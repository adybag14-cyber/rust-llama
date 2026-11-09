use crate::{
    Context, Error, Result,
    context::{
        SequenceMemory,
        owned::{self, PlanEntry, StatePlan, StateStorage},
    },
};

const MAGIC: &[u8; 8] = b"T05STATE";
const VERSION: u32 = 1;

fn serialized_float_bytes(values: &[f32]) -> Result<usize> {
    u32::try_from(values.len())
        .map_err(|_| Error::InvalidState("vector exceeds state v1 length range".into()))?;
    owned::add(4, owned::multiply(values.len(), 4)?)
}

pub(crate) fn encode(context: &Context) -> Result<Vec<u8>> {
    let sequence_count = u32::try_from(context.sequences.len())
        .map_err(|_| Error::InvalidState("too many sequences".into()))?;
    let mut size = 40usize; // fixed header plus CRC32
    for (_, state) in context.sequences.iter() {
        size = owned::add(size, 17)?; // id, position, memory kind, layer count
        match &state.memory {
            SequenceMemory::Attention(layers) => {
                u32::try_from(layers.len())
                    .map_err(|_| Error::InvalidState("too many layers".into()))?;
                for layer in layers {
                    size = owned::add(size, serialized_float_bytes(&layer.keys)?)?;
                    size = owned::add(size, serialized_float_bytes(&layer.values)?)?;
                }
            }
            SequenceMemory::Recurrent(layers) => {
                u32::try_from(layers.len())
                    .map_err(|_| Error::InvalidState("too many layers".into()))?;
                for layer in layers {
                    size = owned::add(size, serialized_float_bytes(&layer.convolution)?)?;
                    size = owned::add(size, serialized_float_bytes(&layer.ssm)?)?;
                }
            }
        }
    }
    // Caller-owned bytes are not retained in the context ledger after return,
    // but the copy must fit alongside resident state and checkpoints at creation.
    let _reservation = context.budget.reserve(size)?;
    let mut out = owned::allocate(size, "serialized state")?;
    out.extend(MAGIC);
    put_u32(&mut out, VERSION);
    put_u32(&mut out, context.model.architecture().id as u32);
    put_u64(&mut out, context.model.fingerprint());
    put_u64(&mut out, context.adapters.fingerprint());
    put_u32(&mut out, sequence_count);
    for (&id, state) in context.sequences.iter() {
        put_u32(&mut out, id as u32);
        put_u64(&mut out, state.position as u64);
        match &state.memory {
            SequenceMemory::Attention(layers) => {
                out.push(0);
                put_u32(&mut out, layers.len() as u32);
                for layer in layers {
                    put_floats(&mut out, &layer.keys);
                    put_floats(&mut out, &layer.values);
                }
            }
            SequenceMemory::Recurrent(layers) => {
                out.push(1);
                put_u32(&mut out, layers.len() as u32);
                for layer in layers {
                    put_floats(&mut out, &layer.convolution);
                    put_floats(&mut out, &layer.ssm);
                }
            }
        }
    }
    let checksum = crc32(&out);
    put_u32(&mut out, checksum);
    debug_assert_eq!(out.len(), size);
    Ok(out)
}

pub(crate) fn decode(context: &mut Context, bytes: &[u8]) -> Result<()> {
    if bytes.len() < 40 {
        return bad("state header truncated");
    }
    let (body, tail) = bytes.split_at(bytes.len() - 4);
    let expected = u32::from_le_bytes(tail.try_into().expect("four byte tail"));
    if crc32(body) != expected {
        return bad("CRC32 mismatch");
    }
    let mut reader = Reader {
        bytes: body,
        position: 0,
    };
    if reader.take(8)? != MAGIC {
        return bad("incorrect state magic");
    }
    if reader.u32()? != VERSION {
        return bad("unsupported state version");
    }
    if reader.u32()? != context.model.architecture().id as u32 {
        return bad("wrong architecture");
    }
    if reader.u64()? != context.model.fingerprint() {
        return bad("wrong model fingerprint");
    }
    if reader.u64()? != context.adapters.fingerprint() {
        return bad("wrong adapter fingerprint");
    }
    let count = reader.u32()? as usize;
    if count > context.options.maximum_sequences {
        return bad("sequence count exceeds context limit");
    }
    if count > reader.remaining() / 17 {
        return bad("sequence count exceeds remaining bytes");
    }
    let payload_start = reader.position;
    let shape = context.sequences.shape;
    let mut plan = StatePlan::new(&context.budget, count)?;
    // First pass validates the whole immutable byte slice before allocating any
    // float vectors. Only bounded sequence descriptors are allocated here.
    for _ in 0..count {
        let id = reader.u32()?;
        if id > i32::MAX as u32 {
            return bad("negative or overflowing sequence ID");
        }
        let position = usize::try_from(reader.u64()?)
            .map_err(|_| Error::InvalidState("position does not fit this host".into()))?;
        if position > context.model.max_context() {
            return bad("position exceeds model context");
        }
        if reader.take(1)?[0] != shape.kind() {
            return bad("memory kind does not match architecture");
        }
        if reader.u32()? as usize != shape.layers() {
            return bad("layer count mismatch");
        }
        let (first, second) = shape.lengths(position)?;
        for _ in 0..shape.layers() {
            reader.floats(first, true)?;
            reader.floats(second, true)?;
        }
        plan.push(PlanEntry {
            id: id as i32,
            source: id as i32,
            position,
        })?;
    }
    if reader.remaining() != 0 {
        return bad("trailing state bytes");
    }
    plan.sort_and_check()?;
    let mut replacement = StateStorage::copied(&context.budget, shape, &plan, None)?;
    reader.position = payload_start;
    for _ in 0..count {
        let id = reader.u32()? as i32;
        let position = reader.u64()? as usize;
        reader.take(5)?; // memory kind and layer count were checked above
        let state = replacement
            .get_mut(&id)
            .expect("validated state descriptor");
        let (first, second) = shape.lengths(position)?;
        match &mut state.memory {
            SequenceMemory::Attention(layers) => {
                for layer in layers {
                    copy_floats(&mut layer.keys, reader.floats(first, false)?)?;
                    copy_floats(&mut layer.values, reader.floats(second, false)?)?;
                }
            }
            SequenceMemory::Recurrent(layers) => {
                for layer in layers {
                    copy_floats(&mut layer.convolution, reader.floats(first, false)?)?;
                    copy_floats(&mut layer.ssm, reader.floats(second, false)?)?;
                }
            }
        }
        state.position = position;
    }
    context.sequences = replacement;
    Ok(())
}

fn copy_floats(destination: &mut Vec<f32>, raw: &[u8]) -> Result<()> {
    let count = raw.len() / 4;
    if destination.capacity() < count {
        return bad("state decoder exceeds its reserved capacity");
    }
    destination.clear();
    destination.extend(
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes)),
    );
    Ok(())
}
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend(value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend(value.to_le_bytes());
}
fn put_floats(out: &mut Vec<u8>, values: &[f32]) {
    // The complete output and every v1 count were checked before allocation.
    put_u32(out, values.len() as u32);
    for value in values {
        out.extend(value.to_le_bytes());
    }
}
fn bad<T>(message: &str) -> Result<T> {
    Err(Error::InvalidState(message.into()))
}
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        if len > self.remaining() {
            return bad("truncated state payload");
        }
        let result = &self.bytes[self.position..self.position + len];
        self.position += len;
        Ok(result)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn floats(&mut self, expected: usize, validate_finite: bool) -> Result<&'a [u8]> {
        let count = self.u32()? as usize;
        if count != expected {
            return bad("serialized vector length does not match model/state shape");
        }
        let size = owned::multiply(count, 4)?;
        let raw = self.take(size)?;
        if validate_finite
            && raw
                .as_chunks::<4>()
                .0
                .iter()
                .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
        {
            return bad("non-finite serialized state value");
        }
        Ok(raw)
    }
}

/// IEEE CRC-32, used only for accidental-corruption detection.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320u32 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_crc32_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}
