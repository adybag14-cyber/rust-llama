use crate::{
    Context, Error, Model, Result,
    context::{SequenceMemory, SequenceState},
    llama::KvLayer,
    mamba::RecurrentLayer,
};
use std::collections::BTreeMap;

const MAGIC: &[u8; 8] = b"T05STATE";
const VERSION: u32 = 1;

pub(crate) fn encode(context: &Context) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend(MAGIC);
    put_u32(&mut out, VERSION);
    put_u32(&mut out, context.model.architecture().id as u32);
    put_u64(&mut out, context.model.fingerprint());
    put_u64(&mut out, context.adapters.fingerprint());
    put_u32(
        &mut out,
        context
            .sequences
            .len()
            .try_into()
            .map_err(|_| Error::InvalidState("too many sequences".into()))?,
    );
    for (&id, state) in &context.sequences {
        put_u32(&mut out, id as u32);
        put_u64(&mut out, state.position as u64);
        match &state.memory {
            SequenceMemory::Attention(layers) => {
                out.push(0);
                put_u32(&mut out, layers.len() as u32);
                for layer in layers {
                    put_floats(&mut out, &layer.keys)?;
                    put_floats(&mut out, &layer.values)?;
                }
            }
            SequenceMemory::Recurrent(layers) => {
                out.push(1);
                put_u32(&mut out, layers.len() as u32);
                for layer in layers {
                    put_floats(&mut out, &layer.convolution)?;
                    put_floats(&mut out, &layer.ssm)?;
                }
            }
        }
    }
    let checksum = crc32(&out);
    put_u32(&mut out, checksum);
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
    // Even empty sequences require 17 bytes of fields. Reject impossible counts
    // before any reserve or loop over an attacker-controlled integer.
    if count > reader.remaining() / 17 {
        return bad("sequence count exceeds remaining bytes");
    }
    let mut sequences = BTreeMap::new();
    for _ in 0..count {
        let id = reader.u32()?;
        if id > i32::MAX as u32 {
            return bad("negative or overflowing sequence ID");
        }
        let id = id as i32;
        if sequences.contains_key(&id) {
            return bad("duplicate sequence ID");
        }
        let position = usize::try_from(reader.u64()?)
            .map_err(|_| Error::InvalidState("position does not fit this host".into()))?;
        if position > context.model.max_context() {
            return bad("position exceeds model context");
        }
        let kind = reader.take(1)?[0];
        let layers = reader.u32()? as usize;
        if layers != context.model.layer_count() {
            return bad("layer count mismatch");
        }
        let memory = match context.model.as_ref() {
            Model::Llama(model) if kind == 0 => {
                let count = position
                    .checked_mul(model.config().kv_width()?)
                    .ok_or_else(|| Error::InvalidState("KV shape overflow".into()))?;
                let mut result = Vec::with_capacity(layers);
                for _ in 0..layers {
                    result.push(KvLayer {
                        keys: reader.floats(count)?,
                        values: reader.floats(count)?,
                    });
                }
                SequenceMemory::Attention(result)
            }
            Model::Mamba(model) if kind == 1 => {
                let c = model.config();
                let conv = c.inner * (c.convolution_width - 1);
                let ssm = c.inner * c.state_size;
                let mut result = Vec::with_capacity(layers);
                for _ in 0..layers {
                    result.push(RecurrentLayer {
                        convolution: reader.floats(conv)?,
                        ssm: reader.floats(ssm)?,
                    });
                }
                SequenceMemory::Recurrent(result)
            }
            _ => return bad("memory kind does not match architecture"),
        };
        sequences.insert(id, SequenceState { position, memory });
    }
    if reader.remaining() != 0 {
        return bad("trailing state bytes");
    }
    context.sequences = sequences;
    Ok(())
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend(value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend(value.to_le_bytes());
}
fn put_floats(out: &mut Vec<u8>, values: &[f32]) -> Result<()> {
    put_u32(
        out,
        values
            .len()
            .try_into()
            .map_err(|_| Error::InvalidState("vector exceeds state v1 length range".into()))?,
    );
    for value in values {
        out.extend(value.to_le_bytes());
    }
    Ok(())
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
    fn floats(&mut self, expected: usize) -> Result<Vec<f32>> {
        let count = self.u32()? as usize;
        if count != expected {
            return bad("serialized vector length does not match model/state shape");
        }
        let size = count
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidState("vector byte size overflow".into()))?;
        let raw = self.take(size)?;
        let values: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        if !values.iter().all(|x| x.is_finite()) {
            return bad("non-finite serialized state value");
        }
        Ok(values)
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
