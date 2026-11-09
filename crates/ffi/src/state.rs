//! Frozen state/session serialization for the supported dense F32/F16 CPU context.
use crate::*;
use llama_model_runtime::{Model as RuntimeModel, NativeState};
use std::ffi::c_char;
use std::path::Path;

const IO_MAGIC: u32 = 0xaf143cd8;
const SESSION_MAGIC: u32 = 0x6767736e;
const SEQUENCE_MAGIC: u32 = 0x67677371;
const MAX_STATE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_SNAPSHOT_BYTES: usize = 1024 * 1024 * 1024;

fn prepared(context: &ContextHandle, sequence: Option<i32>, flags: u32) -> Result<NativeState> {
    let mut state = context
        .runtime
        .prepare_native_state(sequence, flags)
        .map_err(|e| e.to_string())?;
    if let Some(sequence) = sequence {
        let length = state.bytes.len();
        state
            .bytes
            .try_reserve_exact(8)
            .map_err(|_| "sequence header allocation failed")?;
        state.bytes.resize(length + 8, 0);
        state.bytes.copy_within(0..length, 8);
        state.bytes[..4].copy_from_slice(&IO_MAGIC.to_le_bytes());
        state.bytes[4..8].copy_from_slice(&sequence.to_le_bytes());
        state
            .write_boundaries
            .try_reserve_exact(2)
            .map_err(|_| "sequence header boundary allocation failed")?;
        state.write_boundaries.iter_mut().for_each(|end| *end += 8);
        state.write_boundaries.splice(0..0, [4, 8]);
    }
    Ok(state)
}
fn state_size(context: &ContextHandle, sequence: Option<i32>, flags: u32) -> Result<usize> {
    context
        .runtime
        .native_state_size(sequence, flags)
        .map_err(|e| e.to_string())?
        .checked_add(if sequence.is_some() { 8 } else { 0 })
        .ok_or_else(|| "state size overflow".into())
}
unsafe fn state_input<'a>(source: *const u8, size: usize) -> Result<&'a [u8]> {
    if size > MAX_STATE_BYTES || size > isize::MAX as usize {
        return Err("native state exceeds supported byte limit".into());
    }
    if size == 0 {
        return Ok(&[]);
    }
    unsafe {
        reference(source)?;
    }
    Ok(unsafe { slice::from_raw_parts(source, size) })
}
unsafe fn write_state(
    context: &mut ContextHandle,
    sequence: Option<i32>,
    flags: u32,
    target: *mut u8,
    capacity: usize,
) -> Result<usize> {
    let state = prepared(context, sequence, flags)?;
    let required = state.bytes.len();
    let written = state
        .write_boundaries
        .iter()
        .copied()
        .take_while(|end| *end <= capacity)
        .last()
        .unwrap_or(0);
    if written > 0 {
        unsafe {
            reference_mut(target)?;
            ptr::copy_nonoverlapping(state.bytes.as_ptr(), target, written);
        }
    }
    if required > capacity {
        return Err("native state destination is too small".into());
    }
    if flags & 2 != 0 {
        let sequence = sequence.ok_or("device snapshots require a sequence state call")?;
        if !state.tensors.is_empty() {
            let previous = context.state_snapshots.get(&sequence).map_or(0, Vec::len);
            let total = context
                .state_snapshots
                .values()
                .try_fold(0usize, |sum, data| sum.checked_add(data.len()))
                .and_then(|sum| sum.checked_sub(previous))
                .and_then(|sum| sum.checked_add(state.tensors.len()))
                .ok_or("device snapshot size overflow")?;
            if total > MAX_SNAPSHOT_BYTES {
                return Err("CPU snapshot storage exceeds one GiB".into());
            }
            context.state_snapshots.insert(sequence, state.tensors);
        } else {
            context.state_snapshots.entry(sequence).or_default();
        }
    }
    Ok(required)
}
fn read_state(
    context: &mut ContextHandle,
    source: &[u8],
    sequence: Option<i32>,
    flags: u32,
) -> Result<usize> {
    if sequence.is_none() {
        return context
            .runtime
            .restore_native_state(source, None, 0)
            .map_err(|e| e.to_string());
    }
    let header = source.get(..8).ok_or("truncated sequence state header")?;
    if u32::from_le_bytes(header[..4].try_into().expect("four bytes")) != IO_MAGIC {
        return Err("wrong sequence state magic".into());
    }
    let origin = i32::from_le_bytes(header[4..].try_into().expect("four bytes"));
    let consumed = if flags & 2 != 0 {
        let snapshot = context
            .state_snapshots
            .get(&origin)
            .ok_or("device state belongs to a missing or different owning context")?;
        context
            .runtime
            .restore_native_state_with_tensors(&source[8..], sequence, flags, snapshot)
    } else {
        context
            .runtime
            .restore_native_state(&source[8..], sequence, flags)
    };
    consumed.map(|size| size + 8).map_err(|e| e.to_string())
}
ffi!(
    llama_state_get_size(context: *mut ContextHandle) -> usize,
    0,
    { state_size(unsafe { reference(context)? }, None, 0) }
);
ffi!(
    llama_get_state_size(context: *mut ContextHandle) -> usize,
    0,
    { state_size(unsafe { reference(context)? }, None, 0) }
);
ffi!(
    llama_state_get_data(context: *mut ContextHandle, target: *mut u8, size: usize) -> usize,
    0,
    { unsafe { write_state(reference_mut(context)?, None, 0, target, size) } }
);
ffi!(
    llama_copy_state_data(context: *mut ContextHandle, target: *mut u8) -> usize,
    0,
    { unsafe { write_state(reference_mut(context)?, None, 0, target, usize::MAX) } }
);
ffi!(
    llama_state_set_data(context: *mut ContextHandle, source: *const u8, size: usize) -> usize,
    0,
    {
        let source = unsafe { state_input(source, size)? };
        read_state(unsafe { reference_mut(context)? }, source, None, 0)
    }
);
ffi!(
    llama_state_seq_get_size(context: *mut ContextHandle, sequence: i32) -> usize,
    0,
    { state_size(unsafe { reference(context)? }, Some(sequence), 0) }
);
ffi!(
    llama_state_seq_get_size_ext(context: *mut ContextHandle, sequence: i32, flags: u32) -> usize,
    0,
    { state_size(unsafe { reference(context)? }, Some(sequence), flags) }
);
ffi!(
    llama_state_seq_get_data(
        context: *mut ContextHandle,
        target: *mut u8,
        size: usize,
        sequence: i32,
    ) -> usize,
    0,
    { unsafe { write_state(reference_mut(context)?, Some(sequence), 0, target, size) } }
);
ffi!(
    llama_state_seq_get_data_ext(
        context: *mut ContextHandle,
        target: *mut u8,
        size: usize,
        sequence: i32,
        flags: u32,
    ) -> usize,
    0,
    { unsafe { write_state(reference_mut(context)?, Some(sequence), flags, target, size) } }
);
ffi!(
    llama_state_seq_set_data(
        context: *mut ContextHandle,
        source: *const u8,
        size: usize,
        sequence: i32,
    ) -> usize,
    0,
    {
        let source = unsafe { state_input(source, size)? };
        read_state(
            unsafe { reference_mut(context)? },
            source,
            Some(sequence),
            0,
        )
    }
);
ffi!(
    llama_state_seq_set_data_ext(
        context: *mut ContextHandle,
        source: *const u8,
        size: usize,
        sequence: i32,
        flags: u32,
    ) -> usize,
    0,
    {
        let source = unsafe { state_input(source, size)? };
        read_state(
            unsafe { reference_mut(context)? },
            source,
            Some(sequence),
            flags,
        )
    }
);

// The deprecated setter has no length argument. Its original caller contract
// requires a complete, valid state allocation. Read only bounded scalar fields
// first, derive the actual encoded length, then create a slice of that length.
// No arbitrary maximum-length slice is constructed over a smaller allocation.
struct LegacyReader {
    source: *const u8,
    at: usize,
}
impl LegacyReader {
    fn skip(&mut self, size: usize) -> Result<()> {
        self.at = self
            .at
            .checked_add(size)
            .filter(|n| *n <= MAX_STATE_BYTES && *n <= isize::MAX as usize)
            .ok_or("legacy state byte count exceeds supported limit")?;
        Ok(())
    }
    unsafe fn u32(&mut self) -> Result<u32> {
        let offset = self.at;
        self.skip(4)?;
        Ok(u32::from_le(unsafe {
            ptr::read_unaligned(self.source.add(offset).cast::<u32>())
        }))
    }
    unsafe fn u64(&mut self) -> Result<u64> {
        let offset = self.at;
        self.skip(8)?;
        Ok(u64::from_le(unsafe {
            ptr::read_unaligned(self.source.add(offset).cast::<u64>())
        }))
    }
}
unsafe fn legacy_size(context: &ContextHandle, source: *const u8) -> Result<usize> {
    unsafe {
        reference(source)?;
    }
    let RuntimeModel::Llama(model) = context.runtime.model().as_ref() else {
        return Err("unsupported native state model".into());
    };
    let width = model.config().kv_width().map_err(|e| e.to_string())?;
    let (key_type, value_type) = context.runtime.cache_types();
    let mut reader = LegacyReader { source, at: 0 };
    if unsafe { reader.u32()? } != 5 {
        return Err("legacy state architecture length mismatch".into());
    }
    reader.skip(5)?;
    let streams = unsafe { reader.u32()? } as usize;
    let expected = if context.runtime.is_unified() {
        1
    } else {
        context.runtime.sequence_id_limit()
    };
    if streams != expected {
        return Err("legacy native state stream mismatch".into());
    }
    for _ in 0..streams {
        let count = unsafe { reader.u32()? } as usize;
        if count > context.runtime.context_tokens_per_sequence() {
            return Err("legacy state cell count exceeds capacity".into());
        }
        if count == 0 {
            continue;
        }
        for _ in 0..count {
            reader.skip(4)?;
            let ids = unsafe { reader.u32()? } as usize;
            if ids > context.runtime.sequence_id_limit() {
                return Err("legacy state sequence count exceeds limit".into());
            }
            reader.skip(ids * 4)?;
        }
        if unsafe { reader.u32()? } != u32::from(context.runtime.values_are_transposed())
            || unsafe { reader.u32()? } as usize != model.layer_count()
            || unsafe { reader.u32()? } != 0
            || unsafe { reader.u32()? } != 0
        {
            return Err("legacy native state tensor layout mismatch".into());
        }
        let elements = count
            .checked_mul(width)
            .ok_or("legacy tensor shape overflow")?;
        for _ in 0..model.layer_count() {
            if unsafe { reader.u32()? } != key_type as u32
                || unsafe { reader.u64()? } != (width * key_type.size()) as u64
            {
                return Err("legacy state key type/shape mismatch".into());
            }
            reader.skip(
                elements
                    .checked_mul(key_type.size())
                    .ok_or("legacy key byte overflow")?,
            )?;
        }
        for _ in 0..model.layer_count() {
            if unsafe { reader.u32()? } != value_type as u32 {
                return Err("legacy state value type mismatch".into());
            }
            if context.runtime.values_are_transposed() {
                if unsafe { reader.u32()? } != value_type.size() as u32
                    || unsafe { reader.u32()? } as usize != width
                {
                    return Err("legacy state value shape mismatch".into());
                }
            } else if unsafe { reader.u64()? } != (width * value_type.size()) as u64 {
                return Err("legacy state value row size mismatch".into());
            }
            reader.skip(
                elements
                    .checked_mul(value_type.size())
                    .ok_or("legacy value byte overflow")?,
            )?;
        }
    }
    Ok(reader.at)
}
ffi!(
    llama_set_state_data(context: *mut ContextHandle, source: *const u8) -> usize,
    0,
    {
        let context = unsafe { reference_mut(context)? };
        let size = unsafe { legacy_size(context, source)? };
        read_state(context, unsafe { state_input(source, size)? }, None, 0)
    }
);

fn read_bounded(reader: impl std::io::Read, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let maximum = limit.checked_add(1).ok_or("state read limit overflow")?;
    let mut input = reader.take(maximum as u64);
    let mut result = Vec::new();
    let mut chunk = [0u8; 65_536];
    loop {
        let count = input.read(&mut chunk).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        let needed = result
            .len()
            .checked_add(count)
            .ok_or("state file length overflow")?;
        if needed > limit {
            return Err("state file exceeds supported byte limit".into());
        }
        if needed > result.capacity() {
            let capacity = result
                .capacity()
                .saturating_mul(2)
                .max(65_536)
                .max(needed)
                .min(limit);
            result
                .try_reserve_exact(capacity - result.len())
                .map_err(|_| "state file allocation failed")?;
        }
        result.extend_from_slice(&chunk[..count]);
    }
    Ok(result)
}
fn file_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > MAX_STATE_BYTES as u64 {
        return Err("state path is not a supported bounded regular file".into());
    }
    // The limit applies to this opened handle's actual reads even if the file
    // grows after metadata was observed.
    read_bounded(file, MAX_STATE_BYTES)
}
unsafe fn save_file(
    context: *mut ContextHandle,
    path: *const c_char,
    tokens: *const i32,
    count: usize,
    sequence: Option<i32>,
) -> Result<usize> {
    use std::io::Write;
    let path = Path::new(unsafe { c_string(path)? });
    let tokens = unsafe { input_slice(tokens, count)? };
    let context = unsafe { reference(context)? };
    let memory = context
        .runtime
        .save_native_state(sequence, 0)
        .map_err(|e| e.to_string())?;
    let length = tokens
        .len()
        .checked_mul(4)
        .and_then(|n| n.checked_add(12))
        .and_then(|n| n.checked_add(memory.len()))
        .filter(|n| *n <= MAX_STATE_BYTES)
        .ok_or("session file exceeds supported byte limit")?;
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(
        &if sequence.is_some() {
            SEQUENCE_MAGIC
        } else {
            SESSION_MAGIC
        }
        .to_le_bytes(),
    )
    .map_err(|e| e.to_string())?;
    file.write_all(&if sequence.is_some() { 4u32 } else { 11u32 }.to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(
        &u32::try_from(tokens.len())
            .map_err(|_| "too many session tokens")?
            .to_le_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let mut encoded = [0u8; 65_536];
    for chunk in tokens.chunks(encoded.len() / 4) {
        for (raw, token) in encoded.as_chunks_mut::<4>().0.iter_mut().zip(chunk) {
            raw.copy_from_slice(&token.to_le_bytes());
        }
        file.write_all(&encoded[..chunk.len() * 4])
            .map_err(|e| e.to_string())?;
    }
    file.write_all(&memory).map_err(|e| e.to_string())?;
    Ok(length)
}
unsafe fn load_file(
    context: *mut ContextHandle,
    path: *const c_char,
    tokens: *mut i32,
    capacity: usize,
    count_out: *mut usize,
    sequence: Option<i32>,
) -> Result<usize> {
    let path = Path::new(unsafe { c_string(path)? });
    unsafe {
        reference(context)?;
    }
    let bytes = file_bytes(path)?;
    let head = bytes.get(..12).ok_or("truncated session file header")?;
    let u32_at = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().expect("four bytes"));
    if u32_at(0)
        != if sequence.is_some() {
            SEQUENCE_MAGIC
        } else {
            SESSION_MAGIC
        }
        || u32_at(4) != if sequence.is_some() { 4 } else { 11 }
    {
        return Err("session magic or version mismatch".into());
    }
    let count = u32_at(8) as usize;
    let prompt_end = count
        .checked_mul(4)
        .and_then(|n| n.checked_add(12))
        .ok_or("session token byte count overflow")?;
    let prompt = bytes
        .get(12..prompt_end)
        .ok_or("truncated session prompt")?;
    if sequence.is_some() && tokens.is_null() {
        unsafe {
            *reference_mut(count_out)? = count;
        }
        return Ok(12);
    }
    if count > capacity || count > MAX_FOREIGN_ELEMENTS {
        return Err("session token count exceeds output capacity".into());
    }
    if count > 0 {
        unsafe {
            reference_mut(tokens)?;
        }
        for (index, raw) in prompt.as_chunks::<4>().0.iter().enumerate() {
            unsafe {
                *tokens.add(index) = i32::from_le_bytes(*raw);
            }
        }
    }
    unsafe {
        *reference_mut(count_out)? = count;
    }
    let context = unsafe { reference_mut(context)? };
    let consumed = context
        .runtime
        .restore_native_state(&bytes[prompt_end..], sequence, 0)
        .map_err(|e| e.to_string())?;
    if sequence.is_none() && consumed != bytes.len() - prompt_end {
        return Err("session file has trailing state bytes".into());
    }
    if sequence.is_some() && consumed == 0 {
        return Err("sequence file contains no restored state".into());
    }
    Ok(prompt_end + consumed)
}
ffi!(
    llama_state_save_file(
        context: *mut ContextHandle,
        path: *const c_char,
        tokens: *const i32,
        count: usize,
    ) -> bool,
    false,
    {
        unsafe {
            save_file(context, path, tokens, count, None)?;
        }
        Ok(true)
    }
);
ffi!(
    llama_save_session_file(
        context: *mut ContextHandle,
        path: *const c_char,
        tokens: *const i32,
        count: usize,
    ) -> bool,
    false,
    {
        unsafe {
            save_file(context, path, tokens, count, None)?;
        }
        Ok(true)
    }
);
ffi!(
    llama_state_load_file(
        context: *mut ContextHandle,
        path: *const c_char,
        tokens: *mut i32,
        capacity: usize,
        count_out: *mut usize,
    ) -> bool,
    false,
    {
        unsafe {
            load_file(context, path, tokens, capacity, count_out, None)?;
        }
        Ok(true)
    }
);
ffi!(
    llama_load_session_file(
        context: *mut ContextHandle,
        path: *const c_char,
        tokens: *mut i32,
        capacity: usize,
        count_out: *mut usize,
    ) -> bool,
    false,
    {
        unsafe {
            load_file(context, path, tokens, capacity, count_out, None)?;
        }
        Ok(true)
    }
);
ffi!(
    llama_state_seq_save_file(
        context: *mut ContextHandle,
        path: *const c_char,
        sequence: i32,
        tokens: *const i32,
        count: usize,
    ) -> usize,
    0,
    { unsafe { save_file(context, path, tokens, count, Some(sequence)) } }
);
ffi!(
    llama_state_seq_load_file(
        context: *mut ContextHandle,
        path: *const c_char,
        sequence: i32,
        tokens: *mut i32,
        capacity: usize,
        count_out: *mut usize,
    ) -> usize,
    0,
    { unsafe { load_file(context, path, tokens, capacity, count_out, Some(sequence)) } }
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_read_limit_does_not_depend_on_reported_file_length() {
        use std::io::Cursor;
        assert_eq!(
            read_bounded(Cursor::new(b"12345678"), 8).unwrap(),
            b"12345678"
        );
        assert!(read_bounded(Cursor::new(b"123456789MORE"), 8).is_err());
        assert_eq!(read_bounded(Cursor::new(b""), 0).unwrap(), b"");
        assert!(read_bounded(Cursor::new(b"x"), 0).is_err());
    }
    #[test]
    fn legacy_reader_arithmetic_rejects_oversized_offsets_before_access() {
        let mut reader = LegacyReader {
            source: ptr::null(),
            at: MAX_STATE_BYTES - 2,
        };
        assert!(reader.skip(4).is_err());
    }
}
