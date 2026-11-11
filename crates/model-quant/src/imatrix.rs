//! Bounded importance-matrix ingestion for the frozen quantize CLI.
//! Derived from common/imatrix-loader.cpp and tools/quantize/quantize.cpp.
//! Copyright (c) 2023-2026 The ggml authors. MIT license.
use crate::{Error, ImportanceMatrix, Result, invalid};
use llama_gguf::{Endian, Limits as GgufLimits, RawGguf, RawValue};
use llama_quant::DType;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Clone, Debug)]
pub struct ImatrixLimits {
    pub max_file_bytes: u64,
    pub max_entries: usize,
    /// Cumulative encoded sums and counts, including replaced legacy entries.
    pub max_values: usize,
    pub max_name_bytes: usize,
    pub max_dataset_bytes: usize,
    pub max_datasets: usize,
    pub max_metadata_bytes: u64,
    pub max_filter_patterns: usize,
    /// Conservative byte-work bound for substring filters.
    pub max_filter_work: u64,
}
impl Default for ImatrixLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 1024 * 1024 * 1024,
            max_entries: 65_536,
            max_values: 128 * 1024 * 1024,
            max_name_bytes: 4096,
            max_dataset_bytes: 16 * 1024 * 1024,
            max_datasets: 4096,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_filter_patterns: 4096,
            max_filter_work: 256 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug)]
pub struct PreparedImatrix {
    pub entries: ImportanceMatrix,
    /// Native C-string bytes; metadata truncation to 127 bytes happens in the CLI.
    pub datasets: Vec<Vec<u8>>,
    pub chunk_count: i32,
}
impl Default for PreparedImatrix {
    fn default() -> Self {
        Self {
            entries: ImportanceMatrix::new(),
            datasets: Vec::new(),
            chunk_count: -1,
        }
    }
}
fn limit(text: &str) -> Error {
    Error::Limit(format!("importance matrix {text}"))
}
fn prefix(bytes: &[u8]) -> &[u8] {
    &bytes[..bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len())]
}
fn charged(total: &mut usize, count: usize, bound: usize, name: &str) -> Result<()> {
    *total = total
        .checked_add(count)
        .filter(|n| *n <= bound)
        .ok_or_else(|| limit(name))?;
    Ok(())
}
fn allocation<T>(count: usize, name: &str) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| limit(name))?;
    Ok(out)
}
fn read_values(reader: &mut impl Read, count: usize) -> Result<Vec<f32>> {
    let mut values = allocation(count, "value allocation")?;
    let mut buffer = [0u8; 16 * 1024];
    while values.len() < count {
        let n = (count - values.len()).min(buffer.len() / 4);
        reader.read_exact(&mut buffer[..n * 4])?;
        for bytes in buffer[..n * 4].as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*bytes);
            if !value.is_finite() {
                return Err(invalid("importance sums must be finite"));
            }
            values.push(value);
        }
    }
    Ok(values)
}
struct LegacyReader {
    reader: BufReader<File>,
    remaining: u64,
}
impl LegacyReader {
    fn take(&mut self, out: &mut [u8]) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(out.len() as u64)
            .ok_or_else(|| invalid("truncated legacy importance matrix"))?;
        self.reader.read_exact(out)?;
        Ok(())
    }
    fn i32(&mut self) -> Result<i32> {
        let mut bytes = [0; 4];
        self.take(&mut bytes)?;
        Ok(i32::from_le_bytes(bytes))
    }
    fn bytes(&mut self, count: usize) -> Result<Vec<u8>> {
        if count as u64 > self.remaining {
            return Err(invalid("truncated legacy importance string"));
        }
        let mut bytes = allocation(count, "string allocation")?;
        bytes.resize(count, 0);
        self.take(&mut bytes)?;
        Ok(bytes)
    }
}
fn legacy(file: File, file_len: u64, limits: &ImatrixLimits) -> Result<PreparedImatrix> {
    let mut r = LegacyReader {
        reader: BufReader::with_capacity(64 * 1024, file),
        remaining: file_len,
    };
    let count = r.i32()?;
    if count < 1 {
        return Err(invalid("legacy importance matrix has no entries"));
    }
    let count = count as usize;
    if count > limits.max_entries {
        return Err(limit("entry count"));
    }
    if (count as u64)
        .checked_mul(16)
        .is_none_or(|n| n > r.remaining)
    {
        return Err(invalid("truncated legacy entry directory"));
    }
    let mut result = PreparedImatrix {
        chunk_count: 0,
        ..PreparedImatrix::default()
    };
    let mut total = 0;
    for _ in 0..count {
        let name_len = r.i32()?;
        let name_len =
            usize::try_from(name_len).map_err(|_| invalid("negative importance name length"))?;
        if name_len > limits.max_name_bytes {
            return Err(limit("name byte count"));
        }
        let raw = r.bytes(name_len)?;
        let name = String::from_utf8(prefix(&raw).to_vec())
            .map_err(|_| Error::Unsupported("importance names must be UTF-8".into()))?;
        let calls = r.i32()?;
        let values = r.i32()?;
        if values < 1 {
            return Err(invalid("legacy importance entry has no values"));
        }
        let values = values as usize;
        charged(&mut total, values, limits.max_values, "value count")?;
        let byte_count = (values as u64)
            .checked_mul(4)
            .ok_or_else(|| limit("value bytes"))?;
        if byte_count > r.remaining {
            return Err(invalid("truncated importance values"));
        }
        let mut data = read_values(&mut r.reader, values)?;
        r.remaining -= byte_count;
        if calls > 0 {
            for value in &mut data {
                *value /= calls as f32;
            }
        }
        result.entries.insert(name, data);
    }
    // Source optional trailer reads into a zero-initialized i32. Preserve partial
    // little-endian chunk bytes, then ignore any incomplete dataset string.
    if r.remaining != 0 {
        let mut chunks = [0u8; 4];
        let n = r.remaining.min(4) as usize;
        r.take(&mut chunks[..n])?;
        result.chunk_count = i32::from_le_bytes(chunks);
        if n == 4 && r.remaining >= 4 {
            let len = r.i32()?;
            if len > 0 {
                let len = len as usize;
                if len > limits.max_dataset_bytes {
                    return Err(limit("dataset bytes"));
                }
                if len as u64 <= r.remaining {
                    if limits.max_datasets == 0 {
                        return Err(limit("dataset count"));
                    }
                    let data = r.bytes(len)?;
                    result.datasets.push(prefix(&data).to_vec());
                }
            }
        }
    }
    Ok(result)
}
#[derive(Default)]
struct Group {
    sums: Option<usize>,
    counts: Option<usize>,
    activations: Option<usize>,
}
fn metadata_u32(model: &RawGguf<File>, key: &[u8]) -> Result<Option<u32>> {
    match model.metadata_native(key) {
        None => Ok(None),
        Some(RawValue::U32(n)) => Ok(Some(*n)),
        _ => Err(invalid(format!(
            "{} must be UINT32",
            String::from_utf8_lossy(key)
        ))),
    }
}
fn rounded_count(value: f32) -> Result<i64> {
    let value = f64::from(value).round();
    let bound = if cfg!(windows) {
        2_147_483_648.0
    } else {
        9_223_372_036_854_775_808.0
    };
    if !value.is_finite() || value < -bound || value >= bound {
        return Err(invalid(
            "importance count is outside the finite native-long rounding domain",
        ));
    }
    Ok(value as i64)
}
fn gguf(file: File, file_len: u64, limits: &ImatrixLimits) -> Result<PreparedImatrix> {
    let max_tensors = limits
        .max_entries
        .checked_mul(3)
        .ok_or_else(|| limit("tensor count"))?;
    let raw_limits = GgufLimits {
        max_metadata_bytes: limits.max_metadata_bytes,
        max_string_bytes: limits.max_dataset_bytes.max(limits.max_name_bytes),
        max_array_values: limits.max_datasets,
        max_total_values: limits
            .max_datasets
            .checked_add(max_tensors)
            .ok_or_else(|| limit("metadata values"))?,
        max_tensors,
        max_metadata_entries: limits.max_entries,
        max_tensor_read_bytes: 64 * 1024,
    };
    let model = RawGguf::read(file.try_clone()?, raw_limits)?;
    if model.endian != Endian::Little {
        return Err(Error::Unsupported(
            "frozen imatrix loader requires little-endian GGUF".into(),
        ));
    }
    if model.tensors.is_empty() {
        return Err(invalid("GGUF importance matrix has no tensors"));
    }
    let last = model
        .tensors
        .last()
        .ok_or_else(|| invalid("missing importance tensor"))?;
    let alignment = u64::from(model.alignment);
    let padded = last
        .byte_len
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| limit("padded tensor bytes"))?;
    let end = model
        .data_offset
        .checked_add(last.offset)
        .and_then(|n| n.checked_add(padded))
        .ok_or_else(|| limit("data end"))?;
    if end > file_len {
        return Err(invalid("GGUF importance blob lacks final tensor padding"));
    }
    let chunks = metadata_u32(&model, b"imatrix.chunk_count")?;
    let size = metadata_u32(&model, b"imatrix.chunk_size")?;
    let _ = metadata_u32(&model, b"imatrix.n_layer_nextn")?;
    let datasets = model.metadata_native(b"imatrix.datasets");
    if datasets.is_none() || chunks.is_none() || size.is_none() {
        return Err(invalid("missing importance matrix metadata"));
    }
    let mut result = PreparedImatrix {
        chunk_count: chunks.unwrap_or(0) as i32,
        ..PreparedImatrix::default()
    };
    let mut dataset_bytes = 0;
    if let Some(RawValue::Array {
        element_type: 8,
        values,
    }) = datasets
    {
        if values.len() > limits.max_datasets {
            return Err(limit("dataset count"));
        }
        result
            .datasets
            .try_reserve_exact(values.len())
            .map_err(|_| limit("datasets allocation"))?;
        for value in values {
            let bytes = value
                .as_bytes()
                .ok_or_else(|| invalid("invalid importance dataset string"))?;
            charged(
                &mut dataset_bytes,
                bytes.len(),
                limits.max_dataset_bytes,
                "dataset bytes",
            )?;
            result.datasets.push(prefix(bytes).to_vec());
        }
    }
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    for (index, tensor) in model.tensors.iter().enumerate() {
        let name = tensor.native_name();
        if name.is_empty() {
            continue;
        }
        let suffix = [b".in_sum".as_slice(), b".in_sum2", b".counts"]
            .iter()
            .enumerate()
            .find_map(|(kind, suffix)| name.strip_suffix(*suffix).map(|key| (kind, key)));
        let Some((kind, key)) = suffix else {
            continue;
        };
        if key.len() > limits.max_name_bytes {
            return Err(limit("name bytes"));
        }
        let key = String::from_utf8(key.to_vec())
            .map_err(|_| Error::Unsupported("importance names must be UTF-8".into()))?;
        if !groups.contains_key(&key) && groups.len() >= limits.max_entries {
            return Err(limit("entry count"));
        }
        let entry = groups.entry(key).or_default();
        match kind {
            0 => entry.activations = Some(index),
            1 => entry.sums = Some(index),
            _ => entry.counts = Some(index),
        }
    }
    // The cloned handle shares the cursor; all header reads above have completed.
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut total_values = 0;
    for (name, group) in groups {
        let sum_index = group
            .sums
            .ok_or_else(|| invalid(format!("missing sums for {name}")))?;
        let count_index = group
            .counts
            .ok_or_else(|| invalid(format!("missing counts for {name}")))?;
        let sums = &model.tensors[sum_index];
        let counts = &model.tensors[count_index];
        let nval = sums.elements()?;
        let ncounts = counts.elements()?;
        if let Some(index) = group.activations {
            let activation = &model.tensors[index];
            if activation.elements()? != nval {
                return Err(invalid("importance activation size mismatch"));
            }
            if activation.byte_len
                < (nval as u64)
                    .checked_mul(4)
                    .ok_or_else(|| limit("activation bytes"))?
            {
                return Err(Error::Unsupported(
                    "original activation reinterpretation would exceed its tensor".into(),
                ));
            }
        }
        if sums.dtype != DType::F32 || counts.dtype != DType::F32 {
            return Err(invalid("importance sums and counts must be F32"));
        }
        if ncounts == 0 {
            return Err(invalid("importance count vector is empty"));
        }
        charged(&mut total_values, nval, limits.max_values, "value count")?;
        charged(&mut total_values, ncounts, limits.max_values, "value count")?;
        reader.seek(SeekFrom::Start(model.data_offset + sums.offset))?;
        let mut values = read_values(&mut reader, nval)?;
        reader.seek(SeekFrom::Start(model.data_offset + counts.offset))?;
        let width = nval / ncounts;
        for expert in 0..ncounts {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            let count = rounded_count(f32::from_le_bytes(bytes))? as f32;
            let row = &mut values[expert * width..(expert + 1) * width];
            if count > 0.0 {
                for value in row {
                    *value /= count;
                }
            } else {
                row.fill(1.0);
            }
        }
        values[ncounts * width..].fill(0.0);
        result.entries.insert(name, values);
    }
    Ok(result)
}
/// Load and normalize both frozen imatrix formats, then apply source substring
/// filtering. An empty path leaves chunk_count=-1 and an empty map. Filtering
/// never suppresses errors in entries that the source loads first.
pub fn prepare_imatrix(
    path: impl AsRef<Path>,
    included: &[String],
    excluded: &[String],
    limits: &ImatrixLimits,
) -> Result<PreparedImatrix> {
    if included
        .len()
        .checked_add(excluded.len())
        .is_none_or(|n| n > limits.max_filter_patterns)
    {
        return Err(limit("filter pattern count"));
    }
    if included
        .iter()
        .chain(excluded)
        .any(|s| s.len() > limits.max_name_bytes)
    {
        return Err(limit("filter pattern bytes"));
    }
    let path = path.as_ref();
    let mut result = if path.as_os_str().is_empty() {
        PreparedImatrix::default()
    } else {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        if length > limits.max_file_bytes {
            return Err(limit("file bytes"));
        }
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        file.seek(SeekFrom::Start(0))?;
        if &magic == b"GGUF" {
            gguf(file, length, limits)?
        } else {
            legacy(file, length, limits)?
        }
    };
    let mut work = 0u64;
    for name in result.entries.keys() {
        for pattern in included.iter().chain(excluded) {
            work = work
                .checked_add(name.len() as u64)
                .and_then(|n| n.checked_add(pattern.len() as u64))
                .filter(|n| *n <= limits.max_filter_work)
                .ok_or_else(|| limit("filter work"))?;
        }
    }
    result.entries.retain(|name, _| {
        !excluded.iter().any(|pattern| name.contains(pattern))
            && (included.is_empty() || included.iter().any(|pattern| name.contains(pattern)))
    });
    Ok(result)
}
