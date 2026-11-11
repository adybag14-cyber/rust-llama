//! Bounded, seekable GGUF v2/v3 I/O, derived from the frozen `ggml/src/gguf.cpp`.
//! Tensor payloads remain on disk until explicitly requested. No native code is linked.
#![forbid(unsafe_code)]

mod model_metadata;
mod raw;
mod shards;
pub use raw::{RawGguf, RawTensorInfo};
pub mod split;
mod writer;
pub use shards::{GgufSet, ShardInfo, ShardLimits};
pub use writer::{
    RawTensorDescriptor, TensorDescriptor, WriteLayout, write_header, write_ordered,
    write_raw_header, write_tool_header,
};

use llama_quant::DType;
use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
    Limit(String),
    Unsupported(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Invalid(s) => write!(f, "invalid GGUF: {s}"),
            Self::Limit(s) => write!(f, "GGUF limit: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported GGUF: {s}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue<S> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(S),
    /// An exact GGUF STRING payload that is not represented by this facade's
    /// text type. The wire type remains STRING (8); no replacement characters
    /// are inserted. Model views use this for non-UTF-8 C-string prefixes.
    ByteString(Vec<u8>),
    Array {
        element_type: u32,
        values: Vec<MetadataValue<S>>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}
pub type Value = MetadataValue<String>;
pub type RawValue = MetadataValue<Vec<u8>>;
impl<S> MetadataValue<S> {
    pub fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) | Self::ByteString(_) => 8,
            Self::Array { .. } => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I8(v) => Some(*v as i64),
            Self::I16(v) => Some(*v as i64),
            Self::I32(v) => Some(*v as i64),
            Self::I64(v) => Some(*v),
            _ => self.as_u64().and_then(|v| v.try_into().ok()),
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(v) => Some(*v as f64),
            Self::F64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&[MetadataValue<S>]> {
        if let Self::Array { values, .. } = self {
            Some(values)
        } else {
            None
        }
    }
}

impl Value {
    /// Exact string bytes, including values that cannot be exposed as UTF-8.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::String(value) => Some(value.as_bytes()),
            Self::ByteString(value) => Some(value),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
}
impl RawValue {
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::String(value) | Self::ByteString(value) => Some(value),
            _ => None,
        }
    }
    /// C getter view; the stored byte string remains unchanged.
    pub fn as_c_string(&self) -> Option<&[u8]> {
        self.as_bytes().map(raw::c_string)
    }
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_metadata_bytes: u64,
    pub max_string_bytes: usize,
    pub max_array_values: usize,
    pub max_total_values: usize,
    pub max_tensors: usize,
    pub max_metadata_entries: usize,
    pub max_tensor_read_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_metadata_bytes: 256 * 1024 * 1024,
            max_string_bytes: 16 * 1024 * 1024,
            max_array_values: 1_000_000,
            max_total_values: 4_000_000,
            max_tensors: 100_000,
            max_metadata_entries: 100_000,
            max_tensor_read_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}
/// Parser work charged to one header, including data-section alignment padding.
/// A shard set applies its limits to the cumulative values across all files.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeaderUsage {
    pub bytes: u64,
    pub metadata_entries: usize,
    pub metadata_values: usize,
    pub tensors: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: dimension zero is contiguous and determines quantization rows.
    pub dimensions: Vec<u64>,
    pub dtype: DType,
    pub offset: u64,
    pub byte_len: u64,
}
impl TensorInfo {
    pub fn elements(&self) -> Result<usize> {
        dimension_elements(&self.dimensions)
    }
}

pub struct Gguf<R: Read + Seek> {
    reader: R,
    pub version: u32,
    pub endian: Endian,
    pub alignment: u32,
    pub metadata: BTreeMap<String, Value>,
    metadata_order: Vec<String>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub file_len: u64,
    limits: Limits,
    header_usage: HeaderUsage,
}
impl Gguf<File> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::read(File::open(path)?, Limits::default())
    }
    pub fn open_with_limits(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        Self::read(File::open(path)?, limits)
    }
}
impl<R: Read + Seek> Gguf<R> {
    pub fn read(reader: R, limits: Limits) -> Result<Self> {
        RawGguf::read(reader, limits)?.into_text()
    }
    /// Original on-disk metadata key order. The public map remains available for lookup.
    /// Keys removed from that map are omitted; new map entries have no on-disk position.
    pub fn metadata_order(&self) -> impl Iterator<Item = &str> {
        self.metadata_order
            .iter()
            .filter(|key| self.metadata.contains_key(*key))
            .map(String::as_str)
    }
    pub fn header_usage(&self) -> HeaderUsage {
        self.header_usage
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        read_tensor_payload(
            &mut self.reader,
            self.data_offset,
            self.file_len,
            &self.limits,
            info,
        )
    }
    pub fn read_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        read_tensor_f32(
            &mut self.reader,
            self.data_offset,
            self.file_len,
            self.endian,
            &self.limits,
            info,
        )
    }
}

fn dimension_elements(dimensions: &[u64]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |count, &dimension| {
        if dimension > i64::MAX as u64 {
            return Err(invalid("dimension exceeds signed GGML range"));
        }
        let dimension =
            usize::try_from(dimension).map_err(|_| invalid("dimension exceeds usize"))?;
        let result = count
            .checked_mul(dimension)
            .ok_or_else(|| invalid("element count overflow"))?;
        if result as u64 > i64::MAX as u64 {
            return Err(invalid("element count exceeds signed GGML range"));
        }
        Ok(result)
    })
}

struct TensorReadInfo<'a> {
    name: std::borrow::Cow<'a, str>,
    dimensions: &'a [u64],
    dtype: DType,
    offset: u64,
    byte_len: u64,
}
impl<'a> From<&'a TensorInfo> for TensorReadInfo<'a> {
    fn from(info: &'a TensorInfo) -> Self {
        Self {
            name: std::borrow::Cow::Borrowed(&info.name),
            dimensions: &info.dimensions,
            dtype: info.dtype,
            offset: info.offset,
            byte_len: info.byte_len,
        }
    }
}
fn read_tensor_payload<R: Read + Seek>(
    reader: &mut R,
    data_offset: u64,
    file_len: u64,
    limits: &Limits,
    info: &TensorInfo,
) -> Result<Vec<u8>> {
    read_tensor_payload_info(reader, data_offset, file_len, limits, &info.into())
}
fn read_tensor_f32<R: Read + Seek>(
    reader: &mut R,
    data_offset: u64,
    file_len: u64,
    endian: Endian,
    limits: &Limits,
    info: &TensorInfo,
) -> Result<Vec<f32>> {
    read_tensor_f32_info(reader, data_offset, file_len, endian, limits, &info.into())
}

fn read_tensor_payload_info<R: Read + Seek>(
    reader: &mut R,
    data_offset: u64,
    file_len: u64,
    limits: &Limits,
    info: &TensorReadInfo<'_>,
) -> Result<Vec<u8>> {
    let n = usize::try_from(info.byte_len)
        .map_err(|_| Error::Limit("tensor length exceeds usize".into()))?;
    if n > limits.max_tensor_read_bytes {
        return Err(Error::Limit(format!(
            "tensor {} read exceeds {} bytes",
            info.name, limits.max_tensor_read_bytes
        )));
    }
    let offset = data_offset
        .checked_add(info.offset)
        .ok_or_else(|| invalid("tensor absolute offset overflow"))?;
    if offset
        .checked_add(n as u64)
        .is_none_or(|end| end > file_len)
    {
        return Err(invalid("tensor read exceeds file length"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(n)
        .map_err(|_| Error::Limit("tensor allocation".into()))?;
    bytes.resize(n, 0);
    reader.seek(SeekFrom::Start(offset))?;
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_tensor_f32_info<R: Read + Seek>(
    reader: &mut R,
    data_offset: u64,
    file_len: u64,
    endian: Endian,
    limits: &Limits,
    info: &TensorReadInfo<'_>,
) -> Result<Vec<f32>> {
    let n = dimension_elements(info.dimensions)?;
    let decoded = n
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("decoded tensor size overflow".into()))?;
    if decoded > limits.max_tensor_read_bytes {
        return Err(Error::Limit("decoded tensor exceeds read budget".into()));
    }
    let mut bytes = read_tensor_payload_info(reader, data_offset, file_len, limits, info)?;
    if endian == Endian::Big {
        let size = match info.dtype.id() {
            0 | 26 => 4,
            1 | 25 | 30 => 2,
            24 => 1,
            27 | 28 => 8,
            _ => {
                return Err(Error::Unsupported(
                    "big-endian quantized tensor payload".into(),
                ));
            }
        };
        for part in bytes.chunks_exact_mut(size) {
            part.reverse();
        }
    }
    llama_quant::dequantize(info.dtype, &bytes, n).map_err(|e| invalid(e.to_string()))
}

fn align(n: u64, a: u64) -> Result<u64> {
    n.checked_add(a - 1)
        .map(|v| v & !(a - 1))
        .ok_or_else(|| invalid("alignment overflow"))
}
/// Header-only read-ahead. The owned file/reader remains in Gguf<R>.
/// Unlike an unbounded convenience buffer, this allocation is fallible and
/// each refill is capped by the remaining physical file and header allowance.
struct HeaderReader<'a, R: Read + Seek> {
    reader: &'a mut R,
    buffer: Vec<u8>,
    begin: usize,
    end: usize,
    remaining: u64,
    position: u64,
}
impl<'a, R: Read + Seek> HeaderReader<'a, R> {
    fn new(
        reader: &'a mut R,
        position: u64,
        file_remaining: u64,
        header_remaining: u64,
    ) -> Result<Self> {
        let remaining = file_remaining.min(header_remaining);
        let capacity = remaining.min(64 * 1024) as usize;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Limit("GGUF header read buffer allocation".into()))?;
        buffer.resize(capacity, 0);
        Ok(Self {
            reader,
            buffer,
            begin: 0,
            end: 0,
            remaining,
            position,
        })
    }
    fn finish(self) -> Result<()> {
        self.reader.seek(SeekFrom::Start(self.position))?;
        Ok(())
    }
}
impl<R: Read + Seek> Read for HeaderReader<'_, R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.begin == self.end {
            if self.remaining == 0 {
                return Ok(0);
            }
            let request = self.remaining.min(self.buffer.len() as u64) as usize;
            let received = self.reader.read(&mut self.buffer[..request])?;
            if received > request {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "reader returned more bytes than requested",
                ));
            }
            self.remaining -= received as u64;
            self.begin = 0;
            self.end = received;
        }
        let count = output.len().min(self.end - self.begin);
        output[..count].copy_from_slice(&self.buffer[self.begin..self.begin + count]);
        self.begin += count;
        self.position += count as u64;
        Ok(count)
    }
}

struct Parser<'a, R> {
    reader: &'a mut R,
    endian: Endian,
    position: u64,
    file_len: u64,
    limits: &'a Limits,
    total_values: usize,
}
impl<R: Read> Parser<'_, R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.charge(N)?;
        let mut b = [0; N];
        self.reader.read_exact(&mut b)?;
        Ok(b)
    }
    fn charge(&mut self, n: usize) -> Result<()> {
        self.position = self
            .position
            .checked_add(n as u64)
            .ok_or_else(|| invalid("metadata size overflow"))?;
        if self.position > self.file_len {
            return Err(invalid("truncated metadata"));
        }
        if self.position > self.limits.max_metadata_bytes {
            return Err(Error::Limit("metadata byte budget".into()));
        }
        Ok(())
    }
    fn minimum_encoded(&self, count: usize, bytes_each: u64, label: &str) -> Result<()> {
        let minimum = (count as u64)
            .checked_mul(bytes_each)
            .ok_or_else(|| Error::Limit(format!("{label} minimum encoded size overflow")))?;
        if minimum > self.file_len.saturating_sub(self.position) {
            return Err(invalid(format!(
                "{label} minimum encoded size exceeds remaining file bytes"
            )));
        }
        if minimum > self.limits.max_metadata_bytes.saturating_sub(self.position) {
            return Err(Error::Limit(format!(
                "{label} minimum encoded size exceeds remaining metadata budget"
            )));
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.bytes()?;
        Ok(match self.endian {
            Endian::Little => u16::from_le_bytes(b),
            Endian::Big => u16::from_be_bytes(b),
        })
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.bytes()?;
        Ok(match self.endian {
            Endian::Little => u32::from_le_bytes(b),
            Endian::Big => u32::from_be_bytes(b),
        })
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.bytes()?;
        Ok(match self.endian {
            Endian::Little => u64::from_le_bytes(b),
            Endian::Big => u64::from_be_bytes(b),
        })
    }
    fn count(&mut self, max: usize, label: &str) -> Result<usize> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| Error::Limit(label.into()))?;
        if n > max {
            Err(Error::Limit(label.into()))
        } else {
            Ok(n)
        }
    }
    fn string(&mut self) -> Result<Vec<u8>> {
        let n = self.count(self.limits.max_string_bytes, "string length")?;
        self.charge(n)?;
        let mut b = Vec::new();
        b.try_reserve_exact(n)
            .map_err(|_| Error::Limit("string allocation".into()))?;
        b.resize(n, 0);
        self.reader.read_exact(&mut b)?;
        Ok(b)
    }
    fn value(&mut self, ty: u32, in_array: bool) -> Result<RawValue> {
        self.total_values = self
            .total_values
            .checked_add(1)
            .ok_or_else(|| Error::Limit("value count overflow".into()))?;
        if self.total_values > self.limits.max_total_values {
            return Err(Error::Limit("total metadata values".into()));
        }
        Ok(match ty {
            0 => RawValue::U8(self.u8()?),
            1 => RawValue::I8(self.u8()? as i8),
            2 => RawValue::U16(self.u16()?),
            3 => RawValue::I16(self.u16()? as i16),
            4 => RawValue::U32(self.u32()?),
            5 => RawValue::I32(self.u32()? as i32),
            6 => RawValue::F32(f32::from_bits(self.u32()?)),
            7 => RawValue::Bool(self.u8()? != 0),
            8 => RawValue::String(self.string()?),
            9 => {
                if in_array {
                    return Err(invalid("nested arrays are forbidden by GGUF"));
                }
                let element_type = self.u32()?;
                if element_type > 12 || element_type == 9 {
                    return Err(invalid("invalid array element type"));
                }
                let n = self.count(self.limits.max_array_values, "array length")?;
                if n > self
                    .limits
                    .max_total_values
                    .saturating_sub(self.total_values)
                {
                    return Err(Error::Limit("total metadata values".into()));
                }
                let min = match element_type {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    _ => 8,
                };
                self.minimum_encoded(n, min, "array")?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(n)
                    .map_err(|_| Error::Limit("array allocation".into()))?;
                for _ in 0..n {
                    values.push(self.value(element_type, true)?)
                }
                RawValue::Array {
                    element_type,
                    values,
                }
            }
            10 => RawValue::U64(self.u64()?),
            11 => RawValue::I64(self.u64()? as i64),
            12 => RawValue::F64(f64::from_bits(self.u64()?)),
            _ => return Err(invalid(format!("unknown metadata type {ty}"))),
        })
    }
}

/// In-memory input descriptor for the little-endian streaming v3 writer.
/// The writer validates layout but does not requantize or swap payload bytes.
pub struct TensorData<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u64],
    pub dtype: DType,
    pub data: &'a [u8],
}
pub fn write<W: Write>(
    out: W,
    metadata: &BTreeMap<String, Value>,
    tensors: &[TensorData<'_>],
) -> Result<()> {
    writer::write_map(out, metadata, tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn sample() -> Vec<u8> {
        let mut b = Vec::new();
        let mut m = BTreeMap::new();
        m.insert("name".into(), Value::String("hello 界".into()));
        m.insert(
            "array".into(),
            Value::Array {
                element_type: 5,
                values: vec![Value::I32(-3), Value::I32(42)],
            },
        );
        write(
            &mut b,
            &m,
            &[TensorData {
                name: "x",
                dimensions: &[2],
                dtype: DType::from_id(0).unwrap(),
                data: &[0, 0, 128, 63, 0, 0, 0, 192],
            }],
        )
        .unwrap();
        b
    }
    #[test]
    fn roundtrip_preserves_types_and_payload() {
        let mut g = Gguf::read(Cursor::new(sample()), Limits::default()).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.metadata["name"].as_str(), Some("hello 界"));
        assert_eq!(g.read_f32("x").unwrap(), vec![1., -2.]);
        assert!(g.read_tensor("missing").is_err());
    }
    #[test]
    fn every_truncated_header_and_payload_is_rejected() {
        let b = sample();
        let g = Gguf::read(Cursor::new(&b), Limits::default()).unwrap();
        let required = g.data_offset as usize + 8;
        for n in 0..required {
            assert!(
                Gguf::read(Cursor::new(&b[..n]), Limits::default()).is_err(),
                "length={n}"
            )
        }
    }
    #[test]
    fn metadata_budget_and_counts_are_enforced() {
        let b = sample();
        let lim = Limits {
            max_metadata_bytes: 32,
            ..Limits::default()
        };
        assert!(matches!(
            Gguf::read(Cursor::new(&b), lim),
            Err(Error::Limit(_))
        ));
        let mut b = b;
        b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Gguf::read(Cursor::new(b), Limits::default()),
            Err(Error::Limit(_))
        ));
    }
    #[test]
    fn overlap_is_rejected() {
        let mut b = Vec::new();
        let d = [0; 8];
        write(
            &mut b,
            &BTreeMap::new(),
            &[
                TensorData {
                    name: "a",
                    dimensions: &[2],
                    dtype: DType::from_id(0).unwrap(),
                    data: &d,
                },
                TensorData {
                    name: "b",
                    dimensions: &[2],
                    dtype: DType::from_id(0).unwrap(),
                    data: &d,
                },
            ],
        )
        .unwrap();
        let second_offset = 24 + (8 + 1 + 4 + 8 + 4 + 8) + (8 + 1 + 4 + 8 + 4);
        b[second_offset..second_offset + 8].fill(0);
        assert!(
            matches!(Gguf::read(Cursor::new(b),Limits::default()),Err(Error::Invalid(s)) if s.contains("overlap"))
        );
    }
    #[test]
    fn big_endian_metadata_and_scalar_payload() {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_be_bytes());
        b.extend(1u64.to_be_bytes());
        b.extend(0u64.to_be_bytes());
        b.extend(1u64.to_be_bytes());
        b.push(b'x');
        b.extend(1u32.to_be_bytes());
        b.extend(1u64.to_be_bytes());
        b.extend(0u32.to_be_bytes());
        b.extend(0u64.to_be_bytes());
        b.resize(64, 0);
        b.extend(1.25f32.to_be_bytes());
        let mut g = Gguf::read(Cursor::new(b), Limits::default()).unwrap();
        assert_eq!(g.read_f32("x").unwrap(), vec![1.25]);
    }
    #[test]
    fn writer_rejects_invalid_arrays() {
        let mut m = BTreeMap::new();
        m.insert(
            "bad".into(),
            Value::Array {
                element_type: 0,
                values: vec![Value::I8(1)],
            },
        );
        assert!(write(Vec::new(), &m, &[]).is_err());
    }
    #[test]
    fn arrays_preflight_the_remaining_header_budget_before_reservation() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.push(b'a');
        bytes.extend(9u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(64u64.to_le_bytes());
        let value_start = bytes.len();
        bytes.resize(value_start + 64, 0);
        let limits = Limits {
            max_metadata_bytes: value_start as u64 + 1,
            ..Limits::default()
        };
        assert!(matches!(
            Gguf::read(Cursor::new(bytes), limits),
            Err(Error::Limit(message)) if message == "array minimum encoded size exceeds remaining metadata budget"
        ));
    }

    #[test]
    fn tensor_counts_preflight_the_remaining_header_and_file_before_reservation() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(64u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        let mut enough_file = bytes.clone();
        enough_file.resize(24 + 64 * 24, 0);
        let limits = Limits {
            max_metadata_bytes: 32,
            ..Limits::default()
        };
        assert!(matches!(
            Gguf::read(Cursor::new(enough_file), limits),
            Err(Error::Limit(message)) if message == "tensor descriptors minimum encoded size exceeds remaining metadata budget"
        ));
        assert!(matches!(
            Gguf::read(Cursor::new(bytes), Limits::default()),
            Err(Error::Invalid(message)) if message == "tensor descriptors minimum encoded size exceeds remaining file bytes"
        ));
    }
    struct ReadMeter {
        cursor: Cursor<Vec<u8>>,
        calls: usize,
        bytes: usize,
        largest_request: usize,
        max_read: usize,
        fail_at: Option<u64>,
    }
    impl ReadMeter {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                cursor: Cursor::new(bytes),
                calls: 0,
                bytes: 0,
                largest_request: 0,
                max_read: usize::MAX,
                fail_at: None,
            }
        }
    }
    impl Read for ReadMeter {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            self.largest_request = self.largest_request.max(output.len());
            let mut count = output.len().min(self.max_read);
            if let Some(end) = self.fail_at {
                if self.cursor.position() >= end {
                    return Err(io::Error::other("injected header I/O failure"));
                }
                count = count.min((end - self.cursor.position()) as usize);
            }
            let actual = self.cursor.read(&mut output[..count])?;
            self.bytes += actual;
            Ok(actual)
        }
    }
    impl Seek for ReadMeter {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.cursor.seek(position)
        }
    }

    #[test]
    fn large_metadata_uses_bounded_batched_reads_and_preserves_reader_position() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array {
                element_type: 8,
                values: (0..12_000)
                    .map(|index| Value::String(format!("token-{index:06}-界")))
                    .collect(),
            },
        );
        let mut bytes = Vec::new();
        write(&mut bytes, &metadata, &[]).unwrap();
        let mut meter = ReadMeter::new(bytes);
        {
            let parsed = Gguf::read(&mut meter, Limits::default()).unwrap();
            assert_eq!(parsed.metadata, metadata);
            assert!(parsed.tensors.is_empty());
            // The original empty-directory offset is the exact logical end,
            // before any optional writer padding.
            assert_eq!(parsed.reader.cursor.position(), parsed.data_offset);
        }
        assert!(
            meter.calls <= 100,
            "small per-value reads escaped to the backing reader: {}",
            meter.calls
        );
        assert!(meter.largest_request <= 64 * 1024);
    }

    #[test]
    fn header_read_ahead_never_exceeds_the_remaining_byte_budget() {
        let mut bytes = sample();
        bytes.resize(200_000, 0);
        let mut meter = ReadMeter::new(bytes);
        let limits = Limits {
            max_metadata_bytes: 32,
            ..Limits::default()
        };
        assert!(matches!(
            Gguf::read(&mut meter, limits),
            Err(Error::Limit(_))
        ));
        assert!(meter.bytes <= 32, "prefetched {} bytes", meter.bytes);
        assert!(meter.cursor.position() <= 32);
    }

    #[test]
    fn buffered_headers_preserve_short_reads_io_errors_and_payload_seeks() {
        let mut meter = ReadMeter::new(sample());
        meter.max_read = 3;
        {
            let mut parsed = Gguf::read(&mut meter, Limits::default()).unwrap();
            // sample(): 24-byte preamble + 37-byte array + 33-byte string
            // entry + 33-byte tensor descriptor; the data starts at 128.
            assert_eq!(parsed.reader.cursor.position(), 127);
            assert_eq!(parsed.data_offset, 128);
            assert_eq!(parsed.read_f32("x").unwrap(), [1.0, -2.0]);
        }
        assert_eq!(meter.cursor.position(), 136);
        let mut failing = ReadMeter::new(sample());
        failing.max_read = 7;
        failing.fail_at = Some(40);
        assert!(matches!(
            Gguf::read(&mut failing, Limits::default()),
            Err(Error::Io(error)) if error.kind() == io::ErrorKind::Other
        ));
        assert_eq!(failing.cursor.position(), 40);
    }
}
