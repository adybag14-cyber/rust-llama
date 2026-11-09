//! Retained-owner file headers and checked raw ranges.
use crate::{
    Budget, Error, Integer, Limits, Number, Result, Text, Usage, Value, json, platform, reserve,
};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Storage identifiers known to the pinned secondary Safetensors format.
/// Raw access does not require this classification. The frozen converter
/// currently names only the subset returned by is_converter_dtype().
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    F4,
    F6E2M3,
    F6E3M2,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    F8E8M0,
    F8E4M3Fnuz,
    F8E5M2Fnuz,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
}
impl DType {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "F4" => Self::F4,
            "F6_E2M3" => Self::F6E2M3,
            "F6_E3M2" => Self::F6E3M2,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            "F8_E4M3FNUZ" => Self::F8E4M3Fnuz,
            "F8_E5M2FNUZ" => Self::F8E5M2Fnuz,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "C64" => Self::C64,
            "F64" => Self::F64,
            "I64" => Self::I64,
            "U64" => Self::U64,
            _ => return None,
        })
    }
    pub const fn bits(self) -> u32 {
        match self {
            Self::F4 => 4,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::F8E5M2
            | Self::F8E4M3
            | Self::F8E8M0
            | Self::F8E4M3Fnuz
            | Self::F8E5M2Fnuz => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::C64 | Self::F64 | Self::I64 | Self::U64 => 64,
        }
    }
    pub const fn is_converter_dtype(self) -> bool {
        matches!(
            self,
            Self::F64
                | Self::F32
                | Self::Bf16
                | Self::F16
                | Self::I64
                | Self::I32
                | Self::I16
                | Self::U8
                | Self::I8
                | Self::Bool
                | Self::F8E4M3
                | Self::F8E5M2
                | Self::F8E8M0
        )
    }
}

/// A separately validated dense storage layout. This is not a claim that every
/// dtype has a Rust tensor decoder or architecture conversion implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedLayout {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Debug)]
pub struct RawTensorInfo {
    name: Text,
    dtype: Value,
    shape: Vec<Value>,
    offset: Number,
    size: Number,
}
impl RawTensorInfo {
    pub fn name(&self) -> &Text {
        &self.name
    }
    pub fn dtype_value(&self) -> &Value {
        &self.dtype
    }
    pub fn shape_values(&self) -> &[Value] {
        &self.shape
    }
    /// Absolute source-computed position, including any negative relative offset.
    pub fn absolute_offset(&self) -> &Number {
        &self.offset
    }
    pub fn byte_length(&self) -> &Number {
        &self.size
    }

    pub fn typed_layout(&self) -> Result<TypedLayout> {
        let dtype = self
            .dtype
            .as_text()
            .and_then(Text::as_str)
            .and_then(DType::from_name)
            .ok_or(Error::Unsupported("unknown or non-text storage dtype"))?;
        let mut shape = Vec::new();
        reserve(&mut shape, self.shape.len(), "typed dimensions allocation")?;
        let mut elements = 1u64;
        for value in &self.shape {
            let dimension = match value {
                Value::Number(Number::Integer(n)) => n.as_u64(),
                _ => None,
            }
            .ok_or(Error::Invalid(
                "typed shape requires unsigned integer dimensions",
            ))?;
            elements = elements
                .checked_mul(dimension)
                .ok_or(Error::Limit("typed shape element product"))?;
            shape.push(dimension);
        }
        let bits = elements
            .checked_mul(u64::from(dtype.bits()))
            .ok_or(Error::Limit("typed storage bit product"))?;
        if bits % 8 != 0 {
            return Err(Error::Invalid("typed storage is not byte aligned"));
        }
        let bytes = bits / 8;
        if self.size.as_integer().and_then(Integer::as_u64) != Some(bytes) {
            return Err(Error::Invalid("typed storage length does not match shape"));
        }
        Ok(TypedLayout {
            dtype,
            shape,
            elements,
            bytes,
        })
    }
    fn checked_range(&self, file_len: u64) -> Result<(u64, u64)> {
        let offset = self
            .offset
            .as_integer()
            .and_then(Integer::as_u64)
            .ok_or(Error::Invalid(
                "raw offset is not a nonnegative addressable integer",
            ))?;
        let size = self
            .size
            .as_integer()
            .and_then(Integer::as_u64)
            .ok_or(Error::Invalid(
                "raw length is not a nonnegative addressable integer",
            ))?;
        let end = offset
            .checked_add(size)
            .ok_or(Error::Limit("raw range end overflow"))?;
        if end > file_len {
            return Err(Error::Invalid("raw range exceeds retained file"));
        }
        Ok((offset, size))
    }
}

/// An owned, read-only view over one validated header and its original file.
/// Reads use the retained handle; they never reopen a tensor filename. Callers
/// providing File capabilities must keep file contents stable. A concurrent
/// Unix truncation is returned as a checked read error, never a mapped-memory fault.
#[derive(Debug)]
pub struct SafetensorsFile {
    owner: File,
    file_len: u64,
    data_offset: u64,
    tensors: Vec<RawTensorInfo>,
    metadata: Option<Value>,
    max_read_bytes: usize,
    usage: Usage,
}
impl SafetensorsFile {
    /// Explicit trusted single-file path. Unlike Checkpoint::open_directory,
    /// this follows the caller's chosen path, including a trusted symlink.
    pub fn open(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        Self::from_file(platform::open_trusted(path.as_ref())?, limits)
    }
    /// Assemble from an already-opened capability without bypassing parsing.
    pub fn from_file(owner: File, limits: Limits) -> Result<Self> {
        let mut budget = Budget::new(limits)?;
        Self::from_owner(owner, &mut budget)
    }
    pub(crate) fn from_owner(owner: File, budget: &mut Budget) -> Result<Self> {
        budget.files(1)?;
        let meta = owner.metadata()?;
        if !meta.is_file() {
            return Err(Error::Invalid("tensor owner is not a regular file"));
        }
        let file_len = meta.len();
        let mut prefix = [0u8; 8];
        platform::read_exact_at(&owner, &mut prefix, 0)?;
        let header_len = u64::from_le_bytes(prefix);
        let data_offset = 8u64
            .checked_add(header_len)
            .ok_or(Error::Limit("header position overflow"))?;
        if data_offset > file_len {
            return Err(Error::Invalid("header exceeds retained file"));
        }
        budget.header(header_len)?;
        let length =
            usize::try_from(header_len).map_err(|_| Error::Limit("header is not addressable"))?;
        let mut bytes = Vec::new();
        reserve(&mut bytes, length, "header allocation")?;
        bytes.resize(length, 0);
        platform::read_exact_at(&owner, &mut bytes, 8)?;
        let root = json::parse_charged(&bytes, budget)?;
        drop(bytes);
        let Value::Object(root) = root else {
            return Err(Error::Invalid("header root is not an object"));
        };
        let count = root
            .entries()
            .filter(|(k, _)| k.as_wtf8() != b"__metadata__")
            .count();
        budget.tensors(count)?;
        let mut tensors = Vec::new();
        reserve(&mut tensors, count, "tensor descriptors allocation")?;
        let mut metadata = None;
        for member in root.0 {
            if member.key.as_wtf8() == b"__metadata__" {
                metadata = Some(member.value);
                continue;
            }
            let Value::Object(mut record) = member.value else {
                return Err(Error::Invalid("tensor record is not an object"));
            };
            let dtype = record
                .remove("dtype")
                .ok_or(Error::Invalid("tensor dtype missing"))?;
            let shape = source_shape(
                record
                    .remove("shape")
                    .ok_or(Error::Invalid("tensor shape missing"))?,
                budget,
            )?;
            let offsets = record
                .remove("data_offsets")
                .ok_or(Error::Invalid("tensor offsets missing"))?;
            let Value::Array(offsets) = offsets else {
                return Err(Error::Invalid("tensor offsets are not an array"));
            };
            let mut offsets = offsets.into_iter();
            let start = offsets
                .next()
                .ok_or(Error::Invalid("tensor start offset missing"))?
                .numeric(budget)?;
            let end = offsets
                .next()
                .ok_or(Error::Invalid("tensor end offset missing"))?
                .numeric(budget)?;
            let offset = start.add_u64(data_offset, budget)?;
            let size = end.subtract(&start, budget)?;
            tensors.push(RawTensorInfo {
                name: member.key,
                dtype,
                shape,
                offset,
                size,
            });
        }
        tensors.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            owner,
            file_len,
            data_offset,
            tensors,
            metadata,
            max_read_bytes: budget.limits.max_read_bytes,
            usage: budget.used,
        })
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
    pub fn tensors(&self) -> &[RawTensorInfo] {
        &self.tensors
    }
    pub fn metadata(&self) -> Option<&Value> {
        self.metadata.as_ref()
    }
    /// Cumulative parse work up to this file when used inside a checkpoint.
    pub fn usage(&self) -> Usage {
        self.usage
    }
    pub fn find(&self, name: &Text) -> Option<usize> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.cmp(name))
            .ok()
    }
    pub fn find_str(&self, name: &str) -> Option<usize> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_wtf8().cmp(name.as_bytes()))
            .ok()
    }
    pub fn reader(&self, index: usize) -> Result<TensorReader<'_>> {
        let tensor = self
            .tensors
            .get(index)
            .ok_or(Error::Invalid("tensor index out of range"))?;
        let (offset, length) = tensor.checked_range(self.file_len)?;
        Ok(TensorReader {
            owner: &self.owner,
            offset,
            length,
            position: 0,
        })
    }
    pub fn read_tensor(&self, index: usize) -> Result<Vec<u8>> {
        let mut reader = self.reader(index)?;
        let len = usize::try_from(reader.len())
            .map_err(|_| Error::Limit("raw read is not addressable"))?;
        if len > self.max_read_bytes {
            return Err(Error::Limit("raw read bytes"));
        }
        let mut bytes = Vec::new();
        reserve(&mut bytes, len, "raw tensor allocation")?;
        bytes.resize(len, 0);
        reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

fn source_shape(value: Value, budget: &mut Budget) -> Result<Vec<Value>> {
    match value {
        Value::Array(values) => {
            budget.dimensions(values.len())?;
            Ok(values)
        }
        Value::Text(text) => {
            let count = text.codepoints().count();
            budget.dimensions(count)?;
            let mut result = Vec::new();
            reserve(&mut result, count, "string shape allocation")?;
            for code in text.codepoints() {
                budget.node()?;
                result.push(Value::Text(Text::from_codepoints([code], budget)?));
            }
            Ok(result)
        }
        Value::Object(object) => {
            budget.dimensions(object.len())?;
            let mut result = Vec::new();
            reserve(&mut result, object.len(), "object shape allocation")?;
            for member in object.0 {
                result.push(Value::Text(member.key));
            }
            Ok(result)
        }
        _ => Err(Error::Invalid("source shape is not iterable")),
    }
}

/// Streaming reader of one checked range. It borrows its owner and is bounded
/// by the recorded range; payload reads do not allocate internally.
#[derive(Debug)]
pub struct TensorReader<'a> {
    owner: &'a File,
    offset: u64,
    length: u64,
    position: u64,
}
impl TensorReader<'_> {
    pub fn len(&self) -> u64 {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}
impl Read for TensorReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let remaining = self.length.saturating_sub(self.position);
        if output.is_empty() || remaining == 0 {
            return Ok(0);
        }
        let length = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(output.len());
        let position = self.offset.checked_add(self.position).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "tensor position overflow")
        })?;
        let read = platform::read_at(self.owner, &mut output[..length], position)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "retained tensor was truncated",
            ));
        }
        self.position += read as u64;
        Ok(read)
    }
}
impl Seek for TensorReader<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let target = match from {
            SeekFrom::Start(value) => Some(value),
            SeekFrom::End(value) => self.length.checked_add_signed(value),
            SeekFrom::Current(value) => self.position.checked_add_signed(value),
        }
        .filter(|value| *value <= self.length)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek outside tensor range"))?;
        self.position = target;
        Ok(target)
    }
}
