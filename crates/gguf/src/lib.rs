//! Bounded, seekable GGUF v2/v3 I/O, derived from the frozen `ggml/src/gguf.cpp`.
//! Tensor payloads remain on disk until explicitly requested. No native code is linked.
#![forbid(unsafe_code)]

use llama_quant::DType;
use std::{
    collections::{BTreeMap, BTreeSet},
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
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: u32,
        values: Vec<Value>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}
impl Value {
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
            Self::String(_) => 8,
            Self::Array { .. } => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
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
    pub fn as_array(&self) -> Option<&[Value]> {
        if let Self::Array { values, .. } = self {
            Some(values)
        } else {
            None
        }
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
        self.dimensions.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(usize::try_from(b).map_err(|_| invalid("dimension exceeds usize"))?)
                .ok_or_else(|| invalid("element count overflow"))
        })
    }
}

pub struct Gguf<R: Read + Seek> {
    reader: R,
    pub version: u32,
    pub endian: Endian,
    pub alignment: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub file_len: u64,
    limits: Limits,
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
    pub fn read(mut reader: R, limits: Limits) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut magic = [0; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(invalid("magic is not GGUF"));
        }
        let mut version_bytes = [0; 4];
        reader.read_exact(&mut version_bytes)?;
        let (endian, version) = if matches!(u32::from_le_bytes(version_bytes), 2 | 3) {
            (Endian::Little, u32::from_le_bytes(version_bytes))
        } else if matches!(u32::from_be_bytes(version_bytes), 2 | 3) {
            (Endian::Big, u32::from_be_bytes(version_bytes))
        } else {
            return Err(Error::Unsupported(format!(
                "version bytes {version_bytes:?}; supported versions are 2 and 3"
            )));
        };
        let mut p = Parser {
            reader: &mut reader,
            endian,
            position: 8,
            file_len,
            limits: &limits,
            total_values: 0,
        };
        let nt = p.count(limits.max_tensors, "tensor count")?;
        let nk = p.count(limits.max_metadata_entries, "metadata count")?;
        let mut metadata = BTreeMap::new();
        for _ in 0..nk {
            let key = p.string()?;
            if key.is_empty() || key.contains('\0') {
                return Err(invalid("empty/NUL metadata key"));
            }
            let ty = p.u32()?;
            let value = p.value(ty, false)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(invalid(format!("duplicate metadata key {key}")));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(Value::U32(v)) => *v,
            _ => return Err(invalid("general.alignment must be UINT32")),
        };
        if !alignment.is_power_of_two() || alignment > 1024 * 1024 {
            return Err(invalid(
                "alignment must be a power of two within 1..=1048576",
            ));
        }
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(nt)
            .map_err(|_| Error::Limit("tensor descriptors allocation".into()))?;
        let mut names = BTreeSet::new();
        for _ in 0..nt {
            let name = p.string()?;
            if name.is_empty() || name.contains('\0') || !names.insert(name.clone()) {
                return Err(invalid("empty, NUL or duplicate tensor name"));
            }
            let nd = p.u32()?;
            if !(1..=4).contains(&nd) {
                return Err(invalid(format!("tensor {name} has {nd} dimensions")));
            }
            let mut dimensions = Vec::new();
            for _ in 0..nd {
                let n = p.u64()?;
                if n == 0 || n > i64::MAX as u64 {
                    return Err(invalid("invalid tensor dimension"));
                }
                dimensions.push(n)
            }
            let id = p.u32()?;
            let dtype = DType::from_id(id).map_err(|e| invalid(e.to_string()))?;
            let row =
                usize::try_from(dimensions[0]).map_err(|_| invalid("row size exceeds usize"))?;
            let row_bytes = dtype.row_bytes(row).map_err(|e| invalid(e.to_string()))? as u64;
            let byte_len = dimensions[1..].iter().try_fold(row_bytes, |a, &b| {
                a.checked_mul(b)
                    .ok_or_else(|| invalid("tensor byte size overflow"))
            })?;
            let offset = p.u64()?;
            if offset % alignment as u64 != 0 {
                return Err(invalid(format!("tensor {name} offset is unaligned")));
            }
            tensors.push(TensorInfo {
                name,
                dimensions,
                dtype,
                offset,
                byte_len,
            });
        }
        let data_offset = align(p.position, alignment as u64)?;
        let mut ranges = Vec::with_capacity(nt);
        for t in &tensors {
            let start = data_offset
                .checked_add(t.offset)
                .ok_or_else(|| invalid("tensor absolute offset overflow"))?;
            let end = start
                .checked_add(t.byte_len)
                .ok_or_else(|| invalid("tensor end overflow"))?;
            if end > file_len {
                return Err(invalid(format!("tensor {} exceeds file length", t.name)));
            }
            ranges.push((start, end, &t.name));
        }
        ranges.sort_unstable_by_key(|r| r.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(invalid(format!(
                    "overlapping tensor payloads {} and {}",
                    pair[0].2, pair[1].2
                )));
            }
        }
        Ok(Self {
            reader,
            version,
            endian,
            alignment,
            metadata,
            tensors,
            data_offset,
            file_len,
            limits,
        })
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let n = usize::try_from(t.byte_len)
            .map_err(|_| Error::Limit("tensor length exceeds usize".into()))?;
        if n > self.limits.max_tensor_read_bytes {
            return Err(Error::Limit(format!(
                "tensor {name} read exceeds {} bytes",
                self.limits.max_tensor_read_bytes
            )));
        }
        let offset = self
            .data_offset
            .checked_add(t.offset)
            .ok_or_else(|| invalid("tensor absolute offset overflow"))?;
        if offset
            .checked_add(n as u64)
            .is_none_or(|end| end > self.file_len)
        {
            return Err(invalid("tensor read exceeds file length"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(n)
            .map_err(|_| Error::Limit("tensor allocation".into()))?;
        bytes.resize(n, 0);
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    pub fn read_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?
            .clone();
        let n = t.elements()?;
        let max = n
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("decoded tensor size overflow".into()))?;
        if max > self.limits.max_tensor_read_bytes {
            return Err(Error::Limit("decoded tensor exceeds read budget".into()));
        }
        let mut bytes = self.read_tensor(name)?;
        if self.endian == Endian::Big {
            let size = match t.dtype.id() {
                0 => 4,
                1 | 30 => 2,
                24 => 1,
                25 => 2,
                26 => 4,
                27 | 28 => 8,
                _ => {
                    return Err(Error::Unsupported(
                        "big-endian quantized tensor payload".into(),
                    ));
                }
            };
            for part in bytes.chunks_exact_mut(size) {
                part.reverse()
            }
        }
        llama_quant::dequantize(t.dtype, &bytes, n).map_err(|e| invalid(e.to_string()))
    }
}

fn align(n: u64, a: u64) -> Result<u64> {
    n.checked_add(a - 1)
        .map(|v| v & !(a - 1))
        .ok_or_else(|| invalid("alignment overflow"))
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
    fn string(&mut self) -> Result<String> {
        let n = self.count(self.limits.max_string_bytes, "string length")?;
        self.charge(n)?;
        let mut b = Vec::new();
        b.try_reserve_exact(n)
            .map_err(|_| Error::Limit("string allocation".into()))?;
        b.resize(n, 0);
        self.reader.read_exact(&mut b)?;
        String::from_utf8(b).map_err(|_| invalid("non-UTF-8 string"))
    }
    fn value(&mut self, ty: u32, in_array: bool) -> Result<Value> {
        self.total_values = self
            .total_values
            .checked_add(1)
            .ok_or_else(|| Error::Limit("value count overflow".into()))?;
        if self.total_values > self.limits.max_total_values {
            return Err(Error::Limit("total metadata values".into()));
        }
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => match self.u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(invalid("boolean is not 0 or 1")),
            },
            8 => Value::String(self.string()?),
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
                if (n as u64)
                    .checked_mul(min)
                    .is_none_or(|v| v > self.file_len - self.position)
                {
                    return Err(invalid("truncated array"));
                }
                let mut values = Vec::new();
                values
                    .try_reserve_exact(n)
                    .map_err(|_| Error::Limit("array allocation".into()))?;
                for _ in 0..n {
                    values.push(self.value(element_type, true)?)
                }
                Value::Array {
                    element_type,
                    values,
                }
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
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
    mut out: W,
    metadata: &BTreeMap<String, Value>,
    tensors: &[TensorData<'_>],
) -> Result<()> {
    let alignment = match metadata.get("general.alignment") {
        None => 32,
        Some(Value::U32(v)) => *v,
        _ => return Err(invalid("general.alignment must be UINT32")),
    };
    if !alignment.is_power_of_two() || alignment > 1024 * 1024 {
        return Err(invalid("invalid alignment"));
    }
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes())
    }
    fn value(b: &mut Vec<u8>, v: &Value) -> Result<()> {
        match v {
            Value::U8(x) => b.push(*x),
            Value::I8(x) => b.push(*x as u8),
            Value::U16(x) => b.extend(x.to_le_bytes()),
            Value::I16(x) => b.extend(x.to_le_bytes()),
            Value::U32(x) => b.extend(x.to_le_bytes()),
            Value::I32(x) => b.extend(x.to_le_bytes()),
            Value::F32(x) => b.extend(x.to_le_bytes()),
            Value::Bool(x) => b.push(u8::from(*x)),
            Value::String(x) => string(b, x),
            Value::U64(x) => b.extend(x.to_le_bytes()),
            Value::I64(x) => b.extend(x.to_le_bytes()),
            Value::F64(x) => b.extend(x.to_le_bytes()),
            Value::Array {
                element_type,
                values,
            } => {
                if *element_type > 12
                    || *element_type == 9
                    || values.iter().any(|v| v.type_id() != *element_type)
                {
                    return Err(invalid("heterogeneous or nested array"));
                }
                b.extend(element_type.to_le_bytes());
                b.extend((values.len() as u64).to_le_bytes());
                for v in values {
                    value(b, v)?
                }
            }
        }
        Ok(())
    }
    for (k, v) in metadata {
        if k.is_empty() || k.contains('\0') {
            return Err(invalid("invalid metadata key"));
        }
        string(&mut header, k);
        header.extend(v.type_id().to_le_bytes());
        value(&mut header, v)?
    }
    let mut offset = 0u64;
    let mut names = BTreeSet::new();
    for t in tensors {
        if t.name.is_empty()
            || t.name.contains('\0')
            || !names.insert(t.name)
            || !(1..=4).contains(&t.dimensions.len())
            || t.dimensions.contains(&0)
        {
            return Err(invalid("invalid tensor descriptor"));
        }
        let row =
            usize::try_from(t.dimensions[0]).map_err(|_| invalid("dimension exceeds usize"))?;
        let n = t.dimensions[1..].iter().try_fold(
            t.dtype.row_bytes(row).map_err(|e| invalid(e.to_string()))? as u64,
            |a, &b| {
                a.checked_mul(b)
                    .ok_or_else(|| invalid("tensor size overflow"))
            },
        )?;
        if n != t.data.len() as u64 {
            return Err(invalid(format!(
                "tensor {} payload length mismatch",
                t.name
            )));
        }
        string(&mut header, t.name);
        header.extend((t.dimensions.len() as u32).to_le_bytes());
        for d in t.dimensions {
            header.extend(d.to_le_bytes())
        }
        header.extend(t.dtype.id().to_le_bytes());
        header.extend(offset.to_le_bytes());
        offset = align(
            offset
                .checked_add(n)
                .ok_or_else(|| invalid("tensor offset overflow"))?,
            alignment as u64,
        )?;
    }
    let padded = usize::try_from(align(header.len() as u64, alignment as u64)?)
        .map_err(|_| invalid("header size overflow"))?;
    header.resize(padded, 0);
    out.write_all(&header)?;
    for t in tensors {
        out.write_all(t.data)?;
        let padding = align(t.data.len() as u64, alignment as u64)? - t.data.len() as u64;
        io::copy(&mut io::repeat(0).take(padding), &mut out)?;
    }
    Ok(())
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
}
