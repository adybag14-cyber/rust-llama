//! Ordered GGUF headers and bounded streaming payload layout.
//! Validation precedes the first output byte; metadata values stay borrowed.
use crate::raw::c_string;
use crate::{
    DType, Error, Limits, MetadataValue, RawValue, Result, TensorData, Value, align,
    dimension_elements, invalid,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

/// A borrowed tensor description; its payload need not be resident in memory.
#[derive(Debug)]
pub struct TensorDescription<'a, N: ?Sized> {
    pub name: &'a N,
    pub dimensions: &'a [u64],
    pub dtype: DType,
}
impl<N: ?Sized> Copy for TensorDescription<'_, N> {}
impl<N: ?Sized> Clone for TensorDescription<'_, N> {
    fn clone(&self) -> Self {
        *self
    }
}
pub type TensorDescriptor<'a> = TensorDescription<'a, str>;
pub type RawTensorDescriptor<'a> = TensorDescription<'a, [u8]>;

pub(crate) trait WireBytes {
    fn wire_bytes(&self) -> &[u8];
}
impl WireBytes for str {
    fn wire_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}
impl WireBytes for String {
    fn wire_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}
impl WireBytes for [u8] {
    fn wire_bytes(&self) -> &[u8] {
        self
    }
}
impl WireBytes for Vec<u8> {
    fn wire_bytes(&self) -> &[u8] {
        self
    }
}

/// Checked offsets for streaming payloads after an independently written header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteLayout {
    /// Exact encoded header bytes before alignment padding. Producers such as
    /// the frozen HF vocab-only converter intentionally stop at this boundary.
    pub encoded_header_bytes: u64,
    pub data_offset: u64,
    pub alignment: u32,
    pub tensor_offsets: Vec<u64>,
    pub tensor_bytes: Vec<u64>,
    pub file_bytes: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct HeaderPolicy {
    frozen_tool: bool,
    native_header: bool,
    raw_strings: bool,
    converter_arrays: bool,
    version: u32,
    physical_alignment: Option<u32>,
}
impl HeaderPolicy {
    const STANDARD: Self = Self {
        frozen_tool: false,
        native_header: false,
        raw_strings: false,
        converter_arrays: false,
        version: 3,
        physical_alignment: None,
    };
    const RAW: Self = Self {
        raw_strings: true,
        ..Self::STANDARD
    };
    const SOURCE_CONVERSION: Self = Self {
        converter_arrays: true,
        ..Self::RAW
    };
    pub(crate) const FROZEN_TOOL: Self = Self {
        frozen_tool: true,
        raw_strings: true,
        physical_alignment: Some(32),
        ..Self::STANDARD
    };
    fn native(version: u32, alignment: u32) -> Self {
        Self {
            native_header: true,
            raw_strings: true,
            version,
            physical_alignment: Some(alignment),
            ..Self::STANDARD
        }
    }
    fn string(self, value: &[u8]) -> &[u8] {
        if self.frozen_tool {
            c_string(value)
        } else {
            value
        }
    }
    fn name(self, value: &[u8]) -> &[u8] {
        if self.frozen_tool || self.native_header {
            c_string(value)
        } else {
            value
        }
    }
    fn rank(self, dimensions: &[u64]) -> usize {
        if self.frozen_tool || self.native_header {
            dimensions
                .iter()
                .enumerate()
                .rev()
                .find(|&(i, &dimension)| i > 0 && dimension > 1)
                .map_or(1, |(i, _)| i + 1)
        } else {
            dimensions.len()
        }
    }
}

fn add(total: &mut u64, count: u64, limit: u64) -> Result<()> {
    *total = total
        .checked_add(count)
        .ok_or_else(|| Error::Limit("encoded header size overflow".into()))?;
    if *total > limit {
        return Err(Error::Limit(
            "encoded header exceeds metadata byte budget".into(),
        ));
    }
    Ok(())
}
fn string_size(text: &[u8], limits: &Limits, policy: HeaderPolicy) -> Result<u64> {
    let text = policy.string(text);
    if text.len() > limits.max_string_bytes {
        return Err(Error::Limit(
            "encoded string exceeds string byte budget".into(),
        ));
    }
    (text.len() as u64)
        .checked_add(8)
        .ok_or_else(|| Error::Limit("encoded string size overflow".into()))
}
fn value_size<S: WireBytes>(
    value: &MetadataValue<S>,
    limits: &Limits,
    policy: HeaderPolicy,
    total_values: &mut usize,
    array_depth: usize,
) -> Result<u64> {
    *total_values = total_values
        .checked_add(1)
        .filter(|count| *count <= limits.max_total_values)
        .ok_or_else(|| Error::Limit("encoded metadata value budget".into()))?;
    Ok(match value {
        MetadataValue::U8(_) | MetadataValue::I8(_) | MetadataValue::Bool(_) => 1,
        MetadataValue::U16(_) | MetadataValue::I16(_) => 2,
        MetadataValue::U32(_) | MetadataValue::I32(_) | MetadataValue::F32(_) => 4,
        MetadataValue::U64(_) | MetadataValue::I64(_) | MetadataValue::F64(_) => 8,
        MetadataValue::String(text) => string_size(text.wire_bytes(), limits, policy)?,
        MetadataValue::ByteString(bytes) => string_size(bytes, limits, policy)?,
        MetadataValue::Array {
            element_type,
            values,
        } => {
            if *element_type > 12 || (*element_type == 9 && !policy.converter_arrays) {
                return Err(invalid("heterogeneous or nested array"));
            }
            if array_depth >= 64 {
                return Err(Error::Limit("source conversion array depth".into()));
            }
            if policy.converter_arrays && array_depth != 0 && values.is_empty() {
                return Err(invalid("empty nested source conversion array"));
            }
            if values.len() > limits.max_array_values
                || values.len() > limits.max_total_values.saturating_sub(*total_values)
            {
                return Err(Error::Limit("encoded array value budget".into()));
            }
            let mut bytes = 12;
            for item in values {
                if item.type_id() != *element_type {
                    return Err(invalid("heterogeneous or nested array"));
                }
                add(
                    &mut bytes,
                    value_size(item, limits, policy, total_values, array_depth + 1)?,
                    limits.max_metadata_bytes,
                )?;
            }
            bytes
        }
    })
}

pub(crate) fn tensor_bytes<N: ?Sized>(tensor: &TensorDescription<'_, N>) -> Result<u64> {
    if tensor.dimensions.len() > 4 {
        return Err(invalid("tensor has more than four dimensions"));
    }
    dimension_elements(tensor.dimensions)?;
    let row = usize::try_from(tensor.dimensions.first().copied().unwrap_or(1))
        .map_err(|_| invalid("dimension exceeds usize"))?;
    tensor.dimensions.iter().skip(1).try_fold(
        tensor
            .dtype
            .row_bytes(row)
            .map_err(|error| invalid(error.to_string()))? as u64,
        |bytes, &dimension| {
            bytes
                .checked_mul(dimension)
                .ok_or_else(|| invalid("tensor size overflow"))
        },
    )
}

fn layout_inner<N: WireBytes + ?Sized, S: WireBytes>(
    metadata: &[(&N, &MetadataValue<S>)],
    tensors: &[TensorDescription<'_, N>],
    limits: &Limits,
    policy: HeaderPolicy,
) -> Result<WriteLayout> {
    if metadata.len() > limits.max_metadata_entries || tensors.len() > limits.max_tensors {
        return Err(Error::Limit(
            "encoded metadata or tensor count budget".into(),
        ));
    }
    let metadata_alignment = match metadata
        .iter()
        .find(|(key, _)| c_string(key.wire_bytes()) == b"general.alignment")
    {
        None => 32,
        Some((_, MetadataValue::U32(value))) => *value,
        _ => return Err(invalid("general.alignment must be UINT32")),
    };
    if !metadata_alignment.is_power_of_two() || metadata_alignment > 1024 * 1024 {
        return Err(invalid("invalid alignment"));
    }
    let alignment = policy.physical_alignment.unwrap_or(metadata_alignment);
    if !alignment.is_power_of_two() || alignment > 1024 * 1024 {
        return Err(invalid("invalid physical alignment"));
    }
    if !matches!(policy.version, 2 | 3) {
        return Err(Error::Unsupported(
            "writer supports GGUF versions 2 and 3".into(),
        ));
    }
    let mut header = 0;
    add(&mut header, 24, limits.max_metadata_bytes)?;
    let mut keys = BTreeSet::new();
    let mut values = 0;
    for &(key, value) in metadata {
        let key = key.wire_bytes();
        if key.is_empty()
            || (!policy.raw_strings && key.contains(&0))
            || !keys.insert(policy.string(key))
        {
            return Err(invalid("empty, NUL or duplicated metadata key"));
        }
        add(
            &mut header,
            string_size(key, limits, policy)?,
            limits.max_metadata_bytes,
        )?;
        add(&mut header, 4, limits.max_metadata_bytes)?;
        add(
            &mut header,
            value_size(value, limits, policy, &mut values, 0)?,
            limits.max_metadata_bytes,
        )?;
    }
    // A rank-zero tensor with an empty name still needs 24 encoded bytes.
    // Reject impossible descriptor counts before reserving per-tensor arrays.
    let minimum = u64::try_from(tensors.len())
        .ok()
        .and_then(|count| count.checked_mul(24))
        .ok_or_else(|| Error::Limit("encoded tensor descriptor count overflow".into()))?;
    if minimum > limits.max_metadata_bytes.saturating_sub(header) {
        return Err(Error::Limit(
            "tensor descriptors minimum encoded size exceeds remaining metadata budget".into(),
        ));
    }
    #[cfg(test)]
    LAYOUT_RESERVATIONS.with(|calls| calls.set(calls.get() + 1));
    let mut tensor_offsets = Vec::new();
    let mut lengths = Vec::new();
    tensor_offsets
        .try_reserve_exact(tensors.len())
        .map_err(|_| Error::Limit("writer tensor offset allocation".into()))?;
    lengths
        .try_reserve_exact(tensors.len())
        .map_err(|_| Error::Limit("writer tensor length allocation".into()))?;
    let mut names = BTreeSet::new();
    let mut offset = 0u64;
    for tensor in tensors {
        let original_name = tensor.name.wire_bytes();
        let name = policy.name(original_name);
        let duplicate_key = if policy.frozen_tool {
            name
        } else {
            original_name
        };
        if original_name.len() >= 64
            || (!policy.raw_strings && original_name.contains(&0))
            || !names.insert(duplicate_key)
        {
            return Err(invalid("invalid or duplicated tensor name"));
        }
        let bytes = tensor_bytes(tensor)?;
        let rank = policy.rank(tensor.dimensions);
        add(
            &mut header,
            string_size(name, limits, policy)?,
            limits.max_metadata_bytes,
        )?;
        add(&mut header, 16 + rank as u64 * 8, limits.max_metadata_bytes)?;
        tensor_offsets.push(offset);
        lengths.push(bytes);
        offset = offset
            .checked_add(align(bytes, alignment as u64)?)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
    }
    let data_offset = align(header, alignment as u64)?;
    if data_offset > limits.max_metadata_bytes {
        return Err(Error::Limit(
            "encoded header including padding exceeds budget".into(),
        ));
    }
    let file_bytes = data_offset
        .checked_add(offset)
        .ok_or_else(|| invalid("output file size overflow"))?;
    Ok(WriteLayout {
        encoded_header_bytes: header,
        data_offset,
        alignment,
        tensor_offsets,
        tensor_bytes: lengths,
        file_bytes,
    })
}

pub(crate) fn zeros<W: Write>(out: &mut W, mut count: u64) -> Result<()> {
    const ZERO: [u8; 4096] = [0; 4096];
    while count != 0 {
        let part = count.min(ZERO.len() as u64) as usize;
        out.write_all(&ZERO[..part])?;
        count -= part as u64;
    }
    Ok(())
}
struct Encoder<'a, W: Write> {
    out: &'a mut W,
    bytes: u64,
    policy: HeaderPolicy,
}
impl<W: Write> Encoder<'_, W> {
    fn raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.out.write_all(bytes)?;
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("encoded byte count overflow"))?;
        Ok(())
    }
    fn string(&mut self, value: &[u8]) -> Result<()> {
        let value = self.policy.string(value);
        self.raw(&(value.len() as u64).to_le_bytes())?;
        self.raw(value)
    }
    fn value<S: WireBytes>(&mut self, value: &MetadataValue<S>) -> Result<()> {
        match value {
            MetadataValue::U8(value) => self.raw(&[*value]),
            MetadataValue::I8(value) => self.raw(&[*value as u8]),
            MetadataValue::U16(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::I16(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::U32(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::I32(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::F32(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::Bool(value) => self.raw(&[u8::from(*value)]),
            MetadataValue::String(value) => self.string(value.wire_bytes()),
            MetadataValue::ByteString(value) => self.string(value),
            MetadataValue::U64(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::I64(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::F64(value) => self.raw(&value.to_le_bytes()),
            MetadataValue::Array {
                element_type,
                values,
            } => {
                self.raw(&element_type.to_le_bytes())?;
                self.raw(&(values.len() as u64).to_le_bytes())?;
                for value in values {
                    self.value(value)?;
                }
                Ok(())
            }
        }
    }
}
fn header_with_policy_inner<W: Write, N: WireBytes + ?Sized, S: WireBytes>(
    out: &mut W,
    metadata: &[(&N, &MetadataValue<S>)],
    tensors: &[TensorDescription<'_, N>],
    limits: &Limits,
    policy: HeaderPolicy,
) -> Result<WriteLayout> {
    let layout = layout_inner(metadata, tensors, limits, policy)?;
    let mut encoder = Encoder {
        out,
        bytes: 0,
        policy,
    };
    encoder.raw(b"GGUF")?;
    encoder.raw(&policy.version.to_le_bytes())?;
    encoder.raw(&(tensors.len() as u64).to_le_bytes())?;
    encoder.raw(&(metadata.len() as u64).to_le_bytes())?;
    for &(key, value) in metadata {
        encoder.string(key.wire_bytes())?;
        encoder.raw(&value.type_id().to_le_bytes())?;
        encoder.value(value)?;
    }
    for (tensor, &offset) in tensors.iter().zip(&layout.tensor_offsets) {
        encoder.string(policy.name(tensor.name.wire_bytes()))?;
        let rank = policy.rank(tensor.dimensions);
        encoder.raw(&(rank as u32).to_le_bytes())?;
        for dimension in 0..rank {
            encoder.raw(
                &tensor
                    .dimensions
                    .get(dimension)
                    .copied()
                    .unwrap_or(1)
                    .to_le_bytes(),
            )?;
        }
        encoder.raw(&tensor.dtype.id().to_le_bytes())?;
        encoder.raw(&offset.to_le_bytes())?;
    }
    let padding = layout
        .data_offset
        .checked_sub(encoder.bytes)
        .ok_or_else(|| invalid("header preflight and encoding disagree"))?;
    zeros(encoder.out, padding)?;
    Ok(layout)
}

pub(crate) fn header_with_policy<W: Write>(
    out: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[TensorDescriptor<'_>],
    limits: &Limits,
    policy: HeaderPolicy,
) -> Result<WriteLayout> {
    header_with_policy_inner(out, metadata, tensors, limits, policy)
}
pub(crate) fn raw_header_with_policy<W: Write>(
    out: &mut W,
    metadata: &[(&[u8], &RawValue)],
    tensors: &[RawTensorDescriptor<'_>],
    limits: &Limits,
    policy: HeaderPolicy,
) -> Result<WriteLayout> {
    header_with_policy_inner(out, metadata, tensors, limits, policy)
}
/// Write ordered byte strings without UTF-8 conversion or C-string truncation.
/// Names and dimensions retain their supplied wire representation.
pub fn write_raw_header<W: Write>(
    out: &mut W,
    metadata: &[(&[u8], &RawValue)],
    tensors: &[RawTensorDescriptor<'_>],
    limits: &Limits,
) -> Result<WriteLayout> {
    raw_header_with_policy(out, metadata, tensors, limits, HeaderPolicy::RAW)
}
/// Write the frozen copying tools' GGUF v3 header: native C-string projection,
/// trailing singleton rank collapse, and physical 32-byte alignment. Metadata
/// alignment is copied as data, as in the original tool. The caller is responsible
/// for setter ordering and duplicate C-prefix normalization before this call.
pub fn write_tool_header<W: Write>(
    out: &mut W,
    metadata: &[(&[u8], &RawValue)],
    tensors: &[RawTensorDescriptor<'_>],
    limits: &Limits,
) -> Result<WriteLayout> {
    raw_header_with_policy(out, metadata, tensors, limits, HeaderPolicy::FROZEN_TOOL)
}

/// Write the frozen Python converter's ordered metadata representation.
///
/// This explicitly separate compatibility path permits homogeneous nested
/// arrays emitted by the original converter, with a fixed 64-array-level bound.
/// Such output is outside strict GGUF reader admission. The normal writers and
/// readers continue to reject nested arrays. All ordinary value, string, count,
/// layout and total-byte limits are checked before the first output write.
pub fn write_source_conversion_header<W: Write>(
    out: &mut W,
    metadata: &[(&[u8], &RawValue)],
    tensors: &[RawTensorDescriptor<'_>],
    limits: &Limits,
) -> Result<WriteLayout> {
    raw_header_with_policy(
        out,
        metadata,
        tensors,
        limits,
        HeaderPolicy::SOURCE_CONVERSION,
    )
}

pub(crate) fn native_raw_header<W: Write>(
    out: &mut W,
    metadata: &[(&[u8], &RawValue)],
    tensors: &[RawTensorDescriptor<'_>],
    limits: &Limits,
    version: u32,
    alignment: u32,
) -> Result<WriteLayout> {
    raw_header_with_policy(
        out,
        metadata,
        tensors,
        limits,
        HeaderPolicy::native(version, alignment),
    )
}

/// Write a little-endian v3 header in the caller's metadata order, without
/// allocating a header-sized buffer or copying metadata arrays. The returned
/// offsets describe the payloads and their zero padding. Validation completes
/// before the first write; an output I/O failure can still leave partial bytes.
pub fn write_header<W: Write>(
    out: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[TensorDescriptor<'_>],
    limits: &Limits,
) -> Result<WriteLayout> {
    header_with_policy(out, metadata, tensors, limits, HeaderPolicy::STANDARD)
}

/// Ordered counterpart to the existing in-memory payload writer.
pub fn write_ordered<W: Write>(
    mut out: W,
    metadata: &[(&str, &Value)],
    tensors: &[TensorData<'_>],
) -> Result<()> {
    let limits = Limits::default();
    if tensors.len() > limits.max_tensors {
        return Err(Error::Limit("encoded tensor count budget".into()));
    }
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(tensors.len())
        .map_err(|_| Error::Limit("writer tensor descriptor allocation".into()))?;
    for tensor in tensors {
        let descriptor = TensorDescriptor {
            name: tensor.name,
            dimensions: tensor.dimensions,
            dtype: tensor.dtype,
        };
        if tensor_bytes(&descriptor)? != tensor.data.len() as u64 {
            return Err(invalid(format!(
                "tensor {} payload length mismatch",
                tensor.name
            )));
        }
        descriptors.push(descriptor);
    }
    let layout = write_header(&mut out, metadata, &descriptors, &limits)?;
    for (tensor, &bytes) in tensors.iter().zip(&layout.tensor_bytes) {
        out.write_all(tensor.data)?;
        zeros(&mut out, align(bytes, layout.alignment as u64)? - bytes)?;
    }
    Ok(())
}
pub(crate) fn write_map<W: Write>(
    out: W,
    metadata: &BTreeMap<String, Value>,
    tensors: &[TensorData<'_>],
) -> Result<()> {
    if metadata.len() > Limits::default().max_metadata_entries {
        return Err(Error::Limit("encoded metadata entry budget".into()));
    }
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(metadata.len())
        .map_err(|_| Error::Limit("writer metadata index allocation".into()))?;
    ordered.extend(metadata.iter().map(|(key, value)| (key.as_str(), value)));
    write_ordered(out, &ordered, tensors)
}

#[cfg(test)]
std::thread_local! {
    static LAYOUT_RESERVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gguf;
    use std::io::{self, Cursor};

    #[test]
    fn ordered_streaming_header_preserves_order_nan_bits_and_custom_alignment() {
        let first = Value::String("large\0value".into());
        let nan = Value::F32(f32::from_bits(0x7fa1_2345));
        let alignment = Value::U32(64);
        let metadata = [
            ("z", &first),
            ("a", &nan),
            ("general.alignment", &alignment),
        ];
        let descriptors = [TensorDescriptor {
            name: "x",
            dimensions: &[3],
            dtype: DType::I8,
        }];
        let mut bytes = Vec::new();
        let layout = write_header(&mut bytes, &metadata, &descriptors, &Limits::default()).unwrap();
        assert_eq!(layout.data_offset % 64, 0);
        assert_eq!(bytes.len() as u64, layout.data_offset);
        bytes.extend([4, 5, 6]);
        zeros(&mut bytes, 61).unwrap();
        assert_eq!(bytes.len() as u64, layout.file_bytes);
        let parsed = Gguf::read(Cursor::new(bytes), Limits::default()).unwrap();
        assert_eq!(
            parsed.metadata_order().collect::<Vec<_>>(),
            ["z", "a", "general.alignment"]
        );
        assert_eq!(parsed.metadata["z"], first);
        assert!(matches!(parsed.metadata["a"], Value::F32(x) if x.to_bits()==0x7fa1_2345));
    }

    struct NoWrite;
    impl Write for NoWrite {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            panic!("invalid metadata must be rejected before the first output write")
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn all_header_validation_precedes_output_and_enforces_padding_budget() {
        let invalid_array = Value::Array {
            element_type: 0,
            values: vec![Value::I8(2)],
        };
        assert!(
            write_header(
                &mut NoWrite,
                &[("x", &invalid_array)],
                &[],
                &Limits::default()
            )
            .is_err()
        );
        let alignment = Value::U32(64);
        let limits = Limits {
            max_metadata_bytes: 63,
            ..Limits::default()
        };
        assert!(matches!(
            write_header(
                &mut NoWrite,
                &[("general.alignment", &alignment)],
                &[],
                &limits
            ),
            Err(Error::Limit(_))
        ));
        let value = Value::U32(1);
        assert!(
            write_header(
                &mut NoWrite,
                &[("x", &value), ("x", &value)],
                &[],
                &Limits::default()
            )
            .is_err()
        );
        assert!(
            write_header(
                &mut NoWrite,
                &[],
                &[TensorDescriptor {
                    name: "x",
                    dimensions: &[i64::MAX as u64, 2],
                    dtype: DType::F32
                }],
                &Limits::default()
            )
            .is_err()
        );
    }
    #[test]
    fn impossible_header_budget_rejects_before_tensor_layout_reservations() {
        let descriptor = TensorDescriptor {
            name: "x",
            dimensions: &[],
            dtype: DType::I8,
        };
        let descriptors = vec![descriptor; 64];
        let limits = Limits {
            max_metadata_bytes: 32,
            ..Limits::default()
        };
        LAYOUT_RESERVATIONS.with(|calls| calls.set(0));
        assert!(matches!(
            write_header(&mut NoWrite, &[], &descriptors, &limits),
            Err(Error::Limit(_))
        ));
        assert_eq!(LAYOUT_RESERVATIONS.with(|calls| calls.get()), 0);
    }
}
