//! Typed, owned native cache storage. F16 bits are retained across state I/O.
use super::{copy_values, invalid};
use crate::Result;
use std::ops::Range;

/// Supported source KV element types. These are the frozen GGML numeric IDs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum NativeKvType {
    #[default]
    F32 = 0,
    F16 = 1,
}
impl NativeKvType {
    pub const fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
    pub(crate) fn round(self, value: f32) -> f32 {
        match self {
            Self::F32 => value,
            Self::F16 => llama_quant::f16_to_f32(llama_quant::f32_to_f16(value)),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum KvBuffer {
    F32(Vec<f32>),
    F16(Vec<u16>),
}
impl KvBuffer {
    pub(super) fn new(kind: NativeKvType, count: usize) -> Result<Self> {
        fn zeroed<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| invalid("native typed cache allocation failed"))?;
            values.resize(count, T::default());
            Ok(values)
        }
        Ok(match kind {
            NativeKvType::F32 => Self::F32(zeroed(count)?),
            NativeKvType::F16 => Self::F16(zeroed(count)?),
        })
    }
    pub(super) fn kind(&self) -> NativeKvType {
        match self {
            Self::F32(_) => NativeKvType::F32,
            Self::F16(_) => NativeKvType::F16,
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
        }
    }
    pub(super) fn capacity_bytes(&self) -> Result<usize> {
        let capacity = match self {
            Self::F32(v) => v.capacity(),
            Self::F16(v) => v.capacity(),
        };
        capacity
            .checked_mul(self.kind().size())
            .ok_or_else(|| invalid("native typed cache accounting overflow"))
    }
    pub(super) fn try_clone(&self) -> Result<Self> {
        Ok(match self {
            Self::F32(v) => Self::F32(copy_values(v)?),
            Self::F16(v) => Self::F16(copy_values(v)?),
        })
    }
    pub(super) fn clear(&mut self) {
        match self {
            Self::F32(v) => v.fill(0.0),
            Self::F16(v) => v.fill(0),
        }
    }
    pub(super) fn get(&self, index: usize) -> f32 {
        match self {
            Self::F32(v) => v[index],
            Self::F16(v) => llama_quant::f16_to_f32(v[index]),
        }
    }
    pub(super) fn bits(&self, index: usize) -> u32 {
        match self {
            Self::F32(v) => v[index].to_bits(),
            Self::F16(v) => u32::from(v[index]),
        }
    }
    pub(super) fn set_bits(&mut self, index: usize, bits: u32) {
        match self {
            Self::F32(v) => v[index] = f32::from_bits(bits),
            Self::F16(v) => v[index] = bits as u16,
        }
    }
    pub(super) fn set(&mut self, index: usize, value: f32) {
        match self {
            Self::F32(v) => v[index] = value,
            Self::F16(v) => v[index] = llama_quant::f32_to_f16(value),
        }
    }
    pub(super) fn write_f32(&mut self, start: usize, source: &[f32]) {
        match self {
            Self::F32(v) => v[start..start + source.len()].copy_from_slice(source),
            Self::F16(v) => {
                for (destination, source) in v[start..start + source.len()].iter_mut().zip(source) {
                    *destination = llama_quant::f32_to_f16(*source);
                }
            }
        }
    }
    pub(super) fn copy_from(&mut self, source: &Self) -> Result<()> {
        if self.kind() != source.kind() || self.len() != source.len() {
            return Err(invalid("native typed cache copy mismatch"));
        }
        self.copy_range(0, source, 0..source.len())
    }
    pub(super) fn copy_range(
        &mut self,
        destination: usize,
        source: &Self,
        range: Range<usize>,
    ) -> Result<()> {
        if range.start > range.end || range.end > source.len() {
            return Err(invalid("native cache source range mismatch"));
        }
        let end = destination
            .checked_add(range.len())
            .filter(|end| *end <= self.len())
            .ok_or_else(|| invalid("native cache destination range overflow"))?;
        match (self, source) {
            (Self::F32(to), Self::F32(from)) => to[destination..end].copy_from_slice(&from[range]),
            (Self::F16(to), Self::F16(from)) => to[destination..end].copy_from_slice(&from[range]),
            _ => return Err(invalid("native cache copy element type mismatch")),
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub(super) struct StoredKvLayer {
    pub(super) keys: KvBuffer,
    pub(super) values: KvBuffer,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_storage_preserves_every_payload_without_conversion() {
        let mut original = KvBuffer::new(NativeKvType::F16, 65_536).unwrap();
        for bits in 0..65_536 {
            original.set_bits(bits, bits as u32);
        }
        let clone = original.try_clone().unwrap();
        let mut copied = KvBuffer::new(NativeKvType::F16, 65_536).unwrap();
        copied.copy_from(&clone).unwrap();
        for bits in 0..65_536 {
            assert_eq!(copied.bits(bits), bits as u32);
        }
        assert_eq!(copied.capacity_bytes().unwrap(), 131_072);
        copied.clear();
        assert!((0..65_536).all(|index| copied.bits(index) == 0));
    }
    #[test]
    fn typed_copy_rejects_mismatch_without_writes() {
        let source = KvBuffer::new(NativeKvType::F32, 8).unwrap();
        let mut destination = KvBuffer::new(NativeKvType::F16, 8).unwrap();
        destination.set_bits(0, 0x7c01);
        assert!(destination.copy_from(&source).is_err());
        assert!(destination.copy_range(7, &source, 0..2).is_err());
        assert_eq!(destination.bits(0), 0x7c01);
    }
}
