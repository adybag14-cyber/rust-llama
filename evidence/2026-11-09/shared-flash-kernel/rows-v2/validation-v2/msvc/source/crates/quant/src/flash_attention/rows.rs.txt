//! Borrowed contiguous source I32 row maps for FLASH_ATTN_EXT_ROWS.
//! Storage bounds are checked here; physical K/V indices are checked by the
//! plan after execution work/byte preflight and before allocating any buffers.
use super::{Error, Result, add, mul};

#[derive(Clone, Copy, Debug)]
enum Storage<'a> {
    I32(&'a [i32]),
    Bytes(&'a [u8]),
}

/// The source row-map layout is [mask columns, query batches, 1, 1].
/// Each entry chooses a physical row from the shared batch-one K/V tensors.
/// Negative entries are padding. The original one-chunk kernel skips them;
/// the tiled kernel packs physical row zero, with the corresponding -Inf mask.
#[derive(Clone, Copy, Debug)]
pub struct KvRows<'a> {
    storage: Storage<'a>,
    shape: [usize; 2],
    offset: usize,
}
impl<'a> KvRows<'a> {
    pub fn from_i32(values: &'a [i32], shape: [usize; 2], byte_offset: usize) -> Result<Self> {
        Self::new(
            Storage::I32(values),
            mul(values.len(), 4)?,
            shape,
            byte_offset,
        )
    }
    pub fn from_bytes(values: &'a [u8], shape: [usize; 2], byte_offset: usize) -> Result<Self> {
        Self::new(Storage::Bytes(values), values.len(), shape, byte_offset)
    }
    fn new(storage: Storage<'a>, bytes: usize, shape: [usize; 2], offset: usize) -> Result<Self> {
        if shape.contains(&0) || shape.iter().any(|&n| n > i32::MAX as usize) {
            return Err(Error::Invalid(
                "kv_rows dimensions must be nonzero signed-I32 extents",
            ));
        }
        if !offset.is_multiple_of(4) {
            return Err(Error::Invalid("kv_rows byte offset must be I32 aligned"));
        }
        let end = add(offset, mul(mul(shape[0], shape[1])?, 4)?)?;
        if end > bytes {
            return Err(Error::Invalid("kv_rows exceeds its borrowed storage"));
        }
        Ok(Self {
            storage,
            shape,
            offset,
        })
    }
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub const fn byte_offset(&self) -> usize {
        self.offset
    }
    pub(super) fn len(&self) -> usize {
        self.shape[0] * self.shape[1]
    }
    pub(super) fn value(&self, column: usize, batch: usize) -> i32 {
        let index = batch * self.shape[0] + column;
        match self.storage {
            Storage::I32(values) => values[self.offset / 4 + index],
            Storage::Bytes(values) => {
                let at = self.offset + index * 4;
                i32::from_le_bytes(values[at..at + 4].try_into().unwrap())
            }
        }
    }
}
