//! Checked four-dimensional GGML layouts, immutable buffer views and a Rust CPU DAG.
//! Maintained implementation is Rust. The full operation registry is not a support claim.
#![forbid(unsafe_code)]
mod cpu;
mod cpu_attention;
mod cpu_conv;
mod cpu_custom;
mod cpu_extra;
mod cpu_flash;
mod cpu_index;
mod cpu_lightning;
mod cpu_optimizer;
mod cpu_recurrent;
mod cpu_rope_back;
mod cpu_vision_loss;
mod graph;
mod native_sort;
mod registry;
pub use cpu_custom::{CustomKernel, CustomOptions, CustomTask};
pub use cpu_flash::{FlashAttentionError, FlashAttentionOptions};
pub use cpu_optimizer::{AdamWState, OptimizerLimits, adamw_step, sgd_step};
pub use cpu_rope_back::{RopeBackMode, RopeBackOptions};
pub use graph::{
    AdamWNodes, Conv2dParams, Conv3dParams, Graph, NodeId, Op, PoolMode, ResizeMode, RopeMode,
    SortOrder, Triangle,
};
pub use llama_quant::DType;
pub use registry::{GluOp, OpCode, UnaryOp};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    Layout(&'static str),
    Overflow,
    Bounds,
    AllocationFailed,
    Type { expected: DType, actual: DType },
    Shape(&'static str),
    InvalidNode(usize),
    InvalidArity { expected: usize, actual: usize },
    Unsupported(&'static str),
    Parameters(&'static str),
    Cancelled,
    Custom(&'static str),
    Quant(llama_quant::QuantError),
    FlashAttention(FlashAttentionError),
}
impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layout(m)
            | Self::Shape(m)
            | Self::Unsupported(m)
            | Self::Parameters(m)
            | Self::Custom(m) => f.write_str(m),
            Self::Overflow => f.write_str("tensor size or address overflow"),
            Self::Bounds => f.write_str("tensor view or index is outside its buffer"),
            Self::AllocationFailed => f.write_str("tensor allocation failed"),
            Self::Type { expected, actual } => write!(f, "expected {expected:?}, got {actual:?}"),
            Self::InvalidNode(id) => write!(f, "invalid graph node {id}"),
            Self::InvalidArity { expected, actual } => {
                write!(f, "expected {expected} inputs, got {actual}")
            }
            Self::Cancelled => f.write_str("graph execution cancelled"),
            Self::Quant(e) => e.fmt(f),
            Self::FlashAttention(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for TensorError {}
impl From<llama_quant::QuantError> for TensorError {
    fn from(e: llama_quant::QuantError) -> Self {
        Self::Quant(e)
    }
}
pub type Result<T> = std::result::Result<T, TensorError>;

/// Dimension 0 is the fastest dimension. For block types, nb[0] is the
/// byte distance between blocks, and ne[0] counts logical scalar elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub dtype: DType,
    pub ne: [usize; 4],
    pub nb: [usize; 4],
}
impl TensorDesc {
    pub fn contiguous(dtype: DType, ne: [usize; 4]) -> Result<Self> {
        let nb0 = dtype.type_size();
        let nb1 = dtype.row_bytes(ne[0])?;
        let nb2 = nb1.checked_mul(ne[1]).ok_or(TensorError::Overflow)?;
        let nb3 = nb2.checked_mul(ne[2]).ok_or(TensorError::Overflow)?;
        Self::with_strides(dtype, ne, [nb0, nb1, nb2, nb3])
    }
    pub fn with_strides(dtype: DType, ne: [usize; 4], nb: [usize; 4]) -> Result<Self> {
        let desc = Self { dtype, ne, nb };
        desc.validate()?;
        Ok(desc)
    }
    pub fn validate(&self) -> Result<()> {
        self.dtype.row_bytes(self.ne[0])?;
        self.elements()?;
        if !self.ne.contains(&0) && self.nb[0] < self.dtype.type_size() {
            return Err(TensorError::Layout(
                "axis-0 block stride is smaller than its storage size",
            ));
        }
        self.byte_span()?;
        Ok(())
    }
    pub fn elements(&self) -> Result<usize> {
        if self.ne.contains(&0) {
            return Ok(0);
        }
        self.ne
            .into_iter()
            .try_fold(1usize, |n, v| n.checked_mul(v).ok_or(TensorError::Overflow))
    }
    /// Smallest byte span containing every element; holes are included.
    pub fn byte_span(&self) -> Result<usize> {
        if self.ne.contains(&0) {
            return Ok(0);
        }
        if !self.ne[0].is_multiple_of(self.dtype.block_size()) {
            return Err(TensorError::Layout(
                "axis-0 length does not contain complete quantization blocks",
            ));
        }
        let blocks = self.ne[0] / self.dtype.block_size();
        let mut span = self.dtype.type_size();
        for (n, stride) in [blocks, self.ne[1], self.ne[2], self.ne[3]]
            .into_iter()
            .zip(self.nb)
        {
            span = span
                .checked_add((n - 1).checked_mul(stride).ok_or(TensorError::Overflow)?)
                .ok_or(TensorError::Overflow)?;
        }
        Ok(span)
    }
    pub fn is_contiguous(&self) -> bool {
        if self.ne.contains(&0) {
            return true;
        }
        let mut stride = self.dtype.type_size();
        let dims = [
            self.ne[0] / self.dtype.block_size(),
            self.ne[1],
            self.ne[2],
            self.ne[3],
        ];
        for (n, nb) in dims.into_iter().zip(self.nb) {
            if n > 1 && nb != stride {
                return false;
            }
            let Some(next) = stride.checked_mul(n) else {
                return false;
            };
            stride = next;
        }
        true
    }
}

/// Storage is immutable and reference counted. Views can alias safely because
/// every numerical operation allocates its output; no mutable slice is exposed.
#[derive(Clone, Debug)]
pub struct Tensor {
    desc: TensorDesc,
    data: Arc<[u8]>,
    offset: usize,
}
impl Tensor {
    pub fn from_bytes(desc: TensorDesc, bytes: Vec<u8>) -> Result<Self> {
        desc.validate()?;
        if desc.byte_span()? != bytes.len() {
            return Err(TensorError::Layout(
                "owned buffer length must equal the descriptor byte span",
            ));
        }
        Ok(Self {
            desc,
            data: bytes.into(),
            offset: 0,
        })
    }
    pub fn from_f32(ne: [usize; 4], values: &[f32]) -> Result<Self> {
        let desc = TensorDesc::contiguous(DType::F32, ne)?;
        if desc.elements()? != values.len() {
            return Err(TensorError::Shape("F32 data length differs from shape"));
        }
        let mut data = reserved(desc.byte_span()?)?;
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_bytes(desc, data)
    }
    pub fn from_i32(ne: [usize; 4], values: &[i32]) -> Result<Self> {
        let desc = TensorDesc::contiguous(DType::I32, ne)?;
        if desc.elements()? != values.len() {
            return Err(TensorError::Shape("I32 data length differs from shape"));
        }
        let mut data = reserved(desc.byte_span()?)?;
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_bytes(desc, data)
    }
    pub fn from_i64(ne: [usize; 4], values: &[i64]) -> Result<Self> {
        let desc = TensorDesc::contiguous(DType::I64, ne)?;
        if desc.elements()? != values.len() {
            return Err(TensorError::Shape("I64 data length differs from shape"));
        }
        let mut data = reserved(desc.byte_span()?)?;
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_bytes(desc, data)
    }
    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }
    pub fn shape(&self) -> [usize; 4] {
        self.desc.ne
    }
    pub fn dtype(&self) -> DType {
        self.desc.dtype
    }
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
    pub fn view(&self, desc: TensorDesc, byte_offset: usize) -> Result<Self> {
        desc.validate()?;
        if desc.dtype != self.dtype() {
            return Err(TensorError::Layout(
                "a view cannot reinterpret its storage dtype",
            ));
        }
        if !byte_offset.is_multiple_of(self.dtype().type_size()) {
            return Err(TensorError::Layout(
                "view offset must address a complete dtype block",
            ));
        }
        let span = desc.byte_span()?;
        if byte_offset.checked_add(span).ok_or(TensorError::Overflow)? > self.desc.byte_span()? {
            return Err(TensorError::Bounds);
        }
        let offset = self
            .offset
            .checked_add(byte_offset)
            .ok_or(TensorError::Overflow)?;
        if offset.checked_add(span).ok_or(TensorError::Overflow)? > self.data.len() {
            return Err(TensorError::Bounds);
        }
        Ok(Self {
            desc,
            data: Arc::clone(&self.data),
            offset,
        })
    }
    /// Axes follow GGML's mapping: axes[old_axis] = new_axis.
    pub fn permute(&self, axes: [usize; 4]) -> Result<Self> {
        let mut seen = [false; 4];
        for a in axes {
            if a >= 4 || seen[a] {
                return Err(TensorError::Parameters("axes must be a permutation"));
            }
            seen[a] = true;
        }
        if self.dtype().block_size() > 1 && axes[0] != 0 {
            return Err(TensorError::Unsupported(
                "moving a quantized block axis is not supported",
            ));
        }
        let mut ne = [0; 4];
        let mut nb = [0; 4];
        for old in 0..4 {
            ne[axes[old]] = self.desc.ne[old];
            nb[axes[old]] = self.desc.nb[old];
        }
        self.view(TensorDesc::with_strides(self.dtype(), ne, nb)?, 0)
    }
    pub fn reshape(&self, ne: [usize; 4]) -> Result<Self> {
        if !self.desc.is_contiguous() {
            return Err(TensorError::Layout(
                "reshape requires contiguous logical storage",
            ));
        }
        let desc = TensorDesc::contiguous(self.dtype(), ne)?;
        if desc.elements()? != self.desc.elements()? {
            return Err(TensorError::Shape("reshape changes element count"));
        }
        self.view(desc, 0)
    }
    /// Copy logical blocks in GGML element order, excluding view padding.
    /// Unlike floating extraction, this preserves integer and NaN payload bits.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let target = TensorDesc::contiguous(self.dtype(), self.shape())?;
        let mut bytes = reserved(target.byte_span()?)?;
        self.for_each_block(|range| bytes.extend_from_slice(&self.data[range]));
        Ok(bytes)
    }
    pub fn contiguous(&self) -> Result<Self> {
        let target = TensorDesc::contiguous(self.dtype(), self.shape())?;
        let mut bytes = reserved(target.byte_span()?)?;
        self.for_each_block(|range| bytes.extend_from_slice(&self.data[range]));
        Self::from_bytes(target, bytes)
    }
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let mut values = zeros(self.desc.elements()?)?;
        let block_size = self.dtype().block_size();
        let mut out = 0;
        let mut error = None;
        self.for_each_block(|range| {
            if let Err(e) = llama_quant::dequantize_into(
                self.dtype(),
                &self.data[range],
                &mut values[out..out + block_size],
            ) {
                error = Some(e);
            }
            out += block_size;
        });
        if let Some(e) = error {
            return Err(e.into());
        }
        Ok(values)
    }
    pub fn to_i32(&self) -> Result<Vec<i32>> {
        if self.dtype() != DType::I32 {
            return Err(TensorError::Type {
                expected: DType::I32,
                actual: self.dtype(),
            });
        }
        let mut values = reserved(self.desc.elements()?)?;
        self.for_each_block(|range| {
            values.push(i32::from_le_bytes(
                self.data[range].try_into().expect("validated I32"),
            ))
        });
        Ok(values)
    }
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        if self.dtype() != DType::I64 {
            return Err(TensorError::Type {
                expected: DType::I64,
                actual: self.dtype(),
            });
        }
        let mut values = reserved(self.desc.elements()?)?;
        self.for_each_block(|range| {
            values.push(i64::from_le_bytes(
                self.data[range].try_into().expect("validated I64"),
            ));
        });
        Ok(values)
    }
    fn for_each_block(&self, mut visit: impl FnMut(std::ops::Range<usize>)) {
        if self.shape().contains(&0) {
            return;
        }
        let [n0, n1, n2, n3] = self.shape();
        let [s0, s1, s2, s3] = self.desc.nb;
        for i3 in 0..n3 {
            for i2 in 0..n2 {
                for i1 in 0..n1 {
                    for i0 in 0..n0 / self.dtype().block_size() {
                        let start = self.offset + i3 * s3 + i2 * s2 + i1 * s1 + i0 * s0;
                        visit(start..start + self.dtype().type_size());
                    }
                }
            }
        }
    }
}
pub(crate) fn reserved<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| TensorError::AllocationFailed)?;
    Ok(v)
}
pub(crate) fn zeros(n: usize) -> Result<Vec<f32>> {
    let mut v = reserved(n)?;
    v.resize(n, 0.0);
    Ok(v)
}
