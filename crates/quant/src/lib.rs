//! Checked, endian-explicit GGML scalar and block codecs.
//!
//! Layouts and arithmetic are ported from the frozen llama.cpp snapshot
//! (ZIP SHA-256 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122).
//! Source: ggml/include/ggml.h, ggml/src/ggml-common.h, ggml-quants.c and ggml-impl.h.
//! Copyright (c) 2023-2026 The ggml authors. Distributed under the MIT license.
//!
//! This crate never casts input bytes to native structures. GGUF tensor data here
//! is little-endian; callers handling big-endian files must normalize it first.
//! Decoding covers every active type. Scalar encoders cover every source model
//! quantizer, including explicit importance weights where the source requires
//! them. Activation codecs and scalar conversion helpers remain distinct from
//! stored model weight support. Optimized CPU/backend parity is separate.
#![forbid(unsafe_code)]

pub mod cpu_math;
pub mod flash_attention;

mod activation;
mod conversion;
mod decode;
pub use conversion::{convert_f32_to_integer, convert_f32_to_integer_into};
mod encode;
mod encode_iq1;
mod encode_iq2;
mod encode_iq3;
mod encode_iq4;
mod encode_k;
mod iq_lattice;
mod iq_sort;
mod weighted;
pub use weighted::{
    quantize_with_weights, quantize_with_weights_into, requires_importance_weights,
};
mod encode_simple;
mod matvec;
mod matvec_extra;
mod tables;
pub use matvec::{dot_packed, matvec, matvec_activation_dtype};

pub use encode::{quantize, quantize_into};

/// A malformed layout, unrepresentable input or unsupported capability, never an implicit fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantError {
    InvalidType(u32),
    UnsupportedMatVec(DType),
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidMatrixShape { rows: usize, cols: usize },
    NonMultiple { elements: usize, block_size: usize },
    SizeOverflow,
    InvalidLength { expected: usize, actual: usize },
    AllocationFailed,
    UnsupportedQuantization(DType),
    UnsupportedWeightedQuantization(DType),
    ImportanceWeightsRequired(DType),
    InvalidImportanceLength { expected: usize, actual: usize },
    InvalidImportanceWeight { index: usize },
    UnsupportedIntegerConversion(DType),
    IntegerOutOfRange { dtype: DType, index: usize },
    NonFiniteInput { index: usize },
    UnrepresentableScale { block: usize },
}

impl std::fmt::Display for QuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedMatVec(dtype) => write!(
                f,
                "GGML CPU does not define a matrix-vector weight dot for {dtype:?}"
            ),
            Self::InvalidVectorLength { expected, actual } => {
                write!(f, "expected {expected} vector elements, got {actual}")
            }
            Self::InvalidMatrixShape { rows, cols } => {
                write!(f, "matrix dimensions must be positive, got {rows} x {cols}")
            }
            Self::UnsupportedIntegerConversion(dtype) => write!(
                f,
                "source F32 integer assignment is not defined for {dtype:?}"
            ),
            Self::IntegerOutOfRange { dtype, index } => write!(
                f,
                "element {index} is outside the representable range of {dtype:?}"
            ),
            Self::InvalidType(id) => write!(f, "GGML type {id} is unknown or reserved"),
            Self::NonMultiple {
                elements,
                block_size,
            } => write!(
                f,
                "row length {elements} is not a multiple of block size {block_size}"
            ),
            Self::SizeOverflow => f.write_str("tensor byte length exceeds addressable memory"),
            Self::InvalidLength { expected, actual } => {
                write!(f, "expected {expected} bytes, got {actual}")
            }
            Self::AllocationFailed => f.write_str("unable to allocate tensor output"),
            Self::UnsupportedWeightedQuantization(dtype) => write!(
                f,
                "importance-weighted row encoder is not available for {dtype:?}"
            ),
            Self::ImportanceWeightsRequired(dtype) => write!(
                f,
                "the source requires explicit importance weights for {dtype:?}"
            ),
            Self::InvalidImportanceLength { expected, actual } => {
                write!(f, "expected {expected} importance weights, got {actual}")
            }
            Self::InvalidImportanceWeight { index } => write!(
                f,
                "importance weight {index} must be finite and nonnegative"
            ),
            Self::UnsupportedQuantization(dtype) => {
                write!(f, "encoder for {dtype:?} has not been ported")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "non-finite quantizer input at element {index}")
            }
            Self::UnrepresentableScale { block } => {
                write!(f, "quantizer scale is not representable at block {block}")
            }
        }
    }
}
impl std::error::Error for QuantError {}

/// The snapshot's on-disk numeric IDs. Removed IDs deliberately remain absent.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    F8_E4M3 = 43,
}

impl DType {
    pub const ALL: [Self; 36] = [
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::IQ2_XXS,
        Self::IQ2_XS,
        Self::IQ3_XXS,
        Self::IQ1_S,
        Self::IQ4_NL,
        Self::IQ3_S,
        Self::IQ2_S,
        Self::IQ4_XS,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::IQ1_M,
        Self::BF16,
        Self::TQ1_0,
        Self::TQ2_0,
        Self::MXFP4,
        Self::NVFP4,
        Self::Q1_0,
        Self::Q2_0,
        Self::F8_E4M3,
    ];
    pub fn from_id(id: u32) -> Result<Self, QuantError> {
        Self::ALL
            .into_iter()
            .find(|x| x.id() == id)
            .ok_or(QuantError::InvalidType(id))
    }
    pub const fn id(self) -> u32 {
        self as u32
    }
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F8_E4M3 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::NVFP4 | Self::Q2_0 => 64,
            Self::Q1_0 => 128,
            _ => 256,
        }
    }
    /// Bytes per complete quantization block, not bytes per element.
    pub const fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::I8 | Self::F8_E4M3 => 1,
            Self::Q4_0 | Self::IQ4_NL | Self::Q1_0 | Self::Q2_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 | Self::NVFP4 => 36,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS | Self::TQ2_0 => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
            Self::TQ1_0 => 54,
            Self::MXFP4 => 17,
        }
    }
    pub fn row_bytes(self, elements: usize) -> Result<usize, QuantError> {
        if !elements.is_multiple_of(self.block_size()) {
            return Err(QuantError::NonMultiple {
                elements,
                block_size: self.block_size(),
            });
        }
        (elements / self.block_size())
            .checked_mul(self.type_size())
            .ok_or(QuantError::SizeOverflow)
    }
}

/// Decode exactly one or more full blocks. Trailing bytes are rejected.
pub fn dequantize(dtype: DType, bytes: &[u8], elements: usize) -> Result<Vec<f32>, QuantError> {
    validate(dtype, bytes.len(), elements)?;
    let mut out = Vec::new();
    out.try_reserve_exact(elements)
        .map_err(|_| QuantError::AllocationFailed)?;
    out.resize(elements, 0.0);
    dequantize_into(dtype, bytes, &mut out)?;
    Ok(out)
}

/// Decode into caller-owned output after validating the entire input.
pub fn dequantize_into(dtype: DType, bytes: &[u8], out: &mut [f32]) -> Result<(), QuantError> {
    validate(dtype, bytes.len(), out.len())?;
    for (block, values) in bytes
        .chunks_exact(dtype.type_size())
        .zip(out.chunks_exact_mut(dtype.block_size()))
    {
        decode::block(dtype, block, values);
    }
    Ok(())
}

fn validate(dtype: DType, bytes: usize, elements: usize) -> Result<(), QuantError> {
    let expected = dtype.row_bytes(elements)?;
    if bytes != expected {
        return Err(QuantError::InvalidLength {
            expected,
            actual: bytes,
        });
    }
    Ok(())
}

/// IEEE binary16 to binary32, including signed zero, subnormals and infinities.
/// NaNs are quieted as in the scalar upstream arithmetic conversion.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = (bits >> 10) & 31;
    let fraction = (bits & 1023) as u32;
    let result = match exp {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((fraction << shift & 1023) << 13)
        }
        31 => sign | 0x7f800000 | (fraction << 13) | if fraction != 0 { 0x00400000 } else { 0 },
        _ => sign | ((exp as u32 + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(result)
}

/// Binary32 to binary16 with ties-to-even rounding and signed canonical NaNs.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 255) as i32;
    let frac = bits & 0x7fffff;
    if exp == 255 {
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let e = exp - 127;
    if e > 15 {
        return sign | 0x7c00;
    }
    if e >= -14 {
        let mut mant = frac >> 13;
        let rest = frac & 8191;
        if rest > 4096 || (rest == 4096 && mant & 1 != 0) {
            mant += 1;
        }
        let magnitude = (((e + 15) as u32) << 10) + mant;
        return sign | magnitude as u16;
    }
    if e < -25 {
        return sign;
    }
    let mant = frac | 0x800000;
    let shift = (-e - 1) as u32;
    let mut rounded = mant >> shift;
    let rest = mant & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rest > half || (rest == half && rounded & 1 != 0) {
        rounded += 1;
    }
    sign | rounded as u16
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// BF16 ties-to-even; NaNs cannot round into infinity.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fffffff > 0x7f800000 {
        return ((bits >> 16) as u16) | 0x40;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// OCP E4M3 with ties-to-even rounding, finite saturation and canonical NaN.
/// Matches the frozen ggml_fp32_to_f8_e4m3 helper, including signed zero.
pub fn f32_to_f8_e4m3(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = ((bits >> 24) & 128) as u8;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return 0x7f;
    }
    if magnitude == 0 {
        return sign;
    }
    let x = f32::from_bits(magnitude);
    if x >= 448.0 {
        return sign | 0x7e;
    }
    if x < 0.015625 {
        return sign | (x * 512.0).round_ties_even() as u8;
    }
    let mut exponent = ((magnitude >> 23) & 255) - 120;
    let fraction = magnitude & 0x7f_ffff;
    let mut mantissa = fraction >> 20;
    let rest = fraction & 0x0f_ffff;
    if rest > 0x08_0000 || (rest == 0x08_0000 && mantissa & 1 != 0) {
        mantissa += 1;
    }
    if mantissa == 8 {
        mantissa = 0;
        exponent += 1;
    }
    if exponent > 15 || (exponent == 15 && mantissa > 6) {
        return sign | 0x7e;
    }
    sign | ((exponent as u8) << 3) | mantissa as u8
}

/// OCP E4M3 scalar: both 0x7f and 0xff are NaNs, no infinities.
pub fn f8_e4m3_to_f32(bits: u8) -> f32 {
    let abs = bits & 127;
    if abs == 127 {
        return f32::NAN;
    }
    let exp = (abs >> 3) & 15;
    let man = abs & 7;
    let value = if exp == 0 {
        man as f32 * (1.0 / 512.0)
    } else {
        (1.0 + man as f32 * 0.125) * f32::from_bits((exp as u32 + 120) << 23)
    };
    if bits & 128 != 0 { -value } else { value }
}

pub(crate) fn u16le(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}
pub(crate) fn u32le(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]])
}
pub(crate) fn half(b: &[u8], p: usize) -> f32 {
    f16_to_f32(u16le(b, p))
}
