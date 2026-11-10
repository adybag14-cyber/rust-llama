//! Exact Llama 3 RoPE factors for the two captured converter source profiles.
//!
//! The output is one complete little-endian F32 tensor. The source's 32-value
//! vector prefix and scalar tail are determined by the complete dimension;
//! callers must not generate independently chunked slices of one tensor.
//! This module allocates no heap memory. The caller owns and budgets the output.
//! See ROPE.md for source identities, parameter coercion and profile limits.

use crate::codec::SourceProfile;
use std::fmt;

mod torch_pow;

/// Parameters after the converter has resolved the full_attention mapping and
/// head_dim, or the integer hidden_size / num_attention_heads fallback.
/// Float parameters retain Python's binary64 values until each source F32 cast.
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub dimension: i64,
    pub theta: f64,
    pub factor: f64,
    pub low_frequency_factor: f64,
    pub high_frequency_factor: f64,
    pub original_context_length: f64,
}
impl Parameters {
    /// The frozen generate_extra_tensors defaults for a supplied head dimension.
    pub const fn new(dimension: i64) -> Self {
        Self {
            dimension,
            theta: 10000.0,
            factor: 8.0,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
            original_context_length: 8192.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeError {
    NegativeDimension,
    NonFiniteParameter(&'static str),
    ZeroFrequencyFactor(&'static str),
    SizeOverflow,
    OutputLength { expected: usize, actual: usize },
    UnsupportedProfile(SourceProfile),
}
impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeDimension => f.write_str("Llama 3 RoPE dimension is negative"),
            Self::NonFiniteParameter(name) => write!(f, "nonfinite Llama 3 RoPE parameter: {name}"),
            Self::ZeroFrequencyFactor(name) => {
                write!(f, "zero Llama 3 RoPE frequency factor: {name}")
            }
            Self::SizeOverflow => {
                f.write_str("Llama 3 RoPE tensor size overflows the slice domain")
            }
            Self::OutputLength { expected, actual } => write!(
                f,
                "Llama 3 RoPE output length {actual}, expected {expected}"
            ),
            Self::UnsupportedProfile(profile) => write!(
                f,
                "Llama 3 RoPE source profile {profile:?} is unavailable on this target"
            ),
        }
    }
}
impl std::error::Error for RopeError {}

/// Validate finite configuration parameters and return the source arange element count times four.
/// Zero and odd dimensions follow the source arange(0, dimension, 2).
/// Zero low/high factors are rejected even for an empty tensor, as in Python.
/// Other finite signs, underflow and overflow during F32 coercion are retained.
pub fn encoded_len(parameters: &Parameters) -> Result<usize, RopeError> {
    if parameters.dimension < 0 {
        return Err(RopeError::NegativeDimension);
    }
    for (name, value) in [
        ("theta", parameters.theta),
        ("factor", parameters.factor),
        ("low_frequency_factor", parameters.low_frequency_factor),
        ("high_frequency_factor", parameters.high_frequency_factor),
        (
            "original_context_length",
            parameters.original_context_length,
        ),
    ] {
        if !value.is_finite() {
            return Err(RopeError::NonFiniteParameter(name));
        }
    }
    if parameters.low_frequency_factor == 0.0 {
        return Err(RopeError::ZeroFrequencyFactor("low_frequency_factor"));
    }
    if parameters.high_frequency_factor == 0.0 {
        return Err(RopeError::ZeroFrequencyFactor("high_frequency_factor"));
    }
    // The source F32 arange computes its length in the CPU F64 accumulator.
    let elements = ((parameters.dimension as f64) / 2.0).ceil() as u64;
    let elements = usize::try_from(elements).map_err(|_| RopeError::SizeOverflow)?;
    let bytes = elements.checked_mul(4).ok_or(RopeError::SizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(RopeError::SizeOverflow);
    }
    Ok(bytes)
}

/// Validate shape, parameters and the requested profile before caller allocation.
pub fn encoded_len_with_profile(
    parameters: &Parameters,
    profile: SourceProfile,
) -> Result<usize, RopeError> {
    let bytes = encoded_len(parameters)?;
    if profile == SourceProfile::PortableFinite || profile != SourceProfile::current() {
        return Err(RopeError::UnsupportedProfile(profile));
    }
    Ok(bytes)
}

/// Generate using the measured source build associated with the current target.
pub fn generate_into(parameters: &Parameters, output: &mut [u8]) -> Result<(), RopeError> {
    generate_into_with_profile(parameters, SourceProfile::current(), output)
}

/// Generate one complete F32LE factors tensor, without allocating.
/// All errors are detected before the first output write.
///
/// Unlike the format codecs, this helper requires a matching captured source
/// profile and target: the source's scalar power tail uses the OS math runtime.
/// PortableFinite and foreign-host profiles return an explicit error.
pub fn generate_into_with_profile(
    parameters: &Parameters,
    profile: SourceProfile,
    output: &mut [u8],
) -> Result<(), RopeError> {
    let bytes = encoded_len_with_profile(parameters, profile)?;
    if output.len() != bytes {
        return Err(RopeError::OutputLength {
            expected: bytes,
            actual: output.len(),
        });
    }
    let arithmetic = Arithmetic::new(parameters, profile, bytes / 4);
    let vector_end = (bytes / 4) / 32 * 32;
    for (index, dst) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (_, _, power) = arithmetic.power(index, vector_end);
        let frequency = 1.0_f32 / power;
        let (_, _, _, value) = arithmetic.factor(frequency);
        dst.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

struct Arithmetic {
    elements: usize,
    dimension: f32,
    theta: f32,
    factor: f32,
    low: f32,
    width: f32,
    context: f32,
    low_wavelength: f32,
    high_wavelength: f32,
    domain_nan: u32,
}
impl Arithmetic {
    fn new(p: &Parameters, profile: SourceProfile, elements: usize) -> Self {
        Self {
            elements,
            dimension: p.dimension as f32,
            theta: p.theta as f32,
            factor: p.factor as f32,
            low: p.low_frequency_factor as f32,
            width: (p.high_frequency_factor - p.low_frequency_factor) as f32,
            context: p.original_context_length as f32,
            // These two operations take place in Python before the F32 Tensor
            // comparisons. Subtracting the two frequency factors is also F64.
            low_wavelength: (p.original_context_length / p.low_frequency_factor) as f32,
            high_wavelength: (p.original_context_length / p.high_frequency_factor) as f32,
            // SLEEF misc.h defines MSVC NAN through infinity - infinity;
            // GNU uses __builtin_nanf(""). Both original bit streams are fixed.
            domain_nan: match profile {
                SourceProfile::WindowsTorch211Numpy246 => 0xffc0_0000,
                _ => 0x7fc0_0000,
            },
        }
    }
    fn power(&self, index: usize, vector_end: usize) -> (f32, f32, f32) {
        // The checked byte count bounds all index arithmetic within usize.
        let arange = arange_value(index, self.elements);
        let exponent = arange / self.dimension;
        let power = if index < vector_end {
            torch_pow::vector_pow(self.theta, exponent, self.domain_nan)
        } else {
            scalar_power(self.theta, exponent)
        };
        (arange, exponent, power)
    }
    fn factor(&self, frequency: f32) -> (u8, f32, f32, f32) {
        // Tensor.__rtruediv__ computes reciprocal first, then multiplies the
        // weak scalar after its F32 conversion. Reassociation changes bits.
        let wavelength = (1.0_f32 / frequency) * (2.0 * std::f64::consts::PI) as f32;
        if wavelength < self.high_wavelength {
            (0, wavelength, f32::NAN, 1.0)
        } else if wavelength > self.low_wavelength {
            (1, wavelength, f32::NAN, self.factor)
        } else {
            let smooth = ((1.0_f32 / wavelength) * self.context - self.low) / self.width;
            let factor = 1.0_f32 / ((1.0_f32 - smooth) / self.factor + smooth);
            (2, wavelength, smooth, factor)
        }
    }
}

// arange uses REGISTER_DISPATCH (AVX2 here), not the pow kernel's
// ALSO_REGISTER_AVX512_DISPATCH. The source float accumulator type is F64;
// each 8-value vector first narrows its base to F32 before the F64 lane offset.
// Two such vectors form the 16-element prefix, with an F64 scalar remainder.
fn arange_value(index: usize, elements: usize) -> f32 {
    if index < elements / 16 * 16 {
        let base = ((index / 8 * 8) as f64 * 2.0) as f32;
        (f64::from(base) + (index % 8) as f64 * 2.0) as f32
    } else {
        (index as f64 * 2.0) as f32
    }
}

// Keep this as the measured runtime operation even when a caller supplies
// compile-time constants. No project C/C++ or vendor math library is linked.
#[inline(never)]
fn scalar_power(base: f32, exponent: f32) -> f32 {
    std::hint::black_box(base).powf(std::hint::black_box(exponent))
}

#[cfg(test)]
mod tests;
