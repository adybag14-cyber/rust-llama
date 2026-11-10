//! Byte-exact checkpoint preparation and frozen Python converter codecs.
//!
//! These are conversion utilities, separate from llama-quant's C runtime
//! quantizers. Torch preparation and NumPy output casts deliberately differ
//! for signaling NaNs. Inputs/outputs are dense little-endian byte ranges.
//! All validation precedes any output write; this module allocates no heap.
//!
//! Source: frozen conversion/base.py and gguf-py/gguf/quants.py (MIT).
//! NumPy reduction policy is derived from NumPy 2.4.6/2.2.6
//! loops_minmax.dispatch.c.src and SIMD math headers (BSD-3-Clause).
//! See CODEC.md and THIRD_PARTY_NOTICES.md for source/runtime qualifications.

use llama_checkpoint::DType as SourceType;
use llama_quant::DType as TargetType;
use std::fmt;

/// Intermediate dtype retained through row permutation and QKV concatenation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkingType {
    F16,
    F32,
}
impl WorkingType {
    pub const fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Captured Torch preparation and NumPy reduction policy, independent of Rust ISA.
/// Names describe source builds, not detected installed Python packages.
/// PortableFinite rejects profile-dependent nonfinite half promotion/packed
/// encoding; format-defined scalar casts remain available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceProfile {
    WindowsTorch211Numpy246,
    LinuxTorch211Numpy226,
    PortableFinite,
}
impl SourceProfile {
    /// Select the frozen reference profile associated with verified targets.
    /// This does not inspect or claim compatibility with an installed Python.
    pub const fn current() -> Self {
        if cfg!(all(
            target_arch = "x86_64",
            target_os = "windows",
            target_env = "msvc"
        )) {
            Self::WindowsTorch211Numpy246
        } else if cfg!(all(
            target_arch = "x86_64",
            target_os = "linux",
            target_env = "gnu"
        )) {
            Self::LinuxTorch211Numpy226
        } else {
            Self::PortableFinite
        }
    }
    const fn lanes(self) -> usize {
        match self {
            Self::WindowsTorch211Numpy246 => 8,
            Self::LinuxTorch211Numpy226 => 16,
            Self::PortableFinite => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    UnsupportedSource(SourceType),
    UnsupportedTarget(TargetType),
    InputLength {
        item_bytes: usize,
        actual: usize,
    },
    OutputLength {
        expected: usize,
        actual: usize,
    },
    Shape {
        elements: usize,
        row_width: usize,
    },
    /// The sole error corresponding to the source converter's F16 fallback.
    RowWidth {
        row_width: usize,
        block_size: usize,
    },
    SizeOverflow,
    UnverifiedNonfiniteProfile {
        index: usize,
    },
}
impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSource(t) => {
                write!(f, "frozen converter does not accept checkpoint dtype {t:?}")
            }
            Self::UnsupportedTarget(t) => {
                write!(f, "converter codec does not support output dtype {t:?}")
            }
            Self::InputLength { item_bytes, actual } => write!(
                f,
                "input length {actual} is not a multiple of scalar size {item_bytes}"
            ),
            Self::OutputLength { expected, actual } => {
                write!(f, "expected output length {expected}, got {actual}")
            }
            Self::Shape {
                elements,
                row_width,
            } => write!(
                f,
                "{elements} elements cannot form converter rows of width {row_width}"
            ),
            Self::RowWidth {
                row_width,
                block_size,
            } => write!(
                f,
                "row width {row_width} is not a multiple of quantization block size {block_size}"
            ),
            Self::SizeOverflow => f.write_str("converter codec byte size overflow"),
            Self::UnverifiedNonfiniteProfile { index } => write!(
                f,
                "conversion of nonfinite element {index} requires a measured encoder profile"
            ),
        }
    }
}
impl std::error::Error for CodecError {}

pub fn working_type(source: SourceType) -> Result<WorkingType, CodecError> {
    if !source.is_converter_dtype() {
        return Err(CodecError::UnsupportedSource(source));
    }
    Ok(if source == SourceType::F16 {
        WorkingType::F16
    } else {
        WorkingType::F32
    })
}

/// Required caller allocation for preparation. Raw F16/F32 no-op bits survive.
pub fn prepared_len(
    source: SourceType,
    input_len: usize,
    force_f32: bool,
) -> Result<(WorkingType, usize), CodecError> {
    let work = working_type(source)?;
    let bytes = source.bits() as usize / 8;
    let elements = scalar_count(input_len, bytes)?;
    let work = if force_f32 { WorkingType::F32 } else { work };
    let size = elements
        .checked_mul(work.bytes())
        .ok_or(CodecError::SizeOverflow)?;
    Ok((work, size))
}

/// Reproduce source Torch preparation. force_f32 is for actual mixed-type
/// Torch QKV promotion, not a request for NumPy's later F16-to-F32 output cast.
pub fn prepare_into(
    source: SourceType,
    raw_le: &[u8],
    force_f32: bool,
    output: &mut [u8],
) -> Result<WorkingType, CodecError> {
    prepare_into_with_profile(source, raw_le, force_f32, SourceProfile::current(), output)
}

/// Explicit captured preparation policy. The input must be one complete
/// contiguous component after any row permutation. Source Torch cat restarts
/// its conversion prefix/tail for each component; arbitrary chunking can change
/// the GNU scalar-tail NaN bytes.
pub fn prepare_into_with_profile(
    source: SourceType,
    raw_le: &[u8],
    force_f32: bool,
    profile: SourceProfile,
    output: &mut [u8],
) -> Result<WorkingType, CodecError> {
    let (work, expected) = prepared_len(source, raw_le.len(), force_f32)?;
    output_len(output, expected)?;
    if source == SourceType::F32 || work == WorkingType::F16 {
        output.copy_from_slice(raw_le);
        return Ok(work);
    }
    let bytes = source.bits() as usize / 8;
    let elements = raw_le.len() / bytes;
    if source == SourceType::F16 && profile == SourceProfile::PortableFinite {
        for (index, raw) in raw_le.as_chunks::<2>().0.iter().enumerate() {
            if u16::from_le_bytes(*raw) & 0x7fff > 0x7c00 {
                return Err(CodecError::UnverifiedNonfiniteProfile { index });
            }
        }
    }
    let promotion_prefix = elements / 8 * 8;
    for (index, (input, out)) in raw_le
        .chunks_exact(bytes)
        .zip(output.as_chunks_mut::<4>().0.iter_mut())
        .enumerate()
    {
        let value = match source {
            SourceType::F64 => {
                let bits = u64::from_le_bytes(input.try_into().unwrap());
                if bits & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000 {
                    f32::from_bits(
                        ((bits >> 32) as u32 & 0x8000_0000)
                            | 0x7fc0_0000
                            | ((bits >> 29) as u32 & 0x007f_ffff),
                    )
                } else {
                    f64::from_bits(bits) as f32
                }
            }
            SourceType::F16 => {
                let bits = u16::from_le_bytes(input.try_into().unwrap());
                if profile == SourceProfile::LinuxTorch211Numpy226
                    && index >= promotion_prefix
                    && bits & 0x7fff > 0x7c00
                {
                    // Captured GNU Torch cat promotion canonicalizes scalar-tail
                    // NaNs to 0x7fffffff; its full eight-value groups preserve
                    // quieted sign/payload. This is a compiled-profile rule.
                    f32::from_bits(0x7fff_ffff)
                } else {
                    llama_quant::f16_to_f32(bits)
                }
            }
            SourceType::Bf16 => {
                f32::from_bits(u32::from(u16::from_le_bytes(input.try_into().unwrap())) << 16)
            }
            SourceType::I64 => i64::from_le_bytes(input.try_into().unwrap()) as f32,
            SourceType::I32 => i32::from_le_bytes(input.try_into().unwrap()) as f32,
            SourceType::I16 => i16::from_le_bytes(input.try_into().unwrap()) as f32,
            SourceType::I8 => input[0] as i8 as f32,
            SourceType::U8 => f32::from(input[0]),
            SourceType::Bool => f32::from(u8::from(input[0] != 0)),
            SourceType::F8E4M3 => torch_e4m3(input[0]),
            SourceType::F8E5M2 => llama_quant::f16_to_f32(u16::from(input[0]) << 8),
            SourceType::F8E8M0 => f32::from_bits(match input[0] {
                0 => 0x0040_0000,
                255 => 0x7f80_0001,
                n => u32::from(n) << 23,
            }),
            _ => unreachable!("source dtype was validated before any output write"),
        };
        out.copy_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(work)
}

fn torch_e4m3(value: u8) -> f32 {
    let sign = u32::from(value & 0x80) << 24;
    let exp = u32::from((value >> 3) & 15);
    let fraction = u32::from(value & 7);
    if value & 0x7f == 0x7f {
        return f32::from_bits(sign | 0x7ff0_0000);
    }
    if exp == 0 {
        let magnitude = (fraction as f32 * (1.0 / 512.0)).to_bits();
        f32::from_bits(sign | magnitude)
    } else {
        f32::from_bits(sign | ((exp + 120) << 23) | (fraction << 20))
    }
}

fn scalar_count(bytes: usize, item_bytes: usize) -> Result<usize, CodecError> {
    if !bytes.is_multiple_of(item_bytes) {
        return Err(CodecError::InputLength {
            item_bytes,
            actual: bytes,
        });
    }
    Ok(bytes / item_bytes)
}
fn output_len(output: &[u8], expected: usize) -> Result<(), CodecError> {
    if output.len() != expected {
        Err(CodecError::OutputLength {
            expected,
            actual: output.len(),
        })
    } else {
        Ok(())
    }
}

/// Byte length for exactly the supplied rows, with no implicit width fallback.
/// A zero last dimension is accepted only by the direct F16/F32 NumPy casts;
/// the source block-codec row reshape rejects that shape.
pub fn encoded_len(
    work: WorkingType,
    input_len: usize,
    row_width: usize,
    target: TargetType,
) -> Result<usize, CodecError> {
    if !matches!(
        target,
        TargetType::F32
            | TargetType::F16
            | TargetType::BF16
            | TargetType::Q8_0
            | TargetType::TQ1_0
            | TargetType::TQ2_0
    ) {
        return Err(CodecError::UnsupportedTarget(target));
    }
    let elements = scalar_count(input_len, work.bytes())?;
    if row_width == 0 {
        if elements != 0 || !matches!(target, TargetType::F16 | TargetType::F32) {
            return Err(CodecError::Shape {
                elements,
                row_width,
            });
        }
    } else if !elements.is_multiple_of(row_width) {
        return Err(CodecError::Shape {
            elements,
            row_width,
        });
    }
    let block_size = target.block_size();
    if !row_width.is_multiple_of(block_size) {
        return Err(CodecError::RowWidth {
            row_width,
            block_size,
        });
    }
    // The Python ternary packers infer an axis with -1 while n_blocks is zero.
    // NumPy rejects that reshape even if a positive last dimension was valid.
    if elements == 0 && matches!(target, TargetType::TQ1_0 | TargetType::TQ2_0) {
        return Err(CodecError::Shape {
            elements,
            row_width,
        });
    }
    (elements / block_size)
        .checked_mul(target.type_size())
        .ok_or(CodecError::SizeOverflow)
}

pub fn encode_into(
    work: WorkingType,
    data_le: &[u8],
    row_width: usize,
    target: TargetType,
    output: &mut [u8],
) -> Result<(), CodecError> {
    encode_into_with_profile(
        work,
        data_le,
        row_width,
        target,
        SourceProfile::current(),
        output,
    )
}

/// Encode with a fixed measured source policy on any Rust host. This is useful
/// for reproducible conversion and cross-profile oracle comparisons.
pub fn encode_into_with_profile(
    work: WorkingType,
    data_le: &[u8],
    row_width: usize,
    target: TargetType,
    profile: SourceProfile,
    output: &mut [u8],
) -> Result<(), CodecError> {
    let expected = encoded_len(work, data_le.len(), row_width, target)?;
    output_len(output, expected)?;
    let packed = matches!(
        target,
        TargetType::Q8_0 | TargetType::TQ1_0 | TargetType::TQ2_0
    );
    if packed && profile == SourceProfile::PortableFinite {
        for (index, raw) in data_le.chunks_exact(work.bytes()).enumerate() {
            if !numpy_value(work, raw).is_finite() {
                return Err(CodecError::UnverifiedNonfiniteProfile { index });
            }
        }
    }
    if matches!(
        (work, target),
        (WorkingType::F16, TargetType::F16) | (WorkingType::F32, TargetType::F32)
    ) {
        output.copy_from_slice(data_le);
        return Ok(());
    }
    if !packed {
        for (raw, out) in data_le
            .chunks_exact(work.bytes())
            .zip(output.chunks_exact_mut(target.type_size()))
        {
            let value = numpy_value(work, raw);
            match target {
                TargetType::F32 => out.copy_from_slice(&value.to_bits().to_le_bytes()),
                TargetType::F16 => out.copy_from_slice(&numpy_half(value).to_le_bytes()),
                TargetType::BF16 => {
                    out.copy_from_slice(&llama_quant::f32_to_bf16(value).to_le_bytes())
                }
                _ => unreachable!("scalar target was validated"),
            }
        }
        return Ok(());
    }
    let block_size = target.block_size();
    // Fixed stack storage, independent of tensor shape and row count.
    let mut values = [0.0f32; 256];
    let mut symbols = [0u8; 256];
    for (raw, out) in data_le
        .chunks_exact(block_size * work.bytes())
        .zip(output.chunks_exact_mut(target.type_size()))
    {
        for (value, input) in values[..block_size]
            .iter_mut()
            .zip(raw.chunks_exact(work.bytes()))
        {
            *value = numpy_value(work, input);
        }
        let maximum = numpy_max_abs(&values[..block_size], profile);
        let scale = if target == TargetType::Q8_0 {
            if maximum.is_nan() {
                f32::from_bits(maximum.to_bits() | 0x0040_0000)
            } else {
                maximum / 127.0
            }
        } else {
            maximum
        };
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        if target == TargetType::Q8_0 {
            out[..2].copy_from_slice(&numpy_half(scale).to_le_bytes());
            for (&value, byte) in values[..32].iter().zip(&mut out[2..]) {
                *byte = numpy_round_i8(value * inverse) as u8;
            }
        } else {
            for (&value, symbol) in values.iter().zip(&mut symbols) {
                *symbol = numpy_round_i8(value * inverse).wrapping_add(1) as u8;
            }
            if target == TargetType::TQ2_0 {
                for group in 0..2 {
                    for i in 0..32 {
                        let base = group * 128 + i;
                        out[group * 32 + i] = symbols[base]
                            | symbols[base + 32] << 2
                            | symbols[base + 64] << 4
                            | symbols[base + 96] << 6;
                    }
                }
                out[64..].copy_from_slice(&numpy_half(scale).to_le_bytes());
            } else {
                for i in 0..32 {
                    let sum = u16::from(symbols[i]) * 81
                        + u16::from(symbols[32 + i]) * 27
                        + u16::from(symbols[64 + i]) * 9
                        + u16::from(symbols[96 + i]) * 3
                        + u16::from(symbols[128 + i]);
                    out[i] = (sum * 256).div_ceil(243) as u8;
                }
                for i in 0..16 {
                    let sum = u16::from(symbols[160 + i]) * 81
                        + u16::from(symbols[176 + i]) * 27
                        + u16::from(symbols[192 + i]) * 9
                        + u16::from(symbols[208 + i]) * 3
                        + u16::from(symbols[224 + i]);
                    out[32 + i] = (sum * 256).div_ceil(243) as u8;
                }
                for i in 0..4 {
                    let sum = u16::from(symbols[240 + i]) * 81
                        + u16::from(symbols[244 + i]) * 27
                        + u16::from(symbols[248 + i]) * 9
                        + u16::from(symbols[252 + i]) * 3;
                    out[48 + i] = (sum * 256).div_ceil(243) as u8;
                }
                out[52..].copy_from_slice(&numpy_half(scale).to_le_bytes());
            }
        }
    }
    Ok(())
}

/// NumPy half widening preserves signaling payloads, unlike Torch's cast.
fn numpy_value(work: WorkingType, raw: &[u8]) -> f32 {
    match work {
        WorkingType::F32 => f32::from_bits(u32::from_le_bytes(raw.try_into().unwrap())),
        WorkingType::F16 => {
            let bits = u16::from_le_bytes(raw.try_into().unwrap());
            if bits & 0x7fff > 0x7c00 {
                f32::from_bits(
                    (u32::from(bits & 0x8000) << 16)
                        | 0x7f80_0000
                        | (u32::from(bits & 0x03ff) << 13),
                )
            } else {
                llama_quant::f16_to_f32(bits)
            }
        }
    }
}

/// NumPy preserves the high payload/signaling bit when narrowing NaNs.
/// A lost low-only payload becomes one, so a NaN never becomes infinity.
fn numpy_half(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        ((bits >> 16) as u16 & 0x8000) | 0x7c00 | (((bits >> 13) & 0x03ff) as u16).max(1)
    } else {
        llama_quant::f32_to_f16(value)
    }
}

/// NumPy SIMD maximum reduction canonicalizes a NaN in its vector prefix.
/// The original reduction seeds from element zero, consumes complete vectors
/// starting at element one, then preserves the first scalar-tail NaN payload.
fn numpy_max_abs(values: &[f32], profile: SourceProfile) -> f32 {
    let prefix = (values.len() - 1) / profile.lanes() * profile.lanes() + 1;
    let mut maximum = 0.0f32;
    for &value in &values[..prefix] {
        let absolute = f32::from_bits(value.to_bits() & 0x7fff_ffff);
        if absolute.is_nan() {
            maximum = f32::from_bits(0x7fc0_0000);
        } else if !maximum.is_nan() && absolute > maximum {
            maximum = absolute;
        }
    }
    for &value in &values[prefix..] {
        let absolute = f32::from_bits(value.to_bits() & 0x7fff_ffff);
        if !maximum.is_nan() && (absolute.is_nan() || absolute > maximum) {
            maximum = absolute;
        }
    }
    maximum
}

/// np_roundf uses round-away-from-zero. Nonfinite results cast to int8 zero
/// in both measured NumPy builds (the low byte of the invalid conversion).
fn numpy_round_i8(value: f32) -> i8 {
    if value.is_finite() {
        value.round() as i8
    } else {
        0
    }
}
