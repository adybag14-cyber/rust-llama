//! Reference TQ and FP4 encoders from the frozen ggml-quants.c.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{DType, QuantError, f32_to_f16, tables::KVALUES_FP4};

fn maximum(x: &[f32]) -> f32 {
    x.iter().fold(0.0_f32, |a, v| a.max(v.abs()))
}
fn mx_exponent(x: &[f32]) -> Option<u8> {
    let amax = maximum(x);
    if amax == 0.0 {
        return Some(0);
    }
    // Preserve the source log2f rounding, including values close to powers of 2.
    let exponent = amax.log2().floor() - 2.0 + 127.0;
    if (0.0..=255.0).contains(&exponent) {
        Some(exponent as u8)
    } else {
        // The original floating-to-u8 cast is undefined outside this range.
        None
    }
}
pub(crate) fn preflight_mxfp4(values: &[f32]) -> Result<(), QuantError> {
    for (block, x) in values.as_chunks::<32>().0.iter().enumerate() {
        if mx_exponent(x).is_none() {
            return Err(QuantError::UnrepresentableScale { block });
        }
    }
    Ok(())
}
fn e8m0_half(e: u8) -> f32 {
    f32::from_bits(if e < 2 {
        0x0020_0000 << e
    } else {
        (e as u32 - 1) << 23
    })
}
fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 127 {
        return 0.0;
    }
    let exponent = (x >> 3) & 15;
    let mantissa = x & 7;
    let value = if exponent == 0 {
        mantissa as f32 / 512.0
    } else {
        (1.0 + mantissa as f32 / 8.0) * f32::from_bits((exponent as u32 + 120) << 23)
    };
    value * 0.5
}
fn to_ue4m3(x: f32) -> u8 {
    if x <= 0.0 {
        return 0;
    }
    let x = x.min(448.0);
    let bits = x.to_bits();
    let mut exponent = ((bits >> 23) & 255) as i32 - 127 + 7;
    if exponent <= 0 {
        let mantissa = (x * 512.0 + 0.5) as i32;
        return mantissa.clamp(0, 7) as u8;
    }
    if exponent >= 15 {
        return 0x7e;
    }
    // This source UE4M3 helper rounds halfway upward rather than ties-to-even.
    let mut mantissa = ((bits >> 20) & 7) + ((bits >> 19) & 1);
    if mantissa > 7 {
        mantissa = 0;
        exponent += 1;
        if exponent >= 15 {
            return 0x7e;
        }
    }
    (exponent as u8) << 3 | mantissa as u8
}
fn best_fp4(value: f32, scale: f32) -> u8 {
    let mut best = 0;
    let mut error = (KVALUES_FP4[0] as f32 * scale - value).abs();
    for (i, &code) in KVALUES_FP4.iter().enumerate().skip(1) {
        let candidate = (code as f32 * scale - value).abs();
        if candidate < error {
            best = i;
            error = candidate;
        }
    }
    best as u8
}
pub(crate) fn fp4(t: DType, x: &[f32], out: &mut [u8]) {
    if t == DType::MXFP4 {
        let exponent = mx_exponent(x).expect("validated MXFP4 exponent");
        let scale = e8m0_half(exponent);
        out[0] = exponent;
        for j in 0..16 {
            out[1 + j] = best_fp4(x[j], scale) | (best_fp4(x[j + 16], scale) << 4);
        }
    } else {
        for sub in 0..4 {
            let values = &x[sub * 16..(sub + 1) * 16];
            let scale_byte = to_ue4m3(maximum(values) / 6.0);
            out[sub] = scale_byte;
            let scale = ue4m3_half(scale_byte);
            for j in 0..8 {
                out[4 + sub * 8 + j] =
                    best_fp4(values[j], scale) | (best_fp4(values[j + 8], scale) << 4);
            }
        }
    }
}
pub(crate) fn ternary(t: DType, x: &[f32], out: &mut [u8]) {
    let scale = maximum(x);
    let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    let scale_offset = if t == DType::TQ1_0 { 52 } else { 64 };
    out[scale_offset..scale_offset + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
    if t == DType::TQ1_0 {
        for (output_start, input_start, width) in [(0, 0, 32), (32, 160, 16)] {
            for j in 0..width {
                let mut q = 0_u16;
                for n in 0..5 {
                    let trit = (x[input_start + j + n * width] * inverse).round() as i32 + 1;
                    q = q * 3 + trit as u16;
                }
                out[output_start + j] = (q * 256).div_ceil(243) as u8;
            }
        }
        for j in 0..4 {
            let mut q = 0_u16;
            for n in 0..4 {
                let trit = (x[240 + j + n * 4] * inverse).round() as i32 + 1;
                q = q * 3 + trit as u16;
            }
            out[48 + j] = (q * 3 * 256).div_ceil(243) as u8;
        }
    } else {
        for group in 0..2 {
            for j in 0..32 {
                let mut q = 0;
                for n in 0..4 {
                    let trit = (x[group * 128 + j + n * 32] * inverse).round() as i32 + 1;
                    q |= ((trit & 3) as u8) << (2 * n);
                }
                out[group * 32 + j] = q;
            }
        }
    }
}
