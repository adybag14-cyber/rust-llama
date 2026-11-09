use crate::{DType, QuantError, f32_to_bf16, f32_to_f16};

/// Encode reference scalar, legacy, K, ternary and FP4 formats.
/// IQ encoders and importance-weighted encoding remain separate capabilities.
/// Other encoders return an explicit error. Quantized inputs must be finite and
/// their scale calculation must not overflow; IEEE scalar encoders permit NaNs.
pub fn quantize(dtype: DType, values: &[f32]) -> Result<Vec<u8>, QuantError> {
    let bytes = dtype.row_bytes(values.len())?;
    supported(dtype)?;
    if crate::encode_k::supports(dtype) {
        return crate::encode_k::encode(dtype, values);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(bytes)
        .map_err(|_| QuantError::AllocationFailed)?;
    out.resize(bytes, 0);
    quantize_into(dtype, values, &mut out)?;
    Ok(out)
}

/// Encode into a precisely sized buffer. Validation precedes output mutation.
pub fn quantize_into(dtype: DType, values: &[f32], out: &mut [u8]) -> Result<(), QuantError> {
    crate::validate(dtype, out.len(), values.len())?;
    supported(dtype)?;
    if crate::encode_k::supports(dtype) {
        let staged = crate::encode_k::encode(dtype, values)?;
        out.copy_from_slice(&staged);
        return Ok(());
    }
    let scalar = matches!(
        dtype,
        DType::F32 | DType::F16 | DType::BF16 | DType::F64 | DType::F8_E4M3
    );
    if !scalar {
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(QuantError::NonFiniteInput { index });
            }
        }
        if dtype == DType::Q8_K {
            crate::activation::preflight_q8k(values)?;
        } else if dtype == DType::MXFP4 {
            crate::encode_simple::preflight_mxfp4(values)?;
        } else if dtype != DType::NVFP4 {
            for (block, v) in values.chunks_exact(dtype.block_size()).enumerate() {
                let (d, m) = scales(dtype, v);
                if !d.is_finite()
                    || !m.is_finite()
                    || (d != 0.0 && !(1.0 / d).is_finite())
                    || (f32_to_f16(d) & 0x7c00) == 0x7c00
                    || (f32_to_f16(m) & 0x7c00) == 0x7c00
                {
                    return Err(QuantError::UnrepresentableScale { block });
                }
            }
        }
    }
    for (v, b) in values
        .chunks_exact(dtype.block_size())
        .zip(out.chunks_exact_mut(dtype.type_size()))
    {
        block(dtype, v, b);
    }
    Ok(())
}

fn supported(t: DType) -> Result<(), QuantError> {
    use DType::*;
    match t {
        F32 | F16 | BF16 | F64 | Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | F8_E4M3 | TQ1_0 | TQ2_0
        | MXFP4 | NVFP4 | Q8_K | Q1_0 | Q2_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => Ok(()),
        _ => Err(QuantError::UnsupportedQuantization(t)),
    }
}

fn scales(t: DType, x: &[f32]) -> (f32, f32) {
    use DType::*;
    match t {
        Q1_0 => (
            x.iter().fold(0.0, |sum, v| sum + v.abs()) / x.len() as f32,
            0.0,
        ),
        TQ1_0 | TQ2_0 => (x.iter().fold(0.0_f32, |a, v| a.max(v.abs())), 0.0),
        Q2_0 | Q8_0 | Q8_1 => {
            let amax = x.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
            (if t == Q2_0 { amax } else { amax / 127.0 }, 0.0)
        }
        Q4_0 | Q5_0 => {
            let mut amax = 0.0_f32;
            let mut max = 0.0_f32;
            for &v in x {
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            (max / if t == Q4_0 { -8.0 } else { -16.0 }, 0.0)
        }
        Q4_1 | Q5_1 => {
            let min = x.iter().fold(f32::MAX, |a, &v| if v < a { v } else { a });
            let max = x.iter().fold(-f32::MAX, |a, &v| if v > a { v } else { a });
            ((max - min) / if t == Q4_1 { 15.0 } else { 31.0 }, min)
        }
        _ => (0.0, 0.0),
    }
}

fn put_half(b: &mut [u8], p: usize, v: f32) {
    b[p..p + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
}

fn block(t: DType, x: &[f32], b: &mut [u8]) {
    use DType::*;
    match t {
        Q8_K => crate::activation::encode_q8k(x, b),
        TQ1_0 | TQ2_0 => crate::encode_simple::ternary(t, x, b),
        MXFP4 | NVFP4 => crate::encode_simple::fp4(t, x, b),
        F8_E4M3 => b[0] = crate::f32_to_f8_e4m3(x[0]),
        F32 => b.copy_from_slice(&x[0].to_le_bytes()),
        F16 => b.copy_from_slice(&f32_to_f16(x[0]).to_le_bytes()),
        BF16 => b.copy_from_slice(&f32_to_bf16(x[0]).to_le_bytes()),
        F64 => b.copy_from_slice(&(x[0] as f64).to_le_bytes()),
        _ => {
            b.fill(0);
            let (d, m) = scales(t, x);
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            put_half(b, 0, d);
            match t {
                Q1_0 => {
                    for j in 0..128 {
                        if x[j] >= 0.0 {
                            b[2 + j / 8] |= 1 << (j % 8);
                        }
                    }
                }
                Q2_0 => {
                    for j in 0..64 {
                        let q = ((x[j] * inv).round() as i32 + 1).clamp(0, 3) as u8;
                        b[2 + j / 4] |= q << (2 * (j % 4));
                    }
                }
                Q8_0 | Q8_1 => {
                    let p = if t == Q8_0 { 2 } else { 4 };
                    let mut sum = 0i32;
                    for j in 0..32 {
                        let q = (x[j] * inv).round() as i8;
                        b[p + j] = q as u8;
                        sum += q as i32;
                    }
                    if t == Q8_1 {
                        put_half(b, 2, sum as f32 * d);
                    }
                }
                Q4_0 | Q4_1 | Q5_0 | Q5_1 => {
                    let affine = matches!(t, Q4_1 | Q5_1);
                    let five = matches!(t, Q5_0 | Q5_1);
                    if affine {
                        put_half(b, 2, m);
                    }
                    let p = if affine { 4 } else { 2 };
                    let qpos = p + if five { 4 } else { 0 };
                    let offset = if affine {
                        0.5
                    } else if five {
                        16.5
                    } else {
                        8.5
                    };
                    let cap = if five { 31u8 } else { 15u8 };
                    let mut high = 0u32;
                    for j in 0..16 {
                        let q0 = (((if affine { x[j] - m } else { x[j] }) * inv + offset) as u8)
                            .min(cap);
                        let q1 = (((if affine { x[j + 16] - m } else { x[j + 16] }) * inv + offset)
                            as u8)
                            .min(cap);
                        b[qpos + j] = (q0 & 15) | ((q1 & 15) << 4);
                        high |= (((q0 & 16) >> 4) as u32) << j;
                        high |= (((q1 & 16) >> 4) as u32) << (j + 16);
                    }
                    if five {
                        b[p..p + 4].copy_from_slice(&high.to_le_bytes());
                    }
                }
                _ => unreachable!("validated encoder"),
            }
        }
    }
}
