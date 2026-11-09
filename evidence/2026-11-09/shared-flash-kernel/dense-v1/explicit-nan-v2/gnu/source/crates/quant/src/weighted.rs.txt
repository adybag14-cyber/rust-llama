//! Importance-weighted scalar row encoders from the frozen source.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{
    DType, QuantError,
    encode_k::{finite, make_qx, nearest, pack_scale_min, put_half, unpack_scale_min},
    half,
};

/// The original ggml_quantize_requires_imatrix capability, not an encoder-support claim.
pub fn requires_importance_weights(dtype: DType) -> bool {
    matches!(dtype, DType::IQ2_XXS | DType::IQ2_XS | DType::IQ1_S)
}

/// Encode one complete GGML row using explicit nonnegative importance weights.
/// The entire row participates in the legacy formats' variance calculation.
/// Formats that ignore importance weights in the original source retain that behavior.
pub fn quantize_with_weights(
    dtype: DType,
    values: &[f32],
    importance: &[f32],
) -> Result<Vec<u8>, QuantError> {
    let bytes = dtype.row_bytes(values.len())?;
    if importance.len() != values.len() {
        return Err(QuantError::InvalidImportanceLength {
            expected: values.len(),
            actual: importance.len(),
        });
    }
    for (index, &value) in importance.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(QuantError::InvalidImportanceWeight { index });
        }
    }
    match dtype {
        DType::Q1_0
        | DType::Q2_0
        | DType::Q8_0
        | DType::TQ1_0
        | DType::TQ2_0
        | DType::MXFP4
        | DType::NVFP4
        | DType::F8_E4M3
        | DType::F32
        | DType::F16
        | DType::BF16 => return crate::quantize(dtype, values),
        DType::IQ1_S | DType::IQ1_M => {
            return crate::encode_iq1::encode(dtype, values, Some(importance));
        }
        DType::IQ2_XXS | DType::IQ2_XS | DType::IQ2_S => {
            return crate::encode_iq2::encode(dtype, values, Some(importance));
        }
        DType::IQ3_XXS | DType::IQ3_S => {
            return crate::encode_iq3::encode(dtype, values, Some(importance));
        }
        DType::IQ4_NL | DType::IQ4_XS => {
            return crate::encode_iq4::encode(dtype, values, Some(importance));
        }
        DType::Q4_0
        | DType::Q4_1
        | DType::Q5_0
        | DType::Q5_1
        | DType::Q2_K
        | DType::Q3_K
        | DType::Q4_K
        | DType::Q5_K
        | DType::Q6_K => {}
        _ => return Err(QuantError::UnsupportedWeightedQuantization(dtype)),
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantError::NonFiniteInput { index });
        }
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| QuantError::AllocationFailed)?;
    output.resize(bytes, 0);
    if values.is_empty() {
        return Ok(output);
    }
    let legacy = dtype.block_size() == 32;
    let row_sigma2 = if legacy {
        finite(values.iter().fold(0.0, |s, &v| s + v * v) / values.len() as f32)
            .map_err(|()| QuantError::UnrepresentableScale { block: 0 })?
    } else {
        0.0
    };
    let size = dtype.block_size();
    for (block, (x, out)) in values
        .chunks_exact(size)
        .zip(output.chunks_exact_mut(dtype.type_size()))
        .enumerate()
    {
        let w = &importance[block * size..(block + 1) * size];
        let result = if legacy {
            encode_legacy(dtype, x, w, row_sigma2, out)
        } else {
            encode_k(dtype, x, w, out)
        };
        result.map_err(|()| QuantError::UnrepresentableScale { block })?;
    }
    Ok(output)
}

/// Encode atomically into an exactly sized destination.
pub fn quantize_with_weights_into(
    dtype: DType,
    values: &[f32],
    importance: &[f32],
    output: &mut [u8],
) -> Result<(), QuantError> {
    crate::validate(dtype, output.len(), values.len())?;
    let staged = quantize_with_weights(dtype, values, importance)?;
    output.copy_from_slice(&staged);
    Ok(())
}

// The weighted source paths use make_qkx3_quants(-0.9, 0.05, 36, false).
// Its clipping before nearest_int deliberately accepts an intermediate NaN/Inf
// caused by a fitted minimum reaching the maximum; comparisons handle it.
fn make_affine3(
    x: &[f32],
    nmax: i32,
    weights: &[f32],
    levels: &mut [i32],
) -> Result<(f32, f32), ()> {
    let mut minimum = x[0];
    let mut maximum = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..x.len() {
        if x[i] < minimum {
            minimum = x[i];
        }
        if x[i] > maximum {
            maximum = x[i];
        }
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    finite(sum_w)?;
    finite(sum_x)?;
    if minimum > 0.0 {
        minimum = 0.0;
    }
    if maximum <= minimum {
        levels.fill(0);
        return Ok((0.0, -minimum));
    }
    let mut inverse = nmax as f32 / (maximum - minimum);
    let mut scale = 1.0 / inverse;
    let mut best_error = 0.0;
    for i in 0..x.len() {
        levels[i] = nearest(inverse * (x[i] - minimum))?.clamp(0, nmax);
        let delta = scale * levels[i] as f32 + minimum - x[i];
        best_error += weights[i] * (delta * delta);
    }
    let mut candidate = [0; 32];
    for step in 0..=36 {
        inverse = (-0.9 + 0.05 * step as f32 + nmax as f32) / (maximum - minimum);
        let mut sum_l = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_xl = 0.0;
        for i in 0..x.len() {
            let value = inverse * (x[i] - minimum);
            let l = if value > 0.0 {
                nearest(value.min(nmax as f32))?
            } else {
                0
            };
            candidate[i] = l;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * l as f32 * l as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let determinant = sum_w * sum_l2 - sum_l * sum_l;
        if determinant > 0.0 {
            let mut next_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
            let mut next_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if next_minimum > 0.0 {
                next_minimum = 0.0;
                next_scale = sum_xl / sum_l2;
            }
            let mut error = 0.0;
            for i in 0..x.len() {
                let delta = next_scale * candidate[i] as f32 + next_minimum - x[i];
                error += weights[i] * (delta * delta);
            }
            if error < best_error {
                levels.copy_from_slice(&candidate[..x.len()]);
                best_error = error;
                scale = next_scale;
                minimum = next_minimum;
            }
        }
    }
    Ok((finite(scale)?, finite(-minimum)?))
}

// Preserve the source uint8_t scratch assignments, including negative values
// in the IQ2_XXS initial fit; later arithmetic promotes those bytes back to int.
pub(super) fn make_positive(
    x: &[f32],
    nmax: i32,
    levels: &mut [i32],
    weights: &[f32],
) -> Result<f32, ()> {
    let mut maximum = 0.0_f32;
    for &v in x {
        if v > maximum {
            maximum = v;
        }
    }
    if maximum < 1e-15 {
        levels.fill(0);
        return Ok(0.0);
    }
    let mut inverse = nmax as f32 / maximum;
    for (l, &v) in levels.iter_mut().zip(x) {
        *l = nearest(inverse * v)? as u8 as i32;
    }
    let scale = 1.0 / inverse;
    let mut best_error = 0.0;
    for i in 0..x.len() {
        let delta = x[i] - scale * levels[i] as f32;
        best_error += weights[i] * delta * delta;
    }
    for step in -4..=4 {
        if step == 0 {
            continue;
        }
        let next_inverse = (0.1 * step as f32 + nmax as f32) / maximum;
        let next_scale = 1.0 / next_inverse;
        let mut error = 0.0;
        for i in 0..x.len() {
            let l = nearest(next_inverse * x[i])?.min(nmax);
            let delta = x[i] - next_scale * l as f32;
            error += weights[i] * delta * delta;
        }
        if error < best_error {
            best_error = error;
            inverse = next_inverse;
        }
    }
    let mut sum_lx = 0.0;
    let mut sum_l2 = 0.0;
    for i in 0..x.len() {
        let l = nearest(inverse * x[i])?.min(nmax);
        levels[i] = l as u8 as i32;
        sum_lx += weights[i] * x[i] * l as f32;
        sum_l2 += weights[i] * l as f32 * l as f32;
    }
    finite(sum_lx)?;
    finite(sum_l2)?;
    for _ in 0..5 {
        let mut changed = false;
        for i in 0..x.len() {
            let w = weights[i];
            let mut lx = sum_lx - w * x[i] * levels[i] as f32;
            let mut l2 = sum_l2 - w * levels[i] as f32 * levels[i] as f32;
            if lx > 0.0 && l2 > 0.0 {
                let next = nearest(x[i] * l2 / lx)?.min(nmax);
                if next != levels[i] {
                    lx += w * x[i] * next as f32;
                    l2 += w * next as f32 * next as f32;
                    if lx * lx * sum_l2 > sum_lx * sum_lx * l2 {
                        levels[i] = next as u8 as i32;
                        sum_lx = lx;
                        sum_l2 = l2;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    finite(if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 })
}

fn adjusted_weights<const N: usize>(
    x: &[f32],
    weights: &[f32],
    sigma2: f32,
) -> Result<[f32; N], ()> {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = finite(weights[i] * (sigma2 + x[i] * x[i]).sqrt())?;
    }
    Ok(result)
}
fn encode_legacy(
    dtype: DType,
    x: &[f32],
    importance: &[f32],
    sigma2: f32,
    out: &mut [u8],
) -> Result<(), ()> {
    let weights = adjusted_weights::<32>(x, importance, sigma2)?;
    let mut levels = [0; 32];
    let affine = matches!(dtype, DType::Q4_1 | DType::Q5_1);
    let five = matches!(dtype, DType::Q5_0 | DType::Q5_1);
    let scale_bytes = if affine {
        let (d, m) = make_affine3(x, if five { 31 } else { 15 }, &weights, &mut levels)?;
        put_half(out, 0, d)?;
        put_half(out, 2, -m)?;
        4
    } else {
        let d = make_qx(x, if five { 16 } else { 8 }, &mut levels, 1, Some(&weights))?;
        put_half(out, 0, d)?;
        2
    };
    let mut high = 0_u32;
    let qpos = scale_bytes + if five { 4 } else { 0 };
    for j in 0..16 {
        let a = levels[j] as u8;
        let b = levels[j + 16] as u8;
        out[qpos + j] = (a & 15) | ((b & 15) << 4);
        if five {
            high |= (((a & 16) >> 4) as u32) << j;
            high |= (((b & 16) >> 4) as u32) << (j + 16);
        }
    }
    if five {
        out[scale_bytes..scale_bytes + 4].copy_from_slice(&high.to_le_bytes());
    }
    Ok(())
}
fn encode_k(dtype: DType, x: &[f32], importance: &[f32], out: &mut [u8]) -> Result<(), ()> {
    let mut levels = [0; 256];
    if dtype == DType::Q6_K {
        return crate::encode_k::encode_q6(x, out, &mut levels, Some(importance));
    }
    let sum_x2 = x.iter().fold(0.0, |s, &v| s + v * v);
    let sigma2 = finite(if dtype == DType::Q2_K {
        sum_x2 / 256.0
    } else {
        2.0 * sum_x2 / 256.0
    })?;
    if dtype == DType::Q3_K {
        return encode_q3(x, importance, sigma2, out, &mut levels);
    }
    let small = dtype == DType::Q2_K;
    let five = dtype == DType::Q5_K;
    let width = if small { 16 } else { 32 };
    let groups = 256 / width;
    let nmax = if small {
        3
    } else if five {
        31
    } else {
        15
    };
    let max_scale = if small { 15 } else { 63 };
    let mut scales = [0.0; 16];
    let mut minima = [0.0; 16];
    let mut sum_weights = [0.0; 16];
    for sub in 0..groups {
        let values = &x[sub * width..(sub + 1) * width];
        let w = &importance[sub * width..(sub + 1) * width];
        let mut weights = [0.0; 32];
        for j in 0..width {
            weights[j] = finite(w[j] * (sigma2 + values[j] * values[j]).sqrt())?;
            sum_weights[sub] += weights[j];
        }
        let (d, m) = make_affine3(
            values,
            nmax,
            &weights[..width],
            &mut levels[sub * width..(sub + 1) * width],
        )?;
        scales[sub] = d;
        minima[sub] = m;
    }
    let mut scale_levels = [0; 16];
    let mut minimum_levels = [0; 16];
    let d = make_positive(
        &scales[..groups],
        max_scale,
        &mut scale_levels[..groups],
        &sum_weights[..groups],
    )?;
    let m = make_positive(
        &minima[..groups],
        max_scale,
        &mut minimum_levels[..groups],
        &sum_weights[..groups],
    )?;
    let d = put_half(out, if small { 80 } else { 0 }, d)?;
    let m = put_half(out, if small { 82 } else { 2 }, m)?;
    for sub in 0..groups {
        if small {
            out[sub] = scale_levels[sub] as u8 | ((minimum_levels[sub] as u8) << 4);
        } else {
            pack_scale_min(
                &mut out[4..16],
                sub,
                scale_levels[sub] as u8,
                minimum_levels[sub] as u8,
            );
        }
    }
    for sub in 0..groups {
        let (sc, min) = if small {
            (out[sub] & 15, out[sub] >> 4)
        } else {
            unpack_scale_min(&out[4..16], sub)
        };
        let scale = d * sc as f32;
        if scale == 0.0 {
            continue;
        }
        let minimum = m * min as f32;
        for j in 0..width {
            let i = sub * width + j;
            levels[i] = nearest((x[i] + minimum) / scale)?.clamp(0, nmax);
        }
    }
    if small {
        crate::encode_k::pack_two_bits(&levels, &mut out[16..80]);
    } else {
        let qpos = if five { 48 } else { 16 };
        for group in 0..4 {
            for j in 0..32 {
                let a = levels[group * 64 + j] as u8;
                let b = levels[group * 64 + j + 32] as u8;
                if five {
                    if a > 15 {
                        out[16 + j] |= 1 << (2 * group);
                    }
                    if b > 15 {
                        out[16 + j] |= 2 << (2 * group);
                    }
                }
                out[qpos + group * 32 + j] = (a & 15) | ((b & 15) << 4);
            }
        }
    }
    Ok(())
}
fn encode_q3(
    x: &[f32],
    importance: &[f32],
    sigma2: f32,
    out: &mut [u8],
    levels: &mut [i32; 256],
) -> Result<(), ()> {
    let mut scales = [0.0; 16];
    let mut sum_weights = [0.0; 16];
    for sub in 0..16 {
        let values = &x[sub * 16..(sub + 1) * 16];
        let weights =
            adjusted_weights::<16>(values, &importance[sub * 16..(sub + 1) * 16], sigma2)?;
        for &w in &weights {
            sum_weights[sub] += w;
        }
        scales[sub] = make_qx(
            values,
            4,
            &mut levels[sub * 16..(sub + 1) * 16],
            1,
            Some(&weights),
        )?;
    }
    let mut scale_levels = [0; 16];
    let d = make_qx(&scales, 32, &mut scale_levels, 1, Some(&sum_weights))?;
    for sub in 0..16 {
        let l = scale_levels[sub] as u8;
        if sub < 8 {
            out[96 + sub] = l & 15;
        } else {
            out[96 + sub - 8] |= (l & 15) << 4;
        }
        out[104 + sub % 4] |= (l >> 4) << (2 * (sub / 4));
    }
    put_half(out, 108, d)?;
    for sub in 0..16 {
        let low = if sub < 8 {
            out[96 + sub] & 15
        } else {
            out[96 + sub - 8] >> 4
        };
        let sc = (low | (((out[104 + sub % 4] >> (2 * (sub / 4))) & 3) << 4)) as i32 - 32;
        let scale = half(out, 108) * sc as f32;
        if scale == 0.0 {
            continue;
        }
        for j in 0..16 {
            let i = sub * 16 + j;
            levels[i] = nearest(x[i] / scale)?.clamp(-4, 3) + 4;
        }
    }
    for (j, l) in levels.iter_mut().enumerate() {
        if *l > 3 {
            out[j % 32] |= 1 << (j / 32);
            *l -= 4;
        }
    }
    crate::encode_k::pack_two_bits(levels, &mut out[32..96]);
    Ok(())
}
