//! Source scalar K-family reference encoders and their optimization helpers.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{DType, QuantError, f16_to_f32, f32_to_f16, half};

pub(crate) fn supports(t: DType) -> bool {
    matches!(
        t,
        DType::Q2_K | DType::Q3_K | DType::Q4_K | DType::Q5_K | DType::Q6_K
    )
}
pub(crate) fn encode(t: DType, values: &[f32]) -> Result<Vec<u8>, QuantError> {
    let bytes = t.row_bytes(values.len())?;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantError::NonFiniteInput { index });
        }
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| QuantError::AllocationFailed)?;
    output.resize(bytes, 0);
    for (block, (x, out)) in values
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(t.type_size()))
        .enumerate()
    {
        encode_block(t, x, out).map_err(|()| QuantError::UnrepresentableScale { block })?;
    }
    Ok(output)
}
pub(super) fn finite(x: f32) -> Result<f32, ()> {
    if x.is_finite() { Ok(x) } else { Err(()) }
}
pub(super) fn nearest(x: f32) -> Result<i32, ()> {
    if !x.is_finite() || x.abs() > 4_194_303.0 {
        return Err(());
    }
    Ok(x.round_ties_even() as i32)
}
pub(super) fn put_half(out: &mut [u8], offset: usize, x: f32) -> Result<f32, ()> {
    let bits = f32_to_f16(x);
    if bits & 0x7c00 == 0x7c00 {
        return Err(());
    }
    out[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
    Ok(f16_to_f32(bits))
}
fn signed_max(x: &[f32]) -> (f32, f32) {
    let mut magnitude = 0.0;
    let mut maximum = 0.0;
    for &value in x {
        if value.abs() > magnitude {
            magnitude = value.abs();
            maximum = value;
        }
    }
    (maximum, magnitude)
}
fn weight(x: f32, rmse: i32) -> f32 {
    match rmse {
        1 => x * x,
        2 => 1.0,
        3 => x.abs(),
        _ => x.abs().sqrt(),
    }
}
pub(super) fn make_qx(
    x: &[f32],
    nmax: i32,
    levels: &mut [i32],
    rmse: i32,
    weights: Option<&[f32]>,
) -> Result<f32, ()> {
    let (maximum, magnitude) = signed_max(x);
    if magnitude < 1e-15 {
        levels.fill(0);
        return Ok(0.0);
    }
    let mut inverse = -nmax as f32 / maximum;
    if rmse == 0 {
        for (l, &v) in levels.iter_mut().zip(x) {
            *l = nmax + nearest(inverse * v)?.clamp(-nmax, nmax - 1);
        }
        return finite(1.0 / inverse);
    }
    let early = rmse < 0;
    let rmse = rmse.abs();
    let mut sum_lx = 0.0;
    let mut sum_l2 = 0.0;
    for i in 0..x.len() {
        let l = nearest(inverse * x[i])?.clamp(-nmax, nmax - 1);
        levels[i] = l + nmax;
        let w = weights.map_or_else(|| weight(x[i], rmse), |w| w[i]);
        sum_lx += w * x[i] * l as f32;
        sum_l2 += w * l as f32 * l as f32;
    }
    finite(sum_lx)?;
    finite(sum_l2)?;
    let mut scale = if sum_l2 != 0.0 { sum_lx / sum_l2 } else { 0.0 };
    if early {
        return finite(if sum_l2 > 0.0 {
            0.5 * (scale + 1.0 / inverse)
        } else {
            1.0 / inverse
        });
    }
    let mut best = scale * sum_lx;
    for step in -9..=9 {
        if step == 0 {
            continue;
        }
        inverse = -(nmax as f32 + 0.1 * step as f32) / maximum;
        sum_lx = 0.0;
        sum_l2 = 0.0;
        for (i, &v) in x.iter().enumerate() {
            let l = nearest(inverse * v)?.clamp(-nmax, nmax - 1);
            let w = weights.map_or_else(|| weight(v, rmse), |w| w[i]);
            sum_lx += w * v * l as f32;
            sum_l2 += w * l as f32 * l as f32;
        }
        finite(sum_lx)?;
        finite(sum_l2)?;
        // Preserve defined F32 overflow in this comparison: two infinite
        // objective values compare false in the original reference as well.
        if sum_l2 > 0.0 && sum_lx * sum_lx > best * sum_l2 {
            for (l, &v) in levels.iter_mut().zip(x) {
                *l = nmax + nearest(inverse * v)?.clamp(-nmax, nmax - 1);
            }
            scale = sum_lx / sum_l2;
            best = scale * sum_lx;
        }
    }
    finite(scale)
}
fn make_q3(x: &[f32], nmax: i32, levels: &mut [i32]) -> Result<f32, ()> {
    let (maximum, magnitude) = signed_max(x);
    if magnitude < 1e-15 {
        levels.fill(0);
        return Ok(0.0);
    }
    let inverse = -nmax as f32 / maximum;
    let mut sum_lx = 0.0;
    let mut sum_l2 = 0.0;
    for (i, &v) in x.iter().enumerate() {
        let l = nearest(inverse * v)?.clamp(-nmax, nmax - 1);
        levels[i] = l;
        let w = v * v;
        sum_lx += w * v * l as f32;
        sum_l2 += w * l as f32 * l as f32;
    }
    finite(sum_lx)?;
    finite(sum_l2)?;
    for _ in 0..5 {
        let mut changed = false;
        for (i, &v) in x.iter().enumerate() {
            let w = v * v;
            let mut lx = sum_lx - w * v * levels[i] as f32;
            if lx > 0.0 {
                let mut l2 = sum_l2 - w * levels[i] as f32 * levels[i] as f32;
                let next = nearest(v * l2 / lx)?.clamp(-nmax, nmax - 1);
                if next != levels[i] {
                    lx += w * v * next as f32;
                    l2 += w * next as f32 * next as f32;
                    if l2 > 0.0 && lx * lx * sum_l2 > sum_lx * sum_lx * l2 {
                        levels[i] = next;
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
    for l in levels {
        *l += nmax;
    }
    finite(if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 })
}
#[allow(
    clippy::too_many_arguments,
    reason = "Keep the original make_qkx2_quants search parameters explicit for numerical parity."
)]
pub(super) fn make_affine2(
    x: &[f32],
    nmax: i32,
    weights: &[f32],
    levels: &mut [i32],
    rmin: f32,
    rdelta: f32,
    steps: i32,
    use_mad: bool,
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
    if maximum == minimum {
        levels.fill(0);
        return Ok((0.0, -minimum));
    }
    let mut inverse = nmax as f32 / (maximum - minimum);
    let mut scale = 1.0 / inverse;
    let mut best_error = 0.0;
    for i in 0..x.len() {
        levels[i] = nearest(inverse * (x[i] - minimum))?.clamp(0, nmax);
        let difference = scale * levels[i] as f32 + minimum - x[i];
        let error = if use_mad {
            difference.abs()
        } else {
            difference * difference
        };
        best_error += weights[i] * error;
    }
    if steps < 1 {
        return Ok((finite(scale)?, -minimum));
    }
    let mut candidate = [0; 32];
    for step in 0..=steps {
        inverse = (rmin + rdelta * step as f32 + nmax as f32) / (maximum - minimum);
        let mut sum_l = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_xl = 0.0;
        for i in 0..x.len() {
            let l = nearest(inverse * (x[i] - minimum))?.clamp(0, nmax);
            candidate[i] = l;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * l as f32 * l as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let determinant = sum_w * sum_l2 - sum_l * sum_l;
        if determinant > 0.0 {
            let mut candidate_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
            let mut candidate_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if candidate_minimum > 0.0 {
                candidate_minimum = 0.0;
                candidate_scale = sum_xl / sum_l2;
            }
            let mut error = 0.0;
            for i in 0..x.len() {
                let difference = candidate_scale * candidate[i] as f32 + candidate_minimum - x[i];
                let difference = if use_mad {
                    difference.abs()
                } else {
                    difference * difference
                };
                error += weights[i] * difference;
            }
            if error < best_error {
                levels.copy_from_slice(&candidate[..x.len()]);
                best_error = error;
                scale = candidate_scale;
                minimum = candidate_minimum;
            }
        }
    }
    Ok((finite(scale)?, finite(-minimum)?))
}

fn encode_block(t: DType, x: &[f32], out: &mut [u8]) -> Result<(), ()> {
    out.fill(0);
    let mut levels = [0_i32; 256];
    match t {
        DType::Q2_K => encode_q2(x, out, &mut levels),
        DType::Q3_K => encode_q3(x, out, &mut levels),
        DType::Q4_K | DType::Q5_K => encode_q45(t, x, out, &mut levels),
        DType::Q6_K => encode_q6(x, out, &mut levels),
        _ => unreachable!("validated K encoder dtype"),
    }
}
fn encode_q2(x: &[f32], out: &mut [u8], levels: &mut [i32; 256]) -> Result<(), ()> {
    let mut scales = [0.0; 16];
    let mut minima = [0.0; 16];
    let mut maximum_scale = 0.0_f32;
    let mut maximum_minimum = 0.0_f32;
    for sub in 0..16 {
        let values = &x[sub * 16..(sub + 1) * 16];
        let weights = std::array::from_fn::<_, 16, _>(|i| values[i].abs());
        let (scale, minimum) = make_affine2(
            values,
            3,
            &weights,
            &mut levels[sub * 16..(sub + 1) * 16],
            -0.5,
            0.1,
            15,
            true,
        )?;
        scales[sub] = scale;
        minima[sub] = minimum;
        if scale > maximum_scale {
            maximum_scale = scale;
        }
        if minimum > maximum_minimum {
            maximum_minimum = minimum;
        }
    }
    if maximum_scale > 0.0 {
        let inverse = 15.0 / maximum_scale;
        for sub in 0..16 {
            out[sub] = nearest(inverse * scales[sub])? as u8;
        }
        put_half(out, 80, maximum_scale / 15.0)?;
    }
    if maximum_minimum > 0.0 {
        let inverse = 15.0 / maximum_minimum;
        for sub in 0..16 {
            out[sub] |= (nearest(inverse * minima[sub])? as u8) << 4;
        }
        put_half(out, 82, maximum_minimum / 15.0)?;
    }
    for sub in 0..16 {
        let scale = half(out, 80) * (out[sub] & 15) as f32;
        if scale == 0.0 {
            continue;
        }
        let minimum = half(out, 82) * (out[sub] >> 4) as f32;
        for j in 0..16 {
            let index = sub * 16 + j;
            levels[index] = nearest((x[index] + minimum) / scale)?.clamp(0, 3);
        }
    }
    pack_two_bits(levels, &mut out[16..80]);
    Ok(())
}
fn pack_two_bits(levels: &[i32; 256], out: &mut [u8]) {
    for group in 0..2 {
        for j in 0..32 {
            out[group * 32 + j] = levels[group * 128 + j] as u8
                | ((levels[group * 128 + j + 32] as u8) << 2)
                | ((levels[group * 128 + j + 64] as u8) << 4)
                | ((levels[group * 128 + j + 96] as u8) << 6);
        }
    }
}
fn encode_q3(x: &[f32], out: &mut [u8], levels: &mut [i32; 256]) -> Result<(), ()> {
    let mut scales = [0.0; 16];
    let mut maximum_scale = 0.0;
    let mut maximum_magnitude = 0.0;
    for sub in 0..16 {
        scales[sub] = make_q3(
            &x[sub * 16..(sub + 1) * 16],
            4,
            &mut levels[sub * 16..(sub + 1) * 16],
        )?;
        if scales[sub].abs() > maximum_magnitude {
            maximum_magnitude = scales[sub].abs();
            maximum_scale = scales[sub];
        }
    }
    if maximum_scale != 0.0 {
        let inverse = -32.0 / maximum_scale;
        for sub in 0..16 {
            let l = (nearest(inverse * scales[sub])?.clamp(-32, 31) + 32) as u8;
            if sub < 8 {
                out[96 + sub] = l & 15;
            } else {
                out[96 + sub - 8] |= (l & 15) << 4;
            }
            out[104 + sub % 4] |= (l >> 4) << (2 * (sub / 4));
        }
        put_half(out, 108, 1.0 / inverse)?;
    }
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
            let index = sub * 16 + j;
            levels[index] = nearest(x[index] / scale)?.clamp(-4, 3) + 4;
        }
    }
    for (j, l) in levels.iter_mut().enumerate() {
        if *l > 3 {
            out[j % 32] |= 1 << (j / 32);
            *l -= 4;
        }
    }
    pack_two_bits(levels, &mut out[32..96]);
    Ok(())
}
pub(super) fn pack_scale_min(out: &mut [u8], sub: usize, scale: u8, minimum: u8) {
    if sub < 4 {
        out[sub] = scale;
        out[sub + 4] = minimum;
    } else {
        out[sub + 4] = (scale & 15) | ((minimum & 15) << 4);
        out[sub - 4] |= (scale >> 4) << 6;
        out[sub] |= (minimum >> 4) << 6;
    }
}
pub(super) fn unpack_scale_min(out: &[u8], sub: usize) -> (u8, u8) {
    if sub < 4 {
        (out[sub] & 63, out[sub + 4] & 63)
    } else {
        (
            (out[sub + 4] & 15) | ((out[sub - 4] >> 6) << 4),
            (out[sub + 4] >> 4) | ((out[sub] >> 6) << 4),
        )
    }
}
fn encode_q45(t: DType, x: &[f32], out: &mut [u8], levels: &mut [i32; 256]) -> Result<(), ()> {
    let five = t == DType::Q5_K;
    let nmax = if five { 31 } else { 15 };
    let mut scales = [0.0; 8];
    let mut minima = [0.0; 8];
    let mut maximum_scale = 0.0_f32;
    let mut maximum_minimum = 0.0_f32;
    for sub in 0..8 {
        let values = &x[sub * 32..(sub + 1) * 32];
        let square_sum = values.iter().fold(0.0, |sum, &v| sum + v * v);
        let average = finite((square_sum / 32.0).sqrt())?;
        let weights = std::array::from_fn::<_, 32, _>(|i| average + values[i].abs());
        let (scale, minimum) = make_affine2(
            values,
            nmax,
            &weights,
            &mut levels[sub * 32..(sub + 1) * 32],
            if five { -0.5 } else { -1.0 },
            0.1,
            if five { 15 } else { 20 },
            false,
        )?;
        scales[sub] = scale;
        minima[sub] = minimum;
        if scale > maximum_scale {
            maximum_scale = scale;
        }
        if minimum > maximum_minimum {
            maximum_minimum = minimum;
        }
    }
    let inverse_scale = if maximum_scale > 0.0 {
        63.0 / maximum_scale
    } else {
        0.0
    };
    let inverse_minimum = if maximum_minimum > 0.0 {
        63.0 / maximum_minimum
    } else {
        0.0
    };
    for sub in 0..8 {
        let scale = nearest(inverse_scale * scales[sub])?.min(63) as u8;
        let minimum = nearest(inverse_minimum * minima[sub])?.min(63) as u8;
        pack_scale_min(&mut out[4..16], sub, scale, minimum);
    }
    put_half(out, 0, maximum_scale / 63.0)?;
    put_half(out, 2, maximum_minimum / 63.0)?;
    for sub in 0..8 {
        let (sc, min) = unpack_scale_min(&out[4..16], sub);
        let scale = half(out, 0) * sc as f32;
        if scale == 0.0 {
            continue;
        }
        let minimum = half(out, 2) * min as f32;
        for j in 0..32 {
            let index = sub * 32 + j;
            levels[index] = nearest((x[index] + minimum) / scale)?.clamp(0, nmax);
        }
    }
    let qpos = if five { 48 } else { 16 };
    for group in 0..4 {
        for j in 0..32 {
            let low = levels[group * 64 + j] as u8;
            let high = levels[group * 64 + j + 32] as u8;
            if five {
                if low > 15 {
                    out[16 + j] |= 1 << (2 * group);
                }
                if high > 15 {
                    out[16 + j] |= 2 << (2 * group);
                }
            }
            out[qpos + group * 32 + j] = (low & 15) | ((high & 15) << 4);
        }
    }
    Ok(())
}
fn encode_q6(x: &[f32], out: &mut [u8], levels: &mut [i32; 256]) -> Result<(), ()> {
    let mut scales = [0.0; 16];
    let mut maximum_scale = 0.0;
    let mut maximum_magnitude = 0.0;
    for sub in 0..16 {
        scales[sub] = make_qx(
            &x[sub * 16..(sub + 1) * 16],
            32,
            &mut levels[sub * 16..(sub + 1) * 16],
            1,
            None,
        )?;
        if scales[sub].abs() > maximum_magnitude {
            maximum_magnitude = scales[sub].abs();
            maximum_scale = scales[sub];
        }
    }
    if maximum_magnitude < 1e-15 {
        return Ok(());
    }
    let inverse = -128.0 / maximum_scale;
    put_half(out, 208, 1.0 / inverse)?;
    for sub in 0..16 {
        out[192 + sub] = nearest(inverse * scales[sub])?.min(127) as i8 as u8;
    }
    for sub in 0..16 {
        let scale = half(out, 208) * out[192 + sub] as i8 as f32;
        if scale == 0.0 {
            continue;
        }
        for j in 0..16 {
            let index = sub * 16 + j;
            levels[index] = nearest(x[index] / scale)?.clamp(-32, 31) + 32;
        }
    }
    for group in 0..2 {
        for j in 0..32 {
            let a = levels[group * 128 + j] as u8;
            let b = levels[group * 128 + j + 32] as u8;
            let c = levels[group * 128 + j + 64] as u8;
            let d = levels[group * 128 + j + 96] as u8;
            out[group * 64 + j] = (a & 15) | ((c & 15) << 4);
            out[group * 64 + j + 32] = (b & 15) | ((d & 15) << 4);
            out[128 + group * 32 + j] =
                (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((d >> 4) << 6);
        }
    }
    Ok(())
}
