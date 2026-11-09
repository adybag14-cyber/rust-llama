//! IQ1 exhaustive weighted split fitting and exact scalar codebook search.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{
    DType, QuantError,
    encode_k::{finite, nearest, put_half},
    iq_lattice::{self, Grid8, key},
};
const POSITIVE: [f32; 3] = [-0.875, 0.125, 1.125];
const NEGATIVE: [f32; 3] = [-1.125, -0.125, 0.875];
struct Fit {
    scale: f32,
    first: usize,
    second: usize,
    shift: usize,
}
fn fit_s(values: &[f32], weights: &[f32], indices: &[usize]) -> Result<Option<Fit>, ()> {
    let mut sum_x = [0.0; 33];
    let mut sum_w = [0.0; 33];
    for (j, &i) in indices.iter().enumerate() {
        sum_x[j + 1] = sum_x[j] + weights[i] * values[i];
        sum_w[j + 1] = sum_w[j] + weights[i];
    }
    finite(sum_x[32])?;
    finite(sum_w[32])?;
    let mut best = -f32::MAX;
    let mut selected = None;
    for first in 0..=32 {
        for second in first..=32 {
            for (shift, grid) in [POSITIVE, NEGATIVE].iter().enumerate() {
                let sum_qx = (sum_x[first] - sum_x[0]) * grid[0]
                    + (sum_x[second] - sum_x[first]) * grid[1]
                    + (sum_x[32] - sum_x[second]) * grid[2];
                let sum_q2 = (sum_w[first] - sum_w[0]) * grid[0] * grid[0]
                    + (sum_w[second] - sum_w[first]) * grid[1] * grid[1]
                    + (sum_w[32] - sum_w[second]) * grid[2] * grid[2];
                if sum_q2 > 0.0 && sum_qx * sum_qx > best * sum_q2 {
                    let scale = finite(sum_qx / sum_q2)?;
                    best = scale * sum_qx;
                    selected = Some(Fit {
                        scale,
                        first,
                        second,
                        shift,
                    });
                }
            }
        }
    }
    Ok(selected)
}
fn fit_m(values: &[f32], weights: &[f32], indices: &[usize]) -> Result<Option<Fit>, ()> {
    let mut px = [[0.0; 17]; 2];
    let mut pw = [[0.0; 17]; 2];
    for (j, &i) in indices.iter().enumerate() {
        let half = usize::from(i >= 8);
        px[half][j + 1] = px[half][j] + weights[i] * values[i];
        px[1 - half][j + 1] = px[1 - half][j];
        pw[half][j + 1] = pw[half][j] + weights[i];
        pw[1 - half][j + 1] = pw[1 - half][j];
    }
    for half in 0..2 {
        finite(px[half][16])?;
        finite(pw[half][16])?;
    }
    let p2 = POSITIVE[2];
    let m2 = NEGATIVE[2];
    let cp1 = POSITIVE[0] * POSITIVE[0] - POSITIVE[1] * POSITIVE[1];
    let cp2 = POSITIVE[1] * POSITIVE[1] - POSITIVE[2] * POSITIVE[2];
    let cm1 = NEGATIVE[0] * NEGATIVE[0] - NEGATIVE[1] * NEGATIVE[1];
    let cm2 = NEGATIVE[1] * NEGATIVE[1] - NEGATIVE[2] * NEGATIVE[2];
    let mut best = -f32::MAX;
    let mut selected = None;
    for first in 0..=16 {
        for second in first..=16 {
            let mut qx_p = [0.0; 2];
            let mut qx_m = [0.0; 2];
            let mut q2_p = [0.0; 2];
            let mut q2_m = [0.0; 2];
            for half in 0..2 {
                let sum = px[half][first] + px[half][second];
                qx_p[half] = px[half][16] * p2 - sum;
                qx_m[half] = px[half][16] * m2 - sum;
                q2_p[half] =
                    pw[half][16] * p2 * p2 + pw[half][first] * cp1 + pw[half][second] * cp2;
                q2_m[half] =
                    pw[half][16] * m2 * m2 + pw[half][first] * cm1 + pw[half][second] * cm2;
            }
            let qx = [
                qx_p[0] + qx_p[1],
                qx_p[0] + qx_m[1],
                qx_m[0] + qx_p[1],
                qx_m[0] + qx_m[1],
            ];
            let q2 = [
                q2_p[0] + q2_p[1],
                q2_p[0] + q2_m[1],
                q2_m[0] + q2_p[1],
                q2_m[0] + q2_m[1],
            ];
            for shift in 0..4 {
                if q2[shift] > 0.0 && qx[shift] * qx[shift] > best * q2[shift] {
                    let scale = finite(qx[shift] / q2[shift])?;
                    best = scale * qx[shift];
                    selected = Some(Fit {
                        scale,
                        first,
                        second,
                        shift,
                    });
                }
            }
        }
    }
    Ok(selected)
}
fn signed_grid(medium: bool, shift: usize, group: usize) -> &'static [f32; 3] {
    let positive = if medium {
        if group == 0 {
            shift < 2
        } else {
            shift.is_multiple_of(2)
        }
    } else {
        shift == 0
    };
    if positive { &POSITIVE } else { &NEGATIVE }
}
fn best_neighbour(
    grid: &Grid8,
    code: usize,
    values: &[f32],
    weights: &[f32],
    scale: f32,
    signed: &[f32; 3],
) -> Result<usize, ()> {
    let mut best = f32::MAX;
    let mut selected = None;
    for &index in grid.neighbours(code)? {
        let point = &grid.levels[index as usize];
        let mut error = 0.0;
        for j in 0..8 {
            let delta = scale * signed[point[j] as usize] - values[j];
            error += weights[j] * delta * delta;
        }
        if error < best {
            best = error;
            selected = Some(index as usize);
        }
    }
    // The source's exceptional exhaustive fallback indexes xval by a grid
    // index rather than a coordinate. Reject that undefined/out-of-range
    // numerical path instead of reproducing an out-of-bounds read.
    selected.ok_or(())
}
fn weights<const N: usize>(
    values: &[f32],
    importance: Option<&[f32]>,
    sigma2: f32,
) -> Result<[f32; N], ()> {
    let mut output = [0.0; N];
    for j in 0..N {
        output[j] = finite(if let Some(importance) = importance {
            importance[j] * (sigma2 + values[j] * values[j]).sqrt()
        } else {
            values[j] * values[j]
        })?;
    }
    Ok(output)
}
pub(crate) fn encode(
    dtype: DType,
    values: &[f32],
    importance: Option<&[f32]>,
) -> Result<Vec<u8>, QuantError> {
    let bytes = dtype.row_bytes(values.len())?;
    if dtype == DType::IQ1_S && importance.is_none() {
        return Err(QuantError::ImportanceWeightsRequired(dtype));
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantError::NonFiniteInput { index });
        }
    }
    let grid = iq_lattice::grid(dtype)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| QuantError::AllocationFailed)?;
    output.resize(bytes, 0);
    for (block, (x, out)) in values
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(dtype.type_size()))
        .enumerate()
    {
        let w = importance.map(|w| &w[block * 256..(block + 1) * 256]);
        encode_block(dtype, grid, x, w, out)
            .map_err(|()| QuantError::UnrepresentableScale { block })?;
    }
    Ok(output)
}
fn encode_block(
    dtype: DType,
    grid: &Grid8,
    x: &[f32],
    importance: Option<&[f32]>,
    out: &mut [u8],
) -> Result<(), ()> {
    let medium = dtype == DType::IQ1_M;
    let width = if medium { 16 } else { 32 };
    let groups = width / 8;
    let blocks = 256 / width;
    let sum_x2 = x.iter().fold(0.0, |sum, &value| sum + value * value);
    let sigma2 = finite(2.0 * sum_x2 / 256.0)?;
    let mut scales = [0.0; 16];
    let mut shifts = [0; 16];
    let mut max_scale = 0.0_f32;
    for sub in 0..blocks {
        let values = &x[sub * width..(sub + 1) * width];
        let importance = importance.map(|w| &w[sub * width..(sub + 1) * width]);
        let mut weight = [0.0; 32];
        if medium {
            weight[..16].copy_from_slice(&weights::<16>(values, importance, sigma2)?);
        } else {
            weight = weights::<32>(values, importance, sigma2)?;
        }
        let mut magnitude = values[0].abs();
        for &value in &values[1..] {
            if value.abs() > magnitude {
                magnitude = value.abs();
            }
        }
        if magnitude < if medium { 1e-7 } else { 1e-12 } {
            continue;
        }
        let mut order = [0; 32];
        crate::iq_sort::order(values, &mut order[..width]);
        let fit = if medium {
            fit_m(values, &weight[..width], &order[..width])?
        } else {
            fit_s(values, &weight, &order)?
        };
        let Some(mut fit) = fit else {
            continue;
        };
        let mut levels = [0; 32];
        for (j, &i) in order.iter().take(width).enumerate() {
            levels[i] = if j < fit.first {
                0
            } else if j < fit.second {
                1
            } else {
                2
            };
        }
        if fit.scale < 0.0 {
            for level in &mut levels[..width] {
                *level = 2 - *level;
            }
            fit.scale = -fit.scale;
            fit.shift ^= if medium { 3 } else { 1 };
        }
        let mut indices = [0; 4];
        let mut all_on_grid = true;
        for (group, index) in indices.iter_mut().enumerate().take(groups) {
            let part = group * 8..(group + 1) * 8;
            let code = key(&levels[part.clone()]);
            *index = if let Some(index) = grid.index(code) {
                index
            } else {
                all_on_grid = false;
                best_neighbour(
                    grid,
                    code,
                    &values[part.clone()],
                    &weight[part],
                    fit.scale,
                    signed_grid(medium, fit.shift, group),
                )?
            };
        }
        if !all_on_grid {
            let mut sum_qx = 0.0;
            let mut sum_q2 = 0.0;
            for (group, &index) in indices.iter().enumerate().take(groups) {
                let signed = signed_grid(medium, fit.shift, group);
                for j in 0..8 {
                    let q = signed[grid.levels[index][j] as usize];
                    let w = weight[group * 8 + j];
                    sum_qx += w * q * values[group * 8 + j];
                    sum_q2 += w * q * q;
                }
            }
            finite(sum_qx)?;
            finite(sum_q2)?;
            if sum_qx > 0.0 && sum_q2 > 0.0 {
                fit.scale = sum_qx / sum_q2;
            }
        }
        if medium {
            out[sub * 2] = indices[0] as u8;
            out[sub * 2 + 1] = indices[1] as u8;
            out[32 + sub] = (indices[0] >> 8) as u8 | (((indices[1] >> 8) as u8) << 4);
        } else {
            let mut high = 0_u16;
            for (group, &index) in indices.iter().enumerate().take(groups) {
                out[2 + sub * 4 + group] = index as u8;
                high |= ((index >> 8) as u16) << (3 * group);
            }
            out[34 + sub * 2..36 + sub * 2].copy_from_slice(&high.to_le_bytes());
        }
        scales[sub] = finite(fit.scale)?;
        shifts[sub] = fit.shift;
        if fit.scale > max_scale {
            max_scale = fit.scale;
        }
    }
    if max_scale == 0.0 {
        return Ok(());
    }
    let mut d = max_scale / 15.0;
    let inverse = 1.0 / d;
    if !medium {
        put_half(out, 0, d * 1.125)?;
        for sub in 0..8 {
            let mut level = nearest(0.5 * (inverse * scales[sub] - 1.0))?.clamp(0, 7) as u8;
            if shifts[sub] == 1 {
                level |= 8;
            }
            out[35 + sub * 2] |= level << 4;
        }
        return Ok(());
    }
    let mut packed_scales = [0_u16; 4];
    let masks = [0x00, 0x80, 0x08, 0x88];
    let mut sum_qx = 0.0;
    let mut sum_q2 = 0.0;
    for sub in 0..16 {
        let level = nearest(0.5 * (inverse * scales[sub] - 1.0))?.clamp(0, 7) as u16;
        packed_scales[sub / 4] |= level << (3 * (sub % 4));
        out[32 + sub] |= masks[shifts[sub]];
        let values = &x[sub * 16..(sub + 1) * 16];
        let importance = importance.map(|w| &w[sub * 16..(sub + 1) * 16]);
        let weight = weights::<16>(values, importance, sigma2)?;
        for group in 0..2 {
            let signed = signed_grid(true, shifts[sub], group);
            let index = out[2 * sub + group] as usize
                | (((out[32 + sub] as usize) << (8 - 4 * group)) & 0x700);
            for j in 0..8 {
                let q = signed[grid.levels[index][j] as usize] * (2 * level + 1) as f32;
                let w = weight[group * 8 + j];
                sum_qx += w * q * values[group * 8 + j];
                sum_q2 += w * q * q;
            }
        }
    }
    finite(sum_qx)?;
    finite(sum_q2)?;
    if sum_q2 > 0.0 {
        d = sum_qx / sum_q2;
    }
    let mut half = [0; 2];
    put_half(&mut half, 0, d * 1.1125)?;
    let half_bits = u16::from_le_bytes(half);
    for (i, scale) in packed_scales.iter_mut().enumerate() {
        *scale |= ((half_bits >> (i * 4)) & 15) << 12;
        out[48 + i * 2..50 + i * 2].copy_from_slice(&scale.to_le_bytes());
    }
    Ok(())
}
