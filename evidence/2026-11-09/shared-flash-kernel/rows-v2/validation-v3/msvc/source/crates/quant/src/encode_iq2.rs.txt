//! IQ2 exact scalar lattice encoders with explicit importance weights.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{
    DType, QuantError,
    encode_k::{finite, nearest, put_half},
    iq_lattice::{self, Grid8, key},
};

fn quantized_key(x: &[f32], inverse: f32, levels: &mut [i32]) -> Result<usize, ()> {
    for i in 0..8 {
        levels[i] = nearest(0.5 * (inverse * x[i] - 1.0))?.clamp(0, 2);
    }
    Ok(key(levels))
}
pub(crate) fn encode(
    dtype: DType,
    values: &[f32],
    importance: Option<&[f32]>,
) -> Result<Vec<u8>, QuantError> {
    let bytes = dtype.row_bytes(values.len())?;
    if dtype != DType::IQ2_S && importance.is_none() {
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
        let weights = importance.map(|w| &w[block * 256..(block + 1) * 256]);
        encode_block(dtype, grid, x, weights, out)
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
    let xxs = dtype == DType::IQ2_XXS;
    let small = dtype == DType::IQ2_S;
    let width = if xxs { 32 } else { 16 };
    let groups = width / 8;
    let blocks = 256 / width;
    let sum_x2 = x.iter().fold(0.0, |sum, &value| sum + value * value);
    let sigma2 = finite(if small {
        2.0 * sum_x2 / 256.0
    } else {
        sum_x2 / 256.0
    })?;
    let mut scales = [0.0; 16];
    let mut max_scale = 0.0_f32;
    for sub in 0..blocks {
        let values = &x[sub * width..(sub + 1) * width];
        let mut weights = [0.0; 32];
        let mut auxiliary_weights = [0.0; 32];
        for j in 0..width {
            weights[j] = finite(if let Some(importance) = importance {
                importance[sub * width + j] * (sigma2 + values[j] * values[j]).sqrt()
            } else {
                0.25 * sigma2 + values[j] * values[j]
            })?;
            auxiliary_weights[j] = weights[j].sqrt();
        }
        let mut magnitudes = [0.0; 32];
        let mut signs = [0_u8; 4];
        for (group, signs) in signs.iter_mut().enumerate().take(groups) {
            let mut flips = 0;
            let mut sign = 0_u8;
            for j in 0..8 {
                let i = group * 8 + j;
                if values[i] >= 0.0 {
                    magnitudes[i] = values[i];
                } else {
                    magnitudes[i] = -values[i];
                    flips += 1;
                    sign |= 1 << j;
                }
            }
            if !small && flips % 2 != 0 {
                let offset = group * 8;
                let mut selected = 0;
                let mut minimum = weights[offset] * values[offset] * values[offset];
                for j in 1..8 {
                    let i = offset + j;
                    let error = weights[i] * values[i] * values[i];
                    if error < minimum {
                        minimum = error;
                        selected = j;
                    }
                }
                magnitudes[offset + selected] = -magnitudes[offset + selected];
                sign ^= 1 << selected;
            }
            *signs = if small { sign } else { sign & 127 };
        }
        let mut maximum = magnitudes[0];
        for &value in &magnitudes[1..width] {
            if value > maximum {
                maximum = value;
            }
        }
        if maximum < if small { 1e-8 } else { 1e-15 } {
            continue;
        }
        let mut levels = [0; 32];
        let mut candidate = [0; 32];
        let (mut scale, effective_maximum) = if xxs {
            let scale = crate::weighted::make_positive(
                &magnitudes[..width],
                4,
                &mut levels[..width],
                &weights[..width],
            )?;
            for level in &mut levels[..width] {
                *level = *level as u8 as i8 as i32;
            }
            let effective = scale * 3.0;
            if effective <= 0.0 {
                continue;
            }
            (scale, effective)
        } else {
            (maximum / 5.0, maximum)
        };
        let mut on_grid = [true; 4];
        let mut best = 0.0;
        let range = if xxs { 6 } else { 9 };
        for step in -range..=range {
            let inverse = (5.0 + step as f32 * 0.1) / effective_maximum;
            let this_scale = 1.0 / inverse;
            let mut candidate_on_grid = [true; 4];
            for (group, candidate_on_grid) in candidate_on_grid.iter_mut().enumerate().take(groups)
            {
                let part = group * 8..(group + 1) * 8;
                let code = quantized_key(
                    &magnitudes[part.clone()],
                    inverse,
                    &mut candidate[part.clone()],
                )?;
                if grid.index(code).is_none() {
                    *candidate_on_grid = false;
                    grid.nearest_neighbour(
                        code,
                        &magnitudes[part.clone()],
                        &auxiliary_weights[part.clone()],
                        this_scale,
                        &mut candidate[part],
                    )?;
                }
            }
            let mut sum_qx = 0.0;
            let mut sum_q2 = 0.0;
            for j in 0..width {
                let q = (2 * candidate[j] + 1) as f32;
                sum_qx += weights[j] * magnitudes[j] * q;
                sum_q2 += weights[j] * q * q;
            }
            finite(sum_qx)?;
            finite(sum_q2)?;
            if sum_q2 > 0.0 && sum_qx * sum_qx > best * sum_q2 {
                scale = sum_qx / sum_q2;
                best = scale * sum_qx;
                levels.copy_from_slice(&candidate);
                on_grid = candidate_on_grid;
            }
        }
        if scale > 0.0 && (xxs || on_grid[..groups].iter().any(|&on| !on)) {
            let inverse = 1.0 / scale;
            for (group, &on_grid) in on_grid.iter().enumerate().take(groups) {
                if !xxs && on_grid {
                    continue;
                }
                let part = group * 8..(group + 1) * 8;
                let code = quantized_key(
                    &magnitudes[part.clone()],
                    inverse,
                    &mut levels[part.clone()],
                )?;
                if let Some(index) = grid.index(code) {
                    levels[part].copy_from_slice(&grid.levels[index]);
                } else {
                    grid.nearest_neighbour(
                        code,
                        &magnitudes[part.clone()],
                        &auxiliary_weights[part.clone()],
                        scale,
                        &mut levels[part],
                    )?;
                }
            }
            let mut sum_qx = 0.0;
            let mut sum_q2 = 0.0;
            for j in 0..width {
                let q = (2 * levels[j] + 1) as f32;
                sum_qx += weights[j] * magnitudes[j] * q;
                sum_q2 += weights[j] * q * q;
            }
            finite(sum_qx)?;
            finite(sum_q2)?;
            if sum_q2 > 0.0 {
                scale = sum_qx / sum_q2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for sign in &mut signs[..groups] {
                *sign = if small { !*sign } else { !*sign & 127 };
            }
        }
        finite(scale)?;
        let mut sign_word = 0_u32;
        for group in 0..groups {
            let index = grid
                .index(key(&levels[group * 8..(group + 1) * 8]))
                .ok_or(())?;
            if xxs {
                out[2 + sub * 8 + group] = index as u8;
                sign_word |= (signs[group] as u32) << (7 * group);
            } else if small {
                let i = sub * 2 + group;
                out[2 + i] = index as u8;
                out[66 + i / 4] |= ((index >> 8) as u8) << (2 * (i % 4));
                out[34 + i] = signs[group];
            } else {
                let word = index as u16 | ((signs[group] as u16) << 9);
                out[2 + sub * 4 + group * 2..4 + sub * 4 + group * 2]
                    .copy_from_slice(&word.to_le_bytes());
            }
        }
        if xxs {
            out[6 + sub * 8..10 + sub * 8].copy_from_slice(&sign_word.to_le_bytes());
        }
        scales[sub] = scale;
        if scale > max_scale {
            max_scale = scale;
        }
    }
    if max_scale == 0.0 {
        if !small {
            out.fill(0);
        }
        return Ok(());
    }
    let d = max_scale / 31.0;
    put_half(out, 0, if small { d * 0.9875 } else { d })?;
    let inverse = 1.0 / d;
    for sub in 0..blocks {
        let level = nearest(0.5 * (inverse * scales[sub] - 1.0))?.clamp(0, 15) as u8;
        if xxs {
            out[9 + sub * 8] |= level << 4;
        } else {
            let offset = if small { 74 } else { 66 };
            if sub % 2 == 0 {
                out[offset + sub / 2] = level;
            } else {
                out[offset + sub / 2] |= level << 4;
            }
        }
    }
    Ok(())
}
