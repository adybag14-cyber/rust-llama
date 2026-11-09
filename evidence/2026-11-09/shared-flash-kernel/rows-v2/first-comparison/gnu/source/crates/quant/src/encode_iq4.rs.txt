//! Exact IQ4 reference and importance-weighted codebook search.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{
    DType, QuantError,
    encode_k::{finite, nearest, put_half},
    tables::KVALUES_IQ4NL,
};

fn best_index(midpoints: &[f32; 15], value: f32) -> u8 {
    let mut index = 15;
    for &midpoint in midpoints {
        index -= u8::from(value < midpoint);
    }
    index
}

pub(crate) fn encode(
    dtype: DType,
    values: &[f32],
    weights: Option<&[f32]>,
) -> Result<Vec<u8>, QuantError> {
    let bytes = dtype.row_bytes(values.len())?;
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
    let size = dtype.block_size();
    for (block, (x, out)) in values
        .chunks_exact(size)
        .zip(output.chunks_exact_mut(dtype.type_size()))
        .enumerate()
    {
        let w = weights.map(|w| &w[block * size..(block + 1) * size]);
        encode_block(dtype, x, w, out).map_err(|()| QuantError::UnrepresentableScale { block })?;
    }
    Ok(output)
}

fn encode_block(
    dtype: DType,
    x: &[f32],
    importance: Option<&[f32]>,
    out: &mut [u8],
) -> Result<(), ()> {
    let refined = dtype == DType::IQ4_XS || importance.is_some();
    let tries = if refined { 7 } else { -1 };
    let lanes = if tries >= 0 {
        (2 * tries + 2) as usize
    } else {
        1
    };
    let blocks = x.len() / 32;
    let mut sigma2 = 0.0_f32;
    for &v in x {
        sigma2 += v * v;
    }
    sigma2 *= 2.0 / x.len() as f32;
    finite(sigma2)?;
    let midpoints =
        std::array::from_fn(|i| 0.5 * (KVALUES_IQ4NL[i] as f32 + KVALUES_IQ4NL[i + 1] as f32));
    let mut scales = [0.0_f32; 8];
    // The original unrefined IQ4_NL function leaves this scratch array
    // uninitialized for a near-zero block. Give that undefined payload its
    // canonical zero-byte representation; the stored scale is still zero.
    let mut levels = [0_u8; 256];
    let mut maximum_scale = 0.0_f32;
    let mut maximum_magnitude = 0.0_f32;
    for sub in 0..blocks {
        let values = &x[sub * 32..(sub + 1) * 32];
        let mut weight = [0.0; 32];
        let mut maximum = 0.0_f32;
        let mut magnitude = 0.0_f32;
        for j in 0..32 {
            weight[j] = finite(if let Some(importance) = importance {
                importance[sub * 32 + j] * (sigma2 + values[j] * values[j]).sqrt()
            } else {
                values[j] * values[j]
            })?;
            if values[j].abs() > magnitude {
                magnitude = values[j].abs();
                maximum = values[j];
            }
        }
        if magnitude < 1e-15 {
            continue;
        }
        let mut d = if tries > 0 {
            -maximum / KVALUES_IQ4NL[0] as f32
        } else {
            maximum / KVALUES_IQ4NL[0] as f32
        };
        let mut inverse = [0.0; 16];
        let mut sum_qx = [0.0; 16];
        let mut sum_q2 = [0.0; 16];
        inverse[0] = finite(1.0 / d)?;
        for (lane, value) in inverse.iter_mut().enumerate().take(lanes).skip(1) {
            *value = finite((lane as i32 - 1 - tries + KVALUES_IQ4NL[0] as i32) as f32 / maximum)?;
        }
        for j in 0..32 {
            levels[sub * 32 + j] = best_index(&midpoints, inverse[0] * values[j]);
        }
        for j in 0..32 {
            for lane in 0..lanes {
                let q = KVALUES_IQ4NL[best_index(&midpoints, inverse[lane] * values[j]) as usize]
                    as f32;
                sum_qx[lane] += weight[j] * q * values[j];
                sum_q2[lane] += weight[j] * q * q;
            }
        }
        for lane in 0..lanes {
            finite(sum_qx[lane])?;
            finite(sum_q2[lane])?;
        }
        d = if sum_q2[0] > 0.0 {
            sum_qx[0] / sum_q2[0]
        } else {
            0.0
        };
        let mut best = d * sum_qx[0];
        for lane in 1..lanes {
            if sum_q2[lane] > 0.0 && sum_qx[lane] * sum_qx[lane] > best * sum_q2[lane] {
                d = sum_qx[lane] / sum_q2[lane];
                best = d * sum_qx[lane];
            }
        }
        scales[sub] = finite(d)?;
        if d.abs() > maximum_magnitude {
            maximum_magnitude = d.abs();
            maximum_scale = d;
        }
    }
    let qpos = if dtype == DType::IQ4_XS {
        let d = -maximum_scale / 32.0;
        put_half(out, 0, d)?;
        let inverse = if d != 0.0 { finite(1.0 / d)? } else { 0.0 };
        let mut high = 0_u16;
        for sub in 0..blocks {
            let l = nearest(inverse * scales[sub])?.clamp(-32, 31);
            // Source requantizes with the unrounded block scale, not the stored half.
            let scale = d * l as f32;
            let inverse = if scale != 0.0 {
                finite(1.0 / scale)?
            } else {
                0.0
            };
            for j in 0..32 {
                levels[sub * 32 + j] = best_index(&midpoints, inverse * x[sub * 32 + j]);
            }
            let l = (l + 32) as u8;
            if sub % 2 == 0 {
                out[4 + sub / 2] = l & 15;
            } else {
                out[4 + sub / 2] |= (l & 15) << 4;
            }
            high |= ((l >> 4) as u16) << (2 * sub);
        }
        out[2..4].copy_from_slice(&high.to_le_bytes());
        8
    } else {
        put_half(out, 0, scales[0])?;
        if tries > 0 {
            let inverse = if scales[0] != 0.0 {
                finite(1.0 / scales[0])?
            } else {
                0.0
            };
            for j in 0..32 {
                levels[j] = best_index(&midpoints, inverse * x[j]);
            }
        }
        2
    };
    for sub in 0..blocks {
        for j in 0..16 {
            out[qpos + sub * 16 + j] = levels[sub * 32 + j] | (levels[sub * 32 + j + 16] << 4);
        }
    }
    Ok(())
}
