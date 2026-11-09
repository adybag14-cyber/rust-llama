//! Remaining scalar packed dots from the frozen ggml-cpu/quants.c.
//! Codebook values remain integers until the source block scale operation.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::tables::*;
use crate::{DType, f16_to_f32, half, u16le, u32le};

pub(crate) fn dot(t: DType, x: &[u8], y: &[u8]) -> f32 {
    use DType::*;
    match t {
        Q1_0 | Q2_0 => dot_q12(t, x, y),
        MXFP4 | NVFP4 | IQ4_NL => dot_small_codebook(t, x, y),
        TQ1_0 | TQ2_0 => dot_ternary(t, x, y),
        Q2_K => dot_q2k(x, y),
        Q3_K | Q4_K | Q5_K | Q6_K => dot_k_lanes(t, x, y),
        IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S => dot_iq23(t, x, y),
        IQ1_S | IQ1_M => dot_iq1(t, x, y),
        IQ4_XS => dot_iq4xs(x, y),
        _ => unreachable!("validated extended packed dot type"),
    }
}

fn f32le(b: &[u8], p: usize) -> f32 {
    f32::from_bits(u32le(b, p))
}
fn q8k(b: &[u8], i: usize) -> i32 {
    b[4 + i] as i8 as i32
}
fn bsum(b: &[u8], i: usize) -> i32 {
    u16le(b, 260 + 2 * i) as i16 as i32
}
fn signed(signs: u8, bit: usize) -> i32 {
    if signs & (1 << bit) == 0 { 1 } else { -1 }
}

fn dot_q12(t: DType, x: &[u8], y: &[u8]) -> f32 {
    let groups = t.block_size() / 32;
    let mut sum = 0.0;
    for (xb, yb) in x
        .as_chunks::<18>()
        .0
        .iter()
        .zip(y.chunks_exact(groups * 34))
    {
        let mut local = 0.0;
        for group in 0..groups {
            let yr = &yb[group * 34..(group + 1) * 34];
            let mut isum = 0;
            for j in 0..32 {
                let index = group * 32 + j;
                let q = if t == DType::Q1_0 {
                    if xb[2 + index / 8] & (1 << (index % 8)) != 0 {
                        1
                    } else {
                        -1
                    }
                } else {
                    ((xb[2 + index / 4] >> (2 * (index % 4))) & 3) as i32 - 1
                };
                isum += q * yr[2 + j] as i8 as i32;
            }
            local += half(yr, 0) * isum as f32;
        }
        sum += half(xb, 0) * local;
    }
    sum
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

fn dot_small_codebook(t: DType, x: &[u8], y: &[u8]) -> f32 {
    let mut sum = 0.0;
    let yn = t.block_size() / 32 * 34;
    for (xb, yb) in x.chunks_exact(t.type_size()).zip(y.chunks_exact(yn)) {
        if t == DType::NVFP4 {
            for sub in 0..4 {
                let yr = &yb[(sub / 2) * 34..(sub / 2 + 1) * 34];
                let offset = (sub % 2) * 16;
                let mut isum = 0;
                for j in 0..8 {
                    let code = xb[4 + sub * 8 + j];
                    isum +=
                        yr[2 + offset + j] as i8 as i32 * KVALUES_FP4[(code & 15) as usize] as i32;
                    isum += yr[2 + offset + j + 8] as i8 as i32
                        * KVALUES_FP4[(code >> 4) as usize] as i32;
                }
                sum += half(yr, 0) * ue4m3_half(xb[sub]) * isum as f32;
            }
        } else {
            let scale = if t == DType::MXFP4 {
                e8m0_half(xb[0])
            } else {
                half(xb, 0)
            };
            let position = if t == DType::MXFP4 { 1 } else { 2 };
            let table = if t == DType::MXFP4 {
                &KVALUES_FP4
            } else {
                &KVALUES_IQ4NL
            };
            let mut isum = 0;
            for j in 0..16 {
                let code = xb[position + j];
                isum += yb[2 + j] as i8 as i32 * table[(code & 15) as usize] as i32;
                isum += yb[2 + j + 16] as i8 as i32 * table[(code >> 4) as usize] as i32;
            }
            sum += (half(yb, 0) * scale) * isum as f32;
        }
    }
    sum
}

fn dot_ternary(t: DType, x: &[u8], y: &[u8]) -> f32 {
    let mut sum = 0.0;
    for (xb, yb) in x
        .chunks_exact(t.type_size())
        .zip(y.as_chunks::<292>().0.iter())
    {
        let mut isum = 0;
        if t == DType::TQ1_0 {
            let powers = [1u8, 3, 9, 27, 81];
            let mut index = 0;
            for (start, length) in [(0, 32), (32, 16)] {
                for power in powers {
                    for j in 0..length {
                        let code = xb[start + j].wrapping_mul(power);
                        let value = ((code as u16 * 3) >> 8) as i32 - 1;
                        isum += value * q8k(yb, index);
                        index += 1;
                    }
                }
            }
            for power in &powers[..4] {
                for j in 0..4 {
                    let code = xb[48 + j].wrapping_mul(*power);
                    let value = ((code as u16 * 3) >> 8) as i32 - 1;
                    isum += value * q8k(yb, index);
                    index += 1;
                }
            }
            sum += isum as f32 * (half(xb, 52) * f32le(yb, 0));
        } else {
            for j in 0..256 {
                let group = j / 128;
                let lane = (j % 128) / 32;
                let pos = j % 32;
                let code = ((xb[group * 32 + pos] >> (2 * lane)) & 3) as i32 - 1;
                isum += q8k(yb, j) * code;
            }
            sum += isum as f32 * (f32le(yb, 0) * half(xb, 64));
        }
    }
    sum
}

fn dot_q2k(x: &[u8], y: &[u8]) -> f32 {
    let mut sum = 0.0;
    for (xb, yb) in x
        .as_chunks::<84>()
        .0
        .iter()
        .zip(y.as_chunks::<292>().0.iter())
    {
        let mut mins = 0;
        for (j, scale) in xb[..16].iter().enumerate() {
            mins += bsum(yb, j) * (scale >> 4) as i32;
        }
        let mut isum = 0;
        for j in 0..256 {
            let group = j / 128;
            let lane = (j % 128) / 32;
            let pos = j % 32;
            let q = (xb[16 + group * 32 + pos] >> (2 * lane)) & 3;
            isum += (xb[j / 16] & 15) as i32 * q as i32 * q8k(yb, j);
        }
        let d = f32le(yb, 0) * half(xb, 80);
        let min = f32le(yb, 0) * half(xb, 82);
        sum += d * isum as f32 - min * mins as f32;
    }
    sum
}

fn q3_scales(x: &[u8]) -> [i32; 16] {
    let p = &x[96..108];
    let mut scales = [0; 16];
    for j in 0..4 {
        scales[j] = ((p[j] & 15) | ((p[8 + j] & 3) << 4)) as i32 - 32;
        scales[4 + j] = ((p[4 + j] & 15) | (((p[8 + j] >> 2) & 3) << 4)) as i32 - 32;
        scales[8 + j] = ((p[j] >> 4) | (((p[8 + j] >> 4) & 3) << 4)) as i32 - 32;
        scales[12 + j] = ((p[4 + j] >> 4) | (((p[8 + j] >> 6) & 3) << 4)) as i32 - 32;
    }
    scales
}
fn scale_min(p: &[u8], j: usize) -> (i32, i32) {
    if j < 4 {
        ((p[j] & 63) as i32, (p[j + 4] & 63) as i32)
    } else {
        (
            ((p[j + 4] & 15) | ((p[j - 4] >> 6) << 4)) as i32,
            ((p[j + 4] >> 4) | ((p[j] >> 6) << 4)) as i32,
        )
    }
}

fn dot_k_lanes(t: DType, x: &[u8], y: &[u8]) -> f32 {
    // The original generic kernels retain eight floating accumulators across
    // ALL blocks, then add the lanes to the separate minimum correction sum.
    let mut lanes = [0.0; 8];
    let mut sum = 0.0;
    for (xb, yb) in x
        .chunks_exact(t.type_size())
        .zip(y.as_chunks::<292>().0.iter())
    {
        let mut integers = [0; 8];
        let d;
        match t {
            DType::Q3_K => {
                let scales = q3_scales(xb);
                for j in 0..256 {
                    let group = j / 128;
                    let lane = (j % 128) / 32;
                    let pos = j % 32;
                    let low = ((xb[32 + group * 32 + pos] >> (2 * lane)) & 3) as i32;
                    let q = low
                        - if xb[pos] & (1 << (group * 4 + lane)) == 0 {
                            4
                        } else {
                            0
                        };
                    integers[j % 8] += scales[j / 16] * q * q8k(yb, j);
                }
                d = half(xb, 108) * f32le(yb, 0);
            }
            DType::Q4_K | DType::Q5_K => {
                let five = t == DType::Q5_K;
                let qpos = if five { 48 } else { 16 };
                let mut mins = 0;
                for sub in 0..8 {
                    let (scale, min) = scale_min(&xb[4..16], sub);
                    mins += (bsum(yb, 2 * sub) + bsum(yb, 2 * sub + 1)) * min;
                    for l in 0..32 {
                        let j = sub * 32 + l;
                        let mut q = ((xb[qpos + sub / 2 * 32 + l] >> (4 * (sub % 2))) & 15) as i32;
                        if five && xb[16 + l] & (1 << sub) != 0 {
                            q += 16;
                        }
                        integers[j % 8] += scale * q * q8k(yb, j);
                    }
                }
                d = half(xb, 0) * f32le(yb, 0);
                let dmin = half(xb, 2) * f32le(yb, 0);
                sum -= dmin * mins as f32;
            }
            DType::Q6_K => {
                for j in 0..256 {
                    let group = j / 128;
                    let side = (j % 128) / 32;
                    let pos = j % 32;
                    let low = (xb[group * 64 + (side % 2) * 32 + pos] >> (4 * (side / 2))) & 15;
                    let high = (xb[128 + group * 32 + pos] >> (2 * side)) & 3;
                    let q = (low | (high << 4)) as i32 - 32;
                    integers[j % 8] += xb[192 + j / 16] as i8 as i32 * q * q8k(yb, j);
                }
                d = half(xb, 208) * f32le(yb, 0);
            }
            _ => unreachable!("validated K lane dtype"),
        }
        for l in 0..8 {
            lanes[l] += d * integers[l] as f32;
        }
    }
    for lane in lanes {
        sum += lane;
    }
    sum
}

fn dot_iq23(t: DType, x: &[u8], y: &[u8]) -> f32 {
    let mut sum = 0.0;
    for (xb, yb) in x
        .chunks_exact(t.type_size())
        .zip(y.as_chunks::<292>().0.iter())
    {
        let mut block_sum = 0;
        for sub in 0..8 {
            match t {
                DType::IQ2_XXS => {
                    let aux = u32le(xb, 2 + sub * 8 + 4);
                    let scale = (2 * (aux >> 28) + 1) as i32;
                    let mut isum = 0;
                    for l in 0..4 {
                        let grid = IQ2XXS_GRID[xb[2 + sub * 8 + l] as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for (j, &grid_value) in grid.iter().enumerate() {
                            isum += grid_value as i32
                                * q8k(yb, sub * 32 + l * 8 + j)
                                * signed(signs, j);
                        }
                    }
                    block_sum += isum * scale;
                }
                DType::IQ2_XS | DType::IQ2_S => {
                    for part in 0..2 {
                        let scale_byte = xb[if t == DType::IQ2_XS {
                            66 + sub
                        } else {
                            74 + sub
                        }];
                        let scale = 2 * ((scale_byte >> (part * 4)) & 15) as i32 + 1;
                        let mut isum = 0;
                        for l in part * 2..part * 2 + 2 {
                            let (grid, signs) = if t == DType::IQ2_XS {
                                let code = u16le(xb, 2 + sub * 8 + l * 2);
                                (
                                    IQ2XS_GRID[(code & 511) as usize].to_le_bytes(),
                                    KSIGNS_IQ2XS[(code >> 9) as usize],
                                )
                            } else {
                                let code = xb[2 + sub * 4 + l] as usize
                                    | (((xb[66 + sub] as usize) << (8 - 2 * l)) & 0x300);
                                (IQ2S_GRID[code].to_le_bytes(), xb[34 + sub * 4 + l])
                            };
                            for (j, &grid_value) in grid.iter().enumerate() {
                                isum += grid_value as i32
                                    * q8k(yb, sub * 32 + l * 8 + j)
                                    * signed(signs, j);
                            }
                        }
                        block_sum += isum * scale;
                    }
                }
                DType::IQ3_XXS | DType::IQ3_S => {
                    let aux = if t == DType::IQ3_XXS {
                        u32le(xb, 66 + sub * 4)
                    } else {
                        0
                    };
                    let scale = if t == DType::IQ3_XXS {
                        2 * (aux >> 28) as i32 + 1
                    } else {
                        2 * ((xb[106 + sub / 2] >> (4 * (sub % 2))) & 15) as i32 + 1
                    };
                    let mut isum = 0;
                    for l in 0..4 {
                        let signs = if t == DType::IQ3_XXS {
                            KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize]
                        } else {
                            xb[74 + sub * 4 + l]
                        };
                        for part in 0..2 {
                            let code = xb[2 + sub * 8 + l * 2 + part] as usize;
                            let grid = if t == DType::IQ3_XXS {
                                IQ3XXS_GRID[code].to_le_bytes()
                            } else {
                                let high = ((xb[66 + sub] as usize) << (8 - 2 * l - part)) & 256;
                                IQ3S_GRID[code | high].to_le_bytes()
                            };
                            for (j, &grid_value) in grid.iter().enumerate() {
                                isum += grid_value as i32
                                    * q8k(yb, sub * 32 + l * 8 + part * 4 + j)
                                    * signed(signs, part * 4 + j);
                            }
                        }
                    }
                    block_sum += isum * scale;
                }
                _ => unreachable!("validated IQ2/IQ3 dtype"),
            }
        }
        sum += (half(xb, 0) * f32le(yb, 0)) * block_sum as f32;
    }
    match t {
        DType::IQ2_XXS | DType::IQ2_XS | DType::IQ2_S => 0.125 * sum,
        DType::IQ3_XXS => 0.25 * sum,
        _ => sum,
    }
}

fn dot_iq1(t: DType, x: &[u8], y: &[u8]) -> f32 {
    let mut sum = 0.0;
    for (xb, yb) in x
        .chunks_exact(t.type_size())
        .zip(y.as_chunks::<292>().0.iter())
    {
        let mut integer = 0;
        let mut correction = 0;
        let scale;
        if t == DType::IQ1_S {
            scale = half(xb, 0);
            for sub in 0..8 {
                let high = u16le(xb, 34 + sub * 2);
                let local_scale = 2 * ((high >> 12) & 7) as i32 + 1;
                let delta = if high & 0x8000 != 0 { -1 } else { 1 };
                let mut local = 0;
                for l in 0..4 {
                    let code =
                        xb[2 + sub * 4 + l] as usize | ((((high >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[code].to_le_bytes();
                    for (j, &grid_value) in grid.iter().enumerate() {
                        local += q8k(yb, sub * 32 + l * 8 + j) * grid_value as i8 as i32;
                    }
                }
                integer += local_scale * local;
                correction += local_scale * delta * (bsum(yb, sub * 2) + bsum(yb, sub * 2 + 1));
            }
        } else {
            let sc = [u16le(xb, 48), u16le(xb, 50), u16le(xb, 52), u16le(xb, 54)];
            let scale_bits =
                (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
            scale = f16_to_f32(scale_bits);
            for sub in 0..8 {
                let mut local = [0; 2];
                let mut offset = [0; 2];
                for l in 0..4 {
                    let high = xb[32 + sub * 2 + l / 2];
                    let code =
                        xb[sub * 4 + l] as usize | (((high as usize) << (8 - 4 * (l % 2))) & 0x700);
                    let delta = if high & (if l % 2 == 0 { 8 } else { 128 }) != 0 {
                        -1
                    } else {
                        1
                    };
                    let grid = IQ1S_GRID[code].to_le_bytes();
                    let mut sum_grid = 0;
                    let mut sum_q8 = 0;
                    for (j, &grid_value) in grid.iter().enumerate() {
                        let q = q8k(yb, sub * 32 + l * 8 + j);
                        sum_grid += q * grid_value as i8 as i32;
                        sum_q8 += q;
                    }
                    local[l / 2] += sum_grid;
                    offset[l / 2] += sum_q8 * delta;
                }
                let scale1 = 2 * ((sc[sub / 2] >> (6 * (sub % 2))) & 7) as i32 + 1;
                let scale2 = 2 * ((sc[sub / 2] >> (6 * (sub % 2) + 3)) & 7) as i32 + 1;
                integer += local[0] * scale1 + local[1] * scale2;
                correction += offset[0] * scale1 + offset[1] * scale2;
            }
        }
        sum += scale * f32le(yb, 0) * (integer as f32 + 0.125 * correction as f32);
    }
    sum
}

fn dot_iq4xs(x: &[u8], y: &[u8]) -> f32 {
    let mut sum = 0.0;
    for (xb, yb) in x
        .as_chunks::<136>()
        .0
        .iter()
        .zip(y.as_chunks::<292>().0.iter())
    {
        let d = half(xb, 0) * f32le(yb, 0);
        let high = u16le(xb, 2);
        for sub in 0..8 {
            let scale = (((xb[4 + sub / 2] >> (4 * (sub % 2))) & 15) as i32
                | ((((high >> (2 * sub)) & 3) as i32) << 4))
                - 32;
            let mut isum = 0;
            for j in 0..16 {
                let code = xb[8 + sub * 16 + j];
                isum += q8k(yb, sub * 32 + j) * KVALUES_IQ4NL[(code & 15) as usize] as i32;
                isum += q8k(yb, sub * 32 + j + 16) * KVALUES_IQ4NL[(code >> 4) as usize] as i32;
            }
            sum += (d * scale as f32) * isum as f32;
        }
    }
    sum
}
