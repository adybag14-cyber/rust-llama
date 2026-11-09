use crate::tables::*;
use crate::{DType, bf16_to_f32, f8_e4m3_to_f32, half, u16le, u32le};

pub(crate) fn block(t: DType, b: &[u8], y: &mut [f32]) {
    use DType::*;
    match t {
        F32 => y[0] = f32::from_bits(u32le(b, 0)),
        F16 => y[0] = half(b, 0),
        BF16 => y[0] = bf16_to_f32(u16le(b, 0)),
        F64 => y[0] = f64::from_le_bytes(b.try_into().expect("validated scalar")) as f32,
        I8 => y[0] = b[0] as i8 as f32,
        I16 => y[0] = u16le(b, 0) as i16 as f32,
        I32 => y[0] = u32le(b, 0) as i32 as f32,
        I64 => y[0] = i64::from_le_bytes(b.try_into().expect("validated scalar")) as f32,
        F8_E4M3 => y[0] = f8_e4m3_to_f32(b[0]),
        Q4_0 | Q4_1 | Q5_0 | Q5_1 => legacy(t, b, y),
        Q8_0 | Q8_1 => {
            let d = half(b, 0);
            let p = if t == Q8_0 { 2 } else { 4 };
            for j in 0..32 {
                y[j] = b[p + j] as i8 as f32 * d;
            }
        }
        Q1_0 => {
            let d = half(b, 0);
            for j in 0..128 {
                y[j] = if b[2 + j / 8] & (1 << (j % 8)) != 0 {
                    d
                } else {
                    -d
                };
            }
        }
        Q2_0 => {
            let d = half(b, 0);
            for j in 0..64 {
                y[j] = (((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32 * d;
            }
        }
        Q2_K => q2k(b, y),
        Q3_K => q3k(b, y),
        Q4_K | Q5_K => q45k(t, b, y),
        Q6_K => q6k(b, y),
        Q8_K => {
            let d = f32::from_bits(u32le(b, 0));
            for j in 0..256 {
                y[j] = d * b[4 + j] as i8 as f32;
            }
        }
        TQ1_0 => tq1(b, y),
        TQ2_0 => {
            let d = half(b, 64);
            for group in 0..2 {
                for lane in 0..4 {
                    for j in 0..32 {
                        y[128 * group + 32 * lane + j] =
                            (((b[group * 32 + j] >> (2 * lane)) & 3) as i32 - 1) as f32 * d;
                    }
                }
            }
        }
        MXFP4 => {
            // Upstream deliberately maps all exponents, including 255.
            let d = f32::from_bits(if b[0] < 2 {
                0x00200000 << b[0]
            } else {
                (b[0] as u32 - 1) << 23
            });
            for j in 0..16 {
                y[j] = KVALUES_FP4[(b[j + 1] & 15) as usize] as f32 * d;
                y[j + 16] = KVALUES_FP4[(b[j + 1] >> 4) as usize] as f32 * d;
            }
        }
        NVFP4 => {
            for s in 0..4 {
                // Preserve this snapshot's UE4M3 convention: half-scale, 0x7f -> 0,
                // and ignored sign bit. Do not silently replace it with FP8.
                let x = b[s];
                let d = if x == 0 || x == 127 {
                    0.0
                } else {
                    let exp = (x >> 3) & 15;
                    let man = x & 7;
                    let raw = if exp == 0 {
                        man as f32 / 512.0
                    } else {
                        (1.0 + man as f32 / 8.0) * f32::from_bits((exp as u32 + 120) << 23)
                    };
                    raw * 0.5
                };
                for j in 0..8 {
                    let q = b[4 + s * 8 + j];
                    y[s * 16 + j] = KVALUES_FP4[(q & 15) as usize] as f32 * d;
                    y[s * 16 + j + 8] = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
                }
            }
        }
        IQ2_XXS => iq2xxs(b, y),
        IQ2_XS => iq2xs(b, y),
        IQ2_S => iq2s(b, y),
        IQ3_XXS => iq3xxs(b, y),
        IQ3_S => iq3s(b, y),
        IQ1_S => iq1s(b, y),
        IQ1_M => iq1m(b, y),
        IQ4_NL => {
            let d = half(b, 0);
            for j in 0..16 {
                y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 15) as usize] as f32;
                y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
            }
        }
        IQ4_XS => {
            let d = half(b, 0);
            let high = u16le(b, 2);
            for ib in 0..8 {
                let sc = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) as i32
                    | ((((high >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (sc - 32) as f32;
                for j in 0..16 {
                    y[32 * ib + j] = dl * KVALUES_IQ4NL[(b[8 + 16 * ib + j] & 15) as usize] as f32;
                    y[32 * ib + j + 16] =
                        dl * KVALUES_IQ4NL[(b[8 + 16 * ib + j] >> 4) as usize] as f32;
                }
            }
        }
    }
}

fn legacy(t: DType, b: &[u8], y: &mut [f32]) {
    let affine = matches!(t, DType::Q4_1 | DType::Q5_1);
    let five = matches!(t, DType::Q5_0 | DType::Q5_1);
    let d = half(b, 0);
    let min = if affine { half(b, 2) } else { 0.0 };
    let p = if affine { 4 } else { 2 };
    let high = if five { u32le(b, p) } else { 0 };
    let qpos = p + if five { 4 } else { 0 };
    for j in 0..16 {
        let low = b[qpos + j];
        for side in 0..2 {
            let v = if side == 0 { low & 15 } else { low >> 4 };
            let v = v as i32
                | if five {
                    (((high >> (j + side * 16)) & 1) << 4) as i32
                } else {
                    0
                };
            let v = v - if affine {
                0
            } else if five {
                16
            } else {
                8
            };
            let scaled = v as f32 * d;
            y[j + side * 16] = if affine { scaled + min } else { scaled };
        }
    }
}

fn q2k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 80);
    let min = half(b, 82);
    for group in 0..2 {
        for lane in 0..4 {
            for side in 0..2 {
                let sc = b[group * 8 + lane * 2 + side];
                let dl = d * (sc & 15) as f32;
                let ml = min * (sc >> 4) as f32;
                for j in 0..16 {
                    y[group * 128 + lane * 32 + side * 16 + j] =
                        dl * ((b[16 + group * 32 + side * 16 + j] >> (2 * lane)) & 3) as f32 - ml;
                }
            }
        }
    }
}

fn q3k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 108);
    let q = &b[32..96];
    let packed = &b[96..108];
    let mut sc = [0i32; 16];
    for j in 0..4 {
        sc[j] = (packed[j] & 15) as i32 | (((packed[8 + j] & 3) as i32) << 4);
        sc[4 + j] = (packed[4 + j] & 15) as i32 | ((((packed[8 + j] >> 2) & 3) as i32) << 4);
        sc[8 + j] = (packed[j] >> 4) as i32 | ((((packed[8 + j] >> 4) & 3) as i32) << 4);
        sc[12 + j] = (packed[4 + j] >> 4) as i32 | ((((packed[8 + j] >> 6) & 3) as i32) << 4);
    }
    for group in 0..2 {
        for lane in 0..4 {
            let mask = 1 << (group * 4 + lane);
            for side in 0..2 {
                let dl = d * (sc[group * 8 + lane * 2 + side] - 32) as f32;
                for j in 0..16 {
                    let pos = side * 16 + j;
                    let v = ((q[group * 32 + pos] >> (2 * lane)) & 3) as i32
                        - if b[pos] & mask != 0 { 0 } else { 4 };
                    y[group * 128 + lane * 32 + pos] = dl * v as f32;
                }
            }
        }
    }
}

fn scale_min(sc: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        (
            (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
        )
    }
}

fn q45k(t: DType, b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let min = half(b, 2);
    let five = t == DType::Q5_K;
    let qbase = if five { 48 } else { 16 };
    for group in 0..4 {
        for side in 0..2 {
            let (sc, m) = scale_min(&b[4..16], group * 2 + side);
            let dl = d * sc as f32;
            let ml = min * m as f32;
            for j in 0..32 {
                let mut q = (b[qbase + group * 32 + j] >> (side * 4)) & 15;
                if five && b[16 + j] & (1 << (group * 2 + side)) != 0 {
                    q += 16;
                }
                y[group * 64 + side * 32 + j] = dl * q as f32 - ml;
            }
        }
    }
}

fn q6k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 208);
    for group in 0..2 {
        for j in 0..32 {
            let lo0 = b[group * 64 + j];
            let lo1 = b[group * 64 + 32 + j];
            let hi = b[128 + group * 32 + j];
            let values = [
                (lo0 & 15) | ((hi & 3) << 4),
                (lo1 & 15) | (((hi >> 2) & 3) << 4),
                (lo0 >> 4) | (((hi >> 4) & 3) << 4),
                (lo1 >> 4) | (((hi >> 6) & 3) << 4),
            ];
            for side in 0..4 {
                y[group * 128 + side * 32 + j] = d
                    * b[192 + group * 8 + j / 16 + side * 2] as i8 as f32
                    * (values[side] as i32 - 32) as f32;
            }
        }
    }
}

fn tq1(b: &[u8], y: &mut [f32]) {
    let d = half(b, 52);
    let powers = [1u8, 3, 9, 27, 81];
    let mut out = 0;
    for (start, len) in [(0, 32), (32, 16)] {
        for &pow in &powers {
            for j in 0..len {
                let q = b[start + j].wrapping_mul(pow);
                y[out] = (((q as u16 * 3) >> 8) as i32 - 1) as f32 * d;
                out += 1;
            }
        }
    }
    for &pow in &powers[..4] {
        for j in 0..4 {
            let q = b[48 + j].wrapping_mul(pow);
            y[out] = (((q as u16 * 3) >> 8) as i32 - 1) as f32 * d;
            out += 1;
        }
    }
}

fn sign(signs: u8, j: usize) -> f32 {
    if signs & (1 << j) != 0 { -1.0 } else { 1.0 }
}

fn iq2xxs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        let aux = u32le(b, 2 + ib * 8 + 4);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[b[2 + ib * 8 + l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db * grid[j] as f32 * sign(signs, j);
            }
        }
    }
}

fn iq2xs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        for l in 0..4 {
            let db = d * (0.5 + ((b[66 + ib] >> (4 * (l / 2))) & 15) as f32) * 0.25;
            let q = u16le(b, 2 + ib * 8 + l * 2);
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db * grid[j] as f32 * sign(signs, j);
            }
        }
    }
}

fn iq2s(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        for l in 0..4 {
            let db = d * (0.5 + ((b[74 + ib] >> (4 * (l / 2))) & 15) as f32) * 0.25;
            let q = b[2 + ib * 4 + l] as usize | (((b[66 + ib] as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[q].to_le_bytes();
            let signs = b[34 + ib * 4 + l];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db * grid[j] as f32 * sign(signs, j);
            }
        }
    }
}

fn iq3xxs(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        let aux = u32le(b, 66 + ib * 4);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            for part in 0..2 {
                let grid = IQ3XXS_GRID[b[2 + ib * 8 + l * 2 + part] as usize].to_le_bytes();
                for j in 0..4 {
                    y[ib * 32 + l * 8 + part * 4 + j] =
                        db * grid[j] as f32 * sign(signs, part * 4 + j);
                }
            }
        }
    }
}

fn iq3s(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        let db = d * (1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15)) as f32;
        for l in 0..4 {
            let signs = b[74 + ib * 4 + l];
            for part in 0..2 {
                let q = b[2 + ib * 8 + l * 2 + part] as usize
                    | (((b[66 + ib] as usize) << (8 - 2 * l - part)) & 256);
                let grid = IQ3S_GRID[q].to_le_bytes();
                for j in 0..4 {
                    y[ib * 32 + l * 8 + part * 4 + j] =
                        db * grid[j] as f32 * sign(signs, part * 4 + j);
                }
            }
        }
    }
}

fn iq1s(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for ib in 0..8 {
        let high = u16le(b, 34 + ib * 2);
        let dl = d * (2 * ((high >> 12) & 7) + 1) as f32;
        let delta = if high & 0x8000 != 0 { -0.125 } else { 0.125 };
        for l in 0..4 {
            let q = b[2 + ib * 4 + l] as usize | ((((high >> (3 * l)) & 7) as usize) << 8);
            let grid = IQ1S_GRID[q].to_le_bytes();
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid[j] as i8 as f32 + delta);
            }
        }
    }
}

fn iq1m(b: &[u8], y: &mut [f32]) {
    let sc = [u16le(b, 48), u16le(b, 50), u16le(b, 52), u16le(b, 54)];
    let scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
    let d = crate::f16_to_f32(scale);
    for ib in 0..8 {
        for l in 0..4 {
            let dl = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1) as f32;
            let h = b[32 + ib * 2 + l / 2] as usize;
            let q = b[ib * 4 + l] as usize | ((h << (if l % 2 == 0 { 8 } else { 4 })) & 0x700);
            let delta = if h & (if l % 2 == 0 { 8 } else { 128 }) != 0 {
                -0.125
            } else {
                0.125
            };
            let grid = IQ1S_GRID[q].to_le_bytes();
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid[j] as i8 as f32 + delta);
            }
        }
    }
}
