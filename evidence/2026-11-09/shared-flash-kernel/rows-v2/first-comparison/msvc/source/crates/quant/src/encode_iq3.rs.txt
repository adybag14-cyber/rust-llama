//! IQ3 lattice search, scalar source arithmetic and exact block packing.
//! Copyright (c) 2023-2026 The ggml authors; MIT license.
use crate::{
    DType, QuantError,
    encode_k::{finite, nearest, put_half},
};
use std::sync::OnceLock;

struct Grid {
    levels: Vec<[i32; 4]>,
    direct: [i16; 4096],
    neighbours: Vec<Vec<u16>>,
}
impl Grid {
    fn build(indices: &[u16], shells: usize) -> Result<Self, QuantError> {
        let mut levels = Vec::new();
        levels
            .try_reserve_exact(indices.len())
            .map_err(|_| QuantError::AllocationFailed)?;
        let mut direct = [-1_i16; 4096];
        for (index, &code) in indices.iter().enumerate() {
            levels.push(std::array::from_fn(|i| ((code >> (3 * i)) & 7) as i32));
            direct[code as usize] = index as i16;
        }
        let mut neighbours = Vec::new();
        neighbours
            .try_reserve_exact(4096)
            .map_err(|_| QuantError::AllocationFailed)?;
        let mut distances = Vec::new();
        distances
            .try_reserve_exact(indices.len())
            .map_err(|_| QuantError::AllocationFailed)?;
        for (code, &entry) in direct.iter().enumerate() {
            if entry >= 0 {
                neighbours.push(Vec::new());
                continue;
            }
            let position: [i32; 4] = std::array::from_fn(|i| ((code >> (3 * i)) & 7) as i32);
            distances.clear();
            for (index, point) in levels.iter().enumerate() {
                let mut distance = 0;
                for i in 0..4 {
                    let d = 2 * (point[i] - position[i]);
                    distance += d * d;
                }
                distances.push((distance, index as u16));
            }
            distances.sort_unstable();
            let mut last = distances[0].0;
            let mut have = 1;
            let mut count = 0;
            for &(distance, _) in &distances {
                if distance > last {
                    if have == shells {
                        break;
                    }
                    last = distance;
                    have += 1;
                }
                count += 1;
            }
            let mut list = Vec::new();
            list.try_reserve_exact(count)
                .map_err(|_| QuantError::AllocationFailed)?;
            for &(_, index) in &distances[..count] {
                list.push(index);
            }
            neighbours.push(list);
        }
        Ok(Self {
            levels,
            direct,
            neighbours,
        })
    }
    fn index(&self, code: usize) -> Option<usize> {
        let index = self.direct[code];
        (index >= 0).then_some(index as usize)
    }
    fn nearest_neighbour(
        &self,
        code: usize,
        x: &[f32],
        weight: &[f32],
        scale: f32,
        levels: &mut [i32],
    ) -> Result<usize, ()> {
        let mut best = f32::MAX;
        let mut selected = None;
        for &index in &self.neighbours[code] {
            let point = &self.levels[index as usize];
            let mut error = 0.0;
            for i in 0..4 {
                let q = (2 * point[i] + 1) as f32;
                let delta = scale * q - x[i];
                error += weight[i] * delta * delta;
            }
            if error < best {
                best = error;
                selected = Some(index as usize);
            }
        }
        let index = selected.ok_or(())?;
        levels.copy_from_slice(&self.levels[index]);
        Ok(index)
    }
}
fn grid(dtype: DType) -> Result<&'static Grid, QuantError> {
    static XXS: OnceLock<Result<Grid, QuantError>> = OnceLock::new();
    static S: OnceLock<Result<Grid, QuantError>> = OnceLock::new();
    let result = if dtype == DType::IQ3_XXS {
        XXS.get_or_init(|| Grid::build(&KGRID_256, 2))
    } else {
        S.get_or_init(|| Grid::build(&KGRID_512, 3))
    };
    result.as_ref().map_err(Clone::clone)
}
fn key(levels: &[i32]) -> usize {
    let mut code = 0;
    for (i, &level) in levels.iter().take(4).enumerate() {
        code |= (level as usize) << (3 * i);
    }
    code
}
fn quantized_key(x: &[f32], inverse: f32, levels: &mut [i32]) -> Result<usize, ()> {
    for i in 0..4 {
        levels[i] = nearest(0.5 * (inverse * x[i] - 1.0))?.clamp(0, 7);
    }
    Ok(key(levels))
}

pub(crate) fn encode(
    dtype: DType,
    values: &[f32],
    importance: Option<&[f32]>,
) -> Result<Vec<u8>, QuantError> {
    let bytes = dtype.row_bytes(values.len())?;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantError::NonFiniteInput { index });
        }
    }
    let grid = grid(dtype)?;
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
    grid: &Grid,
    x: &[f32],
    importance: Option<&[f32]>,
    out: &mut [u8],
) -> Result<(), ()> {
    let xxs = dtype == DType::IQ3_XXS;
    let sum_x2 = x.iter().fold(0.0, |sum, &value| sum + value * value);
    let sigma2 = finite(2.0 * sum_x2 / 256.0)?;
    let mut scales = [0.0; 8];
    let mut max_scale = 0.0_f32;
    let mut quant_cursor = 2;
    let mut sign_cursor = 74;
    for sub in 0..8 {
        let values = &x[sub * 32..(sub + 1) * 32];
        let mut weights = [0.0; 32];
        let mut auxiliary_weights = [0.0; 32];
        for j in 0..32 {
            weights[j] = finite(if let Some(importance) = importance {
                importance[sub * 32 + j] * (sigma2 + values[j] * values[j]).sqrt()
            } else {
                values[j] * values[j]
            })?;
            auxiliary_weights[j] = weights[j].sqrt();
        }
        let mut magnitudes = [0.0; 32];
        let mut signs = [0_u8; 4];
        for (group, signs) in signs.iter_mut().enumerate() {
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
            if xxs && flips % 2 != 0 {
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
            *signs = if xxs { sign & 127 } else { sign };
        }
        let mut maximum = magnitudes[0];
        for &value in &magnitudes[1..] {
            if value > maximum {
                maximum = value;
            }
        }
        if (xxs && maximum < 1e-8) || (!xxs && maximum == 0.0) {
            // IQ3_S advances its source qs/signs pointers only for nonzero
            // subblocks; preserve that source behavior, including mixed zeros.
            continue;
        }
        let mut levels = [0; 32];
        let mut candidate = [0; 32];
        let mut on_grid = [xxs; 8];
        let mut best = 0.0;
        let mut scale = maximum / 15.0;
        let range = if xxs { 15 } else { 9 };
        for step in -range..=range {
            let inverse = (15.0 + step as f32 * 0.2) / maximum;
            let this_scale = 1.0 / inverse;
            let mut candidate_on_grid = [true; 8];
            for (group, candidate_on_grid) in candidate_on_grid.iter_mut().enumerate() {
                let part = group * 4..(group + 1) * 4;
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
            for j in 0..32 {
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
        if on_grid.iter().any(|&on| !on) && scale > 0.0 {
            let inverse = 1.0 / scale;
            for (group, &on_grid) in on_grid.iter().enumerate() {
                if xxs && on_grid {
                    continue;
                }
                let part = group * 4..(group + 1) * 4;
                let mut next_levels = [0; 4];
                let code = quantized_key(&magnitudes[part.clone()], inverse, &mut next_levels)?;
                let index = if let Some(index) = grid.index(code) {
                    index
                } else {
                    grid.nearest_neighbour(
                        code,
                        &magnitudes[part.clone()],
                        &auxiliary_weights[part.clone()],
                        scale,
                        &mut levels[part.clone()],
                    )?
                };
                levels[part].copy_from_slice(&grid.levels[index]);
            }
            let mut sum_qx = 0.0;
            let mut sum_q2 = 0.0;
            for j in 0..32 {
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
            for sign in &mut signs {
                *sign = if xxs { !*sign & 127 } else { !*sign };
            }
        }
        finite(scale)?;
        for group in 0..8 {
            let index = grid
                .index(key(&levels[group * 4..(group + 1) * 4]))
                .ok_or(())?;
            if xxs {
                out[2 + sub * 8 + group] = index as u8;
            } else {
                out[quant_cursor + group] = index as u8;
                out[66 + sub] |= ((index >> 8) as u8) << group;
            }
        }
        if xxs {
            let combined = signs[0] as u32
                | ((signs[1] as u32) << 7)
                | ((signs[2] as u32) << 14)
                | ((signs[3] as u32) << 21);
            out[66 + sub * 4..70 + sub * 4].copy_from_slice(&combined.to_le_bytes());
        } else {
            quant_cursor += 8;
            out[sign_cursor..sign_cursor + 4].copy_from_slice(&signs);
            sign_cursor += 4;
        }
        scales[sub] = scale;
        if scale > max_scale {
            max_scale = scale;
        }
    }
    if max_scale == 0.0 {
        if xxs {
            out.fill(0);
        }
        return Ok(());
    }
    let d = max_scale / 31.0;
    put_half(out, 0, d * if xxs { 1.0125 } else { 1.033 })?;
    let inverse = 1.0 / d;
    for sub in 0..8 {
        let level = nearest(0.5 * (inverse * scales[sub] - 1.0))?.clamp(0, 15) as u8;
        if xxs {
            out[69 + sub * 4] |= level << 4;
        } else if sub % 2 == 0 {
            out[106 + sub / 2] = level;
        } else {
            out[106 + sub / 2] |= level << 4;
        }
    }
    Ok(())
}

// Original IQ3 lattice index data from frozen ggml-quants.c; MIT.
#[rustfmt::skip]
const KGRID_256: [u16; 256] = [
    0, 2, 4, 9, 11, 15, 16, 18, 25, 34, 59, 61, 65, 67, 72, 74,
    81, 85, 88, 90, 97, 108, 120, 128, 130, 132, 137, 144, 146, 153, 155, 159,
    169, 175, 189, 193, 199, 200, 202, 213, 248, 267, 287, 292, 303, 315, 317, 321,
    327, 346, 362, 413, 436, 456, 460, 462, 483, 497, 513, 515, 520, 522, 529, 531,
    536, 538, 540, 551, 552, 576, 578, 585, 592, 594, 641, 643, 648, 650, 657, 664,
    698, 704, 706, 720, 729, 742, 758, 769, 773, 808, 848, 852, 870, 889, 901, 978,
    992, 1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105,
    1112, 1139, 1143, 1144, 1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228,
    1272, 1276, 1309, 1323, 1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553,
    1555, 1576, 1589, 1594, 1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722,
    1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063,
    2077, 2079, 2091, 2095, 2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389,
    2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746,
    2754, 2795, 2824, 2835, 2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153,
    3185, 3215, 3252, 3288, 3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610,
    3626, 3670, 3680, 3722, 3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
];
#[rustfmt::skip]
const KGRID_512: [u16; 512] = [
    0, 1, 2, 5, 7, 8, 9, 10, 12, 14, 16, 17, 21, 27, 32, 34,
    37, 39, 41, 43, 48, 50, 57, 60, 63, 64, 65, 66, 68, 72, 73, 77,
    80, 83, 87, 89, 93, 100, 113, 117, 122, 128, 129, 133, 135, 136, 139, 142,
    145, 149, 152, 156, 162, 165, 167, 169, 171, 184, 187, 195, 201, 205, 208, 210,
    217, 219, 222, 228, 232, 234, 247, 249, 253, 256, 267, 271, 273, 276, 282, 288,
    291, 297, 312, 322, 324, 336, 338, 342, 347, 353, 357, 359, 374, 379, 390, 393,
    395, 409, 426, 441, 448, 450, 452, 464, 466, 470, 475, 488, 492, 512, 513, 514,
    516, 520, 521, 523, 525, 527, 528, 530, 537, 540, 542, 556, 558, 561, 570, 576,
    577, 579, 582, 584, 588, 593, 600, 603, 609, 616, 618, 632, 638, 640, 650, 653,
    655, 656, 660, 666, 672, 675, 685, 688, 698, 705, 708, 711, 712, 715, 721, 727,
    728, 732, 737, 754, 760, 771, 773, 778, 780, 793, 795, 802, 806, 808, 812, 833,
    840, 843, 849, 856, 858, 873, 912, 916, 919, 932, 934, 961, 963, 968, 970, 977,
    989, 993, 1010, 1016, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047,
    1048, 1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103,
    1106, 1109, 1113, 1116, 1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199,
    1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296,
    1299, 1306, 1309, 1313, 1338, 1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415,
    1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561,
    1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648,
    1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761,
    1763, 1767, 1773, 1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877,
    1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068,
    2071, 2074, 2081, 2088, 2104, 2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177,
    2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269,
    2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520,
    2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634,
    2641, 2650, 2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805,
    2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083,
    3085, 3097, 3099, 3120, 3136, 3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276,
    3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591,
    3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729,
    3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
];
