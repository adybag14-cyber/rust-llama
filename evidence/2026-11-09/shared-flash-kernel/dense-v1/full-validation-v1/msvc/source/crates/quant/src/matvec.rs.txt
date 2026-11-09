//! Scalar CPU dot arithmetic and activation conversion from the frozen GGML CPU traits.
//! Integer products are accumulated before block scales are applied; decoding
//! both operands to F32 is not numerically equivalent to quantized inference.
use crate::{DType, QuantError, dequantize, half, quantize, u32le};

/// Activation dtype selected by the frozen CPU vec_dot trait for implemented paths.
pub fn matvec_activation_dtype(weights: DType) -> Result<DType, QuantError> {
    use DType::*;
    match weights {
        Q1_0 | Q2_0 | Q4_0 | Q5_0 | Q8_0 | MXFP4 | NVFP4 | IQ4_NL => Ok(Q8_0),
        Q4_1 | Q5_1 => Ok(Q8_1),
        Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ1_S
        | IQ1_M | IQ4_XS | TQ1_0 | TQ2_0 => Ok(Q8_K),
        F32 | F16 | F8_E4M3 => Ok(F32),
        BF16 => Ok(BF16),
        _ => Err(QuantError::UnsupportedMatVec(weights)),
    }
}

/// Row-major GGML matrix-vector multiplication with its native activation dtype.
/// Supports every weight dtype with a CPU vec_dot in the frozen source: 25
/// packed formats and scalar F32/F16/BF16/F8_E4M3.
/// This is the scalar arithmetic path, not a claim of ISA-specific accumulation.
pub fn matvec(
    weights_type: DType,
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, QuantError> {
    let activation_type = matvec_activation_dtype(weights_type)?;
    if rows == 0 || cols == 0 {
        return Err(QuantError::InvalidMatrixShape { rows, cols });
    }
    if input.len() != cols {
        return Err(QuantError::InvalidVectorLength {
            expected: cols,
            actual: input.len(),
        });
    }
    let row_bytes = weights_type.row_bytes(cols)?;
    let expected = row_bytes
        .checked_mul(rows)
        .ok_or(QuantError::SizeOverflow)?;
    if weights.len() != expected {
        return Err(QuantError::InvalidLength {
            expected,
            actual: weights.len(),
        });
    }
    for (index, &v) in input.iter().enumerate() {
        if !v.is_finite() {
            return Err(QuantError::NonFiniteInput { index });
        }
    }
    let activation = quantize(activation_type, input)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| QuantError::AllocationFailed)?;
    if weights_type.block_size() > 1 {
        for row in weights.chunks_exact(row_bytes) {
            output.push(dot_packed_unchecked(weights_type, row, &activation));
        }
    } else {
        let y = dequantize(activation_type, &activation, cols)?;
        for row in weights.chunks_exact(row_bytes) {
            let x = dequantize(weights_type, row, cols)?;
            if weights_type == DType::F8_E4M3 {
                // The F8 scalar kernel specifically uses F32 accumulation.
                let mut sum = 0.0_f32;
                for (&a, &b) in x.iter().zip(&y) {
                    sum += a * b;
                }
                output.push(sum);
            } else {
                // ggml_float sum in the other scalar floating-point vector dots.
                let mut sum = 0.0_f64;
                for (&a, &b) in x.iter().zip(&y) {
                    sum += (a * b) as f64;
                }
                output.push(sum as f32);
            }
        }
    }
    Ok(output)
}

/// Dot two already packed rows. The right row must use matvec_activation_dtype().
pub fn dot_packed(
    weights_type: DType,
    weights: &[u8],
    activation: &[u8],
    elements: usize,
) -> Result<f32, QuantError> {
    let activation_type = matvec_activation_dtype(weights_type)?;
    if weights_type.block_size() == 1 {
        return Err(QuantError::UnsupportedMatVec(weights_type));
    }
    crate::validate(weights_type, weights.len(), elements)?;
    crate::validate(activation_type, activation.len(), elements)?;
    Ok(dot_packed_unchecked(weights_type, weights, activation))
}

fn dot_packed_unchecked(t: DType, x: &[u8], y: &[u8]) -> f32 {
    if matches!(
        t,
        DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0
    ) {
        dot_legacy(t, x, y)
    } else {
        crate::matvec_extra::dot(t, x, y)
    }
}
fn dot_legacy(t: DType, x: &[u8], y: &[u8]) -> f32 {
    let affine = matches!(t, DType::Q4_1 | DType::Q5_1);
    let five = matches!(t, DType::Q5_0 | DType::Q5_1);
    let ysize = if affine { 36 } else { 34 };
    let mut sum = 0.0_f32;
    for (xb, yb) in x.chunks_exact(t.type_size()).zip(y.chunks_exact(ysize)) {
        let dx = half(xb, 0);
        let dy = half(yb, 0);
        let mut integer_sum = 0i32;
        if t == DType::Q8_0 {
            for j in 0..32 {
                integer_sum += xb[2 + j] as i8 as i32 * yb[2 + j] as i8 as i32;
            }
            sum += integer_sum as f32 * (dx * dy);
        } else {
            let xp = if affine { 4 } else { 2 };
            let yp = if affine { 4 } else { 2 };
            let high = if five { u32le(xb, xp) } else { 0 };
            let qp = xp + if five { 4 } else { 0 };
            for j in 0..16 {
                let q = xb[qp + j];
                let mut q0 = (q & 15) as i32;
                let mut q1 = (q >> 4) as i32;
                if five {
                    q0 |= (((high >> j) & 1) << 4) as i32;
                    q1 |= (((high >> (j + 16)) & 1) << 4) as i32;
                }
                if !affine {
                    q0 -= if five { 16 } else { 8 };
                    q1 -= if five { 16 } else { 8 };
                }
                integer_sum += q0 * yb[yp + j] as i8 as i32;
                integer_sum += q1 * yb[yp + j + 16] as i8 as i32;
            }
            if affine {
                sum += (dx * dy) * integer_sum as f32 + half(xb, 2) * half(yb, 2);
            } else if t == DType::Q4_0 {
                // Source Q4_0 generic path uses this specific multiplication order.
                sum += integer_sum as f32 * dx * dy;
            } else {
                sum += (dx * dy) * integer_sum as f32;
            }
        }
    }
    sum
}
