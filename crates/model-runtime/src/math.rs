//! Scalar runtime math and checked native quantized matrix-vector execution.
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
    packed: Option<(llama_quant::DType, Vec<u8>)>,
}
impl Matrix {
    /// Row-major weights: rows are output units, columns input units. This is
    /// the dense GGML `{input, output}` memory order, with `ne[0]` contiguous.
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(values.len()) {
            return Err(Error::Shape(format!(
                "{rows} x {cols} matrix has {} values",
                values.len()
            )));
        }
        finite(&values, "matrix weights")?;
        Ok(Self {
            rows,
            cols,
            values,
            packed: None,
        })
    }
    /// Keep the original GGML representation for native activation quantization.
    /// Unsupported native dot formats are rejected at load rather than falling
    /// back to decoded-F32 multiplication.
    pub fn from_quantized(
        rows: usize,
        cols: usize,
        dtype: llama_quant::DType,
        bytes: Vec<u8>,
    ) -> Result<Self> {
        llama_quant::matvec_activation_dtype(dtype)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        if rows == 0 || cols == 0 {
            return Err(Error::Shape(
                "quantized matrix dimensions must be positive".into(),
            ));
        }
        let row_bytes = dtype
            .row_bytes(cols)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let expected = row_bytes
            .checked_mul(rows)
            .ok_or_else(|| Error::Shape("quantized matrix size overflow".into()))?;
        if bytes.len() != expected {
            return Err(Error::Shape(format!(
                "quantized matrix expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Shape("quantized matrix element overflow".into()))?;
        let values = llama_quant::dequantize(dtype, &bytes, elements)
            .map_err(|e| Error::Shape(e.to_string()))?;
        let mut matrix = Self::new(rows, cols, values)?;
        if dtype != llama_quant::DType::F32 {
            matrix.packed = Some((dtype, bytes));
        }
        Ok(matrix)
    }
    /// Included by the model fingerprint because activation conversion changes
    /// arithmetic even when two representations decode to the same F32 values.
    pub(crate) fn storage_identity(&self) -> Option<(u32, &[u8])> {
        self.packed
            .as_ref()
            .map(|(dtype, bytes)| (dtype.id(), bytes.as_slice()))
    }
    pub fn storage_dtype(&self) -> llama_quant::DType {
        self.packed
            .as_ref()
            .map_or(llama_quant::DType::F32, |(dtype, _)| *dtype)
    }
    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        let size = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Shape("matrix size overflow".into()))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(size)
            .map_err(|_| Error::Shape("matrix allocation too large".into()))?;
        values.resize(size, 0.0);
        Self::new(rows, cols, values)
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn row(&self, row: usize) -> Result<&[f32]> {
        if row >= self.rows {
            return Err(Error::Shape(format!(
                "row {row} outside {} rows",
                self.rows
            )));
        }
        Ok(&self.values[row * self.cols..(row + 1) * self.cols])
    }
    pub fn multiply(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.cols {
            return Err(Error::Shape(format!(
                "matrix expected {} inputs, got {}",
                self.cols,
                input.len()
            )));
        }
        finite(input, "matrix input")?;
        let output: Vec<f32> = if let Some((dtype, bytes)) = &self.packed {
            llama_quant::matvec(*dtype, bytes, self.rows, self.cols, input)
                .map_err(|e| Error::Shape(e.to_string()))?
        } else {
            self.values
                .chunks_exact(self.cols)
                .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
                .collect()
        };
        finite(&output, "matrix output")?;
        Ok(output)
    }
    pub(crate) fn shape(&self, rows: usize, cols: usize, name: &str) -> Result<()> {
        if self.rows == rows && self.cols == cols {
            Ok(())
        } else {
            Err(Error::Shape(format!(
                "{name}: expected {rows} x {cols}, got {} x {}",
                self.rows, self.cols
            )))
        }
    }
}
pub(crate) fn finite(values: &[f32], name: &str) -> Result<()> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite(name.into()))
    }
}
pub(crate) fn vector(values: &[f32], len: usize, name: &str) -> Result<()> {
    if values.len() != len {
        return Err(Error::Shape(format!(
            "{name}: expected {len} values, got {}",
            values.len()
        )));
    }
    finite(values, name)
}
pub(crate) fn optional_bias(bias: &Option<Vec<f32>>, len: usize, name: &str) -> Result<()> {
    if let Some(bias) = bias {
        vector(bias, len, name)?;
    }
    Ok(())
}
pub(crate) fn add(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (a, b) in a.iter_mut().zip(b) {
        *a += b;
    }
}
pub(crate) fn bias(a: &mut [f32], bias: &Option<Vec<f32>>) {
    if let Some(b) = bias {
        add(a, b);
    }
}
pub(crate) fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    // GGML scalar norm accumulates squares in ggml_float (double) and rounds
    // the mean to float before adding epsilon.
    let sum: f64 = input.iter().map(|x| (*x * *x) as f64).sum();
    let scale = 1.0 / ((sum / input.len() as f64) as f32 + epsilon).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::NonFinite("RMS normalization scale".into()));
    }
    let output: Vec<f32> = input
        .iter()
        .zip(weight)
        .map(|(x, w)| x * scale * w)
        .collect();
    finite(&output, "RMS normalization output")?;
    Ok(output)
}
pub(crate) fn layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    let mean = input.iter().map(|x| *x as f64).sum::<f64>() as f32 / input.len() as f32;
    let centered: Vec<f32> = input.iter().map(|x| x - mean).collect();
    let variance =
        (centered.iter().map(|x| (*x * *x) as f64).sum::<f64>() / input.len() as f64) as f32;
    let scale = (variance + epsilon).sqrt().recip();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::NonFinite("layer normalization scale".into()));
    }
    let output: Vec<f32> = centered
        .iter()
        .zip(weight)
        .zip(bias)
        .map(|((x, w), b)| x * scale * w + b)
        .collect();
    finite(&output, "layer normalization output")?;
    Ok(output)
}
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
pub(crate) fn gelu(x: f32) -> f32 {
    // The frozen CPU vec.h unconditionally enables GGML_GELU_FP16. Its F32
    // operator uses an F16-indexed / F16-valued table between -10 and 10.
    // Recompute the same table entry in Rust rather than keeping a foreign
    // implementation or silently substituting higher-precision GELU.
    if x <= -10.0 {
        return 0.0;
    }
    if x >= 10.0 {
        return x;
    }
    let x = round_f16(x);
    round_f16(0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh()))
}
fn round_f16(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude < 0x3880_0000 {
        return ((x as f64 * 16_777_216.0).round_ties_even() / 16_777_216.0) as f32;
    }
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    let rounded = (magnitude + 0xfff + ((magnitude >> 13) & 1)) & !0x1fff;
    if rounded >= 0x4780_0000 {
        f32::from_bits((bits & 0x8000_0000) | 0x7f80_0000)
    } else {
        f32::from_bits((bits & 0x8000_0000) | rounded)
    }
}
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
pub(crate) fn softmax(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::Shape("empty attention row".into()));
    }
    finite(values, "attention scores")?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for x in values.iter_mut() {
        *x = (*x - max).exp();
        sum += *x as f64;
    }
    let inverse = (1.0 / sum) as f32;
    for x in values {
        *x *= inverse;
    }
    Ok(())
}

/// Standard adjacent-pair Llama RoPE, not NeoX, MRoPE, YaRN, LongRoPE or scaled
/// frequencies. The model constructor deliberately exposes only this mode.
pub(crate) fn rope(values: &mut [f32], head_dim: usize, position: usize, base: f32) {
    let theta_scale = base.powf(-2.0 / head_dim as f32);
    for head in values.chunks_exact_mut(head_dim) {
        let mut theta = position as f32;
        for pair in head.as_chunks_mut::<2>().0 {
            let (sin, cos) = theta.sin_cos();
            let (x, y) = (pair[0], pair[1]);
            pair[0] = x * cos - y * sin;
            pair[1] = x * sin + y * cos;
            theta *= theta_scale;
        }
    }
}

/// Multi-head attention over dense flattened K/V rows, including GQA.
pub(crate) fn attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let kv_width = kv_heads * head_dim;
    let count = keys.len() / kv_width;
    debug_assert_eq!(keys.len(), values.len());
    debug_assert_eq!(query.len(), heads * head_dim);
    let mut output = vec![0.0; query.len()];
    for head in 0..heads {
        let kv_head = head / (heads / kv_heads);
        let q = &query[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::with_capacity(count);
        for token in 0..count {
            let offset = token * kv_width + kv_head * head_dim;
            let dot: f32 = q
                .iter()
                .zip(&keys[offset..offset + head_dim])
                .map(|(a, b)| a * b)
                .sum();
            scores.push(dot * scale);
        }
        softmax(&mut scores)?;
        for (token, probability) in scores.iter().enumerate() {
            let offset = token * kv_width + kv_head * head_dim;
            for d in 0..head_dim {
                output[head * head_dim + d] += probability * values[offset + d];
            }
        }
    }
    finite(&output, "attention output")?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_rounding_handles_normal_and_subnormal_ties_even() {
        assert_eq!(round_f16(1.0 + 1.0 / 2048.0), 1.0);
        assert_eq!(round_f16(1.0 + 3.0 / 2048.0), 1.0 + 2.0 / 1024.0);
        assert_eq!(round_f16(1.0 / 33_554_432.0), 0.0);
        assert_eq!(round_f16(3.0 / 33_554_432.0), 2.0 / 16_777_216.0);
        assert_eq!(
            round_f16(-1.0 / 33_554_432.0).to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(gelu(-10.0), 0.0);
        assert_eq!(gelu(10.0), 10.0);
    }
    #[test]
    fn overflowing_normalization_cannot_silently_create_zero_embeddings() {
        assert!(rms_norm(&[f32::MAX; 4], &[1.0; 4], 1e-5).is_err());
        assert!(layer_norm(&[f32::MAX; 4], &[1.0; 4], &[0.0; 4], 1e-5).is_err());
    }
}

#[cfg(test)]
mod packed_matrix_tests {
    use super::*;
    use llama_quant::DType;
    #[test]
    fn packed_matrix_preserves_storage_and_native_activation_quantization() {
        let weights = (0..64)
            .map(|i| (i % 7) as f32 * 0.7 - 2.0)
            .collect::<Vec<_>>();
        let input = (0..32).map(|i| (i as f32 * 0.31).sin()).collect::<Vec<_>>();
        let bytes = llama_quant::quantize(DType::Q4_0, &weights).unwrap();
        let decoded = llama_quant::dequantize(DType::Q4_0, &bytes, 64).unwrap();
        let packed = Matrix::from_quantized(2, 32, DType::Q4_0, bytes.clone()).unwrap();
        let dense = Matrix::new(2, 32, decoded.clone()).unwrap();
        assert_eq!(packed.values(), decoded);
        assert_eq!(packed.row(1).unwrap(), &decoded[32..]);
        assert_eq!(packed.storage_identity(), Some((2, bytes.as_slice())));
        assert_eq!(dense.storage_identity(), None);
        assert_eq!(
            packed.multiply(&input).unwrap(),
            llama_quant::matvec(DType::Q4_0, &bytes, 2, 32, &input).unwrap()
        );
        assert!(
            (packed.multiply(&input).unwrap()[0] - dense.multiply(&input).unwrap()[0]).abs() > 1e-4
        );
        assert_eq!(packed.clone(), packed);
    }
    #[test]
    fn unsupported_dot_types_are_rejected_at_load() {
        assert!(matches!(
            Matrix::from_quantized(1, 256, DType::Q4_K, vec![0; 144]),
            Err(Error::Unsupported(_))
        ));
        assert!(Matrix::from_quantized(1, 31, DType::Q4_0, vec![0; 18]).is_err());
        assert!(Matrix::from_quantized(1, 32, DType::Q4_0, vec![0; 17]).is_err());
    }
}
