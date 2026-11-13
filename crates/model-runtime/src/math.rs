//! Source-profile row math and checked native quantized matrix-vector execution.

pub mod native_rows;

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
    pub fn owned_bytes(&self) -> Result<usize> {
        self.values
            .capacity()
            .checked_mul(4)
            .and_then(|bytes| {
                bytes.checked_add(
                    self.packed
                        .as_ref()
                        .map_or(0, |(_, bytes)| bytes.capacity()),
                )
            })
            .ok_or_else(|| Error::Shape("matrix storage accounting overflow".into()))
    }
    pub fn try_clone(&self) -> Result<Self> {
        fn copy<T: Copy>(source: &[T]) -> Result<Vec<T>> {
            let mut out = Vec::new();
            out.try_reserve_exact(source.len())
                .map_err(|_| Error::Shape("matrix copy allocation".into()))?;
            out.extend_from_slice(source);
            Ok(out)
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            values: copy(&self.values)?,
            packed: self
                .packed
                .as_ref()
                .map(|(dtype, bytes)| Ok((*dtype, copy(bytes)?)))
                .transpose()?,
        })
    }
    /// Original immutable packed storage for device upload. F32 matrices expose
    /// their scalar storage through values(); no re-quantization is performed.
    pub fn packed_bytes(&self) -> Option<(llama_quant::DType, &[u8])> {
        self.packed
            .as_ref()
            .map(|(dtype, bytes)| (*dtype, bytes.as_slice()))
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
                .map(|row| {
                    row.iter()
                        .zip(input)
                        .map(|(a, b)| f64::from(a * b))
                        .sum::<f64>() as f32
                })
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
    layer_norm_optional_bias(input, weight, Some(bias), epsilon)
}
/// Affine NORM without synthesizing an ADD when the source omits the bias.
/// Adding positive zero would erase a negative-zero result.
pub(crate) fn layer_norm_optional_bias(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>> {
    if input.is_empty()
        || input.len() != weight.len()
        || bias.is_some_and(|bias| input.len() != bias.len())
    {
        return Err(Error::Shape("invalid layer normalization row".into()));
    }
    let mean = input.iter().map(|x| *x as f64).sum::<f64>() as f32 / input.len() as f32;
    let mut centered = Vec::new();
    centered
        .try_reserve_exact(input.len())
        .map_err(|_| Error::InvalidState("layer normalization allocation failed".into()))?;
    centered.extend_from_slice(input);
    // Source cvar selects its own SSE2 grouped F32 square sum on GNU x64.
    // Return is F64; the NORM graph casts variance once before adding epsilon.
    let variance = native_rows::center_f32_and_variance(&mut centered, mean) as f32;
    let scale = (variance + epsilon).sqrt().recip();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::NonFinite("layer normalization scale".into()));
    }
    for (value, weight) in centered.iter_mut().zip(weight) {
        *value = *value * scale * weight;
    }
    if let Some(bias) = bias {
        for (value, bias) in centered.iter_mut().zip(bias) {
            *value += bias;
        }
    }
    finite(&centered, "layer normalization output")?;
    Ok(centered)
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
    // The frozen ggml_compute_softplus_f32 rounds 1 + exp(x) to F32 before log.
    // log1p(exp(x)) changes small deltas and therefore persistent SSM state.
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
pub(crate) fn softmax(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::Shape("empty attention row".into()));
    }
    if values
        .iter()
        .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(Error::NonFinite("attention scores".into()));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(Error::NonFinite(
            "attention row has no unmasked values".into(),
        ));
    }
    let sum = native_rows::softmax_exp_sum_f32_in_place(values, max);
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::NonFinite("attention normalization sum".into()));
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

pub(crate) fn rope_neox(values: &mut [f32], head_dim: usize, position: usize, base: f32) {
    let theta_scale = base.powf(-2.0 / head_dim as f32);
    for head in values.chunks_exact_mut(head_dim) {
        let (left, right) = head.split_at_mut(head_dim / 2);
        let mut theta = position as f32;
        for (left, right) in left.iter_mut().zip(right) {
            let (sin, cos) = theta.sin_cos();
            let (x, y) = (*left, *right);
            *left = x * cos - y * sin;
            *right = x * sin + y * cos;
            theta *= theta_scale;
        }
    }
}

/// Columns occupied by gathered K/V rows inside the original softmax row.
///
/// GGML keeps empty, unrelated-sequence, future and padded columns in place.
/// Removing them changes SSE2 four-lane grouping and the normalization sum.
/// An absent index map denotes contiguous columns starting at zero. The caller
/// supplies the original row width; it is never inferred from visible rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionLayout<'a> {
    pub row_len: usize,
    pub columns: Option<&'a [usize]>,
}
impl AttentionLayout<'_> {
    /// Dense non-flash graph rows use a multiple of 256 columns, including
    /// BERT/no-cache attention. Zero and overflowing lengths are rejected.
    pub fn contiguous_padded(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::Shape("empty attention row".into()));
        }
        let row_len = count
            .checked_add(255)
            .map(|n| n / 256 * 256)
            .ok_or_else(|| Error::Shape("attention padding overflow".into()))?;
        Ok(Self {
            row_len,
            columns: None,
        })
    }
    fn validate(self, count: usize) -> Result<()> {
        if self.row_len == 0 || count == 0 || count > self.row_len {
            return Err(Error::Shape("invalid attention row width".into()));
        }
        if let Some(columns) = self.columns
            && (columns.len() != count
                || columns.last().is_some_and(|&column| column >= self.row_len)
                || columns.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(Error::Shape("invalid attention physical columns".into()));
        }
        Ok(())
    }
    fn column(self, row: usize) -> usize {
        self.columns.map_or(row, |columns| columns[row])
    }
}

/// Attention with dense contiguous source columns, including GQA. Strict
/// sequential contexts and no-cache encoders use this path. NativeContext
/// supplies its physical hole/stream layout through attention_with_layout.
pub(crate) fn attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let width = kv_heads
        .checked_mul(head_dim)
        .filter(|&width| width != 0)
        .ok_or_else(|| Error::Shape("invalid attention KV width".into()))?;
    attention_with_layout(
        query,
        keys,
        values,
        heads,
        kv_heads,
        head_dim,
        scale,
        AttentionLayout::contiguous_padded(keys.len() / width)?,
    )
}

/// Execute the original full softmax row while storing only visible K/V rows.
/// The grouped shape arguments retain the source head/GQA mapping explicitly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_with_layout(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    layout: AttentionLayout<'_>,
) -> Result<Vec<f32>> {
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .filter(|&width| width != 0)
        .ok_or_else(|| Error::Shape("invalid attention KV width".into()))?;
    if heads == 0
        || !heads.is_multiple_of(kv_heads)
        || heads.checked_mul(head_dim) != Some(query.len())
        || keys.len() != values.len()
        || !keys.len().is_multiple_of(kv_width)
        || !scale.is_finite()
    {
        return Err(Error::Shape("invalid attention tensor shape".into()));
    }
    let count = keys.len() / kv_width;
    layout.validate(count)?;
    // Frozen scalar ggml_vec_dot_f32 rounds each product to F32, then sums
    // in ggml_float (F64). This applies to QK and probability-times-V.
    let mut output = Vec::new();
    output
        .try_reserve_exact(query.len())
        .map_err(|_| Error::InvalidState("attention output allocation failed".into()))?;
    output.resize(query.len(), 0.0f64);
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(layout.row_len)
        .map_err(|_| Error::InvalidState("attention row allocation failed".into()))?;
    scores.resize(layout.row_len, f32::NEG_INFINITY);
    for head in 0..heads {
        scores.fill(f32::NEG_INFINITY);
        let kv_head = head / (heads / kv_heads);
        let q = &query[head * head_dim..(head + 1) * head_dim];
        for token in 0..count {
            let offset = token * kv_width + kv_head * head_dim;
            let dot = q
                .iter()
                .zip(&keys[offset..offset + head_dim])
                .map(|(a, b)| f64::from(a * b))
                .sum::<f64>() as f32;
            let score = dot * scale;
            if !score.is_finite() {
                return Err(Error::NonFinite("attention score".into()));
            }
            scores[layout.column(token)] = score;
        }
        softmax(&mut scores)?;
        for token in 0..count {
            let probability = scores[layout.column(token)];
            let offset = token * kv_width + kv_head * head_dim;
            for d in 0..head_dim {
                output[head * head_dim + d] += f64::from(probability * values[offset + d]);
            }
        }
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(output.len())
        .map_err(|_| Error::InvalidState("attention result allocation failed".into()))?;
    result.extend(output.into_iter().map(|x| x as f32));
    finite(&result, "attention output")?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_masks_and_physical_layout_bounds_fail_safely() {
        let mut row = [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        softmax(&mut row).unwrap();
        assert_eq!(row, [1.0, 0.0, 0.0, 0.0]);
        for mut invalid in [
            [f32::NEG_INFINITY; 4],
            [0.0, f32::INFINITY, 0.0, 0.0],
            [0.0, f32::NAN, 0.0, 0.0],
        ] {
            let before = invalid.map(f32::to_bits);
            assert!(softmax(&mut invalid).is_err());
            assert_eq!(before, invalid.map(f32::to_bits));
        }
        let query = [0.5, 1.0];
        let keys = [1.0, 2.0, 3.0, 4.0];
        for columns in [&[0, 0][..], &[1, 0], &[0, 4], &[0]] {
            assert!(
                attention_with_layout(
                    &query,
                    &keys,
                    &keys,
                    1,
                    1,
                    2,
                    1.0,
                    AttentionLayout {
                        row_len: 4,
                        columns: Some(columns)
                    },
                )
                .is_err()
            );
        }
        for row_len in [0, 1, usize::MAX] {
            assert!(
                attention_with_layout(
                    &query,
                    &keys,
                    &keys,
                    1,
                    1,
                    2,
                    1.0,
                    AttentionLayout {
                        row_len,
                        columns: None
                    },
                )
                .is_err()
            );
        }
        assert!(AttentionLayout::contiguous_padded(0).is_err());
        assert!(AttentionLayout::contiguous_padded(usize::MAX).is_err());
        assert!(layer_norm(&[], &[], &[], 1e-5).is_err());
        assert!(layer_norm(&[1.0], &[], &[0.0], 1e-5).is_err());
    }
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
    fn every_source_cpu_weight_type_loads_and_executes_through_matrix() {
        let mut supported = 0;
        for dtype in DType::ALL {
            if llama_quant::matvec_activation_dtype(dtype).is_err() {
                continue;
            }
            let cols = dtype.block_size();
            let matrix =
                Matrix::from_quantized(1, cols, dtype, vec![0; dtype.type_size()]).unwrap();
            assert_eq!(matrix.storage_dtype(), dtype);
            assert_eq!(
                matrix.multiply(&vec![1.0; cols]).unwrap(),
                vec![0.0],
                "{dtype:?}"
            );
            supported += 1;
        }
        assert_eq!(supported, 29);
    }
    #[test]
    fn unsupported_dot_types_are_rejected_at_load() {
        assert!(matches!(
            Matrix::from_quantized(1, 256, DType::Q8_K, vec![0; 292]),
            Err(Error::Unsupported(_))
        ));
        assert!(Matrix::from_quantized(1, 31, DType::Q4_0, vec![0; 18]).is_err());
        assert!(Matrix::from_quantized(1, 32, DType::Q4_0, vec![0; 17]).is_err());
    }
}

#[cfg(test)]
mod native_attention_tests {
    use super::attention;

    /// Captured from the original DLL's unmodified BERT graph. Inputs and
    /// expected results come from the reference, including later-layer inputs,
    /// so an earlier Rust error cannot contaminate the fixture.
    #[test]
    fn twelve_bert_layers_match_original_scalar_attention() {
        let bytes = include_bytes!("../tests/fixtures/bert-attention-reference.bin");
        assert_eq!(&bytes[..8], b"RLAATTN1");
        let mut offset = 8usize;
        let read_u32 = |offset: &mut usize| {
            let value = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;
            value as usize
        };
        let read_floats = |offset: &mut usize, count: usize| {
            let values: Vec<f32> = bytes[*offset..*offset + count * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            *offset += count * 4;
            values
        };
        let layers = read_u32(&mut offset);
        assert_eq!(layers, 12);
        let mut compared = 0usize;
        for layer in 0..layers {
            let heads = read_u32(&mut offset);
            let kv_heads = read_u32(&mut offset);
            let head_dim = read_u32(&mut offset);
            let count = read_u32(&mut offset);
            assert_eq!((heads, kv_heads, head_dim, count), (12, 12, 32, 10));
            let width = heads * head_dim;
            let q = read_floats(&mut offset, count * width);
            let k = read_floats(&mut offset, count * kv_heads * head_dim);
            let v = read_floats(&mut offset, count * kv_heads * head_dim);
            let expected = read_floats(&mut offset, count * width);
            let scale = (head_dim as f32).sqrt().recip();
            for (token, query) in q.chunks_exact(width).enumerate() {
                let actual = attention(query, &k, &v, heads, kv_heads, head_dim, scale)
                    .expect("valid reference attention");
                for (dimension, (&a, &e)) in actual
                    .iter()
                    .zip(&expected[token * width..(token + 1) * width])
                    .enumerate()
                {
                    let error = (f64::from(a) - f64::from(e)).abs();
                    let tolerance = 1e-6 + 1e-6 * f64::from(e).abs();
                    assert!(
                        a.is_finite() && error <= tolerance,
                        "layer {layer}, token {token}, dimension {dimension}: {a} vs {e}"
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(offset, bytes.len());
        assert_eq!(compared, 46_080);
    }
}
