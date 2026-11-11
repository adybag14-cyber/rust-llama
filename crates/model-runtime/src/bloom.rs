//! BLOOM decoder from frozen src/models/bloom.cpp.
//! Input affine LayerNorm, fixed-bias ALiBi, fused Q/K/V, sequential GELU FFN,
//! final affine LayerNorm, and optional output tying. F32 contiguous KV state.
use crate::llama::KvLayer;
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct BloomConfig {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_context: usize,
    pub norm_epsilon: f32,
}
impl BloomConfig {
    pub fn validate(&self) -> Result<()> {
        check_positive(&[
            ("vocabulary", self.vocabulary),
            ("width", self.width),
            ("feed_forward", self.feed_forward),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
            ("max_context", self.max_context),
        ])?;
        check_epsilon(self.norm_epsilon)?;
        self.kv_width()?
            .checked_mul(self.max_context)
            .ok_or_else(|| Error::InvalidConfig("BLOOM KV size overflow".into()))?;
        self.qkv_width()?;
        Ok(())
    }
    pub fn head_dim(&self) -> Result<usize> {
        crate::checked_head_dimension(self.width, self.heads)
    }
    pub fn kv_width(&self) -> Result<usize> {
        crate::checked_kv_width(self.width, self.heads, self.kv_heads)
    }
    pub fn qkv_width(&self) -> Result<usize> {
        self.kv_width()?
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.width))
            .ok_or_else(|| Error::InvalidConfig("BLOOM QKV size overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub struct BloomLayer {
    pub attention_norm: Vec<f32>,
    pub attention_norm_bias: Vec<f32>,
    pub qkv: Matrix,
    pub qkv_bias: Vec<f32>,
    pub attention_output: Matrix,
    pub attention_output_bias: Vec<f32>,
    pub feed_forward_norm: Vec<f32>,
    pub feed_forward_norm_bias: Vec<f32>,
    pub up: Matrix,
    pub up_bias: Vec<f32>,
    pub down: Matrix,
    pub down_bias: Vec<f32>,
}
impl BloomLayer {
    fn validate(&self, c: &BloomConfig) -> Result<()> {
        for (name, value) in [
            ("attention norm", &self.attention_norm),
            ("attention norm bias", &self.attention_norm_bias),
            ("attention output bias", &self.attention_output_bias),
            ("FFN norm", &self.feed_forward_norm),
            ("FFN norm bias", &self.feed_forward_norm_bias),
            ("FFN down bias", &self.down_bias),
        ] {
            math::vector(value, c.width, name)?;
        }
        math::vector(&self.qkv_bias, c.qkv_width()?, "BLOOM QKV bias")?;
        math::vector(&self.up_bias, c.feed_forward, "BLOOM FFN up bias")?;
        self.qkv.shape(c.qkv_width()?, c.width, "BLOOM QKV")?;
        self.attention_output
            .shape(c.width, c.width, "BLOOM attention output")?;
        self.up.shape(c.feed_forward, c.width, "BLOOM FFN up")?;
        self.down.shape(c.width, c.feed_forward, "BLOOM FFN down")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BloomWeights {
    pub token_embedding: Matrix,
    pub embedding_norm: Vec<f32>,
    pub embedding_norm_bias: Vec<f32>,
    pub layers: Vec<BloomLayer>,
    pub output_norm: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    pub output: Option<Matrix>,
}
#[derive(Debug, Clone)]
pub struct BloomModel {
    config: BloomConfig,
    weights: BloomWeights,
    fingerprint: u64,
}
impl BloomModel {
    pub fn new(config: BloomConfig, weights: BloomWeights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "BLOOM token embedding")?;
        math::vector(&weights.embedding_norm, config.width, "BLOOM input norm")?;
        math::vector(
            &weights.embedding_norm_bias,
            config.width,
            "BLOOM input norm bias",
        )?;
        math::vector(&weights.output_norm, config.width, "BLOOM output norm")?;
        math::vector(
            &weights.output_norm_bias,
            config.width,
            "BLOOM output norm bias",
        )?;
        if let Some(output) = &weights.output {
            output.shape(config.vocabulary, config.width, "BLOOM output")?;
        }
        for layer in &weights.layers {
            layer.validate(&config)?;
        }
        let mut model = Self {
            config,
            weights,
            fingerprint: 0,
        };
        model.fingerprint = model.calculate_fingerprint();
        Ok(model)
    }
    pub fn config(&self) -> &BloomConfig {
        &self.config
    }
    pub fn weights(&self) -> &BloomWeights {
        &self.weights
    }
    pub fn layer_count(&self) -> usize {
        self.weights.layers.len()
    }
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    pub fn weight(&self, name: &str) -> Option<&Matrix> {
        if name == "output.weight" {
            return Some(
                self.weights
                    .output
                    .as_ref()
                    .unwrap_or(&self.weights.token_embedding),
            );
        }
        let (index, field) = name.strip_prefix("blk.")?.split_once('.')?;
        let parsed = index.parse::<usize>().ok()?;
        if parsed.to_string() != index {
            return None;
        }
        let layer = self.weights.layers.get(parsed)?;
        match field {
            "attn_qkv.weight" => Some(&layer.qkv),
            "attn_output.weight" => Some(&layer.attention_output),
            "ffn_up.weight" => Some(&layer.up),
            "ffn_down.weight" => Some(&layer.down),
            _ => None,
        }
    }
    pub(crate) fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= self.config.vocabulary {
            return Err(Error::InvalidToken {
                token,
                vocabulary: self.config.vocabulary,
            });
        }
        Ok(self.weights.token_embedding.row(token as usize)?.to_vec())
    }
    pub(crate) fn step(
        &self,
        mut hidden: Vec<f32>,
        position: usize,
        cache: &mut [KvLayer],
        adapters: &AdapterSet,
        cancel: &mut impl FnMut() -> bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        if position >= c.max_context {
            return Err(Error::ContextFull {
                maximum: c.max_context,
            });
        }
        math::vector(&hidden, c.width, "BLOOM input embedding")?;
        if cache.len() != self.weights.layers.len() {
            return Err(Error::InvalidState("BLOOM cache layer count".into()));
        }
        let expected = position
            .checked_mul(c.kv_width()?)
            .ok_or_else(|| Error::InvalidState("BLOOM cache length overflow".into()))?;
        if cache
            .iter()
            .any(|layer| layer.keys.len() != expected || layer.values.len() != expected)
        {
            return Err(Error::InvalidState("BLOOM cache position mismatch".into()));
        }
        // This affine input normalization is also required for explicit embeddings.
        hidden = math::layer_norm(
            &hidden,
            &self.weights.embedding_norm,
            &self.weights.embedding_norm_bias,
            c.norm_epsilon,
        )?;
        let head_dim = c.head_dim()?;
        let kv_width = c.kv_width()?;
        let scale = (head_dim as f32).sqrt().recip();
        for (index, (layer, kv)) in self.weights.layers.iter().zip(cache).enumerate() {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let normalized = math::layer_norm(
                &hidden,
                &layer.attention_norm,
                &layer.attention_norm_bias,
                c.norm_epsilon,
            )?;
            let mut qkv = adapters.multiply(
                &format!("blk.{index}.attn_qkv.weight"),
                &layer.qkv,
                &normalized,
            )?;
            math::add(&mut qkv, &layer.qkv_bias);
            let q = &qkv[..c.width];
            kv.keys.extend_from_slice(&qkv[c.width..c.width + kv_width]);
            kv.values.extend_from_slice(&qkv[c.width + kv_width..]);
            let attention = alibi_attention(
                q, &kv.keys, &kv.values, c.heads, c.kv_heads, head_dim, scale,
            )?;
            let mut projected = adapters.multiply(
                &format!("blk.{index}.attn_output.weight"),
                &layer.attention_output,
                &attention,
            )?;
            math::add(&mut projected, &layer.attention_output_bias);
            math::add(&mut hidden, &projected);
            let normalized = math::layer_norm(
                &hidden,
                &layer.feed_forward_norm,
                &layer.feed_forward_norm_bias,
                c.norm_epsilon,
            )?;
            let mut up = adapters.multiply(
                &format!("blk.{index}.ffn_up.weight"),
                &layer.up,
                &normalized,
            )?;
            math::add(&mut up, &layer.up_bias);
            for value in &mut up {
                *value = math::gelu(*value);
            }
            let mut output =
                adapters.multiply(&format!("blk.{index}.ffn_down.weight"), &layer.down, &up)?;
            math::add(&mut output, &layer.down_bias);
            math::add(&mut hidden, &output);
            adapters.control(index, &mut hidden);
            math::finite(&hidden, "BLOOM layer output")?;
        }
        let embedding = math::layer_norm(
            &hidden,
            &self.weights.output_norm,
            &self.weights.output_norm_bias,
            c.norm_epsilon,
        )?;
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embedding);
        let logits = adapters.multiply("output.weight", output, &embedding)?;
        math::finite(&logits, "BLOOM logits")?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.bytes(b"bloom-input-norm-alibi-gelu-v1");
        let c = &self.config;
        for value in [
            c.vocabulary,
            c.width,
            c.feed_forward,
            c.heads,
            c.kv_heads,
            c.max_context,
            self.layer_count(),
        ] {
            hash.usize(value);
        }
        hash.floats(&[c.norm_epsilon]);
        hash.matrix(&self.weights.token_embedding);
        hash.floats(&self.weights.embedding_norm);
        hash.floats(&self.weights.embedding_norm_bias);
        hash.floats(&self.weights.output_norm);
        hash.floats(&self.weights.output_norm_bias);
        hash.usize(usize::from(self.weights.output.is_some()));
        if let Some(output) = &self.weights.output {
            hash.matrix(output);
        }
        for layer in &self.weights.layers {
            for vector in [
                &layer.attention_norm,
                &layer.attention_norm_bias,
                &layer.qkv_bias,
                &layer.attention_output_bias,
                &layer.feed_forward_norm,
                &layer.feed_forward_norm_bias,
                &layer.up_bias,
                &layer.down_bias,
            ] {
                hash.floats(vector);
            }
            for matrix in [&layer.qkv, &layer.attention_output, &layer.up, &layer.down] {
                hash.matrix(matrix);
            }
        }
        hash.finish()
    }
}

/// Source nonflash QK -> separate F32 scale -> F32 slope*mask addition ->
/// native-profile softmax over the full padded row -> F32 products / F64 sum.
/// The fixed bias is an architecture constant in bloom.cpp, not a GGUF override.
fn alibi_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .filter(|&n| n != 0)
        .ok_or_else(|| Error::Shape("BLOOM attention KV width".into()))?;
    if heads == 0
        || !heads.is_multiple_of(kv_heads)
        || heads.checked_mul(head_dim) != Some(query.len())
        || keys.len() != values.len()
        || !keys.len().is_multiple_of(kv_width)
        || !scale.is_finite()
    {
        return Err(Error::Shape("BLOOM attention shape".into()));
    }
    let count = keys.len() / kv_width;
    if count == 0 || count > u32::MAX as usize {
        return Err(Error::Shape("BLOOM attention token count".into()));
    }
    let row_len = count
        .checked_add(255)
        .map(|n| n / 256 * 256)
        .ok_or_else(|| Error::Shape("BLOOM attention padding overflow".into()))?;
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(row_len)
        .map_err(|_| Error::InvalidState("BLOOM attention row allocation".into()))?;
    scores.resize(row_len, f32::NEG_INFINITY);
    let mut output = Vec::new();
    output
        .try_reserve_exact(query.len())
        .map_err(|_| Error::InvalidState("BLOOM attention output allocation".into()))?;
    output.resize(query.len(), 0.0f64);
    let head_power = 1usize << heads.ilog2();
    let m0 = 2.0f32.powf(-8.0 / head_power as f32);
    let m1 = 2.0f32.powf(-4.0 / head_power as f32);
    for head in 0..heads {
        scores.fill(f32::NEG_INFINITY);
        let slope = if head < head_power {
            m0.powf((head + 1) as f32)
        } else {
            m1.powf((2 * (head - head_power) + 1) as f32)
        };
        let kv_head = head / (heads / kv_heads);
        let q = &query[head * head_dim..(head + 1) * head_dim];
        for (token, destination) in scores.iter_mut().take(count).enumerate() {
            let offset = token * kv_width + kv_head * head_dim;
            let dot = q
                .iter()
                .zip(&keys[offset..offset + head_dim])
                .map(|(a, b)| f64::from(a * b))
                .sum::<f64>() as f32;
            // The original mask subtracts integer positions before converting.
            // Keep +0 for the diagonal, matching -abs(integer zero).
            let mask = -((count - 1 - token) as i64) as f32;
            let scaled = dot * scale;
            let score = scaled + slope * mask;
            if !score.is_finite() {
                return Err(Error::NonFinite("BLOOM attention score".into()));
            }
            *destination = score;
        }
        math::softmax(&mut scores)?;
        for (token, &probability) in scores.iter().take(count).enumerate() {
            let offset = token * kv_width + kv_head * head_dim;
            for d in 0..head_dim {
                output[head * head_dim + d] += f64::from(probability * values[offset + d]);
            }
        }
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(output.len())
        .map_err(|_| Error::InvalidState("BLOOM attention result allocation".into()))?;
    result.extend(output.into_iter().map(|v| v as f32));
    math::finite(&result, "BLOOM attention output")?;
    Ok(result)
}
