//! Dense Qwen2 graph translated from the frozen src/models/qwen2.cpp.
//! This executable uses full-head NeoX RoPE and the source scalar CPU equations.
use crate::llama::KvLayer;
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_context: usize,
    pub rms_epsilon: f32,
    pub rope_base: f32,
}
impl Qwen2Config {
    pub fn validate(&self) -> Result<()> {
        check_positive(&[
            ("vocabulary", self.vocabulary),
            ("width", self.width),
            ("feed_forward", self.feed_forward),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
            ("max_context", self.max_context),
        ])?;
        check_epsilon(self.rms_epsilon)?;
        if !self.head_dim()?.is_multiple_of(2) {
            return Err(Error::InvalidConfig(
                "Qwen2 full-head NeoX RoPE requires an even head dimension".into(),
            ));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(Error::InvalidConfig(
                "RoPE base must be finite and positive".into(),
            ));
        }
        self.kv_width()?
            .checked_mul(self.max_context)
            .ok_or_else(|| Error::InvalidConfig("Qwen2 KV size overflow".into()))?;
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
            .ok_or_else(|| Error::InvalidConfig("Qwen2 QKV size overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub enum Qwen2Qkv {
    Separate {
        query: Matrix,
        key: Matrix,
        value: Matrix,
        query_bias: Option<Vec<f32>>,
        key_bias: Option<Vec<f32>>,
        value_bias: Option<Vec<f32>>,
    },
    /// The original loader prefers a fused bias when present. Separate biases
    /// are applied only when the fused bias is absent and all three exist.
    Fused {
        projection: Matrix,
        bias: Option<Vec<f32>>,
        query_bias: Option<Vec<f32>>,
        key_bias: Option<Vec<f32>>,
        value_bias: Option<Vec<f32>>,
    },
}
impl Qwen2Qkv {
    fn validate(&self, c: &Qwen2Config) -> Result<()> {
        let kv = c.kv_width()?;
        let (q_bias, k_bias, v_bias) = match self {
            Self::Separate {
                query,
                key,
                value,
                query_bias,
                key_bias,
                value_bias,
            } => {
                query.shape(c.width, c.width, "Qwen2 query")?;
                key.shape(kv, c.width, "Qwen2 key")?;
                value.shape(kv, c.width, "Qwen2 value")?;
                (query_bias, key_bias, value_bias)
            }
            Self::Fused {
                projection,
                bias,
                query_bias,
                key_bias,
                value_bias,
            } => {
                projection.shape(c.qkv_width()?, c.width, "Qwen2 fused QKV")?;
                math::optional_bias(bias, c.qkv_width()?, "Qwen2 fused QKV bias")?;
                if bias.is_some()
                    && (query_bias.is_some() || key_bias.is_some() || value_bias.is_some())
                {
                    return Err(Error::InvalidConfig(
                        "Qwen2 fused and separate QKV biases cannot both be active".into(),
                    ));
                }
                (query_bias, key_bias, value_bias)
            }
        };
        math::optional_bias(q_bias, c.width, "Qwen2 query bias")?;
        math::optional_bias(k_bias, kv, "Qwen2 key bias")?;
        math::optional_bias(v_bias, kv, "Qwen2 value bias")?;
        Ok(())
    }
    fn project(
        &self,
        c: &Qwen2Config,
        index: usize,
        input: &[f32],
        adapters: &AdapterSet,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (mut q, mut k, mut v, q_bias, k_bias, v_bias, apply_separate_bias) = match self {
            Self::Separate {
                query,
                key,
                value,
                query_bias,
                key_bias,
                value_bias,
            } => (
                adapters.multiply(&format!("blk.{index}.attn_q.weight"), query, input)?,
                adapters.multiply(&format!("blk.{index}.attn_k.weight"), key, input)?,
                adapters.multiply(&format!("blk.{index}.attn_v.weight"), value, input)?,
                query_bias,
                key_bias,
                value_bias,
                true,
            ),
            Self::Fused {
                projection,
                bias,
                query_bias,
                key_bias,
                value_bias,
            } => {
                let mut qkv = adapters.multiply(
                    &format!("blk.{index}.attn_qkv.weight"),
                    projection,
                    input,
                )?;
                math::bias(&mut qkv, bias);
                let kv = c.kv_width()?;
                (
                    qkv[..c.width].to_vec(),
                    qkv[c.width..c.width + kv].to_vec(),
                    qkv[c.width + kv..].to_vec(),
                    query_bias,
                    key_bias,
                    value_bias,
                    query_bias.is_some() && key_bias.is_some() && value_bias.is_some(),
                )
            }
        };
        // The source fused graph enables separate Q/K/V biases as one group.
        // Its separate-projection graph applies each optional bias independently.
        if apply_separate_bias {
            math::bias(&mut q, q_bias);
            math::bias(&mut k, k_bias);
            math::bias(&mut v, v_bias);
        }
        Ok((q, k, v))
    }
    fn weight(&self, name: &str) -> Option<&Matrix> {
        match (self, name) {
            (Self::Separate { query, .. }, "attn_q.weight") => Some(query),
            (Self::Separate { key, .. }, "attn_k.weight") => Some(key),
            (Self::Separate { value, .. }, "attn_v.weight") => Some(value),
            (Self::Fused { projection, .. }, "attn_qkv.weight") => Some(projection),
            _ => None,
        }
    }
    fn fingerprint(&self, hash: &mut Fingerprint) {
        let (biases, matrices) = match self {
            Self::Separate {
                query,
                key,
                value,
                query_bias,
                key_bias,
                value_bias,
            } => {
                hash.bytes(b"separate");
                (
                    vec![query_bias, key_bias, value_bias],
                    vec![query, key, value],
                )
            }
            Self::Fused {
                projection,
                bias,
                query_bias,
                key_bias,
                value_bias,
            } => {
                hash.bytes(b"fused");
                (
                    vec![bias, query_bias, key_bias, value_bias],
                    vec![projection],
                )
            }
        };
        for matrix in matrices {
            hash.matrix(matrix);
        }
        for bias in biases {
            hash.usize(usize::from(bias.is_some()));
            if let Some(bias) = bias {
                hash.floats(bias);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2Layer {
    pub attention_norm: Vec<f32>,
    pub qkv: Qwen2Qkv,
    pub attention_output: Matrix,
    pub feed_forward_norm: Vec<f32>,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}
impl Qwen2Layer {
    fn validate(&self, c: &Qwen2Config) -> Result<()> {
        math::vector(&self.attention_norm, c.width, "Qwen2 attention RMS weights")?;
        math::vector(&self.feed_forward_norm, c.width, "Qwen2 FFN RMS weights")?;
        self.qkv.validate(c)?;
        self.attention_output
            .shape(c.width, c.width, "Qwen2 attention output")?;
        self.gate.shape(c.feed_forward, c.width, "Qwen2 FFN gate")?;
        self.up.shape(c.feed_forward, c.width, "Qwen2 FFN up")?;
        self.down.shape(c.width, c.feed_forward, "Qwen2 FFN down")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2Weights {
    pub token_embedding: Matrix,
    pub layers: Vec<Qwen2Layer>,
    pub output_norm: Vec<f32>,
    pub output: Option<Matrix>,
    pub output_bias: Option<Vec<f32>>,
}
#[derive(Debug, Clone)]
pub struct Qwen2Model {
    config: Qwen2Config,
    weights: Qwen2Weights,
    fingerprint: u64,
}
impl Qwen2Model {
    pub fn new(config: Qwen2Config, weights: Qwen2Weights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "Qwen2 token embedding")?;
        math::vector(&weights.output_norm, config.width, "Qwen2 output norm")?;
        if let Some(output) = &weights.output {
            output.shape(config.vocabulary, config.width, "Qwen2 output")?;
        }
        math::optional_bias(&weights.output_bias, config.vocabulary, "Qwen2 output bias")?;
        for layer in &weights.layers {
            layer.validate(&config)?;
        }
        let mut result = Self {
            config,
            weights,
            fingerprint: 0,
        };
        result.fingerprint = result.calculate_fingerprint();
        Ok(result)
    }
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }
    pub fn weights(&self) -> &Qwen2Weights {
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
        layer.qkv.weight(field).or(match field {
            "attn_output.weight" => Some(&layer.attention_output),
            "ffn_gate.weight" => Some(&layer.gate),
            "ffn_up.weight" => Some(&layer.up),
            "ffn_down.weight" => Some(&layer.down),
            _ => None,
        })
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
        let head_dim = c.head_dim()?;
        let scale = (head_dim as f32).sqrt().recip();
        for (index, (layer, kv)) in self.weights.layers.iter().zip(cache).enumerate() {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let normalized = math::rms_norm(&hidden, &layer.attention_norm, c.rms_epsilon)?;
            let (mut q, mut k, v) = layer.qkv.project(c, index, &normalized, adapters)?;
            math::rope_neox(&mut q, head_dim, position, c.rope_base);
            math::rope_neox(&mut k, head_dim, position, c.rope_base);
            kv.keys.extend(k);
            kv.values.extend(v);
            let attention = math::attention(
                &q, &kv.keys, &kv.values, c.heads, c.kv_heads, head_dim, scale,
            )?;
            let projected = adapters.multiply(
                &format!("blk.{index}.attn_output.weight"),
                &layer.attention_output,
                &attention,
            )?;
            math::add(&mut hidden, &projected);
            let normalized = math::rms_norm(&hidden, &layer.feed_forward_norm, c.rms_epsilon)?;
            let mut gate = adapters.multiply(
                &format!("blk.{index}.ffn_gate.weight"),
                &layer.gate,
                &normalized,
            )?;
            let up = adapters.multiply(
                &format!("blk.{index}.ffn_up.weight"),
                &layer.up,
                &normalized,
            )?;
            math::native_rows::silu_f32_in_place(&mut gate);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate *= up;
            }
            let output =
                adapters.multiply(&format!("blk.{index}.ffn_down.weight"), &layer.down, &gate)?;
            math::add(&mut hidden, &output);
            adapters.control(index, &mut hidden);
            math::finite(&hidden, "Qwen2 layer output")?;
        }
        let embedding = math::rms_norm(&hidden, &self.weights.output_norm, c.rms_epsilon)?;
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embedding);
        let mut logits = adapters.multiply("output.weight", output, &embedding)?;
        math::bias(&mut logits, &self.weights.output_bias);
        math::finite(&logits, "Qwen2 logits")?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.bytes(b"qwen2-dense-neox-v1");
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
        hash.floats(&[c.rms_epsilon, c.rope_base]);
        hash.matrix(&self.weights.token_embedding);
        hash.floats(&self.weights.output_norm);
        hash.usize(usize::from(self.weights.output.is_some()));
        if let Some(output) = &self.weights.output {
            hash.matrix(output);
        }
        hash.usize(usize::from(self.weights.output_bias.is_some()));
        if let Some(bias) = &self.weights.output_bias {
            hash.floats(bias);
        }
        for layer in &self.weights.layers {
            hash.floats(&layer.attention_norm);
            layer.qkv.fingerprint(&mut hash);
            hash.floats(&layer.feed_forward_norm);
            for matrix in [&layer.attention_output, &layer.gate, &layer.up, &layer.down] {
                hash.matrix(matrix);
            }
        }
        hash.finish()
    }
}
