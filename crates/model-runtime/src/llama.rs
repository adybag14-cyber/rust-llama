//! Dense F32 Llama graph slice translated from `src/models/llama.cpp`.
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_context: usize,
    pub rms_epsilon: f32,
    pub rope_base: f32,
    /// `None` uses 1/sqrt(head_dim), matching zero in the original hparams.
    pub attention_scale: Option<f32>,
}
impl LlamaConfig {
    fn validate(&self) -> Result<()> {
        check_positive(&[
            ("vocabulary", self.vocabulary),
            ("width", self.width),
            ("feed_forward", self.feed_forward),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
            ("max_context", self.max_context),
        ])?;
        check_epsilon(self.rms_epsilon)?;
        if !self.width.is_multiple_of(self.heads) || !self.heads.is_multiple_of(self.kv_heads) {
            return Err(Error::InvalidConfig(
                "width must divide into heads; heads must divide into KV groups".into(),
            ));
        }
        if !self.head_dim()?.is_multiple_of(2) {
            return Err(Error::InvalidConfig(
                "Llama full-head RoPE requires an even head dimension".into(),
            ));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(Error::InvalidConfig(
                "RoPE base must be finite and positive".into(),
            ));
        }
        if self
            .attention_scale
            .is_some_and(|x| !x.is_finite() || x == 0.0)
        {
            return Err(Error::InvalidConfig(
                "attention scale must be finite and nonzero when provided".into(),
            ));
        }
        self.kv_width()?
            .checked_mul(self.max_context)
            .ok_or_else(|| Error::InvalidConfig("KV size overflow".into()))?;
        Ok(())
    }
    /// Validate the public raw dimensions before computing their quotient.
    pub fn head_dim(&self) -> Result<usize> {
        crate::checked_head_dimension(self.width, self.heads)
    }
    /// Validate head grouping and use checked multiplication, including when
    /// called on a configuration that has not been passed to `LlamaModel::new`.
    pub fn kv_width(&self) -> Result<usize> {
        crate::checked_kv_width(self.width, self.heads, self.kv_heads)
    }
}

#[derive(Debug, Clone)]
pub struct LlamaLayer {
    pub attention_norm: Vec<f32>,
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub attention_output: Matrix,
    pub query_bias: Option<Vec<f32>>,
    pub key_bias: Option<Vec<f32>>,
    pub value_bias: Option<Vec<f32>>,
    pub attention_output_bias: Option<Vec<f32>>,
    pub feed_forward_norm: Vec<f32>,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
    pub gate_bias: Option<Vec<f32>>,
    pub up_bias: Option<Vec<f32>>,
    pub down_bias: Option<Vec<f32>>,
}
impl LlamaLayer {
    fn validate(&self, c: &LlamaConfig) -> Result<()> {
        math::vector(&self.attention_norm, c.width, "attention RMS weights")?;
        math::vector(&self.feed_forward_norm, c.width, "FFN RMS weights")?;
        self.query.shape(c.width, c.width, "query")?;
        self.key.shape(c.kv_width()?, c.width, "key")?;
        self.value.shape(c.kv_width()?, c.width, "value")?;
        self.attention_output
            .shape(c.width, c.width, "attention output")?;
        self.gate.shape(c.feed_forward, c.width, "FFN gate")?;
        self.up.shape(c.feed_forward, c.width, "FFN up")?;
        self.down.shape(c.width, c.feed_forward, "FFN down")?;
        for (value, len, name) in [
            (&self.query_bias, c.width, "query bias"),
            (&self.key_bias, c.kv_width()?, "key bias"),
            (&self.value_bias, c.kv_width()?, "value bias"),
            (
                &self.attention_output_bias,
                c.width,
                "attention output bias",
            ),
            (&self.gate_bias, c.feed_forward, "gate bias"),
            (&self.up_bias, c.feed_forward, "up bias"),
            (&self.down_bias, c.width, "down bias"),
        ] {
            math::optional_bias(value, len, name)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LlamaWeights {
    pub token_embedding: Matrix,
    pub layers: Vec<LlamaLayer>,
    pub output_norm: Vec<f32>,
    /// `None` ties the language-model output to token embeddings.
    pub output: Option<Matrix>,
}

#[derive(Debug, Clone)]
pub struct LlamaModel {
    config: LlamaConfig,
    weights: LlamaWeights,
    fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KvLayer {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

impl LlamaModel {
    pub fn new(config: LlamaConfig, weights: LlamaWeights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "token embedding")?;
        math::vector(&weights.output_norm, config.width, "output RMS weights")?;
        if let Some(output) = &weights.output {
            output.shape(config.vocabulary, config.width, "output")?;
        }
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
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
    pub fn layer_count(&self) -> usize {
        self.weights.layers.len()
    }
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    pub fn weights(&self) -> &LlamaWeights {
        &self.weights
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
        let suffix = name.strip_prefix("blk.")?;
        let (index, field) = suffix.split_once('.')?;
        let parsed = index.parse::<usize>().ok()?;
        if parsed.to_string() != index {
            return None;
        }
        let layer = self.weights.layers.get(parsed)?;
        match field {
            "attn_q.weight" => Some(&layer.query),
            "attn_k.weight" => Some(&layer.key),
            "attn_v.weight" => Some(&layer.value),
            "attn_output.weight" => Some(&layer.attention_output),
            "ffn_gate.weight" => Some(&layer.gate),
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
    pub(crate) fn empty_cache(&self) -> Vec<KvLayer> {
        self.weights
            .layers
            .iter()
            .map(|_| KvLayer {
                keys: Vec::new(),
                values: Vec::new(),
            })
            .collect()
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
        let scale = c
            .attention_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        for (index, (layer, kv)) in self.weights.layers.iter().zip(cache).enumerate() {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let norm = math::rms_norm(&hidden, &layer.attention_norm, c.rms_epsilon)?;
            let mut q =
                adapters.multiply(&format!("blk.{index}.attn_q.weight"), &layer.query, &norm)?;
            let mut k =
                adapters.multiply(&format!("blk.{index}.attn_k.weight"), &layer.key, &norm)?;
            let mut v =
                adapters.multiply(&format!("blk.{index}.attn_v.weight"), &layer.value, &norm)?;
            math::bias(&mut q, &layer.query_bias);
            math::bias(&mut k, &layer.key_bias);
            math::bias(&mut v, &layer.value_bias);
            math::rope(&mut q, c.head_dim()?, position, c.rope_base);
            math::rope(&mut k, c.head_dim()?, position, c.rope_base);
            kv.keys.extend(k);
            kv.values.extend(v);
            let attention = math::attention(
                &q,
                &kv.keys,
                &kv.values,
                c.heads,
                c.kv_heads,
                c.head_dim()?,
                scale,
            )?;
            let mut projected = adapters.multiply(
                &format!("blk.{index}.attn_output.weight"),
                &layer.attention_output,
                &attention,
            )?;
            math::bias(&mut projected, &layer.attention_output_bias);
            math::add(&mut hidden, &projected);
            let norm = math::rms_norm(&hidden, &layer.feed_forward_norm, c.rms_epsilon)?;
            let mut gate =
                adapters.multiply(&format!("blk.{index}.ffn_gate.weight"), &layer.gate, &norm)?;
            let mut up =
                adapters.multiply(&format!("blk.{index}.ffn_up.weight"), &layer.up, &norm)?;
            math::bias(&mut gate, &layer.gate_bias);
            math::bias(&mut up, &layer.up_bias);
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate = math::silu(*gate) * up;
            }
            let mut ffn =
                adapters.multiply(&format!("blk.{index}.ffn_down.weight"), &layer.down, &gate)?;
            math::bias(&mut ffn, &layer.down_bias);
            math::add(&mut hidden, &ffn);
            adapters.control(index, &mut hidden);
            math::finite(&hidden, "Llama layer output")?;
        }
        let embedding = math::rms_norm(&hidden, &self.weights.output_norm, c.rms_epsilon)?;
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embedding);
        let logits = adapters.multiply("output.weight", output, &embedding)?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut h = Fingerprint::new();
        h.bytes(b"llama-dense-f32-v1");
        let c = &self.config;
        for x in [
            c.vocabulary,
            c.width,
            c.feed_forward,
            c.heads,
            c.kv_heads,
            c.max_context,
            self.layer_count(),
        ] {
            h.usize(x);
        }
        h.floats(&[c.rms_epsilon, c.rope_base, c.attention_scale.unwrap_or(0.0)]);
        h.matrix(&self.weights.token_embedding);
        h.floats(&self.weights.output_norm);
        h.usize(usize::from(self.weights.output.is_some()));
        if let Some(output) = &self.weights.output {
            h.matrix(output);
        }
        for layer in &self.weights.layers {
            h.floats(&layer.attention_norm);
            h.floats(&layer.feed_forward_norm);
            for m in [
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.attention_output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                h.matrix(m);
            }
            for b in [
                &layer.query_bias,
                &layer.key_bias,
                &layer.value_bias,
                &layer.attention_output_bias,
                &layer.gate_bias,
                &layer.up_bias,
                &layer.down_bias,
            ] {
                h.usize(usize::from(b.is_some()));
                if let Some(b) = b {
                    h.floats(b);
                }
            }
        }
        h.finish()
    }
}
