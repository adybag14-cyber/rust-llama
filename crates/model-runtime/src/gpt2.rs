//! GPT-2 learned-position decoder from frozen src/models/gpt2.cpp.
//! Fused Q/K/V, affine LayerNorm, sequential GELU FFN and optional tied output.
use crate::llama::KvLayer;
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// Physical number of learned position vectors, independent of request limit.
    pub positions: usize,
    pub max_context: usize,
    pub norm_epsilon: f32,
}
impl Gpt2Config {
    pub fn validate(&self) -> Result<()> {
        check_positive(&[
            ("vocabulary", self.vocabulary),
            ("width", self.width),
            ("feed_forward", self.feed_forward),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
            ("positions", self.positions),
            ("max_context", self.max_context),
        ])?;
        check_epsilon(self.norm_epsilon)?;
        if self.max_context > self.positions {
            return Err(Error::InvalidConfig(
                "GPT-2 context exceeds learned positions".into(),
            ));
        }
        self.kv_width()?
            .checked_mul(self.max_context)
            .ok_or_else(|| Error::InvalidConfig("GPT-2 KV size overflow".into()))?;
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
            .ok_or_else(|| Error::InvalidConfig("GPT-2 QKV size overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub struct Gpt2Layer {
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
impl Gpt2Layer {
    fn validate(&self, c: &Gpt2Config) -> Result<()> {
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
        math::vector(&self.qkv_bias, c.qkv_width()?, "GPT-2 QKV bias")?;
        math::vector(&self.up_bias, c.feed_forward, "GPT-2 FFN up bias")?;
        self.qkv.shape(c.qkv_width()?, c.width, "GPT-2 QKV")?;
        self.attention_output
            .shape(c.width, c.width, "GPT-2 attention output")?;
        self.up.shape(c.feed_forward, c.width, "GPT-2 FFN up")?;
        self.down.shape(c.width, c.feed_forward, "GPT-2 FFN down")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Gpt2Weights {
    pub token_embedding: Matrix,
    pub position_embedding: Matrix,
    pub layers: Vec<Gpt2Layer>,
    pub output_norm: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    pub output: Option<Matrix>,
    pub output_bias: Option<Vec<f32>>,
}
#[derive(Debug, Clone)]
pub struct Gpt2Model {
    config: Gpt2Config,
    weights: Gpt2Weights,
    fingerprint: u64,
}
impl Gpt2Model {
    pub fn new(config: Gpt2Config, weights: Gpt2Weights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "GPT-2 token embedding")?;
        weights.position_embedding.shape(
            config.positions,
            config.width,
            "GPT-2 position embedding",
        )?;
        math::vector(&weights.output_norm, config.width, "GPT-2 output norm")?;
        math::vector(
            &weights.output_norm_bias,
            config.width,
            "GPT-2 output norm bias",
        )?;
        if let Some(output) = &weights.output {
            output.shape(config.vocabulary, config.width, "GPT-2 output")?;
        }
        math::optional_bias(&weights.output_bias, config.vocabulary, "GPT-2 output bias")?;
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
    pub fn config(&self) -> &Gpt2Config {
        &self.config
    }
    pub fn weights(&self) -> &Gpt2Weights {
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
        // Positions remain a separate input in the native graph, including
        // callers that provide an embedding vector instead of a token ID.
        math::add(&mut hidden, self.weights.position_embedding.row(position)?);
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
            let attention = math::attention(
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
            math::finite(&hidden, "GPT-2 layer output")?;
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
        let mut logits = adapters.multiply("output.weight", output, &embedding)?;
        math::bias(&mut logits, &self.weights.output_bias);
        math::finite(&logits, "GPT-2 logits")?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.bytes(b"gpt2-learned-position-gelu-v1");
        let c = &self.config;
        for value in [
            c.vocabulary,
            c.width,
            c.feed_forward,
            c.heads,
            c.kv_heads,
            c.positions,
            c.max_context,
            self.layer_count(),
        ] {
            hash.usize(value);
        }
        hash.floats(&[c.norm_epsilon]);
        hash.matrix(&self.weights.token_embedding);
        hash.matrix(&self.weights.position_embedding);
        hash.floats(&self.weights.output_norm);
        hash.floats(&self.weights.output_norm_bias);
        hash.usize(usize::from(self.weights.output.is_some()));
        if let Some(output) = &self.weights.output {
            hash.matrix(output);
        }
        hash.usize(usize::from(self.weights.output_bias.is_some()));
        if let Some(bias) = &self.weights.output_bias {
            hash.floats(bias);
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
