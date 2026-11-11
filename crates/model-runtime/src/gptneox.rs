//! GPT-NeoX graph from frozen src/models/gptneox.cpp.
//! Partial NeoX RoPE, affine norms, fused QKV and both residual orders.
use crate::llama::KvLayer;
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct GptNeoXConfig {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_context: usize,
    pub norm_epsilon: f32,
    pub rope_base: f32,
    pub rope_dimensions: usize,
    pub parallel_residual: bool,
}
impl GptNeoXConfig {
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
        let head_dim = self.head_dim()?;
        if self.rope_dimensions == 0 {
            return Err(Error::Unsupported(
                "GPT-NeoX without rotary positions is outside the native graph scope".into(),
            ));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(Error::Unsupported(
                "GPT-NeoX native RoPE requires an even head width".into(),
            ));
        }
        if self.rope_dimensions > head_dim || !self.rope_dimensions.is_multiple_of(2) {
            return Err(Error::InvalidConfig(
                "GPT-NeoX rotary dimension must be positive, even and at most the head width"
                    .into(),
            ));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(Error::InvalidConfig(
                "GPT-NeoX RoPE base must be finite and positive".into(),
            ));
        }
        self.kv_width()?
            .checked_mul(self.max_context)
            .ok_or_else(|| Error::InvalidConfig("GPT-NeoX KV size overflow".into()))?;
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
            .ok_or_else(|| Error::InvalidConfig("GPT-NeoX QKV size overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub struct GptNeoXLayer {
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
impl GptNeoXLayer {
    fn validate(&self, c: &GptNeoXConfig) -> Result<()> {
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
        math::vector(&self.qkv_bias, c.qkv_width()?, "GPT-NeoX QKV bias")?;
        math::vector(&self.up_bias, c.feed_forward, "GPT-NeoX FFN up bias")?;
        self.qkv.shape(c.qkv_width()?, c.width, "GPT-NeoX QKV")?;
        self.attention_output
            .shape(c.width, c.width, "GPT-NeoX attention output")?;
        self.up.shape(c.feed_forward, c.width, "GPT-NeoX FFN up")?;
        self.down
            .shape(c.width, c.feed_forward, "GPT-NeoX FFN down")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GptNeoXWeights {
    pub token_embedding: Matrix,
    pub layers: Vec<GptNeoXLayer>,
    pub output_norm: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    pub output: Matrix,
}
#[derive(Debug, Clone)]
pub struct GptNeoXModel {
    config: GptNeoXConfig,
    weights: GptNeoXWeights,
    fingerprint: u64,
}
impl GptNeoXModel {
    pub fn new(config: GptNeoXConfig, weights: GptNeoXWeights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights.token_embedding.shape(
            config.vocabulary,
            config.width,
            "GPT-NeoX token embedding",
        )?;
        weights
            .output
            .shape(config.vocabulary, config.width, "GPT-NeoX output")?;
        math::vector(&weights.output_norm, config.width, "GPT-NeoX output norm")?;
        math::vector(
            &weights.output_norm_bias,
            config.width,
            "GPT-NeoX output norm bias",
        )?;
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
    pub fn config(&self) -> &GptNeoXConfig {
        &self.config
    }
    pub fn weights(&self) -> &GptNeoXWeights {
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
            return Some(&self.weights.output);
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
            let (q, rest) = qkv.split_at_mut(c.width);
            let (k, v) = rest.split_at_mut(kv_width);
            // Native NEOX rotation pairs the two halves of the rotary prefix.
            // The rest of each head is copied unchanged.
            for values in [&mut *q, &mut *k] {
                for head in values.chunks_exact_mut(head_dim) {
                    math::rope_neox(
                        &mut head[..c.rope_dimensions],
                        c.rope_dimensions,
                        position,
                        c.rope_base,
                    );
                }
            }
            kv.keys.extend_from_slice(k);
            kv.values.extend_from_slice(v);
            let attention = math::attention(
                q, &kv.keys, &kv.values, c.heads, c.kv_heads, head_dim, scale,
            )?;
            let mut projected = adapters.multiply(
                &format!("blk.{index}.attn_output.weight"),
                &layer.attention_output,
                &attention,
            )?;
            math::add(&mut projected, &layer.attention_output_bias);
            let ffn_input = if c.parallel_residual {
                &hidden
            } else {
                // Source order: attn_out + input before the second LayerNorm.
                math::add(&mut projected, &hidden);
                &projected
            };
            let normalized = math::layer_norm(
                ffn_input,
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
            // In parallel mode source rounding is (ffn_out + input) + attn_out.
            math::add(&mut output, ffn_input);
            if c.parallel_residual {
                math::add(&mut output, &projected);
            }
            hidden = output;
            adapters.control(index, &mut hidden);
            math::finite(&hidden, "GPT-NeoX layer output")?;
        }
        let embedding = math::layer_norm(
            &hidden,
            &self.weights.output_norm,
            &self.weights.output_norm_bias,
            c.norm_epsilon,
        )?;
        let logits = adapters.multiply("output.weight", &self.weights.output, &embedding)?;
        math::finite(&logits, "GPT-NeoX logits")?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.bytes(b"gptneox-rope-affine-parallel-v1");
        let c = &self.config;
        for value in [
            c.vocabulary,
            c.width,
            c.feed_forward,
            c.heads,
            c.kv_heads,
            c.max_context,
            c.rope_dimensions,
            usize::from(c.parallel_residual),
            self.weights.layers.len(),
        ] {
            hash.usize(value);
        }
        hash.floats(&[c.norm_epsilon, c.rope_base]);
        hash.matrix(&self.weights.token_embedding);
        hash.matrix(&self.weights.output);
        hash.floats(&self.weights.output_norm);
        hash.floats(&self.weights.output_norm_bias);
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
