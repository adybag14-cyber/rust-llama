//! Falcon graph from frozen src/models/falcon.cpp.
//!
//! Shared or separate affine attention/FFN normalization, fused Q/K/V, full
//! NeoX RoPE, GELU FFN and source-order parallel residuals. F32 contiguous KV.
use crate::llama::KvLayer;
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct FalconConfig {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_context: usize,
    pub norm_epsilon: f32,
    pub rope_base: f32,
}
impl FalconConfig {
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
        if !self.head_dim()?.is_multiple_of(2) {
            return Err(Error::Unsupported(
                "Falcon native full-head NeoX RoPE requires an even head width".into(),
            ));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(Error::InvalidConfig(
                "Falcon RoPE base must be finite and positive".into(),
            ));
        }
        self.kv_width()?
            .checked_mul(self.max_context)
            .ok_or_else(|| Error::InvalidConfig("Falcon KV size overflow".into()))?;
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
            .ok_or_else(|| Error::InvalidConfig("Falcon QKV size overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub struct FalconLayer {
    /// Primary normalization feeds FFN and, without attention_norm_2, attention.
    pub attention_norm: Vec<f32>,
    pub attention_norm_bias: Vec<f32>,
    /// Falcon-40B uses its independent second normalization for attention.
    pub attention_norm_2: Option<Vec<f32>>,
    /// The source admits this independently; without the weight it is unused.
    pub attention_norm_2_bias: Option<Vec<f32>>,
    pub qkv: Matrix,
    pub attention_output: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}
impl FalconLayer {
    fn validate(&self, c: &FalconConfig) -> Result<()> {
        math::vector(&self.attention_norm, c.width, "Falcon primary norm")?;
        math::vector(
            &self.attention_norm_bias,
            c.width,
            "Falcon primary norm bias",
        )?;
        if let Some(weight) = &self.attention_norm_2 {
            math::vector(weight, c.width, "Falcon second norm")?;
        }
        if let Some(bias) = &self.attention_norm_2_bias {
            math::vector(bias, c.width, "Falcon second norm bias")?;
        }
        self.qkv.shape(c.qkv_width()?, c.width, "Falcon QKV")?;
        self.attention_output
            .shape(c.width, c.width, "Falcon attention output")?;
        self.up.shape(c.feed_forward, c.width, "Falcon FFN up")?;
        self.down
            .shape(c.width, c.feed_forward, "Falcon FFN down")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FalconWeights {
    pub token_embedding: Matrix,
    pub layers: Vec<FalconLayer>,
    pub output_norm: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    pub output: Option<Matrix>,
}
#[derive(Debug, Clone)]
pub struct FalconModel {
    config: FalconConfig,
    weights: FalconWeights,
    fingerprint: u64,
}
impl FalconModel {
    pub fn new(config: FalconConfig, weights: FalconWeights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "Falcon token embedding")?;
        math::vector(&weights.output_norm, config.width, "Falcon output norm")?;
        math::vector(
            &weights.output_norm_bias,
            config.width,
            "Falcon output norm bias",
        )?;
        if let Some(output) = &weights.output {
            output.shape(config.vocabulary, config.width, "Falcon output")?;
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
    pub fn config(&self) -> &FalconConfig {
        &self.config
    }
    pub fn weights(&self) -> &FalconWeights {
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
        math::vector(&hidden, c.width, "Falcon input embedding")?;
        let kv_width = c.kv_width()?;
        let expected = position
            .checked_mul(kv_width)
            .ok_or_else(|| Error::InvalidState("Falcon cache length overflow".into()))?;
        if cache.len() != self.layer_count()
            || cache
                .iter()
                .any(|layer| layer.keys.len() != expected || layer.values.len() != expected)
        {
            return Err(Error::InvalidState(
                "Falcon cache shape/position mismatch".into(),
            ));
        }
        let head_dim = c.head_dim()?;
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
            let second_norm = layer
                .attention_norm_2
                .as_ref()
                .map(|weight| {
                    math::layer_norm_optional_bias(
                        &hidden,
                        weight,
                        layer.attention_norm_2_bias.as_deref(),
                        c.norm_epsilon,
                    )
                })
                .transpose()?;
            let attention_input = second_norm.as_deref().unwrap_or(&normalized);
            let mut qkv = adapters.multiply(
                &format!("blk.{index}.attn_qkv.weight"),
                &layer.qkv,
                attention_input,
            )?;
            let (q, rest) = qkv.split_at_mut(c.width);
            let (k, v) = rest.split_at_mut(kv_width);
            math::rope_neox(q, head_dim, position, c.rope_base);
            math::rope_neox(k, head_dim, position, c.rope_base);
            kv.keys.extend_from_slice(k);
            kv.values.extend_from_slice(v);
            let attention = math::attention(
                q, &kv.keys, &kv.values, c.heads, c.kv_heads, head_dim, scale,
            )?;
            let projected = adapters.multiply(
                &format!("blk.{index}.attn_output.weight"),
                &layer.attention_output,
                &attention,
            )?;
            // Both paths consume the original layer input. The primary norm
            // always feeds FFN, even when attention uses the second norm.
            let mut up = adapters.multiply(
                &format!("blk.{index}.ffn_up.weight"),
                &layer.up,
                &normalized,
            )?;
            for value in &mut up {
                *value = math::gelu(*value);
            }
            let mut output =
                adapters.multiply(&format!("blk.{index}.ffn_down.weight"), &layer.down, &up)?;
            // Preserve source rounding: (FFN output + attention output) + input.
            math::add(&mut output, &projected);
            math::add(&mut output, &hidden);
            hidden = output;
            adapters.control(index, &mut hidden);
            math::finite(&hidden, "Falcon layer output")?;
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
        math::finite(&logits, "Falcon logits")?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.bytes(b"falcon-neox-parallel-affine-v1");
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
        hash.floats(&[c.norm_epsilon, c.rope_base]);
        hash.matrix(&self.weights.token_embedding);
        hash.floats(&self.weights.output_norm);
        hash.floats(&self.weights.output_norm_bias);
        hash.usize(usize::from(self.weights.output.is_some()));
        if let Some(output) = &self.weights.output {
            hash.matrix(output);
        }
        for layer in &self.weights.layers {
            hash.floats(&layer.attention_norm);
            hash.floats(&layer.attention_norm_bias);
            for optional in [&layer.attention_norm_2, &layer.attention_norm_2_bias] {
                hash.usize(usize::from(optional.is_some()));
                if let Some(value) = optional {
                    hash.floats(value);
                }
            }
            for matrix in [&layer.qkv, &layer.attention_output, &layer.up, &layer.down] {
                hash.matrix(matrix);
            }
        }
        hash.finish()
    }
}
