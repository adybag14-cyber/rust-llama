//! Dense F32 Llama graph slice translated from `src/models/llama.cpp`.
use crate::context::NativeKvType;
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

/// Per-token projection before cache storage conversion. Keeping this boundary
/// separate lets native microbatches publish every current-layer K/V row before
/// any query attends, while the strict sequential path retains its order.
pub(crate) struct LayerProjection {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
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
        if name == "token_embd.weight" {
            return Some(&self.weights.token_embedding);
        }
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
    pub(crate) fn step(
        &self,
        hidden: Vec<f32>,
        position: usize,
        cache: &mut [KvLayer],
        adapters: &AdapterSet,
        cancel: &mut impl FnMut() -> bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let width = self.config.kv_width()?;
        let insertion = cache.first().map_or(0, |layer| layer.keys.len() / width);
        let layout = math::AttentionLayout::contiguous_padded(
            insertion
                .checked_add(1)
                .ok_or_else(|| Error::InvalidState("KV row count overflow".into()))?,
        )?;
        self.step_inserting(
            hidden,
            position,
            cache,
            insertion,
            layout,
            (NativeKvType::F32, NativeKvType::F32),
            adapters,
            cancel,
        )
    }
    // A physical native KV hole can precede already populated cells. Preserve
    // source accumulation order by inserting the new row at that exact index.
    #[allow(clippy::too_many_arguments)] // Carries explicit physical softmax columns with the insertion.
    pub(crate) fn step_inserting(
        &self,
        mut hidden: Vec<f32>,
        position: usize,
        cache: &mut [KvLayer],
        insertion: usize,
        layout: math::AttentionLayout<'_>,
        cache_types: (NativeKvType, NativeKvType),
        adapters: &AdapterSet,
        cancel: &mut impl FnMut() -> bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        let width = c.kv_width()?;
        let offset = insertion
            .checked_mul(width)
            .ok_or_else(|| Error::InvalidState("KV insertion offset overflow".into()))?;
        if cache.len() != self.layer_count()
            || cache.iter().any(|layer| {
                layer.keys.len() != layer.values.len()
                    || !layer.keys.len().is_multiple_of(width)
                    || offset > layer.keys.len()
            })
        {
            return Err(Error::InvalidState("invalid KV insertion shape".into()));
        }
        let head_dim = c.head_dim()?;
        let scale = c
            .attention_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        for (index, kv) in cache.iter_mut().enumerate() {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let LayerProjection {
                query: q,
                key: mut k,
                value: mut v,
            } = self.project_layer(index, &hidden, position, adapters)?;
            // Native cache storage rounds at the original graph's copy into
            // K/V tensors. Nonflash Q and probability operands stay F32.
            k.iter_mut()
                .for_each(|value| *value = cache_types.0.round(*value));
            v.iter_mut()
                .for_each(|value| *value = cache_types.1.round(*value));
            kv.keys.splice(offset..offset, k);
            kv.values.splice(offset..offset, v);
            let attention = math::attention_with_layout(
                &q,
                &kv.keys,
                &kv.values,
                c.heads,
                c.kv_heads,
                c.head_dim()?,
                scale,
                layout,
            )?;
            hidden = self.finish_layer(index, hidden, &attention, adapters)?;
        }
        self.finish_hidden(&hidden, adapters)
    }

    pub(crate) fn project_layer(
        &self,
        index: usize,
        hidden: &[f32],
        position: usize,
        adapters: &AdapterSet,
    ) -> Result<LayerProjection> {
        let c = &self.config;
        let layer = self.weights.layers.get(index).ok_or_else(|| {
            Error::InvalidState("Llama layer projection index is outside the model".into())
        })?;
        let norm = math::rms_norm(hidden, &layer.attention_norm, c.rms_epsilon)?;
        let mut q =
            adapters.multiply(&format!("blk.{index}.attn_q.weight"), &layer.query, &norm)?;
        let mut k = adapters.multiply(&format!("blk.{index}.attn_k.weight"), &layer.key, &norm)?;
        let mut v =
            adapters.multiply(&format!("blk.{index}.attn_v.weight"), &layer.value, &norm)?;
        math::bias(&mut q, &layer.query_bias);
        math::bias(&mut k, &layer.key_bias);
        math::bias(&mut v, &layer.value_bias);
        math::rope(&mut q, c.head_dim()?, position, c.rope_base);
        math::rope(&mut k, c.head_dim()?, position, c.rope_base);
        Ok(LayerProjection {
            query: q,
            key: k,
            value: v,
        })
    }

    pub(crate) fn finish_layer(
        &self,
        index: usize,
        mut hidden: Vec<f32>,
        attention: &[f32],
        adapters: &AdapterSet,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let layer = self.weights.layers.get(index).ok_or_else(|| {
            Error::InvalidState("Llama layer finish index is outside the model".into())
        })?;
        let mut projected = adapters.multiply(
            &format!("blk.{index}.attn_output.weight"),
            &layer.attention_output,
            attention,
        )?;
        math::bias(&mut projected, &layer.attention_output_bias);
        math::add(&mut hidden, &projected);
        let norm = math::rms_norm(&hidden, &layer.feed_forward_norm, c.rms_epsilon)?;
        let mut gate =
            adapters.multiply(&format!("blk.{index}.ffn_gate.weight"), &layer.gate, &norm)?;
        let mut up = adapters.multiply(&format!("blk.{index}.ffn_up.weight"), &layer.up, &norm)?;
        math::bias(&mut gate, &layer.gate_bias);
        math::bias(&mut up, &layer.up_bias);
        math::native_rows::silu_f32_in_place(&mut gate);
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate *= up;
        }
        let mut ffn =
            adapters.multiply(&format!("blk.{index}.ffn_down.weight"), &layer.down, &gate)?;
        math::bias(&mut ffn, &layer.down_bias);
        math::add(&mut hidden, &ffn);
        adapters.control(index, &mut hidden);
        math::finite(&hidden, "Llama layer output")?;
        Ok(hidden)
    }

    pub(crate) fn finish_hidden(
        &self,
        hidden: &[f32],
        adapters: &AdapterSet,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        let embedding = math::rms_norm(hidden, &self.weights.output_norm, c.rms_epsilon)?;
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
