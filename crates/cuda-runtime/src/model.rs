//! Borrowed, checked views of the executable dense model families. These views
//! retain packed matrix representations and never clone or dequantize weights.
use crate::{Error, Result, product};
use llama_model_runtime::{LlamaModel, Matrix, Qwen2Model, Qwen2Qkv, Qwen3Model, Qwen3Qkv};

#[derive(Clone)]
pub(crate) struct Config {
    pub vocabulary: usize,
    pub width: usize,
    pub query_width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub max_context: usize,
    pub rms_epsilon: f32,
    pub rope_base: f32,
    pub attention_scale: Option<f32>,
    pub neox: bool,
}
impl Config {
    pub fn kv_width(&self) -> Result<usize> {
        product(self.kv_heads, self.head_dimension)
    }
    pub fn qkv_width(&self) -> Result<usize> {
        product(self.kv_width()?, 2)?
            .checked_add(self.query_width)
            .ok_or_else(|| Error::Invalid("fused projection width overflow".into()))
    }
    pub fn activation_width(&self) -> usize {
        self.width.max(self.query_width).max(self.feed_forward)
    }
}

pub(crate) enum Qkv<'a> {
    Separate {
        query: &'a Matrix,
        key: &'a Matrix,
        value: &'a Matrix,
        query_bias: Option<&'a [f32]>,
        key_bias: Option<&'a [f32]>,
        value_bias: Option<&'a [f32]>,
    },
    Fused {
        projection: &'a Matrix,
        bias: Option<&'a [f32]>,
        query_bias: Option<&'a [f32]>,
        key_bias: Option<&'a [f32]>,
        value_bias: Option<&'a [f32]>,
    },
}
impl<'a> Qkv<'a> {
    pub fn matrices(&self) -> Vec<&'a Matrix> {
        match self {
            Self::Separate {
                query, key, value, ..
            } => vec![query, key, value],
            Self::Fused { projection, .. } => vec![projection],
        }
    }
    pub fn biases(&self) -> Vec<&'a [f32]> {
        let values = match self {
            Self::Separate {
                query_bias,
                key_bias,
                value_bias,
                ..
            } => [None, *query_bias, *key_bias, *value_bias],
            Self::Fused {
                bias,
                query_bias,
                key_bias,
                value_bias,
                ..
            } => [*bias, *query_bias, *key_bias, *value_bias],
        };
        values.into_iter().flatten().collect()
    }
}
pub(crate) struct Layer<'a> {
    pub attention_norm: &'a [f32],
    pub qkv: Qkv<'a>,
    pub attention_output: &'a Matrix,
    pub attention_output_bias: Option<&'a [f32]>,
    pub query_norm: Option<&'a [f32]>,
    pub key_norm: Option<&'a [f32]>,
    pub feed_forward_norm: &'a [f32],
    pub gate: &'a Matrix,
    pub up: &'a Matrix,
    pub down: &'a Matrix,
    pub gate_bias: Option<&'a [f32]>,
    pub up_bias: Option<&'a [f32]>,
    pub down_bias: Option<&'a [f32]>,
}
pub(crate) struct View<'a> {
    pub config: Config,
    pub token_embedding: &'a Matrix,
    pub output: Option<&'a Matrix>,
    pub output_norm: &'a [f32],
    pub output_bias: Option<&'a [f32]>,
    pub layers: Vec<Layer<'a>>,
}
impl View<'_> {
    pub fn fused(&self) -> bool {
        self.layers
            .iter()
            .any(|l| matches!(l.qkv, Qkv::Fused { .. }))
    }
    pub fn head_norm(&self) -> bool {
        self.layers.iter().any(|l| l.query_norm.is_some())
    }
}
fn runtime_error(error: llama_model_runtime::Error) -> Error {
    Error::Invalid(error.to_string())
}
impl<'a> View<'a> {
    pub fn llama(model: &'a LlamaModel) -> Result<Self> {
        let c = model.config();
        let w = model.weights();
        Ok(Self {
            config: Config {
                vocabulary: c.vocabulary,
                width: c.width,
                query_width: c.width,
                feed_forward: c.feed_forward,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dimension: c.head_dim().map_err(runtime_error)?,
                max_context: c.max_context,
                rms_epsilon: c.rms_epsilon,
                rope_base: c.rope_base,
                attention_scale: c.attention_scale,
                neox: false,
            },
            token_embedding: &w.token_embedding,
            output: w.output.as_ref(),
            output_norm: &w.output_norm,
            output_bias: None,
            layers: w
                .layers
                .iter()
                .map(|l| Layer {
                    attention_norm: &l.attention_norm,
                    qkv: Qkv::Separate {
                        query: &l.query,
                        key: &l.key,
                        value: &l.value,
                        query_bias: l.query_bias.as_deref(),
                        key_bias: l.key_bias.as_deref(),
                        value_bias: l.value_bias.as_deref(),
                    },
                    attention_output: &l.attention_output,
                    attention_output_bias: l.attention_output_bias.as_deref(),
                    query_norm: None,
                    key_norm: None,
                    feed_forward_norm: &l.feed_forward_norm,
                    gate: &l.gate,
                    up: &l.up,
                    down: &l.down,
                    gate_bias: l.gate_bias.as_deref(),
                    up_bias: l.up_bias.as_deref(),
                    down_bias: l.down_bias.as_deref(),
                })
                .collect(),
        })
    }
    pub fn qwen2(model: &'a Qwen2Model) -> Result<Self> {
        let c = model.config();
        let w = model.weights();
        Ok(Self {
            config: Config {
                vocabulary: c.vocabulary,
                width: c.width,
                query_width: c.width,
                feed_forward: c.feed_forward,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dimension: c.head_dim().map_err(runtime_error)?,
                max_context: c.max_context,
                rms_epsilon: c.rms_epsilon,
                rope_base: c.rope_base,
                attention_scale: None,
                neox: true,
            },
            token_embedding: &w.token_embedding,
            output: w.output.as_ref(),
            output_norm: &w.output_norm,
            output_bias: w.output_bias.as_deref(),
            layers: w
                .layers
                .iter()
                .map(|l| Layer {
                    attention_norm: &l.attention_norm,
                    qkv: match &l.qkv {
                        Qwen2Qkv::Separate {
                            query,
                            key,
                            value,
                            query_bias,
                            key_bias,
                            value_bias,
                        } => Qkv::Separate {
                            query,
                            key,
                            value,
                            query_bias: query_bias.as_deref(),
                            key_bias: key_bias.as_deref(),
                            value_bias: value_bias.as_deref(),
                        },
                        Qwen2Qkv::Fused {
                            projection,
                            bias,
                            query_bias,
                            key_bias,
                            value_bias,
                        } => {
                            // The frozen fused graph enables separate biases only
                            // as a complete group, and gives the fused bias priority.
                            let separate = bias.is_none()
                                && query_bias.is_some()
                                && key_bias.is_some()
                                && value_bias.is_some();
                            Qkv::Fused {
                                projection,
                                bias: bias.as_deref(),
                                query_bias: query_bias.as_deref().filter(|_| separate),
                                key_bias: key_bias.as_deref().filter(|_| separate),
                                value_bias: value_bias.as_deref().filter(|_| separate),
                            }
                        }
                    },
                    attention_output: &l.attention_output,
                    attention_output_bias: None,
                    query_norm: None,
                    key_norm: None,
                    feed_forward_norm: &l.feed_forward_norm,
                    gate: &l.gate,
                    up: &l.up,
                    down: &l.down,
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                })
                .collect(),
        })
    }
    pub fn qwen3(model: &'a Qwen3Model) -> Result<Self> {
        let c = model.config();
        let w = model.weights();
        Ok(Self {
            config: Config {
                vocabulary: c.vocabulary,
                width: c.width,
                query_width: c.query_width().map_err(runtime_error)?,
                feed_forward: c.feed_forward,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dimension: c.head_dim().map_err(runtime_error)?,
                max_context: c.max_context,
                rms_epsilon: c.rms_epsilon,
                rope_base: c.rope_base,
                attention_scale: None,
                neox: true,
            },
            token_embedding: &w.token_embedding,
            output: w.output.as_ref(),
            output_norm: &w.output_norm,
            output_bias: None,
            layers: w
                .layers
                .iter()
                .map(|l| Layer {
                    attention_norm: &l.attention_norm,
                    qkv: match &l.qkv {
                        Qwen3Qkv::Separate {
                            query,
                            key,
                            value,
                            query_bias,
                            key_bias,
                            value_bias,
                        } => Qkv::Separate {
                            query,
                            key,
                            value,
                            query_bias: query_bias.as_deref(),
                            key_bias: key_bias.as_deref(),
                            value_bias: value_bias.as_deref(),
                        },
                        Qwen3Qkv::Fused {
                            projection,
                            bias,
                            query_bias,
                            key_bias,
                            value_bias,
                        } => {
                            let separate = bias.is_none()
                                && query_bias.is_some()
                                && key_bias.is_some()
                                && value_bias.is_some();
                            Qkv::Fused {
                                projection,
                                bias: bias.as_deref(),
                                query_bias: query_bias.as_deref().filter(|_| separate),
                                key_bias: key_bias.as_deref().filter(|_| separate),
                                value_bias: value_bias.as_deref().filter(|_| separate),
                            }
                        }
                    },
                    attention_output: &l.attention_output,
                    attention_output_bias: None,
                    query_norm: Some(&l.query_norm),
                    key_norm: Some(&l.key_norm),
                    feed_forward_norm: &l.feed_forward_norm,
                    gate: &l.gate,
                    up: &l.up,
                    down: &l.down,
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                })
                .collect(),
        })
    }
}
