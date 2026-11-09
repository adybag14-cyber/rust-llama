//! Native dense Qwen3 GGUF graph loading.
use crate::loader::{
    bias, details, float, matrix, require_causal, require_unscaled_rope, uint, unsupported, vector,
};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{Qwen3Config, Qwen3Layer, Qwen3Model, Qwen3Qkv, Qwen3Weights};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, path::Path, sync::Arc};

// Frozen GGML binary operations keep the F32 activation as src0. Their CPU
// mixed-type branches cannot consume a low-precision active norm/bias as src1.
// Validate these descriptors before decoding any weights. Partial fused bias
// groups are deliberately ignored by the original graph, so their stored types
// remain admissible and are decoded only for checked model ownership/identity.
fn require_f32_if_present(g: &GgufSet, name: &str) -> Result<()> {
    if let Some(tensor) = g.tensor(name)
        && tensor.dtype.id() != 0
    {
        return Err(unsupported(format!(
            "Qwen3 active norm/bias tensor {name} requires F32 storage"
        )));
    }
    Ok(())
}

fn validate_active_vector_storage(g: &GgufSet, layers: usize) -> Result<()> {
    require_f32_if_present(g, "output_norm.weight")?;
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        for name in [
            "attn_norm.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_norm.weight",
        ] {
            require_f32_if_present(g, &key(name))?;
        }
        let biases = [key("attn_q.bias"), key("attn_k.bias"), key("attn_v.bias")];
        if g.tensor(&key("attn_qkv.weight")).is_none() {
            for name in &biases {
                require_f32_if_present(g, name)?;
            }
        } else if g.tensor(&key("attn_qkv.bias")).is_some() {
            require_f32_if_present(g, &key("attn_qkv.bias"))?;
        } else if biases.iter().all(|name| g.tensor(name).is_some()) {
            for name in &biases {
                require_f32_if_present(g, name)?;
            }
        }
    }
    Ok(())
}

pub(super) fn load(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = g.metadata();
    match metadata.get("general.type") {
        None => {}
        Some(Value::String(kind)) if kind == "model" => {}
        Some(Value::String(kind)) => return Err(unsupported(format!("GGUF general.type={kind}"))),
        Some(_) => return Err(Error::Invalid("general.type must be a string".into())),
    }
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "qwen3.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    let train_context = uint(metadata, "qwen3.context_length", None)?;
    if !metadata.contains_key("qwen3.attention.layer_norm_rms_epsilon") {
        return Err(Error::Invalid(
            "missing qwen3.attention.layer_norm_rms_epsilon".into(),
        ));
    }
    for key in [
        "qwen3.attention.head_count",
        "qwen3.attention.head_count_kv",
        "qwen3.feed_forward_length",
    ] {
        if matches!(metadata.get(key), Some(Value::Array { .. })) {
            return Err(unsupported(format!("per-layer array {key}")));
        }
    }
    let heads = uint(metadata, "qwen3.attention.head_count", None)?;
    let width = uint(metadata, "qwen3.embedding_length", None)?;
    let default_head = width
        .checked_div(heads)
        .ok_or_else(|| Error::Invalid("qwen3 attention head count must be positive".into()))?;
    let key_length = uint(metadata, "qwen3.attention.key_length", Some(default_head))?;
    let value_length = uint(metadata, "qwen3.attention.value_length", Some(default_head))?;
    let config = Qwen3Config {
        vocabulary: tokenizer.vocabulary_size(),
        width,
        feed_forward: uint(metadata, "qwen3.feed_forward_length", None)?,
        heads,
        head_dim: key_length,
        kv_heads: uint(metadata, "qwen3.attention.head_count_kv", Some(heads))?,
        max_context: train_context.min(options.context_length),
        rms_epsilon: float(metadata, "qwen3.attention.layer_norm_rms_epsilon", 1e-5)?,
        rope_base: float(metadata, "qwen3.rope.freq_base", 10000.)?,
    };
    config.validate()?;
    let head_dim = config.head_dim()?;
    let kv_width = config.kv_width()?;
    let query_width = config.query_width()?;
    if value_length != head_dim {
        return Err(unsupported(
            "Qwen3 requires equal key and value head lengths",
        ));
    }
    if uint(metadata, "qwen3.rope.dimension_count", Some(head_dim))? != head_dim {
        return Err(unsupported("Qwen3 requires full-head NeoX RoPE"));
    }
    for key in [
        "qwen3.attention.sliding_window",
        "qwen3.expert_count",
        "qwen3.expert_used_count",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "qwen3", true)?;
    require_unscaled_rope(metadata, "qwen3")?;
    if uint(metadata, "qwen3.pooling_type", Some(0))? != 0 {
        return Err(unsupported("Qwen3 pooled embedding or reranker execution"));
    }
    for tensor in g.tensors() {
        if tensor.name.starts_with("cls.")
            || tensor.name.starts_with("dense_2.")
            || tensor.name == "output.bias"
        {
            return Err(unsupported(format!(
                "Qwen3 LM slice does not execute {}",
                tensor.name
            )));
        }
    }
    validate_active_vector_storage(&g, layers)?;
    let (id, template, decoded_weight_bytes) = details(&g, path, options)?;
    let mut used = BTreeSet::new();
    let token_embedding = matrix(
        &mut g,
        &mut used,
        "token_embd.weight",
        config.vocabulary,
        config.width,
    )?;
    let output_norm = vector(&mut g, &mut used, "output_norm.weight", config.width)?;
    let output = if g.tensor("output.weight").is_some() {
        Some(matrix(
            &mut g,
            &mut used,
            "output.weight",
            config.vocabulary,
            config.width,
        )?)
    } else {
        None
    };
    let mut weights = Vec::with_capacity(layers);
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        let qkv = if g.tensor(&key("attn_qkv.weight")).is_some() {
            let projection = matrix(
                &mut g,
                &mut used,
                &key("attn_qkv.weight"),
                config.qkv_width()?,
                config.width,
            )?;
            let fused_bias = bias(
                &mut g,
                &mut used,
                &key("attn_qkv.bias"),
                config.qkv_width()?,
            )?;
            let (query_bias, key_bias, value_bias) = if fused_bias.is_none() {
                (
                    bias(&mut g, &mut used, &key("attn_q.bias"), query_width)?,
                    bias(&mut g, &mut used, &key("attn_k.bias"), kv_width)?,
                    bias(&mut g, &mut used, &key("attn_v.bias"), kv_width)?,
                )
            } else {
                (None, None, None)
            };
            Qwen3Qkv::Fused {
                projection,
                bias: fused_bias,
                query_bias,
                key_bias,
                value_bias,
            }
        } else {
            Qwen3Qkv::Separate {
                query: matrix(
                    &mut g,
                    &mut used,
                    &key("attn_q.weight"),
                    query_width,
                    config.width,
                )?,
                key: matrix(
                    &mut g,
                    &mut used,
                    &key("attn_k.weight"),
                    kv_width,
                    config.width,
                )?,
                value: matrix(
                    &mut g,
                    &mut used,
                    &key("attn_v.weight"),
                    kv_width,
                    config.width,
                )?,
                query_bias: bias(&mut g, &mut used, &key("attn_q.bias"), query_width)?,
                key_bias: bias(&mut g, &mut used, &key("attn_k.bias"), kv_width)?,
                value_bias: bias(&mut g, &mut used, &key("attn_v.bias"), kv_width)?,
            }
        };
        weights.push(Qwen3Layer {
            attention_norm: vector(&mut g, &mut used, &key("attn_norm.weight"), config.width)?,
            qkv,
            attention_output: matrix(
                &mut g,
                &mut used,
                &key("attn_output.weight"),
                config.width,
                query_width,
            )?,
            query_norm: vector(&mut g, &mut used, &key("attn_q_norm.weight"), head_dim)?,
            key_norm: vector(&mut g, &mut used, &key("attn_k_norm.weight"), head_dim)?,
            feed_forward_norm: vector(&mut g, &mut used, &key("ffn_norm.weight"), config.width)?,
            gate: matrix(
                &mut g,
                &mut used,
                &key("ffn_gate.weight"),
                config.feed_forward,
                config.width,
            )?,
            up: matrix(
                &mut g,
                &mut used,
                &key("ffn_up.weight"),
                config.feed_forward,
                config.width,
            )?,
            down: matrix(
                &mut g,
                &mut used,
                &key("ffn_down.weight"),
                config.width,
                config.feed_forward,
            )?,
        });
    }
    let unconsumed: Vec<_> = g
        .tensors()
        .iter()
        .filter(|tensor| !used.contains(&tensor.name))
        .map(|tensor| tensor.name.as_str())
        .collect();
    if !unconsumed.is_empty() {
        return Err(unsupported(format!(
            "model contains unimplemented tensors: {}",
            unconsumed.join(", ")
        )));
    }
    let model = Qwen3Model::new(
        config,
        Qwen3Weights {
            token_embedding,
            layers: weights,
            output_norm,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Qwen3(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
