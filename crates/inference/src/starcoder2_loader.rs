//! Native dense Starcoder2 GGUF graph loading.
use crate::loader::{
    bias, details, float, matrix, require_causal, require_unscaled_rope, uint, unsupported, vector,
};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{
    Starcoder2Config, Starcoder2Layer, Starcoder2Model, Starcoder2Qkv, Starcoder2Weights,
};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, path::Path, sync::Arc};

pub(super) fn load(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = g.metadata();
    match metadata.get("general.type") {
        None => {}
        Some(Value::String(kind)) if kind == "model" => {}
        Some(Value::String(kind)) => return Err(unsupported(format!("GGUF general.type={kind}"))),
        Some(_) => return Err(Error::Invalid("general.type must be a string".into())),
    }
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "starcoder2.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    let train_context = uint(metadata, "starcoder2.context_length", None)?;
    if !metadata.contains_key("starcoder2.attention.layer_norm_epsilon") {
        return Err(Error::Invalid(
            "missing starcoder2.attention.layer_norm_epsilon".into(),
        ));
    }
    for key in [
        "starcoder2.attention.head_count",
        "starcoder2.attention.head_count_kv",
        "starcoder2.feed_forward_length",
    ] {
        if matches!(metadata.get(key), Some(Value::Array { .. })) {
            return Err(unsupported(format!("per-layer array {key}")));
        }
    }
    let heads = uint(metadata, "starcoder2.attention.head_count", None)?;
    let config = Starcoder2Config {
        vocabulary: tokenizer.vocabulary_size(),
        width: uint(metadata, "starcoder2.embedding_length", None)?,
        feed_forward: uint(metadata, "starcoder2.feed_forward_length", None)?,
        heads,
        kv_heads: uint(metadata, "starcoder2.attention.head_count_kv", Some(heads))?,
        max_context: train_context.min(options.context_length),
        norm_epsilon: float(metadata, "starcoder2.attention.layer_norm_epsilon", 1e-5)?,
        rope_base: float(metadata, "starcoder2.rope.freq_base", 10000.)?,
    };
    config.validate()?;
    let head_dim = config.head_dim()?;
    let kv_width = config.kv_width()?;
    for key in [
        "starcoder2.rope.dimension_count",
        "starcoder2.attention.key_length",
        "starcoder2.attention.value_length",
    ] {
        if uint(metadata, key, Some(head_dim))? != head_dim {
            return Err(unsupported(format!("nonstandard {key}")));
        }
    }
    for key in [
        "starcoder2.attention.sliding_window",
        "starcoder2.expert_count",
        "starcoder2.expert_used_count",
        "starcoder2.pooling_type",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "starcoder2", true)?;
    require_unscaled_rope(metadata, "starcoder2")?;
    for tensor in g.tensors() {
        let inactive_qkv_bias = tensor
            .name
            .strip_prefix("blk.")
            .and_then(|suffix| suffix.split_once('.'))
            .is_some_and(|(index, field)| {
                if !matches!(field, "attn_q.bias" | "attn_k.bias" | "attn_v.bias") {
                    return false;
                }
                let key = |suffix: &str| format!("blk.{index}.{suffix}");
                g.tensor(&key("attn_qkv.weight")).is_some()
                    && g.tensor(&key("attn_qkv.bias")).is_none()
                    && ["attn_q.bias", "attn_k.bias", "attn_v.bias"]
                        .iter()
                        .any(|n| g.tensor(&key(n)).is_none())
            });
        if !inactive_qkv_bias
            && (tensor.name.ends_with(".bias") || tensor.name.ends_with("_norm.weight"))
            && tensor.dtype.id() != 0
        {
            return Err(unsupported(format!(
                "StarCoder2 active norm/bias tensor {} requires F32 storage",
                tensor.name
            )));
        }
    }
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
    let output_norm_bias = vector(&mut g, &mut used, "output_norm.bias", config.width)?;
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
                    bias(&mut g, &mut used, &key("attn_q.bias"), config.width)?,
                    bias(&mut g, &mut used, &key("attn_k.bias"), kv_width)?,
                    bias(&mut g, &mut used, &key("attn_v.bias"), kv_width)?,
                )
            } else {
                (None, None, None)
            };
            Starcoder2Qkv::Fused {
                projection,
                bias: fused_bias,
                query_bias,
                key_bias,
                value_bias,
            }
        } else {
            Starcoder2Qkv::Separate {
                query: matrix(
                    &mut g,
                    &mut used,
                    &key("attn_q.weight"),
                    config.width,
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
                query_bias: bias(&mut g, &mut used, &key("attn_q.bias"), config.width)?,
                key_bias: bias(&mut g, &mut used, &key("attn_k.bias"), kv_width)?,
                value_bias: bias(&mut g, &mut used, &key("attn_v.bias"), kv_width)?,
            }
        };
        weights.push(Starcoder2Layer {
            attention_norm: vector(&mut g, &mut used, &key("attn_norm.weight"), config.width)?,
            attention_norm_bias: vector(&mut g, &mut used, &key("attn_norm.bias"), config.width)?,
            qkv,
            attention_output: matrix(
                &mut g,
                &mut used,
                &key("attn_output.weight"),
                config.width,
                config.width,
            )?,
            attention_output_bias: vector(
                &mut g,
                &mut used,
                &key("attn_output.bias"),
                config.width,
            )?,
            feed_forward_norm: vector(&mut g, &mut used, &key("ffn_norm.weight"), config.width)?,
            feed_forward_norm_bias: vector(&mut g, &mut used, &key("ffn_norm.bias"), config.width)?,
            up_bias: vector(&mut g, &mut used, &key("ffn_up.bias"), config.feed_forward)?,
            down_bias: vector(&mut g, &mut used, &key("ffn_down.bias"), config.width)?,

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
    let model = Starcoder2Model::new(
        config,
        Starcoder2Weights {
            token_embedding,
            layers: weights,
            output_norm,
            output_norm_bias,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Starcoder2(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
