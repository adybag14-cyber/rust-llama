//! Native dense Qwen2 GGUF graph loading.
use crate::loader::{
    bias, details, float, matrix, require_causal, require_unscaled_rope, split_count, uint,
    unsupported, vector,
};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{Gguf, Value};
use llama_model_runtime::{Qwen2Config, Qwen2Layer, Qwen2Model, Qwen2Qkv, Qwen2Weights};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, fs::File, path::Path, sync::Arc};

pub(super) fn load(mut g: Gguf<File>, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = &g.metadata;
    match metadata.get("general.type") {
        None => {}
        Some(Value::String(kind)) if kind == "model" => {}
        Some(Value::String(kind)) => return Err(unsupported(format!("GGUF general.type={kind}"))),
        Some(_) => return Err(Error::Invalid("general.type must be a string".into())),
    }
    let splits = split_count(metadata)?;
    if splits > 1 {
        return Err(unsupported("split GGUF model loading"));
    }
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "qwen2.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    let train_context = uint(metadata, "qwen2.context_length", None)?;
    if !metadata.contains_key("qwen2.attention.layer_norm_rms_epsilon") {
        return Err(Error::Invalid(
            "missing qwen2.attention.layer_norm_rms_epsilon".into(),
        ));
    }
    let heads = uint(metadata, "qwen2.attention.head_count", None)?;
    let config = Qwen2Config {
        vocabulary: tokenizer.vocabulary_size(),
        width: uint(metadata, "qwen2.embedding_length", None)?,
        feed_forward: uint(metadata, "qwen2.feed_forward_length", None)?,
        heads,
        kv_heads: uint(metadata, "qwen2.attention.head_count_kv", Some(heads))?,
        max_context: train_context.min(options.context_length),
        rms_epsilon: float(metadata, "qwen2.attention.layer_norm_rms_epsilon", 1e-5)?,
        rope_base: float(metadata, "qwen2.rope.freq_base", 10000.)?,
    };
    config.validate()?;
    let head_dim = config.head_dim()?;
    let kv_width = config.kv_width()?;
    for key in [
        "qwen2.rope.dimension_count",
        "qwen2.attention.key_length",
        "qwen2.attention.value_length",
    ] {
        if uint(metadata, key, Some(head_dim))? != head_dim {
            return Err(unsupported(format!("nonstandard {key}")));
        }
    }
    for key in [
        "qwen2.attention.sliding_window",
        "qwen2.expert_count",
        "qwen2.expert_used_count",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "qwen2", true)?;
    require_unscaled_rope(metadata, "qwen2")?;
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
    let output_bias = bias(&mut g, &mut used, "output.bias", config.vocabulary)?;
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
            Qwen2Qkv::Fused {
                projection,
                bias: fused_bias,
                query_bias,
                key_bias,
                value_bias,
            }
        } else {
            Qwen2Qkv::Separate {
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
        weights.push(Qwen2Layer {
            attention_norm: vector(&mut g, &mut used, &key("attn_norm.weight"), config.width)?,
            qkv,
            attention_output: matrix(
                &mut g,
                &mut used,
                &key("attn_output.weight"),
                config.width,
                config.width,
            )?,
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
        .tensors
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
    let model = Qwen2Model::new(
        config,
        Qwen2Weights {
            token_embedding,
            layers: weights,
            output_norm,
            output,
            output_bias,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Qwen2(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
