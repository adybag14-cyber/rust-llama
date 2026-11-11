//! GPT-2 GGUF loading: learned positions, fused projection and affine vectors.
use crate::loader::{bias, details, float, matrix, require_causal, uint, unsupported, vector};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{Gpt2Config, Gpt2Layer, Gpt2Model, Gpt2Weights};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, path::Path, sync::Arc};

pub(super) fn load(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = g.metadata();
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "gpt2.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    for key in [
        "gpt2.attention.head_count",
        "gpt2.attention.head_count_kv",
        "gpt2.feed_forward_length",
    ] {
        if matches!(metadata.get(key), Some(Value::Array { .. })) {
            return Err(unsupported(format!("per-layer array {key}")));
        }
    }
    if !metadata.contains_key("gpt2.attention.layer_norm_epsilon") {
        return Err(Error::Invalid(
            "missing gpt2.attention.layer_norm_epsilon".into(),
        ));
    }
    let heads = uint(metadata, "gpt2.attention.head_count", None)?;
    let train_context = uint(metadata, "gpt2.context_length", None)?;
    let config = Gpt2Config {
        vocabulary: tokenizer.vocabulary_size(),
        width: uint(metadata, "gpt2.embedding_length", None)?,
        feed_forward: uint(metadata, "gpt2.feed_forward_length", None)?,
        heads,
        kv_heads: uint(metadata, "gpt2.attention.head_count_kv", Some(heads))?,
        positions: train_context,
        max_context: train_context.min(options.context_length),
        norm_epsilon: float(metadata, "gpt2.attention.layer_norm_epsilon", 1e-5)?,
    };
    config.validate()?;
    for key in ["gpt2.attention.key_length", "gpt2.attention.value_length"] {
        if uint(metadata, key, Some(config.head_dim()?))? != config.head_dim()? {
            return Err(unsupported(format!("nonstandard {key}")));
        }
    }
    for key in [
        "gpt2.attention.sliding_window",
        "gpt2.expert_count",
        "gpt2.expert_used_count",
        "gpt2.pooling_type",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "gpt2", true)?;
    // Native CPU ADD/MUL consume these vectors as src1 of F32 activations.
    // Low-precision active vectors abort in the source; return a checked error.
    for tensor in g.tensors() {
        if (tensor.name.ends_with(".bias") || tensor.name.ends_with("_norm.weight"))
            && tensor.dtype.id() != 0
        {
            return Err(unsupported(format!(
                "GPT-2 active norm/bias tensor {} requires F32 storage",
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
    let position_embedding = matrix(
        &mut g,
        &mut used,
        "position_embd.weight",
        config.positions,
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
    let output_bias = bias(&mut g, &mut used, "output.bias", config.vocabulary)?;
    let mut weights = Vec::with_capacity(layers);
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        weights.push(Gpt2Layer {
            attention_norm: vector(&mut g, &mut used, &key("attn_norm.weight"), config.width)?,
            attention_norm_bias: vector(&mut g, &mut used, &key("attn_norm.bias"), config.width)?,
            qkv: matrix(
                &mut g,
                &mut used,
                &key("attn_qkv.weight"),
                config.qkv_width()?,
                config.width,
            )?,
            qkv_bias: vector(
                &mut g,
                &mut used,
                &key("attn_qkv.bias"),
                config.qkv_width()?,
            )?,
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
            up: matrix(
                &mut g,
                &mut used,
                &key("ffn_up.weight"),
                config.feed_forward,
                config.width,
            )?,
            up_bias: vector(&mut g, &mut used, &key("ffn_up.bias"), config.feed_forward)?,
            down: matrix(
                &mut g,
                &mut used,
                &key("ffn_down.weight"),
                config.width,
                config.feed_forward,
            )?,
            down_bias: vector(&mut g, &mut used, &key("ffn_down.bias"), config.width)?,
        });
    }
    let unconsumed: Vec<_> = g
        .tensors()
        .iter()
        .filter(|t| !used.contains(&t.name))
        .map(|t| t.name.as_str())
        .collect();
    if !unconsumed.is_empty() {
        return Err(unsupported(format!(
            "model contains unimplemented tensors: {}",
            unconsumed.join(", ")
        )));
    }
    let model = Gpt2Model::new(
        config,
        Gpt2Weights {
            token_embedding,
            position_embedding,
            layers: weights,
            output_norm,
            output_norm_bias,
            output,
            output_bias,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Gpt2(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
