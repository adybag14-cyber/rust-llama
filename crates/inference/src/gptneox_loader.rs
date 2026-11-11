//! Dense GPT-NeoX GGUF loading with partial RoPE and explicit residual order.
use crate::loader::{
    details, float, matrix, require_causal, require_unscaled_rope, uint, unsupported, vector,
};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{GptNeoXConfig, GptNeoXLayer, GptNeoXModel, GptNeoXWeights};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, path::Path, sync::Arc};

pub(super) fn load(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = g.metadata();
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "gptneox.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    for key in [
        "gptneox.attention.head_count",
        "gptneox.attention.head_count_kv",
        "gptneox.feed_forward_length",
    ] {
        if matches!(metadata.get(key), Some(Value::Array { .. })) {
            return Err(unsupported(format!("per-layer array {key}")));
        }
    }
    if !metadata.contains_key("gptneox.attention.layer_norm_epsilon") {
        return Err(Error::Invalid(
            "missing gptneox.attention.layer_norm_epsilon".into(),
        ));
    }
    let heads = uint(metadata, "gptneox.attention.head_count", None)?;
    let train_context = uint(metadata, "gptneox.context_length", None)?;
    let parallel_residual = match metadata.get("gptneox.use_parallel_residual") {
        Some(Value::Bool(value)) => *value,
        None => {
            return Err(Error::Invalid(
                "missing gptneox.use_parallel_residual".into(),
            ));
        }
        Some(_) => {
            return Err(Error::Invalid(
                "gptneox.use_parallel_residual must be BOOL".into(),
            ));
        }
    };
    let width = uint(metadata, "gptneox.embedding_length", None)?;
    let config = GptNeoXConfig {
        vocabulary: tokenizer.vocabulary_size(),
        width,
        feed_forward: uint(metadata, "gptneox.feed_forward_length", None)?,
        heads,
        kv_heads: uint(metadata, "gptneox.attention.head_count_kv", Some(heads))?,
        rope_dimensions: uint(
            metadata,
            "gptneox.rope.dimension_count",
            Some(width.checked_div(heads).unwrap_or(0)),
        )?,
        rope_base: float(metadata, "gptneox.rope.freq_base", 10000.0)?,
        parallel_residual,
        max_context: train_context.min(options.context_length),
        norm_epsilon: float(metadata, "gptneox.attention.layer_norm_epsilon", 1e-5)?,
    };
    config.validate()?;
    for key in [
        "gptneox.attention.key_length",
        "gptneox.attention.value_length",
    ] {
        if uint(metadata, key, Some(config.head_dim()?))? != config.head_dim()? {
            return Err(unsupported(format!("nonstandard {key}")));
        }
    }
    for key in [
        "gptneox.attention.sliding_window",
        "gptneox.expert_count",
        "gptneox.expert_used_count",
        "gptneox.pooling_type",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "gptneox", true)?;
    require_unscaled_rope(metadata, "gptneox")?;
    // Native CPU ADD/MUL consume these vectors as src1 of F32 activations.
    // Low-precision active vectors abort in the source; return a checked error.
    for tensor in g.tensors() {
        if (tensor.name.ends_with(".bias") || tensor.name.ends_with("_norm.weight"))
            && tensor.dtype.id() != 0
        {
            return Err(unsupported(format!(
                "GPT-NeoX active norm/bias tensor {} requires F32 storage",
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
    let output = matrix(
        &mut g,
        &mut used,
        "output.weight",
        config.vocabulary,
        config.width,
    )?;
    let mut weights = Vec::with_capacity(layers);
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        weights.push(GptNeoXLayer {
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
    let model = GptNeoXModel::new(
        config,
        GptNeoXWeights {
            token_embedding,
            layers: weights,
            output_norm,
            output_norm_bias,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::GptNeoX(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
