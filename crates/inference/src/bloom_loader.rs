//! BLOOM GGUF loading: input affine normalization, ALiBi and fused Q/K/V.
use crate::loader::{details, float, matrix, require_causal, uint, unsupported, vector};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{BloomConfig, BloomLayer, BloomModel, BloomWeights};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, path::Path, sync::Arc};

pub(super) fn load(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = g.metadata();
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "bloom.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    for key in [
        "bloom.attention.head_count",
        "bloom.attention.head_count_kv",
        "bloom.feed_forward_length",
    ] {
        if matches!(metadata.get(key), Some(Value::Array { .. })) {
            return Err(unsupported(format!("per-layer array {key}")));
        }
    }
    if !metadata.contains_key("bloom.attention.layer_norm_epsilon") {
        return Err(Error::Invalid(
            "missing bloom.attention.layer_norm_epsilon".into(),
        ));
    }
    let heads = uint(metadata, "bloom.attention.head_count", None)?;
    let train_context = uint(metadata, "bloom.context_length", None)?;
    let config = BloomConfig {
        vocabulary: tokenizer.vocabulary_size(),
        width: uint(metadata, "bloom.embedding_length", None)?,
        feed_forward: uint(metadata, "bloom.feed_forward_length", None)?,
        heads,
        kv_heads: uint(metadata, "bloom.attention.head_count_kv", Some(heads))?,
        max_context: train_context.min(options.context_length),
        norm_epsilon: float(metadata, "bloom.attention.layer_norm_epsilon", 1e-5)?,
    };
    config.validate()?;
    for key in ["bloom.attention.key_length", "bloom.attention.value_length"] {
        if uint(metadata, key, Some(config.head_dim()?))? != config.head_dim()? {
            return Err(unsupported(format!("nonstandard {key}")));
        }
    }
    for key in [
        "bloom.attention.sliding_window",
        "bloom.expert_count",
        "bloom.expert_used_count",
        "bloom.pooling_type",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "bloom", true)?;
    // Native CPU ADD/MUL consume these vectors as src1 of F32 activations.
    // Low-precision active vectors abort in the source; return a checked error.
    for tensor in g.tensors() {
        if (tensor.name.ends_with(".bias") || tensor.name.ends_with("_norm.weight"))
            && tensor.dtype.id() != 0
        {
            return Err(unsupported(format!(
                "BLOOM active norm/bias tensor {} requires F32 storage",
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
    let embedding_norm = vector(&mut g, &mut used, "token_embd_norm.weight", config.width)?;
    let embedding_norm_bias = vector(&mut g, &mut used, "token_embd_norm.bias", config.width)?;
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
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(layers)
        .map_err(|_| Error::Invalid("BLOOM layer descriptor allocation".into()))?;
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        weights.push(BloomLayer {
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
    let model = BloomModel::new(
        config,
        BloomWeights {
            token_embedding,
            embedding_norm,
            embedding_norm_bias,
            layers: weights,
            output_norm,
            output_norm_bias,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Bloom(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
