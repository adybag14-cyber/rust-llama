//! Falcon GGUF loading: full NeoX RoPE, fused QKV and parallel GELU FFN.
use crate::loader::{
    details, float, matrix, require_causal, require_unscaled_rope, uint, unsupported, vector,
};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{FalconConfig, FalconLayer, FalconModel, FalconWeights};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, path::Path, sync::Arc};

pub(super) fn load(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = g.metadata();
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "falcon.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    for key in [
        "falcon.attention.head_count",
        "falcon.attention.head_count_kv",
        "falcon.feed_forward_length",
    ] {
        if matches!(metadata.get(key), Some(Value::Array { .. })) {
            return Err(unsupported(format!("per-layer array {key}")));
        }
    }
    if !metadata.contains_key("falcon.attention.layer_norm_epsilon") {
        return Err(Error::Invalid(
            "missing falcon.attention.layer_norm_epsilon".into(),
        ));
    }
    let heads = uint(metadata, "falcon.attention.head_count", None)?;
    let train_context = uint(metadata, "falcon.context_length", None)?;
    let config = FalconConfig {
        vocabulary: tokenizer.vocabulary_size(),
        width: uint(metadata, "falcon.embedding_length", None)?,
        feed_forward: uint(metadata, "falcon.feed_forward_length", None)?,
        heads,
        kv_heads: uint(metadata, "falcon.attention.head_count_kv", Some(heads))?,
        max_context: train_context.min(options.context_length),
        norm_epsilon: float(metadata, "falcon.attention.layer_norm_epsilon", 1e-5)?,
        rope_base: float(metadata, "falcon.rope.freq_base", 10000.0)?,
    };
    config.validate()?;
    for key in [
        "falcon.attention.key_length",
        "falcon.attention.value_length",
        "falcon.rope.dimension_count",
    ] {
        if uint(metadata, key, Some(config.head_dim()?))? != config.head_dim()? {
            return Err(unsupported(format!(
                "nonstandard {key}; Falcon requires full-head RoPE"
            )));
        }
    }
    for key in [
        "falcon.attention.sliding_window",
        "falcon.expert_count",
        "falcon.expert_used_count",
        "falcon.pooling_type",
    ] {
        if uint(metadata, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(metadata, "falcon", true)?;
    require_unscaled_rope(metadata, "falcon")?;
    // Original CPU MUL/ADD require F32 active affine vectors. The independently
    // admitted second bias without a second weight is loaded but never applied.
    for tensor in g.tensors() {
        let active = match tensor.name.strip_suffix("attn_norm_2.bias") {
            Some(prefix) => g.tensor(&format!("{prefix}attn_norm_2.weight")).is_some(),
            None => {
                tensor.name.ends_with(".bias")
                    || tensor.name.ends_with("_norm.weight")
                    || tensor.name.ends_with("attn_norm_2.weight")
            }
        };
        if active && tensor.dtype.id() != 0 {
            return Err(unsupported(format!(
                "Falcon active norm/bias tensor {} requires F32 storage",
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
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(layers)
        .map_err(|_| Error::Invalid("Falcon layer descriptor allocation".into()))?;
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        let attention_norm = vector(&mut g, &mut used, &key("attn_norm.weight"), config.width)?;
        let attention_norm_bias = vector(&mut g, &mut used, &key("attn_norm.bias"), config.width)?;
        let attention_norm_2 =
            optional_vector(&mut g, &mut used, &key("attn_norm_2.weight"), config.width)?;
        let attention_norm_2_bias =
            optional_vector(&mut g, &mut used, &key("attn_norm_2.bias"), config.width)?;
        weights.push(FalconLayer {
            attention_norm,
            attention_norm_bias,
            attention_norm_2,
            attention_norm_2_bias,
            qkv: matrix(
                &mut g,
                &mut used,
                &key("attn_qkv.weight"),
                config.qkv_width()?,
                config.width,
            )?,
            attention_output: matrix(
                &mut g,
                &mut used,
                &key("attn_output.weight"),
                config.width,
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
        .filter(|t| !used.contains(&t.name))
        .map(|t| t.name.as_str())
        .collect();
    if !unconsumed.is_empty() {
        return Err(unsupported(format!(
            "model contains unimplemented tensors: {}",
            unconsumed.join(", ")
        )));
    }
    let model = FalconModel::new(
        config,
        FalconWeights {
            token_embedding,
            layers: weights,
            output_norm,
            output_norm_bias,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Falcon(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}

fn optional_vector(
    g: &mut GgufSet,
    used: &mut BTreeSet<String>,
    name: &str,
    width: usize,
) -> Result<Option<Vec<f32>>> {
    if g.tensor(name).is_some() {
        Ok(Some(vector(g, used, name, width)?))
    } else {
        Ok(None)
    }
}
