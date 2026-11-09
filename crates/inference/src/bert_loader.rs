//! Dense BERT GGUF loader, following the frozen src/models/bert.cpp names/shapes.
use crate::loader::{bias, details, float, matrix, uint, unsupported, vector};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{Gguf, Value};
use llama_model_runtime::{BertConfig, BertLayer, BertModel, BertWeights, Pooling};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, fs::File, path::Path, sync::Arc};

pub(crate) fn load(mut g: Gguf<File>, path: &Path, options: &LoadOptions) -> Result<Model> {
    let m = &g.metadata;
    if let Some(t) = m.get("general.type").and_then(Value::as_str)
        && t != "model"
    {
        return Err(unsupported(format!("GGUF general.type={t}")));
    }
    if uint(m, "split.count", Some(1))? > 1 {
        return Err(unsupported("split GGUF model loading"));
    }
    if m.get("tokenizer.ggml.model").and_then(Value::as_str) != Some("bert") {
        return Err(unsupported(
            "dense BERT currently requires its WordPiece vocabulary",
        ));
    }
    if m.get("bert.attention.causal")
        .is_some_and(|v| v.as_bool() != Some(false))
    {
        return Err(unsupported("causal BERT attention"));
    }
    let tokenizer = Tokenizer::from_metadata(m)?;
    let width = uint(m, "bert.embedding_length", None)?;
    let heads = uint(m, "bert.attention.head_count", None)?;
    let kv_heads = uint(m, "bert.attention.head_count_kv", Some(heads))?;
    let layers = uint(m, "bert.block_count", None)?;
    let train_context = uint(m, "bert.context_length", None)?;
    let token_types = uint(m, "tokenizer.ggml.token_type_count", None)?;
    if width == 0 || heads == 0 || !width.is_multiple_of(heads) {
        return Err(Error::Invalid(
            "invalid BERT embedding/head dimensions".into(),
        ));
    }
    if layers == 0 || layers > 4096 || token_types == 0 || train_context == 0 {
        return Err(Error::Invalid(
            "invalid BERT layer, token-type or context count".into(),
        ));
    }
    for key in [
        "bert.expert_count",
        "bert.expert_used_count",
        "bert.attention.sliding_window",
    ] {
        if uint(m, key, Some(0))? != 0 {
            return Err(unsupported(key));
        }
    }
    for key in ["bert.attention.key_length", "bert.attention.value_length"] {
        if uint(m, key, Some(width / heads))? != width / heads {
            return Err(unsupported(key));
        }
    }
    if !m.contains_key("bert.attention.layer_norm_epsilon") {
        return Err(Error::Invalid(
            "missing bert.attention.layer_norm_epsilon".into(),
        ));
    }
    let config = BertConfig {
        vocabulary: tokenizer.vocabulary_size(),
        width,
        feed_forward: uint(m, "bert.feed_forward_length", None)?,
        heads,
        kv_heads,
        // The complete position table has the training shape. The facade
        // enforces a smaller caller-requested context limit before encoding.
        max_positions: train_context,
        layer_norm_epsilon: float(m, "bert.attention.layer_norm_epsilon", 1e-12)?,
    };
    let embedding_pooling = match uint(m, "bert.pooling_type", Some(0))? {
        0 => Pooling::None,
        1 => Pooling::Mean,
        2 => Pooling::First,
        3 => Pooling::Last,
        value => return Err(unsupported(format!("BERT pooling type {value}"))),
    };
    let kv_width = config.kv_width()?;
    let (id, template, decoded_weight_bytes) = details(&g, path, options)?;
    let mut used = BTreeSet::new();
    let token_embedding = matrix(
        &mut g,
        &mut used,
        "token_embd.weight",
        config.vocabulary,
        width,
    )?;
    let position_embedding = matrix(
        &mut g,
        &mut used,
        "position_embd.weight",
        train_context,
        width,
    )?;
    let token_type_embedding = if g.tensor("token_types.weight").is_some() {
        Some(matrix(
            &mut g,
            &mut used,
            "token_types.weight",
            token_types,
            width,
        )?)
    } else {
        None
    };
    let embedding_norm = vector(&mut g, &mut used, "token_embd_norm.weight", width)?;
    let embedding_norm_bias = vector(&mut g, &mut used, "token_embd_norm.bias", width)?;
    let mut weights = Vec::with_capacity(layers);
    for i in 0..layers {
        let key = |s: &str| format!("blk.{i}.{s}");
        weights.push(BertLayer {
            query: matrix(&mut g, &mut used, &key("attn_q.weight"), width, width)?,
            key: matrix(&mut g, &mut used, &key("attn_k.weight"), kv_width, width)?,
            value: matrix(&mut g, &mut used, &key("attn_v.weight"), kv_width, width)?,
            query_bias: bias(&mut g, &mut used, &key("attn_q.bias"), width)?,
            key_bias: bias(&mut g, &mut used, &key("attn_k.bias"), kv_width)?,
            value_bias: bias(&mut g, &mut used, &key("attn_v.bias"), kv_width)?,
            attention_output: matrix(&mut g, &mut used, &key("attn_output.weight"), width, width)?,
            attention_output_bias: bias(&mut g, &mut used, &key("attn_output.bias"), width)?,
            attention_norm: vector(&mut g, &mut used, &key("attn_output_norm.weight"), width)?,
            attention_norm_bias: vector(&mut g, &mut used, &key("attn_output_norm.bias"), width)?,
            up: matrix(
                &mut g,
                &mut used,
                &key("ffn_up.weight"),
                config.feed_forward,
                width,
            )?,
            up_bias: bias(&mut g, &mut used, &key("ffn_up.bias"), config.feed_forward)?,
            down: matrix(
                &mut g,
                &mut used,
                &key("ffn_down.weight"),
                width,
                config.feed_forward,
            )?,
            down_bias: bias(&mut g, &mut used, &key("ffn_down.bias"), width)?,
            output_norm: vector(&mut g, &mut used, &key("layer_output_norm.weight"), width)?,
            output_norm_bias: vector(&mut g, &mut used, &key("layer_output_norm.bias"), width)?,
        });
    }
    let unused: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| !used.contains(&t.name))
        .map(|t| t.name.as_str())
        .collect();
    if !unused.is_empty() {
        return Err(unsupported(format!(
            "BERT model contains unimplemented tensors: {}",
            unused.join(", ")
        )));
    }
    let bert = BertModel::new(
        config,
        BertWeights {
            token_embedding,
            position_embedding,
            token_type_embedding,
            embedding_norm,
            embedding_norm_bias,
            layers: weights,
        },
    )?;
    Ok(Model {
        runtime: Executor::Bert(Arc::new(bert)),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: Some(embedding_pooling),
    })
}
