use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{GgufSet, Value};
use llama_model_runtime::{LlamaConfig, LlamaLayer, LlamaModel, LlamaWeights, Matrix};
use llama_tokenizer::Tokenizer;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

/// Frozen model hparams use exact GGUF UINT32 values, not arbitrary integers.
pub(super) fn uint(
    m: &BTreeMap<String, Value>,
    key: &str,
    default: Option<usize>,
) -> Result<usize> {
    match m.get(key) {
        Some(Value::U32(value)) => Ok(*value as usize),
        Some(_) => Err(Error::Invalid(format!("{key} must have GGUF UINT32 type"))),
        None => default.ok_or_else(|| Error::Invalid(format!("missing {key}"))),
    }
}
pub(super) fn float(m: &BTreeMap<String, Value>, key: &str, default: f32) -> Result<f32> {
    match m.get(key) {
        None => Ok(default),
        Some(Value::F32(value)) if value.is_finite() => Ok(*value),
        Some(Value::F32(_)) => Err(Error::Invalid(format!("{key} must be finite"))),
        Some(_) => Err(Error::Invalid(format!("{key} must have GGUF FLOAT32 type"))),
    }
}
pub(super) fn require_causal(
    m: &BTreeMap<String, Value>,
    architecture: &str,
    causal: bool,
) -> Result<()> {
    let key = format!("{architecture}.attention.causal");
    match m.get(&key) {
        None => Ok(()),
        Some(Value::Bool(value)) if *value == causal => Ok(()),
        Some(Value::Bool(_)) => Err(unsupported(format!("nondefault {key}"))),
        Some(_) => Err(Error::Invalid(format!("{key} must be a boolean"))),
    }
}
/// Resolve the source's new/legacy frequency-factor precedence, including its
/// zero default and explicit NONE override. Gain and scaled modes stay explicit.
pub(super) fn require_unscaled_rope(m: &BTreeMap<String, Value>, architecture: &str) -> Result<()> {
    let key = format!("{architecture}.rope.scaling.type");
    let kind = match m.get(&key) {
        None => "linear",
        Some(Value::String(value)) => value.as_str(),
        Some(_) => return Err(Error::Invalid(format!("{key} must be a string"))),
    };
    if !matches!(kind, "none" | "linear") {
        return Err(unsupported(format!(
            "{architecture} RoPE scaling mode {kind}"
        )));
    }
    let new_key = format!("{architecture}.rope.scaling.factor");
    let old_key = format!("{architecture}.rope.scale_linear");
    let factor = if m.contains_key(&new_key) {
        float(m, &new_key, 0.0)?
    } else {
        float(m, &old_key, 0.0)?
    };
    if kind != "none" && factor != 0.0 && factor != 1.0 {
        return Err(unsupported(format!(
            "{architecture} scaled RoPE frequencies"
        )));
    }
    if float(m, &format!("{architecture}.rope.scaling.attn_factor"), 1.0)? != 1.0 {
        return Err(unsupported(format!("{architecture} RoPE attention gain")));
    }
    Ok(())
}
fn shared_metadata(m: &BTreeMap<String, Value>, architecture: &str) -> Result<()> {
    match m.get("general.type") {
        None => {}
        Some(Value::String(kind)) if kind == "model" => {}
        Some(Value::String(kind)) => return Err(unsupported(format!("GGUF general.type={kind}"))),
        Some(_) => return Err(Error::Invalid("general.type must be a string".into())),
    }
    if uint(m, &format!("{architecture}.nextn_predict_layers"), Some(0))? != 0 {
        return Err(unsupported(format!("{architecture} NextN/MTP layers")));
    }
    if m.contains_key("prism.hadamard.version") {
        return Err(unsupported("PRISM Hadamard model transforms"));
    }
    Ok(())
}
pub(super) fn unsupported(s: impl Into<String>) -> Error {
    Error::Unsupported(s.into())
}

// GGML tensor shapes have four dimensions, with unspecified trailing axes
// equal to one. The source splitter may canonicalize those serialized axes.
fn same_shape(actual: &[u64], expected: &[u64]) -> bool {
    actual.len() <= 4
        && expected.len() <= 4
        && (0..4).all(|axis| {
            actual.get(axis).copied().unwrap_or(1) == expected.get(axis).copied().unwrap_or(1)
        })
}

pub(super) fn vector(
    g: &mut GgufSet,
    used: &mut BTreeSet<String>,
    name: &str,
    n: usize,
) -> Result<Vec<f32>> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))?;
    if !same_shape(&t.dimensions, &[n as u64]) {
        return Err(Error::Invalid(format!(
            "tensor {name} expected shape [{n}], found {:?}",
            t.dimensions
        )));
    }
    used.insert(name.into());
    Ok(g.read_f32(name)?)
}
pub(super) fn matrix(
    g: &mut GgufSet,
    used: &mut BTreeSet<String>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))?;
    if !same_shape(&t.dimensions, &[cols as u64, rows as u64]) {
        return Err(Error::Invalid(format!(
            "tensor {name} expected GGML shape [{cols},{rows}], found {:?}",
            t.dimensions
        )));
    }
    let dtype = t.dtype;
    used.insert(name.into());
    if dtype.id() == 0 {
        return Ok(Matrix::new(rows, cols, g.read_f32(name)?)?);
    }
    let mut bytes = g.read_tensor(name)?;
    if g.tensor_endian(name) == Some(llama_gguf::Endian::Big) {
        if matches!(dtype.id(), 1 | 30) {
            for word in bytes.as_chunks_mut::<2>().0 {
                word.reverse();
            }
        } else if dtype.id() != 43 {
            return Err(unsupported("big-endian packed matrix arithmetic"));
        }
    }
    Ok(Matrix::from_quantized(rows, cols, dtype, bytes)?)
}
pub(super) fn bias(
    g: &mut GgufSet,
    used: &mut BTreeSet<String>,
    name: &str,
    n: usize,
) -> Result<Option<Vec<f32>>> {
    if g.tensor(name).is_some() {
        Ok(Some(vector(g, used, name, n)?))
    } else {
        Ok(None)
    }
}

pub(super) fn details(
    g: &GgufSet,
    path: &Path,
    options: &LoadOptions,
) -> Result<(String, Option<String>, usize)> {
    let m = g.metadata();
    let id = m
        .get("general.name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
    let template = match m.get("tokenizer.chat_template") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(unsupported("multiple/nonstrings chat templates")),
        None => None,
    };
    let mut decoded_weight_bytes = 0usize;
    for tensor in g.tensors() {
        decoded_weight_bytes = decoded_weight_bytes
            .checked_add(
                tensor
                    .elements()?
                    .checked_mul(4)
                    .ok_or_else(|| Error::Invalid("decoded tensor byte count overflow".into()))?,
            )
            .ok_or_else(|| Error::Invalid("decoded model byte count overflow".into()))?;
    }
    if decoded_weight_bytes > options.max_decoded_weight_bytes {
        return Err(Error::Invalid(format!(
            "decoded model requires {decoded_weight_bytes} bytes, exceeding configured budget {}; the decoded F32 row cache exceeds the limit (packed bytes are additional)",
            options.max_decoded_weight_bytes
        )));
    }

    Ok((id, template, decoded_weight_bytes))
}

pub(crate) fn load(path: &Path, options: &LoadOptions) -> Result<Model> {
    validate_options(options)?;
    load_files(GgufSet::open(path)?, path, options)
}

pub(crate) fn load_splits<P: AsRef<Path>>(paths: &[P], options: &LoadOptions) -> Result<Model> {
    validate_options(options)?;
    let first = paths
        .first()
        .ok_or_else(|| Error::Invalid("list of GGUF splits is empty".into()))?;
    load_files(GgufSet::open_splits(paths)?, first.as_ref(), options)
}

fn validate_options(options: &LoadOptions) -> Result<()> {
    if options.context_length == 0 || options.max_decoded_weight_bytes == 0 {
        return Err(Error::Invalid(
            "context and weight budget must be nonzero".into(),
        ));
    }
    Ok(())
}

pub(crate) fn load_files(mut g: GgufSet, path: &Path, options: &LoadOptions) -> Result<Model> {
    validate_options(options)?;
    let m = g.metadata();
    let architecture = m
        .get("general.architecture")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Invalid("missing general.architecture".into()))?;
    shared_metadata(m, architecture)?;
    if architecture == "bert" {
        return crate::bert_loader::load(g, path, options);
    }
    if architecture == "qwen3" {
        return crate::qwen3_loader::load(g, path, options);
    }
    if architecture == "qwen2" {
        return crate::qwen2_loader::load(g, path, options);
    }
    if architecture == "mamba" {
        return crate::mamba_loader::load(g, path, options);
    }
    if architecture != "llama" {
        return Err(unsupported(format!(
            "model architecture {architecture}; this loader currently executes dense llama, qwen2, qwen3, bert and mamba"
        )));
    }
    if let Some(t) = m.get("general.type").and_then(Value::as_str)
        && t != "model"
    {
        return Err(unsupported(format!("GGUF general.type={t}")));
    }
    let tokenizer = Tokenizer::from_metadata(m)?;
    let width = uint(m, "llama.embedding_length", None)?;
    let heads = uint(m, "llama.attention.head_count", None)?;
    if width == 0 || heads == 0 || !width.is_multiple_of(heads) {
        return Err(Error::Invalid("invalid embedding/head dimensions".into()));
    }
    let head_dim = width / heads;
    let kv_heads = uint(m, "llama.attention.head_count_kv", Some(heads))?;
    let layers = uint(m, "llama.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    let feed_forward = uint(m, "llama.feed_forward_length", None)?;
    let train_context = uint(m, "llama.context_length", None)?;
    if uint(m, "llama.rope.dimension_count", Some(head_dim))? != head_dim {
        return Err(unsupported("partial rotary embedding dimensions"));
    }
    for key in ["llama.attention.key_length", "llama.attention.value_length"] {
        if uint(m, key, Some(head_dim))? != head_dim {
            return Err(unsupported(format!("nonstandard {key}")));
        }
    }
    for key in [
        "llama.attention.sliding_window",
        "llama.expert_count",
        "llama.expert_used_count",
    ] {
        if uint(m, key, Some(0))? != 0 {
            return Err(unsupported(format!("nonzero {key}")));
        }
    }
    require_causal(m, "llama", true)?;
    require_unscaled_rope(m, "llama")?;
    for key in ["llama.attention.qk_norm", "llama.attention.qk_norm_rms"] {
        if m.get(key).is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(unsupported(key.to_string()));
        }
    }
    let config = LlamaConfig {
        vocabulary: tokenizer.vocabulary_size(),
        width,
        feed_forward,
        heads,
        kv_heads,
        max_context: train_context.min(options.context_length),
        rms_epsilon: float(m, "llama.attention.layer_norm_rms_epsilon", 1e-5)?,
        rope_base: float(m, "llama.rope.freq_base", 10000.)?,
        attention_scale: match float(m, "llama.attention.scale", 0.)? {
            0. => None,
            v => Some(v),
        },
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
    let output_norm = vector(&mut g, &mut used, "output_norm.weight", width)?;
    let output = if g.tensor("output.weight").is_some() {
        Some(matrix(
            &mut g,
            &mut used,
            "output.weight",
            config.vocabulary,
            width,
        )?)
    } else {
        None
    };
    let mut weights = Vec::with_capacity(layers);
    for i in 0..layers {
        let key = |s: &str| format!("blk.{i}.{s}");
        weights.push(LlamaLayer {
            attention_norm: vector(&mut g, &mut used, &key("attn_norm.weight"), width)?,
            query: matrix(&mut g, &mut used, &key("attn_q.weight"), width, width)?,
            key: matrix(&mut g, &mut used, &key("attn_k.weight"), kv_width, width)?,
            value: matrix(&mut g, &mut used, &key("attn_v.weight"), kv_width, width)?,
            attention_output: matrix(&mut g, &mut used, &key("attn_output.weight"), width, width)?,
            query_bias: bias(&mut g, &mut used, &key("attn_q.bias"), width)?,
            key_bias: bias(&mut g, &mut used, &key("attn_k.bias"), kv_width)?,
            value_bias: bias(&mut g, &mut used, &key("attn_v.bias"), kv_width)?,
            attention_output_bias: bias(&mut g, &mut used, &key("attn_output.bias"), width)?,
            feed_forward_norm: vector(&mut g, &mut used, &key("ffn_norm.weight"), width)?,
            gate: matrix(
                &mut g,
                &mut used,
                &key("ffn_gate.weight"),
                feed_forward,
                width,
            )?,
            up: matrix(
                &mut g,
                &mut used,
                &key("ffn_up.weight"),
                feed_forward,
                width,
            )?,
            down: matrix(
                &mut g,
                &mut used,
                &key("ffn_down.weight"),
                width,
                feed_forward,
            )?,
            gate_bias: bias(&mut g, &mut used, &key("ffn_gate.bias"), feed_forward)?,
            up_bias: bias(&mut g, &mut used, &key("ffn_up.bias"), feed_forward)?,
            down_bias: bias(&mut g, &mut used, &key("ffn_down.bias"), width)?,
        });
    }
    if let Some(t) = g.tensor("rope_freqs.weight") {
        let n = t.elements()?;
        let values = g.read_f32("rope_freqs.weight")?;
        if n != head_dim / 2 || values.iter().any(|&x| x != 1.) {
            return Err(unsupported("per-dimension RoPE frequency factors"));
        }
        used.insert("rope_freqs.weight".into());
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
    let llama = LlamaModel::new(
        config,
        LlamaWeights {
            token_embedding,
            layers: weights,
            output_norm,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Llama(llama))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
