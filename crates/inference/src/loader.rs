use crate::{Error, LoadOptions, Model, Result};
use llama_gguf::{Gguf, Value};
use llama_model_runtime::{LlamaConfig, LlamaLayer, LlamaModel, LlamaWeights, Matrix};
use llama_tokenizer::Tokenizer;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::Path,
    sync::Arc,
};

fn uint(m: &BTreeMap<String, Value>, key: &str, default: Option<usize>) -> Result<usize> {
    match m.get(key) {
        Some(v) => v
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| Error::Invalid(format!("{key} must be an unsigned integer"))),
        None => default.ok_or_else(|| Error::Invalid(format!("missing {key}"))),
    }
}
fn float(m: &BTreeMap<String, Value>, key: &str, default: f32) -> Result<f32> {
    match m.get(key) {
        None => Ok(default),
        Some(v) => {
            let v = v
                .as_f64()
                .ok_or_else(|| Error::Invalid(format!("{key} must be floating point")))?;
            if !v.is_finite() || v.abs() > f32::MAX as f64 {
                Err(Error::Invalid(format!("{key} is non-finite/out of range")))
            } else {
                Ok(v as f32)
            }
        }
    }
}
fn unsupported(s: impl Into<String>) -> Error {
    Error::Unsupported(s.into())
}

pub(crate) fn load(path: &Path, options: &LoadOptions) -> Result<Model> {
    if options.context_length == 0 || options.max_decoded_weight_bytes == 0 {
        return Err(Error::Invalid(
            "context and weight budget must be nonzero".into(),
        ));
    }
    let mut g = Gguf::open(path)?;
    let m = &g.metadata;
    let architecture = m
        .get("general.architecture")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Invalid("missing general.architecture".into()))?;
    if architecture != "llama" {
        return Err(unsupported(format!(
            "model architecture {architecture}; this loader currently executes dense llama"
        )));
    }
    if let Some(t) = m.get("general.type").and_then(Value::as_str)
        && t != "model"
    {
        return Err(unsupported(format!("GGUF general.type={t}")));
    }
    if uint(m, "split.count", Some(1))? > 1 {
        return Err(unsupported("split GGUF model loading"));
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
    if layers == 0 || layers > 4096 {
        return Err(Error::Invalid("layer count must be 1..=4096".into()));
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
    if m.get("llama.rope.scaling.type")
        .is_some_and(|v| v.as_str() != Some("none"))
    {
        return Err(unsupported(
            "RoPE scaling; this runtime implements unscaled adjacent-pair RoPE",
        ));
    }
    if float(m, "llama.rope.scaling.factor", 1.)? != 1.
        || float(m, "llama.rope.scale_linear", 1.)? != 1.
    {
        return Err(unsupported("scaled RoPE frequencies"));
    }
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
    for tensor in &g.tensors {
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
            "decoded model requires {decoded_weight_bytes} bytes, exceeding configured budget {}; weights are expanded to F32 by this scalar executor",
            options.max_decoded_weight_bytes
        )));
    }
    let mut used = BTreeSet::new();
    fn vector(
        g: &mut Gguf<File>,
        used: &mut BTreeSet<String>,
        name: &str,
        n: usize,
    ) -> Result<Vec<f32>> {
        let t = g
            .tensor(name)
            .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))?;
        if t.dimensions != [n as u64] {
            return Err(Error::Invalid(format!(
                "tensor {name} expected shape [{n}], found {:?}",
                t.dimensions
            )));
        }
        used.insert(name.into());
        Ok(g.read_f32(name)?)
    }
    fn matrix(
        g: &mut Gguf<File>,
        used: &mut BTreeSet<String>,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Matrix> {
        let t = g
            .tensor(name)
            .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))?;
        if t.dimensions != [cols as u64, rows as u64] {
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
        if g.endian == llama_gguf::Endian::Big {
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
    fn bias(
        g: &mut Gguf<File>,
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
        .tensors
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
        runtime: Arc::new(llama_model_runtime::Model::Llama(llama)),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
    })
}
