//! Frozen BLOOM metadata override, distinct from the common TextModel fields.
//! Model inputs can contain head-count sequences even when they do not describe
//! an executable inference graph. Preserve their original GGUF wire types.
use crate::config::Config;
use crate::util::{Metadata, f32_value, integer, text};
use crate::{Error, OutputType, Result, bytes, reserve};
use llama_checkpoint::{Number, Value as Json};
use llama_gguf::RawValue as Value;

pub(crate) fn metadata(config: &Config, out: &mut Metadata, output: OutputType) -> Result<()> {
    let hidden = config.required(&["hidden_size", "n_embed"])?;
    let heads = config.required(&["n_head", "num_attention_heads"])?;
    if matches!(hidden, Json::Null) || matches!(heads, Json::Null) {
        return Err(Error::Invalid("null BLOOM metadata dimensions"));
    }
    out.json_uint(
        "bloom.context_length",
        config.get("seq_length")?.unwrap_or(hidden),
    )?;
    out.json_uint("bloom.embedding_length", hidden)?;
    let feed_forward = integer(hidden)?
        .checked_mul(4)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or(Error::Invalid("BLOOM feed-forward dimension"))?;
    out.uint("bloom.feed_forward_length", feed_forward)?;
    out.uint(
        "bloom.block_count",
        u32::try_from(config.layers).map_err(|_| Error::Limit("BLOOM layer count"))?,
    )?;
    head_count(out, "bloom.attention.head_count", heads)?;
    head_count(out, "bloom.attention.head_count_kv", heads)?;
    out.json_float(
        "bloom.attention.layer_norm_epsilon",
        config.required(&["layer_norm_epsilon"])?,
    )?;
    out.uint(
        "general.file_type",
        output
            .file_type()
            .ok_or(Error::Invalid("unresolved output type"))?,
    )
}
fn head_count(out: &mut Metadata, key: &str, input: &Json) -> Result<()> {
    if matches!(input, Json::Number(Number::Integer(_)) | Json::Bool(_)) {
        return out.json_uint(key, input);
    }
    match input {
        Json::Text(_) => {
            // Python treats a string as a sequence of Unicode scalar strings.
            let text = text(input)?;
            if text.is_empty() {
                return Ok(());
            }
            let mut values = Vec::new();
            reserve(&mut values, text.chars().count(), "BLOOM head characters")?;
            for ch in text.chars() {
                let mut local = [0u8; 4];
                values.push(Value::String(bytes(ch.encode_utf8(&mut local).as_bytes())?));
            }
            out.put(
                key,
                Value::Array {
                    element_type: 8,
                    values,
                },
            )
        }
        Json::Array(values) if values.is_empty() => Ok(()),
        Json::Array(values) => out.put(key, array(values, 0)?),
        _ => Err(Error::Invalid(
            "BLOOM head count is not an integer or sequence",
        )),
    }
}
fn kind(value: &Json) -> Result<u32> {
    Ok(match value {
        Json::Number(Number::Integer(_)) => 5,
        Json::Number(Number::Float(_)) => 6,
        Json::Bool(_) => 7,
        Json::Text(_) => 8,
        Json::Array(_) => 9,
        _ => return Err(Error::Invalid("unsupported BLOOM metadata array item")),
    })
}
fn array(input: &[Json], depth: usize) -> Result<Value> {
    // Bound recursive construction independently of caller-adjustable JSON
    // limits. This is a declared ingestion limit, not a source shape rewrite.
    if depth >= 64 {
        return Err(Error::Limit("BLOOM metadata array depth"));
    }
    let first = input
        .first()
        .ok_or(Error::Invalid("empty nested BLOOM metadata array"))?;
    let element_type = kind(first)?;
    let mut values = Vec::new();
    reserve(&mut values, input.len(), "BLOOM metadata array")?;
    for item in input {
        if kind(item)? != element_type {
            return Err(Error::Invalid("heterogeneous BLOOM metadata array"));
        }
        values.push(match item {
            Json::Number(Number::Integer(_)) => Value::I32(
                i32::try_from(integer(item)?)
                    .map_err(|_| Error::Invalid("INT32 metadata overflow"))?,
            ),
            Json::Number(Number::Float(_)) => Value::F32(f32_value(item)?),
            Json::Bool(value) => Value::Bool(*value),
            Json::Text(_) => Value::String(bytes(text(item)?.as_bytes())?),
            Json::Array(items) => array(items, depth + 1)?,
            _ => return Err(Error::Invalid("unsupported BLOOM metadata array item")),
        });
    }
    Ok(Value::Array {
        element_type,
        values,
    })
}
