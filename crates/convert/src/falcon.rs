//! Frozen Falcon metadata and grouped QKV conversion.
//!
//! The source subclass has its own metadata override and preserves the original
//! tensor shape after moving grouped queries, keys and values into contiguous
//! ranges. Conversion behavior is separate from runtime model admission.
use crate::config::Config;
use crate::util::{Metadata, integer, present, truth};
use crate::{Error, OutputType, Result};
use llama_checkpoint::{Object, Value as Json};

pub(crate) fn valid_auto_config(root: &Object) -> bool {
    // FalconConfig does not validate head division or rotary metadata. It does
    // evaluate the default feed-forward size using the original hidden_size
    // argument (before the legacy n_embed override). Python repetition accepts
    // strings/arrays as well as numbers; None and objects fail construction.
    present(root.get("ffn_hidden_size")).is_some()
        || !matches!(root.get("hidden_size"), Some(Json::Null | Json::Object(_)))
}

pub(crate) fn metadata(config: &Config, out: &mut Metadata, output: OutputType) -> Result<()> {
    let hidden = config.required(&["hidden_size"])?;
    let heads = match present(config.get("num_attention_heads")?) {
        Some(value) => value,
        None => config.required(&["n_head"])?,
    };
    let kv = match present(config.get("num_kv_heads")?) {
        Some(value) => Some(value),
        None => config.get("n_head_kv")?,
    };
    out.uint("falcon.context_length", 2048)?;
    out.string("falcon.tensor_data_layout", "jploski")?;
    out.json_uint("falcon.embedding_length", hidden)?;
    let feed_forward = integer(hidden)?
        .checked_mul(4)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or(Error::Invalid("Falcon feed-forward dimension"))?;
    out.uint("falcon.feed_forward_length", feed_forward)?;
    out.uint(
        "falcon.block_count",
        u32::try_from(config.layers).map_err(|_| Error::Limit("Falcon layer count"))?,
    )?;
    // Both subclasses call the same native GGUF head-count writer. Its
    // integer/sequence behavior is observable even in vocabulary-only files.
    crate::bloom::head_count(out, "falcon.attention.head_count", heads)?;
    if let Some(kv) = kv {
        crate::bloom::head_count(out, "falcon.attention.head_count_kv", kv)?;
    } else {
        out.uint("falcon.attention.head_count_kv", 1)?;
    }
    out.json_float(
        "falcon.attention.layer_norm_epsilon",
        config.required(&["layer_norm_epsilon"])?,
    )?;
    out.uint(
        "general.file_type",
        output
            .file_type()
            .ok_or(Error::Invalid("unresolved output type"))?,
    )
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Qkv {
    groups: usize,
    queries_per_group: usize,
    block_elements: usize,
}
impl Qkv {
    pub(crate) fn plan(config: &Config, name: &str, shape: &[u64]) -> Result<Option<Self>> {
        // The frozen override tests the substring, including aliases and bias
        // names, rather than a particular layer prefix or tensor suffix.
        if !name.contains("query_key_value") {
            return Ok(None);
        }
        let heads = usize::try_from(integer(
            config.required(&["num_attention_heads", "n_head"])?,
        )?)
        .map_err(|_| Error::Invalid("negative Falcon query head count"))?;
        let groups = match config
            .find(&["num_kv_heads", "n_head_kv"])?
            .filter(|value| truth(value))
        {
            Some(Json::Bool(_)) => {
                // Torch view rejects a boolean dimension. A false value has
                // already followed Python's or-one fallback above.
                return Err(Error::Invalid("boolean Falcon QKV group dimension"));
            }
            Some(value) => usize::try_from(integer(value)?)
                .map_err(|_| Error::Invalid("negative Falcon KV head count"))?,
            None => 1,
        };
        let hidden = usize::try_from(integer(config.required(&["hidden_size"])?)?)
            .map_err(|_| Error::Invalid("negative Falcon embedding dimension"))?;
        if heads == 0 || groups == 0 {
            return Err(Error::Invalid("zero Falcon QKV divisor"));
        }
        let head_width = hidden / heads;
        let queries_per_group = heads / groups;
        let row_width = head_width
            .checked_mul(heads)
            .ok_or(Error::Limit("Falcon QKV row width"))?;
        let block_elements = head_width
            .checked_mul(row_width)
            .ok_or(Error::Limit("Falcon QKV head block"))?;
        let grouped_queries = groups
            .checked_mul(queries_per_group)
            .and_then(|n| n.checked_mul(block_elements))
            .ok_or(Error::Limit("Falcon QKV query elements"))?;
        let reshaped_queries = heads
            .checked_mul(block_elements)
            .ok_or(Error::Limit("Falcon QKV query reshape"))?;
        if grouped_queries != reshaped_queries {
            return Err(Error::Invalid("Falcon query reshape element count"));
        }
        let elements = groups
            .checked_mul(
                queries_per_group
                    .checked_add(2)
                    .ok_or(Error::Limit("Falcon QKV grouped heads"))?,
            )
            .and_then(|n| n.checked_mul(block_elements))
            .ok_or(Error::Limit("Falcon QKV elements"))?;
        let actual = shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .ok_or(Error::Limit("Falcon source elements"))?;
        if u64::try_from(elements).ok() != Some(actual) {
            return Err(Error::Invalid("Falcon QKV reshape element count"));
        }
        Ok(Some(Self {
            groups,
            queries_per_group,
            block_elements,
        }))
    }

    pub(crate) fn apply(self, input: &[u8], output: &mut [u8], scalar_bytes: usize) -> Result<()> {
        let block = self
            .block_elements
            .checked_mul(scalar_bytes)
            .ok_or(Error::Limit("Falcon QKV block bytes"))?;
        let queries = self
            .queries_per_group
            .checked_mul(block)
            .ok_or(Error::Limit("Falcon QKV query block bytes"))?;
        let stride = block
            .checked_mul(2)
            .and_then(|n| n.checked_add(queries))
            .ok_or(Error::Limit("Falcon QKV group bytes"))?;
        let expected = stride
            .checked_mul(self.groups)
            .ok_or(Error::Limit("Falcon QKV total bytes"))?;
        if input.len() != expected || output.len() != expected {
            return Err(Error::Invalid("Falcon QKV prepared byte count"));
        }
        if input.is_empty() {
            return Ok(());
        }
        // All products used below are bounded by the checked complete buffer.
        let keys_start = queries * self.groups;
        let values_start = keys_start + block * self.groups;
        for group in 0..self.groups {
            let source = group * stride;
            let query = group * queries;
            output[query..query + queries].copy_from_slice(&input[source..source + queries]);
            let key = keys_start + group * block;
            let value = values_start + group * block;
            output[key..key + block]
                .copy_from_slice(&input[source + queries..source + queries + block]);
            output[value..value + block]
                .copy_from_slice(&input[source + queries + block..source + stride]);
        }
        Ok(())
    }
}
