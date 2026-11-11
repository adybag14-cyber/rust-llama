//! The frozen GPT-2 and GPT-NeoX converter contracts.
//!
//! These families override TextModel metadata rather than inheriting the
//! general Llama/Qwen fields. Tensor transformations act on prepared scalar
//! bytes, preserving F16/NaN payloads before the shared output codec.
use crate::config::{Architecture, Config, floor_div};
use crate::util::{Metadata, integer, real, truth};
use crate::{Error, OutputType, Result};
use llama_checkpoint::Object;

pub(crate) fn valid_neox_config(root: &Object) -> bool {
    let hidden = root.get("hidden_size").map(integer).unwrap_or(Ok(6144));
    let heads = root
        .get("num_attention_heads")
        .map(integer)
        .unwrap_or(Ok(64));
    matches!((hidden, heads), (Ok(h), Ok(n)) if n != 0 && h.checked_rem(n) == Some(0))
}

pub(crate) fn metadata(config: &Config, out: &mut Metadata, output: OutputType) -> Result<()> {
    let family = config.family;
    let prefix = family.name();
    let key = |name: &str| format!("{prefix}.{name}");
    let blocks = u32::try_from(config.layers).map_err(|_| Error::Limit("layer count"))?;
    match family {
        Architecture::Gpt2 => {
            out.uint(&key("block_count"), blocks)?;
            out.json_uint(&key("context_length"), config.required(&["n_ctx"])?)?;
            let hidden = config.required(&["n_embd"])?;
            out.json_uint(&key("embedding_length"), hidden)?;
            let feed_forward = integer(hidden)?
                .checked_mul(4)
                .and_then(|v| u32::try_from(v).ok())
                .ok_or(Error::Invalid("GPT-2 feed-forward dimension"))?;
            // The source always emits 4*n_embd, even when n_inner differs.
            out.uint(&key("feed_forward_length"), feed_forward)?;
            out.json_uint(&key("attention.head_count"), config.required(&["n_head"])?)?;
            out.json_float(
                &key("attention.layer_norm_epsilon"),
                config.required(&["layer_norm_epsilon"])?,
            )?;
            out.uint(
                "general.file_type",
                output
                    .file_type()
                    .ok_or(Error::Invalid("unresolved output type"))?,
            )?;
        }
        Architecture::GptNeoX => {
            out.json_uint(
                &key("context_length"),
                config.required(&["max_position_embeddings"])?,
            )?;
            let hidden = config.required(&["hidden_size"])?;
            let heads = config.required(&["num_attention_heads"])?;
            out.json_uint(&key("embedding_length"), hidden)?;
            out.uint(&key("block_count"), blocks)?;
            out.json_uint(
                &key("feed_forward_length"),
                config.required(&["intermediate_size"])?,
            )?;
            let width = floor_div(integer(hidden)?, integer(heads)?)?;
            let rotary = real(config.required(&["rotary_pct"])?)? * width as f64;
            let rotary = rotary.trunc();
            if !rotary.is_finite() || rotary < 0.0 || rotary > u32::MAX as f64 {
                return Err(Error::Invalid("GPT-NeoX rotary dimension"));
            }
            out.uint(&key("rope.dimension_count"), rotary as u32)?;
            out.json_uint(&key("attention.head_count"), heads)?;
            out.boolean(
                &key("use_parallel_residual"),
                config.get("use_parallel_residual")?.is_none_or(truth),
            )?;
            out.json_float(
                &key("attention.layer_norm_epsilon"),
                config.required(&["layer_norm_eps"])?,
            )?;
            // The frozen GPTNeoXModel override does not write general.file_type
            // or common RoPE metadata; emitting either would change its bytes.
        }
        _ => return Err(Error::Invalid("non-GPT metadata dispatch")),
    }
    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) enum Transform {
    #[default]
    Identity,
    Transpose01 {
        first: usize,
        second: usize,
        tail: usize,
    },
    NeoXQkv {
        heads: usize,
        head_width: usize,
        row_width: usize,
    },
}
impl Transform {
    /// Return a source-specific transform and replace the shape with the
    /// source's resulting shape. No tensor data is read while planning.
    pub(crate) fn plan(config: &Config, name: &str, shape: &mut Vec<u64>) -> Result<Self> {
        if config.family == Architecture::Gpt2
            && [".c_attn.weight", ".c_proj.weight", ".c_fc.weight"]
                .iter()
                .any(|s| name.ends_with(s))
        {
            if shape.len() < 2 {
                return Err(Error::Invalid(
                    "GPT-2 transpose requires at least two dimensions",
                ));
            }
            let first =
                usize::try_from(shape[0]).map_err(|_| Error::Limit("GPT-2 transpose dimension"))?;
            let second =
                usize::try_from(shape[1]).map_err(|_| Error::Limit("GPT-2 transpose dimension"))?;
            let tail = shape[2..]
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
                .ok_or(Error::Limit("GPT-2 transpose tail"))?;
            shape.swap(0, 1);
            return Ok(Self::Transpose01 {
                first,
                second,
                tail,
            });
        }
        if matches!(config.family, Architecture::GptNeoX | Architecture::Bloom) {
            // The subclass asserts these before its name dispatch, including
            // for non-QKV tensors. An explicit n_head:null masks the fallback.
            let heads_value = config.required(&["n_head", "num_attention_heads"])?;
            let hidden_value = config.required(&["hidden_size", "n_embed"])?;
            if matches!(heads_value, llama_checkpoint::Value::Null)
                || matches!(hidden_value, llama_checkpoint::Value::Null)
            {
                return Err(Error::Invalid("null QKV tensor dimensions"));
            }
            // The original re.match is rooted in this exact checkpoint prefix.
            // Canonical/other aliases must not accidentally trigger a reshape.
            let bloom = config.family == Architecture::Bloom;
            let suffix = name
                .strip_prefix(if bloom { "h." } else { "gpt_neox.layers." })
                .and_then(|s| s.split_once('.'))
                .filter(|(layer, _)| !layer.is_empty() && layer.bytes().all(|c| c.is_ascii_digit()))
                .map(|(_, rest)| rest);
            let row_width = match (bloom, suffix) {
                (false, Some("attention.query_key_value.weight"))
                | (true, Some("self_attention.query_key_value.weight")) => None,
                (false, Some("attention.query_key_value.bias"))
                | (true, Some("self_attention.query_key_value.bias")) => Some(1usize),
                _ => return Ok(Self::Identity),
            };
            let heads = usize::try_from(integer(heads_value)?)
                .map_err(|_| Error::Invalid("negative QKV heads"))?;
            let hidden = usize::try_from(integer(hidden_value)?)
                .map_err(|_| Error::Invalid("negative QKV embedding"))?;
            // BLOOM admits an empty bias with hidden=0; the weight path
            // still fails its source reshape(-1, 0), and NeoX retains its
            // independently captured zero-width rejection.
            if heads == 0 || (hidden == 0 && !(bloom && row_width.is_some())) {
                return Err(Error::Invalid("zero QKV dimension"));
            }
            let head_width = hidden / heads;
            let rows = heads
                .checked_mul(head_width)
                .and_then(|n| n.checked_mul(3))
                .ok_or(Error::Limit("GPT-NeoX QKV rows"))?;
            let width = row_width.unwrap_or(hidden);
            let expected = rows
                .checked_mul(width)
                .ok_or(Error::Limit("GPT-NeoX QKV elements"))?;
            let actual = shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d))
                .ok_or(Error::Limit("GPT-NeoX source elements"))?;
            if actual != expected as u64 || (row_width.is_some() && !hidden.is_multiple_of(heads)) {
                return Err(Error::Invalid("GPT-NeoX QKV reshape element count"));
            }
            shape.clear();
            crate::reserve(
                shape,
                if row_width.is_some() { 1 } else { 2 },
                "GPT-NeoX QKV shape",
            )?;
            shape.push(rows as u64);
            if row_width.is_none() {
                shape.push(width as u64);
            }
            return Ok(Self::NeoXQkv {
                heads,
                head_width,
                row_width: width,
            });
        }
        Ok(Self::Identity)
    }
    pub(crate) fn is_identity(self) -> bool {
        matches!(self, Self::Identity)
    }

    /// Copy a checked permutation into a separately charged output buffer.
    pub(crate) fn apply(self, input: &[u8], output: &mut [u8], scalar_bytes: usize) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::Invalid("GPT transform output size"));
        }
        let (first, second, tail) = match self {
            Self::Identity => {
                output.copy_from_slice(input);
                return Ok(());
            }
            Self::Transpose01 {
                first,
                second,
                tail,
            } => (first, second, tail),
            Self::NeoXQkv {
                heads,
                head_width,
                row_width,
            } => (
                heads,
                3,
                head_width
                    .checked_mul(row_width)
                    .ok_or(Error::Limit("GPT-NeoX QKV block"))?,
            ),
        };
        let block = tail
            .checked_mul(scalar_bytes)
            .ok_or(Error::Limit("GPT transform block bytes"))?;
        let expected = first
            .checked_mul(second)
            .and_then(|n| n.checked_mul(block))
            .ok_or(Error::Limit("GPT transform byte count"))?;
        if expected != input.len() {
            return Err(Error::Invalid(
                "GPT transform shape differs from prepared bytes",
            ));
        }
        // Empty trailing dimensions can make the buffer empty even when the
        // leading dimensions are large. No iteration is needed in that case.
        if input.is_empty() {
            return Ok(());
        }
        for i in 0..first {
            for j in 0..second {
                let from = (i * second + j) * block;
                let to = (j * first + i) * block;
                output[to..to + block].copy_from_slice(&input[from..from + block]);
            }
        }
        Ok(())
    }
}
