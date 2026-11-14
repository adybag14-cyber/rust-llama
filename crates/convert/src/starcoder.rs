//! Frozen StarCoder metadata override. StarCoder2 inherits TextModel metadata.
//! Both checkpoint families keep their native linear-weight orientation.
use crate::config::Config;
use crate::util::{Metadata, integer};
use crate::{Error, OutputType, Result};

pub(crate) fn metadata(config: &Config, out: &mut Metadata, output: OutputType) -> Result<()> {
    out.json_uint(
        "starcoder.context_length",
        config.required(&["n_positions"])?,
    )?;
    let hidden = config.required(&["n_embd"])?;
    out.json_uint("starcoder.embedding_length", hidden)?;
    let feed_forward = integer(hidden)?
        .checked_mul(4)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or(Error::Invalid("StarCoder feed-forward dimension"))?;
    // The frozen source ignores n_inner and multi_query here, including
    // configurations that will not form a loadable inference model.
    out.uint("starcoder.feed_forward_length", feed_forward)?;
    out.uint(
        "starcoder.block_count",
        u32::try_from(config.layers).map_err(|_| Error::Limit("StarCoder layer count"))?,
    )?;
    crate::bloom::head_count(
        out,
        "starcoder.attention.head_count",
        config.required(&["n_head"])?,
    )?;
    out.uint("starcoder.attention.head_count_kv", 1)?;
    out.json_float(
        "starcoder.attention.layer_norm_epsilon",
        config.required(&["layer_norm_epsilon"])?,
    )?;
    out.uint(
        "general.file_type",
        output
            .file_type()
            .ok_or(Error::Invalid("unresolved output type"))?,
    )
}
