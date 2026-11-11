//! Bounded native conversion of dense Hugging Face Llama, Qwen2 and Qwen3 checkpoints.
//! The frozen Python converter remains an external comparison oracle only.
#![deny(unsafe_code)]

mod authorship;
pub mod codec;
mod config;
mod frozen;
mod hf_backend;
mod hf_bpe;
mod hf_probe;
mod hf_schema;
mod hf_unicode;
mod model_card;
mod names;
mod pipeline;
mod python_set;
mod python_unicode;
mod qwen;
mod qwen_tables;
pub mod rope;
mod sentencepiece;
mod special;
mod tensors;
mod util;
mod vocab;
pub use model_card::YamlLimits;
pub use pipeline::{Conversion, Summary};

use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Checkpoint(llama_checkpoint::Error),
    Gguf(llama_gguf::Error),
    Codec(codec::CodecError),
    Rope(rope::RopeError),
    Invalid(&'static str),
    Unsupported(&'static str),
    Limit(&'static str),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "converter I/O: {e}"),
            Self::Checkpoint(e) => e.fmt(f),
            Self::Gguf(e) => e.fmt(f),
            Self::Codec(e) => e.fmt(f),
            Self::Rope(e) => e.fmt(f),
            Self::Invalid(e) => write!(f, "invalid conversion input: {e}"),
            Self::Unsupported(e) => write!(f, "unsupported conversion input: {e}"),
            Self::Limit(e) => write!(f, "converter limit: {e}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<llama_checkpoint::Error> for Error {
    fn from(e: llama_checkpoint::Error) -> Self {
        Self::Checkpoint(e)
    }
}
impl From<codec::CodecError> for Error {
    fn from(e: codec::CodecError) -> Self {
        Self::Codec(e)
    }
}
impl From<rope::RopeError> for Error {
    fn from(e: rope::RopeError) -> Self {
        Self::Rope(e)
    }
}
impl From<llama_gguf::Error> for Error {
    fn from(e: llama_gguf::Error) -> Self {
        Self::Gguf(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputType {
    #[default]
    Auto,
    F32,
    F16,
    Bf16,
    Q8_0,
    Tq1_0,
    Tq2_0,
}
impl OutputType {
    pub fn parse(text: &str) -> Result<Self> {
        Ok(match text {
            "auto" => Self::Auto,
            "f32" => Self::F32,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "q8_0" => Self::Q8_0,
            "tq1_0" => Self::Tq1_0,
            "tq2_0" => Self::Tq2_0,
            _ => return Err(Error::Invalid("unknown output type")),
        })
    }
    pub const fn file_type(self) -> Option<u32> {
        Some(match self {
            Self::Auto => return None,
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 32,
            Self::Q8_0 => 7,
            Self::Tq1_0 => 36,
            Self::Tq2_0 => 37,
        })
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub output_type: OutputType,
    /// Explicit captured source-build policy. None uses SourceProfile::current().
    pub source_profile: Option<codec::SourceProfile>,
    pub model_name: Option<String>,
    pub vocab_only: bool,
    pub fuse_qkv: bool,
    /// Optional metadata override relative to the retained input directory.
    pub metadata_override: Option<String>,
}
/// Independent input, descriptor and per-tensor working-set limits.
/// They do not impose a total process or filesystem cache budget.
#[derive(Debug, Clone)]
pub struct Limits {
    pub checkpoint: llama_checkpoint::Limits,
    pub sidecars: llama_checkpoint::SourceLimits,
    pub json: llama_checkpoint::Limits,
    pub yaml: YamlLimits,
    pub gguf: llama_gguf::Limits,
    pub max_vocabulary: usize,
    pub max_proto_fields: usize,
    pub max_tokenizer_operations: usize,
    pub max_layers: usize,
    pub max_tensor_input_bytes: usize,
    pub max_working_bytes: usize,
    pub max_output_bytes: u64,
    pub max_tensor_name_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            checkpoint: llama_checkpoint::Limits::default(),
            sidecars: llama_checkpoint::SourceLimits::default(),
            json: llama_checkpoint::Limits::default(),
            yaml: YamlLimits::default(),
            gguf: llama_gguf::Limits::default(),
            max_vocabulary: 1_000_000,
            max_proto_fields: 4_000_000,
            max_tokenizer_operations: 64_000_000,
            max_layers: 512,
            max_tensor_input_bytes: 1024 * 1024 * 1024,
            max_working_bytes: 2 * 1024 * 1024 * 1024,
            max_output_bytes: 512 * 1024 * 1024 * 1024,
            max_tensor_name_bytes: 1024 * 1024,
        }
    }
}
pub(crate) fn reserve<T>(values: &mut Vec<T>, n: usize, label: &'static str) -> Result<()> {
    values.try_reserve(n).map_err(|_| Error::Limit(label))
}
pub(crate) fn string(text: &str) -> Result<String> {
    let mut value = String::new();
    value
        .try_reserve_exact(text.len())
        .map_err(|_| Error::Limit("string allocation"))?;
    value.push_str(text);
    Ok(value)
}

pub(crate) fn bytes(input: &[u8]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(input.len())
        .map_err(|_| Error::Limit("byte string allocation"))?;
    result.extend_from_slice(input);
    Ok(result)
}

#[cfg(test)]
mod hf_reference;

#[cfg(test)]
mod dense_reference;

#[cfg(test)]
mod hf_schema_reference;
