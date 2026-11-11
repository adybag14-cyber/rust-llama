//! Native, bounded model-file quantization from the frozen llama.cpp snapshot.
//! Source: src/llama-quant.cpp and src/llama-model-loader.h.
//! Copyright (c) 2023-2026 The ggml authors. MIT license.
//!
//! The source ZIP SHA-256 is
//! 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
//! Independent MSVC/GNU type and complete-file captures precede this port.
#![forbid(unsafe_code)]
mod architectures;
pub mod cli;
mod cli_scalar;
mod file;
pub mod imatrix;
mod metadata;
mod planner;
pub use file::{QuantizationReport, TensorReport, quantize_file, quantize_file_with_limits};
pub use planner::{ModelDescriptor, TensorSpec, allows_quantization, assign_types};

use llama_gguf::{RawValue, ShardLimits};
use llama_quant::DType;
use std::collections::BTreeMap;

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Limit(String),
    Unsupported(String),
    Io(std::io::Error),
    Gguf(llama_gguf::Error),
    Quant(llama_quant::QuantError),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid model quantization: {s}"),
            Self::Limit(s) => write!(f, "model quantization limit: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported model quantization: {s}"),
            Self::Io(e) => e.fmt(f),
            Self::Gguf(e) => e.fmt(f),
            Self::Quant(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<llama_gguf::Error> for Error {
    fn from(e: llama_gguf::Error) -> Self {
        Self::Gguf(e)
    }
}
impl From<llama_quant::QuantError> for Error {
    fn from(e: llama_quant::QuantError) -> Self {
        Self::Quant(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

/// Frozen llama_ftype ID. Construction accepts only types implemented by the
/// frozen model-file quantizer; declared-but-rejected NVFP4/F8 IDs stay rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileType(u32);
impl FileType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q8_0: Self = Self(7);
    pub const BF16: Self = Self(32);
    pub const ALL: [Self; 35] = [
        Self(0),
        Self(1),
        Self(2),
        Self(3),
        Self(7),
        Self(8),
        Self(9),
        Self(10),
        Self(11),
        Self(12),
        Self(13),
        Self(14),
        Self(15),
        Self(16),
        Self(17),
        Self(18),
        Self(19),
        Self(20),
        Self(21),
        Self(22),
        Self(23),
        Self(24),
        Self(25),
        Self(26),
        Self(27),
        Self(28),
        Self(29),
        Self(30),
        Self(31),
        Self(32),
        Self(36),
        Self(37),
        Self(38),
        Self(40),
        Self(41),
    ];
    pub fn from_id(id: u32) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|f| f.0 == id)
            .ok_or_else(|| invalid(format!("invalid output file type {id}")))
    }
    pub const fn id(self) -> u32 {
        self.0
    }
    pub const fn default_type(self) -> DType {
        use DType::*;
        match self.0 {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            7 => Q8_0,
            8 => Q5_0,
            9 => Q5_1,
            10 | 21 => Q2_K,
            11..=13 => Q3_K,
            14 | 15 => Q4_K,
            16 | 17 => Q5_K,
            18 => Q6_K,
            19 => IQ2_XXS,
            20 | 28 => IQ2_XS,
            22 | 26 | 27 => IQ3_S,
            23 => IQ3_XXS,
            24 => IQ1_S,
            25 => IQ4_NL,
            29 => IQ2_S,
            30 => IQ4_XS,
            31 => IQ1_M,
            32 => BF16,
            36 => TQ1_0,
            37 => TQ2_0,
            38 => MXFP4,
            40 => Q1_0,
            41 => Q2_0,
            _ => unreachable!(),
        }
    }
    pub const fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            7 => "Q8_0",
            8 => "Q5_0",
            9 => "Q5_1",
            10 => "Q2_K",
            11 => "Q3_K_S",
            12 => "Q3_K_M",
            13 => "Q3_K_L",
            14 => "Q4_K_S",
            15 => "Q4_K_M",
            16 => "Q5_K_S",
            17 => "Q5_K_M",
            18 => "Q6_K",
            19 => "IQ2_XXS",
            20 => "IQ2_XS",
            21 => "Q2_K_S",
            22 => "IQ3_XS",
            23 => "IQ3_XXS",
            24 => "IQ1_S",
            25 => "IQ4_NL",
            26 => "IQ3_S",
            27 => "IQ3_M",
            28 => "IQ2_S",
            29 => "IQ2_M",
            30 => "IQ4_XS",
            31 => "IQ1_M",
            32 => "BF16",
            36 => "TQ1_0",
            37 => "TQ2_0",
            38 => "MXFP4_MOE",
            40 => "Q1_0",
            41 => "Q2_0",
            _ => unreachable!(),
        }
    }
    pub fn parse(text: &str) -> Result<Self> {
        let name = text.to_ascii_uppercase();
        let alias = match name.as_str() {
            "Q3_K" => "Q3_K_M",
            "Q4_K" => "Q4_K_M",
            "Q5_K" => "Q5_K_M",
            n => n,
        };
        if let Some(v) = Self::ALL.into_iter().find(|f| f.name() == alias) {
            return Ok(v);
        }
        if let Ok(id) = name.parse() {
            return Self::from_id(id);
        }
        Err(invalid(format!("unknown file type {text}")))
    }
}
#[derive(Clone, Debug)]
pub struct TensorTypeOverride {
    pub pattern: Vec<u8>,
    pub dtype: DType,
}
/// Typed C-compatible override semantics. Integer output values are UINT32
/// absolute magnitudes, while model hyperparameter lookup uses the signed input.
#[derive(Clone, Debug)]
pub enum OverrideValue {
    Integer(i64),
    Float(f64),
    Bool(bool),
    String(Vec<u8>),
}
#[derive(Clone, Debug)]
pub struct MetadataOverride {
    pub key: Vec<u8>,
    pub value: OverrideValue,
}
impl MetadataOverride {
    pub(crate) fn output_value(&self) -> Result<RawValue> {
        Ok(match &self.value {
            OverrideValue::Integer(v) => {
                RawValue::U32(v.checked_abs().ok_or_else(|| {
                    invalid("INT64_MIN metadata override has no source abs result")
                })? as u32)
            }
            OverrideValue::Float(v) => RawValue::F32(*v as f32),
            OverrideValue::Bool(v) => RawValue::Bool(*v),
            OverrideValue::String(v) => RawValue::String(v.clone()),
        })
    }
}
/// Owned importance rows, keyed by original tensor name before layer pruning.
/// Some(empty) retains the source distinction from a null importance pointer.
pub type ImportanceMatrix = BTreeMap<String, Vec<f32>>;
#[derive(Clone, Debug)]
pub struct QuantizeParams {
    pub nthread: i32,
    pub ftype: FileType,
    pub output_tensor_type: Option<DType>,
    pub token_embedding_type: Option<DType>,
    pub allow_requantize: bool,
    pub quantize_output_tensor: bool,
    pub only_copy: bool,
    pub pure: bool,
    pub keep_split: bool,
    pub dry_run: bool,
    pub imatrix: Option<ImportanceMatrix>,
    pub kv_overrides: Vec<MetadataOverride>,
    pub tt_overrides: Vec<TensorTypeOverride>,
    pub prune_layers: Vec<i32>,
    /// Slab target; one complete row is the minimum. Zero means 8 GiB.
    pub max_buf_size: usize,
}
impl Default for QuantizeParams {
    fn default() -> Self {
        Self {
            nthread: 0,
            ftype: FileType::Q8_0,
            output_tensor_type: None,
            token_embedding_type: None,
            allow_requantize: false,
            quantize_output_tensor: true,
            only_copy: false,
            pure: false,
            keep_split: false,
            dry_run: false,
            imatrix: None,
            kv_overrides: Vec::new(),
            tt_overrides: Vec::new(),
            prune_layers: Vec::new(),
            max_buf_size: 8 * 1024 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug)]
pub struct QuantizeLimits {
    pub input: ShardLimits,
    /// Hard bound on live source/decoded/output slabs and packed worker rows. Scalar encoder
    /// workspaces, metadata, importance rows and OS cache are separate.
    pub max_work_bytes: usize,
    pub max_threads: usize,
    pub max_importance_values: usize,
    pub max_overrides: usize,
}
impl Default for QuantizeLimits {
    fn default() -> Self {
        Self {
            input: ShardLimits::default(),
            max_work_bytes: 256 * 1024 * 1024,
            max_threads: 64,
            max_importance_values: 128 * 1024 * 1024,
            max_overrides: 4096,
        }
    }
}
pub(crate) fn is_quantized(t: DType) -> bool {
    t.block_size() > 1
}
pub(crate) fn requires_imatrix(name: &str, t: DType, f: FileType) -> bool {
    if matches!(
        name,
        "token_embd.weight" | "per_layer_token_embd.weight" | "output.weight"
    ) {
        return false;
    }
    matches!(
        t,
        DType::IQ3_XXS
            | DType::IQ2_XXS
            | DType::IQ2_XS
            | DType::IQ2_S
            | DType::IQ1_M
            | DType::IQ1_S
    ) || (t == DType::Q2_K && f.id() == 21)
}
