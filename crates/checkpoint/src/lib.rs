//! Owned local Safetensors ranges and indexed checkpoints from the frozen converter.
//!
//! Raw metadata and byte-range access are distinct from typed tensor validation.
//! The source's safe noncanonical ranges, duplicate-key behavior and name order
//! are preserved. No Python, Torch, mmap or reference implementation is linked.
#![deny(unsafe_code)]

mod file;
mod index;
pub mod json;
#[allow(unsafe_code)]
mod platform;

pub use file::{DType, RawTensorInfo, SafetensorsFile, TensorReader, TypedLayout};
pub use index::{Checkpoint, CheckpointOptions, NamePolicy, TensorRef};
pub use json::{Integer, Number, Object, Text, Value};

use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(&'static str),
    Limit(&'static str),
    Path(&'static str),
    Unsupported(&'static str),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "checkpoint I/O: {e}"),
            Self::Invalid(s) => write!(f, "invalid checkpoint: {s}"),
            Self::Limit(s) => write!(f, "checkpoint limit: {s}"),
            Self::Path(s) => write!(f, "checkpoint path: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported checkpoint operation: {s}"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Work/input limits, shared cumulatively by every file in one checkpoint.
/// These bound encoded bytes, decoded text, parsed nodes and owned descriptors;
/// they are not a total-process allocator or payload-cache budget.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_files: usize,
    pub max_header_bytes: u64,
    pub max_total_header_bytes: u64,
    pub max_index_bytes: u64,
    pub max_json_nodes: usize,
    pub max_text_bytes: usize,
    pub max_string_bytes: usize,
    pub max_json_depth: usize,
    pub max_integer_digits: usize,
    pub max_tensors: usize,
    pub max_dimensions: usize,
    pub max_directory_entries: usize,
    pub max_path_bytes: usize,
    pub max_path_components: usize,
    pub max_read_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_files: 1024,
            max_header_bytes: 100_000_000,
            max_total_header_bytes: 256 * 1024 * 1024,
            max_index_bytes: 16 * 1024 * 1024,
            max_json_nodes: 4_000_000,
            max_text_bytes: 256 * 1024 * 1024,
            max_string_bytes: 16 * 1024 * 1024,
            max_json_depth: 256,
            max_integer_digits: 4300,
            max_tensors: 100_000,
            max_dimensions: 1_000_000,
            max_directory_entries: 100_000,
            max_path_bytes: 64 * 1024,
            max_path_components: 256,
            max_read_bytes: 256 * 1024 * 1024,
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub files: usize,
    pub header_bytes: u64,
    pub index_bytes: u64,
    pub json_nodes: usize,
    pub text_bytes: usize,
    pub tensors: usize,
    pub dimensions: usize,
    pub directory_entries: usize,
}
pub(crate) struct Budget {
    pub limits: Limits,
    pub used: Usage,
}
impl Budget {
    pub fn new(limits: Limits) -> Result<Self> {
        if limits.max_json_depth > 512 {
            return Err(Error::Limit(
                "JSON depth configuration exceeds the checked recursion ceiling",
            ));
        }
        Ok(Self {
            limits,
            used: Usage::default(),
        })
    }
    pub fn files(&mut self, n: usize) -> Result<()> {
        charge(
            &mut self.used.files,
            n,
            self.limits.max_files,
            "file-owner count",
        )
    }
    pub fn header(&mut self, n: u64) -> Result<()> {
        if n > self.limits.max_header_bytes {
            return Err(Error::Limit("individual header bytes"));
        }
        self.used.header_bytes = self
            .used
            .header_bytes
            .checked_add(n)
            .filter(|x| *x <= self.limits.max_total_header_bytes)
            .ok_or(Error::Limit("cumulative header bytes"))?;
        Ok(())
    }
    pub fn index(&mut self, n: u64) -> Result<()> {
        self.used.index_bytes = self
            .used
            .index_bytes
            .checked_add(n)
            .filter(|x| *x <= self.limits.max_index_bytes)
            .ok_or(Error::Limit("index bytes"))?;
        Ok(())
    }
    pub fn node(&mut self) -> Result<()> {
        charge(
            &mut self.used.json_nodes,
            1,
            self.limits.max_json_nodes,
            "JSON nodes",
        )
    }
    pub fn text(&mut self, n: usize) -> Result<()> {
        charge(
            &mut self.used.text_bytes,
            n,
            self.limits.max_text_bytes,
            "decoded/owned text bytes",
        )
    }
    pub fn tensors(&mut self, n: usize) -> Result<()> {
        charge(
            &mut self.used.tensors,
            n,
            self.limits.max_tensors,
            "tensor descriptors",
        )
    }
    pub fn dimensions(&mut self, n: usize) -> Result<()> {
        charge(
            &mut self.used.dimensions,
            n,
            self.limits.max_dimensions,
            "shape items",
        )
    }
    pub fn entry(&mut self) -> Result<()> {
        charge(
            &mut self.used.directory_entries,
            1,
            self.limits.max_directory_entries,
            "directory entries",
        )
    }
}
fn charge(used: &mut usize, n: usize, max: usize, label: &'static str) -> Result<()> {
    *used = used
        .checked_add(n)
        .filter(|v| *v <= max)
        .ok_or(Error::Limit(label))?;
    Ok(())
}
pub(crate) fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    label: &'static str,
) -> Result<()> {
    values
        .try_reserve(additional)
        .map_err(|_| Error::Limit(label))
}

#[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
mod tests;
