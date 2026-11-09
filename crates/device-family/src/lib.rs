//! Rust-owned progress toward the CUDA, HIP and MUSA backends (track T06).
//!
//! Device discovery is real. An absent driver is an explicit unavailable
//! result. CUDA PTX is generated from the typed instruction representation in
//! [`ptx`]; HIP and MUSA code generation is not falsely inferred from CUDA.
//! This crate does not link any legacy ggml implementation.
pub mod driver;
pub mod planning;
mod probe;
pub mod ptx;
pub mod runtime;

pub use probe::{probe, probe_all};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Cuda,
    Hip,
    Musa,
}
impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cuda => "CUDA",
            Self::Hip => "HIP",
            Self::Musa => "MUSA",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub memory_bytes: usize,
    /// Queried only with a verified family-specific ABI value; never assumed.
    pub subgroup_width: Option<u32>,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    Available,
    Unavailable,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub family: Family,
    pub library: Option<String>,
    pub driver_version: Option<i32>,
    pub devices: Vec<DeviceInfo>,
    pub status: ProbeStatus,
    /// Enumeration is not execution proof. This field says exactly what ran.
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    DriverUnavailable(String),
    MissingSymbol(String),
    Api {
        family: Family,
        operation: &'static str,
        code: i32,
    },
    InvalidArgument(String),
    Unsupported(String),
}
impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverUnavailable(s) => write!(f, "driver unavailable: {s}"),
            Self::MissingSymbol(s) => write!(f, "driver symbol unavailable: {s}"),
            Self::Api {
                family,
                operation,
                code,
            } => write!(f, "{family} {operation} returned {code}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Self::Unsupported(s) => write!(f, "not implemented: {s}"),
        }
    }
}
impl std::error::Error for DeviceError {}
pub(crate) fn check(family: Family, operation: &'static str, code: i32) -> Result<(), DeviceError> {
    if code == 0 {
        Ok(())
    } else {
        Err(DeviceError::Api {
            family,
            operation,
            code,
        })
    }
}

pub(crate) fn checked_range(offset: usize, len: usize, total: usize) -> Result<(), DeviceError> {
    if offset.checked_add(len).is_some_and(|end| end <= total) {
        Ok(())
    } else {
        Err(DeviceError::InvalidArgument(
            "buffer range exceeds allocation".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_cannot_wrap_or_escape() {
        assert!(checked_range(3, 5, 8).is_ok());
        assert!(checked_range(8, 0, 8).is_ok());
        assert!(checked_range(8, 1, 8).is_err());
        assert!(checked_range(usize::MAX, 2, usize::MAX).is_err());
    }
}
