//! Rust-authored media preprocessing for the frozen llama.cpp rewrite.
//!
//! This crate implements a bounded preprocessing slice. It does not load model
//! weights, execute a projector, or claim the full codec set of stb/miniaudio/
//! FFmpeg. See `docs/t09-status.md` for the exact capability boundary.
//!
//! No foreign code, native codec, process, or external crate is used at runtime.
#![forbid(unsafe_code)]

pub mod audio;
pub mod image;
pub mod input;
mod png;
pub mod spectral;

use std::fmt;

/// Errors are recoverable: malformed media never requires an assertion/panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaError {
    Invalid(&'static str),
    Unsupported(&'static str),
    Truncated {
        offset: usize,
        required: usize,
    },
    Limit {
        resource: &'static str,
        requested: u64,
        maximum: u64,
    },
    Overflow,
    Allocation,
    NonFinite,
    State(&'static str),
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid media: {why}"),
            Self::Unsupported(why) => write!(f, "media capability pending: {why}"),
            Self::Truncated { offset, required } => {
                write!(f, "truncated at byte {offset}: need {required} bytes")
            }
            Self::Limit {
                resource,
                requested,
                maximum,
            } => write!(
                f,
                "{resource} limit: requested {requested}, maximum {maximum}"
            ),
            Self::Overflow => f.write_str("media size arithmetic overflow"),
            Self::Allocation => f.write_str("media allocation failed"),
            Self::NonFinite => f.write_str("nonfinite media sample or parameter"),
            Self::State(why) => write!(f, "invalid media state: {why}"),
        }
    }
}
impl std::error::Error for MediaError {}
pub type Result<T> = std::result::Result<T, MediaError>;

/// Resource policy for untrusted inputs. Callers may choose tighter limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    pub max_input_bytes: usize,
    pub max_image_pixels: usize,
    pub max_audio_samples: usize,
}
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 128 * 1024 * 1024,
            max_image_pixels: 32 * 1024 * 1024,
            max_audio_samples: 32 * 1024 * 1024,
        }
    }
}

/// Single intermediate allocation ceiling; protects transforms as well as IO.
pub const MAX_ALLOCATION_BYTES: usize = 256 * 1024 * 1024;

pub(crate) fn check_limit(resource: &'static str, value: usize, maximum: usize) -> Result<()> {
    if value > maximum {
        Err(MediaError::Limit {
            resource,
            requested: value as u64,
            maximum: maximum as u64,
        })
    } else {
        Ok(())
    }
}
pub(crate) fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(MediaError::Overflow)
}
pub(crate) fn zeros<T: Clone>(len: usize, zero: T) -> Result<Vec<T>> {
    check_limit(
        "allocation bytes",
        product(len, std::mem::size_of::<T>())?,
        MAX_ALLOCATION_BYTES,
    )?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| MediaError::Allocation)?;
    out.resize(len, zero);
    Ok(out)
}
pub(crate) fn bytes(input: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    input
        .get(at..at.checked_add(len).ok_or(MediaError::Overflow)?)
        .ok_or(MediaError::Truncated {
            offset: at,
            required: len,
        })
}
pub(crate) fn u16le(input: &[u8], at: usize) -> Result<u16> {
    let b = bytes(input, at, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}
pub(crate) fn u32le(input: &[u8], at: usize) -> Result<u32> {
    let b = bytes(input, at, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
