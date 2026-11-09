//! Rust implementations of the sampling, native chat, grammar and shared data
//! behavior in the frozen llama.cpp snapshot.
//!
//! This crate does not load model weights or execute a model. Its public types
//! are a local integration seam pending the T01/T04/T05 workspace contracts.
#![forbid(unsafe_code)]

pub mod chat;
pub mod grammar;
pub mod json;
pub mod sampling;
pub mod speculative;

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
