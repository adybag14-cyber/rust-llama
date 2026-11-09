//! Independent validation and release utilities for the Rust rewrite.
//!
//! This crate is tooling, not a production inference dependency. Original source
//! is always supplied explicitly by path and is never linked into a Rust product.
#![forbid(unsafe_code)]

pub mod benchmark;
pub mod compare;
pub mod digest;
pub mod inventory;
pub mod package;
pub mod process;

pub mod fixtures;
