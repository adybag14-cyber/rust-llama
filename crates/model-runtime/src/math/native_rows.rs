//! Compatibility facade for the shared safe scalar/SSE2 CPU row routines.
//!
//! Tensor and inference use one maintained implementation. This module preserves
//! the existing model-runtime public API path.
pub use llama_quant::cpu_math::*;
