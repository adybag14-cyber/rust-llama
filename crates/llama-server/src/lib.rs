//! Bounded HTTP/SSE transport over the Rust inference runtime.
//!
//! [`RuntimeProvider`] loads an explicitly supported GGUF through `llama-inference`.
//! The executable connects it with `--model`; without a model, [`UnavailableProvider`]
//! exposes diagnostics and reports unavailable inference. Other embedders can supply
//! their own [`InferenceProvider`].
#![forbid(unsafe_code)]

pub mod client;
pub mod http;
pub mod provider;
pub mod runtime;
pub mod server;
pub mod stream;

pub use provider::{
    GenerationRequest, GenerationSummary, InferenceProvider, ModelInfo, Prompt, ProviderError,
    ProviderStatus, TokenSink, UnavailableProvider,
};
pub use runtime::RuntimeProvider;
pub use server::{RunningServer, Server, ServerConfig};
pub use stream::CancellationToken;
