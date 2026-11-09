use crate::stream::CancellationToken;
use llama_behavior::json::Value;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderStatus {
    Ready,
    Unavailable(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub owned_by: String,
}

/// Fixed embedding capabilities of the currently loaded provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingInfo {
    pub dimensions: usize,
    pub vocabulary_size: usize,
    pub context_length: usize,
    pub pooled: bool,
    pub normalization: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prompt {
    Text(String),
    Chat(Vec<ChatMessage>),
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub model: Option<String>,
    pub prompt: Prompt,
    pub max_tokens: u32,
    pub temperature: f64,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    /// The complete parsed request. A provider can explicitly handle additional
    /// options rather than losing unknown keys at the transport boundary.
    pub parameters: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    /// A user-supplied stop sequence, distinct from the model end token.
    StopSequence,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop | Self::StopSequence => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationSummary {
    pub finish_reason: FinishReason,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
}

impl ProviderError {
    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
        }
    }
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(503, "model_unavailable", message)
    }
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(501, "not_implemented", message)
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request_error", message)
    }
    pub fn cancelled() -> Self {
        Self::new(499, "cancelled", "Generation was cancelled")
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(500, "server_error", message)
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for ProviderError {}

/// Providers emit valid UTF-8 text through a bounded transport sink. Sending can
/// block under backpressure; errors and cancellation must stop generation.
pub trait TokenSink {
    fn send(&mut self, text: &str) -> Result<(), ProviderError>;
}

/// Synchronous integration boundary for T05. Implementations must honor `cancel`
/// between decoding steps and return promptly when `sink.send` fails. Transport
/// limits bound admission, bytes, and queues; they cannot forcibly kill arbitrary
/// provider code. Implementations must validate the `parameters` they support.
pub trait InferenceProvider: Send + Sync + 'static {
    fn status(&self) -> ProviderStatus;
    fn models(&self) -> Vec<ModelInfo>;
    /// Validate embedding availability without executing a model.
    fn embedding_info(&self) -> Result<EmbeddingInfo, ProviderError> {
        Err(ProviderError::unsupported(
            "This provider does not implement embeddings",
        ))
    }
    /// Encode one complete token sequence. Return one vector for a pooled
    /// provider, or one vector per input token when pooling is disabled.
    /// Honor cancellation during model execution. Normalize only pooled vectors.
    fn embed(
        &self,
        _tokens: &[i32],
        _normalization: i32,
        _cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        Err(ProviderError::unsupported(
            "This provider does not implement embeddings",
        ))
    }
    /// Validate runtime-owned options before response headers or inference admission.
    /// This must remain bounded and must not run model decoding.
    fn validate_generation(&self, _request: &GenerationRequest) -> Result<(), ProviderError> {
        Ok(())
    }
    fn generate(
        &self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
        cancel: &CancellationToken,
    ) -> Result<GenerationSummary, ProviderError>;
    fn apply_template(
        &self,
        _messages: &[ChatMessage],
        _add_assistant: bool,
    ) -> Result<String, ProviderError> {
        Err(ProviderError::unsupported(
            "This provider does not implement a chat template",
        ))
    }
    fn tokenize(&self, _text: &str, _add_special: bool) -> Result<Vec<i32>, ProviderError> {
        Err(ProviderError::unsupported(
            "This provider does not implement tokenization",
        ))
    }
    fn detokenize(&self, _tokens: &[i32]) -> Result<String, ProviderError> {
        Err(ProviderError::unsupported(
            "This provider does not implement detokenization",
        ))
    }
}

#[derive(Default)]
pub struct UnavailableProvider;
impl InferenceProvider for UnavailableProvider {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::Unavailable(
            "No Rust model runtime is connected; T01–T08 integration is pending".into(),
        )
    }
    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }
    fn generate(
        &self,
        _: &GenerationRequest,
        _: &mut dyn TokenSink,
        _: &CancellationToken,
    ) -> Result<GenerationSummary, ProviderError> {
        Err(ProviderError::unavailable("No model runtime is connected"))
    }
    fn tokenize(&self, _: &str, _: bool) -> Result<Vec<i32>, ProviderError> {
        Err(ProviderError::unavailable(
            "No model tokenizer is connected",
        ))
    }
    fn detokenize(&self, _: &[i32]) -> Result<String, ProviderError> {
        Err(ProviderError::unavailable(
            "No model tokenizer is connected",
        ))
    }
}
