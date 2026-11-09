//! Adapt actual GGUF inference to the bounded server provider contract.
//!
//! Weights are immutable and shared by `Model`; each generation owns its context,
//! sampler and cancellation state. No reference executable or test provider is
//! called by this module.
use crate::provider::{
    EmbeddingInfo, FinishReason, GenerationRequest, GenerationSummary, InferenceProvider,
    ModelInfo, Prompt, ProviderError, ProviderStatus, TokenSink,
};
use crate::stream::CancellationToken;
use llama_behavior::json::Value;
use llama_inference::{Error, GenerationOptions, LoadOptions, Model, Pooling};
use std::path::Path;

/// Omitted seeds are deterministic until an explicit platform entropy contract
/// is implemented. The original ambient-random sentinel is never silently
/// interpreted as an ordinary seed.
pub const DEFAULT_SEED: u32 = 0;

/// A loaded, validated Rust model. Loading errors propagate to the caller; an
/// invalid model never becomes a ready provider or a synthetic fallback.
pub struct RuntimeProvider {
    model: Model,
    embedding: Option<EmbeddingOptions>,
}

/// Startup options corresponding to --embeddings, --pooling and
/// --embd-normalize. An omitted pooling mode uses the GGUF model default.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingOptions {
    pub pooling: Option<Pooling>,
    pub normalization: i32,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            pooling: None,
            normalization: 2,
        }
    }
}

impl RuntimeProvider {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ProviderError> {
        Self::load_with_options(path, &LoadOptions::default())
    }

    /// Load a model with explicit decoded-weight and context bounds. The weight
    /// budget excludes packed tensors and runtime state; it is not an RSS limit.
    pub fn load_with_options(
        path: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<Self, ProviderError> {
        Model::load_with_options(path, options)
            .map(Self::from_model)
            .map_err(map_error)
    }

    pub fn from_model(model: Model) -> Self {
        Self {
            model,
            embedding: None,
        }
    }

    pub fn with_embeddings(mut self, options: EmbeddingOptions) -> Result<Self, ProviderError> {
        if !self.model.supports_embeddings() {
            return Err(ProviderError::unsupported(
                "Embedding execution is not implemented for this model architecture",
            ));
        }
        self.embedding = Some(options);
        Ok(self)
    }

    fn embedding_pooling(&self) -> Result<Pooling, ProviderError> {
        let options = self.embedding.ok_or_else(|| {
            ProviderError::unsupported(
                "This server does not support embeddings. Start it with --embeddings",
            )
        })?;
        Ok(options
            .pooling
            .or(self.model.default_embedding_pooling())
            .unwrap_or(Pooling::None))
    }

    pub fn model_id(&self) -> &str {
        self.model.model_id()
    }

    fn prompt(&self, request: &GenerationRequest) -> Result<String, ProviderError> {
        match &request.prompt {
            Prompt::Text(text) => Ok(text.clone()),
            Prompt::Chat(messages) => {
                if messages.is_empty() {
                    return Err(ProviderError::invalid("messages must not be empty"));
                }
                let messages: Vec<(&str, &str)> = messages
                    .iter()
                    .map(|message| (message.role.as_str(), message.content.as_str()))
                    .collect();
                self.model.render_chat(&messages, true).map_err(map_error)
            }
        }
    }

    fn options(&self, request: &GenerationRequest) -> Result<GenerationOptions, ProviderError> {
        if !self.model.supports_generation() {
            return Err(ProviderError::unsupported(
                "loaded model provides embeddings; text generation is unavailable",
            ));
        }
        if request
            .model
            .as_deref()
            .is_some_and(|id| id != self.model_id())
        {
            return Err(ProviderError::new(
                404,
                "model_not_found",
                "Requested model is not available",
            ));
        }
        generation_options(request)
    }
}

fn generation_options(request: &GenerationRequest) -> Result<GenerationOptions, ProviderError> {
    // The transport owns these fields. Any other option must acquire a real
    // runtime implementation before the adapter accepts it.
    let parameters = request
        .parameters
        .as_object()
        .ok_or_else(|| ProviderError::invalid("Request parameters must be an object"))?;
    for (key, value) in parameters {
        match key.as_str() {
            "prompt"
            | "messages"
            | "model"
            | "max_tokens"
            | "max_completion_tokens"
            | "n_predict"
            | "temperature"
            | "seed"
            | "stop"
            | "stream"
            | "n" => {}
            "tools" | "tool_choice" | "functions" | "function_call" | "response_format"
            | "logprobs" | "top_logprobs" | "stream_options"
                if value.is_null() => {}
            _ => {
                return Err(ProviderError::unsupported(format!(
                    "The Rust model provider does not implement request option {key}"
                )));
            }
        }
    }
    // Do not silently choose one of contradictory aliases.
    for key in ["max_tokens", "max_completion_tokens", "n_predict"] {
        if let Some(value) = parameters.get(key)
            && value.as_u64() != Some(u64::from(request.max_tokens))
        {
            return Err(ProviderError::invalid(
                "Token-limit aliases must be valid integers with the same value",
            ));
        }
    }
    // Message attributes such as names or tool calls affect formatting. The
    // current native formatter accepts only role and text, so do not drop them.
    if let Some(messages) = parameters.get("messages").and_then(Value::as_array) {
        for message in messages {
            let fields = message
                .as_object()
                .ok_or_else(|| ProviderError::invalid("Each message must be an object"))?;
            for key in fields.keys() {
                if !matches!(key.as_str(), "role" | "content") {
                    return Err(ProviderError::unsupported(format!(
                        "The Rust model provider does not implement message attribute {key}"
                    )));
                }
            }
        }
    }
    if request.max_tokens == 0 || request.max_tokens > 1_000_000 {
        return Err(ProviderError::invalid(
            "Token limit must be an integer from 1 through 1000000",
        ));
    }
    let temperature = request.temperature as f32;
    if !request.temperature.is_finite()
        || request.temperature < 0.0
        || !temperature.is_finite()
        || (request.temperature > 0.0 && temperature == 0.0)
    {
        return Err(ProviderError::invalid(
            "temperature must be finite, nonnegative and representable as f32",
        ));
    }
    let seed = request
        .seed
        .map(u32::try_from)
        .transpose()
        .map_err(|_| ProviderError::invalid("seed must fit in u32 without truncation"))?
        .unwrap_or(DEFAULT_SEED);
    if seed == u32::MAX {
        return Err(ProviderError::unsupported(
            "The ambient-random seed sentinel is not implemented; use an explicit seed below 4294967295",
        ));
    }
    if request.stop.len() > 64 || request.stop.iter().any(String::is_empty) {
        return Err(ProviderError::invalid(
            "stop accepts at most 64 nonempty strings",
        ));
    }
    Ok(GenerationOptions {
        max_tokens: request.max_tokens as usize,
        temperature,
        seed,
        stop: request.stop.clone(),
    })
}

fn map_error(error: Error) -> ProviderError {
    match error {
        Error::Invalid(message) => ProviderError::invalid(message),
        Error::Unsupported(message) => ProviderError::unsupported(message),
        Error::Cancelled => ProviderError::cancelled(),
        Error::Internal(message) => ProviderError::internal(message),
        Error::Io(error) => ProviderError::internal(format!("Model IO failed: {error}")),
    }
}

impl InferenceProvider for RuntimeProvider {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::Ready
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: self.model_id().to_owned(),
            owned_by: "rust-llama".into(),
        }]
    }

    fn embedding_info(&self) -> Result<EmbeddingInfo, ProviderError> {
        let pooling = self.embedding_pooling()?;
        Ok(EmbeddingInfo {
            dimensions: self.model.embedding_dimensions(),
            vocabulary_size: self.model.vocabulary_size(),
            context_length: self.model.context_length(),
            pooled: pooling != Pooling::None,
            normalization: self.embedding.map_or(2, |options| options.normalization),
        })
    }

    fn embed(
        &self,
        tokens: &[i32],
        normalization: i32,
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        let pooling = self.embedding_pooling()?;
        let encoded = self
            .model
            .embed_tokens(tokens, pooling, &|| cancel.is_cancelled())
            .map_err(map_error)?;
        match encoded.pooled {
            Some(mut vector) => {
                crate::embedding::normalize(&mut vector, normalization)?;
                Ok(vec![vector])
            }
            None => Ok(encoded.tokens),
        }
    }

    fn validate_generation(&self, request: &GenerationRequest) -> Result<(), ProviderError> {
        self.options(request)?;
        let prompt = self.prompt(request)?;
        let tokens = self.model.tokenize(&prompt, true).map_err(map_error)?;
        if tokens.is_empty() {
            return Err(ProviderError::invalid("empty prompt with no BOS token"));
        }
        if tokens.len() > self.model.context_length() {
            return Err(ProviderError::invalid(format!(
                "prompt has {} tokens; context limit is {}",
                tokens.len(),
                self.model.context_length()
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
        cancel: &CancellationToken,
    ) -> Result<GenerationSummary, ProviderError> {
        if cancel.is_cancelled() {
            return Err(ProviderError::cancelled());
        }
        let options = self.options(request)?;
        let prompt = self.prompt(request)?;
        let mut sink_error = None;
        let result =
            self.model
                .generate(&prompt, &options, &|| cancel.is_cancelled(), &mut |text| {
                    if cancel.is_cancelled() || sink_error.is_some() {
                        return false;
                    }
                    match sink.send(text) {
                        Ok(()) => true,
                        Err(error) => {
                            sink_error = Some(error);
                            false
                        }
                    }
                });
        // A failed network sink has a more precise transport error than the
        // inference callback's generic cancellation signal.
        if let Some(error) = sink_error {
            return Err(error);
        }
        if cancel.is_cancelled() {
            return Err(ProviderError::cancelled());
        }
        let stats = result.map_err(map_error)?;
        Ok(GenerationSummary {
            finish_reason: match stats.finish_reason {
                llama_inference::FinishReason::Eos => FinishReason::Stop,
                llama_inference::FinishReason::StopSequence => FinishReason::StopSequence,
                llama_inference::FinishReason::Length => FinishReason::Length,
            },
            prompt_tokens: stats.prompt_tokens as u64,
            completion_tokens: stats.completion_tokens as u64,
        })
    }

    fn apply_template(
        &self,
        messages: &[crate::provider::ChatMessage],
        add_assistant: bool,
    ) -> Result<String, ProviderError> {
        let messages: Vec<(&str, &str)> = messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect();
        self.model
            .render_chat(&messages, add_assistant)
            .map_err(map_error)
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>, ProviderError> {
        self.model.tokenize(text, add_special).map_err(map_error)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String, ProviderError> {
        self.model.detokenize(tokens).map_err(map_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_behavior::json;

    fn request(parameters: &str) -> GenerationRequest {
        GenerationRequest {
            model: None,
            prompt: Prompt::Text("test".into()),
            max_tokens: 16,
            temperature: 0.0,
            seed: None,
            stop: Vec::new(),
            parameters: json::parse(parameters).unwrap(),
        }
    }

    #[test]
    fn options_preserve_limits_stops_and_deterministic_seed() {
        let mut request = request(r#"{"max_tokens":16,"n_predict":16,"stream":true}"#);
        request.stop = vec!["END".into(), "ç•Œ".into()];
        let options = generation_options(&request).unwrap();
        assert_eq!(options.max_tokens, 16);
        assert_eq!(options.temperature, 0.0);
        assert_eq!(options.seed, DEFAULT_SEED);
        assert_eq!(options.stop, request.stop);
    }

    #[test]
    fn unimplemented_sampling_and_message_options_are_rejected() {
        for json in [
            r#"{"top_p":0.9}"#,
            r#"{"repeat_penalty":1.1}"#,
            r#"{"messages":[{"role":"user","content":"Hi","name":"Alice"}]}"#,
            r#"{"logprobs":true}"#,
        ] {
            let error = generation_options(&request(json)).unwrap_err();
            assert_eq!(error.status, 501, "{json}");
        }
        assert!(generation_options(&request(r#"{"logprobs":null}"#)).is_ok());
    }

    #[test]
    fn aliases_seed_and_temperature_never_silently_truncate() {
        let invalid = request(r#"{"max_tokens":16,"n_predict":17}"#);
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
        let mut invalid = request("{}");
        invalid.seed = Some(u64::MAX);
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
        invalid.seed = Some(u64::from(u32::MAX));
        assert_eq!(generation_options(&invalid).unwrap_err().status, 501);
        invalid.seed = Some(42);
        invalid.temperature = f64::MAX;
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
        invalid.temperature = -1.0;
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
    }

    #[test]
    fn invalid_stop_and_token_budgets_are_errors() {
        let mut invalid = request("{}");
        invalid.max_tokens = 0;
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
        invalid.max_tokens = 16;
        invalid.stop = vec![String::new()];
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
        invalid.stop = vec!["x".into(); 65];
        assert_eq!(generation_options(&invalid).unwrap_err().status, 400);
    }
}
