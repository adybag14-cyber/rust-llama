//! Native Rust GGUF -> tokenizer -> model -> sampler integration.
//! Model paths: dense Llama/Qwen2, Mamba-1 recurrence and dense BERT embeddings.
//! Unsupported architectures and configuration modes fail before generation.
#![forbid(unsafe_code)]

mod bert_loader;
mod loader;
mod mamba_loader;
mod qwen2_loader;
mod stream;
use llama_behavior::{
    chat,
    sampling::{Chain, Config},
};
use llama_model_runtime::{BatchToken, Context, ContextOptions, Input, OutputSelection};
pub use llama_model_runtime::{Encoded, Pooling};
use llama_tokenizer::Tokenizer;
use std::{fmt, path::Path, sync::Arc};

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
    Cancelled,
    Internal(String),
    Io(std::io::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid input: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Cancelled => f.write_str("generation cancelled"),
            Self::Internal(s) => write!(f, "inference error: {s}"),
            Self::Io(e) => write!(f, "{e}"),
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
        match e {
            llama_gguf::Error::Io(e) => Self::Io(e),
            llama_gguf::Error::Unsupported(s) => Self::Unsupported(s),
            e => Self::Invalid(e.to_string()),
        }
    }
}
impl From<llama_tokenizer::Error> for Error {
    fn from(e: llama_tokenizer::Error) -> Self {
        match e {
            llama_tokenizer::Error::Unsupported(s) => Self::Unsupported(s),
            e => Self::Invalid(e.to_string()),
        }
    }
}
impl From<llama_model_runtime::Error> for Error {
    fn from(e: llama_model_runtime::Error) -> Self {
        use llama_model_runtime::Error as E;
        match e {
            E::Cancelled => Self::Cancelled,
            E::Unsupported(s) | E::UnsupportedArchitecture(s) => Self::Unsupported(s),
            E::InvalidToken { .. } | E::ContextFull { .. } | E::InvalidConfig(_) | E::Shape(_) => {
                Self::Invalid(e.to_string())
            }
            e => Self::Internal(e.to_string()),
        }
    }
}
impl From<llama_behavior::Error> for Error {
    fn from(e: llama_behavior::Error) -> Self {
        Self::Invalid(e.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Decoded F32 weight budget, checked before reading/allocating the weights.
    pub max_decoded_weight_bytes: usize,
    /// Requested context, capped at the model's training context. No scaling.
    pub context_length: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_decoded_weight_bytes: 2 * 1024 * 1024 * 1024,
            context_length: 1024,
        }
    }
}
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: u32,
    pub stop: Vec<String>,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.,
            seed: 0,
            stop: Vec::new(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    StopSequence,
    Length,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub(crate) enum Executor {
    Decoder(Arc<llama_model_runtime::Model>),
    Bert(Arc<llama_model_runtime::BertModel>),
}

#[derive(Debug, Clone)]
pub struct Model {
    pub(crate) runtime: Executor,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) id: String,
    pub(crate) template: Option<String>,
    pub(crate) decoded_weight_bytes: usize,
    pub(crate) context_limit: usize,
    pub(crate) embedding_pooling: Option<Pooling>,
}
impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(path, &LoadOptions::default())
    }
    pub fn load_with_options(path: impl AsRef<Path>, options: &LoadOptions) -> Result<Self> {
        loader::load(path.as_ref(), options)
    }
    pub fn model_id(&self) -> &str {
        &self.id
    }
    pub fn vocabulary_size(&self) -> usize {
        self.tokenizer.vocabulary_size()
    }
    pub fn context_length(&self) -> usize {
        self.context_limit
    }
    /// Whether the loaded architecture supports autoregressive logits/generation.
    pub fn supports_generation(&self) -> bool {
        matches!(self.runtime, Executor::Decoder(_))
    }
    pub fn supports_embeddings(&self) -> bool {
        matches!(self.runtime, Executor::Bert(_))
    }
    pub fn embedding_dimensions(&self) -> usize {
        match &self.runtime {
            Executor::Decoder(model) => model.width(),
            Executor::Bert(model) => model.config().width,
        }
    }
    pub fn default_embedding_pooling(&self) -> Option<Pooling> {
        self.embedding_pooling
    }
    /// Raw final-layer token embeddings, with optional raw pooling.
    /// No normalization is applied. The complete sequence is encoded together.
    pub fn embed_tokens(
        &self,
        tokens: &[i32],
        pooling: Pooling,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Encoded> {
        if cancel() {
            return Err(Error::Cancelled);
        }
        let Executor::Bert(model) = &self.runtime else {
            return Err(Error::Unsupported(
                "embedding API currently requires a dense BERT model".into(),
            ));
        };
        if tokens.len() > self.context_length() {
            return Err(Error::Invalid(
                "token sequence exceeds model context".into(),
            ));
        }
        let ids: Vec<u32> = tokens
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| Error::Invalid("negative token ID".into())))
            .collect::<Result<_>>()?;
        let positions: Vec<usize> = (0..ids.len()).collect();
        Ok(model.encode_positions_with_cancel(&ids, &positions, pooling, cancel)?)
    }
    pub fn embed(
        &self,
        text: &str,
        pooling: Option<Pooling>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Encoded> {
        if cancel() {
            return Err(Error::Cancelled);
        }
        let tokens = self.tokenize(text, true)?;
        self.embed_tokens(
            &tokens,
            pooling.or(self.embedding_pooling).unwrap_or(Pooling::None),
            cancel,
        )
    }
    pub fn decoded_weight_bytes(&self) -> usize {
        self.decoded_weight_bytes
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    /// Share immutable model weights with a separately owned runtime context.
    pub fn runtime_model(&self) -> Result<Arc<llama_model_runtime::Model>> {
        match &self.runtime {
            Executor::Decoder(model) => Ok(model.clone()),
            Executor::Bert(_) => Err(Error::Unsupported(
                "BERT requires complete-sequence encoding, not a decoder context".into(),
            )),
        }
    }
    /// Native server default: parse special strings, independent of BOS/EOS insertion.
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        Ok(self.tokenizer.tokenize(text, add_special, true)?)
    }
    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        String::from_utf8(self.tokenizer.detokenize_bytes(tokens, true, false)?).map_err(|_| {
            Error::Invalid(
                "tokens do not form complete UTF-8; use tokenizer().detokenize_bytes for raw bytes"
                    .into(),
            )
        })
    }
    pub fn render_chat(&self, messages: &[(&str, &str)], add_assistant: bool) -> Result<String> {
        let template = self.template.as_deref().ok_or_else(|| {
            Error::Unsupported("model has no chat template; use a text completion prompt".into())
        })?;
        if chat::detect_template(template).is_none() {
            return Err(Error::Unsupported("model requires a chat template not implemented by the native formatter; full Jinja is pending".into()));
        }
        let messages: Vec<_> = messages
            .iter()
            .map(|(role, content)| chat::Message::new(*role, *content))
            .collect();
        Ok(chat::render(template, &messages, add_assistant)?)
    }
    fn context(&self) -> Result<Context> {
        Ok(Context::new(
            self.runtime_model()?,
            ContextOptions {
                maximum_sequences: 1,
                microbatch_tokens: 32,
                maximum_batch_tokens: 512,
                ..ContextOptions::default()
            },
        )?)
    }
    /// Observable full logits for conformance and embedding/runtime integration.
    pub fn logits_for_tokens(&self, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
        if !self.supports_generation() {
            return Err(Error::Unsupported(
                "BERT produces embeddings, not autoregressive logits".into(),
            ));
        }
        if tokens.len() > self.context_length() {
            return Err(Error::Invalid(
                "token sequence exceeds model context".into(),
            ));
        }
        let mut context = self.context()?;
        let mut result = Vec::new();
        for (position, &id) in tokens.iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| Error::Invalid("negative token ID".into()))?;
            let batch = [BatchToken {
                sequence: 0,
                position,
                input: Input::Token(id),
                output: OutputSelection::Logits,
            }];
            let output = context.decode_batch(&batch)?;
            result.push(
                output
                    .into_iter()
                    .next()
                    .and_then(|v| v.logits)
                    .ok_or_else(|| Error::Internal("missing logits".into()))?,
            )
        }
        Ok(result)
    }
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerationOptions,
        cancel: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(&str) -> bool,
    ) -> Result<GenerationStats> {
        if cancel() {
            return Err(Error::Cancelled);
        }
        if !self.supports_generation() {
            return Err(Error::Unsupported(
                "BERT is an embedding model; generation is unavailable".into(),
            ));
        }
        if !options.temperature.is_finite() || options.temperature < 0. {
            return Err(Error::Invalid(
                "temperature must be finite and nonnegative".into(),
            ));
        }
        if options.max_tokens > 1_000_000 {
            return Err(Error::Invalid("max_tokens exceeds 1000000".into()));
        }
        if options.stop.len() > 256
            || options.stop.iter().any(|s| s.is_empty() || s.len() > 65536)
            || options.stop.iter().map(String::len).sum::<usize>() > 1024 * 1024
        {
            return Err(Error::Invalid(
                "stop strings must be nonempty and bounded to 256 strings / 1 MiB".into(),
            ));
        }
        let tokens = self.tokenize(prompt, true)?;
        if tokens.is_empty() {
            return Err(Error::Invalid("empty prompt with no BOS token".into()));
        }
        if tokens.len() > self.context_length() {
            return Err(Error::Invalid(format!(
                "prompt has {} tokens; context limit is {}",
                tokens.len(),
                self.context_length()
            )));
        }
        let mut stats = GenerationStats {
            prompt_tokens: tokens.len(),
            completion_tokens: 0,
            finish_reason: FinishReason::Length,
        };
        let limit = options.max_tokens.min(self.context_length() - tokens.len());
        if limit == 0 {
            return Ok(stats);
        }
        let configs = if options.temperature == 0. {
            vec![Config::Greedy]
        } else {
            vec![
                Config::Temperature {
                    temperature: options.temperature,
                },
                Config::Dist { seed: options.seed },
            ]
        };
        let mut sampler = Chain::new(configs)?;
        for &token in &tokens {
            sampler.accept(token)?
        }
        let mut context = self.context()?;
        let mut logits = Vec::new();
        for (chunk_index, chunk) in tokens.chunks(512).enumerate() {
            let start = chunk_index * 512;
            let batch: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(i, &id)| BatchToken {
                    sequence: 0,
                    position: start + i,
                    input: Input::Token(id as u32),
                    output: if start + i + 1 == tokens.len() {
                        OutputSelection::Logits
                    } else {
                        OutputSelection::None
                    },
                })
                .collect();
            let outputs = context.decode_batch_with_cancel(&batch, cancel)?;
            for output in outputs {
                if let Some(v) = output.logits {
                    logits = v
                }
            }
        }
        if logits.is_empty() {
            return Err(Error::Internal("prefill produced no logits".into()));
        }
        let mut filter = stream::StopFilter::new(&options.stop);
        let mut utf8 = stream::Utf8Stream::default();
        for step in 0..limit {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let token = sampler.sample_and_accept(&logits)?;
            stats.completion_tokens += 1;
            if self.tokenizer.is_end_of_generation(token) {
                stats.finish_reason = FinishReason::Eos;
                break;
            }
            let piece = self.tokenizer.token_piece(token, false)?;
            let (safe, stopped) = filter.push(&piece);
            utf8.push(&safe, emit)?;
            if stopped {
                stats.finish_reason = FinishReason::StopSequence;
                break;
            }
            if step + 1 < limit {
                let batch = [BatchToken {
                    sequence: 0,
                    position: tokens.len() + step,
                    input: Input::Token(token as u32),
                    output: OutputSelection::Logits,
                }];
                logits = context
                    .decode_batch_with_cancel(&batch, cancel)?
                    .into_iter()
                    .next()
                    .and_then(|v| v.logits)
                    .ok_or_else(|| Error::Internal("decode produced no logits".into()))?;
            }
        }
        if cancel() {
            return Err(Error::Cancelled);
        }
        if stats.finish_reason != FinishReason::StopSequence {
            utf8.push(&filter.finish(), emit)?
        }
        utf8.finish(emit)?;
        Ok(stats)
    }
}
