//! Interchangeable sequential decoder sessions. Model/tokenizer identity remains
//! the caller's responsibility; dimensions and an empty initial prefix are
//! checked by Model::generate_with_session.
use crate::{Error, Result};
use llama_model_runtime::{BatchToken, Context, Input, OutputSelection};

/// An owned autoregressive session. Evaluation appends the supplied token IDs
/// and returns the last token's complete vocabulary logits. Implementations
/// report their actual committed prefix after cancellation or failure.
pub trait DecoderSession {
    fn context_length(&self) -> usize;
    fn vocabulary_size(&self) -> usize;
    fn next_position(&self) -> usize;
    fn evaluate(&mut self, tokens: &[i32], cancel: &dyn Fn() -> bool) -> Result<Vec<f32>>;
}
// Defer decoder context creation until the first evaluation so that shared
// cancellation, architecture, option and zero-limit checks keep their order.
pub(crate) struct CpuSession<'model> {
    model: &'model crate::Model,
    context: Option<Context>,
}
impl<'model> CpuSession<'model> {
    pub(crate) fn new(model: &'model crate::Model) -> Self {
        Self {
            model,
            context: None,
        }
    }
}
impl DecoderSession for CpuSession<'_> {
    fn context_length(&self) -> usize {
        self.model.context_length()
    }
    fn vocabulary_size(&self) -> usize {
        self.model.vocabulary_size()
    }
    fn next_position(&self) -> usize {
        self.context
            .as_ref()
            .and_then(|context| context.next_position(0))
            .unwrap_or(0)
    }
    fn evaluate(&mut self, tokens: &[i32], cancel: &dyn Fn() -> bool) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::Invalid("decoder evaluation requires tokens".into()));
        }
        let start = self.next_position();
        if self.context.is_none() {
            self.context = Some(self.model.context()?);
        }
        let context = self
            .context
            .as_mut()
            .ok_or_else(|| Error::Internal("CPU decoder context was not initialized".into()))?;
        let mut logits = Vec::new();
        for (chunk_index, chunk) in tokens.chunks(512).enumerate() {
            let offset = chunk_index * 512;
            let batch: Vec<BatchToken> = chunk
                .iter()
                .enumerate()
                .map(|(i, &id)| {
                    let token = u32::try_from(id)
                        .map_err(|_| Error::Invalid("negative token ID".into()))?;
                    Ok(BatchToken {
                        sequence: 0,
                        position: start + offset + i,
                        input: Input::Token(token),
                        output: if offset + i + 1 == tokens.len() {
                            OutputSelection::Logits
                        } else {
                            OutputSelection::None
                        },
                    })
                })
                .collect::<Result<_>>()?;
            for output in context.decode_batch_with_cancel(&batch, cancel)? {
                if let Some(value) = output.logits {
                    logits = value;
                }
            }
        }
        if logits.is_empty() {
            return Err(Error::Internal("decoder produced no logits".into()));
        }
        Ok(logits)
    }
}
