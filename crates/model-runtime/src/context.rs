//! Transactional multi-sequence execution with explicit microbatch boundaries.
use crate::{
    AdapterSet, Error, Model, Result, check_positive, llama::KvLayer, mamba::RecurrentLayer, math,
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Token(u32),
    Embedding(Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSelection {
    None,
    Logits,
    Embedding,
    Both,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchToken {
    pub sequence: i32,
    pub position: usize,
    pub input: Input,
    pub output: OutputSelection,
}
impl BatchToken {
    pub fn token(sequence: i32, position: usize, token: u32) -> Self {
        Self {
            sequence,
            position,
            input: Input::Token(token),
            output: OutputSelection::Both,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenOutput {
    pub batch_index: usize,
    pub sequence: i32,
    pub position: usize,
    pub logits: Option<Vec<f32>>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct ContextOptions {
    pub maximum_sequences: usize,
    pub microbatch_tokens: usize,
    pub maximum_batch_tokens: usize,
}
impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            maximum_sequences: 16,
            microbatch_tokens: 32,
            maximum_batch_tokens: 512,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SequenceMemory {
    Attention(Vec<KvLayer>),
    Recurrent(Vec<RecurrentLayer>),
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SequenceState {
    pub position: usize,
    pub memory: SequenceMemory,
}

/// A full owned checkpoint can restore attention and recurrent state. The
/// first implementation clones state; copy-on-write storage is a later T02
/// integration optimization, not a correctness assumption.
#[derive(Debug, Clone)]
pub struct ContextCheckpoint {
    model_fingerprint: u64,
    adapter_fingerprint: u64,
    sequences: BTreeMap<i32, SequenceState>,
}

#[derive(Debug)]
pub struct Context {
    pub(crate) model: Arc<Model>,
    pub(crate) options: ContextOptions,
    pub(crate) adapters: AdapterSet,
    pub(crate) sequences: BTreeMap<i32, SequenceState>,
    generation: u64,
}
impl Context {
    pub fn new(model: Arc<Model>, options: ContextOptions) -> Result<Self> {
        check_positive(&[
            ("maximum_sequences", options.maximum_sequences),
            ("microbatch_tokens", options.microbatch_tokens),
            ("maximum_batch_tokens", options.maximum_batch_tokens),
        ])?;
        Ok(Self {
            model,
            options,
            adapters: AdapterSet::default(),
            sequences: BTreeMap::new(),
            generation: 0,
        })
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }
    pub fn next_position(&self, sequence: i32) -> Option<usize> {
        self.sequences.get(&sequence).map(|s| s.position)
    }
    /// Changes when adapter configuration changes; use in future graph keys.
    pub fn configuration_generation(&self) -> u64 {
        self.generation
    }
    pub fn reset(&mut self) {
        self.sequences.clear();
    }
    pub fn remove_sequence(&mut self, sequence: i32) -> bool {
        self.sequences.remove(&sequence).is_some()
    }
    pub fn fork_sequence(&mut self, source: i32, destination: i32) -> Result<()> {
        if destination < 0 {
            return Err(Error::InvalidSequence(destination));
        }
        if self.sequences.contains_key(&destination) {
            return Err(Error::SequenceExists(destination));
        }
        if self.sequences.len() >= self.options.maximum_sequences {
            return Err(Error::SequenceLimit {
                maximum: self.options.maximum_sequences,
            });
        }
        let state = self
            .sequences
            .get(&source)
            .ok_or(Error::MissingSequence(source))?
            .clone();
        self.sequences.insert(destination, state);
        Ok(())
    }
    /// Suffix truncation is exact for dense KV state. Recurrent state requires
    /// restoring a checkpoint because an SSM update is not reversible.
    pub fn truncate_sequence(&mut self, sequence: i32, position: usize) -> Result<()> {
        let state = self
            .sequences
            .get_mut(&sequence)
            .ok_or(Error::MissingSequence(sequence))?;
        if position > state.position {
            return Err(Error::InvalidPosition {
                expected: state.position,
                actual: position,
            });
        }
        match (&self.model.as_ref(), &mut state.memory) {
            (Model::Llama(model), SequenceMemory::Attention(layers)) => {
                let length = position * model.config().kv_width()?;
                for layer in layers {
                    layer.keys.truncate(length);
                    layer.values.truncate(length);
                }
                state.position = position;
                Ok(())
            }
            (Model::Mamba(_), SequenceMemory::Recurrent(_)) => {
                if position == state.position {
                    Ok(())
                } else {
                    Err(Error::Unsupported(
                        "arbitrary Mamba rewind; restore a ContextCheckpoint instead".into(),
                    ))
                }
            }
            _ => Err(Error::InvalidState("model/memory mismatch".into())),
        }
    }
    pub fn checkpoint(&self) -> ContextCheckpoint {
        ContextCheckpoint {
            model_fingerprint: self.model.fingerprint(),
            adapter_fingerprint: self.adapters.fingerprint(),
            sequences: self.sequences.clone(),
        }
    }
    pub fn restore_checkpoint(&mut self, checkpoint: &ContextCheckpoint) -> Result<()> {
        if checkpoint.model_fingerprint != self.model.fingerprint()
            || checkpoint.adapter_fingerprint != self.adapters.fingerprint()
        {
            return Err(Error::InvalidState(
                "checkpoint belongs to different model weights or adapters".into(),
            ));
        }
        if checkpoint.sequences.len() > self.options.maximum_sequences {
            return Err(Error::SequenceLimit {
                maximum: self.options.maximum_sequences,
            });
        }
        self.sequences = checkpoint.sequences.clone();
        Ok(())
    }
    /// Explicitly clears existing state when replacing adapters. Changing an
    /// adapter while retaining its old keys/values/SSM state would be incorrect.
    pub fn replace_adapters_and_reset(&mut self, adapters: AdapterSet) -> Result<()> {
        adapters.validate(&self.model)?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::InvalidConfig("adapter generation exhausted".into()))?;
        self.adapters = adapters;
        self.sequences.clear();
        self.generation = generation;
        Ok(())
    }
    pub fn decode_batch(&mut self, batch: &[BatchToken]) -> Result<Vec<TokenOutput>> {
        self.decode_batch_with_cancel(batch, || false)
    }
    /// The entire batch commits atomically after all its microbatches succeed.
    /// Cancellation is checked before execution, each microbatch, token, layer
    /// and commit. Failure returns no partial output and leaves state unchanged.
    pub fn decode_batch_with_cancel(
        &mut self,
        batch: &[BatchToken],
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<TokenOutput>> {
        if batch.len() > self.options.maximum_batch_tokens {
            return Err(Error::InvalidConfig(format!(
                "batch of {} exceeds maximum {}",
                batch.len(),
                self.options.maximum_batch_tokens
            )));
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let mut staged = self.sequences.clone();
        let mut outputs = Vec::new();
        for (chunk_index, chunk) in batch.chunks(self.options.microbatch_tokens).enumerate() {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            for (within, item) in chunk.iter().enumerate() {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                if item.sequence < 0 {
                    return Err(Error::InvalidSequence(item.sequence));
                }
                if !staged.contains_key(&item.sequence) {
                    if staged.len() >= self.options.maximum_sequences {
                        return Err(Error::SequenceLimit {
                            maximum: self.options.maximum_sequences,
                        });
                    }
                    let memory = match self.model.as_ref() {
                        Model::Llama(model) => SequenceMemory::Attention(model.empty_cache()),
                        Model::Mamba(model) => SequenceMemory::Recurrent(model.empty_state()),
                    };
                    staged.insert(
                        item.sequence,
                        SequenceState {
                            position: 0,
                            memory,
                        },
                    );
                }
                let state = staged
                    .get_mut(&item.sequence)
                    .expect("sequence inserted above");
                if item.position != state.position {
                    return Err(Error::InvalidPosition {
                        expected: state.position,
                        actual: item.position,
                    });
                }
                if state.position >= self.model.max_context() {
                    return Err(Error::ContextFull {
                        maximum: self.model.max_context(),
                    });
                }
                let input = match &item.input {
                    Input::Token(token) => self.model.embedding(*token)?,
                    Input::Embedding(values) => {
                        math::vector(values, self.model.width(), "external embedding")?;
                        values.clone()
                    }
                };
                let (embedding, logits) = match (self.model.as_ref(), &mut state.memory) {
                    (Model::Llama(model), SequenceMemory::Attention(cache)) => {
                        model.step(input, state.position, cache, &self.adapters, &mut cancelled)?
                    }
                    (Model::Mamba(model), SequenceMemory::Recurrent(cache)) => {
                        model.step(input, cache, &self.adapters, &mut cancelled)?
                    }
                    _ => return Err(Error::InvalidState("model/memory mismatch".into())),
                };
                state.position += 1;
                if item.output != OutputSelection::None {
                    outputs.push(TokenOutput {
                        batch_index: chunk_index * self.options.microbatch_tokens + within,
                        sequence: item.sequence,
                        position: item.position,
                        logits: matches!(
                            item.output,
                            OutputSelection::Logits | OutputSelection::Both
                        )
                        .then_some(logits),
                        embedding: matches!(
                            item.output,
                            OutputSelection::Embedding | OutputSelection::Both
                        )
                        .then_some(embedding),
                    });
                }
            }
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        self.sequences = staged;
        Ok(outputs)
    }
    /// Versioned, little-endian progress format; not llama's native state ABI.
    pub fn save_state(&self) -> Result<Vec<u8>> {
        crate::state::encode(self)
    }
    /// Fully validates before replacing context state. Checks CRC32 corruption,
    /// exact dimensions, finite values, bounded counts, IDs and model identity.
    pub fn restore_state(&mut self, bytes: &[u8]) -> Result<()> {
        crate::state::decode(self, bytes)
    }
}

mod native;
pub use native::{NativeBatchToken, NativeContext, NativeContextOptions, NativeState};
