//! Owned C encoder outputs and the frozen legacy BERT batch conversion.
use super::{ContextHandle, ContextRuntime};
use crate::native_metadata_policy::{Profile, sequence_set_order};
use crate::*;
use llama_model_runtime::bert::native::Options;
use llama_model_runtime::{
    BertModel, Error, Input, NativeBatchToken, NativeExecutionOptions, OutputSelection, Pooling,
};
use std::collections::BTreeMap;
use std::sync::Arc;

const MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
fn native_profile() -> Result<Profile> {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        Ok(Profile::Msvc)
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        Ok(Profile::Gnu)
    } else {
        Err("native BERT sequence ordering requires a measured x64 MSVC or GNU profile".into())
    }
}

pub(super) fn pooling_id(pooling: Pooling) -> i32 {
    match pooling {
        Pooling::None => 0,
        Pooling::Mean => 1,
        Pooling::First => 2,
        Pooling::Last => 3,
        Pooling::Rank => 4,
    }
}
fn pooling(value: i32) -> Result<Pooling> {
    match value {
        0 => Ok(Pooling::None),
        1 => Ok(Pooling::Mean),
        2 => Ok(Pooling::First),
        3 => Ok(Pooling::Last),
        4 => Ok(Pooling::Rank),
        _ => Err("invalid BERT pooling type".into()),
    }
}
pub(crate) struct BertEncoderContext {
    model: Arc<BertModel>,
    pub(super) pooling_id: i32,
    pub(super) causal: bool,
    pub(super) execution: NativeExecutionOptions,
    maximum_sequences: usize,
    storage: Vec<f32>,
    logits_len: usize,
    embeddings_offset: Option<usize>,
    embeddings_len: usize,
    sequences: BTreeMap<i32, Vec<f32>>,
}
impl BertEncoderContext {
    pub(super) fn new(
        model: Arc<BertModel>,
        pooling_id: i32,
        causal: bool,
        maximum_batch: usize,
        maximum_sequences: usize,
        execution: NativeExecutionOptions,
    ) -> Result<Self> {
        pooling(pooling_id)?;
        native_profile()?;
        Options {
            maximum_sequences,
            execution,
        }
        .validate()
        .map_err(|e| e.to_string())?;
        if maximum_sequences > maximum_batch {
            return Err("BERT output row ceiling is smaller than its sequence count".into());
        }
        let mut context = Self {
            model,
            pooling_id,
            causal,
            execution,
            maximum_sequences,
            storage: Vec::new(),
            logits_len: 0,
            embeddings_offset: None,
            embeddings_len: 0,
            sequences: BTreeMap::new(),
        };
        // Before graph reservation the original context assumes a logits graph.
        // Its shared zeroed buffer is observable through the raw pointer getters.
        context.reserve(0, true)?;
        Ok(context)
    }
    pub(super) fn set_threads(&mut self, threads: usize, threads_batch: usize) -> Result<()> {
        let replacement = NativeExecutionOptions {
            threads,
            threads_batch,
            ..self.execution
        };
        Options {
            maximum_sequences: self.maximum_sequences,
            execution: replacement,
        }
        .validate()
        .map_err(|e| e.to_string())?;
        self.execution = replacement;
        Ok(())
    }
    fn reserve(&mut self, outputs: usize, initial: bool) -> Result<()> {
        let rows = outputs.max(self.maximum_sequences);
        let logits_len = if initial || !self.execution.embeddings {
            rows.checked_mul(self.model.config().vocabulary)
                .ok_or("BERT logits allocation overflow")?
        } else {
            0
        };
        let embeddings_len = if self.execution.embeddings {
            rows.checked_mul(self.model.config().width)
                .ok_or("BERT embeddings allocation overflow")?
        } else {
            0
        };
        let count = logits_len
            .checked_add(embeddings_len)
            .ok_or("BERT output allocation overflow")?;
        if count
            .checked_mul(4)
            .is_none_or(|bytes| bytes > MAX_OUTPUT_BYTES)
        {
            return Err("BERT output buffer exceeds 512 MiB".into());
        }
        if count > self.storage.len() {
            let mut storage = Vec::new();
            storage
                .try_reserve_exact(count)
                .map_err(|_| "BERT output allocation failed")?;
            storage.resize(count, 0.0);
            self.storage = storage;
        }
        self.logits_len = logits_len;
        self.embeddings_len = embeddings_len;
        self.embeddings_offset = self.execution.embeddings.then_some(logits_len);
        Ok(())
    }
    pub(super) fn logits_pointer(&mut self) -> *mut f32 {
        if self.logits_len == 0 {
            ptr::null_mut()
        } else {
            self.storage.as_mut_ptr()
        }
    }
    pub(super) fn embeddings_pointer(&mut self) -> *mut f32 {
        self.embeddings_offset
            .map_or(ptr::null_mut(), |offset| unsafe {
                self.storage.as_mut_ptr().add(offset)
            })
    }
    pub(super) fn logits_row_pointer(&mut self, row: usize) -> Result<*mut f32> {
        let offset = row
            .checked_mul(self.model.config().vocabulary)
            .ok_or("BERT logit offset overflow")?;
        if offset >= self.logits_len {
            return Err("BERT logits buffer is unavailable".into());
        }
        Ok(unsafe { self.storage.as_mut_ptr().add(offset) })
    }
    pub(super) fn embeddings_row_pointer(&mut self, row: usize) -> Result<*mut f32> {
        let base = self
            .embeddings_offset
            .ok_or("BERT embeddings were not enabled")?;
        let offset = row
            .checked_mul(self.model.config().width)
            .ok_or("BERT embedding offset overflow")?;
        if offset >= self.embeddings_len {
            return Err("BERT embedding row is unavailable".into());
        }
        Ok(unsafe { self.storage.as_mut_ptr().add(base + offset) })
    }
}

unsafe fn legacy_batch(
    context: &ContextHandle,
    batch: LlamaBatch,
) -> Result<Vec<NativeBatchToken>> {
    let n = count(batch.n_tokens)?;
    if n == 0 || n > context.n_batch as usize {
        return Err("empty or oversized BERT encode batch".into());
    }
    // Original encoding asserts this instead of splitting bidirectional attention.
    if n > context.n_ubatch as usize {
        return Err("BERT encoding requires n_ubatch >= n_tokens".into());
    }
    let ContextRuntime::Encoder(encoder) = &context.runtime else {
        return Err("llama_encode currently requires a native BERT encoder model".into());
    };
    // Reject excessive shapes before forming foreign slices or copying any direct
    // embeddings. Full per-row validation still occurs before changing outputs.
    encoder
        .model
        .validate_native_shape(
            n,
            pooling(encoder.pooling_id)?,
            Options {
                maximum_sequences: encoder.maximum_sequences,
                execution: encoder.execution,
            },
        )
        .map_err(|e| e.to_string())?;
    let tokens = if batch.token.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.token, n)? })
    };
    let width = context.model_handle.data.width as usize;
    let embeddings = if tokens.is_some() || batch.embd.is_null() {
        None
    } else {
        Some(unsafe {
            input_slice(
                batch.embd,
                n.checked_mul(width)
                    .ok_or("BERT embedding input overflow")?,
            )?
        })
    };
    if tokens.is_none() && embeddings.is_none() {
        return Err("BERT batch has neither tokens nor host embeddings".into());
    }
    let positions = if batch.pos.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.pos, n)? })
    };
    let counts = if batch.n_seq_id.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.n_seq_id, n)? })
    };
    let sequences = if batch.seq_id.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.seq_id, n)? })
    };
    let flags = if batch.logits.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.logits, n)? })
    };
    let mut next = BTreeMap::<i32, i32>::new();
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|_| "BERT input row allocation failed")?;
    let mut memberships = 0usize;
    for index in 0..n {
        let count = counts.map_or(Ok(1), |counts| count(counts[index]))?;
        if count == 0 || count > 256 || (sequences.is_none() && count != 1) {
            return Err("invalid BERT per-token sequence count".into());
        }
        memberships = memberships
            .checked_add(count)
            .ok_or("BERT sequence count overflow")?;
        if memberships > 1_048_576 {
            return Err("BERT input exceeds one million memberships".into());
        }
        let ids = if let Some(ids) = sequences {
            unsafe { input_slice(ids[index], count)? }
        } else {
            &[0]
        };
        if ids
            .iter()
            .any(|&id| id < 0 || id as u32 >= context.n_seq_max)
        {
            return Err("BERT batch sequence ID exceeds configured sequence count".into());
        }
        // Legacy conversion increments only the first supplied ID, before the native unordered set is copied.
        let position = if let Some(positions) = positions {
            positions[index]
        } else {
            let next_position = next.entry(ids[0]).or_default();
            let position = *next_position;
            *next_position = position
                .checked_add(1)
                .ok_or("BERT automatic position overflow")?;
            position
        };
        let input = if let Some(tokens) = tokens {
            Input::Token(u32::try_from(tokens[index]).map_err(|_| "negative BERT token ID")?)
        } else {
            let values = embeddings.ok_or("missing BERT input embeddings")?;
            let mut row = Vec::new();
            row.try_reserve_exact(width)
                .map_err(|_| "BERT embedding row allocation failed")?;
            row.extend_from_slice(&values[index * width..(index + 1) * width]);
            Input::Embedding(row)
        };
        let requested =
            context.embeddings_enabled || flags.map_or(index + 1 == n, |flags| flags[index] != 0);
        result.push(NativeBatchToken {
            sequences: sequence_set_order(ids, native_profile()?)?,
            position,
            input,
            output: if !requested {
                OutputSelection::None
            } else if context.embeddings_enabled {
                OutputSelection::Both
            } else {
                OutputSelection::Logits
            },
        });
    }
    Ok(result)
}

pub(super) unsafe fn encode(context: &mut ContextHandle, batch: LlamaBatch) -> Result<i32> {
    let batch = unsafe { legacy_batch(context, batch)? };
    let ContextRuntime::Encoder(encoder) = &mut context.runtime else {
        return Err("llama_encode currently requires a native BERT encoder model".into());
    };
    let pooling = pooling(encoder.pooling_id)?;
    let options = Options {
        maximum_sequences: encoder.maximum_sequences,
        execution: encoder.execution,
    };
    encoder
        .model
        .validate_native_batch(&batch, pooling, options)
        .map_err(|e| e.to_string())?;
    // Validation failures retain prior outputs. Accepted work clears pooled
    // storage and updates row mapping even if the graph is subsequently aborted.
    encoder.sequences.clear();
    let outputs = batch
        .iter()
        .filter(|row| row.output != OutputSelection::None)
        .count();
    if let Err(error) = encoder.reserve(outputs, false) {
        set_error(&error);
        return Ok(-2);
    }
    context.output_ids.fill(None);
    let mut ordinal = 0;
    for (index, row) in batch.iter().enumerate() {
        if row.output != OutputSelection::None {
            context.output_ids[index] = Some(ordinal);
            ordinal += 1;
        }
    }
    context.output_count = outputs;
    let callback = context.abort;
    let data = context.abort_data;
    let result = encoder
        .model
        .encode_native_with_cancel(&batch, pooling, options, || {
            callback.is_some_and(|callback| unsafe { callback(data) })
        });
    let result = match result {
        Ok(output) => output,
        Err(error) => {
            set_error(&error.to_string());
            return Ok(if error == Error::Cancelled { 2 } else { -3 });
        }
    };
    if context.embeddings_enabled {
        if pooling == Pooling::None {
            let base = encoder
                .embeddings_offset
                .ok_or("BERT output buffer missing")?;
            let width = encoder.model.config().width;
            for (index, values) in result.tokens.iter().enumerate() {
                if let Some(row) = context.output_ids[index] {
                    encoder.storage[base + row * width..base + (row + 1) * width]
                        .copy_from_slice(values);
                }
            }
        } else if pooling == Pooling::Rank {
            // Frozen extraction strides by n_cls_out even when an optional head
            // is absent and the graph's physical width is the hidden dimension.
            let count = encoder.model.classifier().output_count;
            let ids: Vec<_> = result.sequences.keys().copied().collect();
            let flattened: Vec<_> = result.sequences.into_values().flatten().collect();
            for (index, id) in ids.into_iter().enumerate() {
                let row = flattened
                    .get(index * count..(index + 1) * count)
                    .ok_or("BERT rank output extraction exceeds graph output")?;
                encoder.sequences.insert(id, row.to_vec());
            }
        } else {
            encoder.sequences = result.sequences;
        }
    }
    Ok(0)
}

ffi!(
    llama_encode(context: *mut ContextHandle, batch: LlamaBatch) -> i32,
    -1,
    {
        let context = unsafe { reference_mut(context)? };
        if !matches!(context.runtime, ContextRuntime::Encoder(_)) {
            return Err("llama_encode currently requires a native BERT encoder model".into());
        }
        unsafe { encode(context, batch) }
    }
);
ffi!(
    llama_get_embeddings_seq(context: *mut ContextHandle, sequence: i32) -> *mut f32,
    ptr::null_mut(),
    {
        let context = unsafe { reference_mut(context)? };
        Ok(match &mut context.runtime {
            ContextRuntime::Encoder(encoder) => encoder
                .sequences
                .get_mut(&sequence)
                .map_or(ptr::null_mut(), |values| values.as_mut_ptr()),
            ContextRuntime::Decoder(_) => ptr::null_mut(),
        })
    }
);

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu"),
    )
))]
mod tests;
