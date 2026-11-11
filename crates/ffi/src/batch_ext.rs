//! Owned extended batches for the admitted native Llama contexts.
//! The setters follow frozen llama-batch.cpp ordering, including the retained
//! empty entry when add_token/add_embd fails after the entry was appended.
use crate::*;
use llama_model_runtime::{Input, NativeBatchToken, OutputSelection};
use std::collections::BTreeSet;

struct Entry {
    token: Option<u32>,
    embedding_offset: Option<usize>,
    sequences: BTreeSet<i32>,
    position: i32,
    output: bool,
}

/// Opaque C-owned batch. It retains copied input values, never foreign pointers.
pub struct BatchExtHandle {
    maximum_tokens: usize,
    maximum_sequences: usize,
    vocabulary: usize,
    width: usize,
    embedding_width: usize,
    entries: Vec<Entry>,
    embeddings: Vec<f32>,
}

impl BatchExtHandle {
    fn add(&mut self, sequence: i32) -> Result<i32> {
        // The capacity check precedes sequence validation in the original.
        if self.entries.len() >= self.maximum_tokens {
            return Ok(-1);
        }
        if sequence < 0 || sequence as usize >= self.maximum_sequences {
            return Ok(-3);
        }
        self.entries
            .try_reserve(1)
            .map_err(|_| "extended batch entry allocation failed")?;
        let index = self.entries.len() as i32;
        self.entries.push(Entry {
            token: None,
            embedding_offset: None,
            sequences: BTreeSet::from([sequence]),
            // The public comment says unset, but frozen token::pos is zero-initialized.
            position: 0,
            output: false,
        });
        Ok(index)
    }

    fn entry(&mut self, index: i32) -> Option<&mut Entry> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.entries.get_mut(i))
    }

    unsafe fn set_embedding(&mut self, index: i32, input: LlamaEmbd) -> Result<bool> {
        let Ok(index) = usize::try_from(index) else {
            return Ok(false);
        };
        if index >= self.entries.len() || input.data.is_null() {
            return Ok(false);
        }
        let Some(total) = input.n_rows.checked_mul(input.n_embd) else {
            return Err("extended embedding dimensions overflow".into());
        };
        if self.embedding_width == 0 {
            if total != self.width {
                return Ok(false);
            }
            self.embedding_width = total;
        } else if total != self.embedding_width {
            return Ok(false);
        }
        if self.entries[index].embedding_offset.is_some() {
            return Ok(false);
        }
        let end = self
            .embeddings
            .len()
            .checked_add(total)
            .ok_or("extended batch embedding storage overflow")?;
        if end > MAX_FOREIGN_ELEMENTS {
            return Err("extended batch embedding storage exceeds checked element limit".into());
        }
        // SAFETY: the caller supplies the declared input; count and alignment are checked.
        let values = unsafe { input_slice(input.data, total)? };
        self.embeddings
            .try_reserve(total)
            .map_err(|_| "extended batch embedding allocation failed")?;
        self.entries[index].embedding_offset = Some(self.embeddings.len());
        self.embeddings.extend_from_slice(values);
        Ok(true)
    }

    fn prepare(&self, context: &ContextHandle) -> Result<Vec<NativeBatchToken>> {
        let (maximum_tokens, maximum_sequences, vocabulary, width, output_all) =
            context.extended_batch_configuration();
        if self.entries.is_empty() || self.entries.len() > maximum_tokens {
            return Err("empty or oversized extended decode batch".into());
        }
        // Original cross-context misuse may assert or interpret mismatched rows.
        // A batch may be reused with another context with compatible dimensions.
        if self.maximum_sequences > maximum_sequences
            || self.vocabulary != vocabulary
            || self.width != width
        {
            return Err("extended batch is incompatible with the target context".into());
        }
        let mut tokens_only = 0;
        let mut embeddings_only = 0;
        let mut both = 0;
        for entry in &self.entries {
            match (entry.token.is_some(), entry.embedding_offset.is_some()) {
                (false, false) => {
                    return Err("extended entry has neither a token nor an embedding".into());
                }
                (true, false) => tokens_only += 1,
                (false, true) => embeddings_only += 1,
                (true, true) => both += 1,
            }
        }
        if both != 0 && both != self.entries.len() {
            return Err(
                "entries carrying both token and embedding cannot mix with other entries".into(),
            );
        }
        let mixed = tokens_only != 0 && embeddings_only != 0;
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.entries.len())
            .map_err(|_| "extended decode planning allocation failed")?;
        for (index, entry) in self.entries.iter().enumerate() {
            // For all-both input the original Llama graph selects its token path.
            let input = if let Some(token) = entry.token {
                Input::Token(token)
            } else {
                // The original mixed path gathers by each token's offset. Its
                // homogeneous embedding path uses append order instead. Preserve
                // that observable distinction for out-of-order setter calls.
                let start = if mixed {
                    entry.embedding_offset.ok_or("missing extended embedding")?
                } else {
                    index
                        .checked_mul(width)
                        .ok_or("extended embedding row overflow")?
                };
                let end = start
                    .checked_add(width)
                    .ok_or("extended embedding end overflow")?;
                let row = self
                    .embeddings
                    .get(start..end)
                    .ok_or("extended embedding row unavailable")?;
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(width)
                    .map_err(|_| "extended embedding row allocation failed")?;
                owned.extend_from_slice(row);
                Input::Embedding(owned)
            };
            let output = if output_all {
                OutputSelection::Both
            } else if entry.output {
                OutputSelection::Logits
            } else {
                OutputSelection::None
            };
            result.push(NativeBatchToken {
                sequences: entry.sequences.iter().copied().collect(),
                position: entry.position,
                input,
                output,
            });
        }
        Ok(result)
    }
}

ffi!(
    llama_batch_ext_init(context: *mut ContextHandle) -> *mut BatchExtHandle,
    ptr::null_mut(),
    {
        let context = unsafe { reference(context)? };
        let (maximum_tokens, maximum_sequences, vocabulary, width, _) =
            context.extended_batch_configuration();
        Ok(Box::into_raw(Box::new(BatchExtHandle {
            maximum_tokens,
            maximum_sequences,
            vocabulary,
            width,
            embedding_width: 0,
            entries: Vec::new(),
            embeddings: Vec::new(),
        })))
    }
);
ffi!(
    llama_batch_ext_free(batch: *mut BatchExtHandle) -> (),
    (),
    {
        if !batch.is_null() {
            unsafe {
                reference_mut(batch)?;
                drop(Box::from_raw(batch));
            }
        }
        Ok(())
    }
);
ffi!(
    llama_batch_ext_clear(batch: *mut BatchExtHandle) -> (),
    (),
    {
        let batch = unsafe { reference_mut(batch)? };
        batch.entries.clear();
        batch.embeddings.clear();
        batch.embedding_width = 0;
        Ok(())
    }
);
ffi!(
    llama_batch_ext_add(batch: *mut BatchExtHandle, sequence: i32) -> i32,
    -1,
    { unsafe { reference_mut(batch)? }.add(sequence) }
);
ffi!(
    llama_batch_ext_add_token(batch: *mut BatchExtHandle, sequence: i32, token: i32) -> i32,
    -1,
    {
        let batch = unsafe { reference_mut(batch)? };
        let index = batch.add(sequence)?;
        if index < 0 {
            return Ok(index);
        }
        if token < 0 || token as usize >= batch.vocabulary {
            return Ok(-2);
        }
        batch.entries[index as usize].token = Some(token as u32);
        Ok(index)
    }
);
ffi!(
    llama_batch_ext_add_embd(
        batch: *mut BatchExtHandle,
        sequence: i32,
        embedding: LlamaEmbd,
    ) -> i32,
    -1,
    {
        let batch = unsafe { reference_mut(batch)? };
        let index = batch.add(sequence)?;
        if index < 0 {
            return Ok(index);
        }
        Ok(if unsafe { batch.set_embedding(index, embedding)? } {
            index
        } else {
            -2
        })
    }
);
ffi!(
    llama_batch_ext_add_seq(batch: *mut BatchExtHandle, index: i32, sequence: i32) -> bool,
    false,
    {
        let batch = unsafe { reference_mut(batch)? };
        if sequence < 0 || sequence as usize >= batch.maximum_sequences {
            return Ok(false);
        }
        let Some(entry) = batch.entry(index) else {
            return Ok(false);
        };
        entry.sequences.insert(sequence);
        Ok(true)
    }
);
ffi!(
    llama_batch_ext_set_pos(batch: *mut BatchExtHandle, index: i32, position: *const i32) -> bool,
    false,
    {
        let batch = unsafe { reference_mut(batch)? };
        let Some(entry) = batch.entry(index) else {
            return Ok(false);
        };
        if position.is_null() {
            return Ok(false);
        }
        // Admitted dense Llama contexts have one position per token or embedding.
        entry.position = *unsafe { reference(position)? };
        Ok(true)
    }
);
ffi!(
    llama_batch_ext_set_embd_token(
        batch: *mut BatchExtHandle,
        index: i32,
        embedding: LlamaEmbd,
    ) -> bool,
    false,
    { unsafe { reference_mut(batch)?.set_embedding(index, embedding) } }
);
ffi!(
    llama_batch_ext_set_embd_state(
        batch: *mut BatchExtHandle,
        index: i32,
        embedding: LlamaEmbd,
    ) -> bool,
    false,
    {
        let batch = unsafe { reference_mut(batch)? };
        if batch.entry(index).is_none() || embedding.data.is_null() {
            return Ok(false);
        }
        // This matches n_embd_state == 0 for every context currently admitted by
        // llama_init_from_model. MTP/LiliCorr context construction remains unsupported.
        Ok(false)
    }
);
ffi!(
    llama_batch_ext_set_output_logits(batch: *mut BatchExtHandle, index: i32, value: u8) -> bool,
    false,
    {
        let batch = unsafe { reference_mut(batch)? };
        let Some(entry) = batch.entry(index) else {
            return Ok(false);
        };
        entry.output = boolean(value)?;
        Ok(true)
    }
);
ffi!(
    llama_batch_ext_set_output_embd(batch: *mut BatchExtHandle, index: i32, value: u8) -> bool,
    false,
    {
        let batch = unsafe { reference_mut(batch)? };
        let Some(entry) = batch.entry(index) else {
            return Ok(false);
        };
        // The frozen public API deliberately shares this flag with logits.
        entry.output = boolean(value)?;
        Ok(true)
    }
);
ffi!(
    llama_process(
        context: *mut ContextHandle,
        process_type: i32,
        batch: *mut BatchExtHandle,
    ) -> i32,
    -1,
    {
        if process_type != 1 {
            return Err(
                "extended processing currently supports LLAMA_PROCESS_TYPE_DECODE only".into(),
            );
        }
        let batch = unsafe { reference(batch)? };
        let context = unsafe { reference_mut(context)? };
        let prepared = batch.prepare(context)?;
        crate::context::execute_native_batch(context, &prepared)
    }
);

#[cfg(test)]
mod tests;
