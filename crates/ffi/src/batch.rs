use crate::*;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

struct BatchStorage {
    tokens: Box<[i32]>,
    embeddings: Box<[f32]>,
    positions: Box<[i32]>,
    counts: Box<[i32]>,
    sequences: Vec<Box<[i32]>>,
    sequence_pointers: Box<[*mut i32]>,
    logits: Box<[i8]>,
}
// SAFETY: each pointer refers to a heap allocation owned by this same object.
// Moving Box/Vec handles does not move those pointees. The registry only allocates,
// validates and frees under a lock; it never dereferences a caller-mutated sequence
// pointer. The foreign caller must not use buffers concurrently with batch_free.
unsafe impl Send for BatchStorage {}
fn registry() -> &'static Mutex<BTreeMap<usize, BatchStorage>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<usize, BatchStorage>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}
impl BatchStorage {
    fn view(&mut self) -> LlamaBatch {
        LlamaBatch {
            n_tokens: 0,
            token: if self.tokens.is_empty() {
                ptr::null_mut()
            } else {
                self.tokens.as_mut_ptr()
            },
            embd: if self.embeddings.is_empty() {
                ptr::null_mut()
            } else {
                self.embeddings.as_mut_ptr()
            },
            pos: self.positions.as_mut_ptr(),
            n_seq_id: self.counts.as_mut_ptr(),
            seq_id: self.sequence_pointers.as_mut_ptr(),
            logits: self.logits.as_mut_ptr(),
        }
    }
}
fn allocate(n_tokens: i32, embedding_width: i32, sequence_max: i32) -> Result<LlamaBatch> {
    let n = count(n_tokens)?;
    let width = count(embedding_width)?;
    let sequences = count(sequence_max)?;
    if n == 0 || n > 65536 || sequences == 0 || sequences > 256 {
        return Err("batch allocation requires 1..65536 tokens and 1..256 sequence slots".into());
    }
    let embedding_elements = n
        .checked_mul(width)
        .ok_or("batch embedding allocation overflow")?;
    let sequence_elements = n
        .checked_mul(sequences)
        .ok_or("batch sequence allocation overflow")?;
    if embedding_elements > MAX_FOREIGN_ELEMENTS || sequence_elements > MAX_FOREIGN_ELEMENTS {
        return Err("batch allocation exceeds checked element limit".into());
    }
    let mut storage = BatchStorage {
        tokens: if width == 0 {
            vec![0; n].into_boxed_slice()
        } else {
            Box::default()
        },
        embeddings: vec![0.0; embedding_elements].into_boxed_slice(),
        positions: vec![0; n].into_boxed_slice(),
        counts: vec![0; n].into_boxed_slice(),
        sequences: (0..n)
            .map(|_| vec![0; sequences].into_boxed_slice())
            .collect(),
        sequence_pointers: Box::default(),
        logits: vec![0; n].into_boxed_slice(),
    };
    let mut pointers: Vec<_> = storage
        .sequences
        .iter_mut()
        .map(|s| s.as_mut_ptr())
        .collect();
    pointers.push(ptr::null_mut()); // matches native sentinel used by llama_batch_free
    storage.sequence_pointers = pointers.into_boxed_slice();
    let batch = storage.view();
    registry()
        .lock()
        .map_err(|_| "batch registry lock poisoned")?
        .insert(batch.seq_id as usize, storage);
    Ok(batch)
}
ffi!(
    llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch,
    LlamaBatch::default(),
    { allocate(n_tokens, embd, n_seq_max) }
);
ffi!(
    llama_batch_get_one(tokens: *mut i32, n_tokens: i32) -> LlamaBatch,
    LlamaBatch::default(),
    {
        let n = count(n_tokens)?;
        if n != 0 {
            unsafe {
                input_slice(tokens, n)?;
            }
        }
        Ok(LlamaBatch {
            n_tokens,
            token: tokens,
            ..LlamaBatch::default()
        })
    }
);
ffi!(llama_batch_free(batch: LlamaBatch) -> (), (), {
    if batch.seq_id.is_null() {
        if batch.token.is_null()
            && batch.embd.is_null()
            && batch.pos.is_null()
            && batch.n_seq_id.is_null()
            && batch.logits.is_null()
        {
            return Ok(());
        }
        return Err("batch_free requires a batch allocated by this library's batch_init".into());
    }
    let mut registry = registry()
        .lock()
        .map_err(|_| "batch registry lock poisoned")?;
    let key = batch.seq_id as usize;
    let storage = registry
        .get_mut(&key)
        .ok_or("batch was not allocated here or was already freed")?;
    let original = storage.view();
    if original.token != batch.token
        || original.embd != batch.embd
        || original.pos != batch.pos
        || original.n_seq_id != batch.n_seq_id
        || original.logits != batch.logits
    {
        return Err("allocated batch pointer fields must not be replaced before batch_free".into());
    }
    registry.remove(&key);
    Ok(())
});

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocated_token_and_embedding_batches_have_native_shape_and_safe_free() {
        unsafe {
            let token = llama_batch_init(3, 0, 2);
            assert_eq!(token.n_tokens, 0);
            assert!(!token.token.is_null());
            assert!(token.embd.is_null());
            assert!((*token.seq_id.add(3)).is_null());
            *(*token.seq_id.add(2)).add(1) = 7;
            *token.token.add(2) = 100;
            llama_batch_free(token);
            set_error("");
            llama_batch_free(token);
            assert!(
                CStr::from_ptr(llama_rust_last_error())
                    .to_str()
                    .unwrap()
                    .contains("already freed")
            );
            let embedding = llama_batch_init(2, 5, 1);
            assert!(embedding.token.is_null());
            assert!(!embedding.embd.is_null());
            *embedding.embd.add(9) = 3.0;
            llama_batch_free(embedding);
        }
    }
    #[test]
    fn unsafe_allocation_sizes_fail_without_allocating() {
        unsafe {
            assert!(llama_batch_init(-1, 0, 1).seq_id.is_null());
            assert!(llama_batch_init(65536, i32::MAX, 1).seq_id.is_null());
            assert!(llama_batch_init(1, 0, 0).seq_id.is_null());
        }
    }
}
