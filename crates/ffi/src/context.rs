use crate::*;
use llama_model_runtime::{BatchToken, Context, ContextOptions, Error, Input, OutputSelection};
use std::collections::BTreeMap;
use std::ffi::c_void;

pub struct ContextHandle {
    // A distinct borrowed handle preserves model access after its original owner is freed.
    model_handle: Box<ModelHandle>,
    runtime: Context,
    n_ctx: u32,
    n_ctx_seq: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    embeddings_enabled: bool,
    abort: AbortCallback,
    abort_data: *mut c_void,
    logits: Vec<f32>,
    embeddings: Vec<f32>,
    output_ids: Vec<Option<usize>>,
    output_count: usize,
}
unsafe fn create(model: *mut ModelHandle, p: LlamaContextParams) -> Result<*mut ContextHandle> {
    let data = unsafe { reference(model)? }.data.clone();
    let loaded = data
        .model
        .as_ref()
        .ok_or("vocabulary-only model cannot create a decoding context")?;
    for value in [
        p.embeddings,
        p.offload_kqv,
        p.no_perf,
        p.op_offload,
        p.swa_full,
        p.kv_unified,
    ] {
        boolean(value)?;
    }
    if p.type_k != 0 || p.type_v != 0 {
        return Err("Rust C ABI currently requires type_k = type_v = GGML_TYPE_F32; F16 cache defaults are not yet implemented".into());
    }
    if p.ctx_type != 0
        || ![-1, 0].contains(&p.rope_scaling_type)
        || ![-1, 0].contains(&p.pooling_type)
        || ![-1, 0].contains(&p.attention_type)
        || ![-1, 0].contains(&p.flash_attn_type)
    {
        return Err("only default decoder, unscaled RoPE, no pooling, causal attention and disabled/automatic flash attention are supported".into());
    }
    if !(p.rope_freq_base == 0.0 || p.rope_freq_base == data.rope_base)
        || !(p.rope_freq_scale == 0.0 || p.rope_freq_scale == 1.0)
        || [
            p.yarn_ext_factor,
            p.yarn_attn_factor,
            p.yarn_beta_fast,
            p.yarn_beta_slow,
        ]
        .iter()
        .any(|v| *v != -1.0)
        || p.yarn_orig_ctx != 0
        || p.defrag_thold != -1.0
    {
        return Err("custom RoPE/YaRN and defragmentation parameters are not implemented".into());
    }
    if !p.cb_eval.is_null()
        || !p.samplers.is_null()
        || p.n_samplers != 0
        || !p.ctx_other.is_null()
        || p.moe_cache_size != 0
        || p.moe_cache_cpu_share != 0.0
        || p.n_rs_seq != 0
        || p.n_outputs_max != 0
        || p.n_outputs_max_per_seq != 1
    {
        return Err("evaluation callbacks, backend samplers, shared/MTP contexts, MoE caches and nondefault output limits are not implemented".into());
    }
    if p.no_perf == 0 || p.kv_unified != 0 {
        return Err("performance counters and unified KV buffers are not implemented".into());
    }
    if p.n_threads <= 0 || p.n_threads_batch <= 0 {
        return Err("thread counts must be positive".into());
    }
    if p.n_seq_max == 0
        || p.n_seq_max > 256
        || p.n_batch == 0
        || p.n_batch > 65536
        || p.n_ubatch == 0
    {
        return Err("invalid context sequence/batch limits".into());
    }
    let n_ctx = if p.n_ctx == 0 {
        data.train_context as u32
    } else {
        p.n_ctx
    };
    let n_ctx_seq = n_ctx / p.n_seq_max;
    if n_ctx_seq == 0 || n_ctx_seq > data.train_context as u32 {
        return Err("per-sequence context must be between 1 and the model training context".into());
    }
    let n_ctx = n_ctx_seq
        .checked_mul(p.n_seq_max)
        .ok_or("context length overflow")?;
    let n_batch = p.n_batch.min(n_ctx);
    let n_ubatch = p.n_ubatch.min(n_batch);
    let runtime = Context::new(
        loaded.runtime_model().map_err(|e| e.to_string())?,
        ContextOptions {
            maximum_sequences: p.n_seq_max as usize,
            microbatch_tokens: n_ubatch as usize,
            maximum_batch_tokens: n_batch as usize,
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(Box::into_raw(Box::new(ContextHandle {
        model_handle: Box::new(ModelHandle { data, owned: false }),
        runtime,
        n_ctx,
        n_ctx_seq,
        n_batch,
        n_ubatch,
        n_seq_max: p.n_seq_max,
        embeddings_enabled: p.embeddings != 0,
        abort: p.abort_callback,
        abort_data: p.abort_callback_data,
        logits: Vec::new(),
        embeddings: Vec::new(),
        output_ids: Vec::new(),
        output_count: 0,
    })))
}
ffi!(
    llama_init_from_model(
        model: *mut ModelHandle,
        params: LlamaContextParams,
    ) -> *mut ContextHandle,
    ptr::null_mut(),
    { unsafe { create(model, params) } }
);
ffi!(
    llama_new_context_with_model(
        model: *mut ModelHandle,
        params: LlamaContextParams,
    ) -> *mut ContextHandle,
    ptr::null_mut(),
    { unsafe { create(model, params) } }
);
ffi!(llama_free(context: *mut ContextHandle) -> (), (), {
    if !context.is_null() {
        unsafe {
            reference(context)?;
            drop(Box::from_raw(context));
        }
    }
    Ok(())
});
macro_rules! query {
    ($name:ident,$field:ident) => {
        ffi!($name(context: *const ContextHandle) -> u32, 0, {
            Ok(unsafe { reference(context)? }.$field)
        });
    };
}
query!(llama_n_ctx, n_ctx);
query!(llama_n_ctx_seq, n_ctx_seq);
query!(llama_n_batch, n_batch);
query!(llama_n_ubatch, n_ubatch);
query!(llama_n_seq_max, n_seq_max);
ffi!(llama_n_rs_seq(context: *const ContextHandle) -> u32, 0, {
    unsafe {
        reference(context)?;
    }
    Ok(0)
});
ffi!(
    llama_get_model(context: *const ContextHandle) -> *const ModelHandle,
    ptr::null(),
    { Ok(unsafe { reference(context)? }.model_handle.as_ref()) }
);
ffi!(
    llama_pooling_type(context: *const ContextHandle) -> i32,
    -1,
    {
        unsafe {
            reference(context)?;
        }
        Ok(0)
    }
);
ffi!(llama_n_threads(context: *mut ContextHandle) -> i32, 0, {
    unsafe {
        reference(context)?;
    }
    Ok(1)
});
ffi!(
    llama_n_threads_batch(context: *mut ContextHandle) -> i32,
    0,
    {
        unsafe {
            reference(context)?;
        }
        Ok(1)
    }
);
ffi!(
    llama_set_n_threads(context: *mut ContextHandle, generation: i32, batch: i32) -> (),
    (),
    {
        unsafe {
            reference_mut(context)?;
        }
        if generation != 1 || batch != 1 {
            return Err("the current scalar executor supports exactly one worker thread".into());
        }
        Ok(())
    }
);
ffi!(
    llama_set_embeddings(context: *mut ContextHandle, enabled: u8) -> (),
    (),
    {
        unsafe { reference_mut(context)? }.embeddings_enabled = boolean(enabled)?;
        Ok(())
    }
);
ffi!(
    llama_get_causal_attn(context: *const ContextHandle) -> bool,
    false,
    {
        unsafe {
            reference(context)?;
        }
        Ok(true)
    }
);
ffi!(
    llama_set_causal_attn(context: *mut ContextHandle, causal: u8) -> (),
    (),
    {
        unsafe {
            reference_mut(context)?;
        }
        if !boolean(causal)? {
            return Err("non-causal attention is not implemented for this decoder".into());
        }
        Ok(())
    }
);
ffi!(
    llama_set_abort_callback(
        context: *mut ContextHandle,
        callback: AbortCallback,
        data: *mut c_void,
    ) -> (),
    (),
    {
        let c = unsafe { reference_mut(context)? };
        c.abort = callback;
        c.abort_data = data;
        Ok(())
    }
);
ffi!(llama_synchronize(context: *mut ContextHandle) -> (), (), {
    // All runtime decoding is synchronous; there is no deferred work to wait for.
    unsafe {
        reference_mut(context)?;
    }
    Ok(())
});

unsafe fn decode(context: *mut ContextHandle, batch: LlamaBatch) -> Result<i32> {
    let c = unsafe { reference_mut(context)? };
    let n = count(batch.n_tokens)?;
    if n == 0 || n > c.n_batch as usize {
        return Err("empty or oversized decode batch".into());
    }
    if batch.token.is_null() == batch.embd.is_null() {
        return Err("decode batch must contain exactly one of token or embd".into());
    }
    let tokens = if batch.token.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.token, n)? })
    };
    let width = c.model_handle.data.width as usize;
    let embeddings = if batch.embd.is_null() {
        None
    } else {
        Some(unsafe {
            input_slice(
                batch.embd,
                n.checked_mul(width)
                    .ok_or("embedding dimensions overflow")?,
            )?
        })
    };
    let positions = if batch.pos.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.pos, n)? })
    };
    let seq_counts = if batch.n_seq_id.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.n_seq_id, n)? })
    };
    let seq_ids = if batch.seq_id.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.seq_id, n)? })
    };
    if seq_counts.is_some() != seq_ids.is_some() {
        return Err("n_seq_id and seq_id must both be provided, or both null".into());
    }
    let flags = if batch.logits.is_null() {
        None
    } else {
        Some(unsafe { input_slice(batch.logits, n)? })
    };
    let mut next = BTreeMap::<i32, usize>::new();
    let mut runtime_batch = Vec::with_capacity(n);
    for i in 0..n {
        let sequence = if let (Some(counts), Some(ids)) = (seq_counts, seq_ids) {
            if counts[i] != 1 {
                return Err("shared-token sequence lists are not yet supported; each token must have exactly one sequence".into());
            }
            *unsafe { reference(ids[i])? }
        } else {
            0
        };
        if sequence < 0 || sequence >= c.n_seq_max as i32 {
            return Err("batch sequence id exceeds configured n_seq_max".into());
        }
        let expected = *next
            .entry(sequence)
            .or_insert_with(|| c.runtime.next_position(sequence).unwrap_or(0));
        let position = if let Some(pos) = positions {
            count(pos[i])?
        } else {
            expected
        };
        if position != expected {
            return Err(format!(
                "sequence {sequence} expected position {expected}, got {position}"
            ));
        }
        if position >= c.n_ctx_seq as usize {
            set_error("sequence context is full");
            return Ok(1);
        }
        next.insert(sequence, position + 1);
        let input = if let Some(tokens) = tokens {
            Input::Token(u32::try_from(tokens[i]).map_err(|_| "negative token id")?)
        } else {
            Input::Embedding(embeddings.unwrap()[i * width..(i + 1) * width].to_vec())
        };
        let requested = flags.map_or(c.embeddings_enabled || i == n - 1, |f| f[i] != 0);
        let output = if !requested {
            OutputSelection::None
        } else if c.embeddings_enabled {
            OutputSelection::Both
        } else {
            OutputSelection::Logits
        };
        runtime_batch.push(BatchToken {
            sequence,
            position,
            input,
            output,
        });
    }
    let abort = c.abort;
    let abort_data = c.abort_data;
    let outputs = c.runtime.decode_batch_with_cancel(&runtime_batch, || {
        abort.is_some_and(|callback| unsafe { callback(abort_data) })
    });
    let outputs = match outputs {
        Ok(v) => v,
        Err(error) => {
            set_error(&error.to_string());
            return Ok(match error {
                Error::Cancelled => 2,
                Error::ContextFull { .. } | Error::SequenceLimit { .. } => 1,
                _ => -1,
            });
        }
    };
    let mut logits = Vec::new();
    let mut embeddings = Vec::new();
    let mut output_ids = vec![None; n];
    let mut output_count = 0;
    for output in outputs {
        if let Some(row) = output.logits {
            output_ids[output.batch_index] = Some(output_count);
            output_count += 1;
            logits.extend(row);
            if let Some(row) = output.embedding {
                embeddings.extend(row);
            }
        }
    }
    c.logits = logits;
    c.embeddings = embeddings;
    c.output_ids = output_ids;
    c.output_count = output_count;
    Ok(0)
}
ffi!(
    llama_decode(context: *mut ContextHandle, batch: LlamaBatch) -> i32,
    -1,
    { unsafe { decode(context, batch) } }
);
fn output_index(c: &ContextHandle, index: i32) -> Result<usize> {
    if index < 0 {
        let resolved = c.output_count as i64 + index as i64;
        if resolved < 0 || resolved >= c.output_count as i64 {
            return Err("negative output index out of range".into());
        }
        Ok(resolved as usize)
    } else {
        c.output_ids
            .get(index as usize)
            .copied()
            .flatten()
            .ok_or_else(|| "batch token did not request an output or index is out of range".into())
    }
}
ffi!(
    llama_get_logits(context: *mut ContextHandle) -> *mut f32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        Ok(if c.logits.is_empty() {
            ptr::null_mut()
        } else {
            c.logits.as_mut_ptr()
        })
    }
);
ffi!(
    llama_get_logits_ith(context: *mut ContextHandle, index: i32) -> *mut f32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        let row = output_index(c, index)?;
        let offset = row
            .checked_mul(c.model_handle.data.vocab.texts.len())
            .ok_or("logit offset overflow")?;
        if offset >= c.logits.len() {
            return Err("logits are unavailable".into());
        }
        Ok(unsafe { c.logits.as_mut_ptr().add(offset) })
    }
);
ffi!(
    llama_get_embeddings(context: *mut ContextHandle) -> *mut f32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        Ok(if c.embeddings.is_empty() {
            ptr::null_mut()
        } else {
            c.embeddings.as_mut_ptr()
        })
    }
);
ffi!(
    llama_get_embeddings_ith(context: *mut ContextHandle, index: i32) -> *mut f32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        let row = output_index(c, index)?;
        let offset = row
            .checked_mul(c.model_handle.data.width as usize)
            .ok_or("embedding offset overflow")?;
        if offset >= c.embeddings.len() {
            return Err("embeddings were not enabled".into());
        }
        Ok(unsafe { c.embeddings.as_mut_ptr().add(offset) })
    }
);

pub(crate) unsafe fn sampling_values(context: *mut ContextHandle, index: i32) -> Result<Vec<f32>> {
    let c = unsafe { reference(context)? };
    let row = output_index(c, index)?;
    let width = c.model_handle.data.vocab.texts.len();
    let offset = row
        .checked_mul(width)
        .ok_or("sampling logit offset overflow")?;
    let end = offset
        .checked_add(width)
        .ok_or("sampling logit length overflow")?;
    c.logits
        .get(offset..end)
        .map(|v| v.to_vec())
        .ok_or_else(|| "requested logits are unavailable".into())
}
