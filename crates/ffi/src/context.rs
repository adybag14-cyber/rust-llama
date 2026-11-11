use crate::*;
use llama_model_runtime::{
    Error, Input, NativeBatchToken, NativeContext, NativeContextOptions, NativeExecutionOptions,
    NativeKvType, OutputSelection,
};
use std::collections::BTreeMap;
use std::ffi::c_void;

pub struct ContextHandle {
    // A distinct borrowed handle preserves model access after its original owner is freed.
    pub(crate) model_handle: Box<ModelHandle>,
    pub(crate) adapters: crate::adapter::ContextAdapters,
    pub(crate) runtime: NativeContext,
    pub(crate) state_snapshots: BTreeMap<i32, Vec<u8>>,
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
    let cache_type = |value| match value {
        0 => Ok(NativeKvType::F32),
        1 => Ok(NativeKvType::F16),
        _ => Err("native CPU cache requires GGML_TYPE_F32 or GGML_TYPE_F16"),
    };
    let key_type = cache_type(p.type_k)?;
    let value_type = cache_type(p.type_v)?;
    if p.ctx_type != 0
        || ![-1, 0].contains(&p.rope_scaling_type)
        || ![-1, 0].contains(&p.pooling_type)
        || ![-1, 0].contains(&p.attention_type)
        || ![-1, 0, 1].contains(&p.flash_attn_type)
    {
        return Err("only default decoder, unscaled RoPE, no pooling, causal attention and defined AUTO/DISABLED/ENABLED flash modes are supported".into());
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
    if p.no_perf == 0 {
        return Err("performance counters are not implemented".into());
    }
    let threads = usize::try_from(p.n_threads).map_err(|_| "negative generation worker count")?;
    let threads_batch =
        usize::try_from(p.n_threads_batch).map_err(|_| "negative batch worker count")?;
    if p.n_seq_max == 0 || p.n_seq_max > 256 || p.n_batch == 0 || p.n_batch > 65536 {
        return Err("invalid context sequence/batch limits".into());
    }
    let n_ctx = if p.n_ctx == 0 {
        data.train_context as u32
    } else {
        p.n_ctx
    };
    let n_batch = p.n_batch.min(n_ctx);
    // The frozen output reserve asserts if n_seq_max exceeds its default
    // n_batch output-row ceiling. Reject that source-invalid configuration
    // before construction instead of exposing a process abort through C.
    if p.n_seq_max > n_batch {
        return Err("configured sequence count exceeds the default output-row ceiling".into());
    }
    let n_ubatch = (if p.n_ubatch == 0 {
        p.n_batch
    } else {
        p.n_ubatch
    })
    .min(n_batch);
    // Frozen ordinary CPU support accepts FLASH_ATTN_EXT and assigns the probe
    // to the same CPU as the model layer. Within this admitted Llama/F32-or-F16
    // domain AUTO therefore resolves to enabled; other architectures/devices
    // remain outside this constructor's compatibility scope.
    let runtime = NativeContext::new_with_execution(
        loaded.runtime_model().map_err(|e| e.to_string())?,
        NativeContextOptions {
            context_tokens: n_ctx as usize,
            maximum_sequences: p.n_seq_max as usize,
            microbatch_tokens: n_ubatch as usize,
            maximum_batch_tokens: n_batch as usize,
            unified: p.kv_unified != 0,
        },
        key_type,
        value_type,
        NativeExecutionOptions {
            flash_attention: p.flash_attn_type != 0,
            threads,
            threads_batch,
            embeddings: p.embeddings != 0,
        },
    )
    .map_err(|e| e.to_string())?;
    let n_ctx = runtime.context_tokens() as u32;
    let n_ctx_seq = runtime.context_tokens_per_sequence() as u32;
    Ok(Box::into_raw(Box::new(ContextHandle {
        model_handle: Box::new(ModelHandle { data, owned: false }),
        adapters: crate::adapter::ContextAdapters::default(),
        runtime,
        state_snapshots: BTreeMap::new(),
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
    Ok(unsafe { reference(context)? }
        .runtime
        .execution_options()
        .threads as i32)
});
ffi!(
    llama_n_threads_batch(context: *mut ContextHandle) -> i32,
    0,
    {
        Ok(unsafe { reference(context)? }
            .runtime
            .execution_options()
            .threads_batch as i32)
    }
);
ffi!(
    llama_set_n_threads(context: *mut ContextHandle, generation: i32, batch: i32) -> (),
    (),
    {
        let generation =
            usize::try_from(generation).map_err(|_| "negative generation worker count")?;
        let batch = usize::try_from(batch).map_err(|_| "negative batch worker count")?;
        unsafe { reference_mut(context)? }
            .runtime
            .set_threads(generation, batch)
            .map_err(|error| error.to_string())
    }
);
ffi!(
    llama_set_embeddings(context: *mut ContextHandle, enabled: u8) -> (),
    (),
    {
        let enabled = boolean(enabled)?;
        let context = unsafe { reference_mut(context)? };
        context.runtime.set_embeddings(enabled);
        context.embeddings_enabled = enabled;
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
        let sequences = if let (Some(counts), Some(ids)) = (seq_counts, seq_ids) {
            let number = count(counts[i])?;
            if number == 0 || number > c.n_seq_max as usize {
                return Err("invalid per-token sequence count".into());
            }
            unsafe { input_slice(ids[i], number)? }.to_vec()
        } else {
            vec![0]
        };
        let sequence = sequences[0];
        for &id in &sequences {
            if id < 0 || id as usize >= c.runtime.sequence_id_limit() {
                return Err("batch sequence ID exceeds the configured native memory limit".into());
            }
        }
        let expected = *next.entry(sequence).or_insert(
            c.runtime
                .next_position(sequence)
                .map_err(|e| e.to_string())? as usize,
        );
        let position = if let Some(pos) = positions {
            usize::try_from(pos[i]).map_err(|_| "negative token position")?
        } else {
            expected
        };
        for &id in &sequences {
            next.insert(id, position + 1);
        }
        let input = if let Some(tokens) = tokens {
            Input::Token(u32::try_from(tokens[i]).map_err(|_| "negative token id")?)
        } else {
            Input::Embedding(embeddings.unwrap()[i * width..(i + 1) * width].to_vec())
        };
        let requested = c.embeddings_enabled || flags.map_or(i == n - 1, |f| f[i] != 0);
        let output = if !requested {
            OutputSelection::None
        } else if c.embeddings_enabled {
            OutputSelection::Both
        } else {
            OutputSelection::Logits
        };
        runtime_batch.push(NativeBatchToken {
            sequences,
            position: i32::try_from(position).map_err(|_| "position exceeds INT32_MAX")?,
            input,
            output,
        });
    }
    execute_native_batch(c, &runtime_batch)
}

pub(crate) fn execute_native_batch(
    c: &mut ContextHandle,
    runtime_batch: &[NativeBatchToken],
) -> Result<i32> {
    let n = runtime_batch.len();
    let abort = c.abort;
    let abort_data = c.abort_data;
    let outputs = c.runtime.decode_batch_with_cancel(runtime_batch, || {
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

impl ContextHandle {
    pub(crate) fn extended_batch_configuration(&self) -> (usize, usize, usize, usize, bool) {
        (
            self.n_batch as usize,
            self.n_seq_max as usize,
            self.model_handle.data.vocab.texts.len(),
            self.model_handle.data.width as usize,
            self.embeddings_enabled,
        )
    }
}
