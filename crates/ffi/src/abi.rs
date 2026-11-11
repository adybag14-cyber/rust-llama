//! Frozen llama.cpp public layouts. Enum slots use i32 and C bool slots use u8,
//! so malformed foreign values are validated rather than materializing invalid Rust enums.
use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::ptr;

pub type ProgressCallback = Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>;
pub type AbortCallback = Option<unsafe extern "C" fn(*mut c_void) -> bool>;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaModelParams {
    pub devices: *mut *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: ProgressCallback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: u8,
    pub check_tensors: u8,
    pub use_extra_bufts: u8,
    pub no_host: u8,
    pub no_alloc: u8,
    pub load_mtp: u8,
}
impl Default for LlamaModelParams {
    fn default() -> Self {
        Self {
            devices: ptr::null_mut(),
            tensor_buft_overrides: ptr::null(),
            n_gpu_layers: -1,
            split_mode: 1,
            load_mode: -1,
            lazy_mode: 1,
            main_gpu: 0,
            tensor_split: ptr::null(),
            progress_callback: None,
            progress_callback_user_data: ptr::null_mut(),
            kv_overrides: ptr::null(),
            vocab_only: 0,
            check_tensors: 0,
            use_extra_bufts: 1,
            no_host: 0,
            no_alloc: 0,
            load_mtp: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub moe_cache_size: usize,
    pub moe_cache_cpu_share: f32,
    pub abort_callback: AbortCallback,
    pub abort_callback_data: *mut c_void,
    pub embeddings: u8,
    pub offload_kqv: u8,
    pub no_perf: u8,
    pub op_offload: u8,
    pub swa_full: u8,
    pub kv_unified: u8,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut c_void,
}
impl Default for LlamaContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 512,
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_rs_seq: 0,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 1,
            n_threads: 4,
            n_threads_batch: 4,
            ctx_type: 0,
            rope_scaling_type: -1,
            pooling_type: -1,
            attention_type: -1,
            flash_attn_type: -1,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: -1.0,
            yarn_beta_fast: -1.0,
            yarn_beta_slow: -1.0,
            yarn_orig_ctx: 0,
            defrag_thold: -1.0,
            cb_eval: ptr::null(),
            cb_eval_user_data: ptr::null_mut(),
            type_k: 1,
            type_v: 1,
            moe_cache_size: 0,
            moe_cache_cpu_share: 0.0,
            abort_callback: None,
            abort_callback_data: ptr::null_mut(),
            embeddings: 0,
            offload_kqv: 1,
            no_perf: 1,
            op_offload: 1,
            swa_full: 1,
            kv_unified: 0,
            samplers: ptr::null_mut(),
            n_samplers: 0,
            ctx_other: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut i32,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}
impl Default for LlamaBatch {
    fn default() -> Self {
        Self {
            n_tokens: 0,
            token: ptr::null_mut(),
            embd: ptr::null_mut(),
            pos: ptr::null_mut(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        }
    }
}

/// ABI inspection extension. The same ordered names appear in ABI_LAYOUT.md.
/// Returns usize::MAX past the last item.
pub fn layout(index: usize) -> usize {
    macro_rules! fields {
        ($ty:ty, $($field:ident),+ $(,)?) => {
            &[size_of::<$ty>(), align_of::<$ty>(), $(offset_of!($ty, $field)),+]
        };
    }
    let model: &[usize] = fields!(
        LlamaModelParams,
        devices,
        tensor_buft_overrides,
        n_gpu_layers,
        split_mode,
        load_mode,
        lazy_mode,
        main_gpu,
        tensor_split,
        progress_callback,
        progress_callback_user_data,
        kv_overrides,
        vocab_only,
        check_tensors,
        use_extra_bufts,
        no_host,
        no_alloc,
        load_mtp
    );
    let context: &[usize] = fields!(
        LlamaContextParams,
        n_ctx,
        n_batch,
        n_ubatch,
        n_seq_max,
        n_rs_seq,
        n_outputs_max,
        n_outputs_max_per_seq,
        n_threads,
        n_threads_batch,
        ctx_type,
        rope_scaling_type,
        pooling_type,
        attention_type,
        flash_attn_type,
        rope_freq_base,
        rope_freq_scale,
        yarn_ext_factor,
        yarn_attn_factor,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_orig_ctx,
        defrag_thold,
        cb_eval,
        cb_eval_user_data,
        type_k,
        type_v,
        moe_cache_size,
        moe_cache_cpu_share,
        abort_callback,
        abort_callback_data,
        embeddings,
        offload_kqv,
        no_perf,
        op_offload,
        swa_full,
        kv_unified,
        samplers,
        n_samplers,
        ctx_other
    );
    let batch: &[usize] = fields!(
        LlamaBatch, n_tokens, token, embd, pos, n_seq_id, seq_id, logits
    );
    model
        .iter()
        .chain(context)
        .chain(batch)
        .nth(index)
        .copied()
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_x64_layout() {
        if size_of::<usize>() == 8 {
            assert_eq!(size_of::<LlamaModelParams>(), 80);
            assert_eq!(align_of::<LlamaModelParams>(), 8);
            assert_eq!(offset_of!(LlamaModelParams, vocab_only), 72);
            assert_eq!(size_of::<LlamaContextParams>(), 176);
            assert_eq!(offset_of!(LlamaContextParams, type_k), 104);
            assert_eq!(offset_of!(LlamaContextParams, samplers), 152);
            assert_eq!(size_of::<LlamaBatch>(), 56);
            assert_eq!(offset_of!(LlamaBatch, seq_id), 40);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaTokenData {
    pub id: i32,
    pub logit: f32,
    pub p: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaTokenDataArray {
    pub data: *mut LlamaTokenData,
    pub size: usize,
    pub selected: i64,
    pub sorted: u8,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaLogitBias {
    pub token: i32,
    pub bias: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaSamplerChainParams {
    pub no_perf: u8,
}
#[repr(C)]
pub struct LlamaSampler {
    pub iface: *mut LlamaSamplerI,
    pub ctx: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaSamplerI {
    pub name: Option<unsafe extern "C" fn(*const LlamaSampler) -> *const std::ffi::c_char>,
    pub accept: Option<unsafe extern "C" fn(*mut LlamaSampler, i32)>,
    pub apply: Option<unsafe extern "C" fn(*mut LlamaSampler, *mut LlamaTokenDataArray)>,
    pub reset: Option<unsafe extern "C" fn(*mut LlamaSampler)>,
    pub clone: Option<unsafe extern "C" fn(*const LlamaSampler) -> *mut LlamaSampler>,
    pub free: Option<unsafe extern "C" fn(*mut LlamaSampler)>,
    pub backend_init: Option<unsafe extern "C" fn(*mut LlamaSampler, *mut c_void, u32) -> bool>,
    pub backend_accept:
        Option<unsafe extern "C" fn(*mut LlamaSampler, *mut c_void, *mut c_void, *mut c_void)>,
    pub backend_apply:
        Option<unsafe extern "C" fn(*mut LlamaSampler, *mut c_void, *mut c_void, *mut c_void)>,
    pub backend_set_input: Option<unsafe extern "C" fn(*mut LlamaSampler)>,
    pub backend_reset: Option<unsafe extern "C" fn(*mut LlamaSampler)>,
    pub copy_state: Option<unsafe extern "C" fn(*const LlamaSampler, *mut LlamaSampler)>,
}

pub fn sampler_layout(index: usize) -> usize {
    let values = [
        size_of::<LlamaTokenData>(),
        align_of::<LlamaTokenData>(),
        offset_of!(LlamaTokenData, id),
        offset_of!(LlamaTokenData, logit),
        offset_of!(LlamaTokenData, p),
        size_of::<LlamaTokenDataArray>(),
        align_of::<LlamaTokenDataArray>(),
        offset_of!(LlamaTokenDataArray, data),
        offset_of!(LlamaTokenDataArray, size),
        offset_of!(LlamaTokenDataArray, selected),
        offset_of!(LlamaTokenDataArray, sorted),
        size_of::<LlamaLogitBias>(),
        align_of::<LlamaLogitBias>(),
        offset_of!(LlamaLogitBias, token),
        offset_of!(LlamaLogitBias, bias),
        size_of::<LlamaSamplerChainParams>(),
        align_of::<LlamaSamplerChainParams>(),
        offset_of!(LlamaSamplerChainParams, no_perf),
        size_of::<LlamaSampler>(),
        align_of::<LlamaSampler>(),
        offset_of!(LlamaSampler, iface),
        offset_of!(LlamaSampler, ctx),
        size_of::<LlamaSamplerI>(),
        align_of::<LlamaSamplerI>(),
        offset_of!(LlamaSamplerI, name),
        offset_of!(LlamaSamplerI, accept),
        offset_of!(LlamaSamplerI, apply),
        offset_of!(LlamaSamplerI, reset),
        offset_of!(LlamaSamplerI, clone),
        offset_of!(LlamaSamplerI, free),
        offset_of!(LlamaSamplerI, backend_init),
        offset_of!(LlamaSamplerI, backend_accept),
        offset_of!(LlamaSamplerI, backend_apply),
        offset_of!(LlamaSamplerI, backend_set_input),
        offset_of!(LlamaSamplerI, backend_reset),
        offset_of!(LlamaSamplerI, copy_state),
    ];
    values.get(index).copied().unwrap_or(usize::MAX)
}

/// Borrowed NUL-terminated native chat strings. The caller owns both strings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaChatMessage {
    pub role: *const std::ffi::c_char,
    pub content: *const std::ffi::c_char,
}

/// Frozen public by-value embedding descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaEmbd {
    pub data: *const f32,
    pub n_rows: usize,
    pub n_embd: usize,
}
