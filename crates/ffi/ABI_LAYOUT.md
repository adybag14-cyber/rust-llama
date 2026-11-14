# ABI inspection indices

`llama_rust_abi_layout(index)` returns the values below using Rust `size_of`, `align_of` and `offset_of`. The external frozen-header harness independently computes the corresponding C values. All indices are zero-based; out-of-range returns `SIZE_MAX`.

| Index | Value |
|---:|---|
| 0 | `sizeof(llama_model_params)` |
| 1 | `alignof(llama_model_params)` |
| 2 | `offsetof(llama_model_params, devices)` |
| 3 | `offsetof(llama_model_params, tensor_buft_overrides)` |
| 4 | `offsetof(llama_model_params, n_gpu_layers)` |
| 5 | `offsetof(llama_model_params, split_mode)` |
| 6 | `offsetof(llama_model_params, load_mode)` |
| 7 | `offsetof(llama_model_params, lazy_mode)` |
| 8 | `offsetof(llama_model_params, main_gpu)` |
| 9 | `offsetof(llama_model_params, tensor_split)` |
| 10 | `offsetof(llama_model_params, progress_callback)` |
| 11 | `offsetof(llama_model_params, progress_callback_user_data)` |
| 12 | `offsetof(llama_model_params, kv_overrides)` |
| 13 | `offsetof(llama_model_params, vocab_only)` |
| 14 | `offsetof(llama_model_params, check_tensors)` |
| 15 | `offsetof(llama_model_params, use_extra_bufts)` |
| 16 | `offsetof(llama_model_params, no_host)` |
| 17 | `offsetof(llama_model_params, no_alloc)` |
| 18 | `offsetof(llama_model_params, load_mtp)` |
| 19 | `sizeof(llama_context_params)` |
| 20 | `alignof(llama_context_params)` |
| 21 | `offsetof(llama_context_params, n_ctx)` |
| 22 | `offsetof(llama_context_params, n_batch)` |
| 23 | `offsetof(llama_context_params, n_ubatch)` |
| 24 | `offsetof(llama_context_params, n_seq_max)` |
| 25 | `offsetof(llama_context_params, n_rs_seq)` |
| 26 | `offsetof(llama_context_params, n_outputs_max)` |
| 27 | `offsetof(llama_context_params, n_outputs_max_per_seq)` |
| 28 | `offsetof(llama_context_params, n_threads)` |
| 29 | `offsetof(llama_context_params, n_threads_batch)` |
| 30 | `offsetof(llama_context_params, ctx_type)` |
| 31 | `offsetof(llama_context_params, rope_scaling_type)` |
| 32 | `offsetof(llama_context_params, pooling_type)` |
| 33 | `offsetof(llama_context_params, attention_type)` |
| 34 | `offsetof(llama_context_params, flash_attn_type)` |
| 35 | `offsetof(llama_context_params, rope_freq_base)` |
| 36 | `offsetof(llama_context_params, rope_freq_scale)` |
| 37 | `offsetof(llama_context_params, yarn_ext_factor)` |
| 38 | `offsetof(llama_context_params, yarn_attn_factor)` |
| 39 | `offsetof(llama_context_params, yarn_beta_fast)` |
| 40 | `offsetof(llama_context_params, yarn_beta_slow)` |
| 41 | `offsetof(llama_context_params, yarn_orig_ctx)` |
| 42 | `offsetof(llama_context_params, defrag_thold)` |
| 43 | `offsetof(llama_context_params, cb_eval)` |
| 44 | `offsetof(llama_context_params, cb_eval_user_data)` |
| 45 | `offsetof(llama_context_params, type_k)` |
| 46 | `offsetof(llama_context_params, type_v)` |
| 47 | `offsetof(llama_context_params, moe_cache_size)` |
| 48 | `offsetof(llama_context_params, moe_cache_cpu_share)` |
| 49 | `offsetof(llama_context_params, abort_callback)` |
| 50 | `offsetof(llama_context_params, abort_callback_data)` |
| 51 | `offsetof(llama_context_params, embeddings)` |
| 52 | `offsetof(llama_context_params, offload_kqv)` |
| 53 | `offsetof(llama_context_params, no_perf)` |
| 54 | `offsetof(llama_context_params, op_offload)` |
| 55 | `offsetof(llama_context_params, swa_full)` |
| 56 | `offsetof(llama_context_params, kv_unified)` |
| 57 | `offsetof(llama_context_params, samplers)` |
| 58 | `offsetof(llama_context_params, n_samplers)` |
| 59 | `offsetof(llama_context_params, ctx_other)` |
| 60 | `sizeof(llama_batch)` |
| 61 | `alignof(llama_batch)` |
| 62 | `offsetof(llama_batch, n_tokens)` |
| 63 | `offsetof(llama_batch, token)` |
| 64 | `offsetof(llama_batch, embd)` |
| 65 | `offsetof(llama_batch, pos)` |
| 66 | `offsetof(llama_batch, n_seq_id)` |
| 67 | `offsetof(llama_batch, seq_id)` |
| 68 | `offsetof(llama_batch, logits)` |

## Sampler layout extension

llama_rust_sampler_abi_layout(index) has its own index space; the original
69-entry llama_rust_abi_layout table remains unchanged. SIZE_MAX marks the end.
The values below were checked against the frozen header on Windows x64.

| Index | Native item | Observed value |
|---|---|---|
| 0 | sizeof(llama_token_data) | 12 |
| 1 | alignof(llama_token_data) | 4 |
| 2 | llama_token_data.id | 0 |
| 3 | llama_token_data.logit | 4 |
| 4 | llama_token_data.p | 8 |
| 5 | sizeof(llama_token_data_array) | 32 |
| 6 | alignof(llama_token_data_array) | 8 |
| 7 | llama_token_data_array.data | 0 |
| 8 | llama_token_data_array.size | 8 |
| 9 | llama_token_data_array.selected | 16 |
| 10 | llama_token_data_array.sorted | 24 |
| 11 | sizeof(llama_logit_bias) | 8 |
| 12 | alignof(llama_logit_bias) | 4 |
| 13 | llama_logit_bias.token | 0 |
| 14 | llama_logit_bias.bias | 4 |
| 15 | sizeof(llama_sampler_chain_params) | 1 |
| 16 | alignof(llama_sampler_chain_params) | 1 |
| 17 | llama_sampler_chain_params.no_perf | 0 |
| 18 | sizeof(llama_sampler) | 16 |
| 19 | alignof(llama_sampler) | 8 |
| 20 | llama_sampler.iface | 0 |
| 21 | llama_sampler.ctx | 8 |
| 22 | sizeof(llama_sampler_i) | 96 |
| 23 | alignof(llama_sampler_i) | 8 |
| 24 | llama_sampler_i.name | 0 |
| 25 | llama_sampler_i.accept | 8 |
| 26 | llama_sampler_i.apply | 16 |
| 27 | llama_sampler_i.reset | 24 |
| 28 | llama_sampler_i.clone | 32 |
| 29 | llama_sampler_i.free | 40 |
| 30 | llama_sampler_i.backend_init | 48 |
| 31 | llama_sampler_i.backend_accept | 56 |
| 32 | llama_sampler_i.backend_apply | 64 |
| 33 | llama_sampler_i.backend_set_input | 72 |
| 34 | llama_sampler_i.backend_reset | 80 |
| 35 | llama_sampler_i.copy_state | 88 |

## Context sampler attachment

The added repr(C) LlamaSamplerSeqConfig matches the frozen
llama_sampler_seq_config on the qualified 64-bit hosts: size 16, alignment 8,
seq_id at offset 0 and sampler at offset 8. It is passed through the existing
context-parameter samplers/n_samplers fields, so their established layout does
not change. The original-header backend-sampling clients exercise constructor
attachments on both MSVC and GNU debug/release libraries. The 36-entry sampler
layout inspection extension is unchanged; no new inspection indices are implied.
See BACKEND_SAMPLING.md and fixtures/ffi/backend-sampling/provenance.json.
