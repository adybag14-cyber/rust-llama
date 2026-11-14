# Frozen C ABI export coverage

The frozen header declares 255 llama C functions, and the original DLL contains
those 255 unmangled symbols. Its 39 additional mangled C++ exports are outside
this public C-header inventory. The Rust library implements **247**, leaves
**8 unexported**, and adds **4 diagnostic/layout extensions**. This is symbol
coverage; accepted configurations and behavioral limits remain in README.md,
SAMPLERS.md, STATE.md, EXECUTION.md, SPLITS.md, ENUMS.md, METADATA.md, CHAT.md, BATCH_EXT.md, PERFORMANCE.md, CONTEXT_PERFORMANCE.md, FILE_POINTER.md, ADAPTERS.md, ENCODER.md and BACKEND_SAMPLING.md.

The context-performance and FILE-pointer checkpoint adds seven real exports.
CONTEXT_PERFORMANCE.md records measured context timing, synchronization/reset
boundaries, dense warmup behavior and truthful runtime feature information.
The direct executor reports zero GGML graph allocation/reuse counts.
FILE_POINTER.md records anonymous embedded model/LoRA loading, caller ownership,
independent MSVC/GNU fixtures and the explicit coherent-cursor behavior boundary.

The quantization checkpoint adds `llama_model_quantize_default_params` and
`llama_model_quantize` with native Rust model-file conversion. The original
64-byte parameter layout and typed override/importance arrays are independently
checked through frozen-header C callers on Windows and GNU debug/release
libraries. Each profile replays 88 requests and 76 exact complete output files.
The [model quantization contract](../model-quant/README.md) records its eight
admitted file families, bounded resource policy, independent metadata corpus
and explicit unsupported domains. Full quantizer/architecture coverage remains
a separate release requirement.

The encoder checkpoint adds `llama_encode`, `llama_get_embeddings_seq`,
`llama_model_n_cls_out` and `llama_model_cls_label` with real complete-batch
BERT execution, native sequence grouping, pooling and optional classifier
heads. It admits full BERT loading and context ownership. ENCODER.md records
the independently captured MSVC/GNU ordering, numerical, output-lifetime and
bounded-error contract. Decoder-only memory, state and adapter execution are
not extended to BERT.

The adapter checkpoint adds ten real LoRA/control-vector functions. Owned GGUF
weights, scalar metadata, invocation-token storage, native ordered composition
and retained partial control vectors are independently checked on MSVC/GNU.
ADAPTERS.md records source behavior, numerical fixtures and resource/lifetime
boundaries. FILE_POINTER.md extends loading to caller-owned FILE streams.

The sampler-performance checkpoint adds real per-chain CPU time/counts and
three perf functions, plus two global logger functions. PERFORMANCE.md records
no_perf, reset/clone/copy rules, callback ownership and independent MSVC/GNU
captures. Context timing is covered separately in CONTEXT_PERFORMANCE.md;
context-attached sampling is covered separately in BACKEND_SAMPLING.md.

The extended batch checkpoint checks debug/release PE and ELF tables: exactly
211 frozen exports, four extensions and 44 absent exports. The original-header
client passes on all four Rust libraries; see `evidence/2026-11-11/c-batch-ext`.
It adds twelve public builders and llama_process
for the existing admitted dense Llama decode contexts. BATCH_EXT.md records
source builder side effects, owned embedding storage, mixed input, output rules
and the independently captured Windows/GNU corpus. The subsequent encoder checkpoint is scoped separately in ENCODER.md;
BERT extended-batch processing and other native context architectures remain open.

The model-metadata/chat extension adds five scalar metadata exports and two
byte-oriented chat exports. Their independent fixtures and native client checks
are documented in METADATA.md and CHAT.md. Symbol presence does not remove their
explicit platform, resource or semantic boundaries.

At the 10 November checkpoint, integrated debug/release PE and ELF export tables were inspected on
both hosts and contained exactly 198 frozen exports and four extensions.
At that checkpoint, 57 exports remained absent. The independent original
libraries still expose the complete 255-symbol set. Original-header chat,
metadata and raw-name clients pass against all four integrated Rust builds;
see `evidence/2026-11-10/continued-port`.

Earlier debug/release PE and ELF export tables were checked against the original
255-symbol inventory. The static enum-helper checkpoint added only
`llama_ftype_name`, `llama_flash_attn_type_name`, `llama_load_mode_name` and
`llama_load_mode_from_str`. Its independent native results and complete symbol
sets are recorded in `evidence/2026-11-09/c-enum-api`. Descriptive helper
availability does not enable an execution, quantizer or loading mode.

The explicit-split checkpoint added `llama_model_load_from_splits`,
`llama_split_path` and `llama_split_prefix`, reaching 187 source exports and
four extensions. Its retained C caller evidence is in
`evidence/2026-11-09/c-split-api`. No absent function is represented by a
placeholder export. Symbol presence does not close all behavioral or platform
requirements.

The earlier native microbatch/flash checkpoint retained 184 source C exports
and the same four extensions. It expanded existing context, worker, decode
and state behavior; its PE/ELF evidence remains unchanged in
`evidence/2026-11-09/native-flash-context`. Earlier native-context, sampler-cpu
and sampler-ffi inventories are also preserved as historical checkpoints.
The separate native C++ symbol surface is not claimed here.

The context-sampling checkpoint adds eight exports with real Rust CPU prefix
execution, source-defined CPU suffixes and context-owned sampled outputs.
BACKEND_SAMPLING.md records supported stages, exact original MSVC/GNU accessor
observations, per-sequence limits, checked attachment lifetimes and remaining
backend-stage gaps. The debug/release PE and ELF inventory is 247 frozen
exports plus four extensions; no absent API has a placeholder symbol.

## Implemented frozen exports

| Symbol |
|---|
| `llama_adapter_get_alora_invocation_tokens` |
| `llama_adapter_get_alora_n_invocation_tokens` |
| `llama_adapter_lora_free` |
| `llama_adapter_lora_init` |
| `llama_adapter_lora_init_from_file_ptr` |
| `llama_adapter_meta_count` |
| `llama_adapter_meta_key_by_index` |
| `llama_adapter_meta_val_str` |
| `llama_adapter_meta_val_str_by_index` |
| `llama_add_bos_token` |
| `llama_add_eos_token` |
| `llama_backend_free` |
| `llama_backend_init` |
| `llama_batch_ext_add` |
| `llama_batch_ext_add_embd` |
| `llama_batch_ext_add_seq` |
| `llama_batch_ext_add_token` |
| `llama_batch_ext_clear` |
| `llama_batch_ext_free` |
| `llama_batch_ext_init` |
| `llama_batch_ext_set_embd_state` |
| `llama_batch_ext_set_embd_token` |
| `llama_batch_ext_set_output_embd` |
| `llama_batch_ext_set_output_logits` |
| `llama_batch_ext_set_pos` |
| `llama_batch_free` |
| `llama_batch_get_one` |
| `llama_batch_init` |
| `llama_chat_apply_template` |
| `llama_chat_builtin_templates` |
| `llama_context_default_params` |
| `llama_copy_state_data` |
| `llama_decode` |
| `llama_detokenize` |
| `llama_encode` |
| `llama_flash_attn_type_name` |
| `llama_free` |
| `llama_free_model` |
| `llama_ftype_name` |
| `llama_get_causal_attn` |
| `llama_get_embeddings` |
| `llama_get_embeddings_ith` |
| `llama_get_embeddings_seq` |
| `llama_get_logits` |
| `llama_get_logits_ith` |
| `llama_get_memory` |
| `llama_get_model` |
| `llama_get_sampled_candidates_count_ith` |
| `llama_get_sampled_candidates_ith` |
| `llama_get_sampled_logits_count_ith` |
| `llama_get_sampled_logits_ith` |
| `llama_get_sampled_probs_count_ith` |
| `llama_get_sampled_probs_ith` |
| `llama_get_sampled_token_ith` |
| `llama_get_state_size` |
| `llama_init_from_model` |
| `llama_load_mode_from_str` |
| `llama_load_mode_name` |
| `llama_load_model_from_file` |
| `llama_load_session_file` |
| `llama_log_get` |
| `llama_log_set` |
| `llama_max_devices` |
| `llama_max_parallel_sequences` |
| `llama_max_tensor_buft_overrides` |
| `llama_memory_can_shift` |
| `llama_memory_clear` |
| `llama_memory_seq_add` |
| `llama_memory_seq_cp` |
| `llama_memory_seq_div` |
| `llama_memory_seq_keep` |
| `llama_memory_seq_pos_max` |
| `llama_memory_seq_pos_min` |
| `llama_memory_seq_rm` |
| `llama_model_chat_template` |
| `llama_model_cls_label` |
| `llama_model_decoder_start_token` |
| `llama_model_default_params` |
| `llama_model_desc` |
| `llama_model_free` |
| `llama_model_ftype` |
| `llama_model_get_vocab` |
| `llama_model_has_decoder` |
| `llama_model_has_encoder` |
| `llama_model_is_diffusion` |
| `llama_model_is_hybrid` |
| `llama_model_is_recurrent` |
| `llama_model_load_from_file` |
| `llama_model_load_from_file_ptr` |
| `llama_model_load_from_splits` |
| `llama_model_meta_count` |
| `llama_model_meta_key_by_index` |
| `llama_model_meta_key_str` |
| `llama_model_meta_val_str` |
| `llama_model_meta_val_str_by_index` |
| `llama_model_n_cls_out` |
| `llama_model_n_ctx_train` |
| `llama_model_n_embd` |
| `llama_model_n_embd_inp` |
| `llama_model_n_embd_out` |
| `llama_model_n_head` |
| `llama_model_n_head_kv` |
| `llama_model_n_layer` |
| `llama_model_n_layer_nextn` |
| `llama_model_n_params` |
| `llama_model_n_swa` |
| `llama_model_quantize` |
| `llama_model_quantize_default_params` |
| `llama_model_rope_freq_scale_train` |
| `llama_model_rope_type` |
| `llama_model_size` |
| `llama_n_batch` |
| `llama_n_ctx` |
| `llama_n_ctx_seq` |
| `llama_n_ctx_train` |
| `llama_n_embd` |
| `llama_n_head` |
| `llama_n_layer` |
| `llama_n_rs_seq` |
| `llama_n_seq_max` |
| `llama_n_threads` |
| `llama_n_threads_batch` |
| `llama_n_ubatch` |
| `llama_n_vocab` |
| `llama_new_context_with_model` |
| `llama_perf_context` |
| `llama_perf_context_print` |
| `llama_perf_context_reset` |
| `llama_perf_sampler` |
| `llama_perf_sampler_print` |
| `llama_perf_sampler_reset` |
| `llama_pooling_type` |
| `llama_print_system_info` |
| `llama_process` |
| `llama_sampler_accept` |
| `llama_sampler_apply` |
| `llama_sampler_chain_add` |
| `llama_sampler_chain_default_params` |
| `llama_sampler_chain_get` |
| `llama_sampler_chain_init` |
| `llama_sampler_chain_n` |
| `llama_sampler_chain_remove` |
| `llama_sampler_clone` |
| `llama_sampler_copy` |
| `llama_sampler_free` |
| `llama_sampler_get_seed` |
| `llama_sampler_init` |
| `llama_sampler_init_adaptive_p` |
| `llama_sampler_init_dist` |
| `llama_sampler_init_dry` |
| `llama_sampler_init_grammar` |
| `llama_sampler_init_grammar_lazy` |
| `llama_sampler_init_grammar_lazy_patterns` |
| `llama_sampler_init_greedy` |
| `llama_sampler_init_infill` |
| `llama_sampler_init_logit_bias` |
| `llama_sampler_init_min_p` |
| `llama_sampler_init_mirostat` |
| `llama_sampler_init_mirostat_v2` |
| `llama_sampler_init_penalties` |
| `llama_sampler_init_temp` |
| `llama_sampler_init_temp_ext` |
| `llama_sampler_init_top_k` |
| `llama_sampler_init_top_n_sigma` |
| `llama_sampler_init_top_p` |
| `llama_sampler_init_typical` |
| `llama_sampler_init_xtc` |
| `llama_sampler_name` |
| `llama_sampler_reset` |
| `llama_sampler_sample` |
| `llama_save_session_file` |
| `llama_set_abort_callback` |
| `llama_set_adapter_cvec` |
| `llama_set_adapters_lora` |
| `llama_set_causal_attn` |
| `llama_set_embeddings` |
| `llama_set_n_threads` |
| `llama_set_sampler` |
| `llama_set_state_data` |
| `llama_set_warmup` |
| `llama_split_path` |
| `llama_split_prefix` |
| `llama_state_get_data` |
| `llama_state_get_size` |
| `llama_state_load_file` |
| `llama_state_save_file` |
| `llama_state_seq_get_data` |
| `llama_state_seq_get_data_ext` |
| `llama_state_seq_get_size` |
| `llama_state_seq_get_size_ext` |
| `llama_state_seq_load_file` |
| `llama_state_seq_save_file` |
| `llama_state_seq_set_data` |
| `llama_state_seq_set_data_ext` |
| `llama_state_set_data` |
| `llama_supports_gpu_offload` |
| `llama_supports_mlock` |
| `llama_supports_mmap` |
| `llama_supports_rpc` |
| `llama_synchronize` |
| `llama_time_us` |
| `llama_token_bos` |
| `llama_token_cls` |
| `llama_token_eos` |
| `llama_token_eot` |
| `llama_token_fim_mid` |
| `llama_token_fim_pad` |
| `llama_token_fim_pre` |
| `llama_token_fim_rep` |
| `llama_token_fim_sep` |
| `llama_token_fim_suf` |
| `llama_token_get_attr` |
| `llama_token_get_score` |
| `llama_token_get_text` |
| `llama_token_is_control` |
| `llama_token_is_eog` |
| `llama_token_nl` |
| `llama_token_pad` |
| `llama_token_sep` |
| `llama_token_to_piece` |
| `llama_tokenize` |
| `llama_version` |
| `llama_vocab_bos` |
| `llama_vocab_cls` |
| `llama_vocab_eos` |
| `llama_vocab_eot` |
| `llama_vocab_fim_mid` |
| `llama_vocab_fim_pad` |
| `llama_vocab_fim_pre` |
| `llama_vocab_fim_rep` |
| `llama_vocab_fim_sep` |
| `llama_vocab_fim_suf` |
| `llama_vocab_get_add_bos` |
| `llama_vocab_get_add_eos` |
| `llama_vocab_get_add_sep` |
| `llama_vocab_get_attr` |
| `llama_vocab_get_score` |
| `llama_vocab_get_suppress_tokens` |
| `llama_vocab_get_text` |
| `llama_vocab_is_control` |
| `llama_vocab_is_eog` |
| `llama_vocab_mask` |
| `llama_vocab_n_tokens` |
| `llama_vocab_nl` |
| `llama_vocab_pad` |
| `llama_vocab_sep` |
| `llama_vocab_type` |

## Rust extensions

| Symbol |
|---|
| `llama_rust_abi_layout` |
| `llama_rust_clear_error` |
| `llama_rust_last_error` |
| `llama_rust_sampler_abi_layout` |

## Not implemented / not exported

| Symbol |
|---|
| `llama_attach_threadpool` |
| `llama_detach_threadpool` |
| `llama_model_init_from_user` |
| `llama_model_save_to_file` |
| `llama_numa_init` |
| `llama_opt_epoch` |
| `llama_opt_init` |
| `llama_opt_param_filter_all` |

## Remaining model construction and save prerequisites

`llama_model_init_from_user` takes a caller-supplied `gguf_context` and a
callback receiving live `ggml_tensor` objects with backend-owned storage.
A complete implementation needs a Rust GGUF C-context API, tensor descriptors,
allocation/backend ownership and initialization callback behavior. Treating an
original C++ GGUF object's opaque storage as a guessed Rust structure, or
linking its implementation, would violate the native Rust boundary. The current
safe GGUF parser/writer does not itself supply that C ABI.

`llama_model_save_to_file` is also not an input-file copy. The frozen
`llama-model-saver.cpp` rebuilds canonical metadata from resolved hparams and
vocabulary, then serializes actual model tensors in architecture creation order.
It includes defaulted fields, tokenizer policies, alias/optional-tensor handling
and model-specific metadata types. Current ModelData owns scalar getter metadata
and execution weights, but not a complete resolved native export representation.
A correct saver needs that owned representation and independent complete-file
captures across admitted architectures, quantized weights and vocabulary
families. Both functions therefore remain absent rather than exporting a
passthrough or an opaque foreign-object cast.
