# Native Qwen2 and Qwen3 checkpoint conversion

The native Rust HF converter now admits dense Qwen2 and Qwen3 checkpoints in addition to its existing dense Llama path. The implementation targets the supplied frozen llama.cpp archive, SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. It does not select the current upstream converter at runtime.

The Rust implementation reads local Safetensors and sidecars, produces the planned GGUF stream, and publishes a completed file. Python, Torch, NumPy, Transformers and Tokenizers are independent reference tools outside the product. The existing reviewed YAML dependency cohort is unchanged.

## Executable scope

The admitted aliases are `Qwen2ForCausalLM`, `Qwen2Model`, `Qwen3ForCausalLM` and `Qwen3Model`, including supported nested text configurations. The three existing Llama aliases remain supported.

| Surface | Qwen behavior |
| --- | --- |
| Tensor names | Frozen family-specific alias tables, filters and insertion order |
| Attention weights | Original Qwen row order; separate/fused QKV projections and compatible biases |
| Qwen3 normalization | Per-head Q/K norm names and independently configured head width |
| Output formats | auto, F32, F16, BF16, Q8_0, TQ1_0, TQ2_0, including original width-dependent F16 fallback |
| Input scalar types | The shared independently qualified cast/packing domain; source packed/prequantized layouts remain separate |
| Configuration | Pinned Transformers defaults, explicit-null behavior, nested text config, and original raw-config fallback |
| Pooling metadata | Source sentence-transformer mean/CLS/last-token precedence and embedding configuration |
| Vocabulary | SentencePiece or the executable reviewed HF BPE pipeline; native Qwen fast-class defaults and generic controls |
| Publication | Retained source handles and a fully written, synced temporary output before destination replacement |

Qwen Q/K matrices retain their native row order. The Llama Q/K row permutation and generated Llama 3 `rope_freqs.weight` tensor apply only to the Llama family. Writing Qwen RoPE or pooling metadata does not establish runtime execution of every such model variant.

Configuration handling preserves the source distinction between successfully resolved Transformers configuration and raw JSON fallback. In particular, the Qwen2 absent sliding-window default and active-window layer thresholds affect whether invalid arguments cause fallback. The edge corpus includes absent, null, disabled and zero windows; invalid, Boolean and fractional thresholds; explicit layer-type lists and dictionary-key iteration. The Qwen3 default head width is taken from the pinned class rather than inferred unconditionally from hidden size.

The precise shared input, allocation and publication contract is in [the converter README](../crates/convert/README.md). Per-tensor work has checked size arithmetic and fallible reservation. These limits are not a process RSS cap or a power-loss durability guarantee for the output directory.

## Tokenizer compatibility

The actual official checkpoints require the exact Qwen Split expression and NFC. The helper executes the captured expression's ordered alternatives rather than borrowing another tokenizer's profile.

Tokenizers 0.22.2 pins `unicode-normalization-alignments 0.1.12`, whose tables are Unicode 9.0.0. The converter keeps those tables and the relevant licenses. Reusing the newer llama runtime tables changes combining classes for 120 later characters. Raw and normalized added-token matching retain their separate native phases.

Tokenizer class selection follows the captured precedence. A non-null class in `tokenizer_config.json` is used directly. An absent or null sidecar class requires successful resolution of the root model configuration, then its root tokenizer class or the pinned class mapping. Nested `text_config` does not override that root selection. An explicit null `model_type` is distinct from an absent key; the latter can trigger the original full-path pattern fallback. Unreviewed root `auto_map` and `configuration_files` indirection remains unsupported when class inference needs it.

Generic fast-class controls apply to Qwen: direct ByteLevel prefix configuration, split-special tokens and cleanup. For Qwen, GPT2 and generic Fast classes, an absent prefix setting—including an absent sidecar—defaults to false, booleans override it, and null/non-boolean values preserve the saved stage after the original caught rebuild failure. Llama preserves its saved prefix for absent/null settings; an explicit non-null setting selects slow SentencePiece reconstruction and is outside this path. Sequence pretokenizers, including a single or nested single child, preserve their child configuration. Qwen's special tokens remain distinct from Llama's reconstructed BOS/EOS template.

Three independent native matrices verify this preparation through the actual converter backend: 106 class-selection cases (82 accepted, 24 rejected; 5,412 encodings per profile), 48 direct-prefix cases (40 accepted, eight rejected; 160 encodings), and 34 missing-sidecar/Sequence cases (all accepted; 136 encodings). Their expected IDs and decoded bytes were captured from the original libraries; retained tokenizer inputs are hash-checked before replay.

[HF_QWEN_TOKENIZER_IMPLEMENTATION.md](HF_QWEN_TOKENIZER_IMPLEMENTATION.md) records the exact expression, normalization algorithm, allocation/work limits, exhaustive scalar references, Unicode conformance corpus, added-token cases and unchanged official tokenizer replays.

## Independent complete-file references

Original converter capture precedes Rust comparison. Each native profile runs the unchanged frozen Qwen model classes with the pinned Python packages, `PYTHONHASHSEED=0`, and one Torch/OMP/MKL worker. The source environment inspection also records package defaults; those defaults are not the capture's execution thread count.

| Ordinary cohort, per native profile | Cases | Exact successful output bytes |
| --- | ---: | ---: |
| Qwen2/Qwen3 conversion, seven output selections and scalar/bias/layout variants | 112: 104 accepted, 8 rejected | 25,723,552 |
| Configuration resolution and raw fallback edges | 32 accepted | 5,318,912 |
| Explicit, absent, null and unknown root model type | 8 accepted | 1,329,824 |
| Combined new complete-file corpus | 152: 144 accepted, 8 rejected | 32,372,288 |

These inputs are synthetic reference data. Binary packs deduplicate identical input/output blobs without transforming their bytes. The manifest records every blob's offset, length and SHA-256; both independently executed source profiles are retained.

The separate pretrained tests use **unmodified official checkpoint files**:

| Model | Pinned Hugging Face revision | Safetensors SHA-256 |
| --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |

Every source sidecar is also hash-checked. Three formats per model produce six complete files totaling **6,342,831,936 bytes per profile**. The original Windows and GNU captures produce the following identical complete-file identities:

| Model | Selection | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Qwen2.5-0.5B-Instruct | F16 | 994,157,120 | `f781a61a06b2dbafd1404187dc27a7981cb45a7bd66fa3268c3fa724b602f38e` |
| Qwen2.5-0.5B-Instruct | Q8_0 | 531,068,480 | `6138491dbbde6c58c3380ac997ec64b4d2fd2e721b02a011a6463a545ab5bfe5` |
| Qwen2.5-0.5B-Instruct | auto/BF16 | 994,157,120 | `6f8d546c52d0da75cc7c8c8f12f97f1494cff84e5afc49e20309a2cb2bcb70ec` |
| Qwen3-0.6B | F16 | 1,509,347,712 | `74878aba1d10648eae3d6cfef822f8c59ce3217b3808b35fce2dd02d975952a2` |
| Qwen3-0.6B | Q8_0 | 804,753,792 | `7e23819b480103d1308753410670c0e035804470452233f1497f76ed1303b92a` |
| Qwen3-0.6B | auto/BF16 | 1,509,347,712 | `d1d00d370d9f7d19673054e2860e62ee2897bb90eb1dd0cda72f54b519c24786` |

Pretrained weights and generated complete files stay outside Git. The committed reference manifest contains identities, provenance and source environments, not redistributed model tensors.

## Running the comparisons

Ordinary tests execute the bundled corpus:

~~~text
cargo test --offline --locked -p llama-convert
cargo test --release --offline --locked -p llama-convert
cargo clippy --offline --locked -p llama-convert --all-targets -- -D warnings
~~~

For the explicit complete pretrained comparisons, set `LLAMA_TEST_QWEN_HF_ROOT` to an external directory containing the two exact named checkpoint folders, and `LLAMA_TEST_QWEN_OUTPUT_ROOT` to a fresh external output directory:

~~~text
cargo test --release --offline --locked -p llama-convert --test qwen_pretrained -- --ignored --nocapture --test-threads=1
~~~

The tests check all source hashes, the predicted output length, the final file length and the complete SHA-256 for every selection. They refuse to reuse an existing model output subdirectory, preserving earlier attempts.

## Preserved findings and remaining scope

The retained development history distinguishes an initial reference-harness configuration error, the fixed family leak that generated Llama-only RoPE tensors for Qwen, the initially unimplemented Qwen tokenizer expression, and later infrastructure interruptions. Independent expected data was not replaced with Rust output and no numerical tolerance was widened. Native file hashes are the acceptance criterion.

HybridDNA tokenizers, InternS1 specialized vocabulary, reranker-head extraction, voyage-style `linear.weight` projection rewriting, mixture-of-experts, Qwen Next/recurrent/MTP, multimodal projectors, adapters, pickle input, remote executable tokenizer code, big-endian conversion and output sharding remain outside this dense path. Nested wrapper identity is preserved when checking specialized variants. These boundaries remain open work in the full port.

The source-specific records and final validation logs live in [the Qwen converter evidence](../evidence/2026-11-11/qwen-converter/README.md). This milestone does not close the complete-conversion, all-model, all-backend or full-release gates.
