# GGUF inference facade

This crate connects checked GGUF loading, native tokenizers and the Rust model runtime. It has no dependency on an original llama/ggml binary and does not invoke a reference executable.

## Executable model scope

| GGUF architecture | Tokenizer | Current high-level execution |
| --- | --- | --- |
| `llama` | Llama SentencePiece or native BPE | Dense autoregressive generation and complete per-token logits, standard adjacent-pair RoPE without scaling, F32 KV state. |
| `qwen2` | Native BPE | Dense generation and complete logits, half-pair NeoX RoPE without scaling, separate or fused Q/K/V with source bias rules, F32 KV state. |
| `qwen3` | Native BPE | Dense generation with independent full head width, learned per-head Q/K RMS, full-head NeoX RoPE without scaling, separate or fused Q/K/V and F32 KV state. |
| `bert` | BERT WordPiece | Dense bidirectional encoding, complete token embeddings and none/mean/first/last pooling. |
| `mamba` | Native tokenizer metadata; SPM and BPE independently exercised | Mamba-1 causal convolution and selective recurrence, tied or separate output weights, optional unweighted dt/B/C RMS normalization. |

Mamba-1 loading validates source tensor names and dimensions before constructing its owned recurrent state. Convolution and SSM A tensors require the source CPU kernel's F32 storage. The frozen loader does not load optional dt/B/C normalization weight tensors; its metadata flag selects unweighted normalization. Other architectures, unsupported tensor arrangements and unimplemented options return explicit errors. The full upstream architecture and runtime contract remains open.

All 29 weight types supported by the frozen scalar CPU dot contract are accepted by the matrix loader. This establishes the available matrix arithmetic, not quality validation for every combination of architecture and quantization. The pretrained corpus exercises F32 and Q4_0 Llama, BPE/Q4_K_M SmolLM2, Q4_K_M Qwen2, Q8_0 Qwen3, F16 Mamba-1 and F16 BERT.

Qwen3 validates independent query/KV widths, per-head Q/K RMS vectors and fused-bias precedence before execution. Active norms and biases require F32 storage in the original CPU graph; ignored partial fused-bias groups retain their original behavior. Its ordinary tests cover automatic and explicit shard sets, state identity and transaction rollback. [Detailed Qwen3 contracts](../../docs/QWEN3_IMPLEMENTATION.md) keep MoE/VL/Next, scaled/partial RoPE, dimension arrays, rerankers and C NativeContext execution explicitly open.

## Loading and memory

`Model::load` uses a 2 GiB decoded F32 weight budget and a requested context of 1,024 tokens, capped to the training context. `Model::load_with_options` lets the caller set those limits. The pinned Qwen2.5-0.5B Q4_K_M model has a separate output matrix and a decoded cache larger than 2 GiB; its comparisons use a 3 GiB weight budget and context 128. The pinned Qwen3-0.6B Q8_0 model needs 2,384,199,680 decoded-weight bytes and also uses an explicit 3 GiB budget; its default-budget rejection is tested.

The single-path APIs automatically discover declared GGUF siblings. The Rust
`Model::load_from_splits` and `load_from_splits_with_options` APIs take explicit
ordered paths; `Model::load_from_gguf` consumes a validated `GgufSet`. Primary
metadata is authoritative, tensor names are globally unique, and each tensor
retains its own file offset and byte order. Limits apply cumulatively across
the set. The decoded-weight allowance covers every tensor before decoding.
Missing or malformed siblings fail transactionally. The [GGUF documentation](../gguf/README.md)
describes filename rules, safe input bounds and the remaining converter/writer work.

Matrices retain decoded F32 values for row access. Supported non-F32 matrices also retain their packed representation so multiplication preserves the original activation conversion and scaling order. The decoded-weight budget does not include packed bytes, activations, KV state, tokenizer data or other allocations. Loading is into host memory; there is no mmap or GPU model placement claim.

The GGUF reader handles little- and big-endian scalar values. F16 and BF16 packed matrix words are converted to the Rust kernel's byte order; F8 requires no word swap. Other big-endian packed arithmetic remains unsupported.

The strict `llama_model_runtime::Context` separately defaults to a 1 GiB `max_state_bytes` budget. Checked reservations cover owned sequence/layer records, KV or recurrent float capacities, planning records, transactional state and retained checkpoints. `checkpoint()` and `ContextCheckpoint::try_clone()` are fallible. Reservations can outlive the context; restoring into another context charges the destination. The budget excludes model weights, execution scratch, returned logits/embeddings, caller buffers and allocator overhead. Serialized output is charged while being constructed, then becomes caller-owned. Native C-compatible `NativeContext` has a distinct resource contract.

## Generation

`generate` creates an independent context and sampler for each call. Its explicit options are a token budget, temperature, u32 seed and stop strings. Temperature zero selects greedy decoding. Output callbacks receive UTF-8 text, with incomplete byte sequences and potential stop-string suffixes withheld across token boundaries.

`logits_for_tokens` returns every vocabulary logit after each supplied token. It is intended for consumers that require raw logits and for independent conformance comparisons.

`runtime_model()` returns `Result<Arc<llama_model_runtime::Model>>`. It rejects an embedding-only executor. Call `supports_generation()` when selecting a task; successful GGUF loading alone does not imply generation capability.

## Dense BERT embeddings

`embed_tokens(tokens, pooling, cancel)` returns:

- `tokens`: one raw embedding vector per input token.
- `pooled`: no vector for `Pooling::None`, the first or last token for those modes, or the source mean-pooling operation.

`embed(text, pooling_override, cancel)` applies the model's WordPiece tokenizer and insertion rules before encoding. An omitted override uses the GGUF pooling metadata; absent metadata preserves the source default of no pooling. The BGE model in the corpus selects its first/CLS token.

Vectors are **not L2-normalized**. The CLI reports `normalized: false`. Consumers that require cosine-normalized vectors must perform that separate operation.

All tokens in a sequence can attend to all other tokens. Position indices default to their sequence indices, and token types follow the frozen BERT graph's row-zero rule. Requested context limits are checked without truncating the trained position-embedding table. Empty sequences, out-of-range token IDs, excessive context and cancellation return errors.

Classifier/reranker tensors, Nomic/Jina/other BERT-derived architectures, causal BERT variants, MoE and unimplemented graph features are rejected. BERT does not provide autoregressive logits or generation through this facade.

## CLI examples

```sh
cargo run --release --offline --locked -p rust-llama-cli -- generate --model /path/to/llama.gguf --prompt "Once upon a time" --n-predict 32
cargo run --release --offline --locked -p rust-llama-cli -- generate --model /path/to/qwen2.gguf --max-weight-mib 3072 --ctx-size 128 --prompt "Once upon a time" --n-predict 8
cargo run --release --offline --locked -p rust-llama-cli -- embed --model /path/to/bert.gguf --prompt "A sentence to encode." --pooling model
cargo run --release --offline --locked -p rust-llama-cli -- embed --model /path/to/bert.gguf --tokens 101,1037,6251,102 --pooling mean
```

The embedding JSON contains model identity, input token IDs, dimensions, pooling mode, every token vector and the optional pooled vector.

## Independent verification

The frozen source archive is `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. Reference C/C++ builds and pretrained weights remain outside the product repository. Expected values in `fixtures/inference` were captured from that original executable before comparison with Rust.

The checks against each matching original profile record:

- Six F32 Llama prompts: 62,976 compared logits and 96 greedy token IDs, bit-identical after preserving source accumulation and attention-row layout.
- Q4_0 Llama: 640,000 unique logits and 16 greedy token IDs, bit-identical with original packed activation arithmetic.
- BPE/Q4_K_M SmolLM2-135M: three prompts, 1,671,168 unique logits and 24 greedy token IDs, bit-identical on the corresponding Windows and GNU profiles.
- F16 Mamba-130M: three prompts, 1,709,520 unique logits and 24 greedy token IDs, bit-identical against both original profiles after matching whole-row SiLU.
- Qwen2.5-0.5B Q4_K_M: three prompts, 5,165,824 unique logits and 24 greedy IDs, bit-identical against each original profile. Twenty-one synthetic variants add 165,888 unique logits and 198 greedy IDs, with metadata, partial-bias and output-bias overflow regressions.
- Qwen3-0.6B Q8_0: three prompts, 5,469,696 unique logits and 24 greedy IDs, bit-identical against each original profile. Dense/split/vector regressions add 294,912 logits and 400 IDs per profile.
- F16 BERT: three prompts, 25 token vectors and nine pooled vectors, totaling 13,056 values, bit-identical after preserving attention, pooling and GNU variance order.
- BERT diagnosis: all 122 captured stages of the first sequence match exactly; a committed independent attention fixture checks 46,080 output values across all 12 layers.

The ordinary shard regression executes 64 model configurations drawn from 90
native admission cases, comparing 464,272 model/embedding values and 590 greedy
IDs per profile. It includes metadata-only first shards and both automatic and
explicit path loading. The two verified x86_64 MSVC/GNU targets require exact
F32 bits; other targets retain the predeclared portability criterion.

The original builds have distinct math profiles: scalar rows on verified MSVC and source SSE2 rows on verified GNU. Captures preserve both profiles; tests choose expected data from target configuration independently of the Rust runtime's arithmetic dispatch. Attention retains masked physical columns and the source's padded row width. A wrong dispatch therefore cannot silently choose a matching wrong expectation.

Portable model tests retain the declared tolerance `2e-4 + 2e-5 * abs(reference)`; exact equality is an observed result for the tested build. This is not a promise about every host math library, instruction set or untested model.

The big-endian BERT regression compares an equivalent independently downloaded encoding with the same little-endian native outputs. The frozen C reader itself rejects that opposite-endian GGUF file, so this check is explicitly a Rust reader capability rather than a claim that the original reader loads it.

Set `LLAMA_TEST_MODEL`, `LLAMA_TEST_Q4_MODEL`, `LLAMA_TEST_BPE_MODEL`, `LLAMA_TEST_MAMBA_MODEL`, `LLAMA_TEST_QWEN2_MODEL`, `LLAMA_TEST_QWEN3_MODEL`, `LLAMA_TEST_BERT_MODEL` and `LLAMA_TEST_BERT_BE_MODEL` to the pinned files, then run:

```sh
cargo test --release --offline --locked -p llama-inference --tests -- --ignored --nocapture
```

Each test checks the model's full size and SHA-256 before loading. CI downloads these explicit test inputs and runs the comparisons separately from ordinary offline workspace tests.
