# GGUF inference facade

This crate connects checked GGUF loading, native tokenizers and the Rust model runtime. It has no dependency on an original llama/ggml binary and does not invoke a reference executable.

## Executable model scope

| GGUF architecture | Tokenizer | Current high-level execution |
| --- | --- | --- |
| `llama` | Llama SentencePiece | Dense autoregressive generation and complete per-token logits, standard adjacent-pair RoPE without scaling, F32 KV state. |
| `bert` | BERT WordPiece | Dense bidirectional encoding, complete token embeddings and none/mean/first/last pooling. |

Typed Mamba execution exists in `llama-model-runtime`; this facade does not yet load Mamba GGUF. Other architectures, unsupported tensor arrangements and unimplemented options return explicit errors. The full upstream architecture and runtime contract remains open.

All 29 weight types supported by the frozen scalar CPU dot contract are accepted by the matrix loader. This establishes the available matrix arithmetic, not quality validation for every combination of architecture and quantization. The current pretrained corpus exercises F32 and Q4_0 Llama and F16 BERT.

## Loading and memory

`Model::load` uses a 2 GiB decoded F32 weight budget and a requested context of 1,024 tokens, capped to the training context. `Model::load_with_options` lets the caller set those limits.

Matrices retain decoded F32 values for row access. Supported non-F32 matrices also retain their packed representation so multiplication preserves the original activation conversion and scaling order. The decoded-weight budget does not include packed bytes, activations, KV state, tokenizer data or other allocations. Loading is into host memory; there is no mmap or GPU model placement claim.

The GGUF reader handles little- and big-endian scalar values. F16 and BF16 packed matrix words are converted to the Rust kernel's byte order; F8 requires no word swap. Other big-endian packed arithmetic remains unsupported.

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
cargo run --release --offline --locked -p rust-llama-cli -- embed --model /path/to/bert.gguf --prompt "A sentence to encode." --pooling model
cargo run --release --offline --locked -p rust-llama-cli -- embed --model /path/to/bert.gguf --tokens 101,1037,6251,102 --pooling mean
```

The embedding JSON contains model identity, input token IDs, dimensions, pooling mode, every token vector and the optional pooled vector.

## Independent verification

The frozen source archive is `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. Reference C/C++ builds and pretrained weights remain outside the product repository. Expected values in `fixtures/inference` were captured from that original executable before comparison with Rust.

The Windows scalar checks record:

- Six F32 Llama prompts: 62,976 logits and 96 greedy token IDs, now bit-identical after matching the original F32-product/F64-sum reduction.
- Q4_0 Llama: 640,000 logits and 16 greedy token IDs, bit-identical with the original packed activation arithmetic.
- F16 BERT: three prompts, 25 token vectors and nine pooled vectors, totaling 13,056 values, bit-identical after the attention and pooling corrections.
- BERT diagnosis: all 122 captured stages of the first sequence match exactly; a committed independent attention fixture checks 46,080 output values across all 12 layers.

Portable model tests retain the declared tolerance `2e-4 + 2e-5 * abs(reference)`; exact equality is an observed result for the tested build. This is not a promise about every host math library, instruction set or untested model.

The big-endian BERT regression compares an equivalent independently downloaded encoding with the same little-endian native outputs. The frozen C reader itself rejects that opposite-endian GGUF file, so this check is explicitly a Rust reader capability rather than a claim that the original reader loads it.

Set `LLAMA_TEST_MODEL`, `LLAMA_TEST_Q4_MODEL`, `LLAMA_TEST_BERT_MODEL` and `LLAMA_TEST_BERT_BE_MODEL` to the pinned files, then run:

```sh
cargo test --release --offline --locked -p llama-inference --tests -- --ignored --nocapture
```

Each test checks the model's full size and SHA-256 before loading. CI downloads these explicit test inputs and runs the comparisons separately from ordinary offline workspace tests.
