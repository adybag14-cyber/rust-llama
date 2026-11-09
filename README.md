# rust-llama

A native Rust implementation of the functionality in the supplied `llama.cpp-master.zip`.

**This is an active, incomplete port.** The repository builds and runs real pretrained Llama, Qwen2, Qwen3 and Mamba-1 generation plus BERT embedding models on its Rust CPU runtime, including packed Q4 inference. The full frozen-source contract still includes all model architectures, tensor operations, backends, converters, multimodal models, tools, browser features and native clients. Those requirements remain tracked; this checkpoint is not a complete replacement for llama.cpp.

The authoritative source input is the 41,458,461-byte ZIP with SHA-256:

```text
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122
```

Its 4,082 regular files have a complete ownership and hash inventory in [ledger/baseline-files.tsv](ledger/baseline-files.tsv). A path assignment establishes responsibility, not implementation parity. See [current implementation status](docs/PORT_STATUS.md), the [12-track contract](docs/plan.md), and [mandatory release gates](release/gates.tsv).

## Build

Rust 1.99.1 is pinned in `rust-toolchain.toml`. The workspace uses local Cargo path dependencies and no downloaded Cargo packages.

```sh
git clone https://github.com/adybag14-cyber/rust-llama.git
cd rust-llama
rustup toolchain install 1.99.1 --profile minimal --component rustfmt,clippy
cargo build --workspace --release --offline --locked
```

The compiler, platform linker and standard library must already be installed for an offline build. Windows builds use the MSVC toolchain; Linux builds use the native Linux toolchain. No original llama/ggml library, C/C++ project build, Python or Node runtime is part of CPU inference.

## Run a model

Provide a supported GGUF model:

```sh
cargo run --release --offline --locked -p rust-llama-cli -- generate \
  --model /path/to/model.gguf --prompt "Once upon a time" --n-predict 32
```

On PowerShell, put the command on one line. Generated text is written to stdout; token counts and timing are written to stderr.

The high-level generation loader supports dense `llama` models with native SPM or BPE tokenization, full adjacent-pair RoPE without scaling, and F32 KV storage. Dense `qwen2` models add half-pair NeoX RoPE, separate or fused Q/K/V projections, source bias handling, and tied or separate output weights. Dense `qwen3` adds independent attention head width and learned per-head Q/K RMS normalization before full-head NeoX RoPE. It also loads Mamba-1 GGUF models with causal convolution, selective recurrent state, optional dt/B/C RMS normalization and tied or separate output weights. Matrix arithmetic implements all 29 weight types in the frozen scalar CPU dot contract, including the K, IQ, ternary and FP4 formats; the [quantization documentation](crates/quant/README.md) lists the precise formats. Unsupported architecture features return an error. A decoder or kernel alone does not establish all-model quality or performance parity.

The runtime retains decoded F32 weights for row access and original packed bytes for native quantized matrix arithmetic. The default decoded-weight budget is 2 GiB; packed storage and runtime/cache allocations are additional. The default context bound is 1,024 tokens and is capped to the model's training limit. Both applications accept the first file of a supported GGUF shard set and automatically discover its declared siblings. The Rust API also accepts explicit ordered paths with cumulative parser limits. Both applications accept `--ctx-size` and `--max-weight-mib`; the pinned Qwen2.5-0.5B Q4_K_M and Qwen3-0.6B Q8_0 corpora need `--max-weight-mib 3072` because its decoded F32 row cache exceeds 2 GiB. These are explicit allocation bounds, not total process memory limits. Generation is currently CPU-based; the separately verified CUDA and OpenCL kernel APIs are not yet a complete GPU model backend.

Other CLI commands expose validated GGUF metadata, tokenization, detokenization and per-token logits:

```sh
cargo run --release --offline --locked -p rust-llama-cli -- --help
cargo run --release --offline --locked -p rust-llama-cli -- inspect --model /path/to/model.gguf
cargo run --release --offline --locked -p rust-llama-cli -- tokenize --model /path/to/model.gguf --prompt "Hello"
```

## Encode a BERT sequence

```sh
cargo run --release --offline --locked -p rust-llama-cli -- embed \
  --model /path/to/bert.gguf --prompt "A sentence to encode." --pooling model
```

Dense `bert` GGUF models use the native WordPiece tokenizer and bidirectional encoder. The JSON result includes input token IDs, every raw token embedding and the selected pooled vector. Pooling supports the model default, none, mean, first and last. Vectors are not L2-normalized. Classifier/reranker tensors and other BERT-derived architectures remain explicit unsupported cases.

See the [inference API and model boundaries](crates/inference/README.md).

## Serve completions and embeddings

```sh
cargo run --release --offline --locked -p llama-server -- \
  --model /path/to/model.gguf --bind 127.0.0.1:8080
```

The server connects native/OpenAI text completions and SSE to real Rust inference, with bounded HTTP parsing, cancellation, stop strings and replay. Text chat requires a supported model template. Unsupported request options fail explicitly. Without `--model`, the executable is a diagnostics service and reports inference unavailable.

To serve a supported BERT model's embeddings:

~~~sh
cargo run --release --offline --locked -p llama-server -- \
  --model /path/to/bert.gguf --embeddings --pooling model
~~~

The native /embedding and /embeddings routes preserve pooled or per-token
vectors; /v1/embeddings provides OpenAI shapes, token usage and optional base64
for pooled output. Pooled results use Euclidean normalization by default.
Text, token, mixed and batched inputs follow the frozen server's rules.

See the [server behavior and supported options](crates/llama-server/README.md). This is not full upstream server/browser parity.

## What is implemented

| Area | Executable implementation |
| --- | --- |
| Quantization | All 36 active dtype decoders; all 29 source model quantizers including required/optional importance weights; 32 non-integer codecs including activation and F64 helpers; checked integer setters; all 29 scalar CPU weight dots. |
| Tensors | Checked four-dimensional layouts, immutable shared views, a CPU DAG evaluator, 75 executable operation IDs plus NONE input leaves, all scalar unary/GLU formulas for supported typed families, and 3,240 distinct native graph cases, including convolution, pooling and resize families. |
| Formats and vocabulary | Bounded GGUF v2/v3 reader and owned shard sets, little-endian writer, scalar endian conversion, Llama SPM, BERT WordPiece, BPE with all 64 reachable metadata-selected profiles / 99 aliases, Unigram, RWKV, PLaMo2/PLaMo3, explicit no-vocabulary mode, and exhaustive frozen Unicode/UTF-8 behavior. |
| Model runtime | Llama, Qwen2, Qwen3, Mamba and BERT Rust model graphs from typed weights; contexts, KV/recurrent state, transactions, adapters and state serialization. The GGUF facade loads dense Llama, Qwen2, Qwen3, Mamba-1 and dense BERT. The distinct Llama NativeContext supports independently selected F32/F16 K/V for explicit nonflash, single-worker execution. Strict contexts account for owned state and retained checkpoints before mutation. |
| Inference and applications | Pretrained GGUF generation and BERT embeddings, CLI, real-model HTTP/SSE and embedding endpoints, and 184 implemented frozen C API exports. |
| Device foundations | Rust CUDA driver execution and generated PTX kernels; generated OpenCL kernels and execution; typed kernel IR and other backend host foundations. |
| Other imported functionality | Selected native media processing, samplers, chat formatters, separate raw-byte native GBNF and safe character grammar APIs, RPC/CBLAS/DSP utilities and Rust audit/packaging tools. |

Detailed support boundaries are in the crate READMEs and [PORT_STATUS.md](docs/PORT_STATUS.md). Numeric registries, backend probes and example programs are not counted as complete implementations.

## Independent comparisons

Reference C/C++ is built and executed outside this product. Committed fixtures contain data and provenance; runtime crates do not call the reference implementation.

The recorded 9 November 2026 evidence includes:

- **Pretrained F32 inference:** six prompts, 96 matching greedy token IDs and 62,976 compared logits. Matching source accumulation and physical attention-row layout gives bit-identical results against the corresponding Windows and GNU captures.
- **Pretrained Q4_0 inference:** 640,000 unique logits and sixteen greedy IDs match the corresponding original profiles. Packed Q8 activation conversion is retained; the earlier failed all-F32 comparison remains recorded.
- **Pretrained BPE/Q4_K_M inference:** 1,671,168 unique SmolLM2 logits and 24 generated IDs match the corresponding Windows and GNU profiles.
- **Pretrained Mamba-1 inference:** 1,709,520 unique logits and 24 generated IDs match the corresponding original profiles. Three synthetic recurrent models provide ordinary loader and state regressions. Earlier GNU drift remains in the historical evidence.
- **Pretrained Qwen2 inference:** 5,165,824 unique logits and 24 generated IDs match each original profile, with independently captured Q4_K_M weights. Twenty-one synthetic Qwen2 variants cover separate/fused projections, partial biases, output configurations and packed weights. Thirteen physical metadata variants preserve source resolution rules.
- **Pretrained Qwen3 inference:** 5,469,696 unique Q8_0 logits and 24 greedy IDs match each original profile. The ordinary dense/split/vector corpus compares 294,912 logits and 400 IDs per profile, including independent head widths, learned Q/K normalization and fused-bias selection. Active low-precision vector cases rejected by the original CPU graph return checked errors. See [the precise Qwen3 scope](docs/QWEN3_IMPLEMENTATION.md).
- **F16 native caches:** independent F32/F16 key and value storage preserves raw half words in memory, sequence state and sessions. The ordinary original-C corpus compares 258,048 exact logits and 1,048 state/session buffers per profile across all four type pairs; a separate external trained/tiny C cohort checks 491,520 exact logits. Flash, AUTO resolution and multiple workers remain separate work; original default parameters are unchanged and still rejected.
- **BERT embeddings:** 13,056 token/pooled values match the corresponding original profiles. The GNU correction preserves the source's four-lane variance reduction. All 122 captured stages of the Windows diagnostic sequence match exactly; 46,080 independent attention outputs remain in the ordinary suite.
- **Sharded loading:** 90 native model/index configurations, including 65 admitted sets and 25 rejections; 64 model configurations compare 464,272 logits/embedding values and 590 greedy IDs per native profile. First-file metadata, per-file tensor origins, cumulative resource limits and retained file handles are checked independently. Ordered streaming split/merge and checkpoint converters remain open.
- **Vocabulary behavior:** the preserved SPM/WordPiece corpora plus 1,499,449 complete BPE vocabulary records across all 14 original BPE models; 5,096 real-model token/piece/detokenization cases; 18,018 GPT-2 alias cases; 23,754 synthetic configuration cases; 22,518 raw-byte cases; 126,896 regex splits. All 64 reachable profiles and 99 aliases are covered on Windows and Linux. Source-inferred special IDs preserve the measured platform container ordering.
- **Remaining tokenizer families:** Unigram, RWKV and PLaMo2/PLaMo3 have 91,476 synthetic family cases, 346,112 original vocabulary records, 10,412 real-vocabulary tokenization cases and 4,067,668 exact normalizer outputs. Invalid source assertion/nonprogress inputs return bounded errors. The no-vocabulary mode is explicit.
- **Unicode behavior:** all 1,114,112 codepoints, 15,957 normalization sequences and 65,809 valid/malformed UTF-8 cases checked against the original source.
- **Quant codecs and dots:** all 36 dtype layouts, exhaustive F16/BF16/F8 patterns, all 29 model quantizers, additional activation/conversion codecs, and complete scalar weight-dot inventory. The weighted/IQ extension adds 3,362 exact original-C encoder rows and 640 native tied-sort cases. Source-undefined payload/access cases have explicit safe behavior documented in the quant crate.
- **Tensor graphs:** 3,240 distinct independently executed frozen GGML graphs. The new convolution/pooling/resize slice adds 981 distinct cases and 530,246 exact output bytes per platform, with exact shapes and strides. Twenty overlapping-gradient cases preserve the measured platform-specific NaN operand order. Indexing compares 599,152 bytes per platform and preserves 88 native MSVC/GNU sort-order differences. The scalar extension checks 256,233 exact outputs, 170,260 transcendental outputs and 96,910 independently rounded high-precision ERF values. A further 1,007 native row/activation/normalization graphs require 2,909,056 exact output bytes per host. Shared scalar/SSE2 row math and the audited system `erff` boundary preserve the exercised dynamic-UCRT/glibc profiles; other providers and remaining ISA policies stay unverified. Independent raw-word gates check 502,058 `erff` and 502,058 scalar quick-GELU inputs per host. Earlier failures and fixed conversion bounds are retained.
- **Actual devices:** 138 CUDA dispatches on RTX 4090 with zero recorded error; OpenCL kernel cases on RTX 4090 and AMD gfx1036 with recorded numerical bounds.
- **C ABI:** 184 of 255 frozen exports, plus four diagnostic extensions, are exercised through independent frozen-header and model harnesses. The latest CPU sampler corpus covers 165 scenarios and 42 real contexts / 585 generated tokens. The BPE/FIM/raw-byte harness compares 1,505,576 vocabulary records and 7,598 tokenization cases. The context/memory extension adds 80 scenarios, 400 checks, 61,440 continuation logits and 1,223,532 native state bytes. The new family getter corpus adds 347,066 vocabulary records and 27,508 tokenization cases; physical metadata adds 23,903 records and 480 outputs across 176 cases. Raw native state restore/reserialization is byte-exact on both platforms; generated Linux floats retain the predeclared numerical budget. Automatic shard C loading adds 96 cases per native profile, including 53 accepted and 43 rejected loads, aggregate model getters and 163,840 exact continuation logit words. Prior core/sampler/WPM fixtures remain passing. The [coverage inventory](crates/ffi/COVERAGE.md) retains all 71 absent exports and the remaining semantic restrictions.
- **Server:** 37 real-model generation requests plus 283 embedding HTTP transcripts per original build (256 tiny-model, 27 pretrained BGE). Both verified profiles require exact corresponding F32 bits and all seven base64 vectors. Direct normalization checks cover another 4,608 values. Real TCP lifecycle tests cover configured load bounds, limits, provider failures, disconnects, shutdown and complete deadline errors.

The original MSVC and GNU builds do not use identical row arithmetic: the verified GNU build enables source SSE2 exponentiation, reductions and variance, while the verified MSVC build uses scalar rows. Separate expected captures preserve those differences. Test expectations select their profile independently of runtime dispatch. The model runtime also preserves 256-column attention padding and physical cache holes. None of these results establishes every CPU instruction-set path.

Evidence and qualifications are under [evidence/2026-11-09](evidence/2026-11-09). These results establish the exercised configurations, not all hardware, models or performance parity. Pretrained weights are not distributed in this repository; the tokenizer-only GGUF is test data with provenance.

## Verify

```sh
cargo fmt --all -- --check
cargo test --workspace --all-targets --offline --locked
cargo test --workspace --all-targets --release --offline --locked
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo xtask audit
cargo xtask fixtures
```

Windows and Linux CI run these checks and separately download hash-pinned F32, Q4_0, BPE/Q4_K_M, Qwen2, Qwen3, Mamba-1 and BERT model inputs for explicit numerical comparisons. Hardware/vendor tests retain their own explicit prerequisites. The corresponding verified original profiles produce exact results for the exercised model cases; unchanged portable error budgets remain available for unverified targets. See [fixtures/inference/README.md](fixtures/inference/README.md), [quant validation](crates/quant/README.md), [tensor validation](crates/tensor/README.md) and [xtask commands](xtask/README.md).

With the frozen source extracted separately:

```sh
cargo xtask verify-source /path/to/llama.cpp-master
```

`cargo xtask release-check` deliberately fails while any mandatory full-port gate remains open.

## Provenance and license

The integration reuses and audits the supplied T05–T08 and T09–T12 Rust progress archives. Their hashes and imported files are recorded in [docs/imports/manifest.json](docs/imports/manifest.json). Historical reports are preserved under `docs/imports/`; current claims require current evidence.

Original project code is MIT licensed. Original ggml attribution is retained in [LICENSE-UPSTREAM](LICENSE-UPSTREAM) and [NOTICE](NOTICE). Component notices identify separately licensed adapted code, including the native tied-sort helpers. Fixture provenance records additional source identities and qualifications.
