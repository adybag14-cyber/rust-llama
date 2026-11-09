# rust-llama

A native Rust implementation of the functionality in the supplied `llama.cpp-master.zip`.

**This is an active, incomplete port.** The repository builds and runs real pretrained Llama generation and BERT embedding models on its Rust CPU runtime, including packed Q4 inference. The full frozen-source contract still includes all model architectures, tensor operations, backends, converters, multimodal models, tools, browser features and native clients. Those requirements remain tracked; this checkpoint is not a complete replacement for llama.cpp.

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

The high-level generation loader supports dense `llama` models with the Llama SPM vocabulary, full adjacent-pair RoPE without scaling, and F32 KV storage. Matrix arithmetic implements all 29 weight types in the frozen scalar CPU dot contract, including the K, IQ, ternary and FP4 formats; the [quantization documentation](crates/quant/README.md) lists the precise formats. Unsupported architecture features return an error. A decoder or kernel alone does not establish all-model quality or performance parity.

The runtime retains decoded F32 weights for row access and original packed bytes for native quantized matrix arithmetic. The default decoded-weight budget is 2 GiB; packed storage and runtime/cache allocations are additional. The default context bound is 1,024 tokens and is capped to the model's training limit. Generation is currently CPU-based; the separately verified CUDA and OpenCL kernel APIs are not yet a complete GPU model backend.

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

## Serve completions

```sh
cargo run --release --offline --locked -p llama-server -- \
  --model /path/to/model.gguf --bind 127.0.0.1:8080
```

The server connects native/OpenAI text completions and SSE to real Rust inference, with bounded HTTP parsing, cancellation, stop strings and replay. Text chat requires a supported model template. Unsupported request options fail explicitly. Without `--model`, the executable is a diagnostics service and reports inference unavailable.

See the [server behavior and supported options](crates/llama-server/README.md). This is not full upstream server/browser parity.

## What is implemented

| Area | Executable implementation |
| --- | --- |
| Quantization | All 36 active dtype decoders; 23 verified codec encoders (including scalar conversions); checked integer setters; source-faithful activation conversion and all 29 scalar CPU weight dots. |
| Tensors | Checked four-dimensional layouts, immutable shared views, a CPU DAG evaluator, and independently compared scalar operators. |
| Formats and vocabulary | Bounded GGUF v2/v3 reader, little-endian writer, scalar endian conversion, Llama SPM, BERT WordPiece, and exhaustive frozen Unicode tables/normalization/UTF-8 behavior. |
| Model runtime | Llama, Mamba and BERT Rust model graphs from typed weights; contexts, KV/recurrent state, transactions, adapters and state serialization. The GGUF facade loads dense Llama and dense BERT. |
| Inference and applications | Pretrained GGUF generation, CLI, real-model HTTP/SSE server, and a growing C API compatibility surface. |
| Device foundations | Rust CUDA driver execution and generated PTX kernels; generated OpenCL kernels and execution; typed kernel IR and other backend host foundations. |
| Other imported functionality | Selected native media processing, samplers, native chat formatters, grammar parsing, RPC/CBLAS/DSP utilities and Rust audit/packaging tools. |

Detailed support boundaries are in the crate READMEs and [PORT_STATUS.md](docs/PORT_STATUS.md). Numeric registries, backend probes and example programs are not counted as complete implementations.

## Independent comparisons

Reference C/C++ is built and executed outside this product. Committed fixtures contain data and provenance; runtime crates do not call the reference implementation.

The recorded 9 November 2026 evidence includes:

- **Pretrained F32 inference:** six prompts, 96 matching greedy token IDs and 62,976 logits, now bit-identical on the frozen Windows comparison after matching source scalar accumulation.
- **Pretrained Q4_0 inference:** 640,000 logits across 20 rows matched the frozen Windows reference exactly after preserving the source's Q8 activation quantization. Sixteen greedy token IDs matched. The earlier failed all-F32 comparison remains recorded.
- **BERT embeddings:** 13,056 token/pooled values exactly match the frozen Windows reference. All 122 captured stages of the diagnostic sequence match exactly; 46,080 independent attention outputs remain in the ordinary regression suite.
- **Vocabulary behavior:** 68 original SPM cases, 1,728 WordPiece cases across nine metadata variants, every one of the 30,522 BERT vocabulary entries, and 72 native special-token fallback cases.
- **Unicode behavior:** all 1,114,112 codepoints, 15,957 normalization sequences and 65,809 valid/malformed UTF-8 cases checked against the original source.
- **Quant codecs and dots:** all 36 dtype layouts, exhaustive F16/BF16/F8 bit patterns, byte comparisons for 23 encoders, 525 cases for the ten newest formats, 12,329 checked integer inputs, and 6,168 exact F16 matmul outputs.
- **Tensor graphs:** 65 independently executed frozen GGML graph cases.
- **Actual devices:** 138 CUDA dispatches on RTX 4090 with zero recorded error; OpenCL kernel cases on RTX 4090 and AMD gfx1036 with recorded numerical bounds.
- **C ABI:** 137 of 255 frozen exports, including model/context/vocabulary calls and sampler/chain ownership, are exercised by independent frozen-header harnesses. The new sampler/WPM harness compares 122,388 floats and 192 generated tokens exactly in each profile; the original core harness now matches all 93,248 values exactly. Exact layouts, seeded sampling, candidate ordering, clone/reset/free behavior and real-context generation are checked; the [coverage inventory](crates/ffi/COVERAGE.md) lists remaining exports.
- **Server:** 38 ordinary tests and 37 additional requests against the real pretrained model, including streaming, stop strings, cancellation, concurrency and replay.

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

Windows and Linux CI run these checks and separately download hash-pinned F32, Q4 and BERT model inputs for explicit numerical comparisons. Hardware/vendor tests retain their own explicit prerequisites. Windows records exact equality for the exercised model cases; Linux remains within the predefined portable limits, with Q4 also exact. See [fixtures/inference/README.md](fixtures/inference/README.md), [quant validation](crates/quant/README.md), [tensor validation](crates/tensor/README.md) and [xtask commands](xtask/README.md).

With the frozen source extracted separately:

```sh
cargo xtask verify-source /path/to/llama.cpp-master
```

`cargo xtask release-check` deliberately fails while any mandatory full-port gate remains open.

## Provenance and license

The integration reuses and audits the supplied T05–T08 and T09–T12 Rust progress archives. Their hashes and imported files are recorded in [docs/imports/manifest.json](docs/imports/manifest.json). Historical reports are preserved under `docs/imports/`; current claims require current evidence.

Original project code is MIT licensed. Original ggml attribution is retained in [LICENSE-UPSTREAM](LICENSE-UPSTREAM) and [NOTICE](NOTICE). Component notices identify separately licensed adapted code, including the native tied-sort helpers. Fixture provenance records additional source identities and qualifications.
