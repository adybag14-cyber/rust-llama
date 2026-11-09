# llama.cpp Rust rewrite — T09–T12 progress checkpoint

**Prepared 8 November 2026. Status: buildable partial implementation, with explicit remaining gates.**

This workspace implements the latest progress on T09 (media), T10 (behavior and CLI), T11 (server transport), and T12 (verification and packaging) from the supplied twelve-track plan. All maintained implementation and tests in this package are Rust. It builds offline using the supplied Rust 1.99.1 toolchain and only local Cargo packages.

This checkpoint does **not** load a language model or perform model inference. T01–T08 implementations were not included in the supplied inputs. The server exposes an `InferenceProvider` interface and its default executable reports unavailable inference; test-only providers exercise transport behavior. Media preprocessing, sampler algorithms, native chat formatting and grammar recognition execute their real Rust implementations independently.

## Packages

| Package | Owner | Runnable or reusable functionality |
|---|---|---|
| `llama-media` | T09 | PNG/PNM/BMP and WAV subsets; image resizing/normalization; FFT/mel/STFT/ISTFT; bounded stream primitives and typed inputs |
| `llama-behavior` | T10 | 16 scalar sampler configurations; MT19937; 54 native chat formats; character GBNF; strict JSON; sampler-side speculation; diagnostic CLI |
| `llama-server` | T11 | Bounded HTTP/SSE transport and replay, provider contract, auth/CORS subset, metrics and Rust-rendered diagnostics |
| `llama-conformance` | T12 | Source inventory/drift checks, exact/numeric comparisons, SHA-256/CRC-32, bounded process probes, microbenchmarks, deterministic ZIPs |
| `xtask` | T12 | Offline build/check/test/audit/fixture/package commands and release gates |

Read `PROGRESS.md` for the final measured results and `docs/integration-handoff.md` for integration with other tracks. The original user plan is preserved in `docs/plan.md`.

## Build and test

Use the uploaded `rust-1.99.1-x86_64-unknown-linux-gnu.tar.xz` or an equivalent installed Rust 1.99.1 toolchain. Ensure its `bin` directory is on `PATH`. The uploaded distribution includes the Linux x86_64 GNU standard library; other targets are untested.

```text
rustc -vV
cargo test --workspace --offline --locked
cargo build --workspace --release --offline --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
```

There are no crates.io downloads, original llama/ggml libraries, Python, Node, FFmpeg or native codecs in the Rust product build. The usual system C linker and OS libraries used by the Rust standard library are external platform dependencies. `release/dependency-inventory.tsv` and `evidence/` record the host boundary and actual checks.

Validation passed **123 tests**, the release build, all-target check, formatting check and Clippy with warnings denied. The same implementation was also tested and built offline from an independently extracted candidate ZIP with a fresh build cache; its complete integrity manifest passed. See `evidence/validation/results.json` and `evidence/validation/clean-extracted-results.json`.

## Try the implemented CLI

These commands use included fixtures and require no model weights:

```text
cargo run --offline --locked -p llama-behavior -- templates
cargo run --offline --locked -p llama-behavior -- chat chatml --input fixtures/t10/chat-request.json
cargo run --offline --locked -p llama-behavior -- sample fixtures/t10/sample-request.json
cargo run --offline --locked -p llama-behavior -- grammar fixtures/t10/grammars/json.gbnf fixtures/t10/sample-request.json
cargo run --offline --locked -p llama-behavior -- speculative fixtures/t10/speculative-request.json
```

`sample` operates on supplied logits. `chat` renders the chosen native format; it does not run a full Jinja interpreter. GBNF recognition accepts real token pieces through its API, but obtaining those pieces from a model vocabulary requires T04.

## Run the server transport

```text
cargo run --offline --locked -p llama-server -- --bind 127.0.0.1:8080
```

The root URL serves a Rust-authored diagnostics page. `/health` returns HTTP 503 with the unavailable runtime status, `/v1/models` returns an empty list, and generation cannot start without a real embedded provider. Type `quit` on the process stdin to shut down. The executable does not ship a model or a synthetic inference mode.

A provider receives typed generation requests, emits valid UTF-8 text through a bounded sink, reports real usage and honors cancellation. The test suite connects a clearly test-only deterministic provider to verify network framing, authentication, completion responses, streams and failures. TLS, full protocol coverage, model routing, tools/MCP, browser/WASM, stored-data migration and native clients remain open.

## Verify source and fixtures

```text
cargo xtask audit
cargo xtask fixtures
cargo xtask verify-source /absolute/path/to/extracted/llama.cpp-master
cargo xtask verify-package
```

`verify-source` checks all 4,082 reference files against their recorded SHA-256 and ownership. It requires the independently extracted original archive. `verify-package` checks the downloaded ZIP's extracted content against its generated `PACKAGE_MANIFEST.tsv`; it is not applicable to a pre-packaging development tree that lacks that manifest. Build caches are excluded.

`ledger/baseline-files.tsv` gives every original file an owner and proposed disposition. Baseline rows remain planned until their complete behavior is reviewed. Domain ledgers report actual partial progress; a file count is never substituted for functionality completion. Registry extraction is lexical discovery with source locations, not a full semantic proof of every macro/dynamic registration.

## Package and resume

```text
cargo xtask release-check
cargo xtask package /absolute/path/to/new-checkpoint.zip
```

The release checker deliberately fails with named open gates. Packaging remains available for labelled progress checkpoints. It audits the source/dependency graph, refuses to overwrite a checkpoint, uses fixed archive metadata and deterministic ordering, and includes an integrity manifest. The original source tree, reference binaries, compiler archives and build outputs are kept outside the ZIP.

The fixture outputs were captured from the frozen original implementations. Their source/compiler identities and hashes are recorded in `docs/t09-oracle.json`, `fixtures/t10/provenance.json` and `evidence/reference/`. Normal Rust builds/tests consume the recorded data directly. Recreating those original oracle adapters is a separate audit task; full end-to-end oracle/build reproducibility is still an open T12 gate.

## Remaining work

- Actual tensor/model/tokenizer/runtime integration from T01–T08; real model weights and CPU/device inference verification.
- All 58 multimodal model builders, learned projector/audio execution, remaining codecs, filtered resampling, video pipelines and full speech generation.
- Full Jinja/JSON-schema/PEG/tool/reasoning behavior, token/lazy grammar integration, remaining sampler/backend details, full CLI/tools/examples and updater.
- Complete server protocol/default/route parity, model lifecycle, tools/MCP, browser storage migrations, WASM UI and Android/Apple clients/connect.
- Full baseline disposition closure, backend/platform CI, memory/fuzz/performance/quality acceptance and M7 release evidence.

These remain requirements in the original plan; this checkpoint does not waive them.
