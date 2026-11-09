# Latest T09–T12 progress — 8 November 2026

**Outcome:** five local Cargo packages build offline with the supplied Rust 1.99.1. All **123 tests pass**, as do the release build, all-target host check, formatting check and Clippy with warnings denied. This is a tested source checkpoint with explicit remaining requirements; it does not yet run a language model.

## Work delivered

| Track | Implemented checkpoint | Tests passed | Main remaining dependency |
|---|---|---:|---|
| T09 | Rust PNG/PNM/BMP and WAV subsets; source-based resize/normalize; FFT/mel/STFT/ISTFT; stream and typed-input primitives | 31 | All 58 model graphs, real embeddings/TTS, remaining codecs/video/filtered resampling and T01–T05 contracts |
| T10 | Sixteen scalar sampler configurations with RNG/state; 54 native chat formats; bounded character GBNF; exact integer JSON; speculative sampler transactions; six CLI commands | 40 | Runtime/tokenizer integration, full Jinja/schema/PEG/tool behavior, full CLI/updater and outstanding sampling/grammar paths |
| T11 | Real HTTP executable and provider interface; native/OpenAI text subsets; SSE/replay; auth/CORS/metrics/limits/cancellation; diagnostics page | 31 | Real T05 provider, full route/schema/default parity, TLS, tools/MCP, router, WASM/storage migration, native clients/connect |
| T12 | Complete baseline ownership/hashes; capability discovery; exact/numeric checks; Rust build/audit/fixture tasks; process probes; deterministic packages/integrity; handoff and release gates | 21 | Full semantic closure, model/device/platform CI, broad independent conformance, performance/memory/quality acceptance |

The T12 count is 18 unit tests plus three tests spanning media/behavior interfaces. T11 includes 17 real TCP integration tests. T10 includes four process-level CLI tests and five independent-reference test groups. Parameterized reference cases are described below; they are not added to the 123 test-function count.

## Independent original-source comparisons

- **Chat:** 540 exact UTF-8 transcripts across all 54 registered native names.
- **Sampling:** 256 original apply/accept steps across 16 configurations. Candidate IDs, selected indices and sorted flags match exactly; floating outputs use `max(3e-6, abs(reference) × 3e-6)`.
- **Grammar:** 232 exact original prefix/completion decisions; all eight bundled GBNF input examples compile.
- **PNG:** 16 inputs and **1,039,347 exact RGB bytes**, including the original source's 429 × 806 OCR image.
- **Resize:** 11 cases and 822 exact RGB bytes.
- **PNM/BMP and WAV:** 72 exact RGB bytes and 35 exact decoded waveform samples across the selected fixtures.
- **Audio/image math:** zero observed absolute differences on the selected normalization, FFT, Hann, mel, log-mel and ISTFT fixtures; the tests retain explicit numerical tolerances for other host evaluations.

The independent expected outputs came from actual original C/C++ implementations in an isolated reference build. The Rust product consumes fixed fixture data and never calls that original code. Detailed corpus identities and qualifications are in `docs/t09-status.md`, `docs/t10-status.md` and the fixture provenance files.

## Integration and robustness

Review and tests cover malformed/truncated media and decompression, byte/shape/compute limits, overflow, non-finite samples/logits, UTF-8 split boundaries, exact large JSON integer IDs, grammar state limits, rejected sampler operations, speculative-prefix RNG/history rollback, and audio streaming state preservation. The speculative snapshot budget is capped at 64 MiB and checked before cloning.

Server tests cover actual socket framing, native/OpenAI response shapes, SSE deltas/finish, exact replay suffixes, offset eviction, auth/CORS, malformed requests, connection/generation admission, slow readers, disconnect/cancellation, provider errors/panics, absolute request/generation deadlines, and shutdown while a client drip-feeds a request. The full browser/native application and stored-data migration are still open.

The built default server was separately probed over loopback: `/health` 503, `/v1/models` 200 with an empty list, `/` 200 diagnostics HTML, and a valid chat-generation request 503. These are the correct results without a model runtime. The process then exited cleanly after `quit`. Successful-generation transport tests use a test-only provider, clearly separated from the executable.

## Frozen source and language/dependency evidence

- Original source: **4,082 files**, **183,804,212 bytes**, with exactly the twelve ownership totals in the supplied plan. Every file has a SHA-256 and proposed disposition.
- Ownership of the selected tracks: T09 **108**, T10 **378**, T11 **1,106**, T12 **447** original files. These counts represent scope, not files fully ported.
- Registry discovery includes 173 architecture enum entries (including UNKNOWN), 176 model source files, 58 multimodal builders, 36 active dtypes, 102 operations, 22 unary variants, seven GLU variants and 18 backend families.
- Additional discovery records CLI flag literals, route registrations, native template names, build options, test registrations and public-function candidates with source locations. Discovery is lexical; full semantic closure remains open.
- Delivered implementation: **35 Rust files / 14,707 physical Rust lines**, including tests and comments. Non-Rust items are explicitly classified data, documentation, fixtures, notices and declarative configuration.
- Cargo graph: five local workspace packages, no external Cargo packages and no native build scripts. Executable dynamic dependencies observed are system `libc`, `libgcc_s`, and for the behavior executable `libm`; no original llama/ggml, C++ runtime, Python, Node or FFmpeg runtime is linked.

## Reproduce and continue

See `README.md` for build, CLI, server and integrity commands. `evidence/validation/` contains exact executed commands/results, the default server transcript and ELF dynamic-link inventories. `release/gates.tsv` prevents this partial checkpoint from being represented as a completed all-runtime rewrite. The release checker is expected to return a failure while mandatory gates remain open.

An independently extracted candidate ZIP passed its CRC checks and complete file manifest, then passed all **123 tests** and the release build using a fresh Cargo build cache with offline, locked dependencies. The final snapshot adds these reports while preserving the exact 44-file implementation/configuration identity recorded in `release/implementation-identity.json`; `evidence/validation/clean-extracted-results.json` records the archived-source checks.

The ZIP contains all current source, tests, independent fixtures, the original plan, ledgers, notices and handoff documentation. It excludes toolchain archives/installations, original foreign-language reference code/binaries and build caches. `PACKAGE_MANIFEST.tsv` covers the saved file contents.

**Next concrete integration:** supply the actual T01–T08 Rust packages/contracts, connect a real tokenizer/model provider, run a fixed-weight greedy text path and real modality embeddings, then expand the remaining feature and target ledgers without dropping any original requirements.
