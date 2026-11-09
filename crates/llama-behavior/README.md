# llama-behavior — T10 progress checkpoint

A dependency-free Rust 2024 crate built with the uploaded Rust 1.99.1 toolchain. It provides real scalar sampling, native chat formatting, strict character and source-compatible byte GBNF recognition, strict JSON and sampler-side speculative transactions. It does not load weights, tokenize strings, execute a model or implement the complete llama command suite.

## Public integration surfaces

| Module | Primary API | Scope |
| --- | --- | --- |
| `sampling` | Candidates, Config, Sampler, Chain, Mt19937; NativeDry, infill_native | 16 safe scalar configurations plus explicit native compatibility operations for ordering, metadata, vocabulary-derived DRY and infill |
| `chat` | `Message`, `BUILTIN_TEMPLATES`, `detect_template`, `render` | All 54 registered native names and the source heuristic detection path; no full Jinja interpreter |
| `grammar` | `Grammar`, `Recognizer`, `Limits` | Character GBNF compilation, bounded incremental UTF-8 input, prefix/completion decisions and caller-provided token-piece filtering |
| native_grammar | NativeGrammar | Bounded source raw-byte GBNF, numeric/named/inverse token terminals, mask/accept state, EOG, clone and reset |
| native_regex | NativeRegex, NativeMatch | Bounded MSVC C-locale ECMAScript byte matching and complete capture offsets for lazy grammar triggers |
| `json` | `Value`, `parse`, `parse_with_limits`, `stringify` | Duplicate-key rejection, Unicode escapes/surrogates, exact i64/u64 integer identity and finite binary64 fractional/exponent numbers |
| `speculative` | `verify_greedy`, `Transaction` | Greedy prefix/bonus verification and bounded sampler-state rollback; no model-cache transaction |

All error-returning sampler application and grammar acceptance paths preserve the previous committed state. `Chain::sample` advances sampling state but requires separate `accept`; `sample_and_accept` performs both. A speculative transaction owns private clones and installs only the accepted prefix. Its conservative state-copy budget is at most 64 MiB and is checked before cloning; candidate arrays and sampler scratch have separate bounds. These scalar copies are a correctness checkpoint, not a performance-accepted inference implementation.

T11 consumes the shared JSON module by Rust path dependency. The separate
llama-ffi crate exports the implemented frozen CPU sampler ABI over this crate
and the real tokenizer/model context. See crates/ffi/SAMPLERS.md for ownership,
pointer contracts, source-specific semantics, checked restrictions and exact
native validation. This behavior crate itself remains safe Rust with no native
implementation linkage.

The safe sampling and strict character Grammar APIs retain their documented
validation and portable semantics. apply_native/clone_native, NativeDry,
infill_native, NativeGrammar and NativeRegex are explicit compatibility paths;
source candidate metadata, raw bytes, token terminals, platform sort/RNG details
and adaptive clone behavior do not silently change the safe API contract.

## CLI

From the workspace root, with Rust 1.99.1 on `PATH`:

```text
cargo run --offline -p llama-behavior --bin llama-behavior -- templates
cargo run --offline -p llama-behavior --bin llama-behavior -- chat chatml --input fixtures/t10/chat-request.json
cargo run --offline -p llama-behavior --bin llama-behavior -- sample fixtures/t10/sample-request.json
cargo run --offline -p llama-behavior --bin llama-behavior -- grammar fixtures/t10/grammars/json.gbnf fixtures/t10/sample-request.json
cargo run --offline -p llama-behavior --bin llama-behavior -- speculative fixtures/t10/speculative-request.json
```

`sample` is a diagnostic over supplied logits and an explicit sampler chain. It returns actual transformed candidates and the terminal sampler's selected token. It is not model generation. `chat` writes exact formatter bytes without an extra newline. Commands accept stdin where documented by `--help`. Exit code 0 means success, 1 means a grammar mismatch, and 2 means invalid input or an unsupported operation.

## Evidence and limitations

Run `cargo test --offline -p llama-behavior`. The reference tests check 540 original native chat transcripts, 256 original sampler apply/accept steps and 232 original grammar decisions. The native fixtures are independently generated from the uploaded source; see `fixtures/t10/PROVENANCE.md`. All eight original `.gbnf` input examples compile.

The committed native compatibility regressions additionally check 383,053
raw-byte grammar decisions and 12,349 regex compile/match/capture outcomes.
The FFI tests replay 165 native sampler scenarios and the external trained-model
harness matches 585 generated tokens across 42 new context cases. Focused Windows
debug/release runs pass 45 behavior unit tests and strict Clippy. These native
measurements, platform scope and resource bounds are recorded in
evidence/2026-11-09/sampler-cpu and fixtures/behavior/*-provenance.json.

The remaining obligations and intentional compatibility corrections are recorded in `docs/t10-status.md`. A passing selected corpus does not establish all-template/Jinja, all-vocabulary, device-sampler, full CLI or real-model parity.
