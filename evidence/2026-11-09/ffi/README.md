# Native Rust C ABI verification — 2026-11-09

This evidence covers the new `llama-ffi` crate and its **partial** implementation
of the frozen llama.cpp public C ABI. Product code is Rust. The test harness
and original C++ build remain outside the repository.

## Results

| Check | Result |
|---|---|
| Windows x64 debug native harness | Passed |
| Windows x64 release native harness | Passed |
| Rust boundary/layout/allocation unit tests | 5 passed in debug; 5 passed in release |
| Strict Clippy, all crate targets | Passed with warnings denied |
| Original-header ABI layouts | 69 sizes, alignments and field offsets identical |
| Model/context default fields | All 17 model and 39 context fields identical |
| Vocabulary metadata | All 512 texts, scores, attributes, EOG and control flags identical |
| Token pieces | All 1,024 ordinary/special forms identical |
| Trained stories260K cases | Six prompts, 27 prompt tokens, 96 greedy tokens matched |
| Numerical comparisons per native build | 93,248 logits/hidden embedding values |
| Maximum absolute error | 0.000017642974853515625 |
| Numerical tolerance | 0.0002 absolute + 0.00002 relative |
| Independent interleaved sequences | Two, matching both fixture rows and freshly executed native outputs |
| Failed batch transaction | Retry after invalid second token reproduces original rows |
| Cancellation | Returns 2; retry proves cache state was not advanced |
| Sparse output indexing | Original-token and negative-output indices verified |
| Embedding input/output | Matches the original DLL |
| Ownership | Decode still works after owning model handle is freed |
| Vocabulary-only loading | Tokenization works; decoding context correctly rejected |
| Unsupported requests | F16 KV default, unknown flash enum, pooling and GPU requests rejected |
| Progress callback cancellation | Stops before model allocation |
| Frozen export coverage | 107 implemented of 255; 148 absent; three Rust extensions |
| Original library linkage | No llama/ggml DLL in the Rust product import tables |

`native-debug.json` and `native-release.json` contain the complete ordered
layout values, model metadata, numerical counters and acceptance flags.
The native harness compares whole vocabulary metadata and all expected rows,
rather than only checking generated text.

The stories260K model reports a 2,048-token training context, width 64, five
layers, eight query heads, four KV heads, 292,800 stored parameters and
1,171,200 tensor bytes. The test contexts use 512 tokens per sequence, F32
KV storage and one CPU thread. These model values were checked directly
against the frozen native DLL.

## Reproduction

Build the Rust crate in the repository:

```powershell
cargo test -p llama-ffi
cargo clippy -p llama-ffi --all-targets -- -D warnings
cargo test -p llama-ffi --release
cargo build -p llama-ffi --release
```

Build the separate native harness from
`C:/Users/adyba/rust-llama-reference/ffi-oracle`:

```powershell
cmake -S . -B build -G 'Visual Studio 18 2026' -A x64
cmake --build build --config Release --parallel 4
```

The native executable takes exactly five arguments: Rust DLL, original DLL,
GGUF model, committed fixture JSON and output report. The successful release
invocation was:

```powershell
.\build\Release\ffi-oracle.exe `
  C:\Users\adyba\rust-llama\target\release\llama_ffi.dll `
  C:\Users\adyba\rust-llama-reference\build\bin\Release\llama.dll `
  C:\Users\adyba\rust-llama-reference\models\stories260K.gguf `
  C:\Users\adyba\rust-llama\fixtures\inference\stories260k-reference.json `
  C:\Users\adyba\rust-llama\evidence\2026-11-09\ffi\native-release.json
```

The debug invocation substitutes `target/debug/llama_ffi.dll` and
`native-debug.json`. Rust toolchain: 1.99.1, MSVC x64 target. The external
harness uses MSVC 19.51.36260.0 and includes the original `llama.h`, the
original GGML headers and the reference tree's JSON header. It imports no
Rust definitions of the structs. It resolves both libraries dynamically and
compares their own default-return functions field by field.

`verification.json` records exact source/header/harness/executable/model/
fixture/product and log hashes, product DLL import tables and background job
IDs. `exports.json` and `exports.tsv` were derived from actual PE exports.
The original DLL exports and header declarations match exactly at 255 names.

## Scope and fixes established here

The native comparison exposed two source details that were corrected before
the final runs: out-of-range special IDs such as UINT32_MAX retain the original
default; and unknown Llama size descriptions use `?B`. Source inspection also
preserved the frozen `seperator_token_id` spelling, guessed ftype ordering and
256-sequence limit. Enum values are checked without constructing invalid Rust
enums, owned batch buffers use retained allocation dimensions, and all exports
contain Rust panics.

This evidence does not assert full ABI compatibility. State/session memory
formats, sampler and grammar exports, memory mutation APIs, additional tokenizer
families, architectures, GPU model offload, quantized KV caches, full platform
coverage and performance parity remain open. The accepted parameter subset
and pointer lifetime/concurrency requirements are documented in
`crates/ffi/README.md`; all missing symbols are named in
`crates/ffi/COVERAGE.md`.
