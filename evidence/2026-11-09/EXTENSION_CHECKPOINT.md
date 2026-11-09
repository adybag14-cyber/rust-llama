# Verified extension checkpoint — 9 November 2026

This checkpoint extends the working Rust implementation with complete scalar CPU weight-dot coverage, additional codecs, native Unicode/WordPiece behavior, pretrained BERT embedding execution and the CPU sampler C ABI. **The full frozen-source port remains incomplete.** The active release gates retain the missing architectures, operations, backends, tools, multimodal and browser/client requirements.

The machine-readable record is `extension-summary.json`. The earlier `checkpoint-summary.json` and its logs remain a historical snapshot.

## Whole-workspace results

| Platform | Debug | Release | Strict Clippy | Release build |
| --- | ---: | ---: | --- | --- |
| Windows x64 / MSVC | 311 passed, 0 failed | 311 passed, 0 failed | Passed | Passed |
| Linux x64 / WSL Ubuntu | 313 passed, 0 failed | 313 passed, 0 failed | Passed | Passed |

Each ordinary profile reports 11 ignored tests. Four are pretrained-model checks that were explicitly executed separately on both platforms. Vendor/reference-service tests retain their own prerequisites. Platform-specific test selection explains the ordinary count difference; it is not a claim that every target executes identical tests.

Formatting, source/dependency audit and fixture integrity pass. There are 113 fixture files and 113 declarations across four manifests. The audit records unsafe review candidates without claiming they are all formally proven.

The first final audit correctly rejected three newly added standalone package license texts until their exact paths were classified. A fixture check also caught the intentionally updated inference README until its documented checksum was refreshed. Numeric expected fixtures were unchanged. Those initial findings remain in the logs. The whitespace check honors existing CRLF files while retaining actual trailing-space checks.

## Executable scope added

- All 29 weight types in the frozen scalar CPU dot contract now have native arithmetic, including K, IQ, ternary and FP4 formats.
- There are 23 verified codec encoders, including scalar serialization helpers, plus separate checked I8/I16/I32 setters. This count is not the number of upstream model-quantizer entry points. IQ encoders, weighted quantization and ISA implementations remain open at this checkpoint.
- WordPiece and the frozen Unicode properties, case tables, NFD/NFC, Hangul and UTF-8 behavior are implemented. BPE/regex dispatch and other tokenizer families remain separate work.
- The high-level loader and CLI execute dense BERT embeddings with no/mean/first/last pooling. Missing pooling metadata resolves to no pooling. Raw output is not L2-normalized.
- The C ABI implements 137 of 255 frozen exports, including 30 new sampler functions. Its 14 sampler constructors, owning chains, custom callbacks, seeded state and actual-context sampling are exercised against the original DLL. The 118 absent exports remain listed explicitly.
- CI now includes downloads and hash checks for the four concrete model files, then executes their release comparisons after ordinary offline checks.

## Independent numerical evidence

| Check | Windows result | Linux result |
| --- | --- | --- |
| Pretrained F32 Llama, 62,976 logits | Exact; 96 greedy IDs exact | Max absolute error 5.722045898e-6; IDs exact |
| Pretrained Q4 Llama, 640,000 logits | Exact; 16 greedy IDs exact | Exact; IDs exact |
| Pretrained F16 BERT, 13,056 values | Exact | Max absolute error 1.348853111e-4 |
| Equivalent big-endian BERT encoding | Exact against the same native corpus | Same bound as little endian |
| BERT attention regression, 46,080 values | Passed | Passed |
| Missing-pooling BERT fixture, 56 values | Exact | Passed ordinary regression |

The fixed model tolerance remains `2e-4 + 2e-5 * abs(reference)`. The BERT diagnosis compared 122 corresponding intermediate stages exactly after correcting F32-product/F64-sum attention and mean pooling. Before/after evidence is retained. The original C reader rejects the big-endian GGUF; the Rust check uses its equivalent little-endian native corpus.

Other independent evidence includes 1,114,112 Unicode codepoints, 15,957 normalization sequences, 65,809 UTF-8 cases, 1,728 WordPiece cases, all 30,522 vocabulary entries and 72 special-token fallback cases. Quantization adds 525 encoder cases, 12,329 integer inputs and 6,168 F16 graph outputs.

For each native Windows build profile, the sampler/WPM harness compares 122,388 floats with zero error, 1,401 candidate-array applications, 89 lifecycle/error checks and 192 generated tokens. The original C model/context harness again matches all 93,248 logit/embedding values exactly. These ABI comparisons are Windows/MSVC evidence; native ABI parity on other platforms remains a separate gate.

## Where to inspect the evidence

- `windows-extension/` and `linux-extension-final/`: complete workspace checks and final audit/build results.
- `linux-extension/`: explicit pretrained model comparisons.
- `bert/`: original numerical failure, per-stage diagnosis, final model tests, CLI and real HTTP behavior.
- `bert-defaults/`: missing optional pooling metadata.
- `tokenizer/`: native Unicode/WPM/SPM comparisons and fallback regressions.
- `sampler-ffi/`: native C layouts, exports, samplers, vocabulary and repeated core ABI proof.
- `crates/quant/tests/fixtures/`: independent codec, matvec and F16 graph data with provenance.

`cargo xtask release-check` still returns 1 because mandatory full-port gates remain open. A passing checkpoint does not mark the user's complete-port objective finished.
