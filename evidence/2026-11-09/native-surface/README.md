# Native scalar, tokenizer, sampler and embedding checkpoint

This checkpoint extends the native Rust implementation. It is not a complete
replacement for the frozen llama.cpp snapshot. The full-release command still
returns exit 1 because mandatory architecture, backend, application, conversion,
performance and distribution gates remain open.

## Final integrated validation

The authoritative final results are [Windows](windows/revalidated/status.json)
and [Linux](linux/revalidated/status.json). Both ran the entire sequence afresh
after the grammar reconstruction described below.

| Gate | Windows x64 MSVC | Linux x64 |
| --- | ---: | ---: |
| Ordinary workspace debug tests | 360 passed, 14 ignored | 361 passed, 14 ignored |
| Ordinary workspace release tests | 360 passed, 14 ignored | 361 passed, 14 ignored |
| Formatting and strict all-target Clippy | Passed | Passed |
| Offline locked release build | Passed | Passed |
| Explicit pretrained model and HTTP tests | 5 passed | 5 passed |
| Standalone embedding CLI scenarios | 11 passed | 11 passed |
| Original generation HTTP regression | 37 requests passed | 37 requests passed |
| Declared fixture integrity | 150 of 150 files | 150 of 150 files |
| Complete frozen-source identity | All 4,082 files match | All 4,082 files match |
| Full-port release gate | Correctly blocked | Correctly blocked |

The 14 ordinary ignored tests are not counted as passing. Five model-dependent
tests were subsequently invoked explicitly with all four hash-pinned model
inputs. The tokenizer and ABI owners also ran their external original-vocabulary
and native-library comparison suites; their separate evidence is linked below.

Both platforms recorded the same 333 immutable execution-input files before and
after validation, SHA-256
5a253504d5a50463d5fd8cc2bef4c26633389e7e385b3141b7d1483deceeb718.
The fingerprint covers Rust sources, Cargo/compiler configuration, CI workflow,
and numerical/tokenizer fixture data and provenance. Mutable evidence and prose
are excluded. The parent commit was
24878b33de096cc6622711197379e12d781f10cb. Each status file records literal commands,
exit codes, timing, and its corresponding raw logs.

## Implemented and independently checked scope

- All 29 defined original scalar model quantizers, including the three IQ
  formats that require importance weights. This comprises 32 non-integer
  storage/activation codecs when Q8_1, Q8_K and F64 conversion are counted.
  There are also checked integer setters. The new independent C corpus contains
  3,362 encoder rows and 640 source-sort cases. SIMD, general CPU operations and
  full backend coverage remain separate requirements.
- All 64 BPE metadata-selected pretokenizer types and 99 aliases reachable in the
  snapshot. Fourteen original vocabularies supplied 1,499,449 full records.
  Native platform differences in regex classes and unordered vocabulary
  precedence are retained. See [tokenizer evidence](../tokenizer-bpe/).
- Native token-aware grammar, lazy grammar, DRY, adaptive-p, infill, and FIM ABI
  integration. The grammar corpus checks 383,053 masks. The CPU sampler corpus
  checks 165 scenarios and 585 generated IDs across 42 actual contexts. The full
  BPE/FIM ABI comparison covers 1,505,576 records and 7,598 tokenization cases.
  See [fresh sampler evidence](../sampler-cpu/revalidated/verification.json).
- Dense BERT embedding routes at /embedding, /embeddings and /v1/embeddings,
  with native pooling, normalization, base64, token/text input forms, bounded
  worker admission and lifecycle handling. See
  [embedding evidence](../server-embeddings/README.md).

The frozen public ABI inventory is 155 implemented exports out of 255, plus four
diagnostic extensions. The remaining 100 exports stay explicitly open. Synthetic
fixtures, individual kernels and tested CPU paths do not establish full model or
device coverage.

## Numerical results

Windows repeated the earlier exact comparisons for 62,976 stories260K logits,
640,000 stories15M Q4_0 logits and 13,056 BERT values per endian encoding. Linux
retained maximum errors of 5.722045898e-6 for stories260K, zero for the Q4_0 corpus
and 1.348853111e-4 for BERT. No existing raw-model tolerance was increased.

The original embedding HTTP corpus comprises 226 tiny-model requests, 30 numeric
coercion requests and 27 pretrained BGE requests. Windows output floats match
exactly. Linux preserves the original raw-encoder budget; native normalization
mode 0 scales coordinates to a maximum absolute magnitude of 32760, so its
separately recorded error propagation budget was calculated before the first
Rust comparison. Its maximum observed scaled difference is 0.75439453125.
The derivation and direct normalization checks are in the embedding evidence.

## Preserved report-writing failure and new grammar revision

The first attempt's status and raw release compilation log remain directly under
each platform directory. A source-hashing loop in an external evidence writer
reused the report-path variable. Its final JSON write consequently targeted
native_grammar.rs, so release compilation correctly failed at the JSON opening
brace. The other 12 recorded source hashes remained unchanged.

The exact prior grammar revision, SHA-256
ae2ce547e2dfa9d54c257a7c1ec8133411504565baa31291d1b62f753d701cd8,
was not recoverable. A complete earlier source was retained. Its final global
parser construction/copy-work and symbol-name byte-budget changes were
reconstructed, independently reviewed, and tested as a new revision:
fd035bd0e967056ad5def35c730ea9ad40f9727061a997c96a0bb8e5f5c3798c.

The new grammar passed the original comparison corpus, focused debug/release
tests and strict Clippy before replacement. The entire integrated sequence was
then rerun on both platforms. No earlier debug result was reused as proof of the
new source. The corrected report writer asserts an evidence-only destination
and checks all source hashes before and after its write. The
[recovery report](../sampler-cpu/revalidated/recovery-report.json) and
[independent review](../sampler-cpu/revalidated/independent-review.md) preserve
the old, corrupt, retained-base and new identities.

## Remaining scope

The complete contract remains in docs/plan.md and release/gates.tsv. Full tensor
operation coverage, instruction-set paths, all 172 named architecture entries,
all 18 backend families, conversion families, multimodal builders, full Jinja and
PEG behavior, browser/native applications, complete protocols and performance
validation are unfinished. This checkpoint does not waive any of those gates.
