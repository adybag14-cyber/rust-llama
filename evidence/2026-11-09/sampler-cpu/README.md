# CPU samplers, byte grammar and vocabulary C ABI evidence

The authoritative current results are under revalidated/. verification.json and
exports.json now describe the freshly tested new grammar revision. Earlier raw
logs and verification-before-report-error.json remain preserved; the report-path
incident and new-revision validation are described below.

This checkpoint adds six frozen CPU sampler constructors and twelve FIM getter
exports/aliases. Actual PE inspection reports **155 of 255 frozen exports**, with
**100 still absent** and four Rust diagnostic/layout extensions. The frozen
header, Rust DLL, native DLL and export inventory are hashed in exports.json and
fixture provenance. The Rust import table contains only system/runtime libraries;
there is no original llama or ggml implementation linkage.

The product source is Rust. The C++ programs below include the unchanged frozen
header and load original and Rust libraries as independent test targets. They
live outside the repository.

## Observed results

All checks below passed in Windows x64 debug and release unless qualified.

| Check | Result |
|---|---|
| New grammar/lazy wrapper corpus | 34 scenarios, 339 candidate applications, 173,568 exact token-mask decisions |
| DRY vocabulary preprocessing/history | 55 scenarios, 699 applications, 357,888 candidate rows; exact float bits and metadata |
| Adaptive-p | 35 scenarios, 1,925 applications; exact seeded selection, probabilities, clone/copy/reset and acceptance behavior |
| Infill | 41 scenarios, 164 applications; exact prefix merges, EOG/fallback, duplicate IDs and candidate metadata |
| Actual trained model with seven new chain configurations | Six prompts each, 42 contexts, 585 exact generated token IDs |
| Independent raw grammar engine fixture | 82 parser cases, 383,053 raw-piece decisions, 178 accepted transitions and two rejected accepts; separate grammar provenance |
| Native byte regex fixture | 137 patterns, 12,349 compile/search/full-match/capture checks |
| Preserved previous sampler harness | 1,401 applications, 45,949 candidate rows, 122,388 float comparisons, 192 generated tokens; maximum absolute error 0 |
| Preserved previous model/context ABI harness | 69 sizes/alignments/offsets, all default fields, 93,248 logits/embedding values; maximum absolute error 0 |
| C BPE/FIM vocabulary harness | All 14 original BPE vocabularies plus the synthetic raw fixture: 1,505,576 entries, 5,096 text cases, 2,502 raw-byte cases; exact outputs |
| Focused Rust unit tests | 45 llama-behavior and 17 llama-ffi tests in each profile |
| Strict all-targets Clippy | Passed for llama-behavior and llama-ffi |

The new sampler JSON comparison is complete structural equality, including every
represented float bit. Previous harnesses retain their original numerical
tolerances; their observed maximum error is zero. Core and prior sampler reports
are preserved as separate JSON files. verification.json records all comparisons
and hashes instead of replacing older checkpoint evidence.

## Independent programs and reproduction

Native source archive SHA-256:
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.

External programs:

- C:/Users/adyba/rust-llama-reference/sampler-cpu-oracle
- C:/Users/adyba/rust-llama-reference/regex-oracle
- C:/Users/adyba/rust-llama-reference/vocab-ffi-oracle
- Prior core and sampler programs remain in ffi-oracle and sampler-oracle.

The sampler and regex programs use MSVC 19.51.36260.0. The vocabulary-only harness
uses MSVC 19.44.35228.0 with the same frozen public header and original DLL; its
report records compiler_msvc=1944. Both target x64. Rust is 1.99.1.

From the repository, run:

    cargo test -p llama-ffi -p llama-behavior --lib
    cargo test -p llama-ffi -p llama-behavior --lib --release
    cargo clippy -p llama-ffi -p llama-behavior --all-targets -- -D warnings
    cargo build -p llama-ffi
    cargo build -p llama-ffi --release

The external CPU oracle takes these arguments:

    sampler-cpu-oracle.exe library.dll stories260K.gguf request-all.json output.json

Run once with the original library and once per Rust profile. Compare complete
parsed JSON values. The committed native-sampler-cpu-reference.json differs only
in whitespace and its absolute model path normalized to stories260K.gguf.
The request, source/executable hashes and exact original model hash are in
fixtures/behavior/native-sampler-cpu-provenance.json.

The external vocabulary oracle takes:

    vocab-ffi-oracle.exe native.dll rust.dll frozen-model-directory repository output.json

It uses fixtures/tokenizer/bpe/vocabulary-reference.json to enumerate all original
models and text cases, then raw-bpe-reference.json for explicit byte-length input.
It compares all special getters including FIM aliases, both piece modes, scores,
attributes, EOG, source model-info defaults, all detokenization flag combinations,
count-only capacities and retained partial writes.

Ordinary Rust tests use a metadata-only 11,360-byte GGUF extracted from the actual
stories260K model. They require no trained weights or native library. The original
model and independent C++ programs remain required for the external execution gate.

## Error, ownership and resource review

The safe Rust behavior API remains distinct from the native compatibility path.
Ordinary tests cover pending adaptive clone/copy rejection before destination
mutation, rejected grammar acceptance without state changes, null/empty grammar
arguments that the source ignores, and owned sampler vocabulary surviving model
handle release.

An independent reviewer identified two regex resource-accounting gaps during
development: pending vector capacity was larger than its counted length, and
backreference slice comparison work was not charged by byte length. Both were
corrected and independently reread. Clone allocation is preflighted with exact
extra-task reservation and checked capacity accounting; every compared backreference
byte consumes the shared work budget. Regression tests cover both cases.

Native grammar separately bounds parser expansion, interned symbol bytes,
transient element copies and recognition state. Resource failures propagate through
the C diagnostic channel; they are never converted to a successful unconstrained
grammar or guessed selection.

## Scope still open

These results establish the documented Windows CPU compatibility slice. Exact
cross-vendor C++ RNG/tie order and ECMAScript locale parity remain open. The byte
regex engine explicitly uses the measured MSVC C-locale profile. Backend-attached
sampling, performance counters, llguidance, the remaining 100 frozen exports,
additional context/architecture execution paths and complete application parity
remain outside this checkpoint.

SAMPLERS.md lists bounded resource restrictions, including the 4,096-candidate
infill prefix-merging limit. COVERAGE.md enumerates every present and absent
frozen symbol. Parent workspace integration runs are a separate validation gate.


## Report-path error and fresh validation

After the initial successful validation, an evidence enhancement script reused
its destination variable inside a read-only source-hashing loop. Its final JSON
write therefore targeted native_grammar.rs instead of verification.json. Full
workspace release attempts subsequently failed at the first unexpected JSON
brace. The only affected source was native_grammar.rs; the other twelve recorded
source hashes remained exact.

The damaged 42,298-byte report is preserved externally with SHA-256
f5727a32b0307b448d91c25d3d25c225541a794c35ae5939454ff441301e8eec.
The exact preceding ae2ce547e2dfa9d54c257a7c1ec8133411504565baa31291d1b62f753d701cd8
source was unavailable. The author reapplied the final parser construction/copy
and cumulative symbol-name bounds to the complete intact earlier source. An
independent reviewer checked the complete Parser and delta before the single-file
replacement. This is explicitly a new revision, with source SHA-256
fd035bd0e967056ad5def35c730ea9ad40f9727061a997c96a0bb8e5f5c3798c.

The new revision then passed fresh focused debug/release tests, strict Clippy and
all native CPU/core/prior-sampler/BPE comparisons in an isolated target directory.
These are new executions; the old successful reports were not relabeled as new
evidence. Job job-1794232599088-757819ff-ca1 completed successfully. The committed
revalidated/recovery-report.json and independent-review.md preserve the cause,
base and new hashes, replacement procedure and review. The schema-2 verification
records hashes for every actual new result and current source file.

Failed full-workspace release logs remain under
evidence/2026-11-09/native-surface/windows/release.log and
evidence/2026-11-09/native-surface/linux/release.log, with their status.json files.
Fresh parent integration runs use each platform's revalidated/ directory.

The corrected report writer uses distinct source and output path variables,
asserts every output is inside this evidence directory with a data/document
suffix, and never writes any source file. All thirteen current recorded source
hashes were checked again after writing the final evidence.
