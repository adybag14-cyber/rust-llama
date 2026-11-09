# T10 status — scalar behavior and application foundations

**Track/package:** T10 / `llama-behavior` 0.1.0  
**Original imported implementation commit:** `ed27c493ff38c71039fa24d9f46662648a9b250e`

**Baseline archive SHA-256:** `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`  
**State:** implemented, compiled and executed for the listed host components; selected behavior independently validated; full T10 completion remains blocked by the obligations below.  
**Targets executed:** original safe-API corpus on Linux x64; integrated native CPU compatibility corpus on Windows x64/MSVC using Rust 1.99.1. Actual trained-model C sampler comparisons are now included. Accelerator, browser and mobile sampler parity remain open.

## What is implemented

### Sampling

Sixteen scalar configurations are implemented: greedy, distribution, top-k, top-p, min-p, typical, fixed temperature, dynamic temperature, XTC, repetition/frequency/presence penalties, top-n-sigma, DRY, Mirostat v1/v2, adaptive-p and logit bias. Chains have explicit stage order. RNG/history/EMA/pending acceptance state is owned, clonable and resettable; invalid application is transactional. The RNG engine is MT19937, with explicit libstdc++-compatible float/double uniform conversion. A singleton distribution consumes a draw, matching the snapshot's fix for RNG alignment.

The independent native corpus runs each algorithm for 16 apply/accept steps. All 256 steps pass: candidate IDs, selected indices and sorted flags exactly, and float outputs within `max(3e-6, abs(reference) * 3e-6)`. The original probability examples also cover several penalty, DRY, typical, XTC and cutoff boundary cases. This does not prove all-vocabulary, tie-order, device or performance parity.

### Frozen native compatibility path

The separate llama-ffi crate now exports all 20 ordinary frozen CPU sampler
constructors and the generic/custom/chain API over real Rust context logits.
It uses explicit native compatibility operations in this crate while retaining
the existing safe Rust API described above. Source candidate ordering, stale
selection/sorted metadata, repeated candidate IDs, MSVC tie/RNG behavior,
raw-byte vocabulary preprocessing and source clone/reset contracts are tested
independently. OS entropy resolves default seeds in the C wrapper.

NativeDry implements the vocabulary overlap algorithm and signed allowed-length
semantics. infill_native implements prefix probability merging, EOG handling and
fallback. Defined adaptive-p cloning retains EMA and restarts its seed stream;
pending-probability clone/copy, which is unsafe in the source, returns a checked
error. The safe Rust Clone continues to preserve all fields.

The new original-DLL corpus matches exactly in debug/release: 34 grammar/lazy,
55 DRY, 35 adaptive-p and 41 infill scenarios. Seven sampler chains over six
trained prompts generate exactly 585 original tokens. NativeGrammar separately
matches 383,053 raw-piece decisions across 82 cases; NativeRegex matches 12,349
MSVC compile/search/full-match/capture checks. Windows focused runs pass 45
behavior unit tests, 17 FFI unit tests and strict all-targets Clippy. See
crates/ffi/SAMPLERS.md and evidence/2026-11-09/sampler-cpu/verification.json.

### Native chat

All 54 names registered in the snapshot's `LLM_CHAT_TEMPLATES` map are ported. The native heuristic detector is ported as an ordered set of predicates; its detected-only DOTS1 path is implemented without adding a new registered alias. Whitespace behavior uses the original C-locale ASCII whitespace set. Formatting accepts borrowed message values, performs checked size arithmetic before each concatenation and bounds intermediate system-message merging.

540 independently generated original transcripts compare exactly, covering each registered name across five histories and both assistant-prompt settings. These are native formatters; the separate full Jinja interpreter, template filters/macros and specialized chat/tool/reasoning parsers are still pending.

### Character GBNF

The parser supports rule references, alternation, grouping, literals/escapes, classes, inverse classes, ranges, wildcard and bounded/unbounded repetitions. It compiles to explicit productions and uses an Earley recognizer with parser, expanded-symbol, recognition-state and input limits. It rejects source-unsupported left recursion and reports unknown/token-level constructs explicitly. Partial UTF-8 bytes are validated against possible next character ranges. Failed candidate acceptance does not alter committed state.

All eight original `.gbnf` data examples compile. A separately compiled original grammar oracle provides 232 prefix/completion decisions, all matched exactly. Grammar-candidate integration accepts actual token pieces from the caller; tokenizer IDs/pieces are never fabricated. This stricter character API remains distinct from NativeGrammar, which now
implements numeric/named/inverse token terminals, source partial-byte decoding,
NUL/EOG behavior and exact mask/accept transitions. The C wrapper adds lazy
word/token/regex triggers with capture-relative suffix replay and bounded
vocabulary snapshots.

### Shared JSON

The crate provides a bounded parser/serializer with UTF-8 and UTF-16 surrogate-pair handling, control-character/escape validation, duplicate-key rejection and deterministic object-key output. Integer lexemes are preserved as `i64`/`u64`, including values beyond binary64's exact-integer range. Out-of-range integer lexemes are rejected. Fraction/exponent values use finite binary64; arbitrary precision mathematics remains outside this component. T11 uses this Rust module through a path dependency.

### Speculation and CLI

Greedy verification returns the accepted draft prefix plus a mismatch replacement or bonus token. Sampler transactions retain private snapshots and move only the accepted prefix into live state; rejected drafts do not advance the live RNG or token history. State estimates include inline RNGs, configuration vectors, history capacities and conservative tree-map overhead. A checked maximum 64 MiB state-copy budget is enforced before cloning, including transient-copy headroom; the caller can choose a smaller budget. Logit/candidate arrays have separate bounds. These copies are a correctness checkpoint; no production throughput claim is made.

The real Rust diagnostic executable supports `templates`, `chat`, `sample`, `grammar`, `json` and `speculative`, bounded file/stdin reads, explicit errors and process exit-code tests. It does not advertise unavailable model generation. It is not yet the source-compatible unified `llama` application.

## Original safe-API validation evidence

| Check | Observed result |
| --- | --- |
| `cargo test --offline -p llama-behavior` | Exit 0: 31 unit, 4 process-level CLI and 5 integration tests passed; 0 ignored |
| Original native chat oracle | 540 exact transcripts passed |
| Original native sampling oracle | 256 apply/accept steps passed at stated float tolerance |
| Original native grammar oracle | 232 exact prefix/completion decisions passed |
| Original `.gbnf` input examples | All 8 compile |
| `cargo clippy --offline -p llama-behavior --all-targets -- -D warnings` | Exit 0 |
| `cargo fmt -p llama-behavior --check` | Exit 0 |
| `cargo build --offline -p llama-behavior --release` | Exit 0; native executable and library built |

The oracle source is the uploaded archive, compiled independently with `g++ (Debian 14.2.0-19) 14.2.0`. Expected outputs, source hashes, adapter hashes and the detailed comparison rules are in `fixtures/t10/provenance.json`. The C++ oracle and adapters are excluded from this Rust product. Normal tests do not require a legacy binary or the original source tree.

## Safe API corrections and native compatibility boundaries

The following original corrections apply to the safe Rust sampling, character
Grammar, chat and JSON surfaces. The separate native compatibility APIs preserve
the defined frozen C behavior where described in crates/ffi/SAMPLERS.md. In
particular, C candidate duplicates/stale flags and default seeds are supported,
and native token grammar uses raw-byte source rules. Native errors remain
checked and transactional at each builtin operation; owning chains apply stages
incrementally.

### Safe API contract

- Stable ordering is used for tied logits and tied typical scores. The original C++ sorting is unstable/platform-dependent, so tied-token order parity is not claimed. Greedy itself preserves the first maximum.
- NaN/+infinity inputs, all-masked sampling, duplicate token IDs/bias IDs, degenerate Mirostat-v1 slope/casts and invalid resource sizes fail with errors instead of copying undefined behavior/assertion failures.
- Underflowed zero probabilities use the mathematical `0 log 0 = 0` entropy contribution. Logit bias and adaptive transformations invalidate the sorted flag when ranking can change; the source can retain stale metadata.
- Rust cloning preserves every adaptive sampler state field, including RNG and pending probability; this is intentional even where the original clone implementation omits fields.
- `u32::MAX`, the original ambient-random seed sentinel, is rejected by sampler constructors until the platform entropy contract is supplied. Explicit seed values are implemented. Cross-vendor C++ RNG-distribution behavior is not guaranteed.
- GBNF duplicate definitions, invalid Unicode escapes, nonproductive/reversed ranges and excessive repetitions fail explicitly. This checkpoint uses stricter resource limits instead of the source's fallback that turns some oversized upper repetition limits into unbounded repetition.
- JSON duplicate keys are errors rather than last-write-wins. Integer values outside signed/unsigned 64-bit range are errors rather than silently becoming imprecise doubles. JSON key ordering is deterministic; byte-for-byte preservation of an imported JSON document requires keeping its original bytes, as the future T11 migration implementation must do.
- Empty roles and embedded NUL in native chat messages fail explicitly where the original C-string contract would be undefined or truncate data.

## Remaining mandatory T10 work

| Work package | Current state and next dependency |
| --- | --- |
| Real greedy generation through unified CLI | Integrated via llama-inference/llama-cli and verified trained models; see root integration status for the supported architectures |
| Backend/device sampling | Planned; requires T02/T05 and relevant backend contracts/hardware |
| Infill sampler and vocabulary-derived DRY breaker discovery | Implemented and independently validated through native compatibility and C APIs; bounded resource restrictions are documented |
| Token-level grammar and lazy triggers | Implemented through NativeGrammar/NativeRegex and C wrappers; cross-vendor regex parity remains open |
| llguidance | Source/version/dependency audit and integration remain open |
| JSON-schema-to-GBNF | Planned; current grammar compiler consumes GBNF input only |
| Full Jinja and specialized PEG/chat/tool/reasoning parsing | Planned; current native-format corpus is not a substitute |
| Speculative draft models, ngram/lookahead/lookup and model cache rollback | Blocked on T05 state transactions; only sampler transaction and greedy verification are implemented |
| Common arguments, aliases/defaults/env/presets and model resolution/cache/downloads | Planned; diagnostic CLI does not claim the original flags |
| Unified `llama` command and standalone inference/conversion/benchmark tools | Planned/blocked on foundation and model tracks |
| Embedding/retrieval, parallel, callback, training and diffusion examples | Real BERT embedding facade/server are integrated separately; remaining examples and architecture coverage remain open |
| Native updater/install/recovery workflow | Planned; separately downloaded installer contracts and source remain an acquisition gate |
| Full shared hashing/subprocess/console/preset replacement | Planned; this crate currently replaces a bounded JSON subset only |
| Performance and platform acceptance | Native Windows sampler/model and broader vocabulary checks are present; no throughput budget or complete cross-vendor sampler parity is claimed |

## Integration handoff

This crate remains dependency-free safe Rust with no original llama/ggml linkage
and no unsafe blocks. The frozen C ABI and foreign pointer boundary are owned by
crates/ffi; the independent C++ reference programs live outside the product.
fixtures/behavior stores compact original-engine and original-DLL expectations
with hashes/provenance; normal Rust tests require no native binary.

The next missing behavior work includes JSON-schema conversion, full Jinja and
specialized chat/tool parsing, llguidance, backend sampling, complete common
arguments/model resolution and speculative model-cache transactions. Source
capabilities outside this checkpoint remain tracked in the root integration
ledgers; the presence of a native CPU sampler corpus does not establish full
application, architecture or backend parity.
