# T10 status — scalar behavior and application foundations

**Track/package:** T10 / `llama-behavior` 0.1.0  
**Implementation commit:** `ed27c493ff38c71039fa24d9f46662648a9b250e`  
**Baseline archive SHA-256:** `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`  
**State:** implemented, compiled and executed for the listed host components; selected behavior independently validated; full T10 completion remains blocked by the obligations below.  
**Target executed:** `x86_64-unknown-linux-gnu`, supplied Rust 1.99.1, host CPU. No accelerator, browser, mobile or real-model execution is claimed.

## What is implemented

### Sampling

Sixteen scalar configurations are implemented: greedy, distribution, top-k, top-p, min-p, typical, fixed temperature, dynamic temperature, XTC, repetition/frequency/presence penalties, top-n-sigma, DRY, Mirostat v1/v2, adaptive-p and logit bias. Chains have explicit stage order. RNG/history/EMA/pending acceptance state is owned, clonable and resettable; invalid application is transactional. The RNG engine is MT19937, with explicit libstdc++-compatible float/double uniform conversion. A singleton distribution consumes a draw, matching the snapshot's fix for RNG alignment.

The independent native corpus runs each algorithm for 16 apply/accept steps. All 256 steps pass: candidate IDs, selected indices and sorted flags exactly, and float outputs within `max(3e-6, abs(reference) * 3e-6)`. The original probability examples also cover several penalty, DRY, typical, XTC and cutoff boundary cases. This does not prove all-vocabulary, tie-order, device or performance parity.

### Native chat

All 54 names registered in the snapshot's `LLM_CHAT_TEMPLATES` map are ported. The native heuristic detector is ported as an ordered set of predicates; its detected-only DOTS1 path is implemented without adding a new registered alias. Whitespace behavior uses the original C-locale ASCII whitespace set. Formatting accepts borrowed message values, performs checked size arithmetic before each concatenation and bounds intermediate system-message merging.

540 independently generated original transcripts compare exactly, covering each registered name across five histories and both assistant-prompt settings. These are native formatters; the separate full Jinja interpreter, template filters/macros and specialized chat/tool/reasoning parsers are still pending.

### Character GBNF

The parser supports rule references, alternation, grouping, literals/escapes, classes, inverse classes, ranges, wildcard and bounded/unbounded repetitions. It compiles to explicit productions and uses an Earley recognizer with parser, expanded-symbol, recognition-state and input limits. It rejects source-unsupported left recursion and reports unknown/token-level constructs explicitly. Partial UTF-8 bytes are validated against possible next character ranges. Failed candidate acceptance does not alter committed state.

All eight original `.gbnf` data examples compile. A separately compiled original grammar oracle provides 232 prefix/completion decisions, all matched exactly. Grammar-candidate integration accepts actual token pieces from the caller; tokenizer IDs/pieces are never fabricated. Lazy triggers, token-level `<[id]>` / `!<[id]>` grammar terminals and vocabulary-specific filtering remain pending.

### Shared JSON

The crate provides a bounded parser/serializer with UTF-8 and UTF-16 surrogate-pair handling, control-character/escape validation, duplicate-key rejection and deterministic object-key output. Integer lexemes are preserved as `i64`/`u64`, including values beyond binary64's exact-integer range. Out-of-range integer lexemes are rejected. Fraction/exponent values use finite binary64; arbitrary precision mathematics remains outside this component. T11 uses this Rust module through a path dependency.

### Speculation and CLI

Greedy verification returns the accepted draft prefix plus a mismatch replacement or bonus token. Sampler transactions retain private snapshots and move only the accepted prefix into live state; rejected drafts do not advance the live RNG or token history. State estimates include inline RNGs, configuration vectors, history capacities and conservative tree-map overhead. A checked maximum 64 MiB state-copy budget is enforced before cloning, including transient-copy headroom; the caller can choose a smaller budget. Logit/candidate arrays have separate bounds. These copies are a correctness checkpoint; no production throughput claim is made.

The real Rust diagnostic executable supports `templates`, `chat`, `sample`, `grammar`, `json` and `speculative`, bounded file/stdin reads, explicit errors and process exit-code tests. It does not advertise unavailable model generation. It is not yet the source-compatible unified `llama` application.

## Validation evidence

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

## Intentional corrections and narrower contracts

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
| Real greedy generation through unified CLI | Blocked on T04 tokenization/GGUF and T05 model/logit/context interfaces; no weights were supplied for this track |
| Backend/device sampling | Planned; requires T02/T05 and relevant backend contracts/hardware |
| Infill sampler and vocabulary-derived DRY breaker discovery | Blocked on real T04 vocabulary semantics |
| Token-level grammar, lazy word/token/regex triggers and llguidance | Planned; llguidance source/version/dependency audit remains open |
| JSON-schema-to-GBNF | Planned; current grammar compiler consumes GBNF input only |
| Full Jinja and specialized PEG/chat/tool/reasoning parsing | Planned; current native-format corpus is not a substitute |
| Speculative draft models, ngram/lookahead/lookup and model cache rollback | Blocked on T05 state transactions; only sampler transaction and greedy verification are implemented |
| Common arguments, aliases/defaults/env/presets and model resolution/cache/downloads | Planned; diagnostic CLI does not claim the original flags |
| Unified `llama` command and standalone inference/conversion/benchmark tools | Planned/blocked on foundation and model tracks |
| Embedding/retrieval, parallel, callback, training and diffusion examples | Planned/blocked on corresponding computation interfaces |
| Native updater/install/recovery workflow | Planned; separately downloaded installer contracts and source remain an acquisition gate |
| Full shared hashing/subprocess/console/preset replacement | Planned; this crate currently replaces a bounded JSON subset only |
| Performance, platform, broad vocabulary and model acceptance | Untested; no throughput/memory regression budget is closed by these unit tests |

## Integration handoff

Only `crates/llama-behavior/**`, `fixtures/t10/**` and `docs/t10*` are owned by this change. No workspace membership or lockfile change is included. There are no non-standard-library Cargo dependencies, no FFI/native implementation linkage and no unsafe blocks (`#![forbid(unsafe_code)]`). The original archive is an oracle input only.

The next bounded task is to align `TokenId`, candidate/logit delivery and sampler acceptance with T01/T04/T05, then drive one real greedy text-generation path with fixed model/tokenizer hashes. Separately, port JSON-schema conversion and token-level/lazy grammar semantics against independent original corpora. Unimplemented source capabilities remain visible in `docs/t10-capabilities.tsv` and the root T12 ledgers.
