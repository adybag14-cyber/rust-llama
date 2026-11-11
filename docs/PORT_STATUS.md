# Current port status

Updated 11 November 2026. The user objective remains a complete native Rust port of the frozen ZIP. **The full port is not complete.** This document records actual executable scope and the work still required. No overall completion percentage is assigned.

## Source and product boundary

The baseline ZIP SHA-256 is `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. All 4,082 source paths have an owner, byte count and checksum in `ledger/baseline-files.tsv`. Source verification compares the complete extracted tree, including additions and removals. Ownership coverage is not semantic completion.

The product's maintained implementation is Rust. Frozen C/C++ reference builds, model downloads and oracle generators remain outside the repository. Numeric tables, tokenizer-only GGUF, reference float/byte arrays, media examples and synthetic binary parser inputs are declared fixture data. Minimal Cargo/CI configuration, licenses and documentation retain their actual formats.

The merged workspace has local path dependencies plus five exactly reviewed Rust registry packages for converter YAML parsing. Locked dependency acquisition precedes offline gates; the audit verifies the actual archives, extracted source, versions, features and licenses. Optional OS/driver/vendor interfaces remain named external boundaries. They do not import the original llama.cpp implementation. See `release/dependency-inventory.tsv`.

## Latest integrated verification — 11 November

The original full attempts passed **728 Windows MSVC / 732 Linux GNU ordinary
tests in each of debug and release**, with zero failures, 33 ignored
prerequisites and 133 test-result suites per run, plus formatting and strict
Clippy. Both attempts then failed the packaging audit because
`crates/model-quant/tests/fixtures/cli/LICENSE` was not classified as an
admitted non-Rust item. Their original reports remain failed, and neither
report records a completed final input guard.

After both attempts stopped, a separate boundary check verified the original
selected inputs before the correction. The correction renamed the license to
`LICENSE.md` without changing its 1,078 bytes, changed only that path in
`docs/model-quant-cli-fixture-manifest.tsv`, and increased the CPU GitHub Actions
timeout from 45 to 60 minutes. All 1,674 other selected files remained
byte-identical. The selected set retained 1,677 files. No Rust implementation
or numerical fixture content changed, and the sealed component evidence was
not replaced.

The Windows and GNU continuations then passed all 29 remaining required
stages with their corrected selected inputs unchanged. Each host separately
passed thirteen pretrained inference/server tests, six converter tests, one
native BGE encoder test and one official Qwen tokenizer replay. These explicit
runs do not imply that every ignored device or external-data test ran.
Formatting, strict Clippy, the actual-cache dependency/source audit, release
builds, and both 4,082-file frozen-tree verifications have successful receipts.
Both hosts verified **1,062 fixture files and declarations across nineteen
manifests**.

All four final debug/release C libraries passed the retained original-header
clients. Each BERT encoder replay covers 192 primary scenarios and 37,384
bit-identical values, plus nine vector-admission cases with 137 compared values
and explicit bounded-error cases. Each quantization replay covers 88 requests,
77 successful cases, eleven rejections and 76 exact complete output files.
Each final export inventory contains **232 frozen exports plus four diagnostic
extensions**, leaving **23 frozen exports absent**. These counts describe the
documented execution scope, not complete semantics for every export.

The schema-3 receipt assigns each of 33 required successful stages to its
actual report and input set: formatting, ordinary debug, ordinary release and
strict Clippy come from the successful stages of the failed original attempts;
the remaining checks come from the successful continuations. Each host has
34 attempted commands including the retained failed audit. The ordinary suites
were not repeated after the correction.

This is selected-input verification. The guard excludes evidence and the two
progress documents, README.md and docs/PORT_STATUS.md; it is not a
whole-repository immutable guard. Both final continuations and the evidence
packager match the 1,677-file manifest with SHA-256
`1ab73dd93269bcfc9b25f647ea06474568aaffc9e73c145b54d93a392c56acf0`.
The original component receipts retain their tested filenames and hashes, with
the old CLI fixture-manifest reference resolved through the retained
pre-correction snapshot. The packaged record contains 534 files and 7,417
verified recorded identity links; historical logs without capture-time hashes
remain explicitly labeled as having packaging-time identities.

All **twenty mandatory full-port gates remain open**. The full-port release
check retains its expected nonzero result, and published-commit CI is reported
separately by GitHub Actions. See
[the schema-3 integrated receipt](../evidence/2026-11-11/continued-port-late/README.md)
and
[the exact correction](../evidence/2026-11-11/continued-port-late/integration-metadata/packaging-correction/report.json).

## Previous integrated verification — 11 November

The integrated continuation passed **691 Windows MSVC / 696 Linux GNU ordinary
tests in each of debug and release**, with zero failures, 29 explicit prerequisite
tests ignored and 124 test-result suites per ordinary run. Separate release runs
passed eleven pretrained inference/server tests, five complete-file converter
tests and one official Qwen tokenizer replay on each host. The ordinary ignored
entries remain explicit; these separate executions do not imply that every
ignored device or external-data test ran.

Formatting, strict Clippy, release builds and the actual-cache dependency/source
audit passed using Rust 1.99.1 / LLVM 23.1.1. Both hosts verified all 959 fixture
files and declarations across fourteen manifests, and all 4,082 files in each
of the supplied and reference frozen trees. Original-header clients exercised
both debug and release C libraries, including the adapter replay, with exact
inventories of 226 frozen exports plus four diagnostic extensions. Twenty-nine
frozen exports were absent at that checkpoint.

Verification was staged, and the original reports retain their actual outcomes.
Both first attempts passed the ordinary suites, formatting and strict Clippy,
then failed the source-language package audit. The correction renamed evidence
logs to admitted log suffixes, moved a binary-attribute rule to the root and
removed the two nested attribute files. The numerical fixtures and runtime
source were unchanged: a separate guard proves all 1,534 retained selected files
identical, with only two changed package-control files and one removed
package-control file in the selected input set.

The first Windows continuation then failed reference-copy verification because
an earlier oracle import had created ten Python bytecode cache files. Those
generated files were moved outside the frozen tree; no baseline source file
changed, and the supplied Downloads tree had already passed. The final GNU and
Windows continuations passed their remaining requested checks with the same
1,536 selected execution inputs unchanged. That selection includes release
controls and attributes, and excludes evidence plus the two progress documents
README.md and docs/PORT_STATUS.md. It is not a whole-repository immutable guard.

The full-port release command still returns its expected failure with
**twenty mandatory gates open**. Published-commit CI is reported separately
by GitHub Actions. See
[the staged receipt](../evidence/2026-11-11/continued-port-next/README.md)
for all five attempt reports, log identities, corrections, native-client
receipts and external driver identities.

## Earlier integrated verification — 11 November

Both native hosts passed the complete ordinary debug and release suites:
662 tests per Windows MSVC build and 667 per GNU build, with zero failures,
27 explicit prerequisite tests ignored, and 117 test-result suites per run.
Separate release checks passed eleven pretrained inference/server tests,
three complete-file converter tests and one official Qwen tokenizer replay.
Both debug and release C libraries passed the retained native clients and exact
216-frozen-plus-four-extension export checks; 39 frozen symbols were absent at that checkpoint.

Rust 1.99.1 / LLVM 23.1.1 builds, strict Clippy, formatting and the reviewed
dependency audit passed. Both hosts verified all 4,082 source files in each
frozen tree and 858 fixture files / declarations across ten manifests.

The original full runs retain their failed execution-input guards: only the
CI workflow changed after launch. The separate review proves all 1,279 other
recorded inputs unchanged; four postflight stages passed with the final
1,280 selected execution inputs unchanged. This is a qualified selected-input
verification, not a whole-repository immutable guard or a published CI result.
All twenty mandatory full-port gates remain open. See
[the complete receipt](../evidence/2026-11-11/continued-port/README.md).

The first published workflow was rejected before scheduling because three
Qwen paths used the runner context at job-level environment scope. The
workflow-only correction moves the same values to their consuming steps;
all 1,279 selected non-CI inputs remain unchanged. The original local
postflight records and failed GitHub run are preserved in the
[correction receipt](../evidence/2026-11-11/continued-port/ci-workflow-context-fix/correction.json).
The corrected published workflow is verified separately by GitHub Actions.

## Historical integrated verification — 10 November

The final integrated input manifest covers 1,047 code, fixture and execution
configuration files and is identical on Windows MSVC and Linux GNU. The complete
ordinary debug and release suites pass 592 tests per Windows profile and 597 per
GNU profile, with zero failures and 22 explicitly ignored external/device tests.
Each host separately passes nine pretrained inference/server tests and the
complete SmolLM2 converter test (three outputs; 686,581,536 exact bytes).

Formatting, strict Clippy, release builds and the actual-cache dependency audit
pass. All 4,082 supplied frozen source files match; all 720 fixture files have
integrity declarations across nine manifests. Original-header clients validate
both debug and release libraries: 35,946 chat cases, 171 model-metadata requests,
38 raw-name models, and actual PE/ELF export inventories of 198 frozen symbols plus
four extensions. Fifty-seven frozen C symbols were absent at that checkpoint.

The retained reports distinguish earlier integration failures, their corrections
and the final successful run. The full-port release check still fails as required
with 20 open mandatory gates. See
[evidence/2026-11-10/continued-port](../evidence/2026-11-10/continued-port/README.md).

## Continued executable scope — 11 November

BLOOM adds a native ALiBi graph, fused QKV, affine input/final LayerNorm,
direct embeddings, tied/untied output, and measured GQA/MQA/non-power-of-two
head variants. Ordinary captures compare 346,644 F32 words and 216 greedy IDs;
the unchanged official F16 and Q8_0 BLOOM-560M models compare another
13,074,432 words and 24 IDs per native profile. The official 560M replay
requests a 4 GiB decoded-weight budget; packed storage and runtime allocations
remain additional. See
[the inference contract](bloom-inference.md) and
[its independent evidence](../evidence/2026-11-11/bloom-inference/README.md).
This high-level model addition does not add BLOOM to the separate native
decoder C execution context.

BLOOM checkpoint conversion adds the sixth admitted dense HF family. Its
113 ordinary native cases per profile include 84 accepted complete files and
29 rejections. Independent tokenizer tests cover all 1,112,064 Unicode scalars,
1,922 Split/encode/decode cases, 87 class configurations and 1,308 shared
six-family filtering decisions. Three unchanged official checkpoint outputs
total 2,871,136,896 exact bytes per host. The explicitly separate source-converter
writer admits the original Python nested-array extension; strict GGUF writers
and readers retain their prior admission boundary. The exact pretrained replay
uses the input directory basename `official-hf`, which contributes to default
metadata. See [the converter contract](BLOOM_HF_CONVERSION.md) and
[the component receipt](../evidence/2026-11-11/bloom-conversion/README.md).

The native BERT C path adds four frozen functions with complete-batch
bidirectional execution, source sequence-container ordering, NONE/MEAN/CLS/LAST/
RANK pooling and optional classifier heads. The primary original-header corpus
contains 192 scenarios and 37,384 exact values; a separate nine-case active-vector
corpus adds 137 values plus checked rejections of five source-aborting inputs.
Both debug and release libraries pass on both hosts. Bounds run before foreign
embedding copies, and output pointer/lifetime behavior is measured separately.
The frozen BERT has-encoder/has-decoder queries retain their original false/true
results; this query behavior does not prevent the implemented encode path.
See [the native encoder contract](../crates/ffi/ENCODER.md) and
[the component receipt](../evidence/2026-11-11/c-bert-encoder/README.md).

The new `llama-model-quant` crate implements model-file conversion and the two
frozen quantization C functions without the original runtime. All 35 source
file modes have independent type and complete-file coverage; the file driver
admits Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BERT, T5 and BLOOM metadata.
T5 file admission does not add T5 inference. The core corpus contains 463
type/default cases and 88 file requests per profile with 76 exact complete
outputs. The separate 346-case metadata corpus checks 140 complete copies,
202 rejections and four outside-family exclusions; ten additional complete
outputs cover NextN-aware Llama 70B selection. Source payloads, decoded rows,
output slabs and temporary packed worker rows have explicit budgets. Existing
destinations survive conversion failure; publication of multiple shards is
not a filesystem-wide transaction. See [the precise file/API contract](../crates/model-quant/README.md)
and [the component receipt](../evidence/2026-11-11/model-quantization/README.md).

The standalone `llama-quantize` executable adds source argument parsing,
default naming, legacy/GGUF importance ingestion and normalization, filtering,
metadata/tensor overrides and dry-run behavior. Independent preimplementation
captures contain 517 helper observations and 188 actual executable cases per
host. The maintained suite compares 117 MSVC / 118 GNU complete outputs byte for
byte; the difference follows the documented native-long metadata range.
The compact output/input fixture contains 99 deduplicated blobs.

The Windows file-identity boundary uses the complete 128-bit file ID plus
volume identity, checked against an independent installed-SDK ABI capture.
Both CLI and library reject write-time aliases while retaining source dry-run
admission. GNU original-C alias writes can corrupt the disposable input or
abort; Rust preserves the input and returns an error. That intentional safety
correction and the private UCRT hexadecimal-rounding correction remain explicit.
The latter preserves every captured public F32 metadata and complete output
byte. See [the CLI contract](../crates/model-quant/CLI.md) and
[its component evidence](../evidence/2026-11-11/model-quant-cli/README.md).

The current implementation has eight admitted model families, 100 executable tensor operation IDs plus NONE input leaves, and 232 frozen C exports plus four diagnostic extensions. The sole ID without an executable constructor is FLASH_ATTN_BACK: its public constructor aborts in the frozen source on both measured profiles. Twenty-three frozen C exports remain absent. Operation and export counts describe the documented execution slices; all twenty mandatory complete-port gates remain open.

The additional tensor work provides extended forward rotary modes and masked/sink/ALiBi softmax, explicit SGD and AdamW state steps, and owned Rust custom callbacks using real scoped workers. The attention corpus retains 3,361 successful original graphs per host: 3,303 admitted graphs compare 11,181,540 bytes exactly and 58 numerical exclusions remain explicit. Optimizers add 300 cases / 440,704 exact output bytes; custom callbacks add 260 cases / 30,064 exact bytes per host. AdamW weight/moment projections execute together through the graph's multi-target API. Immutable snapshots and out-of-place callbacks do not establish full writable backend storage, automatic differentiation, training or general scheduling.

Ten native C adapter functions load owned LoRA GGUF data, expose metadata and invocation tokens, replace active context adapters, and apply control vectors to real dense Llama CPU execution. Each original host contributes 44 load cases, 107 scenarios and 333 decodes with 170,496 exact logit words. Ownership, source map order, retained KV state, token/direct-embedding behavior and partial control rows have explicit contracts. The FILE* adapter loader, full context performance counters and warmup semantics remain open.

The earlier general HF converter checkpoint admitted five dense families: Llama, Qwen2, Qwen3, GPT-2 and GPT-NeoX; BLOOM is added above. The new GPT corpus has 101 native cases per profile: 77 accepted complete files, 24 rejections and 143,810,730 exact successful bytes. Six full pretrained outputs separately total 718,209,760 bytes per profile. The unmodified official GPT-2 checkpoint retains its three original conversion rejections; its positive fixture removes only twelve static causal-mask buffers and preserves all 148 other tensors. A Rust test helper verifies the exact derivative hash. Pythia is unchanged. The original Qwen corpus and six official outputs remain separate evidence; converter families are not additional inference architectures.

See [extended attention](../crates/tensor/ATTENTION_EXT.md), [optimizers](../crates/tensor/OPTIMIZERS.md), [custom callbacks](../crates/tensor/CUSTOM.md), [C adapters](../crates/ffi/ADAPTERS.md), [conversion](../crates/convert/README.md), [tensor support](../crates/tensor/CPU_SUPPORT.md), and [C export coverage](../crates/ffi/COVERAGE.md). Full integrated validation is recorded separately from these component captures.

## Track-by-track scope

| Track | Current implementation and evidence | Requirements still open |
| --- | --- | --- |
| T01: ABI/platform/integration | Shared workspace/loader contracts, real system symbol loading and 232/255 frozen C exports with independent header, ownership, raw-tokenizer and real-context sampler checks. | The remaining 23 exports, full semantics of every implemented export, all platform contracts and cross-platform ABI execution. |
| T02: tensor engine | Four-dimensional GGML layouts and strides, block-aware views, immutable storage lifetime, scalar DAG execution, cancellation, 100 executable operation IDs plus NONE leaves, including the checked shared CPU FLASH_ATTN_EXT adapter, extended forward/backward RoPE, masked/sink/ALiBi softmax, owned optimizer state and custom callbacks, Lightning Indexer, gated delta/hyper-connections and vision/loss kernels; earlier corpora remain preserved. | Complete dtype/shape/parameter and numerical domains; writable/imported/device buffers; scheduler, allocation planning/reuse, events; autodiff and complete optimizer/training integration. |
| T03: CPU/quantization | All 36 active dtype decoders; all 29 source model quantizers with required/optional weights; 32 non-integer codec formats including activation/F64 helpers; three checked integer setters; all 29 scalar CPU weight-dot types; bounded file quantization across 35 modes and eight admitted metadata families. | General CPU operators and all ISA implementations; dispatch, numerical and performance parity across ISAs. Source-undefined codec paths have documented safe behavior. |
| T04: formats/conversion/tokenizers | Bounded shared raw/text GGUF v2/v3 reader, native byte/NUL alias and nonzero-boolean semantics, owned automatic/explicit shard sets, cumulative parser limits, tensor validation, byte-preserving ordered streaming little-endian writer, bounded split/merge executable, scalar endian conversion, raw/indexed Safetensors reading with retained capabilities, native dense Llama/Qwen2/Qwen3/GPT2/GPTNeoX/BLOOM HF conversion with checked publication, raw primary model metadata and native scalar C ordering, SPM, WordPiece, BPE with all 64 metadata-selected profiles / 99 aliases, Unigram, RWKV, PLaMo2/PLaMo3, explicit no-vocabulary mode, native raw-byte behavior and exhaustive Unicode checks. | Other architecture checkpoint converters; remaining arbitrary-byte metadata semantics across model/vocabulary consumers; remaining metadata tools, file interfaces and quantization application semantics. Big-endian packed arithmetic and remaining platform-specific behavior. |
| T05: models/runtime | Typed Llama/Qwen2/Qwen3/GPT2/GPTNeoX/BLOOM/Mamba/BERT graphs, state/cache transactions, sequence isolation, adapters and independent small graph checks. Dense Llama GGUF executes pretrained F32/Q4_0/BPE-Q4_K_M models; Qwen2 GGUF executes pretrained Q4_K_M generation; Qwen3 GGUF executes pretrained Q8_0 generation with independent head width and per-head Q/K normalization; GPT-2 and GPT-NeoX execute pretrained F16 generation with exact native-profile logits; BLOOM executes pretrained F16/Q8_0 generation with ALiBi; Mamba-1 GGUF executes pretrained F16 recurrence; dense BERT GGUF executes pretrained F16 embeddings and native C pooled/classifier execution. | Remaining named architectures among the baseline's 172 entries; full metadata/graph variants and scaled RoPE; memory/cache modes, speculation, batching, model placement and GPU model integration. |
| T06: CUDA/HIP/MUSA | Rust driver interfaces and typed PTX generation; actual CUDA kernels including matvec, stride/broadcast/tail coverage; 138 recorded dispatches on RTX 4090. | Full tensor/backend scheduler integration, upstream operation and dtype coverage, memory/collective modes, optimized kernels; HIP and MUSA hardware execution/parity. |
| T07: portable GPU | Typed kernel IR; generated OpenCL kernels executed on NVIDIA and AMD; Vulkan/other host foundations. | Complete Metal, Vulkan, OpenCL and WebGPU implementation; full compiler/device/OS matrix, scheduler integration and numerical/performance parity. |
| T08: specialty/remote | Checked RPC 8.0.1 subset with transactional scalar graph execution; CBLAS interfaces, DSP ELF/SWIV and virtual-device utilities with explicit capability handling. | Complete BLAS, CANN, ET, SYCL, zDNN, Hexagon, ZenDNN, OpenVINO, RPC and VirtGPU backend semantics and actual required device/vendor testing. |
| T09: multimodal/media | Selected Rust image/audio decode, preprocessing and resampling; independent media fixtures. | All 58 multimodal graph builders; model projections, audio/video model execution and speech generation; full codec/container/device capability coverage. |
| T10: behavior/tools | Safe scalar sampler APIs and a separate native C sampler path; DRY, adaptive-p, infill and lazy/raw-byte GBNF with independent DLL masks/state/context generation; 54 native chat formatters and two byte-oriented C chat exports; inference, bounded model-file quantization with legacy/GGUF importance ingestion, and dense Llama/Qwen2/Qwen3/GPT2/GPTNeoX/BLOOM conversion CLIs. | Complete PEG/Jinja/custom-regex and remaining sampler semantics, complete common options, all tools/examples, converter/updater/unified-app functionality. |
| T11: server/UI/clients | Real-model native/OpenAI completions and SSE; BERT embedding endpoints with all pooling modes, normalization, base64, typed/batched input and usage; bounded HTTP, auth/CORS, replay, cancellation and deadline checks. | Full routing/scheduling/protocols, multimodal and decoder embeddings, reranking, tools/MCP, TLS, browser features/storage migrations/WASM, native clients and connect integration. |
| T12: validation/release | Original ownership/registry ledger, source/fixture audits, independent fixtures, offline Rust tasks and deterministic source packaging; Windows/Linux CI configuration. | Full capability-linked file disposition review; all model/platform/device suites; clean distribution verification, memory/quality/performance budgets and complete release closure. |

The tables and registries in the original plan retain the full scope. A missing feature is not waived because it is difficult or requires unavailable hardware.

## Pretrained inference findings

The first end-to-end model is `stories260K.gguf`, SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`. The independent frozen llama API oracle uses one CPU thread, F32 KV and disabled flash attention. Six prompts produced 96 matching greedy tokens. The initial 62,976-logit comparison had maximum absolute error `1.764297485e-5`. Matching the source's F32 products with F64 accumulation gave exact Windows equality and left a historical GNU difference of `5.722045898e-6`. The new matching-profile captures and physical attention-row correction now give exact equality on both verified targets. The predefined portable bound `2e-4 + 2e-5 * abs(reference)` is unchanged.

The larger `stories15M-q4_0.gguf`, SHA-256 `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`, initially exposed a real bug: expanding all packed matrices to F32 preserved sixteen greedy tokens but changed logits by as much as `0.4481242895126343`. The corrected runtime retains packed weights, converts activations according to the source CPU trait, and evaluates integer block dots with the original scaling order. All 640,000 compared logits then matched the frozen Windows reference exactly. The report before the fix is retained beside the report after the fix.

The F16 BGE-small BERT model, SHA-256 `f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999`, exposed a second real rounding issue. Input embeddings, normalization and Q/K/V projections were exact, but F32 attention accumulation differed from the scalar source's F64 sum. Those small differences crossed GELU table bins in later layers. Correcting attention and mean-pooling order yields exact Windows equality for all 13,056 compared values and all 122 captured diagnostic stages. The earlier GNU comparison had maximum error `1.348853111e-4`, within the original fixed tolerance. Preserving the GNU source's four-lane variance order and comparing its separate native capture now gives exact equality. A 12-layer independent attention fixture exercises 46,080 output values in ordinary tests. The before/after diagnosis remains recorded.

The equivalent big-endian BERT download is separately hash-pinned and produces the same expected results through the Rust reader. The frozen C reader rejects that opposite-endian file; this is reader validation against the equivalent little-endian native corpus, not a claim of original-reader support.

Exact equality is recorded evidence for the exercised build and prompt, against the corresponding original profile. Portable tests retain the previously declared numerical tolerance because platform math libraries and ISA reduction orders can differ. These tests do not establish arbitrary quantized-format support or all-model quality parity.

The SPM corpus uses the original vocabulary-only fixture and verifies 68 cases, 312 pieces, parsing/insertion flags, byte fallback, whitespace and detokenization. WordPiece adds 1,728 native cases across nine metadata variants, all 30,522 original vocabulary entries and 72 special-token fallback cases. Unicode checks cover all 1,114,112 codepoints, 15,957 normalization sequences and 65,809 UTF-8 inputs. The BPE extension now covers every reachable metadata profile: 64 pretokenizer types and 99 aliases. Three other enum names have no metadata dispatch in the frozen source; they are not invented as extra supported profiles. Independent Windows/Linux comparisons cover all 14 original BPE vocabularies, complete property/piece/EOG records, 22,518 raw-byte cases and 126,896 regex splits. Ambiguous inferred EOT/FIM IDs preserve the measured MSVC/libstdc++ container ordering. Full-vector/input tests require explicit original-vocabulary prerequisites; compact synthetic native corpora run ordinarily. The remaining frozen tokenizer families are now implemented: Unigram, RWKV and PLaMo2/PLaMo3, with an explicit no-vocabulary metadata mode. Their independent corpus covers 91,476 synthetic cases, 346,112 original vocabulary records, 10,412 real-vocabulary tokenization cases and 4,067,668 normalizer outputs. This does not close architecture conversion or every native C override configuration.

## Scalar codecs, native behavior and HTTP extensions

The scalar codec slice now covers every defined source model quantizer, including
importance-weighted legacy/K/IQ fitting. Three model formats require caller-supplied
weights. Q8_1/Q8_K remain activation codecs and F64 remains a conversion helper;
they are not counted as extra source model quantizers. The new independent corpus
has 3,362 original-C encoded rows and 640 tied-sort cases. The original IQ4_NL
uninitialized near-zero payload and exceptional IQ1 out-of-bounds fallback are
handled explicitly and excluded from undefined-source byte parity claims.

The raw-byte GBNF engine remains separate from the safe Unicode recognizer. Its
native corpus checks 383,053 candidate masks and source token terminals/partial
byte behavior. The C sampler extension adds strict/lazy grammar, DRY, adaptive-p
and infill, with 165 independent scenarios and 42 real contexts / 585 generated
tokens. Candidate order, flags, float bits, selected indices and lifecycle
operations match the frozen Windows DLL. Native BPE/FIM getter, piece and
detokenization checks add 1,505,576 vocabulary records and 7,598 tokenization cases.
The earlier context/memory extension established 184 exported source symbols, with 71 then absent. Its independent 80-case cohort covers 400 checks, 61,440 continuation logits and 1,223,532 state bytes. Restoring and reserializing original state payloads is byte-exact on both platforms. Windows generated floats are exact; Linux retains the original fixed tolerance. Native sessions, per-sequence state, memory operations, bounded copies and file edge cases have separate frozen-header checks. An additional family C getter cohort covers 347,066 vocabulary records and 27,508 tokenization cases, with 176 physical-metadata cases covering 23,903 records and 480 outputs. Nonnull metadata override lists and native recurrent C memory remain explicitly unsupported.

The server embedding corpora exercise 256 ordinary tiny-model requests and
27 explicit pretrained BGE requests per original build. Both verified profiles
now require exact corresponding F32 bits and all seven base64 vectors. GNU
expectations come from a separately built, unmodified original server. The
original profiles themselves differ in 23,786 of 24,704 embedding values; mode
0 amplifies their maximum absolute difference to 1.4208984375. That source-to-
source difference is not a Rust error. The matching-profile Rust comparisons
record zero error. Earlier cross-profile failures and the unchanged portable
normalization envelope remain recorded. Direct normalization of identical raw
inputs is separately checked over 4,608 outputs.

Transport failure tests exposed a deadline race: the listener's monitor could
cancel the worker before the handler checked its timer, turning a timeout into
a dropped response. Both embedding and nonstream generation now check the
deadline before forwarding a generic cancellation result. The regression
requires a complete HTTP 504 response and verifies that capacity is released.
Disconnect and server-shutdown cancellation remain separate cases.

## Qwen2, source row math and tensor coverage

Dense Qwen2 execution supports half-pair NeoX RoPE without scaling, separate or
fused Q/K/V projections, source bias selection, tied or separate output and
output bias. The pinned Qwen2.5-0.5B Q4_K_M input contributes 5,165,824 unique
logits and 24 greedy predictions per original profile. Both profiles match
their own native output exactly. Its decoded F32 cache requires a 3 GiB weight
budget; the default 2 GiB limit deliberately rejects the load.

Twenty-one synthetic Qwen2 variants cover separate/fused projection and bias
combinations, packed Q4 and output variants. Thirteen physical metadata inputs
check source typing and RoPE resolution. Review caught partial fused-bias
selection and nonfinite output-bias transaction failures; permanent regressions
and the failed baselines are retained.

The verified original GNU build enables SSE2 row exponentiation, four-lane
sum and variance; verified MSVC uses scalar row branches. Rust preserves those
policies, whole-row SiLU, attention padding to 256 columns, masked cache holes
and shared attention widths across active streams. Expected profile selection
is independent of runtime dispatch. Native physical-layout probes include
31,744 continuation logits and 193,868 state bytes, exact on both hosts.
Unverified ISA and platform configurations remain open.

The convolution/pooling/resize extension adds thirteen operation IDs and
981 distinct original graphs, including 530,246 exact output bytes per host.
Twenty overlapping backward-gradient cases preserve native MSVC/GNU NaN
operand selection; the earlier GNU mismatch in ten cases and 66 F32 words is
retained in the evidence.
Shape, stride and parameter/error behavior are tested separately. Tensor
coverage before the flash adapter was 75 executable IDs plus NONE. The adapter
checkpoint raised this to 76 executable IDs plus NONE, with 25 then pending.
That recurrent checkpoint reached 81 executable IDs plus NONE, with 20 then pending.
The earlier tensor corpus retains 3,240 distinct graph inputs and
6,118 native captures; repeated platform executions are not counted as new
graph cases. Shared scalar/SSE2 row math now serves model-runtime and tensor;
1,007 additional original graphs require exact matching-profile bytes. The
system erff boundary is independently verified on the captured x86_64 dynamic
UCRT/glibc profiles; other providers, static targets and remaining ISA behavior
retain explicit unverified or fallback qualifications.

Automatic shard loading now uses a single validated file-set owner through the C facade as well. The new frozen-header regression checks 96 cases per profile, with 53 accepted and 43 rejected loads, exact aggregate getters and 163,840 exact F32 continuation words. That automatic-discovery checkpoint retained 184 frozen exports and four diagnostic extensions. The following explicit-list/filename checkpoint adds three exports as recorded below.

## Dense Qwen3 and native F16 cache support

Dense causal Qwen3 now executes through the Rust inference facade and strict Context, with separate or fused Q/K/V, the source bias precedence, independent attention width, learned Q/K RMS vectors and full-head NeoX RoPE. The pinned official Qwen3-0.6B Q8_0 model contributes 5,469,696 exact logits and 24 greedy IDs per native profile. Ordinary dense/split/vector cases compare another 294,912 logits and 400 IDs. Active non-F32 norms/biases rejected by the original CPU graph return checked errors; ignored partial fused-bias groups remain admissible. Qwen3 participates in the strict state/cancellation/allocation regressions. Its 2,384,199,680 decoded-weight bytes require an explicit 3 GiB budget. [QWEN3_IMPLEMENTATION.md](QWEN3_IMPLEMENTATION.md) records model identity, numerical evidence and the remaining graph variants.

The separate Llama NativeContext now owns F32 or raw-u16 F16 K/V independently. Explicit nonflash execution with one generation and one batch worker is admitted. Copies, deferred rotation, transposition and native state/session I/O preserve all 65,536 half bit patterns. Ordinary C regressions cover 258,048 exact logits, 1,048 buffers and all four K/V type pairs per profile. Evaluation retains checked nonfinite-result rejection; raw NaN state preservation does not imply original NaN-logit parity. The later flash/worker implementation admits the original F16/F16, AUTO and four/four defaults for supported dense-Llama configurations, as described below. The previous manual AUTO-to-nonflash substitution has been replaced by source capability resolution.

## GGUF streaming tool checkpoint

The Rust llama-gguf-split executable and llama_gguf::split APIs now perform
bounded split/merge operations. The public ordered writer preflights complete
headers before writing; tool-private behavior preserves original metadata order,
32-byte physical output layout, rank canonicalization and C-string value
handling. The public writer retains valid metadata-selected alignment and
supplied dimensions. Certain original producer quirks can still create files
that the original reader rejects; the corpus records those outcomes explicitly.

The unchanged original 103-case corpus passes on Windows and GNU in both debug
and release: 63 defined successes, 40 checked errors and 12,179,168 exact
post-operation file bytes per run. Focused validation passes 65/67 ordinary
tests per profile, strict Clippy, formatting and fixture integrity. Safe resource
and alias regressions cover header budgets before reservation, cumulative merge
limits, one-byte streaming buffers and output-path replacement after planning.

Merge deletion starts after the complete output has been flushed and synced.
Earlier failures preserve all inputs. A later cleanup error can leave some
inputs deleted and the complete merged output available. Destructive cleanup
requires stable, exclusively managed paths. The source's earlier deletion timing
and undefined parser inputs are retained in original evidence and qualified.

The [raw GGUF extension](GGUF_BYTE_STRINGS.md) now preserves complete byte
strings in the shared parser and raw writer. Native lookups retain first-prefix
matching; split/merge setters retain last-contribution ordering and first-prefix
tensor shape/payload selection. Every nonzero boolean byte is true, and writers
emit canonical 0/1 values. Independent per-profile coverage comprises 1,605
reader/getter cases, 269,184 exact native serialization bytes, 1,073 public-writer
inputs and 3,218 byte-string/NUL tool cases. The published 103-case streaming
corpus and its output/deletion/undefined-behavior qualifications remain intact.

The strict Gguf entry point retains its text contract. GgufSet now retains raw
primary metadata and a first-C-prefix model projection, including ByteString
values for non-UTF-8 payloads. Consumed settings and tensor names retain checked
boundaries. Dense Llama, Qwen2 and Qwen3 checkpoint conversion is implemented;
other converters, remaining raw-byte semantics and the rest of T04 stay open. All
twenty mandatory release gates remain open.

## Native CPU flash and checkpoint foundation

The shared Rust flash kernel preserves one-chunk, tiled and split-KV execution,
original query geometry, physical masks, unified row maps and measured compiler
NaN operand rules. Both compiler profiles independently pass 3,746 original
graphs and 7,060,000 raw default/forced output words. The tensor adapter adds
114 admitted original graphs and reuses 896 sealed cases, with 3,032,980 exact
output words per profile. Nine native-success adapter domains remain explicitly
unsupported. Worker stacks, scratch and owned results are checked before
execution; the tensor DAG still lacks an aggregate retained-memory budget.

The native C microbatch cohort has 228 requested cases: 226 source-return cases
match exactly, while two original initialization assertions become checked
errors. Per-profile comparisons cover 2,917,888 exact logit words and 14,171,045
binary bytes. Separate trained/tiny state replay covers all four K/V type pairs
under flash/nonflash, including 640 memory scenarios, 640 sequence states, 32
file suites and 1,536 short-buffer checks; all 8,459,616 binary bytes match. The
consumer rejects nonfinite computed model logits rather than claiming the
shared kernel raw-NaN domain as a model-level contract. Cancellation remains
whole-batch transactional. Decode sequence IDs follow n_seq_max; the measured
memory API retains its separate 256-ID domain.

The checkpoint crate provides raw Safetensors parsing, bounded positioned
reads and indexed composition over retained file capabilities. Original file
and index corpora retain platform behavior and safe checked path-policy
differences. Other architecture converters and the remainder of T04 stay open.

At that checkpoint, the combined guard covered every file outside historical
evidence and recognized Git/build caches, including prose, licenses, release
policy, source ledgers, declarations and all fixture types. Earlier 648/744/823/849-file subsets
retain their original meaning and are not described as complete guards. The
actual release list has twenty open rows across twelve ownership tracks.
Historical statements that called the track count a gate count are preserved
with a recorded erratum; no gate status changes with this wording correction.

## Memory and execution qualifications

The high-level model facade loads a dense Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, Mamba-1 or BERT GGUF file set into host memory. Non-matrix vectors are decoded F32. Matrices retain a decoded F32 row-access cache and, for supported non-F32 types, the original packed bytes needed for native dot arithmetic. The decoded-weight limit is not a total process memory limit. Loader hparams use the source UINT32/FLOAT32 types and 512-layer maximum; split.count uses UINT16 and accepts omitted, zero or one for a single file. Unsupported NextN/MTP, PRISM transforms, scaled RoPE and noncausal decoder modes return explicit errors. There is no mmap, tensor streaming, GPU model placement or full asynchronous scheduler claim.

The strict Llama, Qwen2, Qwen3, GPT-2 and GPT-NeoX decoder Context uses F32 KV. The separate Llama NativeContext admits independent F32/F16 key/value storage, native flash/AUTO resolution and actual configured workers for admitted dense-Llama CPU graphs; explicit nonflash execution remains single-worker. Mamba-1 owns F32 causal-convolution and selective-SSM state; cancellation and state restoration are transactional through the strict Rust Context API. Native C recurrent-memory compatibility remains separately open. BERT uses complete bidirectional sequences and its trained position table; it has no decoder KV context through this facade. Model instances are immutable and shared; each generation request owns independent context and sampler state. Stop strings and incomplete UTF-8 suffixes are withheld across token boundaries. Errors and cancellation do not commit partial model state. The C ABI documentation separately states accepted/rejected native parameter configurations.

The strict Rust Context defaults to a 1 GiB max_state_bytes budget. It reserves
owned sequence/layer records, KV and recurrent capacities, planning records,
transactional state and retained checkpoints before mutation. checkpoint() and
ContextCheckpoint::try_clone() return errors on allocation/budget failures.
Reservations survive a dropped context; cross-context restore charges the
destination. The contract excludes model weights, execution scratch, returned
logits/embeddings, caller buffers and allocator overhead. Serialization charges
the output while constructing it, then transfers ownership to the caller.

NativeContext uses a separate budget. Its earlier full-batch KV correction is
retained; review additionally accounted for temporary stream/chunk descriptors
and resident outer-vector storage. A regression rejects the old underestimated
budget atomically and succeeds at the sufficient budget; removing only that
descriptor charge makes the regression fail. Neither state budget is an RSS
limit. The original default combination of F16/F16 cache, AUTO flash and
four/four workers is now admitted for supported dense-Llama CPU configurations.
AUTO follows the frozen CPU capability resolution, and workers execute real
partitions. NativeContext keeps full physical K/V, complete logical masks and
unified row maps through per-layer microbatches. Nonflash multiple workers,
other native C context architectures and remaining backend modes stay open.

The CLI and server accept --ctx-size and --max-weight-mib, and the server
exposes RuntimeProvider::load_with_options for the same checked limits.
Generation currently supports greedy or temperature/distribution sampling with an explicit u32 seed. The presence of other sampler implementations in the behavior crate does not mean every upstream generation option is wired into the CLI/server.

## Validation and honest failure reporting

Current continuation logs are under `evidence/2026-11-11/`. The preceding continuation is under `evidence/2026-11-10/` and `evidence/hf-converter-20261110/`; earlier checkpoint logs remain under `evidence/2026-11-09/`, with imported archive logs under `docs/imports/`. Test outputs are associated with their command, platform and inputs. A test marked ignored requires an explicit external model, native reference library, installed compiler or vendor service. Ordinary workspace tests do not substitute for those explicit runs.

The earlier expanded checkpoint passed 384 ordinary tests on Windows and 385 on Linux in both debug and release, with 17 explicitly gated tests left out of ordinary execution. Seven pretrained tests run separately and pass on both platforms. The 195 fixture declarations, source tree, strict Clippy, offline release builds, 37 generation HTTP checks, embedding CLI lifecycle and SmolLM2/Mamba generation CLI checks also pass. The same 396 execution inputs have SHA-256 050cbc4b0c607a4be1bd7ade8d590f0b49b52b3cbbc284b12c8cfa5ed88413a9. The preserved audit/driver failures and same-fingerprint continuation are explained in evidence/2026-11-09/expanded-runtime/README.md.

After that complete run, independent review found and fixed an undercount of transient KV rows across later native microbatches. Only the NativeContext implementation changed. The incremental correction passes both profiles and strict Clippy on both platforms, with 64/62 focused Windows/Linux tests and 489,158 independent native float comparisons per profile plus state/metadata checks. The prior full-run fingerprint is retained; the corrected 396-input fingerprint is 26399fc8edf90a9c3871cea43e9212757d08094875bb046efe11898504ce2afa. See evidence/2026-11-09/expanded-runtime/final-checkpoint.json for the precise delta and the negative-control regression.

The exact published 6ddeb75eb3b708ace4cb48a365030f94a30192f3 checkpoint then
passed GitHub Actions run 42648885432: Windows 385 and Ubuntu 386 ordinary
tests in each profile, plus seven explicit pretrained tests on each host.
That CI result includes the final native-cache correction and is distinct
from the earlier complete-run/incremental local evidence above.

Earlier Qwen2/native-profile integration evidence is recorded separately under
evidence/2026-11-09/qwen2-and-native-profiles. Focused server verification passes
49 ordinary tests in each profile on each host, plus the explicit pretrained
BGE HTTP test (27 requests, 23,479 numeric comparisons, zero error). Its
567 execution inputs have the same normalized SHA-256
01acfdcf20cb878a2ab54444f8194fe782c7720e2b99a00da1a1d187937afd89
on Windows and GNU. The full integrated run and application results are recorded
in that evidence directory before checkpoint promotion.

The combined Qwen3/F16 checkpoint passes 451 ordinary Windows tests and 454
ordinary GNU tests in both debug and release, with 19 prerequisite tests ignored
in those ordinary runs. Nine explicit pretrained tests pass per host, including
Qwen3; every measured numeric comparison records zero observed error against
its corresponding original profile. Formatting, strict Clippy, 458 fixture
declarations, source verification, release builds, CLI and real HTTP/SSE checks
also pass. All 17 validation steps retain the same 637 execution inputs with
SHA-256 89a863d2145bee98206987803404e1551a9edb45c5174a9ff550cae993e42343.
The release check still fails as required because all twenty mandatory gates are
open. Complete evidence is in evidence/2026-11-09/qwen3-and-f16-caches.

The previous published e0af03a checkpoint passed GitHub Actions run 42673166488
on Windows and Ubuntu: 436/439 ordinary tests per profile and eight explicit
pretrained tests per host. Its first attempt failed before a runner started
with a GitHub internal server error; the identical commit/workflow passed on
attempt two. Both attempts and raw successful job logs are retained.

The combined GGUF streaming checkpoint passes 460 ordinary Windows tests and
464 ordinary GNU tests in both debug and release, with 19 prerequisite tests
ignored in ordinary runs and nine explicit pretrained tests passing per host.
Every measured pretrained numeric comparison records zero error. All 17 checks
cover the same 648 execution inputs, SHA-256
abd901487f73b01db4682a159c75b5a20c7e93b9e2c51cd81f2431923ca44a7c,
including strict Clippy, 465 fixture declarations, release builds, CLI,
HTTP/SSE and verification of all 4,082 frozen source files. The required release
check still fails because all twenty mandatory gates remain open.

Windows source verification first reported two generated Python cache files
from an independent reference import. Those files were archived outside the
frozen source tree, utility.py was compared to the authoritative ZIP, and
source/release checks then completed. GNU ordinary checks passed before a
deliberate resource pause during its release build; its remaining checks ran
under a separate continuation with identical inputs. Original failed/interrupted
status and logs remain preserved. No numerical expected data or source was
changed to complete either run. Full evidence is under
evidence/2026-11-09/gguf-streaming-tools.

The preceding exact published commit 1890f032 passed GitHub Actions run
42683437723 on its first attempt: Windows 451 and Ubuntu 454 ordinary tests per
profile, plus nine explicit pretrained comparisons on each host. Its complete
raw logs and run/job identities are retained with the next checkpoint.

The source audit checks implementation roles and dependency metadata; it reports unsafe review candidates instead of claiming every unsafe block is proven sound. Fixture integrity checks hashes, declared lengths, provenance fields and complete fixture inventory. Numerical tests check semantics separately.

`release/gates.tsv` is the active mandatory gate list. `cargo xtask release-check` must return nonzero until all full-port requirements have independent closure evidence. A green build or a checkpoint commit cannot close that gate.


## Explicit C split lists and byte-oriented filename helpers

At the explicit-split checkpoint the public C surface contained 187 of the
255 frozen exports, plus four diagnostic/layout extensions, with 68 absent.
`llama_model_load_from_splits` uses the existing owned GGUF file-set loader and
the current C model execution/vocabulary contracts. It retains every supplied
path before foreign progress callbacks, accepts raw Unix path bytes, and uses
checked UTF-8 paths on Windows. Automatic file loading also retains its path
before callbacks.

The frozen-header native corpus contains 104 Windows / 106 GNU explicit-list
cases: 55 / 57 accepted, with 49 checked errors on each host. Two errors per
host replace original source assertions. Successful cases compare model and
vocabulary getters, token bytes and 172,032 / 180,224 F32 continuation words
against their own original compiler profile. Those totals include repeated
model/list configurations. Original MSVC/GNU values stay separate.

`llama_split_path` and `llama_split_prefix` operate on C-string bytes without a
UTF-8 requirement. All 7,807 original calls and 506,563 complete output-buffer
bytes agree per profile, including signed minimum-width formatting, embedded
NUL termination, all input byte values, short/zero/null-zero buffers and the
source prefix helper's 32-byte local postfix truncation. Source signed overflow
and null pointer/count errors produce explicit checked diagnostics.

Both measured hosts run the FFI crate tests in debug/release, strict Clippy,
source/fixture audits, built-library C replays and exact 187+4 export checks.
Four isolated mutation controls exercise zero-based numbering, prefix return
semantics, actual caller-list use and ownership of every path before callbacks.
See `crates/ffi/SPLITS.md` and `evidence/2026-11-09/c-split-api`.

Twenty mandatory full-port release gates remain open across twelve ownership
tracks. Complete C ABI semantics, architectures, backends, platform coverage
and performance remain open. Earlier evidence and release-gate reason records
are retained with their original checkpoint counts.

## Static C enum names and load-mode parsing

At the static-enum checkpoint, the public C surface contained 191 of 255 frozen functions and four diagnostic
extensions; 64 functions remain unavailable. This checkpoint adds
`llama_ftype_name`, `llama_flash_attn_type_name`, `llama_load_mode_name` and
`llama_load_mode_from_str` as native Rust static helpers. Returning a mode name
does not enable that model-loading strategy or backend.

Original MSVC/GNU captures agree on all 73,057 file-type inputs, nine valid
enum-name inputs, 1,542 byte-string parser inputs and 43 guessed-prefix pointer
relationships. The file-type corpus contains 76 distinct result strings and
covers all unsigned 16-bit inputs, signed boundaries and additional 32-bit
flags and samples. Inputs stop at their first NUL without requiring UTF-8.
The source's 30 separately captured abort cases return checked Rust errors;
a null parser input is an additional checked foreign-pointer boundary.

`crates/ffi/ENUMS.md` documents ownership, diagnostic sentinels and the exact
scope. `fixtures/ffi/enums` contains independently captured expected values;
`evidence/2026-11-09/c-enum-api` records the validation provenance. Earlier
checkpoint counts and evidence remain historical. All twenty complete-port
release gates remain open.

## 10 November continuation

The recovered dense Llama HF converter is integrated as `llama-convert` and
`convert-hf-to-gguf`. Its ordinary suite retains all 64 original pipeline cases
per native profile: 57 complete files totaling 8,091,032 bytes and seven original
rejections. The independent schema and Transformers class corpora include
1,080 and 76 admission cases respectively. Fresh isolated Windows/GNU pretrained
checks reproduce three complete SmolLM2 outputs totaling 686,581,536 bytes per
host. This is the documented dense Llama conversion path; remaining architecture
converters and runtime scaled RoPE stay open. The five reviewed Rust YAML
packages require an initial locked fetch, followed by offline builds and an
exact archive/source/feature/license audit.

The C surface adds five model metadata functions and two chat functions, reaching
198/255 frozen exports plus four diagnostic extensions, with 57 still absent.
Raw primary GGUF metadata, the first-prefix model projection and the first-scalar
C map remain separate. Native map order and decimal strings follow the measured
MSVC/GNU policies. Their corpus retains 171 model requests, more than 334,000
buffer calls, 12,507 scalar entries, 3,020 tokenizations and 144,384 exact
continuation logit words per profile. Policy tests cover 282 maps through one
million entries and 222,262 raw float patterns per profile. See
[the metadata contract](../crates/ffi/METADATA.md) and
[component evidence](../evidence/2026-11-10/c-model-metadata/summary.json).

The C chat corpus compares 35,946 cases and complete output buffers against both
original hosts. The original frozen-header client also executes both debug and
release Rust libraries. See [the chat contract](../crates/ffi/CHAT.md) and
[component evidence](../evidence/2026-11-10/c-chat-api).

Review of the metadata integration caught a tokenizer behavior regression:
non-UTF-8 `general.name` bytes had silently lost Phi/ModernBERT name markers.
Byte-oriented ASCII matching now preserves the source first-NUL view. Separate
original controls retain 19,456 token attributes and 6,384 tokenizations per
host, including defined 0xff, marker precedence and whitespace effects.

These component results establish their stated scopes. The integrated workspace
verification is recorded separately; all twenty full-port release gates remain
open until their complete requirements are met.

The recurrent tensor extension adds SSM_CONV, SSM_SCAN, RWKV_WKV6, GATED_LINEAR_ATTN and RWKV_WKV7, reaching 81 executable IDs plus NONE, with 20 pending. Of 1,394 independently successful original graphs per host, Rust executes 1,389 with 1,278,140 exact output/state bytes and explicitly rejects five graphs whose competing NaN payloads depend on compiler optimization. All original outputs remain retained. This boundary is documented in crates/tensor/RECURRENT.md; it does not establish additional complete model architectures.
