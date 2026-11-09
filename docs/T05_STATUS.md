# T05 — Model and context implementation checkpoint

**Date:** 8 November 2026
**Status:** executable, independently checked component checkpoint; T05 is not complete.
**Crate:** `llama-model-runtime`
**Production dependencies:** Rust standard library only; `#![forbid(unsafe_code)]`.

The crate executes three deliberately bounded model slices. The other registered
architectures remain visible in the registry with no executable slice. The
registry is an inventory and dispatch guard, not a declaration of completed
model support.

## Implemented behavior

| Area | Implemented and exercised |
| --- | --- |
| Llama dense F32 | Validated hyperparameters and weight dimensions; tied or separate output; RMSNorm; separate Q/K/V with optional biases; grouped-query attention; ordinary full-head adjacent-pair RoPE; causal KV attention; SwiGLU FFN with optional biases; residuals; logits and normalized hidden embeddings |
| Mamba-1 F32 | Validated expansion factor 2; RMSNorm; input split; per-channel causal convolution and bias; SiLU; dt/B/C projections; optional dt/B/C RMS normalization; softplus time steps; selective state-space recurrence; D skip; gate; output projection; residuals; logits |
| BERT dense F32 | Token, position and optional row-zero type embeddings; embedding layer norm; bidirectional attention; post-attention and post-FFN norms; GELU FFN; token embeddings; mean/first/last pooling |
| Context | Shared immutable model weights; owned mutable sequence state; interleaved sequence batches; explicit microbatch splitting; external embedding inputs; per-token logits/embedding selection; finite-value and dimension validation |
| Transactions | Entire-batch commit after all microbatches succeed; cancellation before execution, microbatches, tokens, layers and commit; error/cancellation leaves every sequence unchanged and returns no partial output |
| Sequence operations | Independent prefix copies; removal/reset; exact attention-cache suffix truncation; owned checkpoints for speculative rejection including recurrent state |
| Adapters | Ordinary LoRA A/B matrix composition and upstream alpha/rank scaling; projection-name validation; multiple adapters on a target; post-layer control vectors for layer indices 1 and above |
| Adapter changes | Explicit `replace_adapters_and_reset` validates first, invalidates state and advances a configuration generation; failed changes preserve the old state |
| State persistence | Versioned little-endian progress format with shape/count/ID checks, bounded reads, finite-float checks, model/adapter fingerprints, CRC32 and atomic restoration |

The scalar math is an explicit validation executor pending integration with
T02/T03. It is not a second claimed implementation of their complete tensor
engine or optimized CPU backend. Configurations and weights are checked when
constructing a model; actual GGUF reading, tokenization and quantized weights
remain T04/T03 integration work.

## Independent numerical evidence

The optional **Rust** `reference_oracle` example loads a separately compiled
library from the original source tree and constructs the corresponding native
GGML operator graphs. The reference library is neither bundled nor linked into
the Rust production crate. Normal builds and tests work without that library.

The test used deterministic, synthetic, two-layer weights and tokens
`[1, 2, 3, 4]`. It compared **104 numerical values in 20 vectors**:

| Reference graph | Values compared | Largest observed absolute difference |
| --- | ---: | ---: |
| Llama embeddings and logits at each token | 44 | `1.788139343e-7` |
| Mamba-1 embeddings and logits at each token | 44 | `5.960464478e-8` |
| BERT final embeddings at each token | 16 | `1.192092896e-7` |

Acceptance is `abs(actual - reference) <= 3e-6 + 3e-6 * abs(reference)`.
The exact vectors are in
`crates/model-runtime/tests/data/ggml_reference.tsv`; comparison output is in
`docs/T05_NUMERICAL_EVIDENCE.txt`; input and source hashes are in
`docs/T05_PROVENANCE.json`.

The independent comparison found a real implementation detail: the frozen
CPU's F32 GELU uses an F16-indexed and F16-valued lookup table between -10 and
10. A direct F32 formula differed measurably. The Rust path now reproduces the
half rounding of the table entry and the source's edge behavior. The tolerance
was not relaxed to hide that difference.

These are numerical **operator-graph** checks for the listed slices. They do
not prove full original model-loader/context equivalence, every model option,
or meaningful inference with actual pretrained weights.

## Runtime and state validation

**35 tests passed**: 3 internal numeric/CRC checks, 3 independent captured
reference tests, and 29 runtime/state/adapter tests. Scoped formatting and
`cargo clippy --all-targets -- -D warnings` passed. Exact command results are
saved in `docs/T05_VALIDATION.log`.

The tests cover microbatch versus single-token equivalence; interleaved
sequences; prefix copies; attention suffix truncation; recurrent checkpoint
rollback; late errors after prior microbatches; cancellation inside a layer and
just before commit; invalid positions, IDs and capacity; output selection;
external embedding validation; exact state resumption; truncated/corrupted
state; forged dimensions and non-finite state; different model/adapters;
independently calculated LoRA deltas; control-vector changes; malformed model
weights; arithmetic overflow; BERT bidirectionality/pooling; and a one-tap
Mamba convolution with optional dt/B/C normalization.

Peer review also added checks for malformed public dimension helpers (including
zero heads and overflowing raw dimensions), noncanonical LoRA layer spellings
that previously validated but did not activate, and overflow in BERT mean
pooling of otherwise finite token vectors. Dimension helpers now return
`Result<usize>`, adapter names must match canonical layer indices, and pooling
uses F64 accumulation with a finite-result check. Mamba has no public unchecked
dimensional helper; its constructor rejects malformed dimensions before state
arithmetic.

The state format starts with `T05STATE`, version 1. It is explicitly **not** the
original llama state-file ABI. Model/adapter fingerprints are non-cryptographic
FNV-1a values, and CRC32 detects accidental corruption; neither authenticates
an adversarial file. Lengths and dimensions remain independently checked.

## Registry reconciliation

The executable registry audit was run against the supplied frozen source.

| Item | Observed count / finding |
| --- | --- |
| Architecture enum slots | 173, preserving exact order and numeric IDs |
| Named entries | 172; `(unknown)` occupies ID 172 |
| Canonical name entries | 173 |
| `src/models/*.cpp` | 176 |
| Explicit original model factory cases | 171 |
| Python text registration literals | 312 |
| Python projector registration literals | 92 |
| Total conversion registration literals | 404 |
| Rust executable slices in this checkpoint | 3; full family acceptance remains open for all three |

**Frozen-source findings:** GPTJ is named at enum ID 8 and has a source file,
but the original `llama_model_mapping` switch has no GPTJ factory case.
`qwen3tts.cpp` contains a comment rather than a separate graph implementation;
`llama_model_qwen3tts` inherits `llama_model_qwen3vl`. Its registry mapping
records both files. Helpers such as `mamba-base.cpp` stay in the complete
176-file inventory and are not mislabeled as separate architectures.

**Plan discrepancies:** the supplied plan reports 314 text / 91 projector
conversion aliases. Parsing actual Python class decorators, resolving their
inherited `model_arch` and TextModel/MmprojModel lineage, gives 312 / 92, with
404 literal registrations total. The Rust audit independently rechecks the
exact source-file/alias pairs; this discrepancy is retained openly. No input
was replaced or changed. These aliases are data only, not Rust converters.

The complete per-path ownership ledger contains **218 T05 paths**, matching
the plan and including `src/models/llama.cpp`. The unrelated
`src/llama-version.h.in` remains T12 build/version metadata. The architecture,
factory and conversion registration counts above are separate inventories.

## Build and reproduce

With the supplied Rust 1.99.1 binaries on `PATH`, from the Rust workspace:

```text
cargo test -p llama-model-runtime --all-targets --offline
cargo clippy -p llama-model-runtime --all-targets --offline -- -D warnings
cargo fmt -p llama-model-runtime -- --check
cargo run -p llama-model-runtime --example smoke --offline
cargo run -p llama-model-runtime --example registry_audit --offline -- /path/to/llama.cpp-master
```

For the optional independent native-reference comparison on Linux x86-64,
provide the original source tree separately and an output directory outside
both source trees. The Rust build helper invokes installed gcc/g++ only for
the comparison oracle and emits a complete reference build log:

```text
cargo run -p llama-model-runtime --example build_reference --offline -- /path/to/llama.cpp-master /separate/reference-build
cargo run -p llama-model-runtime --example reference_oracle --offline -- /separate/reference-build/libggml-reference.so
```

The checked reference configuration used gcc/g++ 14.2.0, `-O1`, standard x86-64
compiler defaults and one GGML compute thread, with no AVX-specific flags or
vendor math libraries. The supplied Rust compiler is
`rustc 1.99.1 (58cf8f9c3 2026-10-13)`.

The Rust `build_reference` helper was itself executed successfully into a fresh
separate directory. Running the oracle against that rebuilt library reproduced
the saved TSV byte-for-byte, including all 104 reference values.

## Explicitly unfinished work

- Full architecture graphs, graph caching and T02 graph/scheduler interfaces;
  trained-model fixtures and end-to-end tokenization/generation.
- Quantized weights; model-specific nonstandard head layouts; fused QKV;
  K/Q normalization; MoE/expert offload; nonstandard/scaled RoPE;
  nonstandard cache types; recurrent/hybrid architectures beyond Mamba-1.
- Original `llama_batch` and extended API compatibility, shared multi-sequence
  token entries, encode/decode/process public ABI, attention configuration and
  backend sampler attachment.
- Sliding-window, DSA/ISWA/MSA/DSV4, indexed/hybrid and MoE caches; position
  shifts; original state-file interoperability; bounded copy-on-write state.
- LoRA file loading, embedding LoRA, aLoRA, runtime LoRA for BERT, fit planning,
  lazy/offloaded experts and backend placement.
- BERT classification/reranking heads, specialized pooling, other encoder
  families, audio/diffusion/encoder-decoder graphs, and model-family variants.
- Optimized memory use, threading, device execution and performance evidence.

Ordinary models outside the three listed slices are rejected explicitly by
`require_execution_slice`; no generic decoder silently stands in for them.
