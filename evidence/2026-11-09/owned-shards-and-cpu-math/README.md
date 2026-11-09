# Owned shards and shared native CPU mathematics

This checkpoint combines bounded GGUF shard ownership, buffered header reads,
same-owner C model loading/statistics, and one maintained CPU row implementation
shared by tensor and model inference. It remains a partial native Rust port:
all twelve full release gates remain open.

## Product changes

The GGUF loader validates numbered or explicitly ordered shards, retains one
owner per parsed file, indexes all tensors and reads primary metadata from the
first file. It enforces cumulative header/value/tensor budgets, checks global
tensor-name uniqueness and uses each shard's own data offset and endianness.
A bounded 64 KiB fallible header buffer reduces read calls without changing
the public file owner or the logical reader position. The negative allocation
controls and the original single-run performance observation are retained.

Existing C model loading now obtains metadata, size, parameter count and
guessed file type from the same validated shard set used for model execution.
Vocabulary-only loading also validates the complete declared set. No explicit
split-list C entry point or new exported function is claimed.

Tensor and model inference share safe scalar/SSE2 source row arithmetic.
This improves SiLU, quick GELU and their GLUs, softmax and contiguous
normalization, including measured NaN operand selection and odd packed GLU
widths. It adds no operation IDs: 75 executable IDs plus the NONE leaf are
covered; 26 remain pending. The accumulated corpus has 3,240 distinct inputs
and 6,118 native captures.

The explicit platform boundary calls the installed operating-system float erff
on the two exercised dynamic-CRT targets. It does not link llama, ggml,
a native oracle, a C shim or vendor math code. Other/static targets retain the
documented mathematical fallback. Quant remains dependency-free. The only Cargo
change is tensor's target-specific edge to the existing local platform-loader
crate and its corresponding lockfile entry.

The captured UCRT and glibc identities, real Rust PE/ELF import tables and
provider-qualified numerical claims are retained. CI now hashes the installed
provider file with the existing Rust xtask. That file identity is additional
environment evidence; it is not a proof of dynamic module resolution.

## Independent comparisons

- Sharded model corpus: 90 configurations, 65 accepted sets, 25 rejected sets,
  64 model executions, 464,272 bitwise-equal float values and 590 greedy IDs per
  native profile. The ordinary regression now requires exact bits on the two
  measured targets; the pre-existing portable tolerance remains elsewhere.
- C shard corpus: 96 cases, 53 accepted and 43 rejected, 163,840 exact logit
  words per profile, complete getter statistics and vocabulary-only behavior.
  Two original assertion aborts become checked errors. The ordinary test is
  gated to the two measured targets.
- Shared CPU row corpus: 935 original graphs plus 72 odd packed GLU graphs,
  2,909,056 output bytes per host, all exact including NaN payloads.
- Scalar functions: 502,058 original same-host quick-GELU words and 502,058
  independent OS erff words per host. The existing 96,910 high-precision
  mathematical ERF checks remain unchanged.

Windows and GNU expected results remain separate where the originals differ.
The failed cross-CRT quick-GELU assertion, early tensor mismatches, adapter
failures, and allocation/C-loading negative controls are preserved. No original
expected value or numerical budget was widened to hide a mismatch.

## Combined validation

All 606 execution inputs have the same SHA-256 aggregate on both hosts:
03365c9600e8ad652a4921a1ca0d0e8288f76cf1bb655ae22e8bf700610f4bfe.
The exact input inventory is retained for each run.

The full-v1 stage passed formatting, complete debug/release tests, strict
Clippy, source/dependency audit, installed-provider hashing, all 433 fixture
declarations, release builds, embedding and decoder CLI checks, and Qwen2 HTTP
checks. Its external shard HTTP driver then failed because it requested
normalization mode 0 while comparing raw C API BERT means. Both failures and
the original driver identity remain intact.

The separate completion-v2 stage changes only the external driver's output
directory and normalization selection to the already implemented mode -1.
It repeats the affected application check, then runs the remaining pretrained,
generation HTTP, frozen-source and release-gate checks. It verifies the same
606-input fingerprint before every step and after completion. Rust source,
native references, comparison rules and tolerances do not change.

The shard application driver checks eight CLI runs and nineteen HTTP requests
using original C API outputs from metadata-only first shards. This is model
behavior plus the existing HTTP contract; it is not a new original shard HTTP
capture. The full original embedding/generation HTTP corpora remain separate.

The full release command is expected to return failure because the twelve
project-wide completion gates are still open. Treating that expected refusal
as a successful validation step does not mark any gate complete.

## Evidence layout and provenance

- integration: immutable combined-stage reports, logs, application transcripts,
  preflight formatting failure, root adaptations and external driver hashes.
- shards: original focused validation, input and fixture identities, allocation
  controls and qualified header-I/O measurement.
- c-shards: original C harness reports, real binary export checks, ordinary
  regressions and the deliberate old-loader negative control.
- cpu-math: native numerical validation, linkage evidence, provider identity,
  scalar controls and preserved failures.
- previous-checkpoint-ci: raw successful Windows/Ubuntu job logs and API
  results for ff898aadc24961afc77711e1a690d75458b1f155, run 42666469417.
  That CI tested 417 Windows and 419 Ubuntu ordinary tests in each profile,
  plus eight explicitly enabled pretrained tests per host.

source-provenance.json records the exact origin, byte length and SHA-256 of
every copied artifact. Original C/C++ sources, binaries, full external raw
captures and Python validation drivers remain outside the product; excluded
files encountered while packaging retain identities in the provenance index.

Qwen3, typed F16 KV state, flash attention and GGUF split/merge tooling are
separate workstreams and are not claimed by this checkpoint.

The first final packaging audit rejected three unclassified .patch filenames.
Their exact bytes are now archived as .patch.txt; the original handoffs and
audit policy are unchanged. The failed final-audit logs and this filename-only
adaptation are retained under integration/validation and integration.
