# Qwen2, native CPU profiles and tensor image operations

This is an incremental checkpoint of the incomplete Rust port. It adds dense
Qwen2 inference, bounded strict-context state ownership, the measured scalar/SSE2
model arithmetic, source physical attention rows, thirteen tensor operation IDs,
and application resource controls. All twelve full-port release gates remain
open. Sharded GGUF, typed F16 native KV and flash attention belong to separate
unmerged work at this checkpoint.

## What was executed

The complete integration before the final pooling correction was tested in
Windows MSVC and Ubuntu GNU debug and release builds. The immutable execution
input set contains 567 files and has SHA-256
`b526251fe952ed9d59957918b9d24cdb21adb9f498c51cdf43d995fc3e673031`.
Every step checked that this fingerprint remained unchanged.

| Verification | Windows | GNU/Linux |
| --- | --- | --- |
| Complete workspace ordinary tests, each profile | 416 passed, 0 failed, 18 ignored | 418 passed, 0 failed, 18 ignored |
| Explicit pretrained tests | 8 passed | 8 passed |
| Formatting and strict Clippy | Passed | Passed |
| Dependency/source audit and 403 fixture declarations | Passed | Passed |
| Release workspace build | Passed | Passed |
| Embedding CLI, decoder CLI and Qwen2 HTTP drivers | Passed | Passed |
| Real-model generation HTTP | 37 requests passed | 37 requests passed |
| Frozen source verification | All 4,082 files verified | All 4,082 files verified |
| Full release gate | Rejected as expected: 12 gates open | Rejected as expected: 12 gates open |

The complete base results and logs are under
[integration/validation/full-v1](integration/validation/full-v1). The application
transcripts are under
[integration/validation/applications-v1](integration/validation/applications-v1).
The driver hashes and exact commands are recorded in the platform status files.
Test fixtures are selected by target independently of runtime math dispatch.

### Final pooling delta

Review then added twenty original GGML graphs with overlapping backward-pooling
gradients containing quiet/signaling NaNs, infinities and signed zeros. The old
GNU average-backward implementation differed in ten cases and 66 output words.
The frozen GNU kernel keeps the accumulated NaN; the frozen MSVC kernel and both
max-backward kernels select the incoming gradient payload. An explicit ordered
addition now preserves those observed policies; finite arithmetic is unchanged.

The final source/data input set contains 572 files and has SHA-256
`faf38b62cab6b455fe1267501d3b8a41ed3a944a619af58d1dc40112fa97d083`.
Both hosts subsequently ran the complete tensor suite in debug and release
(26 passed per profile), formatting, strict Clippy, audit and 407 fixture checks.
The new comparison module is gated as a whole to the two verified x86_64 targets,
so an unsupported platform cannot report an empty passing native comparison.

[The merge record](integration/pool-delta-merge.json) records every preimage,
unchanged expected fixture hash and the module-gating adaptation.
[The focused final runs](integration/validation/pool-delta-v1) apply to this
572-file snapshot. The complete workspace/pretrained base run is intentionally
reported separately; it is not relabeled as a complete final-source run.

## Model and application findings

All exercised model outputs match their corresponding original build exactly.
The explicit corpus includes 62,976 F32 Llama logits and 96 greedy IDs, 640,000
unique Q4_0 Llama logits and 16 IDs, 1,671,168 unique SmolLM2 logits and 24 IDs,
1,709,520 unique Mamba-1 logits and 24 IDs, and 5,165,824 unique Qwen2 logits and
24 IDs. BERT compares 13,056 raw/pooled values in both endian variants.
The eighth pretrained test compares the BGE HTTP contract: 27 requests,
23,479 numerical comparisons and three base64 payloads.

Twenty-one synthetic Qwen2 variants cover separate/fused projection weights,
partial bias resolution, tied/untied output and packed types. Thirteen physical
metadata variants preserve original resolution rules. Earlier partial-fused-bias
and output-overflow failures are retained. Atomic rejection replaces a nonfinite
output-producing path.

The original MSVC and GNU builds use different row arithmetic. The verified
MSVC profile uses scalar source rows; the GNU profile uses source SSE2 row
exponentiation and four-lane reductions, including centered BERT variance.
The product follows each measured profile and retains original physical
attention columns, holes and 256-column padding. Cross-platform original values
are not forced into a single expected result. This does not certify other ISAs,
compilers, floating-point environments or CPU schedules.

The server now accepts checked context and decoded-weight limits, with matching
CLI options. Default Qwen loading exceeds the 2 GiB decoded-weight allowance and
is rejected; the pinned Qwen corpus loads with a 3 GiB allowance and a 128-token
context. These limits are not total process-memory limits. Five startup cases
and fourteen Qwen HTTP requests include text, token IDs, streaming and clean
shutdown; five decoder CLI cases cover SmolLM2, Mamba and all three Qwen prompts.
Separate embedding drivers cover EOF and lifecycle behavior.

The GNU embedding HTTP fixtures were captured independently from the original
server using the same inputs and predeclared budgets as the Windows corpus.
Each original profile contains 283 requests: 231 successes and 52 HTTP 400
responses. Corresponding Rust target results match all measured F32 bits and
all seven base64 vectors exactly. The earlier cross-profile differences remain
documented, including the unchanged norm-zero propagated-error qualification.

## Allocation and tensor boundaries

Strict contexts now account for their owned sequence/layer capacities, retained
KV/recurrent buffers, staged transactions and checkpoint lifetimes before
mutation. Checkpoint creation and cloning are fallible; reservations remain
charged while a checkpoint outlives its context. Restore charges the destination
and validates before floating-point buffer allocation. Weights, scratch,
returned outputs, allocator overhead and caller-owned buffers are outside this
specific state limit.

The native-context descriptor correction additionally charges outer stream/layer
descriptors and physical query planning. Its 512-token regression rejects a
32 KiB temporary allowance and accepts 64 KiB; a negative control removing only
the descriptor charge fails. The prior negative-control output is preserved.

Tensor coverage is 75 executable frozen operation IDs plus NONE, with 26 IDs
pending. The convolution/extraction/pooling/resize addition comprises 981
distinct graphs, 530,246 exact output bytes per host, original output shapes and
strides. The earlier 961-graph cohort has identical original bytes on both hosts;
the twenty follow-up gradient graphs retain their measured platform differences.
The complete tensor corpus has 2,233 distinct graph inputs and 4,104 native
captures. Platform recaptures are not counted as new distinct graphs.

General tensor row/ISA dispatch and exact platform `erff` remain separate work.
This checkpoint does not change the earlier tensor transcendental criteria or
replace its independent high-precision mathematical ERF oracle.

## Evidence integrity and failures

No native expected data or numerical budget was adjusted to match Rust output.
The following failures remain available in this package:

- Qwen partial-bias and finite-input/output-overflow failures.
- Earlier GNU BERT variance and model-profile mismatches.
- The strict-state/descriptor allocation negative controls.
- The original pooling adapter enum mistake, kept separately from the actual
  corrected-adapter GNU mismatch.
- The integration preflight that rejected an outdated fixture README hash,
  followed by the explicit metadata hash correction. No numerical fixture changed.
- Earlier formatting, compilation and lint diagnostics from the isolated work.

The root README's en dashes were also restored after a Windows text-decoding
error in a documentation-only edit; the final write uses explicit UTF-8.
The file change is recorded with the pooling documentation update.

The original ZIP, reference C/C++ builds, executable adapters and model downloads
remain external to the product. Committed raw numerical fixtures and provenance
are under the relevant crate and inference fixture directories. The evidence
package records original paths, sizes, SHA-256 hashes, copied logs and any
external-only artifacts. Original implementations are not loaded by Rust model
execution.

The included successful CI report for commit
`6ddeb75eb3b708ace4cb48a365030f94a30192f3` belongs to the previous checkpoint.
It is historical evidence, not a claim that the new commit has already passed CI.
The new commit's exact-head CI must be checked separately after publication.
