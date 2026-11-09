# Frozen native CPU microbatch execution

This extension implements the measured dense Llama CPU context path behind the
existing C exports. It adds no C symbols and does not change public C layouts.
The C execution architecture remains Llama; high-level Rust inference support
for other architectures does not imply native cache or C context parity.

The qualified profiles are x86_64 Windows/MSVC and x86_64 Linux/GNU, using the
frozen source/compiler math policies documented in the shared quant kernel.
Flash rejects other targets. CPU native model contexts, finite numerical
execution and default environment dispatch are the admitted domain. Native
environment overrides, SWA, recurrent/hybrid memory, GPU devices and quantized
cache storage require separate implementations and proofs.

## Constructors and source dispatch

The safe runtime exposes:

- NativeContext::new: unchanged F32/F32, nonflash, one-worker behavior.
- NativeContext::new_with_cache_types: unchanged independent F32/F16 resident
  types, nonflash, one worker.
- NativeContext::new_with_execution: explicit storage types plus
  NativeExecutionOptions { flash_attention, threads, threads_batch, embeddings }.

The strict Context API and its versioned checkpoints are separate. Refactored
Llama projection/finish helpers retain its existing sequential ordering.

The C facade preserves every default field. AUTO resolves to enabled for the
admitted ordinary CPU Llama graph because the frozen supports_op probe accepts
and assigns that attention node to the same CPU. Original AUTO/ENABLED/DISABLED
captures include independent K/V pairs and head widths 4, 6, 32 and 128, in
addition to the two-element-head synthetic fixture and pretrained model.
Untouched context defaults use F16/F16, four workers, separate cache and n_ctx512.
CPU model loading restrictions remain in README.md.

Nonflash multiple-worker execution remains a checked error. AUTO is not an
alias for nonflash. Explicit ENABLED is not downgraded. A batch/context limit
that would trigger the original output-row assertion is rejected during C
initialization with a diagnostic.

## Batch ordering and layer lifetime

Unified input uses the original simple contiguous split. Separate input uses
the source equal-sequence split, or its per-sequence split in embeddings mode.
The planner retains the measured n_ubatch+1 boundary behavior for very small
limits. Input order, sparse output flags and original batch indices are
preserved independently of execution order.

For each native microbatch, all layer projections and all current-layer K/V
writes finish before attention starts. Attention therefore sees every source
physical slot, including equal-position keys. Hidden and Q matrices coexist
until the layer attention result is available. The Q allocation is dropped
before the returned attention values replace its work allowance.

A newly empty sequence may start at a nonnegative position; an existing one
must resume after its stored maximum. Within the submitted sequence, positions
are consecutive and nondecreasing, allowing repeats. Token sequence IDs must be
unique within the token and below configured n_seq_max. Unified shared tokens
require coupled sequences with matching stored position ranges. Separate-mode
tokens naming multiple sequences remain a checked error. Memory API IDs retain
their separate 256-ID contract.

## Physical cache and fused row maps

A physical attention row is determined by the highest occupied cell across
selected streams, rounded to the source 256-column boundary and capped at cache
capacity. Visible-token count is insufficient: prefix/middle holes and hidden
cells keep their original physical columns.

Nonflash attention preserves the full physical row through score/probability
calculation and PV reduction. Its F32 right operands are not half-rounded.

The source flash graph uses F16 K/V views. Resident F16 bytes are copied without
arithmetic conversion; resident F32 values receive the actual source graph's
F16 cast. Every physical lane is retained, including masked holes. Mask values
and complete query/batch geometry are passed to the shared kernel.

Unified flash uses the original fused row map. Contiguous equal-length runs of
one sequence produce logical query slices; shared or ragged sets use one union
slice. The logical width first counts selected sequence memberships, so a
shared physical cell contributes more than once to the source padding bound.
It is then clamped by the physical used maximum and padded. The map itself
lists matching physical cells once each, in ascending order, followed by -1.
Thus 261 unique selected rows can require a 768-column map. A direct source-5
buffer regression protects that distinction.

The shared Plan::new_with_rows receives the complete map and mask. Physical K/V
retain batch extent one. Row maps disable split-KV but retain source tiled
dispatch. No dense substitute or compacted K/V approximation is used.
crates/quant/FLASH_ATTENTION.md documents the independently captured one-chunk,
tiled, split and negative-row arithmetic, including exceptional raw bits.

## Configured workers

The microbatch's actual token count selects threads or threads_batch.
Generation and batch values are independently bounded to 1..=256. A single
worker executes on the caller; multiple workers execute source partitions on
actual scoped OS threads. The coordinator joins every started worker and
combines split partials in source order.

NativeExecutionStats reports executed microbatches, layer flash graphs,
completed worker partitions and distinct threads executing those partitions,
summed over the call. It does not claim to reproduce the original persistent
threadpool or scheduling overhead. Worker changes are validated before
mutation. Invalid changes preserve the prior configuration.

## Owned working-set accounting

The two-GiB decode ceiling is an explicit implementation limit. Checked
arithmetic and fallible reserves cover the allocations created by this path.

| Live category | Bound charged before execution |
|---|---|
| Original and transaction cache | Two resident storage ledgers, including typed tensors, cells and descriptors |
| Planner and slots | Complete batch index storage, microbatch descriptors, slot indices and temporary slot search |
| Hidden and query matrices | Two F32 matrices for the maximum actual microbatch |
| Flash physical casts | Both complete F16 K/V views for selected physical streams |
| Flash mask | F16 physical/logical columns for every current query |
| Unified row maps | I32 full-capacity maps for the maximum logical slice count across every microbatch |
| Projection and outputs | Bounded per-token projection/FFN/vocabulary temporaries plus retained outputs and descriptors |
| Shared kernel | Remaining allowance passed to the kernel's checked output, partial, scratch, descriptor, thread-stack and bounded metadata requirements |

The kernel does not charge the borrowed caller views again. Model storage,
caller-owned input and the existing caller stack are outside this ledger.
The standard library/OS is not treated as a process-wide allocator with an
exact resident-memory guarantee; explicit worker-stack requests and the
documented bounded thread metadata allowance remain charged.

The fixed two-MiB regression uses a 16,384-cell fixture whose simultaneous
payload floor is 132 bytes/cell on the measured targets, before descriptors.
It rejects without committing state. The retained old-formula negative control
omits the row-map charge and is caught by the same independent ceiling.
Separate negative controls detect unique-cell padding and missing decode-ID
validation.

## State, failure and compatibility limits

Nonflash state stores values in transposed order (v_trans1); flash stores
untransposed physical value rows (v_trans0). The destination must match the
encoded layout and independent K/V types. Raw F16 signaling-NaN payloads
survive non-arithmetic state operations. STATE.md specifies write boundaries,
restore ownership and session side effects.

NativeContext rejects nonfinite computed attention/hidden/logit results with a
checked error. The shared primitive's raw NaN corpus is not presented as a
claim that the model context emits the original exceptional logits. Invalid
decode, resource rejection and cancellation leave previous context state,
execution statistics and C output buffers unchanged. This whole-batch atomic
cancellation is stronger than the original's possible partial microbatch
commit and remains an explicit behavioral difference.

The compact ordinary replay covers 205 tiny/generated-model inputs, plus two
checked replacements for original initialization assertions. The full external
C replay includes pretrained inputs and all 228 requests. Expected bytes were
captured from the original libraries before consumer implementation and remain
separate for MSVC and GNU. Five direct map geometries and the original
microbatch-order corpus protect intermediate layout/selection behavior beyond
final logits. Earlier dense-only and nonflash checkpoints, native assertions,
a corrected test-input recipe and infrastructure queue expiry are retained in
the validation evidence.
