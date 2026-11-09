# Tensor FLASH_ATTN_EXT boundary

This is an executable, bounded slice of frozen GGML operation 74. The maintained arithmetic and worker implementation is the single safe Rust module
[llama_quant::flash_attention](../quant/FLASH_ATTENTION.md). The tensor adapter
does not contain a second attention implementation.

## API and source fields

Use Op::FlashAttnExt with FlashAttentionOptions and explicit with_mask,
with_sinks and with_rows flags. Inputs are Q, K, V, then the enabled mask,
sinks and I32 row map in that order. Graph construction checks arity and graph
ownership; source constructor/layout and CPU support checks occur during
evaluation, before a result is published.

Q is [DK, queries, Q heads, batches]. K and V retain their full physical row
extents and original head strides. The result is contiguous F32 in exactly
[DV, Q heads, queries, batches] order. The output owns independent immutable
storage. Source tensor dimensions must fit signed I64; the nonempty kernel
also enforces its original signed-I32 CPU index domain.

FlashAttentionOptions preserves the raw signed source precision slot:
0 (default), 10 (F32) and 15 (BF16) invoke the same frozen CPU arithmetic.
A nonempty node rejects the other values, including source enum values
20/30/40. This field does not request an invented output conversion.
The nonnegative n_kv_max slot is retained even though the frozen CPU does not
read it. Its 0, conservative 1024 and I32_MAX cases have separate original
controls. Scale, max_bias, logit_softcap, worker count, MathPolicy,
CompilerPolicy and force_reference are carried by the shared Params value.
force_reference is the original graph-plan diagnostic switch, distinct from
precision.

## Input contract

The implemented nonempty arithmetic domain is F32 Q and independently
F32/F16 K and V. First-axis channel rows must be contiguous; all scalar byte
origins/strides must satisfy the shared view's alignment checks. Outer strides,
permuted head/query axes, grouped heads and nonzero view origins are retained.
Additional unused V rows are allowed. No input is decoded into a substitute
dense F32 attention calculation.

Masks are contiguous F16, including the original head/batch broadcasting.
The complete mask must cover every logical key column and query; extra mask
columns/queries are preserved. Positive max_bias requires a mask.

A row map is contiguous I32 [logical columns, Q batches, 1, 1]. It requires
a mask with exactly that column count and shared physical K/V batch 1.
Logical columns need not equal either physical K/V extent. Every nonnegative
map entry is checked against both physical extents after byte/work preflight,
including entries that are masked or in otherwise unused columns. The shared
kernel preserves negative-row skipping in the one-chunk path and row-zero
selection in the tiled path. Row maps disable split-KV, as in the source.

### Sinks use a physical flat prefix

The original constructor checks only F32 dtype and a first dimension equal to
the Q-head count. Every frozen CPU branch then reads the first head-count F32
words from sinks->data. It ignores the declared sink strides and trailing
dimensions. The adapter checks those constructor fields, checks the complete
physical prefix against the retained Arc storage, and gives the shared kernel
a normalized prefix view. It does not gather the logical sink elements.

This includes a zero-sized logical sink view with sufficient retained backing
bytes. Direct original strided, trailing-dimension, nonzero-origin and
empty-backed sink graphs agree exactly with separately allocated flat-prefix
controls. Insufficient retained backing returns Bounds. An empty output node
does not read sink storage at all.

### Empty output is a source dispatcher behavior

The original graph skips a node whose output is empty. The adapter therefore
returns a correctly shaped empty F32 tensor after source constructor checks,
without entering the shared kernel. Unsupported precision and nonfinite
arithmetic parameters are not evaluated for those empty outputs. Invalid
mask/sink dtypes and a negative n_kv_max still fail their constructor checks.
Worker-policy bounds and source signed dimension bounds remain checked.

This is distinct from a nonempty output with zero DK, physical KV rows or
logical row-map columns. The original executes those calculations; the
current shared kernel does not implement them. They remain checked errors,
with their original outputs retained as evidence of the remaining gap.

## Resource and cancellation contract

Limits applies to this operator's logical owned-allocation peak and work.
The shared kernel reports disjoint output, staged-result, worker scratch,
descriptor, requested stack and thread metadata categories. The adapter
additionally accounts for conversion into immutable Tensor storage.

For output payload B bytes, conversion has two phases: the F32 Vec plus a byte
Vec, then the byte Vec plus Arc storage. The F32 Vec is explicitly dropped
before Arc allocation. The conversion envelope is therefore 2*B + 64 bytes;
the 64-byte allowance covers the Arc header/alignment category. The operator
requires the maximum of that envelope and the shared-kernel execution peak.
It adds 3*B + 64 conservative copy-work visits to the shared-kernel work count.
Every count and sum is checked before kernel allocation or thread creation.

The caller's borrowed inputs, pre-existing caller stack and other retained DAG
nodes are outside this operator-local envelope. This is not graph-wide
allocation planning or a process resident-memory guarantee. Allocator/OS
bookkeeping is outside the logical categories. Fallible Vec reservations and
worker creation errors are reported; the existing Arc conversion uses Rust's
standard allocation failure behavior, so this API does not promise that every
process-wide out-of-memory condition becomes a recoverable Result.

For more than one configured worker the shared executor starts actual scoped
OS workers, joins all started workers on errors, and returns owned results.
The tensor adapter calls that executor, not the serial-partition diagnostic.
The one-worker case executes in the caller. A fixed one-MiB regression rejects
three requested worker stacks before execution, independently of the reported
estimator.

Cancellation is checked before/after the kernel and while serializing output.
There is no in-kernel cancellation hook. A cancellation during a kernel waits
for its scoped workers to finish, then discards the uncommitted result.
All input storage remains immutable; no partial result is published.

## Independent evidence

The new matching-host contract corpus contains 153 distinct isolated original
graph/constructor controls per host: 123 successful graph captures and
30 asserted/aborted source contracts. Of those successful graphs:

- 109 nonempty cases are admitted and compared at every output bit.
- Five empty-output cases preserve source skip behavior.
- Nine accepted original cases remain explicitly unsupported: three nonempty
  zero-extent domains, four BF16/packed-K dtype cases, and two nonfinite-scale
  cases. Their outputs are retained, not classified as passed comparisons.

Both default and forced-reference outputs for the 114 admitted cases contain
689,820 F32 words. Owner bytes, nonzero origins, byte strides, optional inputs,
source slots and output shapes are stored directly from original GGML.
A separate repeat of 27 original rejection controls saves all exact input
bytes before the abort, for ordinary safe-Rust rejection regressions. Three
empty-output constructor negatives are also checked as ordinary regressions.

The adapter additionally replays 896 already sealed original kernel cases
(320 dense operand-NaN cases and 576 row-map cases) through the DAG, comparing
2,343,160 default/forced words including NaN payloads. These are reused source
captures, not 896 new independent captures. The combined adapter byte gate is
3,032,980 F32 words per matching host and build profile. There is no tolerance
or finite-only exception.

The new fixture files are under tests/fixtures/flash-attention. The shared
replay imports the immutable original readers/fixtures from quant/tests.
Every native corpus was captured independently of the Rust adapter. Default
outputs and forced-reference diagnostics remain separate within the fixture
encoding. No original project library is a product dependency.

## Profiles and remaining coverage

Exact comparison applies to the captured x86_64 MSVC 19.51.36260.0 and
GNU 11.4.0 original builds, with Rust 1.99.1 on their respective hosts.
The unchanged shared module keeps compiler NaN-operand policy distinct from
scalar/SSE2 math policy. Native DLL/SO, source and capture-script hashes are
in the provenance. The shared quant handoff documents the exact flags,
libm/provider identities and floating-environment assumptions.
Other targets do not inherit an unmeasured bit-parity claim. Ordinary
native-fixture test modules are gated to the two verified target triples.

BF16 and quantized attention-cache arithmetic, nonempty zero-width attention,
NUMA scheduling, source profiles outside the measured compilers/ISAs and
unverified OS math providers remain open. General DAG scheduling, aggregate
buffer planning, writable aliases, device execution and FLASH_ATTN_BACK are
not supplied by this adapter. The frozen FLASH_ATTN_BACK constructor itself
aborts before its historical kernel can be used.
