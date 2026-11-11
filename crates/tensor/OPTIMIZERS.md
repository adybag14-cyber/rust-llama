# Native CPU optimizer steps

The Rust tensor engine executes the frozen OPT_STEP_SGD and OPT_STEP_ADAMW
arithmetic for checked F32 parameter rows. These are executable operations,
with independent comparisons of complete parameter and moment buffers.
They do not establish automatic differentiation, a training loop, general
writable backend storage, or the full optimizer/scheduler contract.

## Ownership and graph results

The original GGML operations modify weights and Adam first/second moments in
place. Rust tensors are immutable, reference-counted values. The Rust APIs return
owned replacements and preserve all input storage. AdamWState::step replaces
all three state tensors only after the whole step succeeds. An error, allocation
failure or observed cancellation leaves the state unchanged.

Graph::adamw_step creates an OPT_STEP_ADAMW node and two internal moment
projections. Its primary tensor has the original weight shape and strides; each
moment retains its own original shape and strides. Graph::evaluate_many computes
the union of requested dependencies once, including moments from the same step,
and returns results in requested order. Subsequent graph steps can consume all
three results without recomputing an earlier step. No result is returned if any
requested target fails. An ordinary Op::OptStepAdamW returns only the weight
result; callers retaining optimizer state should use the typed graph builder.

Op::OptStepSgd and the standalone sgd_step return an independent weight
snapshot. Input tensors used as parameters do not need a separate mutable
parameter flag: the typed optimizer constructor declares that role.

## Arithmetic and layouts

SGD reads two physical F32 parameter words: learning rate and weight decay.
AdamW reads seven: learning rate, beta1, beta2, epsilon, decay, beta1 correction
and beta2 correction. The caller supplies bias corrections, exactly as for the
original GGML operation. The step does not recompute them or introduce hidden
iteration counters.

Updates preserve the source F32 expression order, including separate weight
decay and the square root after second-moment correction. Parameters outside
usual training ranges are not silently clamped. The reference corpus includes
negative learning rates, unusual beta values, zero learning rate, signed zero,
subnormals, infinities and individual quiet/signaling NaNs. Results are compared
as raw words against each corresponding native capture. The two captured
profiles produced identical corpus bytes. This is not a universal NaN payload,
floating environment, ISA or optimized-kernel guarantee.

The source kernel uses the weight tensor's row offsets for gradients and moments,
even when those operands have different strides. The Rust implementation
preserves that physical addressing and checks that each operand contains every
access. Parameter tensors similarly use their physical prefix. Padding bytes
are copied and remain unchanged unless the original physical address calculation
actually updates them.

The admitted domain has aligned contiguous F32 axis-0 rows, nonoverlapping
monotonically separated row planes, source-sized integer dimensions, and
independent input buffers. Overlapping input storage, overlapping/permuted rows,
non-F32 operands and insufficient physical spans return checked errors.
Cross-input alias side effects and writable imported/device storage remain open.
Empty tensors return immediately after validation.

OptimizerLimits bounds logical element count and the total owned result bytes,
including padding and all three Adam state buffers. Defaults are 128 Mi elements
and 512 MiB of output. Explicit result-vector reservations are fallible;
cancellation is checked before work, at row/element intervals, and before result
publication. Existing Vec-to-Arc tensor ownership uses the standard allocation
boundary. The output limit does not bound temporary peak allocations or total
retained graph memory.

## Independent evidence

The tests/fixtures/optimizer directory contains separately captured Windows MSVC
and GNU results from the actual frozen public constructors and CPU graph executor.
Each profile has **300 cases** and **440,704 exact output bytes**. The cases
include one and seven repeated updates, one and three original workers,
four-dimensional and padded state, different operand strides, strided parameter
metadata and exceptional values. Rust exercises both the state API and chained
graph projections against those external results, then checks original inputs
are unchanged.

The first reference harness attempted to mark a GGML view as a parameter.
Both original profiles asserted that only a leaf can be marked. That failed
harness is retained outside the product. The corrected capture uses independent
owned leaf allocations, setting the public shape/stride descriptor within fully
initialized backing storage. No reference implementation is called by the Rust
product or its ordinary tests.

Separate tests cover cancellation, allocation/work budgets, malformed shapes,
physical bounds, alias rejection, multi-target identity/liveness and graph
failure behavior. Full integration results and the preserved reference-harness
failure are indexed under evidence/2026-11-11/optimizer-custom.
