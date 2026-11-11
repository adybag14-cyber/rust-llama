# Rust custom tensor callbacks

The tensor graph executes MAP_CUSTOM1, MAP_CUSTOM2, MAP_CUSTOM3 and CUSTOM
through owned Rust callbacks. This is the out-of-place custom-operation family.
Mutable alias/in-place custom callbacks and the complete backend scheduler remain
open. No arbitrary C function pointer or raw userdata lifetime is fabricated.

## Callback contract

CustomKernel owns a Send + Sync Rust callback and its captured userdata.
Cloning a kernel shares that owner; graphs retain it until their nodes are
dropped. A CustomTask exposes the actual worker index and task count, immutable
input tensors, checked physical-byte reads, the output descriptor, checked
output reads/writes, and a cooperative cancellation check.

MAP_CUSTOM1/2/3 derive a contiguous output from the first input's dtype and
shape, matching the original duplicate-tensor constructor. CUSTOM declares its
own dtype and four-dimensional shape and accepts zero through nine inputs,
matching the original public constructor's input limit. Block-type output
geometry is validated by the normal tensor descriptor contract.

Input physical reads include padding and respect a tensor view's offset and
span. Writes never reach an input buffer. The output is separately owned and
synchronized; byte coverage is tracked. Reading an unwritten output byte or
returning without writing every result byte is a checked error, rather than
exposing the original allocator's uninitialized memory.

The task count is min(requested tasks, configured workers), with -1 selecting
all configured workers. Zero/other negative task counts and invalid worker
counts are rejected. At most 64 actual workers may execute one operation.
They are real scoped threads, with worker zero on the calling thread. A launch
gate prevents callbacks from starting with only a subset of workers if thread
creation fails. Every started thread is joined on success and failure.

An empty output invokes no callbacks, preserving the original CPU executor's
observed empty-node suppression. It does not create worker threads.

## Failure and resource behavior

CustomOptions defaults to one worker and a 256 MiB output bound. Explicit output
and initialization-map vector reservations are fallible; final Vec-to-Arc tensor
ownership uses the existing standard allocation boundary. The output limit does
not include the initialization bitmap, thread stacks or temporary peak storage.
Callback panics, callback errors,
out-of-range accesses, uninitialized reads and incomplete results return errors.
No partial result tensor is published.

Callbacks are ordinary trusted caller code and must cooperate with cancellation.
Their external side effects cannot be rolled back by the graph. Blocking or
nonterminating user code is not sandboxed or forcibly interrupted. Deterministic
parallel callbacks should use disjoint output regions and must coordinate any
caller-owned shared state. These rules are not a claim of general parallel DAG
scheduling or an aggregate graph-memory budget.

## Independent verification

The original public constructors and CPU graph planner were exercised on the
frozen Windows MSVC and Linux GNU builds before the Rust callback implementation.
Each capture contains **260 cases** and **30,064 exact output bytes**. It checks
all four operation IDs, F32/F16/I32/Q4_0 layouts, padded sources, zero through nine
generic inputs, task capping, actual distinct native worker threads, source
pointer order, userdata, and empty output behavior.

The reference callback itself uses a byte/integer transform chosen as input code;
GGML supplies its dispatch, task counts, descriptors and source pointers. The Rust
test performs the same independent transform through the safe task API and
requires exact stored native outputs and invocation records. Both native
profiles produced identical binary corpus bytes. The Rust test also proves
distinct actual worker threads and one execution when a graph target is requested
twice through evaluate_many.

The initial capture incorrectly expected empty-output callbacks to run. Both
native profiles disproved that expectation; the failure is preserved and the
corrected corpus explicitly records zero invocations. Separate checked tests
cover callback ownership, panic/error cleanup, all-worker joining, cancellation,
bounds, byte initialization and resource limits.

The data and provenance are in tests/fixtures/custom. Integrated results and
reference-harness diagnoses are under evidence/2026-11-11/optimizer-custom.
