# Dense F32/F16 native memory and state compatibility

This slice implements the frozen Llama dense F32/F16 cache contract through
llama-model-runtime::NativeContext and 28 memory/state/session C exports.
It is separate from the safe sequential Context API, its checkpoints, and its
versioned Rust state format. No original C/C++ object is linked into the product.

The frozen source archive SHA-256 is
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
Compatibility is tied to that source, not arbitrary later llama.cpp states.

## Memory operations

llama_get_memory borrows the context's cache. The pointer remains valid until
the context is freed and must not be freed separately. All context and memory
calls require exclusive access to the same context.

| Operation | Implemented contract |
|---|---|
| Clear | Clears occupancy, IDs, positions and shift metadata; optionally zeroes tensors. Pending cross-stream copies remain queued, matching the frozen source. |
| Remove | Removes selected membership over the exact half-open position range, including prefixes, middle ranges and suffixes. ID -1 selects every sequence. Cells become free when their final membership is removed. |
| Copy, unified | Shares existing physical cells with the destination over the requested range. |
| Copy, separate | Copies source metadata immediately and queues a full physical tensor-buffer copy for the next decode. Partial cross-stream copies fail explicitly; the original path asserts. |
| Keep | Retains only the selected sequence in its own stream. Other separate streams remain intact, as in the original. |
| Add/divide positions | Applies signed changes and deferred key rotation. Negative resulting positions free cells. Overflow and nonpositive divisors fail before mutation. |
| Position queries | Return minimum/maximum occupied position, or -1 for an empty sequence. |
| Can shift | Returns true for the supported dense cache. This does not advertise recurrent-cache support. |

The cache retains physical holes and membership independently of sequence
length. Deferred copies execute in queue order using existing destination
storage. Deferred shifts run after copies. seq_div retains the source's
old_position - divided_position accumulated shift.

Causal dense Llama supports independent F32/F16 keys and values in NativeContext
through new_with_cache_types. The existing new constructor and strict Context
remain F32. This slice requires explicit nonflash execution and one worker;
AUTO/ENABLED flash, quantized, recurrent/hybrid, SWA and GPU caches remain open. Duplicate
positions within one submitted batch require batched layer execution and return
a diagnostic. Shared-token input is supported in unified mode; coupled input in
separate mode is not implemented.

## State and file formats

| Surface | Frozen encoding |
|---|---|
| Full raw state | Architecture string, then stream count, occupied-cell metadata and typed F32/F16 tensor data. |
| Sequence raw state | 0xaf143cd8, source sequence ID, then selected memory metadata/data. |
| Full session file | Magic 0x6767736e, version 11, prompt count/IDs, then full raw state. |
| Sequence session file | Magic 0x67677371, version 4, prompt count/IDs, then selected memory without the raw sequence-I/O prefix. |

Integer fields are little-endian on tested x64 platforms. Occupied cells
serialize in physical order. Keys use native row order; values use the frozen
transposed dimension/cell order. Type, element size, width and layer headers are
validated before restoration and must match the context's independent key/value
types. F32 data uses four-byte elements; F16 data uses two-byte elements and raw
u16 storage. Copies, transposition, deferred tensor copies and serialization
preserve every bit, including signaling NaN payloads. Numerical evaluation alone
widens and rounds half values according to the source operation.

Size queries calculate the result without constructing a serialized buffer.
Save operations reproduce original write-chunk boundaries: an insufficient
destination receives only whole chunks that fit, returns zero and records an
error. Sentinels after the supplied capacity remain intact.

A valid host restore returns the consumed byte count and permits a trailing
buffer suffix, as the original does. Malformed state fails atomically in Rust.
Full restore clears and compacts physical cells. Sequence restore replaces only
a serialized nonempty source. An empty sequence state leaves an existing
destination unchanged, matching the source.

The state does not serialize pending tensor-copy or key-shift work. A live
context with pending work and a context restored from its saved bytes can
therefore have different continuations. The corpus records both; tests retain
this distinction.

### Extended sequence flags

PARTIAL_ONLY (bit value 1) produces the source's metadata-only sequence-I/O
result for this dense cache. ON_DEVICE (bit value 2) stores tensor bytes
separately inside the owning CPU context and emits the original small metadata
buffer. This is CPU ownership behavior, not GPU execution. Restore uses the
latest snapshot for the encoded source ID in that same context. Missing
ownership returns a checked error instead of the native assertion.
Empty/partial captures preserve a prior nonempty tensor store. Other flag bits
retain the measured frozen behavior.

### Session failure side effects

A sequence-file call with null token output performs the source count-only
operation: it returns 12, writes the token count and leaves memory untouched.
A prompt output capacity failure preserves existing memory and count.

Full-session trailing bytes are rejected **after** valid prompt outputs and
memory have been restored. The call returns false but those writes remain,
matching the frozen source. Sequence-file trailing bytes are accepted and the
return value reports only consumed bytes. Independent native fixtures and
ordinary regressions cover these side effects.

## Bounds and foreign-pointer requirements

Per-cell membership uses a fixed 32-byte bitset, with no allocation per ID.
Constructor and transaction checks account for cache tensors, cells, staged
copies, outputs and temporary state buffers under a two-GiB working budget.
Resident storage is charged using each K/V type's actual element size; gathered
computation remains charged for F32 values even when resident storage is half.
Restore includes the transaction copy, typed layer copies, descriptors, input
bytes and temporary transposed V storage. Large vectors reserve fallibly. Pending tensor copies reuse destination
allocations. CPU-owned snapshots additionally have a one-GiB aggregate storage
limit. These are explicit implementation limits, not native maximum-size or
performance claims.

Session files are opened once, checked for a regular-file type, and read through
an actual byte cap. A growing file cannot bypass the cap through stale metadata.
Saves stream prompt IDs and serialized state without a duplicate whole-session
buffer. State inputs are capped at two GiB and pointer-sized slice limits.

The deprecated llama_set_state_data has no length argument. Its caller must
supply the complete live allocation containing valid state. A bounded scanner
validates counts, shapes and checked offsets before constructing a slice of the
derived encoded length. A pointer alone cannot reveal a foreign allocation's
actual boundary. Prefer the explicit-size API where possible.

The README's pointer, ownership, alignment and concurrency requirements apply.
Panic containment cannot recover from invalid foreign memory or allocator
aborts. Decode cancellation remains atomic across a whole batch; the original
can retain completed microbatches, so that behavior is outside current parity.

## Independent validation

The external harness includes the frozen llama.h and loads the original and
Rust libraries in separate captures. Explicit nonflash, single-worker execution
was compared for all four K/V dtype pairs on both original platform profiles,
using stories260K and the small deterministic 512-token-vocabulary GGUF. The
synthetic model has no text-quality claim.

The ordinary typed-cache replay commits the tiny-model cohort. Per profile it
covers 160 memory scenarios, 160 sequence-state/flag scenarios, eight session
suites and 384 short-buffer cases. It compares 258,048 continuation logit words
and 1,048 state/partial-output buffers totaling 290,328 bytes exactly.
Every original raw state also restores and reserializes identically before
evaluation. The 520 original binary artifacts per profile and repeated logit
rows are stored in a shared lossless byte pool; reconstruction is checked
against the original hashes. Provenance is in
fixtures/ffi/f16-context-provenance.json.

F32/F32 remains in the typed corpus as a control, and prior F32 tests/budgets
are unchanged. Earlier native-context evidence records its original
2e-4 + 2e-5 * abs(reference) GNU bound and initial drift before shared compiler
profile arithmetic was integrated. Those historical reports remain preserved.
The new typed cohort compares to its independently captured MSVC or GNU source
profile bitwise; it does not widen an earlier bound or substitute generated
Rust expectations.

All 65,536 F16 bit patterns also pass raw storage/state preservation tests,
including type-mismatch and truncation rejection without mutation. A separate
original graph corpus checks the F16-by-F32 KQ/PV matmul layouts: 125 cases and
245,154 output values per host, including strided views and physical padding.
Explicitly half-rounding the F32 right operand changes 239,587 native output
values, so storage type is not used to invent an operand conversion.

The original default parameter structure is unchanged and requests AUTO flash
attention and four workers. Those settings remain rejected; this slice does
not claim successful untouched default-context initialization, native flash
dispatch, recurrent cache parity or full frozen-ABI completion.
