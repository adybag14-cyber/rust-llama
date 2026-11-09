# Dense F32 native memory and state compatibility

This slice implements the frozen Llama dense F32 cache contract through
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

Only causal dense Llama F32 keys and values are implemented in NativeContext.
F16/quantized, recurrent/hybrid, SWA and GPU caches remain open. Duplicate
positions within one submitted batch require batched layer execution and return
a diagnostic. Shared-token input is supported in unified mode; coupled input in
separate mode is not implemented.

## State and file formats

| Surface | Frozen encoding |
|---|---|
| Full raw state | Architecture string, then stream count, occupied-cell metadata and F32 tensor data. |
| Sequence raw state | 0xaf143cd8, source sequence ID, then selected memory metadata/data. |
| Full session file | Magic 0x6767736e, version 11, prompt count/IDs, then full raw state. |
| Sequence session file | Magic 0x67677371, version 4, prompt count/IDs, then selected memory without the raw sequence-I/O prefix. |

Integer fields are little-endian on tested x64 platforms. Occupied cells
serialize in physical order. Keys use native row order; values use the frozen
transposed dimension/cell order. Type, element size, width and layer headers are
validated before restoration. Raw float bits, including nonfinite payloads,
are preserved; numerical execution applies separate finite checks.

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
Large vectors reserve fallibly. Pending tensor copies reuse destination
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

The external C++ harness includes the untouched original llama.h and loads the
original and Rust libraries in separate captures. It exercises the trained
stories260K model and a small deterministic GGUF with 512 vocabulary entries,
two dense layers, biases and grouped-query attention. The synthetic model has
no text-quality claim.

Each model/profile covers 40 memory scenarios, 40 sequence-state/flag scenarios,
two session suites and 96 short-buffer cases. Its 130 binary artifacts total
1,329,186 bytes for the trained model and 58,890 for the synthetic model.
Ordinary tests use committed synthetic expected bytes from independent Windows
and GNU captures and require original raw restore/reserialize identity on both.

Windows generated state data and continuations match bitwise. GNU defines
__SSE2__ and selects the source vector exponential for SiLU/softmax; the MSVC
reference uses scalar expf. Both source F32 dot paths use scalar products with
F64 accumulation. Linux generated float checks retain the established
2e-4 + 2e-5 * abs(reference) bound. Metadata, IDs, sizes, write boundaries and
sentinels remain exact. Observed debug maxima were 5.7220458984375e-6 for the
trained cohort and 1.1920928955078125e-7 for the synthetic cohort. Actual build
flags, per-profile results and source hashes are recorded under
evidence/2026-11-09/native-context.

The preserved core C harness still checks 69 layout entries, every default
field and 93,248 trained-model values. The full sampler/context corpus also
passes. These results establish this subset and do not complete every frozen
export or native cache architecture.
