# Strict Context state allocation boundary

This external Qwen2 draft adds a checked, fallible allocation boundary to the
existing strict Rust Context. It does not change the separate NativeContext or
the original C ABI state format.

## Budget and scope

ContextOptions::max_state_bytes defaults to 1 GiB and must be nonzero. Each
Context owns a ledger shared with the checkpoints created from it. The ledger
reserves bytes before allocation and rejects checked arithmetic overflow,
budget exhaustion, and failed Vec reservations through Error::InvalidState.

The charged payload includes:

- Sequence record vector capacity, layer record vector capacities, and every
  KV/convolution/SSM F32 vector capacity.
- Sequence allocation plans and the complete staged replacement state for an
  atomic batch, fork, restore, or checkpoint copy.
- Live checkpoints and their independent try_clone copies, until those objects
  are dropped. Dropping the originating Context does not release reservations
  still owned by a checkpoint.
- A serialized state output allocation while save_state creates the copy.

This is not a total-process memory limit. Immutable model weights, execution
scratch, output logits/embeddings, allocator overhead, small ledger/control
objects, and caller-owned input/output byte buffers are outside persistent
state accounting. The Vec returned by save_state becomes caller-owned after
the checked allocation finishes. Restoring a checkpoint into another Context
charges the newly copied state to the destination ledger.

state_bytes reports the Context's resident state capacities.
reserved_state_bytes also includes its retained checkpoints and active
reservations. peak_reserved_state_bytes reports the high-water reservation;
state_budget_bytes reports the configured limit. These are reservation metrics,
not operating-system resident memory measurements.

## Allocation and transaction behavior

The internal sorted sequence Vec replaces BTreeMap so sequence records can
also be reserved fallibly. A batch validates all positions, token/embedding
inputs, and sequence growth before constructing replacement state. All KV
vectors reserve capacity for the final position of the complete batch before
any layer runs. Recurrent state retains its fixed model-defined size regardless
of token count.

The old committed state remains alive while replacement storage is reserved
and filled. Failure or cancellation drops the replacement and its reservation.
The same accounting covers forks, checkpoint creation/copy, checkpoint restore,
and serialized-state restore. Float storage is allocated with try_reserve_exact;
there is no infallible collection or Vec clone on these state-copy paths.

Truncation retains existing vector capacity and keeps it charged. A subsequent
transaction compacts storage to its planned shape. Sequence removal releases
its dropped payload; reset releases all resident state. Held checkpoints remain
charged independently.

State decoding makes a complete validating pass before allocating float
vectors. It checks the unchanged v1 format, CRC, model/adapter identity, count
and position bounds, memory kind, shapes, finite values, duplicate IDs, and
trailing data. A second pass fills already-reserved replacement buffers.
Serialization order and v1 byte representation remain unchanged.

## Public API changes

- ContextOptions adds max_state_bytes; existing constructors can use
  ..ContextOptions::default().
- Context::checkpoint returns Result<ContextCheckpoint>.
- ContextCheckpoint has a fallible try_clone method instead of infallible Clone.
- The four reservation inspection methods described above are available.

No new Error enum variant or dependency was added. The two existing runtime
test checkpoint callers now unwrap the Result. Root updated the high-level
inference constructor and the older external FFI constructor to use the default.
The obsolete private empty_cache methods in Llama/Qwen2 and empty_state method
in Mamba were removed with root authorization; runtime equations were untouched.

For later product integration, preserve the NativeContext module declaration
and exports already added in the product. This draft's changes are confined to
strict Context storage, its private codec/helpers, and related tests.

## Verification

Windows and Linux each pass nine new allocation tests and all 29 existing
runtime integration tests in debug and release, with strict all-target Clippy.
The new tests cover exact budget thresholds, pending KV growth, fixed recurrent
state, sequence multiplicity, checkpoint lifetime and copies, atomic copy and
restore rejection, serialization allocation, truncation/removal accounting,
checked overflow, and concurrent reservations. A test-only scoped fault injector
enumerates every owned state Vec reservation in each copy/restore path, including
failures after earlier buffers have been allocated, and verifies unchanged state
and released reservations.

Logs are retained in validation/windows-after-count-and-test-lint and
validation/linux-after-count-and-test-lint. Earlier logs retain the stale smoke
registry count failure (three versus the new four executable slices) and a
test-only Clippy predicate style failure. No numeric expected data was modified.
Root independently runs the native Qwen2 model/metadata corpora and full focused
model/inference checks after the allocation source is stable.
