# Native BERT encoder and classifier C interface

This package implements real complete-batch BERT execution behind
`llama_encode` and `llama_get_embeddings_seq`, and real classifier metadata
behind `llama_model_n_cls_out` and `llama_model_cls_label`. It extends full
model loading, context construction, `llama_decode`, embedding getters and
context configuration for this architecture. Existing Llama decoder execution
continues through its separate native decoder context.

The reference is the frozen source archive
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
The defining paths are `src/models/bert.cpp`, `src/llama-context.cpp`,
`src/llama-batch.cpp`, `src/llama-batch.h`, `src/llama-graph.cpp` and
`include/llama.h`. The source, original libraries, C++ clients, generators
and compiler output remain external comparison inputs. No original object,
library, Python runtime or C++ helper is linked into the Rust library.

## Execution and ownership

The C facade owns an explicit decoder-or-encoder runtime. Its BERT variant
holds an `Arc<BertModel>`, native execution settings, a shared output buffer
and owned pooled vectors. It has no decoder KV cache. Consequently
`llama_get_memory` returns null for BERT, and decoder state/snapshot,
adapter composition and extended `llama_process` execution do not become
BERT implementations. Their checked dispatch rejects missing decoder memory.

The frozen `llama_model_has_encoder` query returns **false** for BERT and
`llama_model_has_decoder` returns **true**. Those original public results are
preserved. The source `llama_decode` implementation calls encoding when
memory is absent; the Rust BERT context follows that route. `llama_encode`
is presently admitted for BERT, with an explicit unsupported error for the
existing Llama decoder context.

A context owns a distinct borrowed model handle sharing immutable model data.
Its model, vocabulary, label and output storage remain alive with the context
after the original Rust model owner is released. This is an owned Rust lifetime
guarantee; it does not license freeing arbitrary source-library handles while
they are in use. Output and borrowed handles must never be freed by callers.
The crate-level foreign-pointer, exclusive-access and non-reentrant callback
contract applies to every call.

## Batch and sequence semantics

One encode operation evaluates the entire bidirectional batch. It cannot
microbatch independent pieces of a sequence: `n_tokens <= n_ubatch` is
required. Source-invalid configurations that assert or access invalid memory,
including empty sequence memberships and an undersized microbatch, return a
bounded error.

A batch accepts token IDs or host F32 input embeddings of the model width.
When both are provided, token execution takes precedence, as in the frozen
BERT graph. The token/type/position additions retain the source grouping,
followed by layer normalization, Q/K/V attention, the attention residual and
normalization, source GELU, the feed-forward residual and final normalization.
The optional token-type table requires F32: the original binary-add graph
aborts for F16/BF16 token-type tables. Other admitted packed matrix formats
retain their native activation conversion rather than becoming decoded-F32
matrix products. This package's new storage captures cover F32, F16 and BF16
matrix weights, plus the pinned F16 pretrained BGE model.

Stored norm weights and active bias vectors likewise require F32 because the
native CPU binary dispatcher has no F32-left/F16-or-BF16-right case. Loading
rejects those unsupported stored vectors before graph creation. Classifier
biases are active for this model admission check when their corresponding
matrix exists; this strict model-level boundary also applies when a later
context would select ordinary embedding pooling. Biases without a paired
classifier matrix remain inactive and retain their decodable stored types.
A separate native addendum captures five source graph failures, both ignored
half-precision classifier-bias cases and two existing F32 controls.

Absent per-token sequence counts default to one; absent sequence pointers
default to sequence zero. Counts larger than one with no sequence pointer
are rejected. Absent positions start at zero on each encode and increment
only the **first supplied sequence ID**, before container conversion.
Explicit positions may repeat, but must be nonnegative, fit the learned
position table, be nondecreasing within each sequence and contain no gaps.

The source stores memberships in `std::unordered_set<int32_t>`, then copies
the token into its batch. The first ID in that resulting container determines
the no-cache attention group. This is compiler-profile behavior:

- MSVC hashes the four-byte integer representation, uses its native bucket
  growth and reinserts set iteration order during the copy. Collision-chain
  ordering can therefore change in the copy.
- GNU uses its native integer hash, bucket growth and list ordering; copying
  preserves that list order.

The Rust C path reproduces those independently captured profiles. It does not
sort memberships or replace the mask with arbitrary set intersection.
Duplicate membership inputs are deduplicated. Every actual membership still
contributes to pooling. Pooling output sequence IDs are ascending.

Encoding forces noncausal attention during execution even when the public
causal flag is true, then preserves that flag afterward. Default attention
uses the model metadata; missing causal metadata follows the source's true
default. Explicit causal/noncausal settings affect the context's reported
configuration and initial batch cap, not the encoder's bidirectional graph.

## Pooling and classifier behavior

| Mode | Execution and extraction |
| --- | --- |
| NONE (0) | Final-layer rows are copied to the token embedding buffer. |
| MEAN (1) | Every row belonging to a sequence contributes an F32 reciprocal product accumulated in F64. |
| CLS (2) | Select the first row at the minimum position in the sequence. |
| LAST (3) | Select the last row at the maximum position in the sequence. |
| RANK (4) | Select CLS, or MEAN when the UINT32 classifier pooling metadata is one; then execute the optional classifier head. |

Pool `-1` resolves model metadata, with NONE when pooling metadata is absent.
CLS/LAST position tie handling follows the original graph rather than assuming
that arbitrary input row order is a position ordering.

Optional `cls.weight` and `cls.bias` form a dense layer followed by tanh.
Optional `cls.output.weight` and `cls.output.bias` form the final classifier
projection. Biases without their corresponding matrix are retained/validated
and have no independent execution. Ordinary embedding pooling does not run
the classifier. The Rust embedding facade also executes RANK, including its
single-sequence source extraction width.

`bert.classifier.output_labels` determines the class count when nonempty;
the default count is one. Labels are owned byte strings with a terminating
NUL, and out-of-range label indices return null. Embedded NUL bytes retain
the original C-string view. These getters do not manufacture a classifier
head: corresponding optional weights are loaded, validated and executed.

The original C extraction uses `n_cls_out` as its flat stride even when an
optional output projection is absent and the graph still has hidden width.
That measured behavior, including dense-only heads and multiple sequences,
is retained. An extraction count greater than the physical graph width is
rejected instead of reading beyond a graph result.

The independently retained INT32 classifier-pooling fixture is rejected by
both originals and Rust. Its separate UINT32 counterpart exercises successful
mean pooling and head execution. The negative fixture was not relabeled as a
successful classifier case.

## Output pointers and failure lifecycle

All pointer getters borrow context-owned storage. Repeated sequence getters
are stable until an operation replaces or clears that sequence's outputs.
Callers must copy values before a subsequent encode, decode or context free.

The original raw output buffer has observable allocation behavior even when
its regions are not populated by the BERT graph. The Rust context retains it:

- At construction, raw embeddings exist when enabled, and the initial graph
  assumption also allocates raw logits. No indexed outputs exist yet.
- After BERT graph reservation, raw logits are absent in embedding mode.
  In logits mode the buffer may still exist even though BERT has no language
  model output head. These bytes are zeroed or retained allocation contents,
  not calculated token logits.
- Pooled modes return actual vectors through `llama_get_embeddings_seq`.
  Their raw per-token buffer remains allocated but is not populated by pooled
  extraction, retaining the source zero/stale contents.
- `llama_set_embeddings` changes the next execution configuration. Existing
  pointer views change when the next accepted encode reserves output storage.
- Growing the shared buffer clears its new allocation; reusing its capacity
  preserves unwritten bytes. Positive indexed getters use batch indices;
  negative indices count backward through requested output rows.
- Embedding mode outputs every token regardless of output flags. With
  embeddings disabled, flags select rows; absent flags select the last row.

Invalid batches leave prior output mappings and pooled vectors intact.
An accepted operation clears pooled vectors and replaces output-row mapping
before graph execution. Cancellation returns two, leaves those pooled results
cleared and preserves the source's unwritten raw buffer contents. Recovery on
the next valid call is tested. Synchronization is immediate because execution
is synchronous. Exact callback poll counts are not a compatibility claim.

Prevalidation failures return minus one; output-buffer allocation failure
returns minus two; an executor failure returns minus three. The original
process-asserting inputs are covered by explicit Rust rejection tests.
No claim is made that allocator exhaustion or every original diagnostic
message has been reproduced.

## Accepted configuration and resource bounds

The native C encoder is qualified on x86-64 MSVC Windows and x86-64 GNU Linux.
Foreign sequence counts and unique sequence IDs are bounded to 256; raw
membership lists are also bounded to 256 entries per token. Context batches
are bounded to 65,536 rows and one million total memberships. Every input
slice, checked dimension, classifier shape and learned position is validated.

The source context's 256-position padding and unified/separate sequence
capacity calculations are retained for queries. These reported capacities do
not enlarge a learned BERT position table. Causal configuration caps the
logical batch by the raw requested context before padding; noncausal
configuration retains the requested batch.

Execution preflight bounds the encoder's live data and scratch payload to
512 MiB and scalar-work estimate to 2^36 items. The allocation-free shape
check runs before any foreign input slice or direct-embedding copy. Full
per-row validation then runs before outputs change. Membership bookkeeping
and classifier result/work estimates are included. Immutable model weights,
caller-owned inputs and allocator bookkeeping are excluded from that budget.
The flash engine independently accounts for its output, scratch, descriptors,
requested thread stacks and worker overhead against a 512 MiB limit.
The retained C output buffer has a separate 512 MiB limit. These are explicit
payload limits, not a total process peak-memory guarantee.

Existing model loading budgets still apply; labels are additionally bounded
to 65,536 entries and 16 MiB including terminators. Output-count/shape overflow
is checked before allocation. Unsupported model tensors/configurations remain
explicit loader errors.

Nonflash execution currently uses one worker. Native CPU flash supports the
measured AUTO/DISABLED/ENABLED choice, F16 K/V conversion and actual scoped
workers from one through 256, using generation workers for a one-row batch
and batch workers otherwise. The original physical softmax columns, masked
holes and 256-column padding remain present. No decoder KV storage is
allocated for BERT.

Existing unsupported context options remain unsupported, including scheduler
callbacks, backend sampling and shared/MTP contexts. Context elapsed/count
metrics accept both no_perf settings as documented in CONTEXT_PERFORMANCE.md;
GGML graph metrics remain zero for direct execution. GPU discovery/offload, other encoder
architectures, BERT adapters and generalized encoder/decoder graphs are
separate open requirements.

## Independent validation data

`fixtures/ffi/encoder` contains twenty tiny GGUF models, separate native
MSVC/GNU C observation corpora and separate I32-set corpora. Provenance records
the external original executable, source/caller, matrix and every native
output identity. The primary corpus was sealed at the clean preimplementation
base before Rust edits. Corrected source-type and set-order addenda remain
separate and retain their earlier failed attempts.

Each host's C corpus has 192 scenarios: 188 tiny scenarios (including thirteen
expected model-load rejections) and four pinned pretrained BGE scenarios, each
with both single-sequence and interleaved-sequence calls. Each I32-set corpus
has 569 cases, covering every size through 256, rehash boundaries,
permutations, collisions, duplicates and copying. Both compiler profiles are
replayed by the ordinary set-policy test. The vector-storage addendum has
nine further cases per host: five active-type failures handled by bounded
Rust rejection, two ignored-bias executions and two F32 controls.

C observations check model flags and labels, context dimensions, default and
explicit pooling, token/direct embeddings, shared and sparse sequence IDs,
position validation/ties, output flags, getters and offsets, cancellation,
recovery, embedding toggles, decode routing and flash workers. Rust ownership
and bounded source-invalid inputs have additional ordinary tests.

Float comparisons retain the pre-existing model-domain tolerances:
tiny values use `2e-6 + 2e-6 * abs(reference)`; BGE uses
`2e-4 + 2e-5 * abs(reference)`. All statuses, model/context facts, nullness,
indices, offsets, booleans, labels and set orders are exact. Raw source words
are retained so bitwise agreement and maximum errors can be reported
separately. No tolerance was widened to accommodate the shared-sequence bug.

The package's focused validation uses debug/release tests for
`llama-model-runtime`, `llama-inference` and `llama-ffi`, strict Clippy,
format checks, the pinned pretrained tests, independent original-header C
replay against both build modes, and exact PE/ELF export inventory checks.
Receipts and failed attempts are under
`evidence/2026-11-11/c-bert-encoder`. This is a bounded encoder milestone;
it does not close the full-port release gates.
