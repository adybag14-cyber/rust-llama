# Partial native Rust llama C ABI

This crate builds a cdylib, staticlib and rlib. The Windows DLL is
`target/release/llama_ffi.dll`; the import library is `llama_ffi.dll.lib`.
It executes the Rust GGUF loader, all nine frozen vocabulary family identifiers, CPU samplers and the dense CPU decoder from this
workspace. No original llama/ggml library is linked or loaded by the product.

**This is an operational subset, not full llama.cpp ABI compatibility.**
`COVERAGE.md` enumerates the exact frozen exports implemented and still absent.
Unavailable functions have no placeholder exports. The native reference DLL is
used only by an external test process.

## Frozen interface and compatibility checks

The interface comes from `include/llama.h` in the original archive with SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Model params, context params and batch use their exact C field order. On tested
Windows x64 their sizes are 80, 176 and 56 bytes, each with alignment 8.
All 69 sizes, alignments and field offsets are checked against the real header
by the external MSVC C++ harness, along with every model and context default
field returned by the original DLL. Enum slots are represented as i32 and bool
slots as u8 internally, allowing validation before interpreting foreign values.

`ABI_LAYOUT.md` documents the inspection extension's index order. No layout or
calling-convention compatibility is claimed for an untested target. Parent
workspace builds additionally check compilation on supported Rust targets.

## Operational configuration

The accepted execution path is dense Llama, using the validated GGUF loader and
tokenizer from `llama-inference`. Vocabulary-only loading covers the frozen
NONE, SPM, BPE, WordPiece, UGM, RWKV, PLaMo-2, test and PLaMo-3 identifiers.
Admitted architecture metadata is llama, bert, gpt2, gptneox, mpt, falcon,
command-r, refact, starcoder2, qwen2, gemma4, nomic-bert-moe, baichuan and phi3.
Execution of non-Llama architectures is explicitly rejected in this C layer.
Vocabulary-only loading retains the source's unloaded model-info defaults:
context/embedding/layer dimensions are zero, description is empty and file type
is F32. Attention-head queries return a diagnostic because the native source
aborts when no model layers were loaded. BERT's public architecture routing
flags and RoPE enum follow the frozen header/source values even though C context
execution is unavailable.

Weights use the same validated GGUF loader and its memory budget, numerical
formats and explicit architecture restrictions. File loading automatically
discovers the frozen numbered GGUF shards when primary split.count exceeds one.
Vocabulary-only loading validates every declared shard too. Architecture,
tokenizer and string metadata come only from the primary file; tensor byte and
parameter totals, and guessed file type, cover the complete validated set.
Full model loading consumes those same owned file handles. A primary split.count
of zero, one or absent retains the original single-file interpretation.
Missing siblings, malformed headers, duplicate tensor names, invalid indices or
payload extents fail with a diagnostic. The explicit C split-list and filename
buffer APIs remain unexported.

Start with the frozen parameter functions and select CPU model loading. On the
qualified Windows x64/MSVC and Linux x64/GNU CPU profiles, an admitted dense
Llama model now accepts the untouched context defaults:

```c
struct llama_model_params mp = llama_model_default_params();
mp.n_gpu_layers = 0;
mp.load_mode = LLAMA_LOAD_MODE_NONE;
mp.lazy_mode = LLAMA_LAZY_MODE_OFF;

struct llama_context_params cp = llama_context_default_params();
/* F16 K/V, AUTO flash, four generation/batch workers, separate cache. */
```

| Context selection | Accepted execution |
|---|---|
| Untouched context defaults | Source CPU AUTO resolution enables flash; the configured four workers execute. |
| AUTO or ENABLED flash, independent F32/F16 K/V | Source one-chunk, tiled or split-KV dispatch, with unified row maps where selected by the source. Worker counts are independently configurable in 1..=256. |
| DISABLED flash, independent F32/F16 K/V | Source nonflash attention with worker counts 1/1. |
| Other cache types, nonflash multiple workers, unqualified flash target | Checked error; no silent conversion or fallback. |

Model default n_gpu_layers=-1 resolves to CPU because this C ABI registers no
GPU backend. Positive GPU-layer requests fail. Automatic CPU attention dispatch
is backed by the frozen device-support probe and separate native AUTO, ENABLED
and DISABLED captures; a caller's ENABLED request never silently runs nonflash.

The shared safe flash kernel executes actual scoped OS workers. A one-worker
configuration executes on the caller; a larger configuration starts the stated
number of worker threads and joins them before returning. Thread query and
setter functions reflect the actual configuration. The microbatch's token count
selects generation versus batch workers. Unsupported changes fail atomically.
This reproduces the captured worker arithmetic and dispatch, without claiming
the original persistent threadpool topology or equivalent throughput.

Separate and unified caches retain independent F32/F16 key/value storage.
Half storage preserves raw u16 payloads through copies and state/session I/O,
including nonfinite payloads. Numerical writes perform the source F16 rounding.
Nonflash query/probability operands remain F32; the source flash graph casts
physical K/V to F16 even when their resident storage is F32. Complete current-layer
K/V writes precede attention for every native microbatch.

Cache capacity follows the source 256-cell padding. Physical holes, sequence
sharing, queued copies and signed position transforms remain observable.
Unified memory APIs retain their measured 256-ID domain; submitted decode IDs
must additionally be below configured n_seq_max. Invalid IDs preserve both
previous output buffers and serialized state. n_seq_max must fit the default
output-row ceiling, min(n_batch, resolved n_ctx); source-asserting configurations
return a checked initialization error. Batch size remains bounded to 65,536.
A zero n_ctx resolves to the model's training context.

Newly empty sequences may begin at a nonnegative position. Existing sequences
resume after their stored maximum. Within each submitted sequence, positions
may repeat or advance by one, including across internal microbatches. Null
positions compute the next position; null sequence arrays select sequence 0.
Token and embedding inputs remain mutually exclusive. Unified shared-token
input is supported; coupled multi-sequence input in separate mode remains a
checked error.

Logit and embedding rows follow the original output flags and embeddings mode:
when embeddings are enabled, every input row is emitted. Positive indexed
getters use original caller batch positions, even when native sequence splitting
changes execution order; negative indices count backward through emitted rows.
Invalid or unrequested rows return NULL. External embedding input and normalized
hidden output are implemented. Pooling and noncausal decoding are outside this
C executor. EXECUTION.md details row maps, workers and resource accounting;
STATE.md documents the raw cache and session contracts.

## Unsupported options

Context creation rejects KV types other than F32/F16, nondefault context types,
custom RoPE/YaRN, pooling, noncausal attention, nonflash multiple-worker execution, evaluator
callbacks, backend samplers, shared/MTP contexts, MoE caches, recurrent-state
slots, nondefault output capacity fields and performance counters. SWA preferences have no effect for the supported non-SWA model.
All boolean and enum inputs are validated.

Model loading rejects device lists, tensor buffer overrides, tensor splitting,
metadata overrides, positive GPU layer counts, mmap/mlock/direct I/O, forced
lazy loading, no-host/no-allocation requests and MTP. CPU capability queries
honestly report the absence of mapped loading, locking, GPU offload and RPC.

The explicit-length tokenization entry point accepts arbitrary bytes, including
invalid UTF-8 and embedded NUL, with the source behavior of each tokenizer family.
Byte pieces and detokenized outputs can contain arbitrary bytes. Input and buffer
element counts remain bounded. Special-token metadata with an out-of-range ID
retains the source default, including UINT32_MAX padding IDs. The frozen
misspelling tokenizer.ggml.seperator_token_id is intentional. BOS/EOS/EOT/FIM,
padding, masks, EOG attributes, newline IDs and insertion flags use the shared
tokenizer's normalized loader results. Ambiguous inferred special spellings
follow the independently measured MSVC/libstdc++ source map order.

The suppression-token pointer/count getter preserves source order and duplicates,
and its count pointer is optional. NONE retains its nominal vocabulary count
and special IDs, returns empty detokenization, and rejects unavailable token
data/pieces/tokenization with a checked diagnostic instead of the native abort.

Dense F32/F16 memory manipulation and frozen state/session formats are implemented
as described in STATE.md. Backend-attached sampling, recurrent native caches,
adapters, encoder entry points, model saving, explicit split-list loading and
performance APIs remain absent. Source custom chat-template lookup
is implemented; template application is not exported.

## Automatic shard regression

The ordinary model_shard_tests replay uses independent frozen-header C captures
from Windows MSVC and GNU x86_64. Each profile checks 96 auto-load cases:
53 accepted and 43 rejected, including metadata-only first files, subsidiary
metadata conflicts, inferred file types, missing split.count partial totals,
missing siblings and malformed shards. Successful cases check model dimensions,
description, type, byte/parameter totals, vocabulary, token IDs/pieces and
detokenization. Full Llama cases additionally compare 163,840 F32 logit words
exactly, including greedy continuations. Expected values are deduplicated to
16,384 stored words per profile, with the original captures retained separately.
fixtures/ffi/shards-provenance.json records input and capture hashes.

## CPU sampling

SAMPLERS.md documents all 20 ordinary CPU sampler constructors, generic/custom
interfaces, owning chains, raw-byte grammar/lazy triggers, vocabulary-derived DRY
breakers, adaptive-p, infill, and actual-context sampling. The sampler ABI has an
independent 36-entry layout check. Backend-attached sampling, llguidance integration
and sampler performance collection remain unavailable. Source-undefined pending
adaptive clone/copy is rejected with a diagnostic.

## Memory, lifetime and error contract

Every non-null foreign pointer must identify a live, aligned allocation of the
declared length. Output buffers must be writable, exclusive and disjoint from
inputs. The caller must provide a NUL terminator for C-string arguments; the
tokenizer's explicit-length input does not require one. Handles must originate
from this library. Use of stale, fabricated or incorrectly sized pointers is
outside the ABI contract; Rust cannot validate foreign allocation boundaries.

Model and context allocations have explicit owners. A context keeps an Arc to
its model weights and vocabulary, so it continues to work after the original
owning model handle is released. `llama_get_model` returns a distinct borrowed
handle; it must not be freed. `llama_model_get_vocab` returns a borrowed pointer.
It stays valid while a model or context owning the shared data remains alive.

Context calls require exclusive access, including getters during decoding.
Independent contexts may run concurrently. Callbacks must be valid, must not
unwind and must not free or reenter their active handle.

`llama_batch_init` allocates all arrays, with a trailing NULL sequence pointer.
It returns n_tokens = 0; fields are zero-initialized as a safe extension of the
native uninitialized allocation. Populate the per-token fields before decoding.
Do not replace pointer fields in an owned batch. Free it using the same
library's `llama_batch_free`. A registry retains allocation dimensions and
ownership, so freeing never trusts mutable n_tokens or a sequence sentinel.
A detected duplicate/foreign free sets an error instead of dereferencing it.
`llama_batch_get_one` borrows the caller's token array and must not be freed.

Returned output buffers belong to the context and remain valid until the next
successful decode or context destruction. Failed or cancelled decode batches
publish no KV changes and retain previous output buffers. This implementation
commits the complete batch atomically, including across microbatches.
Decode returns 0 on success, 1 for exhausted logical capacity, 2 when an abort
callback cancels, and -1 for invalid/unsupported input or numerical failure.

All exports catch Rust panics before the C boundary. Allocator aborts and invalid
foreign memory are not recoverable. Pointer-returning failures return NULL;
count-based tokenizer failures return INT32_MIN, while insufficient capacity
returns the negative required size. String description functions follow snprintf
length/truncation/NUL-termination semantics. No token-piece or detokenization
terminator is appended. Detokenize count-only and insufficient-buffer returns
use the native pre-cleanup byte count; a successful WordPiece call can return
fewer bytes after space cleanup. Insufficient detokenize buffers retain the
complete token pieces that fitted before the first capacity failure.

The Rust extensions are:

- `const char * llama_rust_last_error(void)`: latest error on the current thread;
  valid until that thread's next error or explicit clear.
- `void llama_rust_clear_error(void)`: clears that thread's diagnostic.
- `size_t llama_rust_abi_layout(size_t index)`: sizes/offsets in ABI_LAYOUT.md;
  SIZE_MAX beyond the final index.
- size_t llama_rust_sampler_abi_layout(size_t index): sampler sizes/offsets;
  SIZE_MAX beyond its 36-entry table.

Successful calls do not erase an earlier diagnostic. Void APIs report rejected
changes through this extension.

## Validation and reproduction

Run `cargo test -p llama-ffi`, `cargo test -p llama-ffi --release` and
`cargo clippy -p llama-ffi --all-targets -- -D warnings`.
The unit tests cover panic containment, thread-local diagnostics, checked
buffer conventions, allocation limits, owned token/embedding batch frees,
sampler ownership and cycles, checked history, native tied-sort regressions,
XTC pointer semantics and RNG clone/reset/error behavior.

The native MSVC harness is intentionally outside the product at
`C:/Users/adyba/rust-llama-reference/ffi-oracle`. It includes the untouched frozen
header, dynamically loads both the Rust DLL and the independently built
reference DLL, and runs the committed stories260K oracle fixture.
The exact harness/build/reference/model hashes and command lines are recorded
under `evidence/2026-11-09/ffi`. No native harness or reference object is linked
into the product.

The harness checks exact ABI layout/defaults, every vocabulary text/score/
attribute/control/EOG entry, token pieces, model dimensions/size/parameter count/
description, six trained prompt/greedy cases, contiguous/sparse/negative output
indexing, independent interleaved sequences, mid-batch rollback, abort rollback,
embedding input/output, model ownership after handle release, vocabulary-only
loading and explicit rejection of unsupported settings.

The additional CPU sampler, WordPiece and BPE/FIM C harnesses, source-specific
fixes and exact measurements are documented in SAMPLERS.md and
evidence/2026-11-09/sampler-cpu. The new vocabulary harness checks 1,505,576 entries
across all 14 frozen BPE vocabularies plus the synthetic raw-byte fixture,
5,096 text cases and 2,502 raw-byte cases, including buffer sizes and partial
detokenization writes. Prior sampler evidence remains preserved separately.

The additional native context corpus uses both MSVC and GNU builds of the original
source. Ordinary tests replay all 40 memory cases, 40 sequence-state cases and 96
short-buffer cases against committed synthetic-model fixtures. Session files,
legacy aliases, trailing-file side effects and malformed-state atomicity are
covered. Original raw buffers restore and reserialize bitwise exactly. Generated
Windows state floats/logits were bitwise exact in that initial external cohort;
its GNU report retains the original predeclared bound and initial arithmetic
drift. Subsequent compiler-profile arithmetic, typed-cache and flash cohorts
compare independently captured native outputs bitwise on each qualified host.
Historical expectations and failure logs remain intact. See STATE.md and the
corresponding evidence directories for exact scope and source/build identities.

These checks establish the stated subset on Windows x64 and Linux x64. They do
not close the full ABI, architecture, backend, platform or performance gates.

## Ordinary typed-cache regression

The f16_context_reference_tests module replays independent original MSVC and
GNU tiny-model captures for F32/F32, F16/F16, F16/F32 and F32/F16 storage.
Each profile covers 160 memory scenarios, 160 sequence-state/flag scenarios,
eight session suites and 384 short-buffer cases. It checks 258,048 logit words
and 1,048 state/partial-output buffers (290,328 bytes) exactly, including
physical holes, deferred copies, shifts and restore/reserialize without
evaluation. The original 520 binary artifacts per profile are losslessly
deduplicated with the continuation rows; no Rust output generates expectations.
See fixtures/ffi/f16-context-provenance.json for hashes and reconstruction proof.

The independent F16 cache matmul graph corpus additionally checks 125 original
graphs and 245,154 F32 outputs per platform, including head-offset views, GQA,
strided K caches and 256/512/768-column V products. Trained-model typed-cache
execution remains covered by the external C harness, while this ordinary
state/decode cohort deliberately uses the small deterministic fixture.

## Native microbatch and flash regression

The ordinary native_microbatch_reference_tests replay covers 205 independent
tiny/generated-model configurations per measured profile. It checks 2,398,720
logit words, 588 embedding words, 319 exact serialized states (1,098,603 bytes),
default parameter fields, worker setters, native input/output ordering, all four
K/V storage pairs, duplicate positions, shared sequence membership, physical
holes and shifts. Two additional inputs that assert in the original context
constructor must produce checked initialization errors.

The full external C adapter retains 228 requested configurations per host:
226 compare original return values and bytes, and two compare the documented
checked replacement for the original initialization assertion. Its successful
comparisons cover 2,917,888 logit words and 14,171,045 binary bytes, including
pretrained stories260K. The compact ordinary replay excludes external weights
and deduplicates identical inputs only after comparing the original results.

Both nonflash and enabled-flash state/session suites are replayed for each of
the four K/V storage pairs. Each mode retains the 258,048-logit and 1,048-buffer
tiny-model checks described above. The flash suite independently verifies
untransposed V state, host/device sequence flags, all short-buffer write
boundaries and continuation behavior. Original expected bytes are in
fixtures/ffi/{native-microbatch,flash-context}-provenance.json and their adjacent
reference/data files. No Rust execution generates expected numerical output.

The model-runtime row-map regression separately compares source-5 I32 buffers
and Q/mask geometry. It includes 768 logical columns caused by shared sequence
memberships even though only 261 unique rows are selected. Retained negative
controls show that unique-row padding, omitted row-map memory accounting and
omitted decode-ID bounds are each detected. Qualified flash arithmetic lives in
llama_quant::flash_attention; its own independent graph corpus is documented in
crates/quant/FLASH_ATTENTION.md. NativeContext continues to reject nonfinite
computed results and makes no exceptional-logit parity claim.
