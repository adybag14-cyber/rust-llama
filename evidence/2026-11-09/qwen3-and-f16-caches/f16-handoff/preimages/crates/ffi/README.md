# Partial native Rust llama C ABI

This crate builds a cdylib, staticlib and rlib. The Windows DLL is
`target/release/llama_ffi.dll`; the import library is `llama_ffi.dll.lib`.
It executes the Rust GGUF loader, all nine frozen vocabulary family identifiers, CPU samplers and scalar decoder from this
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
formats and explicit architecture restrictions.

Start with the frozen default parameter functions, then select CPU loading,
F32 KV caches and causal decoding. For example, a client using the frozen C
header can set:

```c
struct llama_model_params mp = llama_model_default_params();
mp.n_gpu_layers = 0;
mp.load_mode = LLAMA_LOAD_MODE_NONE;
mp.lazy_mode = LLAMA_LAZY_MODE_OFF;

struct llama_context_params cp = llama_context_default_params();
cp.n_ctx = 512;
cp.type_k = GGML_TYPE_F32;
cp.type_v = GGML_TYPE_F32;
cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
cp.n_threads = 1;
cp.n_threads_batch = 1;
cp.offload_kqv = false;
cp.op_offload = false;
```

The default structure itself remains source-faithful: it returns F16 cache types.
Context creation rejects those unsupported types with a diagnostic; it never
substitutes F32 silently. Model default `n_gpu_layers = -1` and automatic
attention/offload preferences resolve to CPU because this C ABI registers no
GPU backend. Positive GPU-layer requests and forced flash attention fail.

The executor runs one worker. Context creation requires thread counts 1/1,
and the thread query functions report that actual value.
`llama_set_n_threads` accepts 1/1; unsupported changes leave state unchanged
and set the diagnostic.

The compatibility executor supports separate and unified physical F32 KV
buffers. Context padding, per-stream capacity, metadata holes, sequence sharing,
queued cross-stream copies and signed position transforms follow the measured
frozen cache behavior. Separate mode has the configured sequence streams;
unified mode accepts the frozen 256-ID range, with per-token membership bounded
by n_seq_max. Both modes pad cache capacity to the source's 256-cell boundary.
Batch limits are bounded to 65,536 tokens. A zero n_ctx resolves to the model
training context. STATE.md documents precise memory and serialization behavior.

A newly empty sequence may begin at any nonnegative position. Existing sequences
resume after their stored maximum; positions within a batch must advance
consecutively. Null positions compute the next position for each sequence.
Null sequence arrays mean sequence 0. Both sequence arrays must otherwise be
provided. Token and embedding inputs are mutually exclusive. Duplicate positions
inside one batch and coupled multi-sequence token input in separate-buffer mode
are explicitly unsupported; unified shared-token input is implemented.

Logit and embedding rows follow the requested batch flags. Positive
`llama_get_logits_ith` and `llama_get_embeddings_ith` indices refer to original
batch token positions; negative indices count backward through output rows.
Invalid or unrequested rows return NULL. External embedding input and normalized
hidden output are implemented. Pooling and noncausal attention are not.

## Unsupported options

Context creation rejects non-F32 KV caches, nondefault context types, custom
RoPE/YaRN, pooling, noncausal attention, forced flash attention, evaluator
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

Dense F32 memory manipulation and frozen state/session formats are implemented
as described in STATE.md. Backend-attached sampling, recurrent native caches,
adapters, encoder entry points, model saving, split-file loading and performance
APIs remain absent. Source custom chat-template lookup
is implemented; template application is not exported.

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
Windows state floats/logits are bitwise exact in the external cohort; Linux
generated floats use the predeclared model conformance bound because the frozen
GNU source selects a different vector exponential path. See STATE.md and
evidence/2026-11-09/native-context for measured errors and actual build flags.

These checks establish the stated subset on Windows x64 and Linux x64. They do
not close the full ABI, architecture, backend, platform or performance gates.
