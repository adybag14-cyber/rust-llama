# Native LoRA and control-vector C APIs

The Rust library implements ten frozen adapter exports:

- `llama_adapter_lora_init` and `llama_adapter_lora_free`;
- `llama_adapter_meta_count`, `llama_adapter_meta_val_str`,
  `llama_adapter_meta_key_by_index` and `llama_adapter_meta_val_str_by_index`;
- `llama_adapter_get_alora_n_invocation_tokens` and
  `llama_adapter_get_alora_invocation_tokens`;
- `llama_set_adapters_lora` and `llama_set_adapter_cvec`.

These functions load owned weights and change actual dense Llama CPU execution.
The implementation does not link, load or call the reference library.
`llama_adapter_lora_init_from_file_ptr` remains unexported. This package does not
add a backend scheduler, native context performance counters or warmup exports.

## Source and admitted behavior

The contract comes from `include/llama.h`, `src/llama-adapter.cpp`,
`src/llama-context.cpp` and `src/llama-graph.cpp` in the frozen archive with SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Independent original-header callers capture Windows x64/MSVC and Linux x64/GNU
behavior. Platform-specific metadata formatting and unordered-map iteration
follow the already measured native policies described in METADATA.md.

A LoRA file must contain matching `general.type = adapter`, the base model's
`general.architecture`, and `adapter.type = lora` strings. The optional
`adapter.lora.alpha` must be a finite FLOAT32; absent alpha becomes zero.
Nonzero alpha scales a delta by `user_scale * alpha / rank`. Zero alpha keeps
the user scale without dividing by rank. Negative alpha and negative scales
are supported. Invocation tokens are an optional ARRAY of UINT32: their bits
are preserved when returned through signed `llama_token`, including negative
representations. This native API exposes the invocation sequence; it does not
automatically gate LoRA application on matching that sequence.

Every adapted base weight must exist physically in the loaded GGUF and have a
complete `.lora_a`/`.lora_b` pair. Ordinary matrices use A [input, rank] and
B [rank, output] in GGML dimension order. Token embeddings use A [rank,
vocabulary] and B [rank, embedding width]. The token path gathers the A row,
multiplies B and adds the scaled delta to the base embedding. Caller-supplied
embedding rows bypass the token embedding gather and its LoRA while retaining
adapters on later matrices. Normalization tensors ending in `_norm.weight` are
ignored, as in the source; other unexpected tensor suffixes are rejected.
Metadata-only and normalization-only adapter files are valid.

The retained execution corpus covers F32, F16 and BF16 LoRA payloads. Matrices
retain their packed representation when activation conversion depends on it;
there is no decoded-F32 fallback for a native dot type that is unsupported by
the shared runtime. The loader explicitly rejects unsupported tensor dimensions,
shape mismatches and non-matrix targets. It accepts one complete file, not a
sharded adapter set. Big-endian adapter payloads and GGML tensor names of 64 or
more bytes are outside the admitted profile. Filename bytes follow the model
loader's host boundary: arbitrary Unix path bytes, UTF-8 Windows paths.

The existing C context boundary still permits only admitted dense Llama models.
Token embedding LoRA on a tied output model is rejected when selected: the
frozen graph applies the same pair to an incompatible output matmul. This
checked error replaces a source assertion; it does not claim tied-head parity.

## Metadata and borrowed token storage

Only scalar GGUF fields enter the metadata map. Arrays, including invocation
tokens, are absent from scalar lookup. Raw key/value bytes are preserved through
the first C NUL, including invalid UTF-8. First insertion wins for native key
aliases. Enumeration uses the measured MSVC or GNU map order, and floating-point
scalar formatting uses the matching compiler profile.

String getters return the full byte length. A positive-capacity output receives
at most capacity minus one bytes followed by NUL. Zero capacity does not write;
a null output is accepted only at zero capacity. Missing keys and invalid
indices return -1 and write an empty string when capacity is positive. The
invocation-token pointer borrows immutable adapter-owned storage and remains
valid until that handle is released. The null invocation-count query returns
zero. Other invalid null/misaligned pointers produce checked diagnostics when
they can be detected without dereferencing foreign memory.

## Applying and replacing LoRA

Zero-scale entries are ignored. For repeated handles, the first nonzero entry
wins. Passing an empty list clears LoRA. When the nonzero count, identities and
scales equal the active map, the source fast path retains existing map order
even if the caller reorders its list. Otherwise the replacement uses the
source pointer hash, bucket growth and iteration policy. Multiple overlapping
adapters are applied sequentially in that order, preserving float rounding.

Pointer allocation can change the order of an MSVC two-adapter map when both
identities occupy the same bucket. The fixtures independently capture both
orders; they do not assume that reference and Rust allocators return identical
addresses. Separate deterministic pointer-map traces check both native policies
through multiple rehash boundaries. GNU and MSVC numerical expectations are
retained separately, with exact F32-word comparisons on each qualified host.

An adapter must belong to the context's base model. Scales must be finite and
there may be at most 256 entries in one setter call. Null pointer arrays are
valid only with zero count. A zero-scale adapter pointer is not dereferenced.
On success, replacing or clearing adapters retains already evaluated KV state,
source-style. It also retains the previous output buffers until the next
successful decode. On failure, active adapters, control vectors, cache and
outputs are unchanged. The separate strict Rust `Context` retains its existing
`replace_adapters_and_reset` contract.

## Control-vector updates

Control rows exist for layers 1 through `n_layer - 1`; layer zero is never
modified. For layer i, the input row starts at `(i - 1) * n_embd`. Only complete
rows are copied. A short trailing row preserves the previous value of that
layer, including previously zero-initialized rows. Complete input rows are
retained even when outside the currently active layer range.

The active range is inclusive. Negative, reversed or out-of-range endpoints
retain the source's intersection behavior. A non-null, zero-length call can
reactivate stored rows under a new range. Null data disables the active range
without erasing stored rows and ignores the other parameters. A non-null call
with a mismatched embedding width fails without changing the old configuration.
Consumed complete rows must be finite. The runtime adds active rows after the
corresponding layer's residual output.

## Ownership and resource boundaries

A model owns all loaded adapter handles and frees those not explicitly released.
Manual free removes the handle from that registry. Contexts already own a strong
reference to their base model, so releasing the public owning model handle does
not invalidate a live context or its model's adapters. Final model destruction
releases the remaining handles. Callers must not use or free stale handles.

The active Rust context retains immutable owned adapter data and a validated
execution copy. Transactional native decode shares that immutable execution
copy; it does not clone all adapter matrices per microbatch. Explicitly freeing
an active handle cannot leave Rust execution with a dangling weight pointer.
This is a stronger ownership guarantee than the source caller-lifetime rule.
The adapter file may be replaced or removed after a successful load.

The loader bounds metadata headers to 32 MiB, aggregate metadata storage to
64 MiB and tensor entries to 32,768. It preflights decoded and packed weight
storage before reading matrices. A model admits at most 1,024 adapter handles
and at most 512 MiB of aggregate retained adapter payload. Context composition
also bounds original plus replacement adapter storage to 512 MiB. The native
context checks its two-GiB owned working-set limit before allocating the
replacement execution copy. Decode accounts adapter storage and rank-dependent
projection/activation scratch in its existing ledger. These are owned-payload
budgets, not process-RSS or allocator-overhead guarantees.

All existing foreign-memory rules remain in force: live aligned allocations,
correct lengths, exclusive context access, valid terminated strings and handles
from this library. Allocation failure, nonfinite numerical output and invalid
input return the documented checked error without partial publication. Rust
allocator aborts and invalid foreign allocations cannot be recovered.

## Independent evidence and reproduction

`fixtures/ffi/adapters` retains 47 GGUF inputs, scenario definitions, compiler
specific reference records, exact deduplicated byte/word payloads and provenance
hashes. Expectations came from original libraries before Rust replay. The
external packer only stores original bytes and words; it computes no numerical
answer. No C/C++ implementation, source oracle or reference binary is maintained
inside this product repository.

Each compiler profile covers:

- 44 load cases: 34 accepted and 10 rejected; all scalar index/lookup buffers
  at capacities 0, 1, 2, 8, 64 and 160, plus invocation-token bits;
- 107 execution scenarios and 333 decodes: 170,496 exact F32 logit words;
- every dense projection target, whole-model adapters, F32/F16/BF16 weights,
  absent/zero/positive/negative alpha, positive/negative/zero scale, duplicate
  handles, overlapping adapters, clear/reapply and retained KV continuation;
- token and direct embedding inputs, partial control rows, disabled/reactivated
  vectors, inclusive range boundaries, width errors and combined LoRA/control;
- pointer-map traces for 0, 1, 2, 3, 7, 8, 13, 17, 29, 64 and 128 identities.

Rust-only safety tests additionally verify wrong-model and invalid-input
rollback, ownership after model-handle release, automatic versus manual free,
file replacement after load, source-asserting tied embedding rejection and
resource-overflow checks. The same original-header C client is run against the
Rust debug and release libraries on both hosts. Export inventories are checked
against all 255 original C exports; the adapter checkpoint implements 226,
retains 29 absent functions and adds only the four existing Rust extensions.
Detailed command, binary and output identities are retained in
`evidence/2026-11-11/c-adapters`.

Run `cargo test --locked --offline -p llama-ffi adapter::tests` for this corpus,
and `cargo test --locked --offline -p llama-ffi -p llama-model-runtime` for the
full affected crate suites. The coverage and evidence qualify this stated
adapter subset; they do not close all ABI, architecture, platform, scheduler,
backend or performance requirements.
