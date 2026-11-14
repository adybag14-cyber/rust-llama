# CPU sampler C ABI

The C sampler implementation executes the workspace's Rust behavior samplers.
The original llama/ggml DLLs are only external test oracles. The CPU sampler
surface implements 36 frozen public C functions, including all 20 ordinary CPU
sampler constructors, and one Rust layout inspection function. Eight additional
context-sampling functions and the admitted CPU prefix/suffix execution are
documented in BACKEND_SAMPLING.md. Three additional sampler-chain performance functions
and the global logger are documented in PERFORMANCE.md.

## Implemented surface

| Area | Functions |
|---|---|
| Generic interface | init, name, accept, apply, reset, clone, copy, free, get_seed |
| Chains | chain_default_params, chain_init, chain_add, chain_get, chain_n, chain_remove |
| Selection | init_greedy, init_dist, init_mirostat, init_mirostat_v2, sample |
| Filters and transforms | init_top_k, init_top_p, init_min_p, init_typical, init_temp, init_temp_ext, init_xtc, init_top_n_sigma |
| History and bias | init_penalties, init_logit_bias, init_dry |
| Adaptive selection and infill | init_adaptive_p, init_infill |
| Grammar | init_grammar, init_grammar_lazy, init_grammar_lazy_patterns |

Every name in this table has the llama_sampler_ prefix. In particular,
init_penalties has the frozen **five** arguments: vocabulary size, history
length, repeat, frequency, and presence. It does not use an older four-argument
llama.cpp signature.

The public llama_token_data, llama_token_data_array, llama_logit_bias,
llama_sampler_chain_params, llama_sampler, and twelve-slot llama_sampler_i
layouts match the frozen header. The additional inspection extension is
size_t llama_rust_sampler_abi_layout(size_t index); its 36 entries are listed in
ABI_LAYOUT.md. Built-in backend callback slots are NULL because this layer does
not expose ggml graph sampling. A custom interface can provide its own callbacks;
CPU wrappers dispatch only the corresponding CPU slots.

## Candidate behavior

The selected field is an index into the active candidate array, not a token ID.
Terminal samplers set it. Filters retain the previous index exactly as the source
does, even if shrinking or reordering makes it stale. A subsequent terminal
sampler must establish the selection before the application uses it.

The implementation preserves source ordering and flags, including the source's
unchanged sorted flag after logit bias. XTC advances the caller's data pointer
past the discarded prefix; it reduces size and leaves the original allocation
owned by the caller. Retained XTC probabilities are not renormalized. The caller
must free its original allocation, not the advanced pointer.

Native top-p's adaptive 256-candidate sorting buffer is reproduced: tied token
order can differ from a complete sort. On the verified MSVC target, heap,
bucket, typical-score, and full-sort tie order and the standard library's
MT19937 canonical random-number extraction are reproduced. These choices matter
for seeded selected tokens even when all probability values match. The original
C++ standard library does not promise identical tie order or random distribution
mapping across platforms. The context-sampling checkpoint separately verifies
GNU/Linux typical-score/full-sort ordering and seeded distribution on its
512-candidate CPU suffix corpus. The GNU ordering policy is shared in algorithm
with the independently verified tensor implementation. Complete native stream
and sampler-domain parity on other targets remains open.

Repeated candidate IDs are accepted by the C compatibility path because the
source operates on array positions; infill and bias have observable duplicate-ID
behavior. The safe Rust behavior API retains its unique-ID validation, stable
ordering, and established portable random stream. Sampler::apply_native is a separate entry
point for the frozen C behavior. Errors in one built-in apply validate before
committing either its state or candidate array. A chain applies stages in order:
if a later stage fails, successful preceding stages remain applied. This follows
the source's incremental chain behavior; there is no chain-wide rollback claim.

Disabled constructor settings create the source's real empty samplers with its
?name labels. An empty sampler has the source's no-op semantics. Empty candidate
arrays are accepted for the defined empty/greedy/dist cases. Greedy selects index
0 even when every logit is masked, as the source does; callers must check the
array size before reading that index. Dist returns selected = -1 on an empty
array.

Logit bias preserves duplicate semantics: direct token-ID slots receive all
matching biases, while the fallback search applies the first matching entry.
Negative or absent bias token IDs have no effect. A negative-infinity bias can
mask a candidate.

## RNG, history, clone, reset, and copy

Explicit seeds reproduce the tested native streams. The default UINT32_MAX
sentinel resolves through the operating-system random source: BCryptGenRandom
on Windows and /dev/urandom on Unix. An unsupported OS must provide an explicit
seed. OS RNG failure returns an error; there is no deterministic fallback.

get_seed reports the source's seed for dist and the two Mirostat variants.
Chains search children from the end. XTC, adaptive-p and non-reporting stages return the
source's UINT32_MAX sentinel. Reset restores the original explicit-seed stream;
a default seed obtains fresh entropy. Distribution/Mirostat clones preserve the
source stream and history behavior. The frozen default-seed Mirostat clone reports
a freshly resolved seed while copying the old RNG state.

Adaptive-p acceptance commits its EMA only for the pending selected token.
Defined adaptive clones preserve EMA and restart the RNG at the requested seed;
a default-seed clone obtains fresh OS entropy. The original clone drops its pending
probability buffer, making clone-before-accept unsafe. This C implementation
returns NULL with a diagnostic for that case; copy also rejects it before changing
the destination. Accept or reset the pending selection before cloning. The safe
Rust Clone implementation continues to preserve every state field.

Copy requires the same sampler interface/type. Existing numerical builtin
samplers require identical configuration; DRY, grammar, infill and bias replace
the destination state with the source clone, matching their source fallback-copy
behavior. Chain copy requires the same topology and preserves existing child
handle identities. Chain copying is incremental if a later child fails. Custom
copy_state is used if provided; otherwise the wrapper clones the source context,
frees the destination's previous context, and transfers the cloned context
without replacing the destination handle. Copying a sampler to itself is a no-op.

llama_sampler_sample obtains the context row and returns an already selected
backend token when present. Otherwise it constructs candidates from the sampled
logits and IDs, or from the ordinary full-vocabulary row, then applies the CPU
stages and checks the selected index. It accepts the selected token once and
returns its token ID. BACKEND_SAMPLING.md defines initialized prefix skipping. It does
not decode that token. A failure returns LLAMA_TOKEN_NULL (-1) and records the
thread-local diagnostic. Context logit indexing follows the same positive batch
positions and negative output-row conventions as llama_get_logits_ith.

## Vocabulary-dependent samplers and raw-byte grammar

DRY performs the frozen vocabulary overlap search on actual detokenized words.
It skips null/empty breakers, retains at most 64 distinct breakers after
truncating each to 40 bytes, tokenizes unmatched suffixes with the tokenizer's
raw-byte API, and retains at most 20 tail tokens. Every overlapping occurrence
is considered. Signed allowed_length, source window/reset behavior, clamped
Z-array reuse, and the source double-precision power intermediate are preserved.

Infill performs native softmax, EOG selection, prefix-piece merging, the two
probability-threshold passes, and EOT/EOS fallback. It preserves the source
selection/sorted metadata and duplicate candidate-ID behavior. Missing fallback
tokens produce an explicit diagnostic.

The grammar compatibility engine is separate from the stricter character
Grammar/Recognizer API. NativeGrammar implements source token terminals,
numeric/inverse/named tokens, partial byte decoding, leading-NUL termination,
EOG masks and the source's distinct mask-versus-accept transitions. Active
filtering changes only rejected logits to negative infinity. It preserves
candidate order, probabilities, size, pointer and selection metadata.

Lazy token triggers, legacy word triggers and ECMAScript byte patterns retain
buffered token boundaries. Activation replays the suffix beginning at the first
nonempty capture, or the whole-match start when no capture is nonempty. A match
inside a token replays that token's suffix with its original token ID. Clone and
copy retain trigger buffers and grammar state; reset restores the awaiting state.
Null/empty grammar is unconstrained and does not dereference unused arguments.

The bounded regex engine reproduces the measured MSVC C-locale profile, including
capture resets, backreferences, lookahead, greedy/lazy order and raw byte offsets.
The frozen legacy word helper's MSVC replacement of metacharacters with a literal
dollar-zero is preserved. Cross-vendor std::regex and locale equivalence remains
unverified; these platform-specific semantics are explicit in the fixture
provenance. No regex error is converted to an unconstrained successful result.

Grammar and infill own bounded vocabulary snapshots. DRY owns its processed
breakers. They remain valid after the original model handle is released; the
constructors still require a live vocabulary while building those owned values.

## Ownership and callback requirements

A chain takes ownership when a child is added. chain_get returns a borrowed
pointer; index -1 returns the chain itself. chain_remove transfers ownership
back to the caller. Freeing a chain recursively releases all children. Deep
clone allocates independent children. The implementation rejects duplicate
ownership, cycles, freeing an owned child, and detected foreign or duplicate
frees before dereferencing an unregistered handle.

All sampler shells, including custom ones, must be created by this DLL's
constructors or llama_sampler_init. A custom clone callback must use the same
library's init function and return a new unowned shell. Its free callback frees
its context only; the generic wrapper frees the shell. A context-free custom
sampler can omit clone and free callbacks. Interfaces and custom contexts must
remain live for every callback and every clone that refers to them.

Each sampler and its complete owning chain tree require exclusive access during
a call. Callbacks must not unwind, free or reenter the active tree, or retain
transient array pointers. Separate trees can be used concurrently. Built-in
interfaces and opaque contexts must not be replaced or modified by the caller.
A registry can reject known invalid handles but does not make arbitrary foreign
pointer dereferences or stale-pointer address reuse safe.

All C exports and built-in vtable entry points contain Rust unwinding. Invalid
foreign allocations, process aborts, and foreign callbacks that violate their
contract cannot be recovered. Diagnostics follow the README's thread-local
last-error contract; successful calls do not clear earlier errors.

## Explicit restrictions

Candidate and history capacities are capped at one million elements. Chains
allow at most 128 direct stages and 64 nested levels. Probabilities must be
finite and nonnegative, token IDs nonnegative, and logits finite or negative
infinity. Parameter values must be finite; repeat factors must be positive.
Penalty acceptance must be inside the declared vocabulary. Mirostat learning
rates must be nonnegative and v1 needs a supported, nondegenerate slope.

The implementation rejects source-invalid numerical cases, including a
zero-probability typical-sampling entropy and an undefined Mirostat float-to-int
conversion. It does not publish a fake successful selection for those cases.

Chain performance collection supports both no_perf settings, with real monotonic
timing and accepted-token counts. PERFORMANCE.md specifies counter ownership,
reset, clone and copy behavior. Unsupported backend sampler stages and llguidance
integration remain outside this surface.

Vocabulary snapshots are capped at one million entries and 256 MiB of pieces.
DRY preprocessing bounds generated sequences and tail storage; exponent overflow
outside the source's defined integer domain fails explicitly. Infill's quadratic
prefix-merging path is capped at 4,096 candidates; its earlier EOG-only path can
handle larger supported arrays. A real source assertion or invalid allocation
is represented by a checked error.

Lazy grammar buffers are capped at one MiB, with at most 128 trigger patterns.
Each pattern is limited to 64 KiB, 16,384 nodes, 128 captures and 128 nesting
levels. Matching has a shared five-million work-unit budget, charges compared
backreference bytes, and bounds queued allocation capacity to 64 MiB before
cloning continuations. NativeGrammar separately bounds parser expansion,
interned symbols, stack states and input pieces. All such failures propagate
through the C error channel. Grammar acceptance commits only after the complete
token or lazy replay succeeds.

## Verification

The independent native harness is outside the product at
C:/Users/adyba/rust-llama-reference/sampler-oracle. It includes the unchanged
frozen public header and dynamically loads both independent libraries. It checks
the layout, constructor names/defaults, seeded streams, selected indices, sorted
flags, XTC pointer advancement and removed-prefix writes, ties near sorting
cutoffs, clone/copy/reset, custom vtable dispatch, child ownership transfer,
default-seed relationships, error handling, and actual-context sampling.

It also checks the C vocabulary-only WordPiece path: all vocabulary metadata
and both piece modes, tokenization, all detokenization flag combinations,
count-only sizes, and partial-buffer writes. Native detokenization reports the
pre-cleanup required capacity; a successful call can return fewer bytes after
space cleanup. This buffer convention is distinct from the high-level Rust
detokenize_bytes return length.

The new frozen-DLL corpus covers 34 grammar/lazy scenarios (173,568 candidate
mask decisions), 55 DRY scenarios (357,888 candidate rows), 35 adaptive-p scenarios
(1,925 applications), and 41 infill scenarios (164 applications). Every recorded
bit and state field matches in debug and release. Seven new chain configurations
on six trained prompts produce exactly 585 native tokens across 42 contexts.
The separate raw grammar engine corpus checks 383,053 decisions, and the regex
corpus checks 12,349 compile/match/capture outcomes.

Ordinary Rust tests replay the committed vocabulary-only and native reference
fixtures without a native library. They also verify pending adaptive clone/copy
errors, failed grammar acceptance, ignored empty-grammar arguments and model-free
sampler lifetime. Full trained-model execution remains an external native gate.

Evidence, source/compiler hashes, PE export inventory, native BPE/FIM/raw-byte
checks and debug/release results are in evidence/2026-11-09/sampler-cpu. The
previous sampler checkpoint and core checkpoint remain preserved under
evidence/2026-11-09/sampler-ffi and evidence/2026-11-09/ffi.
Microsoft STL algorithm attribution and its complete license are preserved in
THIRD_PARTY_LICENSES.md.
