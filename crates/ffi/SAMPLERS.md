# CPU sampler C ABI

The C sampler implementation executes the workspace's Rust behavior samplers.
The original llama/ggml DLLs are only external test oracles. It adds 30 frozen
public C functions and one Rust layout inspection function. This is the CPU
sampling subset; grammar, DRY vocabulary preprocessing, adaptive-p, infill,
backend sampling, and sampler performance counters are not exported.

## Implemented surface

| Area | Functions |
|---|---|
| Generic interface | init, name, accept, apply, reset, clone, copy, free, get_seed |
| Chains | chain_default_params, chain_init, chain_add, chain_get, chain_n, chain_remove |
| Selection | init_greedy, init_dist, init_mirostat, init_mirostat_v2, sample |
| Filters and transforms | init_top_k, init_top_p, init_min_p, init_typical, init_temp, init_temp_ext, init_xtc, init_top_n_sigma |
| History and bias | init_penalties, init_logit_bias |

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
mapping across platforms. Exact native stream/order parity on non-MSVC targets
remains unverified; cross-platform Rust compilation is a separate gate.

The safe Rust behavior API keeps its existing validation, stable ordering, and
established portable random stream. Sampler::apply_native is a separate entry
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
Chains search children from the end. XTC and non-reporting stages return the
source's UINT32_MAX sentinel. Reset restores the original explicit-seed stream;
a default seed obtains fresh entropy. Clone preserves stream and history state.
The frozen default-seed Mirostat clone reports a freshly resolved seed while
copying the old RNG state; this observable source behavior is retained.

Copy requires the same sampler type and configuration. Chain copy requires the
same topology and preserves existing child handle identities. Custom
copy_state is used if provided; otherwise the wrapper clones the source context,
frees the destination's previous context, and transfers the cloned context
without replacing the destination handle. Copying a sampler to itself is a no-op.

llama_sampler_sample obtains an actual logit row from the Rust context, creates
the vocabulary candidate array, applies the sampler, checks the selected index,
accepts the selected token into the sampler, and returns its token ID. It does
not decode that token. A failure returns LLAMA_TOKEN_NULL (-1) and records the
thread-local diagnostic. Context logit indexing follows the same positive batch
positions and negative output-row conventions as llama_get_logits_ith.

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

Chain performance collection (no_perf = false) is explicitly rejected.
Grammar initialization/filtering, DRY's overlapping vocabulary sequence-breaker
construction, adaptive-p's pending-probability clone semantics, infill and
backend-attached sampling are pending separate work. The source's adaptive-p
clone omits its pending probability buffer; that unsafe reference case has not
been exposed through a guessed implementation.

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

Evidence, actual comparison counts, source/compiler hashes, PE export inventory,
and debug/release results are in evidence/2026-11-09/sampler-ffi. The prior core
ABI checkpoint remains recorded separately under evidence/2026-11-09/ffi.
Microsoft STL algorithm attribution and its complete license are preserved in
THIRD_PARTY_LICENSES.md.
