# Context-attached CPU sampling

This checkpoint implements the eight frozen context-sampling exports with
Rust-owned output buffers and real Rust execution of the admitted sampling
prefix. The external original libraries are verification oracles only.

| Function | Behavior |
|---|---|
| llama_set_sampler | Attach a supported nonempty sampler chain to a sequence; a null sampler detaches it. |
| llama_get_sampled_token_ith | Read the selected token, or LLAMA_TOKEN_NULL when this output has no terminal selection. |
| llama_get_sampled_logits_ith | Read the transformed/gathered logit buffer, or NULL when absent. |
| llama_get_sampled_logits_count_ith | Read that output's logit count and the source's unallocated-buffer fallback. |
| llama_get_sampled_probs_ith | Read a probability buffer when produced; the admitted prefixes produce none. |
| llama_get_sampled_probs_count_ith | Return the probability count; it is zero for the admitted prefixes. |
| llama_get_sampled_candidates_ith | Read gathered token IDs, or the source's full-vocabulary ID fallback. |
| llama_get_sampled_candidates_count_ith | Read the gathered candidate count, independently of the fallback pointer. |

Symbol availability does not imply that every frozen backend sampler is
implemented. The supported execution domain is the existing dense Llama native
decoder on CPU. BERT encoding, other native architectures, GPU sampling and
caller-supplied GGML graph callbacks remain outside this checkpoint.

## Prefix and suffix execution

The original chain initializes a contiguous prefix of stages that have backend
implementations. The first stage whose original interface has no backend
initializer ends that prefix; the remaining stages execute through the ordinary
CPU sampler interface. Rust preserves this distinction rather than treating an
unsupported backend stage as an implicit CPU boundary.

The admitted prefix stages are:

| Stage | Native behavior |
|---|---|
| Source-disabled empty sampler | No operation; preserves the source's ?name label. |
| Greedy | Rust tensor ARGMAX selects the last tied maximum, matching the backend operator. |
| Top-k | Rust tensor TOP_K preserves the captured MSVC/GNU partial-sort order and gathers logits and token IDs. |
| Positive temperature | Scale logits by the F32 reciprocal, with the source multiply-plus-zero operation order. |
| Zero/negative temperature | Gather the last-argmax token into a one-candidate output. |

This backend greedy tie policy differs from the existing standalone CPU greedy
sampler, whose first-maximum behavior remains unchanged. Top-k must fit its
current candidate width. Source-disabled top-k settings are real empty stages,
not an oversized gather. The entire prefix is validated before attachment,
including native tensor ordering and shape checks.

Typical, XTC, Mirostat, adaptive-p, DRY, infill, grammar and other source CPU-only
stages can establish a CPU suffix. A distribution sampler after such a boundary
uses the existing CPU distribution implementation. A distribution, top-p, min-p,
dynamic-temperature, penalty or logit-bias stage encountered in the backend
prefix is a checked unsupported error. Nested backend chains and custom GGML
backend callbacks are also rejected. The admitted prefix does not create a
probability tensor; the probability getters' NULL/zero result is the measured
original behavior for these configurations, not a claim of backend distribution
support.

Initialized numerical prefix stages expose the original +name labels. Ordinary
chain apply skips precisely that initialized prefix. llama_sampler_sample first
uses a selected token already produced by decoding. If no terminal prefix stage
selected a token, it builds the CPU suffix input from the sampled logits and
candidate IDs, preserving their source ordering. It accepts the selected token
once and retains the chain's actual acceptance/performance accounting.

## Attachment and lifetime

A caller may provide llama_sampler_seq_config entries through the existing
context parameters or call llama_set_sampler after creation. The x64 sequence
configuration has the frozen 16-byte layout, with sequence ID at offset zero and
sampler pointer at offset eight. Original-header callers exercise the constructor
configuration directly.

The caller owns each attached chain. The context holds a checked lifetime lease:
freeing an attached chain returns a diagnostic instead of producing a dangling
context pointer. Detachment or context destruction releases that lease. Free the
chain after detaching it or after freeing the context. Independent contexts and
chains require the existing exclusive-call discipline.

An initialized chain cannot be initialized again or have children added/removed.
This checked boundary avoids the original's unsafe reinitialization/topology
cases. Cloning creates a fresh, unattached chain, with unprefixed child names.
Copying compatible state preserves the destination's initialization, child
identities and prefix names. Numerical configuration compatibility is unchanged.
Reset preserves backend initialization and resets the ordinary CPU suffix state.

A null sampler detaches and returns true. An empty chain or a nonchain produces
false and removes the prior attachment, following the source's rejection path.
Unsupported prefix validation leaves an existing attachment unchanged. Source
calls that reinterpret a direct greedy sampler as a chain can abort; Rust returns
a checked rejection instead.

## Output indexing, buffers and transactions

Positive indices refer to original caller batch token positions; negative indices
count backward through emitted output rows. Interleaved sequence execution does
not change that indexing. A batch token uses the sampler attached to its first
sequence ID. The output limit counts all listed sequence memberships.

The existing default n_outputs_max_per_seq of one is enforced for every attached
sequence. Nondefault output-capacity parameters remain unsupported. Too many
requested outputs are rejected before native memory or previous output buffers
change. At most 256 chains may be attached, vocabulary size is bounded to one
million, and checked output/scratch accounting is bounded to 256 MiB in addition
to the native context's existing memory limits.

The original accessor fallbacks are deliberately distinct:

- Before sampling buffers exist, sampled logit count is the vocabulary size even
  though the sampled-logit pointer is NULL. Probability/candidate counts are zero.
- Once sampling buffers exist, an unselected or invalid row has zero counts and
  null float pointers. Its token is LLAMA_TOKEN_NULL.
- The candidate pointer falls back to a full-vocabulary ID array even when its
  reported candidate count is zero.
- llama_get_logits_ith prefers sampled logits when they exist and otherwise uses
  the normal full-vocabulary row.
- Detaching retains the most recent buffers until the next successful decode.

Returned non-null buffers are writable context-owned storage. Their addresses
remain valid until the next successful decode or context destruction. Getter
mutations are observable on later reads and CPU suffix input. As in the existing
native C contract, callers must not use freed, misaligned or undersized pointers.

Sampling execution runs through the native decoder's output-consumer hook before
staged KV memory, deferred operations, statistics or outputs are committed. An
error or cancellation preserves the previous context state and output buffers.
The runtime regression explicitly rejects an output consumer after computation,
checks state/statistics preservation, and then retries successfully.

For fully sampled rows the original can leave the contiguous raw llama_get_logits
storage unfilled; its bytes are not part of this claim. Rust retains the computed
raw logits. Use llama_get_logits_ith or the sampled accessors for the defined
sampled-row contract.

## Independent acceptance

fixtures/ffi/backend-sampling contains separately captured original-header
MSVC and GNU observations. Each platform records 26 scenarios and all accessor
fields through initial state, attachment, decode, invalid indices, detachment and
subsequent decode. The corpus includes mixed sequence ordering, absent/empty
samplers, tied greedy/top-k behavior, positive/zero/negative temperatures, CPU
suffixes, initial sampler configuration and output-limit rejection/recovery.

A second original-header corpus records the complete typical and distribution
CPU-stage candidate arrays for both single-sequence and interleaved contexts.
It exposed a pre-existing GNU fallback to stable ordering. The behavior crate
now uses the same independently expressed GNU sorting policy already verified by
the tensor crate. Every candidate ID, F32 logit/probability word, selection and
sorted flag is checked exactly. This expands GNU evidence for this suffix
configuration; it does not certify every ordinary sampler's full GNU input domain.

Ordinary FFI tests replay the independent data without a native library. External
clients compiled against the untouched frozen header also replay both corpora
against debug/release Rust DLLs and shared objects. The focused evidence directory
records exact commands, artifact hashes, test results, PE/ELF inventories and
retained failures. No Rust output generates expected numerical values. See
evidence/2026-11-14/c-backend-sampling.

Full C ABI, sampler/backend, architecture, platform and performance release gates
remain open. This is a bounded CPU implementation checkpoint.
