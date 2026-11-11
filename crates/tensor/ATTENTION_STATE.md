# Gated delta and DeepSeek V4 hyper-connections

Four original operation IDs execute through five explicit Rust constructors.
All inputs are immutable; every successful output owns independent storage.

| ID | Rust operation | Inputs in source order |
| ---: | --- | --- |
| 86 | `GatedDeltaNet { snapshots }` | Q, K, V, gate, beta, initial state |
| 88 | `Dsv4HcComb { epsilon, iterations }` | Mix channels, scales, base values |
| 89 | `Dsv4HcPre` | Stream embeddings, per-stream weights |
| 89 | `Dsv4HcPreGated { scale }` | Stream embeddings, per-channel gates |
| 90 | `Dsv4HcPost { with_comb }` | X, residual streams, post weights, optional mixing matrix |

## Gated delta state and addressing

V has shape `[S,H,T,B]`: state width, heads, tokens and sequences. Initial state
has exactly `[S,S,H,B]`. Gate is scalar or vector, with axis zero either one or S;
beta has axis zero one. Gate and beta must cover every destination head, token
and sequence, and all three state/gate/beta buffers must be contiguous.

Q and K have at least S channels and T tokens. Their heads repeat cyclically:
the source selects `head % source_heads`. Sequence groups instead repeat
contiguous groups, selecting `sequence / (B / source_sequences)`. Source
sequence counts must divide B. Q/K can have surplus heads, channels and tokens;
only the original kernel's prefix reads occur. Native controls cover these
surplus shapes, as well as grouped sequences and non-divisor head counts.

Q, K and V require contiguous physical inner rows. Their outer byte strides
remain effective, including row and sequence padding. Every scalar read stays
within its declared view. State uses the source's physical sequence stride and
the original transposed matrix storage: `state[j*S+i]` is matrix entry S[i,j].

Each token first multiplies state by `exp(gate)`, either per input channel or
once per head. It forms each delta as `(V - state_dot_K) * beta`, then performs
the K/delta outer-product update. Attention output is `state_dot_Q / sqrt(S)`.
Each dot product rounds every multiplication to F32, accumulates those products
in F64, and converts once to F32. The decay, subtraction, outer update and final
scale retain separate F32 operations; they are not fused multiply-adds.

Output shape is `[S*H, T*B + snapshots*S*B, 1, 1]`. It contains flattened
attention scores in `[S,H,T,B]` order followed by complete `[S,S,H,B]` state
snapshots. Snapshot zero is the latest state; later slots go backwards in time.
Snapshots must be between one and T. When the original caller asks for more
slots than tokens, older slots remain caller-owned/unwritten; this immutable
adapter rejects that domain rather than inventing old states.

## Hyper-connection arithmetic

`Dsv4HcComb` requires mixes `[24,T,1,1]`, a scale vector with at least three
entries, and base `[24,1,1,1]`. The original CPU supports exactly four streams.
Only mix/base channels 8 through 23 and scale entry two are read. Nonfinite
values in ignored fields do not cause a rejection, and an independent native
case checks that behavior.

For each source stream, the operator applies the source F32 affine transform,
max-subtracted scalar exponential, row softmax, then adds epsilon to each entry.
It normalizes columns once. Each subsequent iteration normalizes rows and then
columns; every normalization sum starts at epsilon. The result is
`[destination_stream,source_stream,token,1]`. Epsilon may be any finite F32,
including negative values and either zero sign. Iterations must fit positive
signed I32; native negative-epsilon and multi-iteration controls are preserved.

`Dsv4HcPre` reduces X `[E,H,T,1]` over streams using weights `[H,T,1,1]`.
The gated constructor instead accepts `[E,H,T,1]` gate values, applies
`1 / (1 + exp(-gate))` at each channel/stream/token, then multiplies the
stream sum by its explicit scale. The ungated constructor uses scale one,
matching the original public API.

`Dsv4HcPost` starts from `X[channel,token] * post[destination,token]`.
With no mixing matrix it adds that destination stream's residual. With a
`[H,H,T,1]` matrix it adds each residual source stream in original source order,
using matrix `[destination,source,token]`. Result shape is `[E,H,T,1]`.

The three hyper-connection kernels read every original byte stride, including
nonunit axis-zero strides and padded outer axes. They retain F32 reduction
order. Unlike the combination kernel, PRE and POST accept arbitrary positive
stream counts.

## Checked domains and resource limits

All source dimensions fit positive signed I32 and all byte strides are aligned
to F32 storage. Gated delta also checks that the squared state width fits the
original signed-int vector-helper length. Shape relationships, sequence groups,
address arithmetic and view bounds are checked before unsafe-sized work.

This slice admits finite values actually read by the kernel and finite returned
values. It returns an explicit `Unsupported` error for nonfinite reads or
nonfinite results, including overflow from finite inputs. This is a remaining
source-success domain, not an assertion that the original kernel rejects it.
Unreferenced mix/scale/base entries remain ignored.

Each new attention operation limits its output payload to 256 MiB. Requests
exceeding that scalar-adapter budget return `Unsupported` before output/scratch
allocation or recurrent work. This protects against large state-snapshot
amplification and huge logical tensors backed by tiny overlapping views.
Allocation is fallible, size/address calculations are checked, and iterative
loops check cancellation. A global scheduler or caller-configurable allocation
planner remains open.

The accompanying vision/loss follow-up preflights complete flat source spans
before allocating window/relative outputs. It retains nonmonotonic unpartition
addresses when stored windows are smaller than the requested window size.
Window/relative output payloads have the same 256 MiB limit; scalar cross-entropy
work is limited to 64 million logical cells.

## Independent reference evidence

The primary corpus contains 867 actual original CPU graphs per host and
5,185,540 output bytes. It covers S=1,3,7,16,33, cyclic heads, grouped sequences,
scalar/vector decay, padded rows, single/multiple snapshots, aligned nonunit
strides, several normalization iterations/epsilons, arbitrary stream counts,
both post forms and sigmoid saturation/subnormal/signed-zero values.

A second corpus contains 89 original graphs and 40,244 output bytes per host.
Sixteen admitted controls add 32,096 exact output bytes, including surplus
GDN dimensions, three-worker disjoint-output cases, negative epsilon and ignored
nonfinite fields. The other 73 graphs succeed in the original C implementation
but are explicitly rejected by the Rust finite-domain boundary. All 8,148 bytes
of their original successful output remain untouched and are read by the test.

Thus Rust executes **883 independently captured graphs**, comparing
**5,217,636 output bytes per host** exactly. The remaining 73 original successful
graphs assert checked rejection. No tolerance, NaN comparison exemption or
replacement expected output is used. The native generators import no Rust,
and the primary capture predates this implementation. Fixture/provenance hashes,
validation reports and external-generator hashes are recorded under
`evidence/2026-11-11/tensor-attention-state`.

These results establish the measured scalar CPU arithmetic on the original
x86_64 MSVC and GNU profiles. Native one/three-worker controls have disjoint
outputs; this Rust adapter remains serial. Other ISA paths, floating exception
flags, non-default rounding/FTZ/DAZ modes, GPU/device execution, scheduler
integration, and full gated-delta/DeepSeek model execution remain open.

Reference-derived fixtures and implementation retain the GGML MIT attribution
in root `NOTICE` and `LICENSE-UPSTREAM`. All reference C/C++ programs and
native binaries remain outside the product.
