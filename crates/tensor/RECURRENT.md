# Frozen recurrent CPU operators

This is an executable scalar Rust slice for original operation IDs 76, 77, 82,
83 and 84. It does not implement a complete recurrent model loader, model
architecture or parallel graph scheduler. The implementation is
`src/cpu_recurrent.rs`; `Op` registration and dispatch are in `graph.rs` and
`cpu.rs`. All numerical inputs and outputs are F32 except the I32 SSM state IDs.
Dimensions and pointer arithmetic are checked before accesses. Outputs own their
storage; source inputs, including state tensors, remain immutable.

The source baseline is the frozen archive whose SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Constructors are in `ggml/src/ggml.c`, CPU kernels in
`ggml/src/ggml-cpu/ops.cpp`, and the SSM softplus helper in
`ggml/src/ggml-impl.h`. The independently built reference libraries and capture
generators remain outside the product. The recurrent source dependencies were
also checked against the user's frozen `llama.cpp-master (4)` directory before
native capture.

## API and source operand order

Dimension zero is the fastest axis. Shapes below include unit trailing axes
where needed; byte strides are retained in every native fixture.

| Rust constructor | Original ID | Input order |
|---|---:|---|
| `SsmConv` | 76 | X, convolution coefficients |
| `SsmScan { snapshots }` | 77 | Initial state, X, delta, A, B, C, state IDs |
| `RwkvWkv6` | 82 | K, V, R, time-first, time-decay, initial state |
| `GatedLinearAttn { scale }` | 83 | K, V, Q, gate, initial state |
| `RwkvWkv7` | 84 | R, W, K, V, A, B, initial state |

Missing or surplus inputs are graph arity errors. Source F32-only kernels do not
implicitly cast other storage dtypes.

### SSM convolution

X has shape `[K - 1 + T, C, S, 1]`, coefficients have shape `[K, C, 1, 1]`,
and output has shape `[C, T, S, 1]`. K must be positive. X requires a packed
first-two-axis plane, while its physical sequence stride is honored. Coefficients
require a four-byte scalar stride.

The original one-worker loop advances coefficient rows by K scalar words even
when their declared row stride includes padding. Rust preserves that addressing
and checks every read against the declared view span. It does not silently
reinterpret the coefficients as a logical strided matrix. A descriptor whose
flattened read would escape its view returns `Bounds`, even if a larger parent
allocation contains bytes there.

Each output is a left-to-right F32 multiply/add reduction. No F64 accumulator or
fused multiply-add is substituted. Empty output follows the original dispatcher:
it returns an empty tensor without running the numerical kernel. The original
multi-worker coefficient-row stepping can differ on padded layouts; that
scheduling mode is not exposed by this scalar implementation. Dense three-worker
original controls are captured separately.

### SSM scan and snapshots

| Input | Shape and layout |
|---|---|
| State | `[N, D, H, stored_sequences]`; contiguous F32 storage |
| X | `[D, H, T, S]`; first two axes packed, token and sequence strides honored |
| Delta | `[H, T, S, 1]`; contiguous |
| A | `[N, H, 1, 1]` for per-state decay or `[1, H, 1, 1]` for per-head decay; contiguous |
| B and C | `[N, G, T, S]`; matching shapes, packed first two axes, physical outer strides |
| State IDs | `[S, 1, 1, 1]`; I32 with four-byte scalar stride |

Heads must divide into G nonempty groups. Head h selects group
`h / (H / G)`. Destination sequence s begins from source state
`state_ids[s]`; repeated or permuted valid IDs are allowed, with independent
destination state. Negative IDs and IDs beyond stored state return `Bounds`.

The original helper computes delta as `x > 20 ? x : logf(1 + expf(x))`.
It is deliberately not changed to `log1pf(expf(x))`. Per-state decay computes
the exponential separately for each state element; per-head decay shares it
across a head. State updates and C reductions preserve source F32 operation order.

The result is a one-dimensional tensor. It contains flattened Y first, followed
by `snapshots` state slots. Each slot has `N * D * H * S` words. Slot zero
is the state after the final token. Subsequent slots contain states in reverse
time order: slot one is after the penultimate token, slot two after the token
before that, and so on. Within every slot, destination sequences retain input
sequence order.

All state/head/token/sequence dimensions must be nonempty. Snapshots must be in
`1..=T`; per-state A requires exactly one snapshot. The source accepts some
larger snapshot counts but leaves output state slots unwritten, so Rust returns
a checked parameter error for that domain. Zero-token scans also have unwritten
native state and are rejected.

The native destination uses source-state strides. For multiple destination
sequences or snapshots, the state sequence stride must cover exactly one full
state. Snapshot copies additionally require the canonical state head stride.
Unusual unit-axis strides that could cause overlapping or unwritten destination
state are explicit layout errors. Flattened state loops must fit signed I32.

### RWKV6, gated linear attention and RWKV7

Ordinary token inputs are contiguous `[W, H, T, ...]` tensors. The original
kernels use their first three axes and ignore additional fourth-axis slabs;
Rust preserves that behavior and validates the accessed first slab. RWKV6's
time-first operand needs at least `W * H` contiguous scalar values and is
shared across tokens. Its unused surplus values are not evaluated.

The number of sequences is `state.ne[1]`. Initial state is contiguous with
exactly `W * W * H * sequences` elements. Tokens must be a positive multiple
of sequences, so each sequence owns an equal nonempty consecutive token run.
Zero-sized heads, widths, tokens or sequences are rejected before source-style
division or state indexing.

The output shape is
`[W * H, T + W * sequences, 1, 1]`: flattened model output followed by all
final state matrices. Sequence transitions reload that sequence's initial state;
tokens within a sequence use the newly updated state. State is never written
back into the input tensor.

RWKV6 retains separate output and state formulas: time-first affects the
current output contribution, while time-decay affects stored state. Gated linear
attention applies the explicit F32 scale to Q, updates gated state, and then
accumulates output. RWKV7 first reduces A against the previous state row, then
updates every row element using W, V, K and B before accumulating R-weighted
output. This two-pass order is preserved.

## Floating-point and worker scope

The measured MSVC and GNU reference builds both select the source's scalar
recurrent branch. GNU's SSE2 macro alone does not enable `GGML_SIMD` in these
kernels: the relevant source mapping requires SSE3 or a later supported profile.
This differs from the separate SSE2 row-math paths used by some existing tensor
operators.

Finite arithmetic, signed zeros, gradual underflow, infinities and admitted NaNs
are compared as raw output bytes on each matching host. A single NaN operand
propagates its original sign and payload with the quiet bit set. If a multiply
or add actually encounters two different quieted NaN encodings, evaluation
returns:

`Unsupported("recurrent arithmetic with competing NaN payloads is unsupported")`

The check applies at the arithmetic operation, including collisions with a NaN
created by invalid arithmetic. Different NaNs in independent rows are allowed.
The boundary is required because original MSVC/GNU operand selection differs
with compiler register allocation and loop position; ordinary Rust also changes
selection between debug and release optimization. This is a documented
unsupported arithmetic domain, not a relaxed equality comparison. The original
competing-payload outputs remain in the native fixtures.

Floating exception flags, non-default process rounding modes, FTZ/DAZ modes,
other ISA builds, other CRT implementations and general worker schedules are
not certified by this slice. The tested environment uses nearest-even rounding
and gradual underflow. Rust execution is one worker; dense native three-worker
controls establish matching results for those specific cases, not a parallel
scheduler implementation.

Cancellation is checked at sequence/token/row boundaries. An error or
cancellation discards the local output and leaves every input unchanged. No
partially updated state is returned.

## Independent fixtures and validation

`tests/fixtures/recurrent/{msvc,gnu}/recurrent-reference.bin` contains 944
original graphs per host, captured before comparison with the new Rust kernels.
Each record preserves input dtype, all shapes and byte strides, complete physical
input bytes, parameters, worker count, output metadata and all original output
and state bytes. Expected values never come from Rust.

| Original operation | Main cases per host |
|---|---:|
| SSM_CONV | 157 |
| SSM_SCAN | 424 |
| RWKV_WKV6 | 121 |
| GATED_LINEAR_ATTN | 108 |
| RWKV_WKV7 | 134 |

The main corpus contains 1,147,172 original output/state bytes per host. Five
competing-NaN graphs explicitly exercise the unsupported boundary. The other
939 graphs require exact equality for 1,145,804 output/state bytes, output
descriptors and immutable input storage. No numerical tolerance is used.

Cases cover odd and aligned state widths through 64, multiple heads and groups,
per-state and per-head A, remapped stored states, physical token/sequence gaps,
multiple destination sequences, all retained state snapshots, source
fourth-slab behavior, raw IEEE values, and original one/three-worker execution.
Separate checked-domain tests cover invalid IDs, shape/group mismatches,
unsupported dtypes, noncontiguous token storage, native undefined/unsafe
dimensions, out-of-view physical reads, insufficient time-first storage,
competing NaNs and cancellation. The complete original source expectations are
retained even for deliberately rejected domains.

Main fixture hashes:

| Profile | Bytes | SHA-256 |
|---|---:|---|
| MSVC | 4,148,299 | `cbb30b51df847adfbbb6978bed5b537c493d9a6f6629f8c572075877ebbc9e89` |
| GNU | 4,148,299 | `4628aced7c2ba465391b89d045c8facc7b0ad2d44d763dfcbf5346ee10de9945` |

The provenance records also identify the original library, source helper,
compiler macro probe and external generator hashes. New probes are captured in
separate files; these main records are never overwritten to obtain a pass.

### Additional raw IEEE controls

A separate immutable `recurrent-ieee-reference.bin` adds 450 original graphs
per host and 132,336 exact output/state bytes. It exercises positive and negative
quiet/signaling NaNs, single-word and whole-input placement, uniform raw values,
maximal finite values, infinities and their state propagation. All 450 results
are compared exactly; none uses a checked rejection or NaN exemption.

The combined recurrent cohort therefore has **1,394 captured graphs per host**:
**1,389 execute with 1,278,140 exactly matching output/state bytes**, while five
retain their original outputs and require the documented competing-NaN error.
There are 1,279,508 original output/state bytes in the full retained cohort.

Both hosts produced the same 758,880-byte IEEE extension:
`3651a4cedc8fe64d137be0c02987633829bd2bb6a4171c400be12d04298a20d9`.
The matching `provenance-ieee.json` describes this extension precisely.
The external initial extension capture and its generator remain retained;
a repeated original capture corrected inherited descriptive provenance text
and was verified byte-identical to that initial capture. No numerical input
or expected output was changed.
