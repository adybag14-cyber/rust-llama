# Extended CPU rotary and softmax attention

This package extends existing frozen ROPE and SOFT_MAX operation IDs. It does
not increase the count of executable IDs, and it does not establish model,
backend, ISA, scheduling or full-port closure.

## Forward rotary operation

`Op::RopeExt { options, with_factors }` takes values, I32 positions and an
optional F32 frequency-factor tensor. `RopeExtOptions` and `RopeExtMode`
are public aliases of the existing backward parameter and mode types. The
original CPU implements forward and backward with one kernel and a sine-sign
parameter; Rust now follows the same sharing. Existing `Op::RopeBack` and
basic `Op::Rope` callers retain their APIs.

The extended forward path supports F32/F16, normal, NeoX, multi-axis, vision and
interleaved modes; YaRN extrapolation/interpolation, attention scaling, beta
parameters, optional truncation, offsets and frequency factors. It preserves
the source's physical position/factor prefixes, aligned strided scalar rows,
and raw prefix/suffix copies. Its independent forward corpus is captured by
invoking original forward constructors and CPU execution. No inverse-derived
or Rust-generated values supply the expected forward outputs.

The shared checked bounds retain the established backward contract:
positive even dimensions, even width and offset, signed-I32 parameters,
contained rotations, source vision/section restrictions and physical byte-span
checks. Empty admitted outputs return before numerical work. Rotated values,
used coefficients and F32 products must be finite. Unrotated channels preserve
all raw payloads, including signaling NaNs; ignored YaRN fields and unused
frequency-factor values retain source behavior. A per-allocation 256 MiB bound
and 256 Mi scalar-work bound precede execution. These are local bounds, not
an aggregate graph or process-memory limit.

## Masked softmax, ALiBi and sinks

`Op::SoftmaxExt { scale, max_bias, with_mask, with_sinks }` takes F32 logits,
then enabled mask and sink inputs. Logits and F32/F16 masks must satisfy the
original contiguous-layout constructor contract. Mask axis 0 matches logit
width, mask axis 1 can contain surplus rows, and mask heads/batches repeat
periodically into the corresponding logits axes.

ALiBi uses the source floor-power-of-two head split and its two powf sequences.
A positive maximum bias requires a mask. Negative and zero bias disable the
slope; a NaN bias also disables it because the original comparison is
`max_bias > 0`. Positive infinity is retained when the resulting scores
remain defined. Scores are scaled, then the per-head mask slope is added,
before the maximum and native-profile exponential/reduction.

Sinks are F32 with first-axis length equal to the number of heads. The original
setter does not require contiguous sinks or restrict their other axes; the CPU
reads a flat physical prefix. Rust checks that prefix span and preserves it,
including surplus axes and noncontiguous sink descriptors. Each sink participates
in the maximum and normalization denominator, without an output probability
column. A finite sink can therefore yield an all-zero ordinary output row when
every ordinary score is negative infinity. This is measured release-library
behavior; the original debug-only pre-sink sum assertion would reject that case.

Empty outputs preserve the source dispatcher behavior after constructor checks.
For nonempty execution, scale is finite, sizes fit signed I32, local output is
limited to 256 MiB and 64 Mi scalar elements, and all resulting probabilities
must be finite. NaN scores, positive-infinite scores/sinks, NaN sinks, an
undefined all-negative-infinity normalization and failed arithmetic return
checked errors. Their successful original release outputs remain recorded in
the fixture rather than being replaced with invented expected values.

The local output vector and final immutable byte buffer may both be resident
during publication. Input/model storage, other graph values and allocator
overhead are outside these local bounds. No aggregate memory budget or
multithreaded softmax execution is claimed.

## Backward bias and ownership

The frozen CPU unconditionally asserts that SOFT_MAX_BACK maximum bias equals
zero. Four isolated original rejection probes on both hosts confirm nonzero
backward bias, positive forward bias without a mask, an undersized mask row
axis, and mismatched sink head count. The existing `Op::SoftmaxBack { scale }`
continues to represent the original zero-bias behavior. There is no invented
backward ALiBi implementation.

Both extended operations return independent immutable outputs. Input bytes are
unchanged on success, checked errors and cancellation. Checks occur before
publication; a later-row softmax error does not expose earlier-row output.

## Independent evidence

The main and edge corpora were captured on each native profile before new Rust implementation. An additional independent broadcast corpus was captured after the initial implementation and before adding its replay test:

| Corpus | Native graphs | Admitted graphs | Exact admitted output bytes | Retained native exclusions |
| --- | ---: | ---: | ---: | ---: |
| Forward rotary main | 780 | 780 | 1,031,360 | 0 |
| Forward rotary edges | 112 | 69 | 133,380 | 43 |
| Extended softmax main, including zero-bias backward | 2,122 | 2,122 | 7,279,720 | 0 |
| Extended softmax edges | 267 | 252 | 371,160 | 15 |
| Intermediate mask head/batch divisors | 80 | 80 | 2,365,920 | 0 |

The rotary corpus covers every finite half input word, raw copied half payloads,
five modes, extended parameters, original one/three-worker graphs, strided inputs,
and the checked finite-domain boundaries. The softmax corpus covers varying row
widths and head counts, F32/F16 masks, all broadcast dimensions, raw sink-prefix
semantics, source one/three-worker graphs, empty dimensions, sink dominance,
subnormal/large values and exceptional-value boundaries. The 104 original
zero-bias backward cases also replay through the existing backward operation.

Each profile compares exact descriptors and bytes against its corresponding
original capture. GNU SSE2 row exponentials/reductions and MSVC scalar rows
remain distinct; no cross-ISA or cross-compiler bit-parity claim is made.
Additional Rust checks cover constructor failures, invalid arity, physical spans,
dtype/layout rejection, output amplification and cancellation without mutation.

Fixtures are under `tests/fixtures/rope-forward` and
`tests/fixtures/softmax`, with raw case ledgers and source/library hashes.
Oracle scripts and C/C++ libraries remain outside the product under
`rust-llama-reference/continue-20261111-next/attention-oracle`.
GGML MIT and the existing YaRN MIT attribution apply. Validation and retained
failures are recorded in `evidence/2026-11-11/attention-extended`.

The supplemental corpus distinguishes periodic mask indexing from grouped head
selection for intermediate divisors. Its first capture inherited an inaccurate
pre-implementation phrase from the original harness. The corrected v2 capture
records the actual timing and reproduces every binary byte; both external
captures remain retained. It never reads or executes Rust to obtain expected data.
