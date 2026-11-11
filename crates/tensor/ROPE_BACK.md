# Backward rotary position encoding

`Op::RopeBack { options, with_factors }` executes original operation ID 49.
Its inputs are the F32/F16 gradient/value tensor, an I32 position vector and,
when requested, an F32 frequency-factor tensor. The output has the input
shape and dtype, contiguous storage and independent ownership. All inputs
remain immutable.

## Modes and parameters

| `RopeBackMode` | Original ID | Paired channels | Position vectors |
| --- | ---: | --- | --- |
| Normal | 0 | Adjacent channels | One |
| NeoX | 2 | First/second halves of the rotated span | One |
| Multi | 8 | First/second halves; contiguous T/H/W/extra sections | Four |
| Vision | 24 | First/second halves of the entire row; independent section frequencies | Four |
| Interleaved | 40 | First/second halves; original interleaved T/H/W/extra selection | Four |

`RopeBackOptions::new(dimensions, mode)` selects the original unscaled defaults:
base 10,000, frequency scale one, extension factor zero, attention factor one,
context/betas/sections/offset zero and truncation enabled. Multi-axis callers
also supply nonzero T/H/W sections.

| Option | Meaning and current boundary |
| --- | --- |
| dimensions | Positive even number of rotated channels; vision uses half the complete row width. |
| mode | One of the five explicit modes above. |
| context_original | Original YaRN context, nonnegative signed I32; positive for active YaRN. |
| frequency_base | Finite and strictly positive. |
| frequency_scale | Finite; zero/negative are supported with extension disabled, positive when active. |
| extension_factor | Finite; nonzero activates source YaRN interpolation/extrapolation. |
| attention_factor | Finite magnitude applied to both cosine and sine, including zero or negative values. |
| beta_fast / beta_slow | Finite and positive for active YaRN; ignored values retain source behavior when extension is zero. |
| sections | Four nonnegative signed-I32 counts; multi-axis modes need a nonzero T/H/W section and total at most the row width. |
| offset | Even starting channel of the rotated span, contained in the row; zero for vision. |
| truncate | Applies source floor/ceil correction boundaries when enabled. |

Every input extent and integer option fits signed I32. The nonempty source row
count also fits signed I32. Row width is even; the source's final pair-copy
loop otherwise reads beyond an odd row. The rotated span fits the row.
Vision requires `dimensions == width/2` and offset zero.

## Physical input contract

The value tensor has shape `[width,heads,tokens,batches]`. Nonempty rows have
the native scalar inner stride, with aligned outer byte strides. Padded,
permuted outer rows and overlapping immutable views retain physical addressing.
Output rows are contiguous.

Positions have shape `[tokens,1,1,1]` for Normal/NeoX and
`[4*tokens,1,1,1]` for the other modes. The original kernel reads a contiguous
I32 prefix from the position tensor's data pointer, including when its logical
axis-zero stride has gaps. The adapter preserves that flat prefix rather than
gathering logical elements. Multi-axis positions are four consecutive token
vectors, shared by attention heads and batches.

Frequency factors have F32 type and axis-zero length at least
`dimensions/2`, matching the constructor. The original cache then reads
`width/2` contiguous F32 words regardless of its logical descriptor. The
adapter checks that this complete prefix fits the declared backing span before
allocation or execution. Extra logical rows can supply that physical prefix;
a short backing span returns Bounds. Factors used by rotated channels must
be finite and nonzero. Nonfinite words affecting only unused cache entries
are retained as supported source controls.

The original dispatcher skips empty output tensors. Valid empty head, token
and batch dimensions return an empty tensor with the original output
descriptor before kernel scalar-layout and numerical checks. The Rust API
still validates its typed constructor geometry and integer parameter domain.

## Arithmetic and exceptional values

The implementation follows the original backward kernel directly. It keeps
the source F32 power, frequency stepping, YaRN correction and interpolation
order, attention magnitude and per-pair scalar products. Backward changes
the sine sign; both sine and cosine retain the source attention magnitude.
Normal/NeoX, contiguous MRoPE sectors, vision frequency resets and interleaved
MRoPE selection each follow their original branch.

F16 input words convert to F32 for arithmetic and round to F16 only when each
rotated output word is stored. F32 output retains its original arithmetic.
Prefix/suffix channels outside the rotation are copied as raw bytes, preserving
all half NaN encodings, signaling payloads, infinities and signed zeros.

Rotated input values, used sine/cosine coefficients and each of the four F32
products in a pair must be finite. Finite final additions can overflow to
infinity, and F16 storage can overflow after rounding; those source outcomes
are preserved. A nonfinite rotated input, coefficient or product returns
Unsupported. Nonfinite or zero used frequency factors, nonfinite active
frequency parameters, nonpositive active YaRN context/scale/betas and
nonpositive frequency bases are also explicit Unsupported domains.

Unused prefix/suffix values, unused frequency-cache words and inactive YaRN
beta fields have separate original controls and remain supported. The
implementation does not reject an entire tensor merely because an unrotated
or unused word is nonfinite.

## Resource and cancellation boundaries

Output payload and the F32 trigonometric cache each have a 256 MiB limit.
A checked estimate of `elements + width*tokens*batches` is bounded by
268,435,456 scalar units before allocation. This prevents overlapping tiny
input views from amplifying into unbounded output work. Every accessed byte
is checked against the declared view span, and allocation is fallible.

Cancellation is checked before token and head work. Failure or cancellation
publishes no partial tensor and changes no shared input storage. The adapter
is serial; the original three-worker controls validate disjoint native row
behavior. General CPU scheduling and graph-wide allocation planning remain
open.

## Independent original backward evidence

The reference generators called `ggml_rope_ext_back` and
`ggml_rope_multi_back`, then executed actual original CPU graphs before the
Rust implementation existed. Each fixture retains original parameters, raw
input bytes, shapes, byte strides, worker count and the complete expected
output descriptor and bytes.

| Corpus per host | Original graphs | Exact replay graphs | Exact output bytes | Explicit native-success exclusions | Retained excluded bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main | 780 | 780 | 1,031,360 | 0 | 0 |
| Empty / IEEE / boundaries | 112 | 69 | 133,380 | 43 | 5,216 |
| Total | 892 | 849 | 1,164,740 | 43 | 5,216 |

The 43 excluded cases succeeded in the original CPU; their original output
bytes remain unchanged and tests require an explicit Unsupported result.
All admitted output bytes, including exceptional copied payloads and stored
infinities, compare exactly. There is no tolerance or blanket NaN exemption.

Coverage spans F32/F16, all five modes, partial rotations and offsets,
one/three workers, padded value rows, strided position/frequency descriptors,
physical prefix reads, source section resets/interleaving, YaRN truncation,
negative/zero unscaled frequencies, attention magnitudes and empty dispatch.
Two original graphs rotate every finite half input word. Another graph copies
all 65,536 half words unchanged outside its rotated span.

The MSVC and GNU main outputs differ, so matching-profile captures remain
separate. All eight binary/provenance files are copied without modification
under `tests/fixtures/rope-back`. Source/library/generator hashes and the
external case ledgers are recorded in provenance and the evidence directory.
Native C/C++ and harness artifacts remain outside the maintained product.

Other ISA/compiler/CRT profiles and floating-point exception or FTZ/DAZ
settings are unverified. This explicit backward kernel does not provide graph
automatic differentiation or extend the separate forward RoPE API.
The frozen GGML and YaRN MIT attributions are retained in
[ROPE_BACK_NOTICES.md](ROPE_BACK_NOTICES.md).
