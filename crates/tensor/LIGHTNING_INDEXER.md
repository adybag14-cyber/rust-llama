# Lightning Indexer

`Op::LightningIndexer` preserves original operation ID 87 and takes four inputs
in source order: queries, keys, per-head weights and an F16 numerical mask.
Every output owns independent storage; no input or view is modified.

## Geometry and physical rows

| Tensor | Type | Shape |
| --- | --- | --- |
| Queries Q | F32 | `[E,H,T,B]` |
| Keys K | One of the 29 types below | `[E,1,N,B]` |
| Weights W | F32 | `[H,T,1,B]` |
| Mask M | F16 | `[N,T,1,MB]` |
| Output | F32 | `[N,T,1,B]` |

All extents fit signed I32. Mask streams MB must be positive and divide B.
Stream b reads mask stream `b % MB`, preserving the original modulo
selection. Q, K and W have the same stream count; key rows and weights do not
implicitly broadcast across streams. Nonempty outputs require each input's
axis-zero byte stride to equal its native storage type size, even when that
axis has only one element. Q and W strides are F32-aligned; mask strides are
F16-aligned. Outer byte strides, overlapping immutable views and row padding
remain explicit. Every physical read stays within the input's declared span.

The original CPU dispatcher skips empty outputs before kernel assertions.
This adapter likewise returns empty F32 output when N, T or B is zero,
after constructor dtype/shape checks but before row-layout and key-decoder
checks. Original captures include each empty-output axis and all seven key
types without a nonempty decoder. Zero heads or zero embedding with a nonempty
output preserve the original zero score followed by mask addition.

## Key decoding and source arithmetic

| Family | Supported key storage |
| --- | --- |
| Scalar | F32, F16, BF16, F8_E4M3 |
| Classic block | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q1_0, Q2_0 |
| K block | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K |
| Importance block | IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M |
| Other block | TQ1_0, TQ2_0, MXFP4, NVFP4 |

Quantized rows contain complete blocks. Nonempty execution rejects Q8_1,
Q8_K, I8, I16, I32, I64 and F64 keys because the original CPU has no matching
`to_float` trait. A generic Rust codec is not treated as evidence that the
original operator supports that storage type.

Each key row is decoded to F32 using the existing native-compatible Rust
quantization implementation. The scalar dot multiplies Q and K in F32,
accumulates each rounded product in F64, and rounds the final dot to F32.
For each head, the source `MAX(dot, 0.0f)` comparison returns the dot only
when it is strictly positive. A negative dot, either zero sign and a NaN dot
therefore contribute positive zero before multiplication by the weight.
Weighted contributions accumulate in F32 in head order.

The F16 mask is a numerical additive bias. It can contain positive or negative
infinity, signed zero, subnormals and every half NaN encoding. With a finite
pre-mask score, the original mask NaN is quieted while its sign and payload
are retained. The explicit Rust policy preserves those independently measured
words.

## Checked exceptional and resource boundaries

Actually read weights and the final pre-mask score must be finite. An
unsupported result is returned when a nonfinite weight or score is encountered,
including finite inputs that overflow during the weighted reduction.
Queries and keys may contain nonfinite words when the original ReLU suppresses
their resulting NaN or negative-infinite dot and the final score stays finite.
Nonfinite mask values are supported. With no heads, unused weights are not read.

Output payload and decoded-key scratch are each limited to 256 MiB. Before
allocation or numerical work, a checked work estimate must not exceed
268,435,456 scalar units. The estimate is
`N*T*B*(E*(H+1)+H)`: one key decode, H dot products and H weighted reductions
per output. The limit prevents overlapping small input views from creating
unbounded scalar execution even when the output itself is tiny. Checked
overflow, invalid shapes/strides and out-of-view addresses return errors.
Allocation is fallible. Cancellation is checked before token and key work;
partial results are never published.

These are local adapter limits, not a general DAG allocation plan or aggregate
graph-memory budget. The adapter is serial. Native one/three-worker controls
cover disjoint original output work, not a new Rust parallel scheduler.

## Independent native evidence

Expected values were captured from the frozen original CPU on both x86_64
MSVC and GNU/Linux before this Rust implementation. Each configuration runs
two original graphs: its supplied mask, and an all-zero mask yielding the
native pre-mask score. Admission classification comes from that stored second
original output, never a Rust-generated answer.

| Corpus | Configurations | Original graphs | Exact replay graphs | Exact output bytes | Explicitly rejected graphs | Retained excluded bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 652 | 1,304 | 1,304 | 2,802,864 | 0 | 0 |
| Empty/IEEE edges | 51 | 102 | 78 | 2,784 | 24 | 1,152 |
| Total per host | 703 | 1,406 | 1,382 | 2,805,648 | 24 | 1,152 |

The 24 exclusions are successful original graphs with nonfinite pre-mask
scores. Their complete native outputs remain unchanged in the fixture;
tests require an explicit Rust Unsupported result. No tolerance or blanket
NaN exemption is used for the admitted graphs.

Coverage includes all 29 key decoders, original quantizer-generated key rows,
one/two/three quantization blocks, padded outer strides, multiple heads/tokens/
keys/streams, modulo mask sharing, one/three original workers, all 65,536
F16 mask words at three different scores, all F16/BF16 key words except positive
infinity, all 256 F8 key words, and explicit zero shapes and IEEE controls.
The supplied and zero-mask results both compare byte for byte. Output
descriptors, independent ownership and input immutability are checked too.

The separately captured MSVC and GNU binary corpora were byte-identical.
One copy of each raw binary is stored with both unchanged profile provenance
files in `tests/fixtures/lightning-indexer`; no expected values were merged,
rounded or regenerated. Original source/library/generator hashes and native
case ledgers remain auditable through the provenance and evidence records.
The standalone native harnesses and libraries remain outside the product.

Other ISA/compiler profiles, alternative floating-point exception or FTZ/DAZ
settings, arbitrary graph scheduling and complete model parity remain open.
The implementation and original-derived fixtures retain the frozen GGML MIT
attribution in root `NOTICE` and `LICENSE-UPSTREAM`.
