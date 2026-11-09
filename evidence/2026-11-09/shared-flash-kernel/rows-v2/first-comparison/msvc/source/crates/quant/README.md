# GGML codecs and scalar matrix-vector kernels in Rust

This crate implements checked, little-endian codecs for the **36 active dtype IDs** in the frozen source ZIP (`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`). It is a quantization foundation, not a claim of complete GGML CPU/backend parity.

## Implemented

- Exact numeric IDs, block lengths and byte lengths; removed IDs remain rejected.
- Decoding of F32, F16, BF16, F64, I8/I16/I32/I64 and F8_E4M3.
- Decoding of Q1_0, Q2_0, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1 and Q2_K through Q8_K.
- Decoding of every active IQ format: IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S and IQ4_NL/IQ4_XS.
- Decoding of TQ1_0/TQ2_0, MXFP4 and NVFP4.
- Scalar encoding of every source model quantizer: F32/F16/BF16/F8_E4M3; Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0; Q2_K/Q3_K/Q4_K/Q5_K/Q6_K; all nine IQ types; TQ1_0/TQ2_0; MXFP4/NVFP4. The 29 source model formats include three requiring explicit importance weights.
- Additional Q8_1/Q8_K activation codecs and an F32-to-F64 serialization helper. Together there are 32 non-integer codec formats: 29 have an unweighted entry point and three require weights.
- Exact optional importance-weighted legacy, K and IQ fitting, including source lattice initialization, neighbour search, sign parity, row variance, scale fitting and bit packing.
- Checked F32-to-I8/I16/I32 tensor assignments using the source's truncation toward zero. These are exposed separately from quantization: the snapshot has no integer row quantizer and no F32-to-I64 setter.
- Checked native matrix-vector arithmetic for all 29 weight dtypes with a CPU vec_dot in the frozen snapshot: 25 packed types and scalar F32/F16/BF16/F8_E4M3.
- Bit-based IEEE conversion with ties-to-even rounding for F16/BF16, preserving signed zero and explicit NaN behavior.

All input lengths and complete quantization blocks are checked before decoding or writing a destination. No alignment assumption, native-structure pointer cast or unsafe code is used. `dequantize` has fallible allocation. An existing destination remains unchanged on validation or encoder preflight failure. Block encoding rejects non-finite inputs and unrepresentable scale computations. The legacy block formats require finite F16 scales, while Q8_K requires a finite inverse and F32 scale; scalar IEEE conversion permits non-finite values. Unsupported encoders produce a typed error.

Q8_1 and Q8_K are intermediate formats in GGML; offering a codec does not establish that the original model loader accepts these as stored model weights. F64 serialization is a separately validated C-cast conversion helper; the snapshot has no F64 `from_float` or model `quantize_chunk` entry. Integer/F64 conversions to F32 have normal Rust/C rounding and range loss. NVFP4 decoding deliberately retains the frozen source's UE4M3 scale behavior (half-scale, ignored sign bit, 0x7f mapped to zero); the format is not treated as the standalone signed F8_E4M3 dtype.

## Importance-weighted encoding

`quantize_with_weights(dtype, values, importance)` encodes one complete row. The importance slice must have exactly the row length and contain finite, nonnegative values. It is never synthesized: `quantize` returns `ImportanceWeightsRequired` for IQ2_XXS, IQ2_XS and IQ1_S. The corresponding `_into` APIs stage their output, so an invalid later block cannot partly overwrite the destination.

For the legacy weighted formats, variance is calculated across the entire supplied row, as in the source. IQ and K encoders preserve their own block/subblock weighting and search algorithms. Formats whose original model quantizer ignores weights retain that behavior after validation. Q8_1, Q8_K, F64 and integer types have no original model quantizer accepting importance weights and return `UnsupportedWeightedQuantization`.

IQ1 prefix fitting depends on the order of tied input values. An independent finite-value median-of-three index sorter reproduces the frozen native UCRT comparator order, including its maximum-selection short-range finish. It is checked with 640 native calls; no platform sorting runtime is linked. C does not specify equal-value `qsort` ordering, so another original C platform may produce different encoded bytes for ties. The Rust result intentionally preserves the frozen reference build.

Two undefined original-source paths have explicit safe behavior:

- The unrefined IQ4_NL reference encoder does not initialize its local payload levels for a block whose magnitude is below `1e-15`. Rust writes a canonical zero scale and zero payload. These source-undefined bytes are excluded from byte parity claims; the defined weighted zero paths remain fixture checked.
- IQ1's exceptional exhaustive-neighbour fallback indexes an eight-element input span with a codebook index. If numerical scores cannot select a valid finite neighbour, Rust returns `UnrepresentableScale` atomically instead of performing that out-of-bounds access.

## Matrix-vector arithmetic

`matvec_activation_dtype` is the capability gate; `matvec` validates the complete matrix and vector, converts the activation once, and evaluates every row. `dot_packed` accepts already converted activation blocks for all 25 implemented packed weight formats. The seven active dtypes without a source CPU weight dot (Q8_1/Q8_K, I8/I16/I32/I64, F64) return `UnsupportedMatVec` before evaluation, even when a dequantizer exists.

| Weight dtype | Activation dtype in this snapshot |
| --- | --- |
| Q1_0, Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, MXFP4, NVFP4 | Q8_0 |
| Q4_1, Q5_1 | Q8_1 |
| Q2_K through Q6_K, remaining IQ formats, TQ1_0/TQ2_0 | Q8_K |
| F32, F16, F8_E4M3 | F32 |
| BF16 | BF16 |

For the packed block formats, integer products are accumulated first, then the original scale/minimum arithmetic is applied with the source's multiplication order. A decoded-weight F32 dot would omit activation quantization and produces different logits. Q8_0/Q8_1 activation conversion here follows the original scalar reference path, including `roundf` tie behavior. ISA-specific vectorized conversion and reduction paths can differ in rounding; matching those paths remains separate work. F16 uses F32 activations in this frozen snapshot; rounding them to F16 would change its behavior.

Q3_K/Q4_K/Q5_K/Q6_K preserve the original eight floating accumulation lanes across all row blocks and their final reduction order. IQ2/IQ3 kernels retain the original post-accumulation scale factors; IQ1 kernels use integer offset sums. No packed path substitutes decoded F32 multiplication. Q8_K activation encoding preserves the source's first signed maximum and ties-to-even rounding, stores its F32 scale and sixteen I16 block sums, and rejects an unrepresentable inverse scale before output mutation. Its all-zero blocks have canonical zero sums; the original C function leaves those sum bytes untouched.

## Independent validation

`cargo test -p llama-quant --offline` checks the committed fixtures against a separately compiled C reference. The C code and build remain outside the Rust product. `tests/fixtures/provenance.json` records the source hashes, compiler, fixture hashes and qualifications.

- All 36 decoder layouts are checked bit-for-bit (NaNs compared by classification). Most have 19 arbitrary packed blocks, exercising codebooks and bit packing independently of the Rust encoders.
- Every one of the 65,536 F16 encodings, 65,536 BF16 encodings and 256 F8_E4M3 encodings is checked against the C output.
- Twelve encoders are checked against reference bytes over ten structured/random rows each; F16/BF16 also have 20,021 raw F32 bit patterns apiece.
- Another 162 independently generated matrix-vector cases cover nine weight dtypes, six widths (32 through 1,536), three activation patterns and three output rows per case. All five packed formats match unmodified original scalar C dot functions bit-for-bit. All nine types also match separately executed original GGML CPU graphs within `2e-5 * (1 + abs(expected))`, allowing scalar/SSE reduction-order differences. `tests/fixtures/matvec-provenance.json` records these two oracles and their source/binary hashes.
- A further 480 matrix-vector cases exercise every remaining packed dtype with 1/2/3/8 blocks and arbitrary packed codebooks, signed/zero/subnormal/max F16 scales, scale edge encodings, and random/zero/tie/sparse/constant/wide-range activations. All 1,440 outputs match both unmodified generic C and actual original GGML CPU graphs bit-for-bit on the frozen reference build. Thirty additional Q8_K encoder cases match original C bytes, including F32 maximum and small representable scales. Provenance is recorded in `tests/fixtures/matvec-extended-provenance.json`.
- The ten additional encoder formats match exact C bytes in 525 cases covering zero/signed zero, constants, ramps, ternary ties, random/sparse rows, half-scale extremes, tiny scales, all normal FP4 exponent boundaries, and 103,214 raw F32 values/rounding neighborhoods for F8. Three integer conversion records cover 12,329 finite inputs and exact truncation/range boundaries. `tests/fixtures/encoder-extended-provenance.json` records the independent oracle.
- F16 matrix multiplication is additionally checked bit-for-bit against 68 original GGML CPU graphs and 6,168 outputs, including 384/1,536 input widths, 384 output rows, seven activation columns, and cancellation around 32/64-element boundaries. The frozen MSVC build uses the pure scalar F16 fallback (F32 products accumulated in F64); substituting a hypothetical SSE chunk reduction disagrees on 4,018 of these outputs. `tests/fixtures/f16-matmul-provenance.json` records this evidence.
- The weighted/IQ encoder extension matches 3,362 independently generated original-C rows exactly: 1,820 legacy/K/IQ4 cases, 390 IQ3 cases, 624 IQ2 cases and 528 IQ1 cases. These cover one to seven blocks, random/constant/repeated/sparse/signed-zero inputs, skipped zero groups, subnormal weights, wide scales and optional/unit/varying/random/sparse/zero importance. The three required-importance formats are only called with explicit weights. The four `*-encoder-provenance.json` files record source, generator and oracle binary hashes; the IQ1 provenance also records 640 native sort cases.
- Directed tests cover half rounding ties/subnormal boundaries, reserved IDs, truncated/oversized input, incomplete blocks, checked overflow, multi-block separation, non-finite quantizer input and unchanged destinations on errors.

The reference consists of unmodified functions extracted from the frozen `ggml-quants.c`, including its original numeric codebooks and scalar conversion helpers. Q8_1 has no upstream dequantization entry point, so its expected output is produced by an independent C loop over the original struct's scale and quants. Integer/F32/F64 reference conversion uses C casts or copying. These exceptions are recorded in the provenance.

## Shared safe CPU row arithmetic

The dependency-free `cpu_math` module is shared by tensor execution and the
model-runtime compatibility facade. It implements frozen scalar/SSE2 row
SiLU, quick GELU, softmax exponentials/sums and centered variance using safe
Rust and explicit F32/F64 rounding/grouping. Four-element vector prefixes
restart at each source row; scalar tails use the current host's standard
math lowering. Explicit MathPolicy selects grouping and does not emulate a
foreign operating-system CRT.

Existing model-runtime native-row fixtures remain the regression gate for
the moved code. The tensor row-policy corpus adds 1,007 matching native graph
cases per host, including F16/BF16 controls and odd packed GLU rows. A separate
original-inline scalar quick-GELU oracle checks 502,058 raw F32 words on each
host; portable SSE2 quick prefixes are also checked on both hosts against
the GNU native graph bytes. This module adds no C dependency to this crate.

## Remaining T03 work

Scalar codec and original weighted-encoder coverage is complete for the defined source entry points, with the explicit undefined-source qualifications above. Optimized CPU kernels/dispatch, ISA-specific numerical/performance checks and general GGML operation coverage remain open. Completing the scalar codec and CPU dot inventories does not close those gates. This crate does not claim throughput, pretrained-model parity or complete T03 coverage.
