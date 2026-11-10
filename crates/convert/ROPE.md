# Llama 3 RoPE factor generation

This module ports the numeric branch of the frozen LlamaModel.generate_extra_tensors method for rope_type = llama3. It writes a complete little-endian F32 tensor without heap allocation. Its implementation is Rust. Python, Torch, NumPy, original llama.cpp code and downloaded SLEEF C sources are comparison inputs outside the product.

The scalar power tail uses Rust f32::powf, which lowers to the target OS math runtime on the two measured targets. No original project binary or SLEEF shared library is called by the implementation. The vector-prefix calculation is a scalar Rust transcription of the exact SLEEF algorithm selected by the captured Torch builds.

## Public interface and caller responsibilities

~~~rust
use llama_convert::{codec::SourceProfile, rope};

let parameters = rope::Parameters::new(head_dimension);
let output_bytes = rope::encoded_len_with_profile(&parameters, selected_profile)?;
// The caller checks its aggregate budget and fallibly allocates output_bytes.
rope::generate_into_with_profile(&parameters, selected_profile, &mut output)?;
~~~

Parameters contains a signed 64-bit dimension and five binary64 values: theta, factor, low_frequency_factor, high_frequency_factor and original_context_length. Parameters::new uses the frozen numeric defaults 10000, 8, 1, 4 and 8192.

The converter caller selects the full_attention mapping when present, recognizes the llama3 rope type, and resolves the dimension from head_dim when that field is not None. Otherwise the frozen method uses integer hidden_size / num_attention_heads. A zero head_dim is an explicitly supplied dimension, rather than a request for that fallback. Metadata lookup, tensor naming, target-format choice and output-file writing are outside this numeric helper.

Generate one complete component. The complete output count determines the vector and scalar tails. Generating independent pieces with different dimensions does not preserve that membership. The API provides no chunk-offset argument, and a shorter output slice is an error.

encoded_len validates parameters; encoded_len_with_profile also validates the requested policy before caller allocation. Both check the element/byte count against usize and the signed slice-length domain. It allocates nothing. generate_into and generate_into_with_profile detect every error before writing output. The caller supplies the exact byte length and owns both output allocation and the aggregate converter working-set limit. Execution is linear in the number of output elements, with constant local scratch and no threads or hidden heap buffers. The helper does not impose a total process-memory limit or cancellation mechanism.

## Source parameter and dtype rules

The generated tensor is F32 from the first arange operation. Python configuration values remain binary64 until the specific Torch weak-scalar conversion in the source expression.

| Step | Preserved source operation |
| --- | --- |
| Shape | CPU F32 arange length is computed using the source binary64 accumulator and checked conversion, including the source rounding above the exact-integer F64 range. Negative dimensions are rejected. |
| Index values | arange uses the captured AVX2 dispatch: two eight-value vectors per prefix block, with each vector base narrowed to F32 before its F64 lane offset; the remainder computes F64 index times step before narrowing. |
| Exponent | Each F32 arange value is divided by the dimension narrowed to F32. |
| Power prefix | Two sixteen-value vectors per complete 32-element block use the pinned SLEEF AVX512 FMA power arithmetic. Rust explicitly preserves each fused multiply-add and double-float correction. |
| Power tail | Remaining values use the matching target's scalar float power runtime. |
| Frequency | F32 reciprocal of the power. |
| Wavelength | F32 reciprocal of frequency, then multiplication by the F32 conversion of Python's binary64 2*pi. |
| Thresholds | original_context_length / low_frequency_factor and / high_frequency_factor are computed in binary64 before their F32 Tensor comparisons. |
| Blend denominator | high_frequency_factor - low_frequency_factor is computed in binary64, then narrowed for F32 division. |
| Smooth blend | Each reciprocal, multiplication, subtraction, division and addition preserves the original F32 order; no algebraic reassociation is used. |

Zero dimensions yield empty tensors. Odd dimensions retain the final even arange index, as the original does. Negative dimensions and zero low/high frequency factors are errors; the latter remain errors for an empty tensor because Python computes both threshold divisions before entering the loop.

Other finite signs and special intermediate results are preserved. A zero factor, negative base, negative context, equal or reversed nonzero frequency factors, and finite binary64 parameters that narrow to F32 zero or infinity are not silently normalized. They can produce signed zeros, infinities or NaNs, whose original output bits remain covered.

Nonfinite binary64 configuration parameters are an explicit unsupported input domain of this bounded helper. They are rejected before output mutation. This restriction does not reject a finite configuration value merely because an intermediate F32 conversion overflows.

## Explicit source profiles

The public SourceProfile comes from the converter codec module. Its name identifies captured source builds, not installed Python packages or the Rust process's detected vector width.

| SourceProfile | Admitted target for this helper | Captured Torch | Captured NumPy |
| --- | --- | --- | --- |
| WindowsTorch211Numpy246 | x86_64-pc-windows-msvc | 2.11.0+cpu | 2.4.6 |
| LinuxTorch211Numpy226 | x86_64-unknown-linux-gnu | 2.11.0+cpu | 2.2.6 |
| PortableFinite | Explicit UnsupportedProfile error | Unverified | Unverified |

A foreign source profile is rejected even when a particular parameter set happens to produce the same bytes on both hosts. This preserves a clear contract for the scalar tail. The format codecs can reproduce both profiles on either measured Rust host; this RoPE helper has a narrower platform contract.

Both captured Torch builds select AVX512 for power. Range generation uses REGISTER_DISPATCH, whose AVX512 entry is null, so the original dispatch falls back to AVX2. The implementation preserves those different source choices without requiring AVX512 instructions in the Rust implementation.

SLEEF's domain-NaN constant is also source-build dependent. The pinned GNU path uses __builtin_nanf and produces 0x7fc00000 for a finite negative base raised to a noninteger vector exponent. The pinned MSVC path defines NAN through infinity minus infinity and produces 0xffc00000. SourceProfile selects this constant explicitly. General input-NaN mask behavior in the private SLEEF helper remains distinct from this domain-NaN constant.

The recorded runtime-identity fixtures identify the actual loaded scalar powf provider and original Torch CPU binary with full SHA-256 and file size. Profile selection does not inspect an installed Torch library. Exact fixture comparisons on another provider version are new compatibility evidence; the target triple alone is not a claim of arbitrary runtime-version equivalence.

The numerical proof assumes round-to-nearest ties-even and gradual underflow, with no caller changes to rounding mode, FTZ or DAZ.

## Frozen source identities

The archive is the same frozen user-provided llama.cpp snapshot used by the rest of this port, SHA-256 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.

The original factor generator is conversion/llama.py, 21,299 bytes, SHA-256 33d3068838f923bdb3f3aea8113ad43bdbe932a25e4554f35fb80ee278c1dde2.

Both captured Torch distributions identify commit 70d99e998b4955e0049d13a98d77ae1b14db1f45. Its SLEEF submodule points to commit 5a1d179df9cf652951b59010a2d2075372d67f68; the installed header reports SLEEF 3.8.0.

The external source manifest pins these primary files and installed headers:

- Torch Pow.cpp and PowKernel.cpp establish the scalar-base redispatch and tensor/tensor vector/scalar kernels.
- Torch BinaryOps.cpp and BinaryOpsKernel.cpp establish weak-scalar coercion, reciprocal and F32 arithmetic.
- Torch RangeFactories.cpp, RangeFactoriesKernel.cpp, AccumulateType.h, DispatchStub.h, Loops.h and the float vector headers establish length, F64 indexing and the two distinct vector/tail boundaries.
- SLEEF sleefsimdsp.c supplies xpowf, logkf and expkf.
- SLEEF df.h supplies the double-float corrections; helperavx512f.h supplies fused multiply-add and mantissa/exponent behavior; misc.h supplies source-profile NaN definitions.

The Rust transcription retains the original coefficient literals and rounding points. Approximate-constant/precision lint allowances are limited to that source-derived private math module. Attribution and the complete Boost Software License are in ROPE_NOTICES.md; the crate license expression must include BSL-1.0.

## Independent comparisons and retained failures

The authoritative final output in the complete-generator fixtures comes from calling the unmodified original LlamaModel.generate_extra_tensors method. Seven intermediate F32 stages are also captured by a separate Torch transcription of that source expression, whose final bytes must match the authoritative method before a case can be admitted. Off-branch smooth values are diagnostic NaN sentinels and are excluded from arithmetic comparison.

The first corpus has 579 cases per host: 576 original successes and three original errors, with 90,880 authoritative output bytes. The additional corpus has 1,336 cases per host: 1,308 original successes and 28 original errors, with 446,584 authoritative output bytes. Together this is 1,915 original method calls, 1,884 successful tensors, 31 original rejections and 537,464 compared final output bytes per host. Structured and seeded random cases cover vector/tail boundaries, threshold neighbors, odd and empty shapes, positive/negative/zero parameters, subnormal bases and extreme binary64-to-F32 coercions.

A separate corpus of 852 original Torch operator windows per host, with 72,270 F32 stage words, tests large global arange indices without allocating a multi-petabyte tensor. Each window starts at a global multiple of 32 and preserves either a complete original prefix block or the complete final tail. It exposes 42 direct-integer-cast counterexamples per host and tests arange, exponent and power as operator diagnostics; it is not counted as an additional complete original LlamaModel generator invocation.

All numerical checks compare raw F32 words, including zero signs and NaN signs/payloads. No tolerance is used.

The preserved development history includes:

1. The first capture wrote its binary reference before a NumPy integer diagnostic count failed JSON serialization. A new capture revision fixed only diagnostic serialization; the initial artifact and failed log remain available.
2. Direct OS powf and binary64 math.pow narrowed to F32 are retained as diagnostic counterexamples to the source vector-prefix algorithm.
3. Cross-host prototype checks differ in scalar-tail intermediate words. They are retained as negative controls for explicit profile rejection.
4. The first extended Windows comparison differed in 1,136 final NaN words from negative vector bases; the same corpus was exact on GNU. Inspection of the pinned SLEEF misc.h established the source-defined NaN-sign correction. Those failures and both untouched expected corpora remain preserved.
5. The first checked-helper gate passed all 15 numerical/atomicity and existing codec tests on Windows, then strict Clippy required the pinned compiler's array-chunk iterator spelling. The preimage and failure log are preserved; the iteration changed without changing arithmetic or error behavior.

The handoff validation report records exact final source hashes, fixture identities, host/compiler/provider identities, and debug/release/fmt/strict-Clippy results. It must be read together with the stated profile and finite-configuration restrictions.
