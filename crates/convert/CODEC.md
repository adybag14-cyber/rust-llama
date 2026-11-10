# Frozen Python converter codecs

This module implements checkpoint dtype preparation and the scalar/packed encoders used by the admitted dense converter. It is a safe Rust implementation of the frozen Python conversion path. It does not call the C runtime quantizers and does not load Python, Torch, NumPy, llama.cpp, or GGML at runtime.

## Exact scope

Preparation accepts the frozen LazyTorchTensor dtype map's F64, F32, BF16, F16, I64, I32, I16, U8, I8, BOOL, F8_E4M3, F8_E5M2, and F8_E8M0. Natural preparation retains F16 as F16 and F32 as F32, preserving their raw no-op bytes. Other accepted inputs become F32. Wider unsigned integers, complex, F4/F6 and FNUZ formats remain explicit errors. Accepting a source FP8 cast does not implement preserved FP8 or ModelOpt output conversion.

The output codecs are F32, F16, BF16, Q8_0, TQ1_0, and TQ2_0. Their block layouts come from the frozen GGUF specification; their rounding, reduction, casts, and packing come from `gguf-py/gguf/quants.py`. Runtime GGML quantizers have a separate API and are not substitutes for these Python encoders.

All byte spans are dense and little endian. Row permutation and QKV concatenation belong to the converter caller.

## API and allocation contract

`working_type` and `prepared_len` validate the source dtype and compute the exact preparation size. `prepare_into[_with_profile]` writes into an exactly sized caller allocation and returns the intermediate WorkingType.

`encoded_len` validates scalar alignment, complete row shape, target support, block divisibility, and checked output size. `encode_into[_with_profile]` writes into the exact output span. Input shape, output length, and any profile-dependent nonfinite restriction are checked before the first output write. All error paths leave the caller's output unchanged.

The codec allocates no heap. Its packed path has one fixed 256-element F32 array and one 256-byte symbol array: 1,280 bytes of logical array storage, independent of tensor dimensions. This is not an OS thread-stack size claim. Callers must separately bound and reserve source, intermediate, permutation, concatenation, and output buffers; the codec length functions do not impose a whole-converter resource limit.

Only `CodecError::RowWidth` corresponds to the original `QuantError` branch that falls back to F16. Unsupported dtype, malformed shape, overflow, short output, and unverified profile errors must not be converted into that fallback.

For empty tensors, direct F32/F16 casts accept a zero last dimension. Block codecs reject it because the original row reshape fails. With a positive valid row width and zero rows, BF16/Q8_0 produce empty bytes, whereas both ternary encoders reject the zero-block inferred reshape. These distinctions are independently captured.

## Preparation order and complete components

The caller must retain the following source order:

1. Prepare each entire source component with `force_f32 = false`.
2. Apply the source's Q/K row permutation to its raw working bytes.
3. For an actual mixed F16/F32 QKV concatenation, promote each complete permuted F16 component with `force_f32 = true`, using the selected profile.
4. Concatenate complete Q, K, and V components in source order, then encode the resulting rows.

Do not promote arbitrary chunks independently. The captured GNU Torch conversion has a complete eight-element prefix followed by a scalar tail for each component. Tail membership can change NaN bytes. Separate `encode_into(F16, ..., F32)` is the later NumPy cast and must not be used for Torch promotion.

The actual Torch-cat corpus includes 1D bias components, 2D permuted Q/K components, mixed dtype combinations, and sizes around 32,768/65,536 elements. Exact equality of these captured domains does not assert compatibility with every future Torch implementation or arbitrary noncontiguous TensorIterator geometry.

## Explicit source profiles

| Profile | Captured source builds | Captured reduction |
| --- | --- | --- |
| WindowsTorch211Numpy246 | Torch 2.11.0+cpu; NumPy 2.4.6; Python 3.14.2; x86_64 MSVC | NumPy maximum dispatch X86_V3, eight F32 lanes |
| LinuxTorch211Numpy226 | Torch 2.11.0+cpu; NumPy 2.2.6; Python 3.10.12; x86_64 GNU | NumPy maximum dispatch AVX512_SKX, sixteen F32 lanes |
| PortableFinite | Format-defined scalar casts and finite packed inputs | No claim for profile-dependent nonfinite promotion or packed reduction |

Both explicit captured profiles are implemented in scalar safe Rust and can be selected on either Rust host. The profile is a source-build policy, not runtime ISA dispatch and not a query of installed Python packages. The converter should record its chosen profile in provenance.

`SourceProfile::current()` is only a convenience association: verified x86_64 Windows/MSVC chooses the captured Windows profile, verified x86_64 Linux/GNU chooses the captured GNU profile, and other targets choose PortableFinite. Selecting a captured profile does not establish parity with a different installed Torch/NumPy version. PortableFinite rejects half-promotion NaNs and packed nonfinite inputs before writing; format-defined direct scalar casts remain available.

Original captures used one Torch computation thread and one interop thread, default round-to-nearest and gradual underflow. The base provenance's nested environment report was collected during environment installation and contains its earlier thread defaults; the capture scripts themselves set both counts to one before executing any case. NumPy runtime output is retained in capture stdout; the separate dispatch JSON files identify the selected maximum kernel. Altered floating environments and unmeasured source package/dispatch combinations are not covered.

## NaNs, signed zero, and source rounding

Raw F16/F32 no-op preparation and no-op output casts preserve all bits. Torch F16-to-F32 promotion quiets signaling NaNs; the GNU scalar tail produces positive 0x7fffffff, while its complete eight-value groups and the captured Windows path preserve the quieted sign and payload. This is a measured compiled-profile rule, supported by every 16-bit half pattern through a one-element original cat and an independent Tensor.to control.

NumPy F16-to-F32 widening preserves the signaling payload. NumPy F32-to-F16 narrowing retains the high payload/signaling bit and sign; a low-only payload becomes one so a NaN cannot become infinity. BF16 uses the frozen encoder's quiet-NaN and round-to-nearest-even rules. Torch's FP8 casts retain their independently measured special bit patterns, including E8M0's signaling NaN.

The captured NumPy absolute-maximum reduction seeds from element zero, consumes complete SIMD vectors starting at element one, then handles the remaining scalar tail. A NaN in the vector prefix canonicalizes; a first tail NaN can retain its payload. Different eight/sixteen-lane profiles therefore have different packed bytes for some nonfinite blocks. Q8's scale division quiets the NaN; ternary scale storage need not. These exact source outputs remain separate fixtures.

Packed encoders preserve original round-away-from-zero, scale arithmetic, int8 invalid-conversion bytes, and ternary packing. Source-supported nonfinite and overflow/subnormal rows are compared byte for byte; finite-only comparisons or relaxed numerical tolerances are not used.

## Independent fixture inventory

Each cohort has immutable Windows and GNU references generated before its Rust comparison.

| Cohort | Scenarios per source profile | Compared output bytes per source profile |
| --- | ---: | ---: |
| Base Torch preparation and frozen encoders | 1,510 (1,376 accepted; 134 original errors) | 4,372,488 |
| NaN reductions and special cast tails | 1,206 | 94,720 |
| Actual post-permutation mixed QKV cat | 158 | 2,427,672 |
| All one-element half patterns | 65,536 | 262,144 |
| Total | 68,410 (68,276 accepted; 134 errors) | 7,157,024 |

Both explicit source profiles are executed by the ordinary Rust tests on each verified Rust host. The extra Tensor.to bytes in the half-scalar fixture are original-to-original diagnostics, not a second implemented converter capability. Additional ordinary tests cover malformed lengths, overflow, unsupported dtypes, exact no-op bits, distinct Torch/NumPy casts, and transactional rejection of a nonfinite value at the end of a multi-block input.

HCOD0001 stores each independent operation, raw input, exact original output or error class, and name. Its record header is 52 bytes (`<IIIIQQQIII`). HCAT0001 stores complete post-permutation components and the original cat output. HSCL0001 stores the 65,536 original cat output words and separate original Tensor.to words. The adjacent provenance files specify the full layouts, hashes, source archive, and package identities.

## Preserved failure history and limitations

The evidence preserves the first diagnostic reader's incorrect 56-byte record stride, the first Rust empty-ternary mismatch, and the subsequent GNU scalar-tail NaN mismatch. Those failures were corrected in the adapter or Rust implementation; original expected bytes were not changed. Initial Clippy failures are retained with their source preimages.

This module closes the admitted dtype/cast/encoder slice, not all model conversion. Tensor naming, metadata, tokenizer conversion, source model families, RoPE extra tensors, preserved FP8/ModelOpt, sharding, and CLI parity have separate implementation and validation records. Source conversion warnings are captured for provenance; this API returns exact bytes or typed errors and does not reproduce Python warning text.

## Source identity and licensing

Frozen archive SHA-256: `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

Primary algorithm sources: `conversion/base.py`, `gguf-py/gguf/quants.py`, NumPy 2.4.6/2.2.6 `loops_minmax.dispatch.c.src` and SIMD math headers, and PyTorch commit `70d99e998b4955e0049d13a98d77ae1b14db1f45` copy/cat and half conversion source. Exact source hashes and retrieval URLs accompany the external handoff. Downloaded C/C++ source is reference evidence only and is not part of the maintained Rust runtime.

The codec portion carries MIT and BSD-3-Clause attribution. The complete crate also includes the metadata/tokenizer and RoPE terms declared in Cargo.toml. Retain THIRD_PARTY_NOTICES.md, METADATA_NOTICES.md and ROPE_NOTICES.md.
