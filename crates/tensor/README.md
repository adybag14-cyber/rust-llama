# Rust tensor layouts and scalar CPU graphs

This crate is an executable, partial T02 port of the frozen GGML tensor and CPU graph contracts. It currently exposes **75 operation IDs**, plus input leaves, from the frozen 102-ID registry. The full per-ID inventory and dtype restrictions are in [CPU_SUPPORT.md](CPU_SUPPORT.md). Registration of all IDs is an ABI inventory, not a claim that all operators, parameter modes or schedulers execute.

## Storage and graph contract

`TensorDesc` preserves GGML's four logical dimensions (`ne`) and byte strides (`nb`), with dimension 0 fastest. For quantized dtypes, `nb[0]` measures whole stored blocks while `ne[0]` counts scalar values. Constructors check complete block rows, address/size arithmetic, the full strided span and buffer bounds. Owned buffers must match that span exactly. Empty shapes are permitted for storage and copying; operations needing nonempty rows reject them.

`Tensor` holds immutable `Arc<[u8]>` storage. Views, permutations, transposes and contiguous reshapes can share storage safely. Numeric outputs allocate separate buffers. Public APIs never expose mutable aliases. Reshaping requires contiguous storage and the same element count; a quantized block axis cannot move out of dimension 0. View origins must be multiples of the dtype block's byte size, so some otherwise expressible padded quantized subviews remain unsupported.

`to_bytes` extracts logical raw blocks without numeric conversion and preserves NaN payloads and integer bits. Empty tensors return immediately, including extreme outer dimensions. `contiguous` copies logical blocks and removes stride holes without converting dtype. `to_f32` uses the independently validated `llama-quant` decoder. I32 and I64 constructors/readers preserve integer bits directly, including I64 values above F32/F64's exact integer range.

A `Graph` has graph-specific node IDs. Operations only reference earlier nodes of the same graph, making cycles unrepresentable. Evaluation discovers target dependencies iteratively and ignores unrelated branches. Cancellation is checked during graph traversal, between operations and within scalar loops. Failed or cancelled evaluations leave inputs unchanged. `Arange` is a real zero-input graph node; `View` is an explicit checked storage view.

## Executable CPU slice

The additions include CUMSUM, ARGMAX, COUNT_EQUAL, REPEAT_BACK, GROUP_NORM, L2_NORM, OUT_PROD, DIAG, signed/circular PAD, PAD_REFLECT_1D, ROLL, ARANGE, TIMESTEP_EMBEDDING, LEAKY_RELU, TRI, FILL, and explicit SILU/RMS_NORM/SOFT_MAX/GET_ROWS backward kernels. They build on periodic F32 arithmetic, reductions, concatenation, views, basic RoPE, normalization, softmax and row gathering.

All 22 unary formulas execute for F32/F16/BF16. XIELU's coefficients and epsilon are required explicitly; the bare XIELU enum does not invent defaults. All seven GLU formulas execute for their source-supported dtypes. Plain REGLU/GEGLU/SWIGLU/GEGLU_ERF/GEGLU_QUICK support F32/F16/BF16; OAI and clamped SWIGLU support F32/F16, as the frozen CPU does. Both split and packed rows are available, including gate swaps and the native floor-half behavior for odd packed row widths.

SUM supports F32/F16/BF16. F32 contiguous rows first form F64 row totals and then add those totals; its strided first-axis path scans individual values in F64. F16/BF16 form F32 row totals, accumulate them in F32, then convert the final scalar back to the input dtype. NORM similarly preserves the source's F64 contiguous-row reductions and F32 reductions when dimension 0 is strided. These distinctions have independent cancellation-sensitive fixtures: replacing either path with a generic decoded reduction changes native results.

COUNT_EQUAL retains the frozen CPU's unusual higher-axis address formula. Rust checks every resulting offset against the declared input span; source formulas that would read outside it return Bounds. Circular left padding greater than its input axis is also rejected because the frozen C remainder expression can address before the buffer. These errors do not fabricate a result for source-undefined memory access.

### Matrix arithmetic

MUL_MAT supports all **29 source CPU weight-dot formats** through the quant crate's native activation mapping. F32 right-hand rows are converted to the actual Q8_0/Q8_1/Q8_K/BF16 activation format where required. Already packed activation rows retain their original scales and payloads; they are not decoded and requantized. Integer/block-scale dot order is preserved. The scalar F32/F16/BF16 paths accumulate F32 products in F64; F8 follows its F32 accumulation kernel. Source-supported F16/BF16 RHS widening and grouped batch division are covered.

OUT_PROD supports the **28** frozen left formats with actual CPU kernels; BF16 remains explicitly unsupported. This operator follows source decoding followed by F32 multiply/add accumulation. Non-F32 source batches must match; F32 uses the native grouped expansion. No packed activation arithmetic is substituted into OUT_PROD.

These paths are compared with actual GGML graphs for every supported weight dtype and every supported packed RHS family. A separate noncanonical activation-scale regression prevents accidental decode/requantize substitution.

### Indexed selection, conversion and updates

ADD_ID adds the selected expert bias to each F32 row; ID -1 copies that row unchanged. It preserves the source's first-slab index/bias addressing across axis-3 batches.

MUL_MAT_ID supports all **29** source weight-dot formats. Expert ID -1 produces positive zero; other IDs are checked before execution. Activation rows are packed once in their native format, and slot broadcasting uses `slot % input_slots`. Already packed activation rows are retained. Unlike ordinary MUL_MAT, the source MUL_MAT_ID dispatcher only accepts F32 or the exact activation dtype as its RHS; other half/bfloat widening is rejected.

SET_ROWS accepts F32/F16 values with I32/I64 indices and all **25** destinations having a native CPU `from_float` converter. It retains destination byte strides and untouched bytes, and duplicate indices follow the source's one-thread final-write order. F16-to-F16 writes copy payload bits directly. The output owns independent storage, so writes cannot mutate aliased inputs.

The 25 CPU conversion destinations are F32, F16, BF16, F8_E4M3, I32, Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4 and NVFP4. Stored-model encoder availability does not imply CPU conversion availability.

`Cast { dtype }` executes the CPY same-shape conversion slice: same-type raw copying; F32/F16/BF16 to those 25 destinations; F8 to F32/F16/BF16; I32 to F32; and native quantized input to F32 where the source has a decoder trait. Q8_1/Q8_K have no native CPU `to_float` trait, so conversion from them is rejected even though the generic quant crate can decode them. Arbitrary destination-view CPY and shape-changing CPY remain open.

ACC and SET accept explicit byte strides and an offset into a contiguous base. ACC is F32; SET is F32/I32. Overlapping ACC writes always add to the original base, matching the source's out-of-place kernel. Complete final write extents, scalar alignment, signed parameter limits, source shapes and every index are checked.

ARGSORT and TOP_K return I32 indices. The source's flattened `row * nb[1]` addressing is preserved with span checks. TOP_K also preserves the original kernel's deliberate first-two-index swap. Tied values follow **MSVC STL 14.51.36231** on MSVC targets and the independently verified **GNU 11.4.0** policy on GNU/Linux; other target policies return an explicit error. Eighty-eight captured ARGSORT cases have different native tie orders between platforms, and each matches its own reference exactly. NaN sort keys are rejected because the original comparison does not establish a strict weak ordering for arbitrary NaN-containing rows; there is no cross-platform NaN-order claim. Integer conversions similarly reject nonfinite/out-of-range values that would invoke undefined C casts.

TOP_K_MASK supports F32/F16 masks and every combination of optional score, block and extra-index tensors. Native invalid selected/column indices are ignored. Scores of NaN or negative infinity suppress a selection; extra indices apply independently. Retained mask bytes preserve their original signed zeros and NaN payloads.

### Convolution, extraction, pooling and resize

Thirteen additional original CPU IDs execute through checked `Op` variants.
`Conv2dParams` and `Conv3dParams` carry signed I32 stride, padding and dilation
arrays in spatial-axis order. Active strides and dilations must be positive.
All operations keep inputs immutable and return independent storage.

| API | Source layout and dtype contract |
|---|---|
| `Im2col` | Kernel then image; 1D image `[IW,IC,N,1]` or 2D `[IW,IH,IC,N]`; F32 to F32/F16, or exact F16 payload copying. |
| `Im2colBack` | Gradient then kernel metadata, matching the actual CPU operand order; explicit original image shape; F32 gradient, F32/F16 kernel metadata. |
| `Im2col3d` | Kernel `[KW,KH,KD,IC*OC]`, image `[IW,IH,ID,IC*N]`; explicit input-channel count, F32 to F32/F16. |
| `Col2im1d` | Contiguous `[K*OC,T,1,1]` matrix to `[Tout,OC,1,1]`; F32/F16/BF16, F32 gather sums and final dtype rounding. |
| `ConvTranspose1d` | Kernel `[K,Cout,Cin,1]`, F32 image `[L,Cin,1,1]`; F32/F16 weights, positive stride, source no-padding/unit-dilation form. |
| `Conv2d`, `Conv3d` | Original direct operators; contiguous F32/F16 weights and strided F32 input. The 3D variant checks packed channel/batch axes. |
| `Conv2dDepthwise` | F32/F16 `[KW,KH,1,C]` weights and F32 image, preserving native WHCN/CWHN layout selection and output strides. |
| `ConvTranspose2d` | F32/F16 `[KW,KH,Cout,Cin]` weights and F32 image; source no-padding form and common spatial stride. |
| `Pool1d`, `Pool2d` | F32/F16 input to F32; `PoolMode::Max` or `Average`; explicit kernel, stride and padding. |
| `Pool2dBack` | F32 gradient then original F32 forward input; explicit matching pooling geometry. |
| `Resize` | F32, explicit four-dimensional output shape, `Nearest`/`Bilinear`/`Bicubic`, align-corners and bilinear antialias flags. |

The implementation preserves several source distinctions that a generic image
library would miss. IM2COL 1D/2D flattens the spatial plane inside each channel,
whereas IM2COL_3D reads every input stride. IM2COL_BACK's CPU operand order is the
reverse of the misleading names in the frozen header comments. Direct
convolution uses F32 products with F64 channel-dot accumulation; its half weights
do not imply half-quantized input activations. Transpose-2D deliberately flattens
each kernel spatial plane during preparation. Depthwise uses F32 accumulation and
retains the source's output byte strides, including the unit-axis case where the
constructor's channel-layout predicate and the CPU's contiguous-layout predicate
both hold.

POOL_1D averages only valid samples. POOL_2D divides by the entire kernel area,
including padding. Float padding first affects the constructor's F32 output-size
calculation and is then truncated to I32 for actual sample coordinates. Max
pooling starts at negative F32 maximum and preserves the original comparator
argument order, signed zeros and NaN payloads. Backward max pooling preserves the
source's first-maximum tie behavior and explicit NaN comparisons. Forward average pooling
retains the first NaN accumulated by the original kernels; its initial payload
mismatch and all native values are preserved in external evidence.

Resize keeps the original F32 coordinate and weight arithmetic, cubic coefficient
-0.75, bilinear triangle filter for antialiasing, align-corners scales and nearest
selection on the two outer axes. It reads actual input byte strides. Dimensions,
packed channel relationships, scalar addresses and allocations are checked.
Source address formulas that would escape the declared view or produce an
invalid native output layout return errors even if a larger shared allocation
happens to contain bytes at that address.

The new cohort contains **961** actual original one-thread GGML graphs:
446 main cases, 490 reduction/layout edge cases and 25 constructor geometry
corners. Each was captured separately from original MSVC and GNU libraries before
its Rust comparison. Both platform captures are byte-identical. All **529,046
output bytes**, shapes and original output strides are compared without numerical
tolerance. The three raw fixtures and six platform provenance records remain
separate so later fixes cannot silently replace the earlier expected values.

Twenty additional original backward-pooling graphs use overlapping gradients
with quiet/signaling NaNs, opposite infinities and signed zeros. They add
1,200 exact output bytes per host. The old Rust GNU path differed in ten
average-pooling cases and 66 F32 words: GNU retains an accumulated NaN, while
the frozen MSVC path and both max-backward paths select the incoming gradient
payload. Explicit operand selection now preserves each measured source policy;
finite arithmetic is unchanged. The earlier mismatch and an initial test-adapter
enum error remain in the evidence. These exceptional-value comparisons run only
on the two verified x86_64 MSVC/GNU targets; other toolchains remain unverified.

The independently verified GNU vector-row policies for existing SiLU, softmax
and NORM are a separate integration item. Their model-runtime helpers do not
automatically change this tensor executor, whose earlier scalar fixture set
targets the frozen MSVC profile.

### Source activation rounding

GELU preserves the frozen half lookup's input/output rounding on F32 and F16 paths. BF16 GELU instead evaluates the direct F32 expression before BF16 conversion. F16 GEGLU and GEGLU_QUICK preserve the intermediate half lookup rounding before multiplication by the second input.

The source has two different softplus expressions. UNARY SOFTPLUS uses `log1pf(expf(x))` below its threshold. XIELU's constructor coefficient helper uses `logf(1 + expf(x))`. The implementation and tiny-value fixtures keep that distinction. REGLU/RELU and hard-sigmoid also preserve the source's NaN and signed-zero branches.

## ERF accuracy and native parity are separate gates

The Rust 1.99.1 toolchain still reports `f32::erf` as the unstable `float_erf` feature. The maintained implementation therefore uses a dependency-free positive-term F64 series, then one F32 conversion. For `|x| >= 4`, the exact ERF value rounds to the returned signed 1 in F32.

For smaller arguments, successive series ratios decrease. After term `t` with next ratio `r < 1`, the omitted positive tail is bounded by `t*r/(1-r)`. Iteration only stops when that bound is at most one eighth of F64 epsilon times the sum; the maximum is 160 terms. This bounds truncation separately from floating evaluation and the standard F64 exp/sqrt implementation. The final F32 rounding contributes at most half the adjacent F32 spacing. Correct rounding for every possible F32 input is not formally proved.

An independent 80-decimal-digit mpmath oracle supplies **48,455 nonnegative arguments**, also checked at their negative counterparts: **96,910** accuracy checks. Inputs include all finite F16/BF16 activation arguments in the nonsaturated range, a dense grid, powers of two and nearby boundary values. Every tested result is the correctly rounded F32 value; the observed maximum absolute rounding error is **2.98014634e-8**. Those mathematical references never replace the native GGML outputs.

**Exact Windows UCRT erff equivalence remains open.** For example, input GELU x = -2.0546875 produces native ERF bits bf75c885, while the correctly rounded mathematical ERF has bits bf75c883. That difference crosses a final F16 rounding boundary: native GELU is -0.040985107421875 and Rust is -0.041015625. The first failing 295-case capture and logs are preserved outside the product with SHA256 `74d04184c941a9058c5321cc568c7e81c458fbe3dcec9aa8ab64a6dc7ae4fa89`. The committed expanded native corpus retains the raw original values and exhaustively checks the finite F16/BF16 ERF domains.

### Comparison criteria, independent of observed maxima

Arithmetic, layout, integer and packed-dot cases compare exact output bits, except NaN payloads. Exhaustive finite F16 GELU/GELU_QUICK and BF16 GELU cases also use exact comparisons.

The F32 transcendental comparison budget remains `B = 2e-6 + 2e-6*abs(reference)`. This is a numerical validation criterion, not a claim of exact platform libm reproduction. For F16/BF16 outputs the bound additionally includes half the adjacent output spacing for each result. If two underlying F32 values differ by at most B and each undergoes a final nearest-even conversion Q, then the triangle inequality gives:

`|Q(a) - Q(b)| <= B + 0.5*spacing(Q(a)) + 0.5*spacing(Q(b))`.

The test derives these spacing terms from dtype bits; it does not set a tolerance from the measured maximum. Finite/nonfinite classifications must still match, and all arithmetic gates stay exact. Four comparisons in the expanded corpus require this final-conversion envelope: three distinct finite F16 GELU_ERF inputs and one duplicate ordinary case. Numerical comparison success does **not** close exact UCRT ERF parity.

## Independent evidence

The fixtures come from independent programs outside this product:

| Fixture | Evidence |
|---|---|
| `cpu-reference.bin` / `provenance.json` | Original 65 actual frozen GGML CPU graphs, one CPU thread. |
| `cpu-extended-reference.bin` / `cpu-extended-provenance.json` | 297 additional actual frozen GGML CPU graphs, retaining raw dtypes, shape, strides, input payloads, parameters and output bytes. Inputs are encoded only by original GGML routines. |
| `cpu-indexing-msvc-reference.bin` / `cpu-indexing-msvc-provenance.json` | 890 new original CPU graph cases captured on Windows; 599,152 output bytes require exact equality. |
| `cpu-indexing-gnu-reference.bin` / `cpu-indexing-gnu-provenance.json` | The same 890 cases independently captured from original GNU/Linux shared libraries; platform tie ordering retained. |
| `cpu-convolution-reference.bin` / `cpu-convolution-{msvc,gnu}-provenance.json` | 446 main image-operation graphs; both native captures have identical bytes. |
| `cpu-convolution-edges-reference.bin` / `cpu-convolution-edges-{msvc,gnu}-provenance.json` | 490 original reduction-payload and unit-axis layout cases; both platform captures identical. |
| `cpu-convolution-corners-reference.bin` / `cpu-convolution-corners-{msvc,gnu}-provenance.json` | 25 signed-division and fractional-padding constructor cases; both platform captures identical. |
| `backward-nan/{msvc,gnu}/cpu-convolution-{reference.bin,provenance.json}` | Twenty additional original overlapping-gradient graphs per host; 1,200 output bytes, including measured NaN operand selection. |
| `erf-high-precision-reference.bin` / `erf-high-precision-provenance.json` | Independent 80-digit mathematical ERF values and explicit nearest-even F32 rounding; no Rust output is used. |

The original **362 native graph cases** include every native weight matmul and out-product dtype, packed RHS input, typed unary/GLU kernels, batch expansion, row padding/permutation, uneven normalization groups, cancellation-sensitive reductions, signed zero/NaN branches, cropping/wrapping, repeated row indices and explicit backward operators. Exhaustive activation captures contain every one of the **63,488 finite F16 bit patterns** and every one of the **65,280 finite BF16 bit patterns**.

In the 297-case extension, **256,233 floating outputs use bit-exact checks** (including exact activation-table cases); **170,260 use the documented transcendental criteria**. On Windows, the latter contain **1,815 bit differences** from native, with maximum F32 difference **4.7683716e-7**. On GNU/Linux they contain **1,838 bit differences**, with maximum F32 difference **1.9073486e-6**; the maximum across all output dtypes is **9.765625e-4**, one F16 step. These observed maxima are reported separately from the criteria above.

The indexing corpus adds **890 distinct graph cases per platform**: 6 ADD_ID, 59 MUL_MAT_ID, 51 SET_ROWS, 116 CPY/cast, 3 ACC, 6 SET, 225 ARGSORT, 408 TOP_K and 16 TOP_K_MASK. All new numerical outputs compare exact bytes in debug and release; no transcendental tolerance applies. Together with the 961 convolution/extraction/pooling/resize cases and 20 additional overlapping backward-pooling controls, the total is **2,233 distinct native graph cases**. The 890 indexing and 981 convolution/pooling cases were independently captured on both platforms, giving **4,104 native graph captures overall**.

Malformed-input tests cover unsupported parameterized variants, native undefined offsets, half-row strides, unsupported BF16 clamped GLU, shape/stride overflow, invalid indices, empty/extreme descriptors, allocation size checks, storage aliasing, I64 identity, node ownership, graph cancellation and zero-input nodes.

Focused validation commands:

- `cargo test -p llama-tensor --offline`
- `cargo test -p llama-tensor --offline --release`
- `cargo clippy -p llama-tensor --all-targets --offline -- -D warnings`
- `cargo fmt -p llama-tensor --check`

Windows and GNU/Linux validation use Rust 1.99.1. The Windows native reference uses MSVC 19.51.36260.0 and STL 14.51.36231; the Linux reference uses GNU 11.4.0. Actual CMake option values and library/header hashes are recorded in the indexing provenance files. The build configurations are not asserted to be identical, and each original graph call uses one CPU thread. GNU's default `__SSE2__` also enables original SiLU/softmax exponential and NORM variance paths that MSVC does not select; the indexing and image-operation cases do not execute those operators, and this slice does not claim their native cross-platform equivalence. Focused debug/release/strict-Clippy results are recorded against the current source hashes; previous checkpoint results are not reused as proof.

## Open T02 gates

The remaining 26 operator IDs, missing dtype/parameter combinations within implemented IDs, advanced RoPE, mask/sink/ALiBi softmax, exact platform ERF behavior, optimized CPU scheduling/SIMD, writable/imported/device storage, asynchronous execution/events, allocation planning/reuse, graph automatic differentiation and optimizer integration remain open. Explicit backward kernels do not imply a training framework. Scalar graph fixtures alone do not establish general pretrained-model inference parity.

Implementation follows the frozen GGML layout/operator semantics under the MIT license, copyright (c) 2023-2026 The ggml authors. The adapted Microsoft STL sorting helpers use Apache-2.0 WITH LLVM-exception. The combined crate license is MIT AND (Apache-2.0 WITH LLVM-exception); see THIRD_PARTY_NOTICES.md and the included license texts.
