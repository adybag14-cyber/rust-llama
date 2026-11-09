# Rust tensor layouts and scalar CPU graphs

This is the first executable T02 slice of the frozen llama.cpp rewrite. It does not close T02's scheduler, training or full operator gates.

## Storage and layout

`TensorDesc` preserves GGML's four logical dimensions (`ne`) and byte strides (`nb`). Dimension 0 is fastest. For quantized dtypes, `nb[0]` measures complete blocks while `ne[0]` counts scalar values. Construction checks complete block rows, multiplication/addition overflow, the full strided storage span and buffer bounds. `Tensor::from_bytes` requires a buffer matching the descriptor span. Empty shapes are permitted for storage and copying; operations requiring nonempty rows reject them explicitly.

`Tensor` holds immutable `Arc<[u8]>` storage. Views, permutations, transposes and contiguous reshapes share storage and preserve its lifetime. Numeric operations allocate a separate output, so aliased inputs cannot be overwritten. Public APIs never return a mutable view. A reshape requires contiguous logical storage and an unchanged element count. Moving a quantized block axis out of axis 0 is explicitly unsupported. View offsets are conservatively required to be multiples of the dtype block's byte size; general padded quantized subview origins need further work.

The `contiguous` operation copies logical blocks and removes stride holes without changing dtype. `to_f32` uses the independently validated `llama-quant` decoder; integer row gathering preserves I32 output when the source is I32.

## Graph execution

A `Graph` has graph-specific `NodeId`s. Appended operations may only reference existing nodes of the same graph, making cycles unrepresentable. Evaluation marks target dependencies iteratively, computes them in order and ignores unrelated branches. Repeated evaluation is supported. Cancellation is checked during dependency traversal, between operators and within the scalar operator loops; errors and cancellation leave immutable inputs intact.

The exact frozen registries contain 102 operation IDs, 22 unary IDs and 7 GLU IDs. Registry membership is only an ABI inventory. `Op` exposes actual executable paths, and the two pending unary / three pending GLU variants return explicit errors, including for empty tensors.

Implemented scalar F32 paths:

- Duplication/contiguous copy, reshape, permutation and transpose.
- ADD/SUB/MUL/DIV with GGML periodic broadcasting, scalar ADD1 and SCALE with bias.
- SQR/SQRT/LOG/SIN/COS, SUM/SUM_ROWS/MEAN, REPEAT and CONCAT across any axis.
- MUL_MAT with GGML's dimension ordering and grouped batch expansion. The left batch index uses division by the repetition factor, not modulo.
- SOFT_MAX (scale, no auxiliary mask/sinks/ALiBi), RMS_NORM and NORM.
- Twenty unary variants: ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK, SILU, HARDSWISH, HARDSIGMOID, EXP, EXPM1, SOFTPLUS, FLOOR, CEIL, ROUND and TRUNC.
- Binary REGLU/GEGLU/SWIGLU/GEGLU_QUICK with an explicit gate swap option.
- CLAMP, causal DIAG_MASK_INF/DIAG_MASK_ZERO, and basic normal/NeoX RoPE with positions, frequency base and frequency scale.
- GET_ROWS with checked indices, batch dimensions and I32 preservation. Quantized source rows can be decoded through the codec crate.

The frozen CPU defines `GGML_GELU_FP16`; GELU and GEGLU therefore preserve its half-rounded lookup input/output and saturation behavior. Other numerical paths use Rust scalar arithmetic. No vector throughput claim is made.

## Evidence

`tests/fixtures/cpu-reference.bin` contains 65 independent cases constructed and executed by the actual frozen GGML CPU DLLs, one CPU thread, outside this product. The fixture includes raw inputs, operator parameters, output shapes/types and reference results. No Rust output was used to generate expected values. `tests/fixtures/provenance.json` records all case names, the fixture hash and DLL hashes.

The oracle cases cover the implemented operations, four concat axes, grouped-batch matmul, swapped GLU gates, normal/NeoX RoPE, permuted views, and batched/I32 gathering. Numerical comparisons use `3e-6 + 3e-6*abs(reference)` with exact shape/type checks and explicit infinity/NaN handling. Additional tests cover strided/quantized views, lifetime/aliasing, malformed layouts, overflow, cancellation, graph ownership, re-evaluation, unrelated pending branches and invalid operation parameters.

Validated on Windows x86-64 with Rust 1.99.1:

- `cargo test -p llama-tensor --offline`
- `cargo test -p llama-tensor --offline --release`
- `cargo clippy -p llama-tensor --all-targets --offline -- -D warnings`
- `cargo fmt -p llama-tensor --check`

## Open T02 gates

The full operation set, GELU_ERF/XIELU, remaining GLUs, advanced RoPE, masked/sink/ALiBi softmax, mixed-dtype math, optimized CPU scheduling, writable/imported/device buffers, asynchronous execution/events, allocation planning/reuse, automatic differentiation and optimizers remain open. No implicit CPU fallback or fake accelerator result is provided. General model execution is being integrated separately; these scalar graph tests alone do not establish pretrained-model inference parity.

The implementation is derived from the frozen GGML layout/operator semantics under the MIT license, copyright (c) 2023-2026 The ggml authors; see the repository's preserved upstream notices.
