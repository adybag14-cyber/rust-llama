# Frozen CPU operation inventory

This table inventories all **102** frozen operation IDs. **62 IDs** have explicit `Op` constructors backed by Rust execution, and `NONE` is represented by an input leaf. The other **39 IDs** do not have executable constructors. An exposed constructor does not imply that every source dtype, shape, stride or parameter mode has been ported: evaluation checks the restrictions below and returns an error for unsupported inputs.

Numeric IDs are preserved in `src/registry.rs`, derived from the frozen `ggml/include/ggml.h`. Actual execution is in `src/cpu.rs`, `src/cpu_extra.rs`, `src/cpu_index.rs` and `src/native_sort.rs`; registration alone is not execution evidence. All new supported paths are exercised by independent original-GGML graphs described in the crate README. The explicit backward kernels do not constitute graph automatic differentiation.

| ID | Frozen operation | Execution status | Dtypes / result | Current contract |
|---:|---|---|---|---|
| 0 | `NONE` | Input leaf | All storage dtypes | Existing immutable tensor value. |
| 1 | `DUP` | Implemented slice | All active storage dtypes | Immutable storage/view paths; complete quantized rows and checked byte spans; quantized block axis stays axis 0. |
| 2 | `ADD` | Implemented slice | F32 | Periodic broadcasting; ADD1 requires one scalar; SCALE supports explicit bias. |
| 3 | `ADD_ID` | Implemented slice | F32 + F32 + I32 → F32 | Checked expert biases; ID -1 copies a row; source first-slab addressing retained. |
| 4 | `ADD1` | Implemented slice | F32 | Periodic broadcasting; ADD1 requires one scalar; SCALE supports explicit bias. |
| 5 | `ACC` | Implemented slice | F32 | Checked byte-strided out-of-place updates; overlaps read the original base; aligned signed source parameters. |
| 6 | `SUB` | Implemented slice | F32 | Periodic broadcasting; ADD1 requires one scalar; SCALE supports explicit bias. |
| 7 | `MUL` | Implemented slice | F32 | Periodic broadcasting; ADD1 requires one scalar; SCALE supports explicit bias. |
| 8 | `DIV` | Implemented slice | F32 | Periodic broadcasting; ADD1 requires one scalar; SCALE supports explicit bias. |
| 9 | `SQR` | Implemented slice | F32 / F16 / BF16 | Compute source scalar expression in F32, then preserve output dtype. |
| 10 | `SQRT` | Implemented slice | F32 / F16 / BF16 | Compute source scalar expression in F32, then preserve output dtype. |
| 11 | `LOG` | Implemented slice | F32 / F16 / BF16 | Compute source scalar expression in F32, then preserve output dtype. |
| 12 | `SIN` | Implemented slice | F32 / F16 / BF16 | Compute source scalar expression in F32, then preserve output dtype. |
| 13 | `COS` | Implemented slice | F32 / F16 / BF16 | Compute source scalar expression in F32, then preserve output dtype. |
| 14 | `SUM` | Implemented slice | F32 / F16 / BF16 | F32 row grouping/strided scan and F32 accumulation for F16/BF16; half/bfloat rows need contiguous axis 0. |
| 15 | `SUM_ROWS` | Implemented slice | F32 | Nonempty rows; source accumulation order and casts. |
| 16 | `CUMSUM` | Implemented slice | F32 | Nonempty rows; source accumulation order and casts. |
| 17 | `MEAN` | Implemented slice | F32 | Nonempty rows; source accumulation order and casts. |
| 18 | `ARGMAX` | Implemented slice | F32 → I32 | Matrix input only; source last-maximum tie and NaN ordering. |
| 19 | `COUNT_EQUAL` | Implemented slice | I32 → I64 | Checked source address formula, including frozen higher-axis behavior; invalid source addresses are errors. |
| 20 | `REPEAT` | Implemented slice | F32 | Four-dimensional tiling; REPEAT_BACK preserves per-addition F32 rounding. |
| 21 | `REPEAT_BACK` | Implemented slice | F32 | Four-dimensional tiling; REPEAT_BACK preserves per-addition F32 rounding. |
| 22 | `CONCAT` | Implemented slice | F32 | All four axes; remaining dimensions must match. |
| 23 | `SILU_BACK` | Implemented slice | F32 / F16 | Matching input dtypes/shapes; explicit backward kernel. |
| 24 | `NORM` | Implemented slice | F32 | Explicit finite nonnegative epsilon; GROUP_NORM supports uneven groups; NORM preserves strided F32 versus contiguous F64 reduction. |
| 25 | `RMS_NORM` | Implemented slice | F32 | Explicit finite nonnegative epsilon; GROUP_NORM supports uneven groups; NORM preserves strided F32 versus contiguous F64 reduction. |
| 26 | `RMS_NORM_BACK` | Implemented slice | F32 | Explicit backward kernels; softmax scale supported, positive max-bias/ALiBi derivative remains absent. |
| 27 | `GROUP_NORM` | Implemented slice | F32 | Explicit finite nonnegative epsilon; GROUP_NORM supports uneven groups; NORM preserves strided F32 versus contiguous F64 reduction. |
| 28 | `L2_NORM` | Implemented slice | F32 | Explicit finite nonnegative epsilon; GROUP_NORM supports uneven groups; NORM preserves strided F32 versus contiguous F64 reduction. |
| 29 | `MUL_MAT` | Implemented slice | 29 native weight dtypes → F32 | Native activation packing/dot arithmetic and grouped batches; prepacked RHS retained; source F16/BF16 widening where supported. |
| 30 | `MUL_MAT_ID` | Implemented slice | 29 native weight formats → F32 | Checked expert IDs, -1 zero slots, native activation packing, slot broadcasting; RHS F32 or exact activation dtype only. |
| 31 | `OUT_PROD` | Implemented slice | 28 native left dtypes × F32 → F32 | F32 multiply/add accumulation after source decoding; BF16 has no frozen CPU kernel; non-F32 batches must match. |
| 32 | `SCALE` | Implemented slice | F32 | Periodic broadcasting; ADD1 requires one scalar; SCALE supports explicit bias. |
| 33 | `SET` | Implemented slice | F32 / I32 | Independent storage; checked byte-strided writes into a contiguous base; untouched bytes preserved. |
| 34 | `CPY` | Implemented slice | Same dtype / source CPU conversions | Cast exposes same-shape conversion; 25 CPU encoders and source-supported decoder pairs; arbitrary destination views/reshapes remain open. |
| 35 | `CONT` | Implemented slice | All active storage dtypes | Immutable storage/view paths; complete quantized rows and checked byte spans; quantized block axis stays axis 0. |
| 36 | `RESHAPE` | Implemented slice | All active storage dtypes | Immutable storage/view paths; complete quantized rows and checked byte spans; quantized block axis stays axis 0. |
| 37 | `VIEW` | Implemented slice | All active storage dtypes | Immutable storage/view paths; complete quantized rows and checked byte spans; quantized block axis stays axis 0. |
| 38 | `PERMUTE` | Implemented slice | All active storage dtypes | Immutable storage/view paths; complete quantized rows and checked byte spans; quantized block axis stays axis 0. |
| 39 | `TRANSPOSE` | Implemented slice | All active storage dtypes | Immutable storage/view paths; complete quantized rows and checked byte spans; quantized block axis stays axis 0. |
| 40 | `GET_ROWS` | Implemented slice | Decoder-supported input; I32 retained | Checked batched indices; non-I32 output is F32; source-type coverage is narrower than the generic decoder API. |
| 41 | `GET_ROWS_BACK` | Implemented slice | F32 / F16 + I32 → F32 | Matrix/vector form; checked indices and repeated-index accumulation; explicit output row count. |
| 42 | `SET_ROWS` | Implemented slice | F32/F16 + I32/I64 → 25 CPU destination types | Preserves destination strides and untouched bytes; one-thread duplicate write order; F16 raw copying. |
| 43 | `DIAG` | Implemented slice | F32 | ne1 == 1; supports batches. |
| 44 | `DIAG_MASK_INF` | Implemented slice | F32 | Explicit causal past length. |
| 45 | `DIAG_MASK_ZERO` | Implemented slice | F32 | Explicit causal past length. |
| 46 | `SOFT_MAX` | Implemented slice | F32 | Scalar scale only; auxiliary masks, sinks and ALiBi remain pending. |
| 47 | `SOFT_MAX_BACK` | Implemented slice | F32 | Explicit backward kernels; softmax scale supported, positive max-bias/ALiBi derivative remains absent. |
| 48 | `ROPE` | Implemented slice | F32 | Basic normal and NeoX modes, positions/base/scale; extended/multimodal/YaRN modes remain pending. |
| 49 | `ROPE_BACK` | Pending | — | No executable Op constructor; no fallback result. |
| 50 | `CLAMP` | Implemented slice | F32 | Finite ordered limits. |
| 51 | `CONV_TRANSPOSE_1D` | Pending | — | No executable Op constructor; no fallback result. |
| 52 | `IM2COL` | Pending | — | No executable Op constructor; no fallback result. |
| 53 | `IM2COL_BACK` | Pending | — | No executable Op constructor; no fallback result. |
| 54 | `IM2COL_3D` | Pending | — | No executable Op constructor; no fallback result. |
| 55 | `COL2IM_1D` | Pending | — | No executable Op constructor; no fallback result. |
| 56 | `CONV_2D` | Pending | — | No executable Op constructor; no fallback result. |
| 57 | `CONV_3D` | Pending | — | No executable Op constructor; no fallback result. |
| 58 | `CONV_2D_DW` | Pending | — | No executable Op constructor; no fallback result. |
| 59 | `CONV_TRANSPOSE_2D` | Pending | — | No executable Op constructor; no fallback result. |
| 60 | `POOL_1D` | Pending | — | No executable Op constructor; no fallback result. |
| 61 | `POOL_2D` | Pending | — | No executable Op constructor; no fallback result. |
| 62 | `POOL_2D_BACK` | Pending | — | No executable Op constructor; no fallback result. |
| 63 | `UPSCALE` | Pending | — | No executable Op constructor; no fallback result. |
| 64 | `PAD` | Implemented slice | F32 | Signed crop/pad and circular forms; checked source-defined address domain. |
| 65 | `PAD_REFLECT_1D` | Implemented slice | F32 | Reflection pads strictly smaller than the input row. |
| 66 | `ROLL` | Implemented slice | F32 | Four axes; absolute shift must be smaller than its axis. |
| 67 | `ARANGE` | Implemented slice | No inputs → F32 | Finite start/stop/positive step, increasing range and checked allocation. |
| 68 | `TIMESTEP_EMBEDDING` | Implemented slice | F32 | Vector input, positive I32 dimensions/period; odd final channel is zero. |
| 69 | `ARGSORT` | Implemented slice | F32 → I32 | Native ASC/DESC with checked source flattened row addressing; MSVC/GNU tie policies; NaN keys rejected. |
| 70 | `TOP_K` | Implemented slice | F32 → I32 | Native partial selection and deliberate first-two swap; checked k; platform-verified tie policy; NaN keys rejected. |
| 71 | `LEAKY_RELU` | Implemented slice | F32 / F16 | Explicit slope or fill value. |
| 72 | `TRI` | Implemented slice | F32 | Square batched matrices; four exact native triangle IDs. |
| 73 | `FILL` | Implemented slice | F32 / F16 | Explicit slope or fill value. |
| 74 | `FLASH_ATTN_EXT` | Pending | — | No executable Op constructor; no fallback result. |
| 75 | `FLASH_ATTN_BACK` | Pending | — | No executable Op constructor; no fallback result. |
| 76 | `SSM_CONV` | Pending | — | No executable Op constructor; no fallback result. |
| 77 | `SSM_SCAN` | Pending | — | No executable Op constructor; no fallback result. |
| 78 | `WIN_PART` | Pending | — | No executable Op constructor; no fallback result. |
| 79 | `WIN_UNPART` | Pending | — | No executable Op constructor; no fallback result. |
| 80 | `GET_REL_POS` | Pending | — | No executable Op constructor; no fallback result. |
| 81 | `ADD_REL_POS` | Pending | — | No executable Op constructor; no fallback result. |
| 82 | `RWKV_WKV6` | Pending | — | No executable Op constructor; no fallback result. |
| 83 | `GATED_LINEAR_ATTN` | Pending | — | No executable Op constructor; no fallback result. |
| 84 | `RWKV_WKV7` | Pending | — | No executable Op constructor; no fallback result. |
| 85 | `SOLVE_TRI` | Pending | — | No executable Op constructor; no fallback result. |
| 86 | `GATED_DELTA_NET` | Pending | — | No executable Op constructor; no fallback result. |
| 87 | `LIGHTNING_INDEXER` | Pending | — | No executable Op constructor; no fallback result. |
| 88 | `DSV4_HC_COMB` | Pending | — | No executable Op constructor; no fallback result. |
| 89 | `DSV4_HC_PRE` | Pending | — | No executable Op constructor; no fallback result. |
| 90 | `DSV4_HC_POST` | Pending | — | No executable Op constructor; no fallback result. |
| 91 | `TOP_K_MASK` | Implemented slice | F32/F16 mask + I32 indices | Optional score/block/extra inputs, source invalid-index skipping, batch repeats, raw retained payload bits. |
| 92 | `UNARY` | Implemented slice | F32 / F16 / BF16 | All 22 scalar formulas; XIELU needs explicit parameters; exact UCRT ERF bit parity remains open. |
| 93 | `MAP_CUSTOM1` | Pending | — | No executable Op constructor; no fallback result. |
| 94 | `MAP_CUSTOM2` | Pending | — | No executable Op constructor; no fallback result. |
| 95 | `MAP_CUSTOM3` | Pending | — | No executable Op constructor; no fallback result. |
| 96 | `CUSTOM` | Pending | — | No executable Op constructor; no fallback result. |
| 97 | `CROSS_ENTROPY_LOSS` | Pending | — | No executable Op constructor; no fallback result. |
| 98 | `CROSS_ENTROPY_LOSS_BACK` | Pending | — | No executable Op constructor; no fallback result. |
| 99 | `OPT_STEP_ADAMW` | Pending | — | No executable Op constructor; no fallback result. |
| 100 | `OPT_STEP_SGD` | Pending | — | No executable Op constructor; no fallback result. |
| 101 | `GLU` | Implemented slice | F32 / F16 / BF16, with limits below | All seven scalar variants; OAI/CLAMP support F32/F16 only; packed/split and gate swaps. |

## Unary variants

All 22 unary IDs have scalar execution for F32/F16/BF16. Twenty-one use `Op::Unary`; XIELU uses `Op::Xielu { alpha_n, alpha_p, beta, epsilon }`, because its coefficients are constructor parameters. Calling the bare XIELU enum returns an explicit error, including for an empty tensor.

GELU is intentionally dtype-specific: F32 follows the frozen half lookup with its source saturation handling, F16 uses the half table value, and BF16 computes the direct F32 formula before BF16 conversion. GELU_QUICK's half path rounds its lookup result before a later GLU multiplication. UNARY SOFTPLUS uses source `log1pf(expf(x))`; XIELU's coefficient helper instead uses source `logf(1 + expf(x))`. ERF has the mathematical-accuracy and native-output distinction documented in the README.

## GLU variants

| ID | Variant | Dtypes | API |
|---:|---|---|---|
| 0 | REGLU | F32/F16/BF16 | `Glu` / `GluPacked` |
| 1 | GEGLU | F32/F16/BF16 | `Glu` / `GluPacked`; dtype-specific GELU rounding |
| 2 | SWIGLU | F32/F16/BF16 | `Glu` / `GluPacked` |
| 3 | SWIGLU_OAI | F32/F16 | `SwiGluOai` / `SwiGluOaiPacked`; explicit alpha and limit |
| 4 | GEGLU_ERF | F32/F16/BF16 | `Glu` / `GluPacked`; portable ERF numerical contract |
| 5 | GEGLU_QUICK | F32/F16/BF16 | `Glu` / `GluPacked` |
| 6 | SWIGLU_CLAMP | F32/F16 | `SwiGluClamp` / `SwiGluClampPacked`; explicit limit |

Packed GLUs split each row into two floor-sized halves, matching the source's odd-row behavior, and can swap the gate half. Split tensors must have matching shapes and dtypes. Bare OAI/CLAMP enum variants reject missing required parameters rather than inventing defaults. BF16 OAI/CLAMP are explicitly unsupported because the frozen CPU has no such kernels.

## Indexing domains and platform ordering

The README lists every native CPU conversion destination and CPY pair. Missing CPU traits return errors even when a generic quant codec exists. Writable aliasing, arbitrary CPY destination shapes/views, additional native thread schedules and platform sort policies beyond pinned MSVC/GNU remain open. NaN sort keys violate the source comparator's strict weak ordering and are rejected; no portable NaN ordering is claimed. Nonfinite/out-of-range integer casts and source byte addresses outside their declared buffer spans are errors.

## Still open

Operation IDs marked pending, unsupported dtype combinations within implemented IDs, advanced attention/RoPE modes, allocation planning, automatic differentiation, optimizer integration, CPU parallel scheduling, SIMD optimization, imported/writable/device storage and asynchronous events remain open. The portable ERF implementation's exact equivalence to Windows UCRT `erff` also remains open; a numerical-comparison pass does not close that requirement.
