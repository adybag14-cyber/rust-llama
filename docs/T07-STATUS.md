# T07 — Portable accelerator and shared kernel progress

**Date:** 8 November 2026. **Baseline:** uploaded archive SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

This is a tested implementation checkpoint. None of the four complete backend exit gates is closed. The default Rust workspace builds offline without external Cargo packages. No old GGML implementation or handwritten native shader supplies a production capability.

## Implemented behavior

`llama-kernel-ir` provides IR version 1: scalar F32/U32/I32/Bool values, typed storage buffers, read/write permissions, checked declarations and expressions, byte/element addressing, conditionals, counted loops, bitcasts, and IEEE binary16 decoding. There is no raw source statement or template escape hatch. Integer/layout arithmetic, identifiers, dimensions, workgroup sizes and bindings are checked. Writable-buffer reads are rejected because synchronization semantics are not yet represented.

Four emitters traverse those Rust semantics and produce OpenCL C 1.2, Vulkan GLSL 450, WGSL and MSL. The generated files are derived artifacts, never the authoritative implementation. SHA-256 identifies every emitted source and binary. Cache keys include source digest, compiler, target, device, driver, capabilities and flags with length-delimited encoding. No cache key substitutes for actual device capability validation.

The nine built-in kernels are F32 add, multiply, SiLU, scale, row sum, RMSNorm, stable finite-row softmax, matrix multiplication, and Q4_0 matrix-vector multiplication. Strided inputs, padded outputs and partial workgroups are exercised. Q4_0 supports arbitrary byte offsets, row padding, packed half scales and strided input vectors. Its two 16-value nibble halves match the frozen `block_q4_0` representation. Other quantization formats and fused attention are not implemented.

`DispatchPlan` is a sealed safe launch boundary. Only checked built-in builders can create one; exposing and editing a cloned generic IR does not modify a plan. The public generic IR remains usable for source generation and bounded interpretation, but cannot be submitted through the safe OpenCL execution API. This avoids presenting type checking as a proof of arbitrary kernel memory safety.

The scalar interpreter checks actual reads/writes, disallows competing work items from writing the same element, enforces fuel for invocations/statements/loop iterations, and commits output buffers only after the whole dispatch succeeds. It is a validation engine, not GPU execution or a production T03 CPU backend. It supports the same loop-bound snapshot rule as all four emitters.

## Host and compiler boundaries

| Path | Implemented | Actually established on this host |
| --- | --- | --- |
| OpenCL | Real loader/platform/device queries; driver/version/extensions/IL reports; checked context/queue/program/kernel/buffer ownership; source build diagnostics; synchronous upload, dispatch and readback for sealed plans | Loader opens, but `clGetPlatformIDs` returns `CL_PLATFORM_NOT_FOUND_KHR (-1001)`. No device kernel ran. |
| Vulkan | Real loader version, instance creation and physical-device enumeration with bounded retry/cleanup | Loader is installed. Exact instance/device result is in the delivery probe log. Buffer/pipeline/command execution remains pending. |
| Vulkan compiler | Optional external `libshaderc`, with owned compiler/results, source-hash validation and SPIR-V magic/size checks | All nine generated GLSL programs compile to actual SPIR-V. Invalid shader and stale source-provenance inputs are rejected. This is compiler evidence, not device execution. |
| Metal | Rust-derived MSL output | MSL compiler validation and Metal host/resource execution are pending; no Apple runner is available. |
| WebGPU | Rust-derived WGSL output | Native/browser host adapter, WASM integration, asynchronous mapping/device loss and WGSL compiler/device validation are pending. |

The installed Clang library was also examined as an optional OpenCL compiler route, but its resource header `opencl-c-base.h` is absent. A small OpenCL translation-unit probe failed with that exact missing-header diagnostic. No OpenCL source-compiler pass is claimed from this failed probe.

The optional GLSL compiler is an external build service, named explicitly. It is not bundled as a Rust rewrite. OpenCL compilation, when a device is available, similarly uses the device's external compiler. These are allowed compiler/platform boundaries under the supplied plan.

## Numerical and independent verification

The ordinary Rust suite compares arithmetic with independent formulas/F64 accumulation, checks strides and untouched padding, rejects overflow/type/name errors, verifies all 65,536 binary16 encodings against a separately expressed normalization algorithm, and tests bounds/race/fuel rollback. SHA-256 has published empty/`abc`/million-`a` test vectors. Mutable loop bounds, deeply nested selects and empty dispatches have explicit regression tests.

The opt-in `frozen_quant_oracle` test loads a separately built frozen GGML reference only in the test process. It compares every binary16 encoding with the actual native conversion (NaNs compared by class) and compares unaligned/strided Q4_0 matvec with actual `dequantize_row_q4_0` output plus independent F64 dots. Observed outputs for the three-row fixture are `[54.0, -88.75, 1.7285347e-6]` on both sides. This verifies the small numerical slice; it does not establish complete dtype/backend parity.

Cross-track tests exercise the overlapping T06/T07 semantics on the same strided/broadcast add and padded Q4_0 fixtures. These expose inconsistent contracts between the two provisional representations. Shared lowering and T02/T03 integration remain necessary; agreement between interpreters is not independent GPU proof.

## Review corrections

Independent peer review produced concrete fixes before packaging:

- Charge loop iterations and dispatch invocations before decrementing fuel; return errors instead of underflow or unbounded empty work.
- Infer each `Select` subtree once, avoiding exponential validation on deeply nested expressions.
- Snapshot loop bounds once in emitted code, matching the interpreter when an outer bound variable is reassigned.
- Restrict safe device launches to sealed checked plans, leaving generic IR unable to bypass device bounds/write-ownership requirements.
- Recompute and validate source SHA-256 at the compiler boundary.
- Distinguish Vulkan absence from loader/API failures and account for both uploaded and downloaded buffer bytes.
- Keep result publication transactional and avoid copying unrelated buffers.

The driver paths still need real-device fault, cancellation, teardown and allocation-stress tests. Successful Rust compilation does not close those gates.

## Frozen coverage denominator

T07 owns 601 archived paths: 597 files beneath its four backend directories plus four public headers. Directory contents are:

| Directory | Total files | Device source count |
| --- | ---: | ---: |
| `ggml-metal` | 57 | 38 `.metal` |
| `ggml-vulkan` | 203 | 162 `.comp`, 30 `.glsl` |
| `ggml-opencl` | 278 | 271 `.cl` |
| `ggml-webgpu` | 59 | 47 `.wgsl`, 7 templates |

The hashed responsibility ledger leaves all source capability rows open. Nine portable kernels do not replace those complete backend surfaces. Subgroups, cooperative/tensor matrices, tiled reductions/GEMM, all quantization formats, BF16/FP8, images/textures, atomics/barriers, training, graph/fusion dispatch, allocation imports, API validation layers, browsers, device loss and performance parity remain pending.

## Reproduction

With the supplied Rust 1.99.1 binaries on `PATH`, run from the workspace root:

```text
cargo test --offline --locked -p llama-kernel-ir -p llama-portable-gpu
cargo test --offline --locked -p llama-portable-gpu --test compiler_validation glsl_corpus_compiles_to_real_spirv -- --ignored --nocapture
cargo run --offline --locked -p track-runner -- generate-kernels generated --compile-vulkan
cargo run --offline --locked -p track-runner -- opencl-smoke
```

The last command requires a real OpenCL platform and must fail explicitly when one is absent. The compiler test requires `libshaderc`. For the independent quantization oracle, use the T05 Rust `build_reference` helper to build the frozen reference outside the product, then set `LLAMA_REFERENCE_LIBRARY` for the explicitly ignored `frozen_quant_oracle` test. See `VERIFICATION.md` for the exact recorded commands and outputs.

## Primary API references

- Frozen source: `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`, and the four backend directories above.
- [Khronos OpenCL API specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html).
- [Khronos command-queue C signature](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clCreateCommandQueue.html).
- [W3C WGSL specification](https://www.w3.org/TR/WGSL/).
- [Khronos physical-device enumeration](https://github.khronos.org/Vulkan-Site/refpages/latest/refpages/source/vkEnumeratePhysicalDevices.html).
- [Google shaderc C interface](https://github.com/google/shaderc/blob/main/libshaderc/include/shaderc/shaderc.h).

These documents informed ABI/syntax work; only the recorded local experiments are execution evidence. Accessed 8 November 2026.
