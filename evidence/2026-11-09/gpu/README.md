# Native Windows GPU verification — 9 November 2026

## Implemented changes

- Added a checked CUDA `MatVecF32` primitive to the existing typed Rust PTX representation. It supports independent matrix row/column strides, input-vector strides, output strides, input broadcasting and partial launch tails. Buffer extents and output aliasing are checked before launch. This is scalar-per-row device execution; it is not a tensor-core kernel or a full model-offload implementation.
- Added two independent semantic regression tests for the new primitive.
- Added a Windows dynamic-loader test that resolves and calls the actual `GetCurrentProcessId` system symbol and verifies missing/invalid-width symbol errors.
- Added `cuda_conformance` and `opencl_conformance` executable examples. They compare device results with direct scalar/F64 formulas and frozen native GGML fixture data, without using either kernel interpreter as the device oracle. They verify unchanged input buffers, output padding, noncontiguous layouts, launch tails and invalid-buffer rejection.

## Hardware execution actually observed

| API / device | Coverage in each debug/release run | Maximum absolute error |
| --- | --- | --- |
| CUDA / NVIDIA GeForce RTX 4090, driver 621.01, compute capability 8.9 | 138 real dispatches: 69 with PTX 7.0 / SM60 and 69 with PTX 7.8 / SM89. Five kinds: add, scale, sum, F32 matvec, Q4_0 matvec. Each target exercises all 256 frozen native Q4_0 reference values plus padded-block/row and strided-vector cases. | 0 |
| OpenCL / NVIDIA GeForce RTX 4090, OpenCL 3.0 CUDA | Ten real compiler/dispatch/readback cases spanning all nine built-ins: add, multiply, SiLU, scale, row sum, RMS normalization, softmax, matrix multiplication, Q4_0 matvec; an additional large-logit softmax case. | 2.3841858e-7 |
| OpenCL / AMD gfx1036, driver 3652.0 (PAL,LC), OpenCL 2.0 | The same ten independently checked cases. | 1.1920929e-7 |

OpenCL Q4_0 uses an unaligned three-byte base, padded rows and a strided vector. CUDA Q4_0 additionally tests padded blocks. Output padding is checked exactly where preserved; numeric results use the explicit tolerances in the examples. Both APIs reject undersized buffers while preserving caller data.

The archived CUDA smoke also passed device and pinned-memory transfer plus stream/event synchronization. Its first execution used the available Rust 1.97.1 with `--ignore-rust-version`; this is labelled `baseline-*` only. All expanded conformance and final checks used the installed pinned **Rust 1.99.1**.

## Native source differential check

`frozen_quant_oracle.log` records an explicit run against the newly compiled, separate reference:

`C:\Users\adyba\rust-llama-reference\build\bin\Release\ggml-base.dll`

- All **65,536 binary16 encodings** match native `ggml_fp16_to_fp32`; NaNs are compared by class.
- Unaligned, padded, strided Q4_0 matvec matches native dequantization followed by an independent F64 dot: `[54.0, -88.75, 1.7285347e-6]` on both sides.
- The native library remains outside the product, used only by the opt-in differential test.

The reference DLL and source ZIP hashes are recorded in `verification.json`.

## Validation

- Focused default suites: **39 tests passed**, zero failed, two external tests intentionally ignored.
- The native differential test above was then enabled explicitly and passed: **40 distinct tests exercised** in total. Debug/release repetitions are not added as new tests.
- Focused release suites: the same **39 default tests passed**.
- Warning-denied Clippy passed for all targets of the four assigned crates.
- Rustfmt passed on all changed Rust files.
- Debug and release GPU conformance both passed. Counts in the hardware table are per run.

The remaining ignored external test requires shaderc. No new Vulkan shader-compiler or Vulkan execution result is claimed here.

## Reproduction from the repository root

```text
cargo test --offline --locked -p llama-device-family -p llama-platform-loader -p llama-kernel-ir -p llama-portable-gpu
cargo clippy --offline --locked -p llama-device-family -p llama-platform-loader -p llama-kernel-ir -p llama-portable-gpu --all-targets -- -D warnings
cargo run --offline --locked -p llama-device-family --example cuda_conformance
cargo run --offline --locked -p llama-portable-gpu --example opencl_conformance
cargo run --release --offline --locked -p llama-device-family --example cuda_conformance
cargo run --release --offline --locked -p llama-portable-gpu --example opencl_conformance
```

For the frozen-reference check, set `LLAMA_REFERENCE_LIBRARY` to a separately built compatible GGML library, then:

```text
cargo test --offline --locked -p llama-portable-gpu --test frozen_quant_oracle -- --ignored --nocapture
```

CUDA conformance selects SM60 and also SM89 when the actual device supports it. An unavailable CUDA driver exits 77, not success. OpenCL conformance executes on every enumerated device; absent devices or compilation/dispatch errors fail the command. Neither example substitutes a CPU result for missing hardware.

## Remaining limitations

This establishes actual Windows CUDA/OpenCL primitive execution on the two named devices. It does not establish full model GPU offload, full dtype/operator coverage, tensor-core optimization, multi-GPU computation, fault/cancellation stress, performance parity, HIP/MUSA execution, or Metal/Vulkan/WebGPU host execution. All complete backend release gates remain open.
