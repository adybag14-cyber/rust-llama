# Target and runtime evidence

| Surface | Status in this checkpoint | Evidence boundary |
|---|---|---|
| Linux x86_64 GNU host Rust 1.99.1 | Compiled and executed | Pure Rust media, behavior, server transport and validation tools; final command logs in `evidence/`. |
| Linux CPU model inference backend | Not supplied | Host scalar preprocessing/sampling is not a completed T03 tensor backend or T05 runtime. |
| Other Linux architectures / musl | Untested | No corresponding Rust standard libraries or runner was installed. |
| Windows | Untested | No Windows target build, OS integration or native client validation. |
| Apple/macOS/iOS | Untested | No Apple SDK, native app or Metal execution. |
| Android | Untested | No Android target/SDK or app lifecycle run. |
| Browser/WASM | Untested | No WASM standard library/browser bindings/UI migration execution. The server can emit a Rust-authored diagnostics HTML page. |
| Device backends | Not supplied | T06–T08 work belongs to separate track outputs, which were not attached. |

## Backend closure

Every backend in the original plan remains in scope. This package does not claim any of these backends is complete.

| Backend | Owner | Current required next input |
|---|---|---|
| CPU | T03 | Rust tensor kernels plus T02/T05 end-to-end inference and ISA checks |
| CUDA | T06 | Reviewed Rust host/device implementation, CUDA SDK/device evidence |
| HIP | T06 | Reviewed Rust host/device implementation, HIP SDK/device evidence |
| MUSA | T06 | Reviewed Rust host/device implementation, MUSA SDK/device evidence |
| Metal | T07 | Rust host/device implementation and Apple target/device proof |
| Vulkan | T07 | Rust host/IR lowering and Vulkan device evidence |
| OpenCL | T07 | Rust host/IR lowering and supported OpenCL device evidence |
| WebGPU | T07 | Rust host/IR lowering and actual browser/device evidence |
| BLAS | T08 | Rust backend adapter and explicit external vendor-library boundary |
| CANN | T08 | Rust backend and intended accelerator evidence |
| ET | T08 | Rust backend/compiler/firmware contract and runner evidence |
| SYCL | T08 | Rust backend/compiler strategy and target evidence |
| zDNN | T08 | Rust backend and IBM target evidence |
| Hexagon | T08 | Rust host/DSP implementation and FastRPC/HMX/QuRT evidence |
| ZenDNN | T08 | Rust backend plus validated vendor ABI/behavior |
| OpenVINO | T08 | Rust graph/state frontend and target evidence |
| RPC | T08 | Rust wire/client/server implementation over real downstream backend |
| VirtGPU | T08 | Rust guest/host implementation and real shared-buffer synchronization |

Model weights, GPU throughput, end-to-end generation quality, complete browser/client behavior and full media-codec coverage remain separate gates. No hardware or model result is inferred from successful compilation of these host crates.
