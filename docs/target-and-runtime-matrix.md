# Current target and runtime evidence

Updated 9 November 2026. This matrix replaces the host-only matrix inherited
from the standalone T09–T12 archive. The [import manifest](imports/manifest.json)
retains that archive's identity. Current executable scope is recorded in
[PORT_STATUS.md](PORT_STATUS.md); full requirements remain in the
[original plan](plan.md) and [release gates](../release/gates.tsv).

## Tested configurations

| Surface | Verified scope | Evidence boundary |
| --- | --- | --- |
| Windows x86_64 MSVC, Rust 1.99.1 | Offline workspace debug/release builds and tests, strict Clippy, source/fixture audits, CLI and HTTP execution. | [Expanded runtime verification](../evidence/2026-11-09/expanded-runtime/README.md) records the exact input fingerprint and command results. This does not establish native GUI/client, other Windows architecture or every OS integration contract. |
| Linux x86_64 GNU under Ubuntu/WSL, Rust 1.99.1 | The same workspace gates, explicit pretrained comparisons, CLI and HTTP execution. | The same [verification report](../evidence/2026-11-09/expanded-runtime/README.md) separates Linux results from Windows. Other distributions, libc versions and CPU ISAs require their own checks. |
| CPU model inference on these hosts | Dense Llama generation with SPM/BPE and native packed dots; Mamba-1 recurrence; dense BERT embeddings and pooling. | [Inference contracts](../crates/inference/README.md) identify the exercised pretrained models, fixed error budgets and unsupported metadata. Model families and all graph variants remain separate requirements. |
| Scalar tensor execution on these hosts | 62 operation IDs plus NONE input leaves; 1,252 distinct original-GGML graph cases. | [CPU operation inventory](../crates/tensor/CPU_SUPPORT.md) states dtype/shape restrictions. MSVC/GNU tied-sort differences have separate native references; exact UCRT ERF bit parity remains open. |
| Native C ABI on these hosts | 184 frozen C exports plus four diagnostic extensions; dense Llama context/memory/session and all vocabulary-family identifiers within accepted parameters. | [Export coverage](../crates/ffi/COVERAGE.md), [ABI contract](../crates/ffi/README.md) and [state contract](../crates/ffi/STATE.md) retain all 71 missing exports and semantic limitations. Native recurrent memory and nonnull metadata overrides remain unsupported. |
| CUDA device slice | Selected Rust-generated kernels executed on an RTX 4090. | [T06 evidence](T06-STATUS.md) records 138 dispatches. This is not complete backend scheduling, model offload, all operation/dtype or performance parity. |
| OpenCL device slice | Selected Rust-generated kernels executed on NVIDIA and AMD devices. | [T07 evidence](T07-STATUS.md) records the exercised device/compiler configuration. Other GPU frontends remain partial foundations. |
| Other Linux architectures / musl | Not validated by this checkpoint. | No successful target execution is inferred from the x64 GNU results. |
| Apple/macOS/iOS | Not validated by this checkpoint. | Apple SDK, native client and Metal execution requirements remain open. |
| Android | Not validated by this checkpoint. | Target/SDK, device and application lifecycle requirements remain open. |
| Browser/WASM | Not validated by this checkpoint. | The server's diagnostic HTML page does not supply the browser application, WASM runtime, bindings or storage migrations. |

The linked verification report associates its totals with a specific execution
fingerprint. A later source revision or added regression needs its own recorded
validation; this matrix does not transfer earlier results to changed inputs.
Frozen C/C++ libraries are external comparison tools. They are not linked or
loaded by the Rust product.

## Backend closure

Every backend in the original plan remains in scope. Current device and scalar
results establish bounded slices; no complete backend gate is closed here.

| Backend | Owner | Remaining execution and integration requirements |
| --- | --- | --- |
| CPU | T03 | Complete tensor operators, ISA implementations/dispatch, scheduler and model integration, numerical and performance checks across supported hosts. |
| CUDA | T06 | Full tensor/backend scheduler and model integration, remaining operation/dtype coverage, memory/collective modes and optimized-kernel/device parity. |
| HIP | T06 | Complete host/device implementation, actual AMD execution and full integration/parity. |
| MUSA | T06 | Complete host/device implementation, actual MUSA execution and full integration/parity. |
| Metal | T07 | Complete Rust host/device implementation and Apple target/device evidence. |
| Vulkan | T07 | Complete host/IR lowering, device execution, scheduler integration and parity. |
| OpenCL | T07 | Complete operation/dtype coverage and scheduler/model integration beyond the measured kernels. |
| WebGPU | T07 | Complete host/IR lowering and actual browser/device execution. |
| BLAS | T08 | Complete adapter semantics, supported vendor ABI matrix and independently exercised end-to-end integration. |
| CANN | T08 | Complete Rust backend and intended accelerator execution. |
| ET | T08 | Complete backend/compiler/firmware contract and runner evidence. |
| SYCL | T08 | Complete backend/compiler strategy and target execution. |
| zDNN | T08 | Complete Rust backend and IBM target execution. |
| Hexagon | T08 | Complete host/DSP implementation and FastRPC/HMX/QuRT execution; parser and rewrite fixtures do not establish DSP execution. |
| ZenDNN | T08 | Complete Rust backend, validated vendor ABI/behavior and target execution. |
| OpenVINO | T08 | Complete graph/state frontend and target execution. |
| RPC | T08 | Complete wire/client/server semantics and integration beyond the checked RPC 8.0.1 scalar graph subset. |
| VirtGPU | T08 | Complete guest/host implementation and real shared-buffer synchronization. |

Model quality and performance across the full corpus, complete browser/native
clients and all media/model paths remain explicit release requirements.
Hardware discovery or successful compilation does not establish device execution.
