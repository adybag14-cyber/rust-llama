# T06 — CUDA, HIP and MUSA progress checkpoint

**Baseline:** uploaded archive SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
**Host:** Linux x86-64, supplied Rust 1.99.1, LLVM 23.1.1.
**Date:** 8 November 2026.
**Status:** executable Rust host and kernel-generation subset; no T06 hardware execution in this environment; full T06 exit gates remain open.

## What is implemented

| Area | Rust implementation | Proof in this checkpoint |
| --- | --- | --- |
| Discovery | Independent CUDA, HIP and MUSA dynamic-library loads, initialization, device count/name/memory/version queries; CUDA warp and compute-capability queries | Real local calls attempted; each family reports its missing driver independently |
| CUDA ownership | Primary-context retain/release with scoped context-stack restoration; device and pinned allocations; initialized memory; bounded host/device copies | Compiles; range/size validation tests; real-driver smoke executable is available but could not run |
| CUDA execution resources | Streams, events, waits, PTX module/function ownership, synchronous kernel submission | Compiles; holds libraries, contexts, modules, buffers and argument storage through completion; hardware validation pending |
| HIP host resources | Direct HIP allocation/free, copies, stream/event ownership and device restoration | Compiles; actual HIP library unavailable; no HIP device execution claimed |
| MUSA host resources | Independently loaded MUSA runtime allocation/free, copies, stream/event ownership and device restoration | Compiles; actual MUSA driver/runtime unavailable; no MUSA device execution claimed |
| CUDA code generation | Typed Rust instruction tree emits PTX for strided F32 add, scale, scalar sum and noncontiguous Q4_0 × F32 matvec | Type/shape validation plus instruction interpretation passes; PTX assembly/JIT acceptance is not yet established |
| Numerical verification | Interpreter executes the same instruction tree that emits PTX; an external frozen C oracle supplies independent Q4_0 values | 256 basis-vector values agree with original dequantization; additional independent dense dot/stride/tail tests pass |
| Scheduling contracts | Integer weighted/tiled row partitions, byte-budget checks, directional peer-copy routing and explicit host staging | Boundary/property-grid tests; no actual multi-device transfer has run |
| Graph/cache contracts | Device, topology, tensor shape, buffer generation, adapter and code-generation invalidation; structural cache keys | Replay-decision tests; actual vendor graph capture/replay is pending |

The four generated operations use scalar work per output lane. Sum runs in one lane; matvec uses one lane per row. These establish a first correctness path. They are not optimized reductions, GEMM, tensor-core or throughput parity.

The local PTX representation is a T06-owned provisional extension. It must be reconciled with T07's shared kernel representation and T02's backend/submission contracts. This crate is not registered as a completed GGML backend. T01–T04 production contracts were not supplied to this work order.

## Actual verification

| Command or experiment | Actual result |
| --- | --- |
| `cargo test --offline -p llama-device-family` | Exit 0; 19 unit tests passed; none ignored |
| `cargo test --offline --release -p llama-device-family` | Exit 0; the same 19 unit tests passed |
| `cargo clippy --offline -p llama-device-family --all-targets -- -D warnings` | Exit 0 |
| `cargo fmt -p llama-device-family --check` | Exit 0 |
| Optional Rust oracle recorder | Compiled the original quantization translation unit in an isolated directory; captured 8 Q4_0 blocks and 256 F32 values |
| Real CUDA/HIP/MUSA discovery | All three explicitly unavailable; exact loader messages are in `T06-PROBE.txt` |
| `cuda_smoke` | Exit 77: **execution not run**, because CUDA driver library is absent |
| Direct Rust NVPTX compile | Exit 1, E0463: `core` is not installed for `nvptx64-nvidia-cuda` |
| Direct Rust AMDGPU compile | Exit 1, E0463: `core` is not installed for `amdgcn-amd-amdhsa` |

The 19 tests cover generated instruction semantics, out-of-bounds prevention, bad shape/stride rejection, output alias rejection, instruction fuel limits, launch tails, FP16 special/subnormal values, Q4_0 nibble ordering, source-oracle values, integer partition overflow, directional peers, warp/wave mismatch and graph invalidation. Debug and release are two runs of the same 19 tests, not 38 distinct tests.

Reference Q4_0 comparisons use ordinary F32 numerical equality on finite fixture values. Matvec accumulation can change the sign bit of zero, so the result is not a claim of identical signed-zero bit patterns. No GPU numerical result was inferred from interpretation.

### Device availability

| Family | Actual observation | Consequence |
| --- | --- | --- |
| CUDA | Neither `libcuda.so.1` nor `libcuda.so` can be opened | No enumeration, allocation, driver JIT, kernel execution or CUDA performance result |
| HIP | `libamdhip64.so`, `.so.7` and `.so.6` cannot be opened | No AMD enumeration, allocation, code-object load or execution result |
| MUSA | Neither `libmusa.so.1` nor `libmusa.so` can be opened | No MUSA enumeration, allocation, compiler or execution result |

The supplied compiler lists NVPTX and AMDGPU targets, but its installed target libraries do not contain their `core` crates. `T06-DEVICE-COMPILER-PROBES.txt` contains the exact compiler commands and diagnostics. No MUSA target appeared in that compiler's target list. This does not establish that another Rust device toolchain cannot support MUSA; that route remains unproved.

## Running the work

From the workspace root, with the supplied compiler's `bin` directory on `PATH`:

```bash
cargo test --offline -p llama-device-family
cargo clippy --offline -p llama-device-family --all-targets -- -D warnings
cargo run --offline -p llama-device-family --example device_probe
cargo run --offline -p llama-device-family --example cuda_smoke
```

`device_probe` is read-only. `cuda_smoke` allocates small temporary resources on CUDA device 0 and exercises the four generated kernels when a real driver/device is available. Exit 77 means unavailable, exit 1 means a failed check, and exit 0 means those limited checks passed. It does not test full-model inference.

`RuntimeDriver::load(Family::Hip)` and `RuntimeDriver::load(Family::Musa)` expose `device(ordinal)?.memory_roundtrip(bytes)`. These are real allocation/copy/event checks. They are not kernel execution and cannot establish either backend's compute support.

## Frozen source coverage

`T06-SOURCE-COVERAGE.tsv` inventories **295 source paths**, with each path's byte size and SHA-256:

| Original directory | Files |
| --- | ---: |
| `ggml/src/ggml-cuda` | 290 |
| `ggml/src/ggml-hip` | 1 |
| `ggml/src/ggml-musa` | 3 |
| `ggml/include/ggml-cuda.h` | 1 |

The small HIP and MUSA directory counts reflect their reuse of CUDA source through vendor mappings. They do not make those backends small or complete. See their original `CMakeLists.txt` files and `ggml-cuda/vendors/hip.h` / `musa.h`.

`T06-OP-CASE-INVENTORY.tsv` records **90 distinct `GGML_OP_*` case labels** observed in `ggml-cuda.cu`. This is an inventory, not an exact source-supported operation matrix: conditionals, type/shape/layout restrictions and dispatch conditions still require semantic extraction. Each row is explicit about the narrow Rust subset or pending work. No original source file is marked completely rewritten.

## Remaining work

| Required capability | State |
| --- | --- |
| Complete operation/type/layout dispatch and T02 backend registration | Pending |
| Remaining quantization formats, GEMV/GEMM, attention, recurrent, MoE and training kernels | Pending |
| CUDA tensor-core, subgroup, cooperative and architecture-specific fast paths | Pending |
| HIP Rust device builds, wave32/wave64 kernels, MFMA/WMMA and code-object loading | Pending; no compatible device binary generated |
| MUSA proven compiler/emitter route and device modules | Pending; CUDA PTX is not passed off as a MUSA implementation |
| Actual graph capture, graph replay/update and cancellation | Pending; only invalidation decisions implemented |
| Virtual memory, imported allocations and asynchronous submission ownership | Pending |
| Actual peer copies, multi-GPU computation, collectives/NCCL/RCCL/MCCL | Pending; row/copy planning only |
| Vendor BLAS/math integration and feature audit | Pending; none silently linked |
| Windows portability and old/future SM targets | Untested; no cross-target standard library supplied |
| Rust backend full-model logits/state parity and performance | Pending |
| Driver JIT acceptance of generated PTX | Pending because no CUDA driver/assembler is available |

## Safety and dependency boundary

Normal builds use only Rust standard libraries and the workspace's Rust dynamic loader. No `build.rs`, native compiler, old ggml binary or CUDA source is part of the normal dependency graph. CUDA/HIP/MUSA drivers are separately installed external vendor dependencies and are loaded only on request.

`unsafe` is confined to explicit external ABI calls, opaque handle operations and the CUDA pinned-host slice boundary. Handles retain their libraries and contexts. The types deliberately do not implement `Send` or `Sync`; there are no blanket unsafe thread-safety assertions. Generated execution is blocking and keeps all memory alive through synchronization. The arbitrary raw CUDA launch function is `unsafe` and documents its memory/ABI requirements.

Destructors make best-effort releases; they cannot return driver shutdown errors. Safe operations surface synchronous errors. Hardware loss and asynchronous driver errors still need real-device fault testing. The provisional resource layer has not been subjected to a hardware lifecycle stress test.

Peer review found and corrected a pageable-upload ordering hazard: CUDA's host-to-device call may return after staging while DMA is still pending, and a nonblocking stream does not establish an automatic dependency on that transfer. Every blocking `copy_from` now explicitly synchronizes its CUDA context or HIP/MUSA device before returning. This deliberately trades concurrency for a clear ownership/order contract at this checkpoint. A future asynchronous implementation must retain transfer memory and event dependencies.

The arbitrary CUDA module-image loader is `unsafe`: its vendor API accepts a pointer without a byte length and can sniff binary image formats. The safe `compile_generated` entry accepts only immutable typed-generator artifacts; it cannot be supplied a truncated binary image disguised as a C string.

The independent oracle recorder is an optional **test/reference tool**. It invokes a native compiler on the separately supplied frozen reference source; the compiled oracle lives outside the Rust product and is not bundled or linked into production. Only its small binary data fixtures are retained. See `T06-Q4-FIXTURES.md`.

## ABI and compiler references

The frozen source remains the feature denominator. These official documents were checked for external ABI signatures and PTX target versions:

- [NVIDIA CUDA Driver API: device management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html)
- [NVIDIA CUDA Driver API: execution control](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)
- [NVIDIA CUDA Driver API: synchronization behavior](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-driver-api/api-sync-behavior.html)
- [NVIDIA CUDA driver data types and attribute values](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html)
- [NVIDIA PTX ISA 7.8: target introduction/version table](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)
- [AMD HIP initialization and device information](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___driver.html)
- [AMD HIP runtime API declarations](https://rocm.docs.amd.com/projects/HIP/en/develop/doxygen/html/hip__runtime__api_8h.html)
- [AMD HIP memory management](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/doxygen/html/group___memory.html)
- [AMD HIP module-management contract](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/doxygen/html/group___module.html)
- [Moore Threads MUSA driver API 5.2.0](https://docs.mthreads.com/musa-sdk/version-5.2.0/libraries/core_api/driver_api_reference/)
- [Moore Threads MUSA runtime API 5.1.0](https://docs.mthreads.com/musa-sdk/version-5.1.0/libraries/core_api/runtime_api_reference/)

Accessed 8 November 2026. Availability of a documented API is separate from successfully calling it on this machine.
