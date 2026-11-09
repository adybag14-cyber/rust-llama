# Integrated T05–T08 progress

**Checkpoint: 8 November 2026. Implementation commit: `e5a6e57b38d0ba05bb174fd82e8e760467b30cd9`.**

The four requested tracks are integrated in one offline-buildable Rust workspace. All **115 distinct tests passed in debug and release**, including all seven tests that require an external reference library, shader compiler or CBLAS provider. Formatting, warning-denied Clippy, release compilation and the integrated execution demo also passed. Exact commands, outputs and hashes are retained in `evidence/` and explained in `VERIFICATION.md`.

This is a substantial implementation checkpoint. **T05, T06, T07 and T08 remain incomplete under the supplied plan; no complete track exit gate is closed.** In particular, there is no claim of full llama.cpp replacement, pretrained-model inference, full public ABI compatibility or successful accelerator execution. The source mapping keeps unfinished capabilities visible.

## Workspace and integration

| Package | Responsibility |
| --- | --- |
| `llama-model-runtime` | Model slices, exact architecture inventory, batch/context/state/adapters |
| `llama-device-family` | CUDA/HIP/MUSA ownership and probes; CUDA PTX semantics and validation |
| `llama-kernel-ir` | Shared checked kernel semantics, scalar validation and four source emitters |
| `llama-portable-gpu` | Portable platform boundaries, OpenCL execution path, Vulkan discovery and shaderc integration |
| `llama-vendor-remote` | RPC, external BLAS, Hexagon/ET utilities and vendor feasibility boundaries |
| `llama-platform-loader` | Owned dynamic libraries and explicit unsafe ABI lookup boundary |
| `track-runner` | Integrated demo/probes, artifact generation, inventory/provenance audit and deterministic ZIP packaging |

All Cargo dependencies are local workspace packages. Default builds need the supplied Rust compiler and standard library, with no crates.io download, native project build or original GGML linkage. Optional vendor calls use explicitly identified external libraries. Optional reference examples deliberately compile/load the original implementation outside the product so the differential checks can be repeated.

The supplied T01–T04 implementations were not available. `docs/INTEGRATION-CONTRACT.md` therefore marks the small local tensor/context interfaces as provisional. Scalar model and IR execution provide validation while the authoritative tensor engine, scheduler, CPU kernels, GGUF ingestion and tokenizer interfaces remain to be integrated.

## T05 — models and runtime

Three owned F32 execution slices are implemented:

- **Llama:** dense layers with grouped-query attention, ordinary full-head RoPE, RMSNorm, SwiGLU, causal KV state, optional projection biases and tied/separate output weights.
- **Mamba-1:** causal depthwise convolution, selective state-space recurrence, softplus time steps, gating, optional dt/B/C normalization and recurrent state.
- **BERT:** bidirectional attention, embedding/post-layer normalization, the frozen implementation's half-rounded GELU behavior, token embeddings and mean/first/last pooling.

The context supports interleaved sequences, microbatches, selected outputs, external embeddings, whole-batch transactional publication, per-layer cancellation, sequence forks/removal, attention-cache truncation and checkpoint restoration of recurrent state. LoRA and control-vector changes are validated and invalidate prior state explicitly. Bounded versioned state files include model/adapter fingerprints and CRC32; this local format is not the original state-file ABI.

The registry preserves **173 enum slots, including UNKNOWN**, and records **172 named architectures, 176 source files and 171 original factory cases**. Those inventory entries do not imply executable support. The source audit found **312 text and 92 projector conversion registrations**, versus the plan's 314/91; both findings are recorded rather than silently changing the denominator.

The three model slices passed **35 tests**. An actual separately built frozen GGML library provided **104 values in 20 numerical vectors**, with a largest absolute error of **`1.788139343e-7`**. The shipped Rust reference-build helper reproduced the saved reference vectors. These checks use small synthetic weights and corresponding native operator graphs; they do not certify trained-model loader/context equivalence.

Remaining work includes the other architecture graphs and family variants, quantized weights, specialized cache layouts and attention, MoE/expert placement, public ABI integration, original state interoperability, model-file ingestion and real pretrained-model acceptance.

## T06 — CUDA, HIP and MUSA

Independent vendor loaders and discovery paths are present. Owned host APIs cover contexts/device guards, device and pinned memory, copies, streams and events. CUDA also has owned modules/functions and a synchronous generated-kernel launch path. HIP/MUSA have separate runtime ownership and transfer paths; they are not reported as CUDA substitutes.

A typed Rust PTX representation generates strided F32 add, scale, sum and Q4_0 matvec. The interpreter executes the same representation, with independent reference data for quantization. Weighted aligned device splits, memory budgets, directional peer-routing decisions and graph invalidation have explicit contracts and tests. **19 tests passed**, including all 256 values of the native-derived Q4 fixture and boundary/error cases.

The checkpoint includes four generated PTX files with provenance hashes. **No PTX assembler/JIT pass or GPU execution was possible:** this environment has no usable CUDA, HIP or MUSA driver. Direct Rust GPU-target probes also recorded missing target standard-library support. HIP/MUSA program generation, broad dtype/operator coverage, real multi-device scheduling and hardware performance/failure validation remain open.

## T07 — portable GPUs and shared semantics

IR version 1 represents scalar types, typed buffers, checked arithmetic and access, loops/conditions, binary16 conversion, packed-byte addressing and read/write permissions. Four Rust emitters generate **OpenCL C, Vulkan GLSL, WGSL and MSL** without a raw-source escape hatch.

The nine checked built-ins are add, multiply, SiLU, scale, row sum, RMSNorm, finite-row softmax, matrix multiplication and Q4_0 matvec. They exercise strides, padding and partial workgroups. Generic IR can be interpreted transactionally with bounds/race/fuel checks. Only sealed, checked built-in plans can reach the safe OpenCL launch API.

The host implementation includes real OpenCL resource ownership, compile/dispatch/readback and real Vulkan loader/instance/device discovery. All **nine GLSL programs compiled to SPIR-V** through external shaderc. The independent native quantization comparison checked **all 65,536 binary16 encodings** (NaNs by class) and an unaligned, padded and strided Q4_0 matvec. T06/T07 shared fixtures also check overlapping contracts.

There are **36 generated portable shader sources and nine SPIR-V binaries**, plus the four T06 PTX sources: **49 generated artifacts** with a SHA-256 manifest. Compilation evidence is limited to Vulkan GLSL. OpenCL has no installed platform, Vulkan has no compatible ICD, and Metal/WebGPU host adapters are pending. Device execution, broader kernel/dtype coverage, optimized/fused kernels, synchronization primitives, full backend graphs and performance evidence remain open.

## T08 — vendor and remote paths

The RPC implementation explicitly encodes **protocol 8.0.1**, all **24 command IDs**, the **296-byte packed tensor** and HELLO/layout fields. The TCP client/server implements 20 handlers and a bounded, non-aliased contiguous F32 graph slice. Allocations, clear/set/get/copy/memset, 2D transfers, graph replay and cleanup run in the integrated demo. Graph execution is transactional; unsupported operations are errors.

The original native CPU server accepted the Rust client through transfers, actual F32 matrix multiplication and replay. The original native client also completed its allocation/transfer/free roundtrip with the Rust server. The shipped Rust build recipe and FFI driver rebuilt the native comparison library from **124 pinned input hashes** and reran all three unchanged native integration tests. Native sources and binaries remain external to the product.

The external LP64 CBLAS adapter passed real matrix/stride and reduction calls. Hexagon ELF32/ELF64 SWIV handling matched both original Python-derived byte fixtures. The ET forbidden-instruction text guard and VirtGPU APIR handshake/control codecs are implemented. Ten-family probes report availability and open ABI/hardware gates honestly. **39 T08 tests passed** when the five external tests were enabled.

Full RPC operations/dtypes, persistent remote caching, asynchronous scheduling, RDMA/collectives, VirtGPU transport and the actual CANN/SYCL/zDNN/OpenVINO/Hexagon/ET/ZenDNN device implementations remain open. A loader or symbol being present is not backend execution evidence.

## Review corrections included

Review addressed malformed model dimensions, noncanonical LoRA targets, BERT mean-pooling overflow, asynchronous transfer ordering and unsafe raw-module loading. IR review fixed fuel accounting, repeated subtree validation, loop-bound semantics, unsafe generic launch exposure, source-hash validation and transactional buffer publication. Native RPC comparison exposed the original COMPUTE flag requirement; the Rust path now matches it and has a regression check.

These are concrete correctness improvements in the final tested implementation. The remaining host driver paths still require real-device lifecycle, fault, teardown and stress validation.

## Coverage accounting and continuation

The hashed responsibility ledger contains **1,685 assigned source paths**: T05 **218**, T06 **295**, T07 **601**, T08 **571**, matching the plan's ownership rules. Every capability-completion row stays open. File counts and the Rust source line count are inventory measures, not completion percentages.

Recommended continuation order:

1. Integrate the authoritative T01/T02 contracts and T03/T04 execution/ingestion paths, preserving all existing differential fixtures.
2. Run the implemented CUDA/OpenCL launch paths on real supported devices; add compiler/device validation for HIP/MUSA, Metal and WebGPU.
3. Unify T06/T07 kernel lowering and expand dtype/operator coverage against the frozen source ledger.
4. Add architecture families and trained-model tests, then expand RPC/vendor operations with corresponding native and hardware evidence.
5. Close each plan gate only after its full behavior, compatibility, resource and performance requirements are verified.

The delivery contains source, manifests, tests, generated artifacts, reports, raw validation logs and a Linux x86-64 release runner. It excludes the original source archive, the supplied Rust distribution, native comparison binaries and Cargo build caches. Keep the original inputs separately to repeat the optional native comparisons.
