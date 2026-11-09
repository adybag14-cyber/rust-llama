# llama.cpp Rust rewrite — T05–T08 progress checkpoint

**8 November 2026 · Rust 1.99.1 · Linux x86-64 verified**

This archive contains the latest integrated implementation for the four requested tracks, following the supplied twelve-track plan. It is a working, independently buildable Rust progress workspace. **It is not a completed llama.cpp replacement, and no complete T05–T08 exit gate is claimed closed.**

The original plan is retained at `docs/original-12-track-plan.md`. The frozen input identity is recorded in `evidence/input-manifest.json`. All four source ownership counts match the plan: **T05 218, T06 295, T07 601, T08 571**, for **1,685 source paths**. The per-path ledger records responsibility and outstanding work; these counts are not a conversion percentage.

## What runs now

| Track | Implemented checkpoint | Evidence and practical limits |
| --- | --- | --- |
| **T05 — models/runtime** | Owned dense F32 Llama, Mamba-1 and BERT slices; exact architecture registry; checked model weights; multi-sequence batches; transactional KV/recurrent state; cancellation and checkpoint rollback; LoRA/control vectors; bounded state encoding | Numerical comparisons use actual frozen GGML operator graphs and synthetic tiny weights. No trained-model/GGUF/tokenizer or full architecture-family acceptance is claimed. |
| **T06 — CUDA/HIP/MUSA** | Independent vendor loaders/probes; owned contexts, memory, copies, streams/events; CUDA typed PTX for four operations; numerical instruction interpreter; multi-device split and graph-invalidation contracts | Rust code builds and semantic/reference tests run. GPU libraries/devices are unavailable here; no GPU execution or PTX assembler/JIT pass is claimed. HIP/MUSA device program generation remains open. |
| **T07 — portable GPUs** | Versioned typed IR; nine scalar kernels; OpenCL C, Vulkan GLSL, WGSL and MSL generation; sealed safe launch plans; real OpenCL host path; Vulkan enumeration; external shaderc integration; provenance/cache keys | Nine GLSL programs compile to SPIR-V. OpenCL reports no platform and Vulkan no compatible ICD. WGSL/MSL compiler/device validation and most host/backend coverage remain open. |
| **T08 — vendor/remote** | RPC 8.0.1 TCP client/server and bounded F32 graph slice; real external CBLAS adapter; Hexagon ELF/SWIV tooling; ET forbidden-instruction guard; VirtGPU APIR control codecs; ten-family feasibility reporting | Real TCP interoperability with the frozen C++ implementation in both directions, real CBLAS calls, and exact native/Python-derived fixture comparisons. Full remote, vendor and DSP backend coverage remains open. |

The demo reports Llama/Mamba logits, BERT pooled embeddings, matching state restoration, an RPC sequence with **zero retained bytes**, and the CBLAS/IR matrix result `[58, 64, 139, 154]`. These are small, source-derived execution checks, not an application benchmark.

## Build and run

Use the supplied Rust 1.99.1 distribution, or the same compiler installed locally. Put its `bin` directory on `PATH`, then run from this directory:

```text
cargo build --workspace --release --offline --locked
cargo test --workspace --offline --locked
cargo run --offline --locked -p track-runner -- smoke
cargo run --offline --locked -p track-runner -- probe
```

There are no crates.io dependencies. Normal builds do not invoke a native compiler, Python or Node, and do not link the original llama/ggml implementation. Some tests/examples deliberately exercise separately supplied reference code or external vendor libraries; they are identified in `VERIFICATION.md` and the track reports.

The delivery also includes `bin/track-runner-linux-x86_64`, built from this source with the supplied compiler. It targets Linux x86-64, not Windows. Its hash and dependencies are recorded with the build evidence. The source's other platform branches have not been cross-compiled or executed in this checkpoint.

```text
./bin/track-runner-linux-x86_64 smoke
./bin/track-runner-linux-x86_64 probe
./bin/track-runner-linux-x86_64 help
```

## Files to read first

- **`PROGRESS.md`** — integrated result, boundaries and continuation priorities.
- **`VERIFICATION.md`** — exact verified commands, test totals, reference comparisons and hardware blockers.
- **`docs/T05_STATUS.md`**, **`docs/T06-STATUS.md`**, **`docs/T07-STATUS.md`**, **`docs/T08-PROGRESS.md`** — domain APIs, source mapping, evidence and unfinished work.
- **`docs/INTEGRATION-CONTRACT.md`** — provisional boundaries and how this work fits T01–T04.
- **`docs/reference-path-ledger.tsv`** — all 1,685 assigned source paths with sizes/hashes and open capability status.
- **`generated/artifact-manifest.tsv`** — hashes of 36 portable shader sources, four typed-IR PTX outputs and nine compiled SPIR-V binaries. PTX assembler/device acceptance remains unverified.
- **`docs/fixture-manifest.tsv`** — exact binary data allowlist and independent reference provenance.
- **`evidence/`** — input/toolchain identity and final integrated command outputs.

## Kernel and packaging commands

```text
cargo run --offline --locked -p track-runner -- generate-kernels generated --compile-vulkan
cargo run --offline --locked -p track-runner -- opencl-smoke
cargo run --offline --locked -p track-runner -- audit-reference /path/to/llama.cpp-master docs
cargo run --offline --locked -p track-runner -- audit-product .
cargo run --offline --locked -p track-runner -- package . /outside/this/tree/new-progress.zip
```

`--compile-vulkan` explicitly requires the external shaderc compiler. `opencl-smoke` requires an actual OpenCL platform and executes generated kernels there; absence is an error, never a CPU fallback reported as GPU success. The package command creates a deterministic ZIP, rejects unsafe source paths, omits `.git` and Cargo caches, and refuses to overwrite an existing destination. Generated shaders are outputs of Rust semantics, not maintained foreign-language implementations.

## Integration with the other eight tracks

Only the plan, original source archive and toolchain were supplied for this task. T01–T04 implementation artifacts were not available, so this workspace supplies narrow provisional internal APIs and small scalar validation paths. It does not pretend that a production tensor engine, GGUF/tokenizer pipeline or public llama C ABI has been integrated.

The next integration should adapt these domain APIs to the authoritative T01/T02 contracts, replace local scalar validation execution with T02/T03 scheduling and kernels, bind T04 weight/tokenizer ingestion, and retain the current fixtures as differential checks. T06's typed PTX extension and T07's shared IR have overlapping cross-track tests; a unified lowering interface remains to be implemented.

All remaining architectures, variants, numeric formats, backends, kernels, device APIs and runtime behavior in the original plan remain in scope. Explicit unsupported/error results and unclosed rows are intentional progress accounting, not exemptions.

## License and external boundaries

The original MIT license is retained in `LICENSE`. Reference-derived algorithms, registries and fixtures retain that provenance. No trained model weights are included. External GPU/vendor APIs and math remain external; `libblas.so.3` is not claimed as rewritten Rust. Optional comparison tools build/read the frozen native reference outside this source tree, and are excluded from normal product linkage.
