# Current port status

Updated 9 November 2026. The user objective remains a complete native Rust port of the frozen ZIP. **The full port is not complete.** This document records actual executable scope and the work still required. No overall completion percentage is assigned.

## Source and product boundary

The baseline ZIP SHA-256 is `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. All 4,082 source paths have an owner, byte count and checksum in `ledger/baseline-files.tsv`. Source verification compares the complete extracted tree, including additions and removals. Ownership coverage is not semantic completion.

The product's maintained implementation is Rust. Frozen C/C++ reference builds, model downloads and oracle generators remain outside the repository. Numeric tables, tokenizer-only GGUF, reference float/byte arrays, media examples and synthetic binary parser inputs are declared fixture data. Minimal Cargo/CI configuration, licenses and documentation retain their actual formats.

The merged workspace has local path dependencies. Optional OS/driver/vendor interfaces remain named external boundaries. They do not import the original llama.cpp implementation. See `release/dependency-inventory.tsv`.

## Track-by-track scope

| Track | Current implementation and evidence | Requirements still open |
| --- | --- | --- |
| T01: ABI/platform/integration | Shared workspace and loader contracts, real system symbol loading, typed errors, and native C ABI work backed by the Rust model/context APIs. | Complete frozen header export/semantic coverage, all platform contracts, exhaustive FFI ownership review and cross-platform ABI execution. |
| T02: tensor engine | Four-dimensional GGML layouts and strides, block-aware views, immutable storage lifetime, scalar DAG execution, cancellation and 65 native graph fixtures. | Full operator set; writable/imported/device buffers; scheduler, allocation planning/reuse, events; autodiff and optimizers; advanced RoPE and softmax modes. |
| T03: CPU/quantization | All 36 active dtype decoders, 12 reference encoders, exhaustive half/BF16/F8 conversions, and nine source-supported matvec weight types with independent scalar/native graph checks. | Remaining packed dots and encoders; weighted quantization; all CPU operators and ISA implementations; dispatch, rounding and performance parity across ISAs. |
| T04: formats/conversion/tokenizers | Bounded GGUF v2/v3 reader, all metadata types, tensor validation, little-endian writer, scalar endian conversion, Llama SPM with 68 original test cases. | Other tokenizer families and pre-tokenizers; complete Unicode behavior; checkpoint/Safetensors and sharded model loading; all converters, metadata tools and quantization applications. Big-endian packed tensor arithmetic beyond implemented scalar types. |
| T05: models/runtime | Typed Llama/Mamba/BERT graphs, state/cache transactions, sequence isolation, adapters and independent small graph checks. Dense Llama GGUF integration executes actual pretrained F32 and Q4 models. | Remaining named architectures among the baseline's 172 entries; full metadata/graph variants and scaled RoPE; memory/cache modes, speculation, batching, model placement and GPU model integration. |
| T06: CUDA/HIP/MUSA | Rust driver interfaces and typed PTX generation; actual CUDA kernels including matvec, stride/broadcast/tail coverage; 138 recorded dispatches on RTX 4090. | Full tensor/backend scheduler integration, upstream operation and dtype coverage, memory/collective modes, optimized kernels; HIP and MUSA hardware execution/parity. |
| T07: portable GPU | Typed kernel IR; generated OpenCL kernels executed on NVIDIA and AMD; Vulkan/other host foundations. | Complete Metal, Vulkan, OpenCL and WebGPU implementation; full compiler/device/OS matrix, scheduler integration and numerical/performance parity. |
| T08: specialty/remote | Checked RPC 8.0.1 subset with transactional scalar graph execution; CBLAS interfaces, DSP ELF/SWIV and virtual-device utilities with explicit capability handling. | Complete BLAS, CANN, ET, SYCL, zDNN, Hexagon, ZenDNN, OpenVINO, RPC and VirtGPU backend semantics and actual required device/vendor testing. |
| T09: multimodal/media | Selected Rust image/audio decode, preprocessing and resampling; independent media fixtures. | All 58 multimodal graph builders; model projections, audio/video model execution and speech generation; full codec/container/device capability coverage. |
| T10: behavior/tools | Sixteen sampler methods, 54 native chat formatters and independent corpora, GBNF subset, native inference CLI. | Remaining sampler/grammar/PEG/Jinja behavior, complete common options, all tools/examples, converter/updater/unified-app functionality. |
| T11: server/UI/clients | Real-model native/OpenAI text completions, SSE, bounded HTTP parsing, auth/CORS, replay, cancellation and stop handling; actual model TCP tests. | Full server routing/scheduling/protocols, tools/MCP, TLS, all browser features/storage migrations/WASM behavior, native clients and connect integration. |
| T12: validation/release | Original ownership/registry ledger, source/fixture audits, independent fixtures, offline Rust tasks and deterministic source packaging; Windows/Linux CI configuration. | Full capability-linked file disposition review; all model/platform/device suites; clean distribution verification, memory/quality/performance budgets and complete release closure. |

The tables and registries in the original plan retain the full scope. A missing feature is not waived because it is difficult or requires unavailable hardware.

## Pretrained inference findings

The first end-to-end model is `stories260K.gguf`, SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`. The independent frozen llama API oracle uses one CPU thread, F32 KV and disabled flash attention. Six prompts produced 96 matching greedy tokens. The 62,976 compared logits had maximum absolute error `1.764297485e-5`, within the predefined bound `2e-4 + 2e-5 * abs(reference)`.

The larger `stories15M-q4_0.gguf`, SHA-256 `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`, initially exposed a real bug: expanding all packed matrices to F32 preserved sixteen greedy tokens but changed logits by as much as `0.4481242895126343`. The corrected runtime retains packed weights, converts activations according to the source CPU trait, and evaluates integer block dots with the original scaling order. All 640,000 compared logits then matched the frozen Windows reference exactly. The report before the fix is retained beside the report after the fix.

Exact equality is recorded evidence for that exercised build and prompt. Portable tests retain the previously declared numerical tolerance because platform math libraries and ISA reduction orders can differ. These tests do not establish arbitrary quantized-format support or all-model quality parity.

The SPM corpus uses the original vocabulary-only fixture rather than a fabricated small vocabulary. It verifies 68 cases, 312 pieces, parsing/insertion flags, byte fallback, Unicode, whitespace and detokenization behavior.

## Memory and execution qualifications

The high-level model facade currently loads a single dense Llama GGUF into host memory. Non-matrix vectors are decoded F32. Matrices retain a decoded F32 row-access cache and, for supported non-F32 types, the original packed bytes needed for native dot arithmetic. The decoded-weight limit is not a total process memory limit. There is no mmap, tensor streaming, GPU model placement or full asynchronous scheduler claim.

Context state uses F32 KV. Model instances are immutable and shared; each generation request owns independent context and sampler state. Stop strings and incomplete UTF-8 suffixes are withheld across token boundaries. Errors and cancellation do not commit partial model state. The C ABI documentation separately states accepted/rejected native parameter configurations.

Generation currently supports greedy or temperature/distribution sampling with an explicit u32 seed. The presence of other sampler implementations in the behavior crate does not mean every upstream generation option is wired into the CLI/server.

## Validation and honest failure reporting

Current validation logs are under `evidence/2026-11-09/`; historical archive logs remain under `docs/imports/`. Test outputs are associated with their command, platform and inputs. A test marked ignored requires an explicit external model, native reference library, installed compiler or vendor service. Ordinary workspace tests do not substitute for those explicit runs.

The source audit checks implementation roles and dependency metadata; it reports unsafe review candidates instead of claiming every unsafe block is proven sound. Fixture integrity checks hashes, declared lengths, provenance fields and complete fixture inventory. Numerical tests check semantics separately.

`release/gates.tsv` is the active mandatory gate list. `cargo xtask release-check` must return nonzero until all full-port requirements have independent closure evidence. A green build or a checkpoint commit cannot close that gate.
