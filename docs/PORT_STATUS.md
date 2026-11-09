# Current port status

Updated 9 November 2026. The user objective remains a complete native Rust port of the frozen ZIP. **The full port is not complete.** This document records actual executable scope and the work still required. No overall completion percentage is assigned.

## Source and product boundary

The baseline ZIP SHA-256 is `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. All 4,082 source paths have an owner, byte count and checksum in `ledger/baseline-files.tsv`. Source verification compares the complete extracted tree, including additions and removals. Ownership coverage is not semantic completion.

The product's maintained implementation is Rust. Frozen C/C++ reference builds, model downloads and oracle generators remain outside the repository. Numeric tables, tokenizer-only GGUF, reference float/byte arrays, media examples and synthetic binary parser inputs are declared fixture data. Minimal Cargo/CI configuration, licenses and documentation retain their actual formats.

The merged workspace has local path dependencies. Optional OS/driver/vendor interfaces remain named external boundaries. They do not import the original llama.cpp implementation. See `release/dependency-inventory.tsv`.

## Track-by-track scope

| Track | Current implementation and evidence | Requirements still open |
| --- | --- | --- |
| T01: ABI/platform/integration | Shared workspace/loader contracts, real system symbol loading and 155/255 frozen C exports with independent header, ownership, raw-tokenizer and real-context sampler checks. | The remaining 100 exports, full semantics of every implemented export, all platform contracts and cross-platform ABI execution. |
| T02: tensor engine | Four-dimensional GGML layouts and strides, block-aware views, immutable storage lifetime, scalar DAG execution, cancellation and 65 native graph fixtures. | Full operator set; writable/imported/device buffers; scheduler, allocation planning/reuse, events; autodiff and optimizers; advanced RoPE and softmax modes. |
| T03: CPU/quantization | All 36 active dtype decoders; all 29 source model quantizers with required/optional weights; 32 non-integer codec formats including activation/F64 helpers; three checked integer setters; all 29 scalar CPU weight-dot types. | General CPU operators and all ISA implementations; dispatch, numerical and performance parity across ISAs. Source-undefined codec paths have documented safe behavior. |
| T04: formats/conversion/tokenizers | Bounded GGUF v2/v3 reader, tensor validation, little-endian writer, scalar endian conversion, SPM, WordPiece, BPE with all 64 metadata-selected profiles / 99 aliases, native raw-byte behavior and exhaustive Unicode checks. | Unigram, RWKV and PLaMo families; checkpoint/Safetensors and sharded loading; all converters, metadata tools and quantization applications. Big-endian packed arithmetic and remaining platform-specific behavior. |
| T05: models/runtime | Typed Llama/Mamba/BERT graphs, state/cache transactions, sequence isolation, adapters and independent small graph checks. Dense Llama GGUF executes pretrained F32/Q4 models; dense BERT GGUF executes pretrained F16 embeddings with all supported pooling modes. | Remaining named architectures among the baseline's 172 entries; full metadata/graph variants and scaled RoPE; memory/cache modes, speculation, batching, model placement and GPU model integration. |
| T06: CUDA/HIP/MUSA | Rust driver interfaces and typed PTX generation; actual CUDA kernels including matvec, stride/broadcast/tail coverage; 138 recorded dispatches on RTX 4090. | Full tensor/backend scheduler integration, upstream operation and dtype coverage, memory/collective modes, optimized kernels; HIP and MUSA hardware execution/parity. |
| T07: portable GPU | Typed kernel IR; generated OpenCL kernels executed on NVIDIA and AMD; Vulkan/other host foundations. | Complete Metal, Vulkan, OpenCL and WebGPU implementation; full compiler/device/OS matrix, scheduler integration and numerical/performance parity. |
| T08: specialty/remote | Checked RPC 8.0.1 subset with transactional scalar graph execution; CBLAS interfaces, DSP ELF/SWIV and virtual-device utilities with explicit capability handling. | Complete BLAS, CANN, ET, SYCL, zDNN, Hexagon, ZenDNN, OpenVINO, RPC and VirtGPU backend semantics and actual required device/vendor testing. |
| T09: multimodal/media | Selected Rust image/audio decode, preprocessing and resampling; independent media fixtures. | All 58 multimodal graph builders; model projections, audio/video model execution and speech generation; full codec/container/device capability coverage. |
| T10: behavior/tools | Safe scalar sampler APIs and a separate native C sampler path; DRY, adaptive-p, infill and lazy/raw-byte GBNF with independent DLL masks/state/context generation; 54 native chat formatters; inference CLI. | Complete PEG/Jinja/custom-regex and remaining sampler semantics, complete common options, all tools/examples, converter/updater/unified-app functionality. |
| T11: server/UI/clients | Real-model native/OpenAI completions and SSE; BERT embedding endpoints with all pooling modes, normalization, base64, typed/batched input and usage; bounded HTTP, auth/CORS, replay, cancellation and deadline checks. | Full routing/scheduling/protocols, multimodal and decoder embeddings, reranking, tools/MCP, TLS, browser features/storage migrations/WASM, native clients and connect integration. |
| T12: validation/release | Original ownership/registry ledger, source/fixture audits, independent fixtures, offline Rust tasks and deterministic source packaging; Windows/Linux CI configuration. | Full capability-linked file disposition review; all model/platform/device suites; clean distribution verification, memory/quality/performance budgets and complete release closure. |

The tables and registries in the original plan retain the full scope. A missing feature is not waived because it is difficult or requires unavailable hardware.

## Pretrained inference findings

The first end-to-end model is `stories260K.gguf`, SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`. The independent frozen llama API oracle uses one CPU thread, F32 KV and disabled flash attention. Six prompts produced 96 matching greedy tokens. The initial 62,976-logit comparison had maximum absolute error `1.764297485e-5`. Matching the source's F32 products with F64 accumulation now gives exact equality on Windows and maximum absolute error `5.722045898e-6` on Linux, within the unchanged predefined bound `2e-4 + 2e-5 * abs(reference)`.

The larger `stories15M-q4_0.gguf`, SHA-256 `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`, initially exposed a real bug: expanding all packed matrices to F32 preserved sixteen greedy tokens but changed logits by as much as `0.4481242895126343`. The corrected runtime retains packed weights, converts activations according to the source CPU trait, and evaluates integer block dots with the original scaling order. All 640,000 compared logits then matched the frozen Windows reference exactly. The report before the fix is retained beside the report after the fix.

The F16 BGE-small BERT model, SHA-256 `f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999`, exposed a second real rounding issue. Input embeddings, normalization and Q/K/V projections were exact, but F32 attention accumulation differed from the scalar source's F64 sum. Those small differences crossed GELU table bins in later layers. Correcting attention and mean-pooling order yields exact Windows equality for all 13,056 compared values and all 122 captured diagnostic stages. Linux maximum error is `1.348853111e-4`, within the original fixed tolerance. A 12-layer independent attention fixture exercises 46,080 output values in ordinary tests. The before/after diagnosis remains recorded.

The equivalent big-endian BERT download is separately hash-pinned and produces the same expected results through the Rust reader. The frozen C reader rejects that opposite-endian file; this is reader validation against the equivalent little-endian native corpus, not a claim of original-reader support.

Exact equality is recorded evidence for that exercised build and prompt. Portable tests retain the previously declared numerical tolerance because platform math libraries and ISA reduction orders can differ. These tests do not establish arbitrary quantized-format support or all-model quality parity.

The SPM corpus uses the original vocabulary-only fixture and verifies 68 cases, 312 pieces, parsing/insertion flags, byte fallback, whitespace and detokenization. WordPiece adds 1,728 native cases across nine metadata variants, all 30,522 original vocabulary entries and 72 special-token fallback cases. Unicode checks cover all 1,114,112 codepoints, 15,957 normalization sequences and 65,809 UTF-8 inputs. The BPE extension now covers every reachable metadata profile: 64 pretokenizer types and 99 aliases. Three other enum names have no metadata dispatch in the frozen source; they are not invented as extra supported profiles. Independent Windows/Linux comparisons cover all 14 original BPE vocabularies, complete property/piece/EOG records, 22,518 raw-byte cases and 126,896 regex splits. Ambiguous inferred EOT/FIM IDs preserve the measured MSVC/libstdc++ container ordering. Full-vector/input tests require explicit original-vocabulary prerequisites; compact synthetic native corpora run ordinarily. Remaining tokenizer families stay open.

## Scalar codecs, native behavior and HTTP extensions

The scalar codec slice now covers every defined source model quantizer, including
importance-weighted legacy/K/IQ fitting. Three model formats require caller-supplied
weights. Q8_1/Q8_K remain activation codecs and F64 remains a conversion helper;
they are not counted as extra source model quantizers. The new independent corpus
has 3,362 original-C encoded rows and 640 tied-sort cases. The original IQ4_NL
uninitialized near-zero payload and exceptional IQ1 out-of-bounds fallback are
handled explicitly and excluded from undefined-source byte parity claims.

The raw-byte GBNF engine remains separate from the safe Unicode recognizer. Its
native corpus checks 383,053 candidate masks and source token terminals/partial
byte behavior. The C sampler extension adds strict/lazy grammar, DRY, adaptive-p
and infill, with 165 independent scenarios and 42 real contexts / 585 generated
tokens. Candidate order, flags, float bits, selected indices and lifecycle
operations match the frozen Windows DLL. Native BPE/FIM getter, piece and
detokenization checks add 1,505,576 vocabulary records and 7,598 tokenization cases.
These results cover 155 exported source symbols; 100 remain absent.

The server embedding corpora exercise 256 ordinary tiny-model requests and
27 explicit pretrained BGE requests. Windows output bits and seven base64
payloads match the native HTTP reference. Linux retains the existing raw
encoder budget. Normalization 0 deliberately scales into +/-32760, so its
portable bound is derived from the unchanged raw-vector bound before testing;
the measured largest absolute scaled difference is 0.75439453125. This is
separate from direct normalization of identical native raw inputs, checked
over 4,608 outputs with exact Windows bits and at most two portable ULPs.

Transport failure tests exposed a deadline race: the listener's monitor could
cancel the worker before the handler checked its timer, turning a timeout into
a dropped response. Both embedding and nonstream generation now check the
deadline before forwarding a generic cancellation result. The regression
requires a complete HTTP 504 response and verifies that capacity is released.
Disconnect and server-shutdown cancellation remain separate cases.

## Memory and execution qualifications

The high-level model facade loads a single dense Llama or BERT GGUF into host memory. Non-matrix vectors are decoded F32. Matrices retain a decoded F32 row-access cache and, for supported non-F32 types, the original packed bytes needed for native dot arithmetic. The decoded-weight limit is not a total process memory limit. There is no mmap, tensor streaming, GPU model placement or full asynchronous scheduler claim.

Decoder context state uses F32 KV. BERT uses complete bidirectional sequences and its trained position table; it has no decoder KV context through this facade. Model instances are immutable and shared; each generation request owns independent context and sampler state. Stop strings and incomplete UTF-8 suffixes are withheld across token boundaries. Errors and cancellation do not commit partial model state. The C ABI documentation separately states accepted/rejected native parameter configurations.

Generation currently supports greedy or temperature/distribution sampling with an explicit u32 seed. The presence of other sampler implementations in the behavior crate does not mean every upstream generation option is wired into the CLI/server.

## Validation and honest failure reporting

Current validation logs are under `evidence/2026-11-09/`; historical archive logs remain under `docs/imports/`. Test outputs are associated with their command, platform and inputs. A test marked ignored requires an explicit external model, native reference library, installed compiler or vendor service. Ordinary workspace tests do not substitute for those explicit runs.

The source audit checks implementation roles and dependency metadata; it reports unsafe review candidates instead of claiming every unsafe block is proven sound. Fixture integrity checks hashes, declared lengths, provenance fields and complete fixture inventory. Numerical tests check semantics separately.

`release/gates.tsv` is the active mandatory gate list. `cargo xtask release-check` must return nonzero until all full-port requirements have independent closure evidence. A green build or a checkpoint commit cannot close that gate.
