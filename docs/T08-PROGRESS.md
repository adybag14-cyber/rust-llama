# T08 — Vendor, DSP, distributed and virtual backend checkpoint

**Baseline:** uploaded `llama.cpp-master.zip`, SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

**Status:** substantive partial implementation. None of the ten complete backend exit gates is claimed closed. The Rust package builds offline with no crates.io dependencies. Its only library dependency is the workspace's provisional external-library loader. No original llama/ggml object, C++ shim, Python runtime or native reference executable is linked into this package.

## Implemented and executed

| Component | Implemented behavior | Evidence | Remaining scope |
| --- | --- | --- | --- |
| RPC TCP | Exact version 8.0.1 IDs and request/response framing; explicit 296-byte packed tensor codec; HELLO and capability handling; 20 command handlers covering allocation, clear/free, size/alignment/memory, tensor initialization, set/get/copy/memset, 2D transfers, graph submission/replay. Hash-cache request returns a miss with caching disabled. | All-Rust loopback plus both directions against a freshly compiled frozen C++ reference. Rust client exercises real native CPU matmul and replay, with exact output comparisons. | RDMA, collectives, persistent tensor cache, backend registration/public ABI, asynchronous dispatch, all operations and dtypes. |
| RPC scalar execution | F32 contiguous, nonaliased graph slice: copies, add/subtract/multiply/divide with checked broadcasting, square/sqrt/log/sin/cos, affine scale, and GGML-shaped matmul with higher-dimension broadcast. `GGML_TENSOR_FLAG_COMPUTE` is respected. Results commit after every node succeeds. | Unit tests for IEEE division behavior, broadcasts, required COMPUTE flag, rollback, graph invalidation and resource bounds. Native matmul/replay interoperability. | A provisional validation executor, not T02/T03's production CPU backend. Views, alias scheduling, non-F32 compute, training and general inference graphs remain pending. |
| BLAS | Rust matrix views and packing, checked LP64 CBLAS invocation, GGML F32 matmul/out-product orientation, original minimum-size/Hadamard dispatch preference. | Actual `libblas.so.3` `cblas_sgemm` execution: `[58, 64, 139, 154]`, padded/strided inputs and out-product fixture. | Quantized conversions, 4D batching, thread/vendor selection, host-buffer registration and complete op dispatch. Vendor CBLAS remains explicitly external native math. |
| Hexagon utility | Bounded ELF32/ELF64 little-endian parsing, load/BSS CRC32, SWIV section generation and verification in Rust. | Byte-for-byte comparison with outputs produced by frozen `htp-swiv.py` for ELF32 with two BSS tails and ELF64 with the ELF header inside a load segment. Standard and incremental CRC fixtures. | Full FastRPC IDL/host/skeleton, DSP kernels, HVX/HMX, SDK/signing and physical execution. Utility has explicit size/table limits and rejects table growth that would overwrite contents. General filesystem in-place replacement semantics are not yet ported. |
| ET utility | The original 14-instruction forbidden-mnemonic guard implemented in Rust, with line-level findings. | Exact-mnemonic and comment handling tests. | This audits assembly/disassembly text, not binary opcodes. Full disassembler/source mapping, custom device instruction output, runtime and execution remain pending. |
| VirtGPU control data | APIR protocol 0.2, handshake magic `0xab1e`, handshake payload codec, version diagnostics, all 23 backend dispatch IDs. | Exact-byte fixtures and complete command-ID checks. | Guest/host execution, DRM resources, shared-memory ownership, fences and transport are not implemented. |

The accepted transfer-layout table covers 14 scalar/common-quantized dtypes, with explicit rejection of the remaining T03 layouts. A transfer layout is not an implementation of quantized computation. The default server advertises one local Rust validation device, **not** any accelerator family. Missing SDK/device implementations are never routed to this CPU slice under another backend label.

## Live feasibility results

The exact recorded output is in `T08-evidence/live-probes.txt`.

| Family | Observed on this host | What that establishes |
| --- | --- | --- |
| BLAS | `libblas.so.3` loaded; actual CBLAS matrix calls passed. | The external LP64 CBLAS route works for the implemented F32 adapter. |
| CANN | `libascendcl.so` unavailable. | Ascend execution remains blocked. |
| SYCL route | `libze_loader.so.1` and required symbols present; actual `zeInit(0)` returned `0x78000001`. | The loader is installed; a usable driver/device was not established. No SYCL or XMX execution. |
| zDNN | `libzdnn.so` unavailable. | IBM Z/NNPA execution remains blocked. |
| OpenVINO | `libopenvino_c.so` unavailable. | Vendor graph compilation/inference remains blocked. |
| Hexagon | `libcdsprpc.so` unavailable. | Rust ELF/SWIV data tooling runs; FastRPC and DSP execution do not. |
| ET | Default/configured ET platform directory absent; source uses C++ `rt::IRuntime` and `deviceLayer::IDeviceLayer`. | Runtime ABI/source and target toolchain gates remain open. |
| ZenDNN | `libzendnnl.so`/`libzendnn.so` unavailable; source uses C++ `zendnnl.hpp` and `matmul_direct`. | No audited C ABI or Rust replacement behavior. Snapshot pin remains `1f399a75cc0993778374a51bea49b64a57879595`. |
| RPC | Native/reference interoperability and Rust loopback passed. | Only the exercised wire/operation subset is established. |
| VirtGPU | `libdrm.so.2` required symbols present; `/dev/dri` absent. | Library availability, not a working virtual GPU or guest/host pair. |

The Level Zero historical initialization/enumeration signature was checked against the [official core API 1.9.1](https://oneapi-src.github.io/level-zero-spec/level-zero/1.9.1/core/api.html), [driver API](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/deprecated/driver.html), and [device API](https://oneapi-src.github.io/level-zero-spec/level-zero/1.20.41/core/api/apis/device.html). `zeInit`/`zeDriverGet` must not be mixed with `zeInitDrivers` in one process; this standalone probe uses the historical pair only. Future runtime integration must choose one initialization path across consumers.

## Bounds, ownership and failure handling

Default per-session limits are 8 MiB frames, 32 MiB single buffers, 64 MiB total buffers, 4,096 graph nodes, 8,192 graph tensors, 16 MiB cached graph metadata and 64 million scalar operation units. Scalar workspace is separately bounded to the total-buffer budget. Checked arithmetic covers dimensions, byte strides, range ends, graph counts, cache totals and 2D transfer spans. The graph-cache count remains 1,024; exceeding the memory budget fails explicitly instead of silently evicting at a point incompatible with the original client's cache sequence.

Every session owns its buffers and graphs. Clean disconnects, truncated frames, timeouts, unknown commands and response failures clear session allocations and cache. Unknown/stale handles, repeated HELLO, invalid device IDs, malformed tensor names/types/ops and out-of-range transfers fail. The native wire has no general error response, so the server closes on failure. A malformed response poisons the Rust client connection. The Rust demo server binds loopback only.

Scalar F32 division follows IEEE behavior, including infinities and NaNs at zero denominators; it does not panic or invent a domain rejection absent from ggml. Unsupported operators, views or aliasing return an explicit error before staged graph values commit. Freeing any buffer invalidates cached graphs.

There are no `unsafe` blocks in the RPC, ELF/SWIV, ET guard or APIR codecs. In the product library, `unsafe` is limited to external symbol lookup and the CBLAS/Level Zero call boundaries with documented signatures, checked buffers and loaded-library lifetime. The optional separate reference-driver example also uses documented original C ABI calls solely for independent comparison; ordinary product execution never loads original llama/ggml. No blanket `Send`/`Sync` assertion is made for vendor handles.

## Running the checkpoint

From the workspace root, with the supplied Rust 1.99.1 binaries on `PATH`:

```text
cargo test --offline -p llama-vendor-remote
cargo clippy --offline -p llama-vendor-remote --all-targets -- -D warnings
cargo run --offline -p llama-vendor-remote --example t08_check
cargo run --offline -p llama-vendor-remote --example t08_check -- serve-once 127.0.0.1:50052
cargo run --offline -p llama-vendor-remote --example t08_check -- client 127.0.0.1:50052
cargo run --offline -p llama-vendor-remote --example t08_check -- swiv input.elf output.elf
cargo run --offline -p llama-vendor-remote --example t08_check -- et-audit disassembly.txt
```

Two real CBLAS tests and three native reference tests are explicitly ignored by default because they require external artifacts. They fail, rather than silently pass, if requested without their prerequisites. In this session all five were run successfully together with the ordinary tests:

```text
T08_NATIVE_RPC_ORACLE=/path/to/isolated/rpc-oracle cargo test --offline -p llama-vendor-remote -- --include-ignored --nocapture
```

**Final recorded verification:** 34 unit tests, 3 native reference tests and 2 original SWIV fixture tests: **39 passed, 0 failed**, including the external gates listed above. Clippy with `-D warnings` passed. Exact commands and output are retained in `T08-evidence/`; timestamps and SHA-256 hashes are in its manifest.

The native comparisons are now reproducible from the shipped Rust examples plus the separately supplied frozen source. See `T08-REFERENCE-REBUILD.md`: `t08_build_reference` verifies 124 frozen inputs, builds original objects only in an external directory, and installs a Rust public-ABI comparison driver as `rpc-oracle`. All three unchanged native tests were rerun successfully through that rebuilt driver. No C++ implementation or native build artifact is added to the checkpoint.

## Integration handoff

`llama_vendor_remote::self_check()` returns a local TCP result and actual CBLAS result or explicit unavailable message. `probe_all()` returns ten separate reports with remaining gates. `RpcClient`, `ServerSession`, `RpcTensor`, `Graph` and `Limits` are owned safe-Rust boundaries. They are provisional pending T01/T02 public/backend/scheduler integration; they do not expose a compatible llama C API or claim full backend registration.

Next useful work is to connect the RPC executor to the authoritative T02/T03 graph and dtype contracts, add graph/dtype differential fixtures and asynchronous resources, then implement RPC collectives/cache/RDMA independently. In parallel, pursue the ET and ZenDNN C++ ABI gates and real Hexagon/VirtGPU runners. Availability alone must never advance a backend to “implemented” or “validated”.

`T08-source-inventory.tsv` records source paths, sizes and hashes, plus partial/pending capability disposition. It is a coverage ledger, not a completion percentage.
