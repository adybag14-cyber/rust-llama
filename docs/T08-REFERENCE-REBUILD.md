# T08 — Rebuilding the independent native comparison

The checkpoint includes a complete **optional Rust-authored build recipe and Rust FFI driver** for rerunning all three native RPC integration tests. This replaces the earlier manual harness reconstruction requirement. It does not add original llama/ggml code to the Rust product or to ordinary Cargo builds.

The supplied original `llama.cpp-master.zip` is still required; its SHA-256 is `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. Extract it independently, outside the Rust workspace. The recipe checks the SHA-256 of **124 relevant source/header/build-template files** against `crates/vendor-remote/examples/reference/inputs.sha256` before compiling. A different snapshot or a changed checked input fails explicitly.

## Commands

Validated environment: x86-64 little-endian Linux, supplied Rust 1.99.1, GCC/G++ 14.2.0, GNU `sha256sum`, system pthread/dl/libm and the C/C++ standard libraries. No CMake, Python, network dependency download or crates.io package is required. `CC` and `CXX` can select other compiler executables, but their compatibility and numerical output have not been certified by this checkpoint.

From the Rust workspace root, after putting the supplied toolchain on `PATH`:

```bash
cargo build --offline -p llama-vendor-remote --examples

# Replace these two paths with an extracted original tree and a separate,
# writable build directory. Both paths may contain spaces.
T08_FROZEN_SOURCE="/absolute/path/to/llama.cpp-master"
T08_COMPARE_BUILD="/absolute/path/outside/both/source/trees/t08-reference"

./target/debug/examples/t08_build_reference \
  "$T08_FROZEN_SOURCE" "$T08_COMPARE_BUILD"

T08_NATIVE_RPC_ORACLE="$T08_COMPARE_BUILD/rpc-oracle" \
  cargo test --offline -p llama-vendor-remote --test reference_interop \
  -- --include-ignored --nocapture
```

The build directory must be outside both the original source tree and the Rust workspace. The helper creates and writes only that external output directory. A build uses at most four native compiler subprocesses concurrently. Its 28 explicitly listed translation units are the frozen base, CPU and RPC implementations. Native compiler/linker commands, compiler versions, per-file logs, input hashes, layout output and generated-artifact hashes remain in that directory.

The optional complete test command also requires a usable LP64 CBLAS library such as `libblas.so.3`:

```bash
T08_NATIVE_RPC_ORACLE="$T08_COMPARE_BUILD/rpc-oracle" \
  cargo test --offline -p llama-vendor-remote -- --include-ignored --nocapture
```

That command passed **39 tests, zero failures**, including the two actual CBLAS tests, all three native RPC comparisons and both original SWIV byte fixtures. Its exact output is retained in `T08-evidence/rebuilt-reference-tests.txt`. The follow-up Clippy result and generated artifact hashes are in `T08-evidence/reference-rebuild-manifest.json` and the companion text files. Native artifact hashes identify this observed build; byte-identical binaries across compiler versions or absolute build paths are not claimed.

## What the driver executes

`t08_build_reference.rs` compiles the **unchanged** original sources into `libggml-reference-rpc.so`, generates the frozen version-header template in the external build directory, then copies the already compiled Rust `t08_reference_driver` into that directory as `rpc-oracle`. No C++ transport driver or graph implementation is generated.

The copied driver preserves the existing integration-test interface:

| Mode | Independent native behavior exercised |
| --- | --- |
| `rpc-oracle --layout` | Runs the native declaration/layout comparison process and prints actual packed tensor sizes/offsets, HELLO message sizes, command count, plus the public `ggml_init_params` ABI. |
| `rpc-oracle 127.0.0.1:PORT` | Calls the original exported CPU registry/device APIs and `ggml_backend_rpc_start_server`, with one CPU device and two CPU threads. The parent integration test owns and terminates this process. |
| `rpc-oracle --client 127.0.0.1:PORT` | Calls original exported RPC backend, context/tensor allocation, naming, buffer clear, tensor set/get/free and device-memory APIs. The actual original C++ client performs its handshake, dispatching and serialization. A responding memory query follows explicit free. The F32 test values are `[1.25, -2.5, 3.75, 4]`. |

The driver treats tensor/context/backend handles as opaque pointers. It checks the one by-value public struct against the native probe before using FFI and frees buffers, context metadata and backend/dispatcher in that order while the library remains loaded. It accepts only explicit IPv4 loopback endpoints. `T08_REFERENCE_DIR` can override the adjacent artifact directory when running the example directly from Cargo's target directory.

The native server test runs the unchanged Rust `exercise_client`: allocations, transfers, 2D transfers, copies, memset, **actual original CPU F32 matrix multiplication**, graph replay and cleanup. The native client test runs against the Rust server and checks the original public-API client's exact F32 roundtrip. These are independent native/Rust directions, not an all-Rust pair labeled as reference interoperability.

## Private layout probe boundary

The original `rpc_tensor` and message declarations are private to `ggml-rpc.cpp`. A Rust program cannot query their C++ layout through the public native ABI. The builder therefore extracts **only the four original declarations** (`rpc_tensor`, `rpc_cmd`, HELLO request and response), includes original headers for their constants, and emits a temporary `rpc-layout.cpp` containing those declarations and `sizeof`/`offsetof` output expressions specified in Rust. No native algorithm is authored or copied into that probe. The C++ probe source and its binary exist only in the external comparison output and are not shipped in the Rust checkpoint.

The observed layout was:

| Field | Native value |
| --- | ---: |
| `sizeof(rpc_tensor)` | 296 |
| `offsetof(rpc_tensor, data)` | 220 |
| `offsetof(rpc_tensor, name)` | 228 |
| `offsetof(rpc_tensor, use_count)` | 292 |
| HELLO request / response sizes | 24 / 28 |
| `RPC_CMD_COUNT` | 24 |
| `sizeof(ggml_init_params)` | 24 |
| `offsetof(ggml_init_params, no_alloc)` | 16 |

## Scope of reproduction

The shipped Rust examples, pinned input manifest and separately supplied original source are sufficient to rebuild and rerun **all three existing native integration tests**. No handwritten reconstruction of the earlier C++ harness is needed. The original manual-harness hashes and logs remain as historical evidence, with the new independently rebuilt output recorded separately.

This recipe verifies the TCP CPU subset on the stated host. It does not establish accelerator hardware, RDMA, collectives, all graph operations, all tensor types or complete T08 parity. The product remains independent of original native runtime objects; the optional reference driver deliberately loads those objects solely in its separate comparison process.
