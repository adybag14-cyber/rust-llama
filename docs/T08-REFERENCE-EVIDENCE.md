# T08 independent reference evidence

The comparison executable was built outside the Rust product from the frozen uploaded source. It is not shipped or linked by Cargo. The exact baseline archive hash and executable/harness hashes are in `T08-evidence/manifest.json`.

**Reproducibility follow-up:** `T08-REFERENCE-REBUILD.md` provides a complete shipped Rust build recipe and Rust public-ABI driver that rebuilt the untouched original library and passed all three existing native integration tests. The earlier manual harness described below is retained as historical evidence; it is no longer required to rerun the comparisons. New exact outputs and hashes are recorded separately in `T08-evidence/rebuilt-reference-tests.txt` and `reference-rebuild-manifest.json`.

## Native oracle construction

The native harness includes the **unaltered** `ggml/src/ggml-rpc/ggml-rpc.cpp`. This makes its actual packed definitions, static framing functions and `negotiate_hello` available to a tiny driver. The server path calls `ggml_backend_rpc_start_server` with the actual frozen CPU device. The client path calls the original `socket_t`, `negotiate_hello` and `send_rpc_cmd`, using original request structures. This is a protocol/native-backend oracle, not a reimplementation substituted for the reference.

The native build used GCC/G++ with `-O1 -ffunction-sections -fdata-sections -pthread -D_GNU_SOURCE -DGGML_CPU_GENERIC -DGGML_USE_CPU -DGGML_USE_RPC`, C11/C++17, the frozen `ggml/include`, `ggml/src`, and `ggml/src/ggml-cpu` include paths, and a generated `ggml-version.h`. The linked files were:

- Base: `ggml.c`, `ggml.cpp`, `ggml-alloc.c`, `ggml-backend.cpp`, `ggml-backend-meta.cpp`, `ggml-opt.cpp`, `ggml-threading.cpp`, `ggml-quants.c`, `gguf.cpp`, `ggml-backend-dl.cpp`, `ggml-backend-reg.cpp`.
- CPU: `ggml-cpu.c`, `ggml-cpu.cpp`, `repack.cpp`, `hbm.cpp`, `quants.c`, `traits.cpp`, `amx/amx.cpp`, `amx/mmq.cpp`, `tiled/tiled.cpp`, `tiled/tiled-kernel.cpp`, `binary-ops.cpp`, `unary-ops.cpp`, `vec.cpp`, `ops.cpp`.
- RPC: `log.cpp`, `transport.cpp`, and `ggml-rpc.cpp` included by the isolated oracle driver.

Linker flags: `-pthread -Wl,--gc-sections -ldl -lm`. RDMA was disabled. This build proves the TCP CPU protocol subset; it does not establish native accelerator or RDMA execution.

The driver supports three actions, which another independent harness can reproduce:

1. `--layout`: print `sizeof(rpc_tensor)`, `offsetof(data/name/use_count)`, HELLO structure sizes and command count. Observed: **296**, **220/228/292**, **24/28**, **24**.
2. `IP:PORT`: start the real frozen CPU RPC server with one device and two CPU threads.
3. `--client IP:PORT`: original handshake, device count, 64-byte allocation, base query, F32 set/get of `[1.25, -2.5, 3.75, 4]`, explicit free, then a responding device-count barrier. The implementation uses the original packed structures and send/receive helpers.

The Rust integration tests in `crates/vendor-remote/tests/reference_interop.rs` invoke these modes only when explicitly requested with `T08_NATIVE_RPC_ORACLE`. They do not build or locate an arbitrary reference implicitly. The test process owns and terminates the native server it starts.

## Important mismatch found and corrected

The first native graph test exposed a source-specific requirement: the CPU skips nodes lacking `GGML_TENSOR_FLAG_COMPUTE` (16). The initial Rust-only graph worked because it had not yet respected that flag. The checkpoint now preserves this behavior, and the shared probe sets COMPUTE plus OUTPUT on the result. A separate negative test verifies that a node without COMPUTE leaves its result unchanged. Final native tests pass. This finding is why independent reference execution is stronger evidence than a self-consistent Rust client/server pair.

## SWIV fixture derivation

Four small binary data files in `crates/vendor-remote/tests/fixtures/` are synthetic input/output pairs. ELF32 contains two loadable segments with BSS tails; ELF64 includes the ELF header itself in a loadable segment, so the changed section count participates in the checksum. The frozen `ggml/src/ggml-hexagon/htp/htp-swiv.py` generated the expected outputs with its `-i INPUT -o OUTPUT` interface. Both byte sequences match the Rust generator exactly and verify successfully after parsing. Fixture names, sizes and hashes, and the frozen script hash, are in `fixtures/manifest.json`.

The original Python utility was used only as an isolated comparison oracle. Normal builds/tests consume the generated binary fixtures and need no Python. The two fixtures establish exact behavior for these cases; they are not a claim that every ELF topology, filesystem replacement mode, or DSP load has been validated.
