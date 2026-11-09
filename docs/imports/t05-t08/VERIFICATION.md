# Verification of the integrated checkpoint

**Date:** 8 November 2026. **Source implementation commit:** `e5a6e57b38d0ba05bb174fd82e8e760467b30cd9`.

The commands below ran against the final integrated Rust implementation. Subsequent delivery changes add reports, captured evidence and a copy of the verified release binary. `evidence/command-results.json` records command arguments, start times, elapsed times, actual/expected exit codes and SHA-256 hashes of all captured logs.

## Verified build and test results

| Check | Observed result | Evidence |
| --- | --- | --- |
| `cargo fmt --all -- --check` | Exit 0 | `evidence/format.log` |
| `cargo clippy --workspace --all-targets --offline --locked -- -D warnings` | Exit 0; warnings denied | `evidence/clippy.log` |
| `cargo test --workspace --offline --locked -- --include-ignored --nocapture` | **115 passed, 0 failed, 0 ignored** | `evidence/tests-debug-all.log` |
| `cargo test --workspace --release --offline --locked -- --include-ignored` | **115 passed, 0 failed, 0 ignored** | `evidence/tests-release-all.log` |
| `cargo build --workspace --release --offline --locked` | Exit 0 | `evidence/build-release.log` |
| Frozen architecture/registration audit | Exact source inventories reported; exit 0 | `evidence/model-registry-audit.log` |
| Native model reference rerun | Native vectors regenerated; exit 0 | `evidence/model-reference-rerun.log` |
| Integrated model/RPC/CBLAS/IR smoke | Actual outputs and state roundtrips; exit 0 | `evidence/integration-smoke.log` |
| Platform/vendor probes | Explicit availability results; exit 0 | `evidence/device-probes.log` |
| Source/provenance audit | 55 generated/fixture hashes checked; exit 0 | `evidence/source-policy-audit.log` |
| Linux runner dynamic dependencies | Recorded with `ldd`; exit 0 | `evidence/binary-linkage.log` |

The debug and release runs execute the **same 115 distinct tests**, not 230 different tests. The ordinary default suite has **108 runnable tests and seven explicitly ignored external tests**. All seven were enabled and passed in the recorded final runs.

| Component | Default tests | External tests enabled separately | Total |
| --- | ---: | ---: | ---: |
| T05 model/context/state | 35 | 0 | 35 |
| T06 device-family semantics | 19 | 0 | 19 |
| T07 kernel IR and portable compiler/reference checks | 16 | 2 | 18 |
| T08 vendor/RPC/SWIV | 34 | 5 | 39 |
| Shared library ownership | 2 | 0 | 2 |
| T06/T07 integration fixtures | 2 | 0 | 2 |
| **Total** | **108** | **7** | **115** |

No GPU hardware test is hidden in those passing totals. Hardware attempts are reported separately below.

## Numerical and interoperability evidence

### Model operator graphs

The optional Rust T05 reference tool calls actual frozen GGML operations in a separately built native library. Synthetic two-layer fixtures compare Llama and Mamba embeddings/logits plus BERT token embeddings. **104 values in 20 vectors** pass the tolerance `3e-6 + 3e-6 * abs(reference)`.

| Slice | Values | Largest observed absolute difference |
| --- | ---: | ---: |
| Llama | 44 | `1.788139343e-7` |
| Mamba-1 | 44 | `5.960464478e-8` |
| BERT | 16 | `1.192092896e-7` |

The three ordinary numerical tests consume the captured native vectors, so default tests do not require the reference library. The Rust reference builder was separately exercised into a fresh output directory and reproduced those reference TSV values byte-for-byte. Detailed derivation and source hashes are in `docs/T05_STATUS.md`, `docs/T05_PROVENANCE.json` and `docs/T05_NUMERICAL_EVIDENCE.txt`.

These checks certify the exercised operator-graph slices with synthetic data. They do not establish full trained-model behavior, tokenizer/GGUF ingestion, the original context API or all variants of the three architecture families.

### Kernel representation and shader compiler

The T06 Q4 fixture contains eight native-derived blocks and all 256 reference F32 values; basis-vector checks cover those values. T07 checks every binary16 encoding against an independent Rust expression and, in its optional native test, against the frozen library's actual conversion. NaN results compare by classification. Unaligned/strided/padded Q4_0 matvec gives `[54.0, -88.75, 1.7285347e-6]` in both the Rust IR interpreter and native dequantization plus independent F64 dots.

The external shaderc test compiled all nine generated Vulkan GLSL programs to SPIR-V. Invalid source and a changed source with a stale digest are rejected. Compiled outputs and hashes are retained in `generated/`. This does not validate GPU dispatch, PTX assembly, WGSL/MSL compilation or an unavailable OpenCL compiler.

### RPC, CBLAS and SWIV

Three optional RPC tests compare native layout, run the Rust client against the original native CPU server, and run the original native client against the Rust server. The server direction includes actual native F32 matrix multiplication and graph replay. The native layout is 296 bytes for `rpc_tensor`, offsets 220/228/292 for data/name/use_count, HELLO sizes 24/28 and 24 command IDs.

The final integrated tests used the newly rebuilt reference library and Rust driver described in `docs/T08-REFERENCE-REBUILD.md`. The recipe checks 124 frozen input hashes before building 28 original translation units outside the Rust workspace. Both CBLAS tests use actual `libblas.so.3` calls. Two synthetic ELF fixtures match the original SWIV utility's bytes exactly. Detailed logs and provenance are retained in `docs/T08-evidence/` and `docs/T08-REFERENCE-EVIDENCE.md`.

The integrated smoke also performed 36 RPC commands, reported **zero retained allocated bytes after cleanup**, checked state roundtrips, and produced `[58, 64, 139, 154]` through CBLAS and the independent scalar IR demo. This is a functional smoke check, not a throughput benchmark.

## Hardware attempts and remaining evidence gaps

| Path | Actual observation | Consequence |
| --- | --- | --- |
| CUDA | Driver libraries unavailable; `cuda_smoke` exited **77** | Execution was **not run**. PTX assembly/JIT and device correctness remain unverified. |
| HIP / MUSA | Runtime/driver libraries unavailable | No device execution; program generation and hardware validation remain open. |
| OpenCL | Loader present; `clGetPlatformIDs` returned **-1001**, no platform | `opencl-smoke` exited **1** as expected for absence. No kernel executed. |
| Vulkan | Loader 1.4.309 present; `vkCreateInstance` returned **-9**, incompatible driver | No compatible device or runtime execution. Only external GLSL compilation passed. |
| Metal / WebGPU | Source emitters implemented; host adapters pending | No corresponding compiler/device result is claimed. |
| SYCL / Level Zero | Loader present; `zeInit(0)` returned `0x78000001` | No driver/device execution established. |
| VirtGPU | `libdrm.so.2` present; `/dev/dri` absent | Control-codec coverage only; no guest/host transport execution. |
| CANN / zDNN / OpenVINO / Hexagon | Required external libraries unavailable | Utility/probe implementation does not establish backend execution. |
| ET / ZenDNN | Vendor ABI gate remains open | No invented shim or unverified ABI is presented as an implementation. |

The command runner distinguishes an **expected absence result** from a successful hardware test. `expectation_met: true` for OpenCL/CUDA means the documented absence was observed; it does not mean an accelerator test passed. Exact diagnostics are in `evidence/opencl-execution-attempt.log`, `evidence/cuda-execution-attempt.log` and `evidence/device-probes.log`.

The T06 direct compiler-target experiments and the failed Clang OpenCL header probe are recorded in the track documents. No performance, sanitizer, cross-platform or physical-device certification is claimed beyond the checks above.

## Reproduce the ordinary checkpoint

Put the supplied Rust 1.99.1 `bin` directory on `PATH`, then run from the extracted workspace:

```bash
cargo build --workspace --release --offline --locked
cargo test --workspace --offline --locked
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo run --offline --locked -p track-runner -- smoke
cargo run --offline --locked -p track-runner -- probe
```

The default tests need no native reference build, shader compiler or BLAS installation. The integrated `smoke` command does exercise external CBLAS and reports that provider's availability; the reference and compiler tests remain opt-in. The release runner's exact hash, platform and shared-library dependencies are recorded in `evidence/binary-manifest.json`.

## Reproduce all external tests

Keep the original source archive separately and verify it against `evidence/input-manifest.json`. Extract it outside this workspace. On a compatible Linux host with GCC/G++, the two Rust build recipes create the independent oracles outside both source trees:

```bash
# Replace these absolute paths with local paths.
FROZEN_LLAMA_SOURCE="/absolute/path/to/llama.cpp-master"
MODEL_COMPARE_BUILD="/absolute/separate/model-reference"
RPC_COMPARE_BUILD="/absolute/separate/rpc-reference"

cargo run --offline --locked -p llama-model-runtime --example build_reference -- \
  "$FROZEN_LLAMA_SOURCE" "$MODEL_COMPARE_BUILD"
cargo build --offline --locked -p llama-vendor-remote --examples
./target/debug/examples/t08_build_reference \
  "$FROZEN_LLAMA_SOURCE" "$RPC_COMPARE_BUILD"

export LLAMA_REFERENCE_LIBRARY="$MODEL_COMPARE_BUILD/libggml-reference.so"
export T08_NATIVE_RPC_ORACLE="$RPC_COMPARE_BUILD/rpc-oracle"

# Also install a compatible external libshaderc and LP64 CBLAS provider.
cargo test --workspace --offline --locked -- --include-ignored --nocapture
cargo test --workspace --release --offline --locked -- --include-ignored
```

The external tests are: `glsl_corpus_compiles_to_real_spirv`, `q4_and_half_match_frozen_ggml_symbols`, the two `external_blas_*` tests and the three tests in `reference_interop.rs`. Their prerequisites are intentional. A missing reference library or compiler must fail an explicitly enabled test rather than silently convert it to a pass.

T05's graph-vector and registry audits can additionally be repeated using its documented `reference_oracle` and `registry_audit` examples. T08's build recipe details compiler options, native ABI boundaries and inputs in `docs/T08-REFERENCE-REBUILD.md`.

## Provenance and delivery audit

`evidence/input-manifest.json` records all three uploaded input hashes. `evidence/environment.json` records the actual Rust 1.99.1/Cargo versions, Linux host, native compiler and external shaderc identity. The reference ledger hashes all **1,685 scoped original paths**; the six binary test fixtures and 49 generated artifacts have separate SHA-256 manifests.

The product audit is deliberately limited: it checks accepted paths/extensions, exact fixture identities and generated-artifact provenance. It does not prove that all semantics are rewritten or that any complete plan gate is closed. No maintained C/C++/CUDA/Python/JavaScript/shell implementation files are shipped in the product. Generated OpenCL/GLSL/WGSL/MSL/PTX sources derive from Rust semantics; the optional native layout probe is produced only in an external comparison directory.

The Rust ZIP writer sorts entries, records CRC32, uses fixed timestamps, excludes `.git` and `target`, preserves the included runner's executable mode, rejects unsafe paths and refuses to overwrite an existing destination. The final archive is independently checked and extracted for an offline build/test before delivery. Those package checks occur after this report is written; the final delivery message reports their observed result.
