# Native dense Llama CUDA runtime

## Implemented path

The `llama-cuda-runtime` crate owns a persistent, single-sequence dense Llama
decoder on a selected NVIDIA CUDA device. Its kernels are generated from the
Rust typed instruction representation in `llama-device-family`; the CUDA driver
JIT compiles the generated PTX. The product does not link to or launch the
original llama.cpp, GGML, CUDA C++ kernels, Python, or an external inference
server.

Device execution covers token embedding lookup, RMS normalization, Q/K/V
projections and optional biases, adjacent-pair full-head RoPE, F32 KV cache
writes, causal grouped-query attention, softmax, attention output, residual
addition, SwiGLU feed-forward layers, final normalization and output projection.
Weights, scratch buffers, KV data and compiled modules remain live between
tokens. The host loads and validates GGUF, retains the existing decoded matrix
caches, calculates fixed RoPE phases, tokenizes, samples and streams text.
Complete model decoding therefore executes on the GPU; token handling remains
on the CPU.

The public Rust entry points are `CudaLlama::new` for an already validated
`LlamaModel` and `CudaLlama::from_inference_model` for a high-level loaded model.
`decode` returns the complete vocabulary logits and the final normalized
embedding. The inference crate's `DecoderSession` interface shares sampling,
stop matching, UTF-8 streaming, seed handling and token statistics between
CPU and CUDA generation. A supplied session must belong to the same weights,
have compatible dimensions and have an empty committed prefix. The trait
cannot authenticate arbitrary external model weights.

## Admission and boundaries

| Capability | Implemented scope |
| --- | --- |
| Model | Dense Llama using the existing validated Llama loader |
| Matrix storage | F32, F16, Q4_0 and Q8_0; mixed supported types are accepted |
| Attention | Causal MHA, MQA or GQA; F32 K/V; one sequence |
| RoPE | Adjacent-pair, full-head rotation and the loader's supported unscaled configuration |
| Output | Tied or untied matrix; logits and final embedding |
| Biases | Optional query, key, value, attention-output, gate, up and down biases |
| Device | Explicit ordinal, NVIDIA compute capability at least 6.0; measured hardware is RTX 4090 |
| State | Reset, truncate, decode cancellation and a reported committed prefix |
| CLI | Explicit CUDA generation and logits modes |
| Other public routes | The HTTP server and frozen C API retain their existing CPU execution scope |

Other architectures, packed weight types, alternate KV formats, flash
attention, scaled/partial RoPE, adapters, multi-sequence scheduling, multi-GPU
placement and CPU/GPU tensor partitioning are not admitted by this runtime.
The CUDA choice returns an error for an unsupported model or unavailable
device; it does not silently execute the CPU model graph.

This implementation does not establish the entire original CUDA backend,
its optimized algorithms, all named architectures, every CUDA device, or the
other accelerator families. Kernel availability is not a model-support claim.
The original 20 mandatory completion gates remain open.

## Numerical choices

Dense dot products use rounded F32 products and F64 accumulation where the
frozen scalar reference uses that order. F16 weights and activations preserve
the reference conversion semantics. Q4_0 and Q8_0 paths quantize activation
blocks to the source Q8_0 representation on device, including half-rounded
scales and round-away-from-zero integer decisions. They then use integer block
sums and the source's F32 scale-multiplication order. A former implementation
that multiplied quantized weights by unconverted F32 activations failed the
pretrained comparison; it was replaced rather than accommodated with a wider
error allowance.

The typed instruction representation covers F64 operations and conversions,
F16 conversion, bounded loads, scalar comparisons and finite predicates.
Its interpreter is useful for instruction-level checking. Numerical model
expectations are independently captured original C-API outputs, not outputs
from that interpreter.

Two independently reviewed error classes have permanent regressions:

* Adding 0.5 before integer truncation can round a representable number just
  below a half-integer onto the boundary. The kernel now compares an exact
  fractional remainder and restores the sign. Tests check immediately below,
  at and above boundaries, signed values, zero and subnormal scales.
* Overflow can create nonfinite intermediate values that a subsequent maximum,
  normalization or integer conversion would otherwise erase. Typed finite
  checks preserve an invalid marker through the graph, and the host rejects
  nonfinite output without committing the token. Both finite-huge-weight
  counterexamples retain actual before/after GPU evidence.

The initial dense-activation trial also used a provisional 0.15 absolute bound
and still failed. That provisional bound was removed. Every accepted replay
uses the unchanged existing pretrained gate:

`abs(actual - reference) <= 2e-4 + 2e-5 * abs(reference)`.

Greedy token IDs, generated UTF-8 bytes, finish reason and token counts must
match exactly. Bitwise equality is reported when observed, rather than
asserted as a portable promise across drivers and operating systems.

## Memory and failure semantics

`Options::max_device_bytes` bounds the sum of requested live CUDA-buffer
lengths. The plan is checked with overflow-aware arithmetic before device
allocation, including the decoded embedding, weights, optional biases,
RoPE tables, KV cache and scratch buffers. Individual kernel addressing is
bounded before PTX construction. The default requested-buffer limit is 8 GiB,
clamped to the addressable size on smaller hosts.

This is **not a total VRAM ceiling**. CUDA allocation granularity, context and
module overhead are additional. CPU model storage and decoded weight caches
also remain allocated. `Statistics::allocated_bytes` reports the same requested
buffer-byte measure. Uploads use bounded temporary chunks.

A compiled kernel retains its module and device context. Launches reject
foreign-context streams or buffers, short inputs and output aliasing. Scoped
resources remain alive until the driver call has synchronized. Dynamic token
and position uniforms have guards before device memory access.

The sequence position advances only after a complete successful decode.
Cancellation between layers, invalid tokens and rejected numerical results
leave the committed prefix unchanged. An uncommitted KV row may have been
written; retry overwrites that row before it can become an active prefix.
Truncation cannot extend the prefix. Reset and truncate change the active
prefix; upload, dispatch and successful-token statistics remain cumulative for
the lifetime of the session. Multi-token CUDA evaluation commits
successful tokens individually; CPU batch evaluation retains its existing
chunk transaction boundary. The session interface reports the actual prefix
in either case.

## CLI use

Load a supported GGUF and request CUDA explicitly:

```text
cargo run --release --offline --locked -p rust-llama-cli --bin rust-llama -- generate --model MODEL.gguf --backend cuda --device 0 --ctx-size 128 --max-gpu-mib 8192 --prompt "Once upon a time" --n-predict 16
cargo run --release --offline --locked -p rust-llama-cli --bin rust-llama -- logits --model MODEL.gguf --backend cuda --tokens 1,2,3
```

`--backend`, `--device` and `--max-gpu-mib` apply to generation and logits.
Device and GPU-budget options require the CUDA backend. Text goes to stdout.
Generation statistics on stderr identify the backend, device name, requested
CUDA-buffer bytes and kernel dispatch count. These counts establish the
executed path; they are not throughput measurements.

The public CPU generation API preserves its existing cancellation and
validation ordering, including embedding-model rejection and zero-token
short circuits.

## Reproducible checks

The ordinary workspace suite executes interpreter and input-validation tests
without requiring a GPU. Live-driver tests are explicitly ignored by default
and must be selected on a machine with the stated prerequisites:

```text
cargo test --offline --locked -p llama-device-family --test decoder_kernels -- --include-ignored --nocapture --test-threads=1
cargo test --offline --locked -p llama-cuda-runtime --test runtime -- --include-ignored --nocapture --test-threads=1
cargo test --release --offline --locked -p llama-cuda-runtime --test pretrained -- --ignored --nocapture --test-threads=1
```

The pretrained test requires these external, hash-pinned files:

| Variable | Model | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `LLAMA_TEST_MODEL` | stories260K F32 | 1,185,376 | `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d` |
| `LLAMA_TEST_Q4_MODEL` | stories15M Q4_0 with Q8_0 matrices | 19,077,344 | `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739` |
| `LLAMA_TEST_F16_MODEL` | Native-original F16 conversion of stories260K | 601,248 | `4cd29bbe9cb3b735750837816a43c3dd4b473dbd658e8612816f7c4d3926ac21` |

F32 reuses the existing platform-specific original C-API corpus. Q4 reuses the
existing original scalar reference. The new F16 corpus was captured with the
original quantizer and original MSVC C API before its first Rust GPU replay;
both host targets compare that same reference. The F16 conversion command is
recorded with executable and DLL identities in
[the fixture provenance](../fixtures/inference/cuda/provenance.json).
The model weights and original implementations remain outside this repository.

The Windows focused replay compared 62,976 F32 logits (96 greedy IDs),
59,904 unique F16 logits (96 IDs), and 640,000 quantized logits (16 IDs).
Maximum absolute errors were 7.152557373046875e-6, 6.556510925292969e-6 and
zero, respectively. The GNU debug replay recorded 6.020069122314453e-6 for F32,
the same F16 maximum and exact Q4 logits. Windows and Ubuntu WSL use the same
physical RTX 4090 with NVIDIA driver 621.01; they are separate host profiles. The F32 count includes duplicated final-prompt/first-greedy
rows, consistently with its existing corpus. The same test also replays
complete autoregressive generated text, cross-token stops, reset, truncate,
invalid tokens and cancellation recovery.

The live kernel corpus has 135 analytical cases with two persistent launches
per case, seven malformed-uniform cases and ownership/lifetime checks.
A separate biased GQA model checks 44 original GGML values, its exact
3,812 requested buffer-byte allocation plan, the one-byte-below rejection,
and numerical rollback. These synthetic cases complement the pretrained
models; they do not substitute for pretrained coverage.

See [GPU execution receipts](../evidence/2026-11-13/cuda-runtime/README.md)
and [independent review evidence](../evidence/2026-11-13/cuda-review/README.md)
for actual platform/profile results and retained failed attempts. Performance,
larger models and additional hardware remain unverified.
