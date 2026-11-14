# Native dense Llama and Qwen CUDA runtime

## Implemented path

The `llama-cuda-runtime` crate owns a persistent, single-sequence dense Llama,
Qwen2 or Qwen3 decoder on a selected NVIDIA CUDA device. Its kernels are generated from the
Rust typed instruction representation in `llama-device-family`; the CUDA driver
JIT compiles the generated PTX. The product does not link to or launch the
original llama.cpp, GGML, CUDA C++ kernels, Python, or an external inference
server.

Device execution covers token embedding lookup, RMS normalization, Q/K/V
separate or fused projections and optional biases, per-head Q/K normalization,
adjacent-pair or half-pair full-head RoPE, F32 KV cache
writes, causal grouped-query attention, softmax, attention output, residual
addition, SwiGLU feed-forward layers, final normalization and output projection.
Weights, scratch buffers, KV data and compiled modules remain live between
tokens. The host loads and validates GGUF, retains the existing decoded matrix
caches, calculates fixed RoPE phases, tokenizes, samples and streams text.
Complete model decoding therefore executes on the GPU; token handling remains
on the CPU.

The public Rust entry points are `CudaDecoder::new`, `new_qwen2` and
`new_qwen3` for validated typed models, and
`CudaDecoder::from_inference_model` for a high-level loaded model.
`CudaLlama` remains a source-compatible type alias. Borrowed model views
preserve packed storage without cloning model weights.
`decode` returns the complete vocabulary logits and the final normalized
embedding. The inference crate's `DecoderSession` interface shares sampling,
stop matching, UTF-8 streaming, seed handling and token statistics between
CPU and CUDA generation. A supplied session must belong to the same weights,
have compatible dimensions and have an empty committed prefix. The trait
cannot authenticate arbitrary external model weights.

## Admission and boundaries

| Capability | Implemented scope |
| --- | --- |
| Model | Dense Llama, Qwen2 and Qwen3 using their validated GGUF loaders |
| Matrix storage | F32, F16, Q4_0 and Q8_0; mixed supported types are accepted |
| Attention | Causal MHA, MQA or GQA; F32 K/V; one sequence |
| RoPE | Llama adjacent pairs; Qwen2/Qwen3 half pairs; full head, unscaled configurations |
| Output | Tied or untied matrix; optional Qwen2 output bias; logits and final embedding |
| Biases | Source-specific separate/fused QKV bias precedence; optional Llama attention/FFN biases |
| Qwen3 heads | Independent query width, shared per-head Q/K RMS weights and grouped KV heads |
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
frozen scalar reference uses that order. F16 weights are decoded for F32
products; activations remain F32, matching the frozen F16-by-F32 dot contract.
Q4_0 and Q8_0 paths quantize activation
blocks to the source Q8_0 representation on device, including half-rounded
scales and round-away-from-zero integer decisions. They then use integer block
sums and the source's F32 scale-multiplication order. A former implementation
that multiplied quantized weights by unconverted F32 activations failed the
pretrained comparison; it was replaced rather than accommodated with a wider
error allowance.

RMS kernels divide their F64 square sum by the row width before rounding;
multiplication by a rounded reciprocal can cross a boundary at non-power-of-two
widths. Exact 98-element regressions cover both whole-row and per-head RMS.
Softmax keeps the F64 reciprocal until its final F32 conversion. Scalar
exponentials use F64 range reduction and a degree-12 polynomial before F32
rounding. GNU x86-64 builds emit the frozen no-FMA SSE2 exponential policy,
four-lane F32 softmax summation and scalar SiLU tails. The row arithmetic
runs on the device. These policies preserve the independently measured host
reference profiles; they are not a claim of vendor CUDA-kernel equivalence.

The typed instruction representation covers F64 operations and conversions,
F16 conversion, bounded loads, scalar comparisons and finite predicates.
Its interpreter is useful for instruction-level checking. Numerical model
expectations are independently captured original C-API outputs, not outputs
from that interpreter.

The numerical regressions cover these error classes:

* Adding 0.5 before integer truncation can round a representable number just
  below a half-integer onto the boundary. The kernel now compares an exact
  fractional remainder and restores the sign. Tests check immediately below,
  at and above boundaries, signed values, zero and subnormal scales.
* Overflow can create nonfinite intermediate values that a subsequent maximum,
  normalization or integer conversion would otherwise erase. Typed finite
  checks preserve an invalid marker through the graph, and the host rejects
  nonfinite output without committing the token. Both finite-huge-weight
  counterexamples retain actual before/after GPU evidence.

* An approximate device exponential changed an attention output by one ULP
  in Qwen3, crossed a Q8 activation boundary, and amplified into a wrong logit
  on the next token. The original pretrained fixture retains this regression.
  Per-layer traces established the cause before the numerical implementation
  changed; the accepted comparison keeps its original error allowance.

The initial dense-activation trial also used a provisional 0.15 absolute bound
and still failed. That provisional bound was removed. Pretrained model replays
use the unchanged existing gate:

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
cargo test --release --offline --locked -p llama-cuda-runtime --test qwen -- --ignored --nocapture --test-threads=1
```

Repeat the live test commands without `--release` for debug acceptance. The
Qwen structural tests use committed original fixtures; the pretrained tests
require these external, hash-pinned files:

| Variable | Model | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `LLAMA_TEST_MODEL` | stories260K F32 | 1,185,376 | `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d` |
| `LLAMA_TEST_Q4_MODEL` | stories15M Q4_0 with Q8_0 matrices | 19,077,344 | `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739` |
| `LLAMA_TEST_F16_MODEL` | Native-original F16 conversion of stories260K | 601,248 | `4cd29bbe9cb3b735750837816a43c3dd4b473dbd658e8612816f7c4d3926ac21` |
| `LLAMA_TEST_QWEN2_F16_MODEL` | Qwen2.5-0.5B-Instruct F16 | 994,157,120 | `f781a61a06b2dbafd1404187dc27a7981cb45a7bd66fa3268c3fa724b602f38e` |
| `LLAMA_TEST_QWEN2_Q8_MODEL` | Qwen2.5-0.5B-Instruct Q8_0 | 531,068,480 | `6138491dbbde6c58c3380ac997ec64b4d2fd2e721b02a011a6463a545ab5bfe5` |
| `LLAMA_TEST_QWEN3_MODEL` | Qwen3-0.6B Q8_0 | 639,446,688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |

F32 reuses the existing platform-specific original C-API corpus. Q4 reuses the
existing original scalar reference. The new F16 corpus was captured with the
original quantizer and original MSVC C API before its first Rust GPU replay;
both host targets compare that same reference. The F16 conversion command is
recorded with executable and DLL identities in
[the fixture provenance](../fixtures/inference/cuda/provenance.json).
The model weights and original implementations remain outside this repository.

Qwen2 pretrained references were independently captured through the original
MSVC and GNU C APIs before their first Rust GPU replay. The converted model
bytes reuse the previously accepted original conversion outputs. Model,
request, caller, native-library and output identities are recorded in
`fixtures/inference/qwen2-cuda-pretrained/`; the complete fixture set is listed
in [the Qwen CUDA manifest](qwen-cuda-fixture-manifest.tsv). Qwen3 reuses its
unchanged original CPU inference corpus.

## 14 November validation corpus

| Independent model corpus | Executed models | Compared logits per host/profile | Exact greedy IDs |
| --- | ---: | ---: | ---: |
| Qwen2 structural | 21 | 165,888 | 198 |
| Qwen3 structural and vector storage | 38 | 251,904 | 340 |
| Qwen2.5-0.5B-Instruct F16 | 1 | 1,975,168 | 8 |
| Qwen2.5-0.5B-Instruct Q8_0 | 1 | 1,975,168 | 8 |
| Qwen3-0.6B Q8_0 | 1 | 5,469,696 | 24 |

The five Qwen tests check 9,837,824 original logits and 578 greedy IDs per
completed profile, along with complete generated UTF-8, finish reasons and
token counts. The same tests check invalid-token rejection, cancellation,
truncate/reset replay and explicit rejection of six Q4_K configurations.
The Qwen2 structural corpus retains its original `2e-6 + 2e-6 * abs(reference)`
bound; the other model comparisons retain their existing pretrained bound.

All four measured host/build profiles passed. GNU debug and release matched
every Qwen comparison bit for bit. MSVC debug and release matched all Qwen2
structural values and both pretrained Q8_0 models exactly. MSVC Qwen2 F16
matched 245,131 of 1,975,168 values bit for bit, with maximum absolute error
`5.4836273193359375e-6`; the Qwen3 structural corpus matched 251,482 of 251,904
values exactly, with maximum absolute error `7.450580596923828e-8`. Every
comparison remains within its unchanged bound and every greedy ID matches.

The live kernel corpus now has 162 analytical cases with two persistent
launches per case, eight malformed-uniform cases and ownership/lifetime
checks. It covers full-row and per-head RMS rounding, half-pair RoPE,
slice guards, scalar/SSE2 exponentials and SiLU range/tail behavior.
A separate biased GQA model checks 44 original GGML values, its exact
3,812 requested buffer-byte allocation plan, the one-byte-below rejection,
and numerical rollback. These cases complement the pretrained models.

The [14 November GPU receipts](../evidence/2026-11-14/cuda-runtime/README.md)
record each actual Windows MSVC and Linux GNU debug/release result, unchanged
execution-input identities and the original failed numerical attempts. The
[independent review](../evidence/2026-11-14/cuda-review/README.md) records the
RMS boundary counterexample and its correction. Windows and Ubuntu WSL use
the same physical RTX 4090 with NVIDIA driver 621.01; they are separate host
profiles. Performance, larger Qwen models and additional hardware remain
unverified.

## Retained 13 November Llama measurements

The Windows focused replay compared 62,976 F32 logits (96 greedy IDs),
59,904 unique F16 logits (96 IDs), and 640,000 quantized logits (16 IDs).
Maximum absolute errors were 7.152557373046875e-6, 6.556510925292969e-6 and
zero, respectively. The GNU debug replay recorded 6.020069122314453e-6 for F32,
the same F16 maximum and exact Q4 logits. The F32 count includes duplicated final-prompt/first-greedy
rows, consistently with its existing corpus. The same test also replays
complete autoregressive generated text, cross-token stops, reset, truncate,
invalid tokens and cancellation recovery.

The current continuation reruns all three Llama models after the shared
numerical-kernel changes. See the [historical GPU receipts](../evidence/2026-11-13/cuda-runtime/README.md)
and [13 November review](../evidence/2026-11-13/cuda-review/README.md)
for the original 135-case kernel corpus and retained earlier failures.
