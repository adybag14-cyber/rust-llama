# rust-llama

A native Rust implementation of the functionality in the supplied `llama.cpp-master.zip`.

**This is an active, incomplete port.** The repository builds and runs real pretrained Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BLOOM, StarCoder, StarCoder2 and Mamba-1 generation plus BERT embedding models on its Rust CPU runtime, including packed Q4 inference. Falcon CPU execution has independent synthetic-model coverage; pretrained Falcon acceptance remains open. The CLI also executes complete dense Llama, Qwen2 and Qwen3 F32/F16/Q4_0/Q8_0 graphs on CUDA. The full frozen-source contract still includes all model architectures, tensor operations, backends, converters, multimodal models, tools, browser features and native clients. Those requirements remain tracked; this checkpoint is not a complete replacement for llama.cpp.

The authoritative source input is the 41,458,461-byte ZIP with SHA-256:

```text
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122
```

Its 4,082 regular files have a complete ownership and hash inventory in [ledger/baseline-files.tsv](ledger/baseline-files.tsv). A path assignment establishes responsibility, not implementation parity. See [current implementation status](docs/PORT_STATUS.md), the [12-track contract](docs/plan.md), and [mandatory release gates](release/gates.tsv).

## Build

Rust 1.99.1 is pinned in `rust-toolchain.toml`. The workspace uses local Cargo path dependencies and five checksum-pinned Rust registry packages for the converter's YAML parser. Their complete source, feature and license review is enforced by the dependency audit.

```sh
git clone https://github.com/adybag14-cyber/rust-llama.git
cd rust-llama
rustup toolchain install 1.99.1 --profile minimal --component rustfmt,clippy
cargo fetch --locked
cargo build --workspace --release --offline --locked
```

The initial fetch acquires the locked dependency archives. The compiler, platform linker, standard library and those archives must already be installed for subsequent offline builds. Windows builds use the MSVC toolchain; Linux builds use the native Linux toolchain. No original llama/ggml library, C/C++ project build, Python or Node runtime is part of CPU inference.

## Run a model

Provide a supported GGUF model:

```sh
cargo run --release --offline --locked -p rust-llama-cli -- generate \
  --model /path/to/model.gguf --prompt "Once upon a time" --n-predict 32
```

On PowerShell, put the command on one line. Generated text is written to stdout; token counts and timing are written to stderr.

The high-level generation loader supports dense `llama` models with native SPM or BPE tokenization, full adjacent-pair RoPE without scaling, and F32 KV storage. Dense `qwen2` models add half-pair NeoX RoPE, separate or fused Q/K/V projections, source bias handling, and tied or separate output weights. Dense `qwen3` adds independent attention head width and learned per-head Q/K RMS normalization before full-head NeoX RoPE. Dense `gpt2` adds learned position embeddings and biased affine-LayerNorm blocks; dense `gptneox` adds partial NeoX RoPE and the original parallel or sequential residual order. See the [GPT-2](docs/GPT2_IMPLEMENTATION.md) and [GPT-NeoX](docs/GPTNEOX_IMPLEMENTATION.md) contracts for supported variants. Dense `bloom` models use fused QKV, affine input/final LayerNorm and ALiBi attention without RoPE; [the BLOOM contract](docs/bloom-inference.md) records supported head and output variants. Dense `falcon` models add fused QKV, full-head NeoX RoPE, optional second affine normalization and parallel GELU feed-forward execution; [the Falcon contract](docs/falcon-inference.md) records its exact graph and fixture boundaries. It also loads Mamba-1 GGUF models with causal convolution, selective recurrent state, optional dt/B/C RMS normalization and tied or separate output weights. Matrix arithmetic implements all 29 weight types in the frozen scalar CPU dot contract, including the K, IQ, ternary and FP4 formats; the [quantization documentation](crates/quant/README.md) lists the precise formats. Unsupported architecture features return an error. A decoder or kernel alone does not establish all-model quality or performance parity.

The runtime retains decoded F32 weights for row access and original packed bytes for native quantized matrix arithmetic. The default decoded-weight budget is 2 GiB; packed storage and runtime/cache allocations are additional. The default context bound is 1,024 tokens and is capped to the model's training limit. Both applications accept the first file of a supported GGUF shard set and automatically discover its declared siblings. The Rust API also accepts explicit ordered paths with cumulative parser limits. Both applications accept `--ctx-size` and `--max-weight-mib`; the pinned Qwen2.5-0.5B Q4_K_M and Qwen3-0.6B Q8_0 corpora use `--max-weight-mib 3072` because their decoded F32 row caches exceed 2 GiB. The official BLOOM-560M F16 and Q8_0 replays use `--max-weight-mib 4096`. These are explicit allocation bounds, not total process memory limits. CPU is the default backend. The CLI's `generate` and `logits` commands can select `--backend cuda` for single-sequence dense Llama, Qwen2 or Qwen3 with F32, F16, Q4_0 or Q8_0 matrices and F32 KV storage. `--device` selects the CUDA ordinal, and `--max-gpu-mib` bounds requested CUDA buffer lengths. Host decoded and packed weights remain retained; driver/context overhead is additional. [The CUDA contract](docs/CUDA_RUNTIME.md) records execution, state, resource and comparison boundaries. Other model, format and backend configurations retain explicit unsupported errors.

Dense `starcoder` models add learned position embeddings, fused QKV and
sequential affine-LayerNorm/GELU blocks. Dense `starcoder2` models use full-head
NeoX RoPE, affine normalization, sequential GELU and the source's required
projection biases. Both execute through the CPU generation and logits paths;
[their runtime contract](docs/starcoder-inference.md) records the supported
variants. The pinned StarCoder2 3B Q4_K_M replay uses `--max-weight-mib 16384`
for its decoded F32 row cache. That limit excludes packed weights and other
process allocations.

Other CLI commands expose validated GGUF metadata, tokenization, detokenization and per-token logits:

```sh
cargo run --release --offline --locked -p rust-llama-cli -- --help
cargo run --release --offline --locked -p rust-llama-cli -- inspect --model /path/to/model.gguf
cargo run --release --offline --locked -p rust-llama-cli -- tokenize --model /path/to/model.gguf --prompt "Hello"
```

## Split and merge GGUF files

The native Rust llama-gguf-split executable streams tensor payloads through a
bounded buffer and preserves the frozen tool's defined metadata order and file
layout:

~~~sh
cargo run --release --offline --locked -p rust-llama-cli --bin llama-gguf-split -- --split-max-size 4G model.gguf output/model
cargo run --release --offline --locked -p rust-llama-cli --bin llama-gguf-split -- --merge output/model-00001-of-00003.gguf merged.gguf
~~~

Tensor-count limits, metadata-only first splits, dry runs and deletion after a
complete merge are supported. Destructive cleanup requires stable paths under
the caller's control. See [the precise format and tool contract](crates/gguf/README.md)
for resource limits and original producer quirks. The [raw byte-string contract](docs/GGUF_BYTE_STRINGS.md)
covers non-UTF-8/NUL wire strings, native alias semantics and boolean normalization;
model loading now retains raw primary metadata and its native C-prefix projection,
with checked boundaries for consumed settings and tensor names.

## Convert a Hugging Face checkpoint

The native Rust converter accepts local dense Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BLOOM, Falcon, StarCoder and StarCoder2 Safetensors checkpoints. Falcon includes the frozen RW class alias; [its conversion contract](docs/FALCON_HF_CONVERSION.md) preserves the measured source configuration and QKV quirks:

~~~sh
cargo run --release --offline --locked -p llama-convert --bin convert-hf-to-gguf -- \
  /path/to/checkpoint --outfile /path/to/model-f16.gguf --outtype f16
~~~

It supports SentencePiece and the documented Hugging Face BPE paths, indexed
shards, model-card metadata, optional QKV fusion, vocabulary-only output, and
auto/F32/F16/BF16/Q8_0/TQ1_0/TQ2_0 output. A completed temporary output is synced
before replacing the destination. `--dry-run` reports the validated plan without
publishing a file. Python and Torch are not runtime dependencies.

See [the converter contract](crates/convert/README.md) for input and allocation
limits, native source profiles and unsupported modes. Other architecture
converters remain open. Conversion of scaled-RoPE metadata does not establish
scaled-RoPE inference support.

The [StarCoder conversion contract](docs/STARCODER_HF_CONVERSION.md) records
the original GPTBigCode/StarCoder2 aliases, metadata and tokenizer behavior.
Tiny StarCoder has unchanged pretrained checkpoint acceptance. StarCoder2 HF
conversion has synthetic-weight and official-sidecar coverage; its independent
pretrained GGUF inference result does not establish pretrained HF conversion.

## Quantize a GGUF model

The native Rust `llama-quantize` executable converts supported GGUF weights
without loading the model's inference graph:

~~~sh
cargo run --release --offline --locked -p llama-model-quant --bin llama-quantize -- \
  model-f16.gguf model-q4_k_m.gguf Q4_K_M 4
cargo run --release --offline --locked -p llama-model-quant --bin llama-quantize -- \
  --dry-run model-f16.gguf Q4_K_M
~~~

Options precede the input path. The tool supports the 35 frozen file modes,
`COPY`, mixed precision, requantization, layer pruning, metadata and tensor-type
overrides, split preservation, and legacy or GGUF importance data through
`--imatrix`. Its file loader admits Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX,
BERT, T5 and BLOOM metadata; T5 quantization does not add T5 inference.

The default live source/decoded/output slab budget is 256 MiB; importance maps,
codec workspaces and other allocations have separate bounds. Write-time
same-file aliases are rejected, including hard links. Dry runs plan the layout
without publishing output or inspecting the destination. See
[the CLI contract](crates/model-quant/CLI.md) for argument rules, importance
normalization, explicit safety corrections and resource limits, and
[the library/C contract](crates/model-quant/README.md) for publication semantics.

## Encode a BERT sequence

```sh
cargo run --release --offline --locked -p rust-llama-cli -- embed \
  --model /path/to/bert.gguf --prompt "A sentence to encode." --pooling model
```

Dense `bert` GGUF models use the native WordPiece tokenizer and bidirectional encoder. The JSON result includes input token IDs, every raw token embedding and the selected pooled vector. Pooling supports the model default, none, mean, first and last. Vectors are not L2-normalized. Optional BERT classifier heads and model-selected RANK pooling execute through the Rust embedding facade. The [native encoder C interface](crates/ffi/ENCODER.md) additionally preserves source sequence grouping, pooled output pointers and classifier metadata. Other BERT-derived architectures remain explicit unsupported cases.

See the [inference API and model boundaries](crates/inference/README.md).

## Serve completions and embeddings

```sh
cargo run --release --offline --locked -p llama-server -- \
  --model /path/to/model.gguf --bind 127.0.0.1:8080
```

The server connects native/OpenAI text completions and SSE to real Rust inference, with bounded HTTP parsing, cancellation, stop strings and replay. Text chat requires a supported model template. Unsupported request options fail explicitly. Without `--model`, the executable is a diagnostics service and reports inference unavailable.

To serve a supported BERT model's embeddings:

~~~sh
cargo run --release --offline --locked -p llama-server -- \
  --model /path/to/bert.gguf --embeddings --pooling model
~~~

The native /embedding and /embeddings routes preserve pooled or per-token
vectors; /v1/embeddings provides OpenAI shapes, token usage and optional base64
for pooled output. Pooled results use Euclidean normalization by default.
Text, token, mixed and batched inputs follow the frozen server's rules.

See the [server behavior and supported options](crates/llama-server/README.md). This is not full upstream server/browser parity.

## What is implemented

| Area | Executable implementation |
| --- | --- |
| Quantization | All 36 active dtype decoders; all 29 source model quantizers including required/optional importance weights; 32 non-integer codecs including activation and F64 helpers; checked integer setters; all 29 scalar CPU weight dots. |
| Model-file quantization | Native bounded streaming conversion with 35 source file modes across eight admitted metadata families, a standalone quantize CLI, legacy/GGUF importance ingestion, mixed precision, ordered overrides, pruning, copy/requantization, split preservation and frozen C entrypoints. See [the complete contract](crates/model-quant/README.md). |
| Tensors | Checked four-dimensional layouts, immutable shared views, a CPU DAG evaluator, 100 executable operation IDs plus NONE input leaves, scalar unary/GLU formulas for supported typed families, convolution/pooling/resize, recurrent and gated-delta operators, hyper-connections, Lightning Indexer, extended forward/backward RoPE, masked/sink/ALiBi softmax, owned SGD/AdamW state, threaded custom callbacks, vision/loss kernels, and a checked FLASH_ATTN_EXT adapter over the shared native CPU flash kernel. |
| Formats and vocabulary | Bounded shared raw/text GGUF v2/v3 reader and owned shard sets, byte-preserving ordered little-endian streaming writer, bounded split/merge executable, raw/indexed Safetensors reading with retained file capabilities, scalar endian conversion, Llama SPM, BERT WordPiece, BPE with all 64 reachable metadata-selected profiles / 99 aliases, Unigram, RWKV, PLaMo2/PLaMo3, explicit no-vocabulary mode, and exhaustive frozen Unicode/UTF-8 behavior. |
| Model runtime | Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BLOOM, Falcon, StarCoder, StarCoder2, Mamba and BERT Rust model graphs from typed weights; contexts, KV/recurrent state, transactions, adapters and state serialization. The GGUF facade loads dense Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BLOOM, Falcon, StarCoder, StarCoder2, Mamba-1 and dense BERT. The distinct Llama NativeContext supports independent F32/F16 K/V, source CPU flash/AUTO resolution, actual configured workers and unified row maps for admitted dense-Llama configurations. Explicit nonflash execution remains single-worker. Strict contexts account for owned state and retained checkpoints before mutation. |
| Inference and applications | Pretrained GGUF generation and BERT embeddings, CLI, real-model HTTP/SSE and embedding endpoints, and 247 implemented frozen C API exports, including context-attached sampler chains and sampled-output accessors, LoRA adapters, control vectors, native BERT encoding, model-file quantization, context performance/warmup/system information, and caller-owned FILE model/LoRA loading within documented cursor and ownership boundaries. |
| Checkpoint conversion | Native dense Llama/Qwen2/Qwen3/GPT2/GPTNeoX/BLOOM/Falcon/StarCoder/StarCoder2 HF Safetensors to GGUF conversion, bounded sidecars and tensor working sets, source tokenizer/metadata behavior including the Qwen Split expression, Falcon punctuation and digit-triple stages, pinned NFC and class resolution, seven output type selections and complete-file publication. |
| Devices | Persistent native Rust dense Llama/Qwen2/Qwen3 CUDA model execution, retained weights/KV/scratch and typed PTX kernels; generated OpenCL kernel execution and other backend foundations. Full architecture, format, scheduler, hardware and performance coverage remains open. |
| Other imported functionality | Selected native media processing, samplers, chat formatters, separate raw-byte native GBNF and safe character grammar APIs, RPC/CBLAS/DSP utilities and Rust audit/packaging tools. |

Detailed support boundaries are in the crate READMEs and [PORT_STATUS.md](docs/PORT_STATUS.md). Numeric registries, backend probes and example programs are not counted as complete implementations.

## Independent comparisons

The 14 November continuation adds StarCoder and StarCoder2 CPU execution and
HF conversion, Qwen2/Qwen3 CUDA graphs, and eight sampled-output C API exports.
The [StarCoder runtime contract](docs/starcoder-inference.md),
[CUDA contract](docs/CUDA_RUNTIME.md) and [current status](docs/PORT_STATUS.md)
record the admitted variants and remaining boundaries. Component and combined
receipts are indexed under [evidence/2026-11-14](evidence/2026-11-14).

The combined 14 November runs pass **768 Windows MSVC / 772 Linux GNU
ordinary tests in each of debug and release**, with zero failures, 49 explicit
prerequisites ignored and 147 result suites per run. Formatting, strict Clippy,
locked offline builds, sixteen separately invoked pretrained inference/server
tests and one native BERT encoder test pass per host. All four final C library
profiles pass retained clients and the new sampling/context/FILE replay with
exact inventories of **247 frozen exports plus four extensions**.

Four actual RTX 4090 host/profile runs compare 9,837,824 Qwen logits and 578
exact greedy IDs per profile, in addition to the existing Llama acceptance.
GNU Qwen values are bit exact; Windows F16/small-model differences remain within
the unchanged numerical bounds. Fifteen new release CLI commands per host
check Qwen CUDA and StarCoder CPU generation, full logits, statistics and
explicit rejections. Both hosts verify 1,143 fixture files covered by 1,154
integrity declarations across 30 manifests and both complete 4,082-file frozen
source trees.

The [combined validation](evidence/2026-11-14/integrated-port/README.md)
preserves the three failed attempts and their narrowly scoped corrections.
The [final audits](evidence/2026-11-14/final-audits/README.md) record the
subsequent CI/documentation delta; all Rust, Cargo and fixture inputs remain
identical. **All twenty mandatory full-port gates remain open.**

The following 13 November measurements retain their original scope.

The 13 November continuation added native dense Llama CUDA execution, Falcon CPU
inference and HF conversion, and seven frozen context/FILE API exports.
[CUDA](docs/CUDA_RUNTIME.md), [Falcon runtime](docs/falcon-inference.md),
[Falcon conversion](docs/FALCON_HF_CONVERSION.md),
[context performance](crates/ffi/CONTEXT_PERFORMANCE.md) and
[FILE loading](crates/ffi/FILE_POINTER.md) give the exact supported contracts.
Component and combined integration receipts are under
[evidence/2026-11-13](evidence/2026-11-13). All twenty broad completion gates
remain open.

The combined 13 November runtime checks passed **753 Windows MSVC / 757 Linux
GNU ordinary tests in each of debug and release**, with zero test failures.
Each ordinary run recorded 40 explicit prerequisites ignored and 142
test-result suites. Separate runs passed thirteen pretrained inference/server
tests and one native BERT encoder test per host. Original-header clients
passed against both debug/release libraries with exactly **239 frozen exports
plus four extensions**, leaving sixteen frozen exports absent.

Four actual CUDA host/profile replays each compare 762,880 native logits and
208 exact greedy IDs across three model files; the Q4_0/Q8_0 model is bit exact.
A separate 128-command CLI replay checks generated text, stop behavior, logits,
allocation statistics and explicit unsupported cases. See the
[combined runtime receipts](evidence/2026-11-13/integrated-port/README.md),
[GPU receipts](evidence/2026-11-13/cuda-runtime/README.md) and
[CLI receipts](evidence/2026-11-13/cuda-cli/README.md). The retained failed
attempts identify an obsolete external export-count guard and a collision with
an older quantizer output directory; subsequent checks preserve the original
expected outputs. The successful continuations retain the same 1,623 selected
execution inputs. Documentation and the final package audit are sealed
separately.

The final Windows and GNU audits verified **1,094 fixture files across 24
manifests**, both 4,082-file frozen source trees, and the reviewed source and
dependency boundaries. The full-release check retains its expected refusal
with all twenty mandatory gates open. After runtime verification, the only
change within the selected input set was the documented dependency-boundary
inventory update; all Rust, Cargo and fixture inputs remain unchanged.
[Final audit receipts](evidence/2026-11-13/final-audits/README.md) record that
delta and the separate documentation/evidence sealing boundary.

The following 11 November figures retain their original scope.

The 11 November continuation added BLOOM inference and checkpoint conversion,
native BERT encoder/classifier C execution, and bounded model-file quantization.
The frozen MSVC/GNU BLOOM captures compare 13,074,432 pretrained F32 words and
24 greedy IDs per host. Its unchanged official checkpoint also produces three
exact complete converter files totaling 2,871,136,896 bytes per host.

The encoder's primary original-header replay covers 192 scenarios and 37,384
exact values per library, with a separate nine-case active-vector boundary
corpus. The quantizer covers all 35 source file modes, 463 type/default cases,
76 primary complete outputs and 150 metadata/NextN complete outputs per
profile. Both additions have original-header debug/release C checks on Windows
and Linux. See the [current scope and qualifications](docs/PORT_STATUS.md).

The 11 November integrated checkpoint passed **728 Windows MSVC / 732 Linux GNU
ordinary tests in each of debug and release**, with zero failures. Each
ordinary run recorded 33 ignored prerequisites and 133 test-result suites.
Separate continuation checks passed thirteen pretrained inference/server tests,
six converter tests, one native BGE encoder test and one official Qwen tokenizer
test per host. Both hosts verified **1,062 fixture files and declarations across
nineteen manifests**, both 4,082-file frozen source trees, and the debug/release
C libraries' exact 232 frozen exports plus four extensions. Twenty-three frozen
exports were absent at that checkpoint.

Verification was staged. Both original full attempts passed formatting,
the ordinary suites and strict Clippy, then failed the packaging audit because
the CLI fixture's `LICENSE` filename was unclassified. The correction preserved
all 1,078 license bytes while renaming it to `LICENSE.md`, updated its one
fixture-manifest path, and raised the CPU CI timeout from 45 to 60 minutes.
Rust implementation and numerical fixture content did not change.

The schema-3 receipt preserves both failed attempts, the separate post-attempt
input check, the exact correction and the successful continuation checks.
Each host has 33 required successful stages with explicit provenance.
The ordinary suites were not repeated after this packaging and CI configuration
correction. All twenty mandatory full-port gates remain open. See
[the latest staged integrated evidence](evidence/2026-11-11/continued-port-late/README.md)
for stage provenance, input hashes and qualifications. Published-commit CI is
reported separately by GitHub Actions.

The preceding integrated continuation passed **691 Windows / 696 GNU ordinary
tests in each of debug and release**, with zero failures and 29 explicit
prerequisite tests ignored per ordinary run. Separate checks passed eleven
pretrained inference/server tests, five complete-file converter tests and the
official Qwen tokenizer replay per host. Formatting, strict Clippy, reviewed
dependency/source audits, release builds, all 959 fixture checks across fourteen
manifests, and both 4,082-file frozen-tree verifications passed. Original-header
clients checked both debug and release libraries, including the new adapters,
with exactly 226 frozen exports plus four extensions and 29 absent exports.

This validation is recorded in stages. The initial runs stopped at packaging
audit, and the first Windows continuation stopped on generated bytecode in the
reference copy. The retained reports preserve those failures and their
corrections. Runtime code and numerical fixture bytes stayed unchanged; both
final continuations retained the same 1,536 selected execution inputs.
All twenty mandatory full-port gates remain open. See
[the staged integrated evidence](evidence/2026-11-11/continued-port-next/README.md)
for commands, hashes, exact coverage and input-scope qualifications.
Published-commit CI is reported separately by GitHub Actions.

The earlier 11 November continuation passed **662 Windows / 667 GNU ordinary tests in
each of debug and release**, with zero failures and 27 explicit external/device
prerequisites ignored in each ordinary run. Both hosts verified all 4,082 supplied
source files and all 858 fixture files across ten manifests. Separate release
runs passed eleven pretrained inference/server tests, three complete-file
converter tests, and the official Qwen tokenizer replay. Original-header clients
and debug/release export inventories checked 216 frozen C exports plus four
extensions; 39 frozen exports were absent at that checkpoint. The retained qualification explains
the CI-only change detected by the original execution-input guard, the unchanged
1,279 other recorded inputs, and the successful four-stage postflight. Published-commit
CI is reported separately by GitHub Actions. All twenty mandatory
full-port gates remain open. See [the integrated evidence](evidence/2026-11-11/continued-port/README.md).

The complete 10 November integration passed **592 Windows / 597 Linux ordinary
tests in each of debug and release**, strict Clippy, formatting, release builds
and the dependency audit. Each host also passed nine explicit pretrained
inference/server tests and the complete converter test. All 4,082 supplied source
files and 720 fixture files were verified. Debug/release libraries passed the
original-header C clients and exact 198-export inventories. The 22 ordinary-suite
ignored entries keep their external-data/device prerequisites; the separately
executed pretrained tests are recorded explicitly. See the
[integrated verification](evidence/2026-11-10/continued-port/README.md) for raw
logs, input hashes, retained failure diagnoses and the 20 still-open full-port gates.

Reference C/C++ is built and executed outside this product. Committed fixtures contain data and provenance; runtime crates do not call the reference implementation.

The 10 November extension adds independently captured model metadata, byte-oriented
C chat APIs, and a native dense Llama checkpoint converter. The metadata corpus
contains 171 model requests, over 334,000 buffer calls and 144,384 continuation
logit words per profile; the chat corpus contains 35,946 cases per profile.
The converter's ordinary suite compares 57 complete files totaling 8,091,032
bytes and seven original rejections. Its separate pinned SmolLM2 test verifies
three complete outputs totaling 686,581,536 bytes. See
[metadata](crates/ffi/METADATA.md), [chat](crates/ffi/CHAT.md), and
[conversion](crates/convert/README.md) for the exact boundaries and provenance.

The retained 9 November 2026 evidence includes:

- **Pretrained F32 inference:** six prompts, 96 matching greedy token IDs and 62,976 compared logits. Matching source accumulation and physical attention-row layout gives bit-identical results against the corresponding Windows and GNU captures.
- **Pretrained Q4_0 inference:** 640,000 unique logits and sixteen greedy IDs match the corresponding original profiles. Packed Q8 activation conversion is retained; the earlier failed all-F32 comparison remains recorded.
- **Pretrained BPE/Q4_K_M inference:** 1,671,168 unique SmolLM2 logits and 24 generated IDs match the corresponding Windows and GNU profiles.
- **Pretrained Mamba-1 inference:** 1,709,520 unique logits and 24 generated IDs match the corresponding original profiles. Three synthetic recurrent models provide ordinary loader and state regressions. Earlier GNU drift remains in the historical evidence.
- **Pretrained Qwen2 inference:** 5,165,824 unique logits and 24 generated IDs match each original profile, with independently captured Q4_K_M weights. Twenty-one synthetic Qwen2 variants cover separate/fused projections, partial biases, output configurations and packed weights. Thirteen physical metadata variants preserve source resolution rules.
- **Pretrained Qwen3 inference:** 5,469,696 unique Q8_0 logits and 24 greedy IDs match each original profile. The ordinary dense/split/vector corpus compares 294,912 logits and 400 IDs per profile, including independent head widths, learned Q/K normalization and fused-bias selection. Active low-precision vector cases rejected by the original CPU graph return checked errors. See [the precise Qwen3 scope](docs/QWEN3_IMPLEMENTATION.md).
- **F16 native caches:** independent F32/F16 key and value storage preserves raw half words in memory, sequence state and sessions. The ordinary original-C corpus compares 258,048 exact logits and 1,048 state/session buffers per profile across all four type pairs; a separate external trained/tiny C cohort checks 491,520 exact logits. The native microbatch/flash path now accepts the original default CPU configuration for admitted dense Llama models. Its 228 frozen-header cases contain 226 defined native returns and two source initialization assertions converted to checked errors, with 2,917,888 exact logit words and 14,171,045 exact binary bytes per profile. Separate flash/nonflash state replay checks 8,459,616 exact bytes. The strict high-level Context retains its existing execution contract.
- **BERT embeddings:** 13,056 token/pooled values match the corresponding original profiles. The GNU correction preserves the source's four-lane variance reduction. All 122 captured stages of the Windows diagnostic sequence match exactly; 46,080 independent attention outputs remain in the ordinary suite.
- **Sharded loading:** 90 native model/index configurations, including 65 admitted sets and 25 rejections; 64 model configurations compare 464,272 logits/embedding values and 590 greedy IDs per native profile. First-file metadata, per-file tensor origins, cumulative resource limits and retained file handles are checked independently. Architecture-specific checkpoint converters and remaining format boundaries stay open.
- **GGUF streaming tools:** all 103 independent original CLI cases per compiler profile pass, with 63 defined successes and 40 checked errors. Complete post-operation comparisons cover 12,179,168 exact file bytes. Cumulative budgets, one-byte copy buffers, retained-handle alias protection and pre-completion input preservation are checked separately. The [raw GGUF extension](docs/GGUF_BYTE_STRINGS.md) adds 1,605 native reader/getter cases, 269,184 exact native serialization bytes, 1,073 public-writer inputs and 3,218 byte/NUL tool cases per native profile. Raw APIs and split/merge preserve full wire bytes and the original prefix/overwrite rules; the later model-metadata extension adds a raw primary owner and native model projection. Source UB produces checked errors.
- **Vocabulary behavior:** the preserved SPM/WordPiece corpora plus 1,499,449 complete BPE vocabulary records across all 14 original BPE models; 5,096 real-model token/piece/detokenization cases; 18,018 GPT-2 alias cases; 23,754 synthetic configuration cases; 22,518 raw-byte cases; 126,896 regex splits. All 64 reachable profiles and 99 aliases are covered on Windows and Linux. Source-inferred special IDs preserve the measured platform container ordering.
- **Remaining tokenizer families:** Unigram, RWKV and PLaMo2/PLaMo3 have 91,476 synthetic family cases, 346,112 original vocabulary records, 10,412 real-vocabulary tokenization cases and 4,067,668 exact normalizer outputs. Invalid source assertion/nonprogress inputs return bounded errors. The no-vocabulary mode is explicit.
- **Unicode behavior:** all 1,114,112 codepoints, 15,957 normalization sequences and 65,809 valid/malformed UTF-8 cases checked against the original source.
- **Quant codecs and dots:** all 36 dtype layouts, exhaustive F16/BF16/F8 patterns, all 29 model quantizers, additional activation/conversion codecs, and complete scalar weight-dot inventory. The weighted/IQ extension adds 3,362 exact original-C encoder rows and 640 native tied-sort cases. Source-undefined payload/access cases have explicit safe behavior documented in the quant crate.
- **Tensor graphs:** 3,240 distinct independently executed frozen GGML graphs. The new convolution/pooling/resize slice adds 981 distinct cases and 530,246 exact output bytes per platform, with exact shapes and strides. Twenty overlapping-gradient cases preserve the measured platform-specific NaN operand order. Indexing compares 599,152 bytes per platform and preserves 88 native MSVC/GNU sort-order differences. The scalar extension checks 256,233 exact outputs, 170,260 transcendental outputs and 96,910 independently rounded high-precision ERF values. A further 1,007 native row/activation/normalization graphs require 2,909,056 exact output bytes per host. Shared scalar/SSE2 row math and the audited system `erff` boundary preserve the exercised dynamic-UCRT/glibc profiles; other providers and remaining ISA policies stay unverified. Independent raw-word gates check 502,058 `erff` and 502,058 scalar quick-GELU inputs per host. Earlier failures and fixed conversion bounds are retained.
- **Shared flash and tensor adapter:** 3,746 original graphs check 7,060,000 raw default/forced output words per compiler profile, including source NaN operand selection. One-chunk, tiled and split-KV paths retain original geometry, physical masks, row maps and real worker partitions. The tensor adapter separately checks 114 new admitted graphs plus 896 reused sealed cases, with 3,032,980 exact output words per profile. Unsupported domains remain explicit; the tensor DAG has no aggregate retained-memory budget.
- **Checkpoint foundation:** raw/indexed Safetensors reading, bounded positioned reads and a retained-directory capability boundary. Independent original file/index corpora and safe checked path-policy differences are recorded in [the checkpoint crate](crates/checkpoint/README.md). Other architecture converters and remaining format boundaries stay open.
- **Actual devices:** 138 CUDA dispatches on RTX 4090 with zero recorded error; OpenCL kernel cases on RTX 4090 and AMD gfx1036 with recorded numerical bounds.
- **C ABI:** At that checkpoint, 191 of 255 frozen exports, plus four diagnostic extensions, were exercised through independent frozen-header and model harnesses. The latest CPU sampler corpus covers 165 scenarios and 42 real contexts / 585 generated tokens. The BPE/FIM/raw-byte harness compares 1,505,576 vocabulary records and 7,598 tokenization cases. The context/memory extension adds 80 scenarios, 400 checks, 61,440 continuation logits and 1,223,532 native state bytes. The new family getter corpus adds 347,066 vocabulary records and 27,508 tokenization cases; physical metadata adds 23,903 records and 480 outputs across 176 cases. Raw native state restore/reserialization is byte-exact on both platforms; generated Linux floats retain the predeclared numerical budget. Automatic shard C loading adds 96 cases per native profile, including 53 accepted and 43 rejected loads, aggregate model getters and 163,840 exact continuation logit words. Explicit split-list C loading adds 104 Windows / 106 GNU cases, including raw Unix filenames, with 172,032 / 180,224 exact continuation logit words. The two byte-oriented filename helpers match 7,807 original calls and 506,563 complete output-buffer bytes per profile; [SPLITS.md](crates/ffi/SPLITS.md) records their truncation, signed-number and callback ownership contracts. The four static enum helpers reproduce 73,057 original file-type values, nine enum names and 1,542 byte-string load modes; their 30 original abort cases become checked errors, as documented in [ENUMS.md](crates/ffi/ENUMS.md). Prior core/sampler/WPM fixtures remain passing. The current [coverage inventory](crates/ffi/COVERAGE.md) retains all eight absent exports and the remaining semantic restrictions.
- **Server:** 37 real-model generation requests plus 283 embedding HTTP transcripts per original build (256 tiny-model, 27 pretrained BGE). Both verified profiles require exact corresponding F32 bits and all seven base64 vectors. Direct normalization checks cover another 4,608 values. Real TCP lifecycle tests cover configured load bounds, limits, provider failures, disconnects, shutdown and complete deadline errors.

The original MSVC and GNU builds do not use identical row arithmetic: the verified GNU build enables source SSE2 exponentiation, reductions and variance, while the verified MSVC build uses scalar rows. Separate expected captures preserve those differences. Test expectations select their profile independently of runtime dispatch. The model runtime also preserves 256-column attention padding and physical cache holes. None of these results establishes every CPU instruction-set path.

Evidence and qualifications are under [evidence/2026-11-09](evidence/2026-11-09). These results establish the exercised configurations, not all hardware, models or performance parity. Pretrained weights are not distributed in this repository; the tokenizer-only GGUF is test data with provenance.

## Verify

```sh
cargo fmt --all -- --check
cargo test --workspace --all-targets --offline --locked
cargo test --workspace --all-targets --release --offline --locked
cargo clippy --workspace --all-targets --offline --locked -- -D warnings
cargo xtask audit
cargo xtask fixtures
```

Windows and Linux CI run these checks and separately download hash-pinned F32, Q4_0, BPE/Q4_K_M, Qwen2, Qwen3, Mamba-1 and BERT model inputs for explicit numerical comparisons. CI also downloads exact official GPT-2 and Pythia Safetensors revisions and uses the Rust xtask preparer to reconstruct pinned GGUF inputs for inference tests.

Separate conversion jobs verify every input hash and complete outputs for Qwen, GPT and BLOOM checkpoints. BLOOM jobs reuse their exact F16/Q8_0 conversions for independent inference replay on each host. The general converter admits Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BLOOM, Falcon, StarCoder and StarCoder2. StarCoder has unchanged pretrained F16/Q8_0/auto complete-file acceptance; StarCoder2 conversion has synthetic weights and pinned official sidecar acceptance, independently of its pretrained GGUF runtime coverage. Falcon's independent synthetic conversion corpus and official tokenizer sidecars do not establish pretrained Falcon weight acceptance. Its GPT test preserves the original official GPT-2 rejection and separately verifies the disclosed test derivative; [GPT_HF_CONVERSION.md](docs/GPT_HF_CONVERSION.md) records that boundary. Hardware/vendor tests retain their own explicit prerequisites. Exact numerical claims apply to the corresponding measured original profiles; portable error budgets remain available for unverified targets. See [inference fixtures](fixtures/inference/README.md), [quant validation](crates/quant/README.md), [tensor validation](crates/tensor/README.md) and [xtask commands](xtask/README.md).

With the frozen source extracted separately:

```sh
cargo xtask verify-source /path/to/llama.cpp-master
```

`cargo xtask release-check` deliberately fails while any mandatory full-port gate remains open. The active list has twenty open gate rows across twelve ownership tracks; these are different counts.

## Provenance and license

The integration reuses and audits the supplied T05-T08 and T09-T12 Rust progress archives. Their hashes and imported files are recorded in [docs/imports/manifest.json](docs/imports/manifest.json). Historical reports are preserved under `docs/imports/`; current claims require current evidence.

Original project code is MIT licensed. Original ggml attribution is retained in [LICENSE-UPSTREAM](LICENSE-UPSTREAM) and [NOTICE](NOTICE). Component notices identify separately licensed adapted code, including the native tied-sort helpers. Fixture provenance records additional source identities and qualifications.
