# Pinned pretrained fixture preparation

The `xtask prepare-pretrained-fixture` command reproduces the exact GPT2 and Pythia GGUF inputs used by the independent inference comparisons. It accepts only the two official checkpoint files and the two recipes recorded here. This is test-data preparation. It does not add a general GPT2 or GPT-NeoX checkpoint converter.

The Rust implementation lives in `xtask/src/pretrained.rs`. It uses the existing native Rust SafeTensors parser, SHA-256 implementation, and F16 conversion. CI downloads ordinary pinned official SafeTensors files and invokes this command; neither Python nor a llama.cpp runtime is required to prepare or test the files.

## Reproduce the inputs

Download each `model.safetensors` from its exact revision below, then run from the repository root:

```text
cargo run --release --offline --locked -p xtask -- prepare-pretrained-fixture gpt2-124m-f16 GPT2.safetensors gpt2-124m-f16.gguf
cargo run --release --offline --locked -p xtask -- prepare-pretrained-fixture pythia-14m-f16 PYTHIA.safetensors pythia-14m-f16.gguf
```

The destination must be absent and its parent directory must already exist. Use a caller-owned directory with hard-link support. Preparation refuses an existing destination and does not replace a destination created concurrently. The temporary name is resolved during publication, so the directory must not be modified by an untrusted process.

Set `LLAMA_TEST_GPT2_MODEL` and `LLAMA_TEST_GPTNEOX_MODEL` to the resulting files for the ignored inference tests. The CI workflow supplies both variables alongside its existing model inputs and continues to run all ignored inference/server tests.

| Fixture | Official repository and revision | Official bytes | Prepared GGUF bytes |
| --- | --- | ---: | ---: |
| GPT2 124M | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e), `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 548,105,171 | 252,470,528 |
| Pythia 14M | [EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac), `cf967c0a9a04383db6f7b1108d86b2962634b4ac` | 28,143,920 | 29,929,440 |

Official source SHA-256:

- GPT2: `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
- Pythia: `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1`.

Complete prepared GGUF SHA-256:

- GPT2: `c7cb2936027231a4f2159264f9a635742d0d3bbbf1ddfd82074337b0d1c6e3a7`.
- Pythia: `4fefeed58ed27e1fc93b31145494ba87ce539cdf74cef809122b3c61bb0570a8`.

## Independent origin of the recipes

The source baseline is the user-pinned llama.cpp archive with SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. The original converter and original C API produced both existing GGUF files and their MSVC/GNU inference expectations before these recipes or the Rust preparer existed. Their original acquisition, conversion, and inference provenance is retained in `fixtures/inference/gpt2/provenance.json` and `fixtures/inference/gptneox/provenance.json`.

An independent read-only audit applied the source transformations from the frozen converter to all official source tensors and compared every target payload byte with the existing original GGUF files. All 224 learned output tensors matched. Source and original-GGUF whole-file hashes were checked before and after that audit using retained file handles. The audit records source/target names, shapes, dtypes, byte ranges, transform parameters and individual payload hashes in `evidence/2026-11-11/pretrained-preparation/audit`.

The `*-header.bin` files are exact prefixes of those independently produced GGUF files, from byte zero through the aligned start of tensor data. They contain metadata, tokenizer vocabulary/merges, tensor descriptors and alignment padding; they contain no learned tensor payloads. Metadata strings such as GPT2's original directory-derived name `Official Hf Without Static Masks` and Pythia's `Official Hf` are preserved. The command deliberately reconstructs these particular original artifacts without regenerating arbitrary metadata or tokenizers.

The `*-recipe.json` files are data extracted from that audit. They describe every admitted source tensor and its exact output payload. The embedded recipe checksum, header checksum, per-tensor hashes, and complete GGUF checksum are all checked by Rust. The independent source audit and extraction scripts remain outside the maintained repository; `provenance.json` records their paths and hashes.

## GPT2 qualification and transformations

The official GPT2 checkpoint has 160 F32 tensors. Twelve are serialized static causal masks named `h.0.attn.bias` through `h.11.attn.bias`, each with shape `[1, 1, 1024, 1024]`. The frozen original converter cannot directly accept those buffers. Its successful reference conversion used a separately derived checkpoint that omitted exactly these twelve masks while preserving all 148 learned tensors unchanged.

This preparer reads and verifies the original unmodified official checkpoint. It admits exactly that same omission: the source names, layer range, F32 dtype, shape and payload hash of all twelve masks must match the sealed recipe. The independent audit also verified that every mask value is exactly the lower-triangular causal 0/1 pattern. Each mask's SHA-256 is `e16420835229addd98c66a061dd8a9bea9b405eff3738f4bb00ff603f98400d1`. No learned tensor is omitted.

Each of the twelve layers has four transposed Conv1D matrices: attention QKV, attention output, feed-forward up, and feed-forward down. That gives 48 transposes. Token and position embeddings are not transposed. The frozen converter specifically forces the position embedding matrix to F32 even with F16 output; vectors and norms are also F32. The other admitted matrices are F16. The resulting 148 tensors comprise 99 F32 tensors and 49 F16 tensors, with 250,695,168 payload bytes. The input has no separate learned output matrix; the existing model graph uses tied token/output weights.

## Pythia transformations

Pythia has 76 F16 tensors and needs no omitted buffers. It has six layers, hidden width 128, four attention heads, and head width 32. For each layer, the fused QKV matrix `[384, 128]` and its bias `[384]` are reordered from source rows arranged as `[head, Q/K/V, position]` into all Q rows, all K rows, then all V rows. These twelve transformations perform no rotary permutation.

Vectors and norms are promoted to F32, while matrices remain F16. The resulting 76 tensors comprise 50 F32 tensors and 26 F16 tensors, with 28,155,904 payload bytes.

## Bounds and publication

The preparer checks the complete source size and SHA-256 before parsing or transforming data. The source handle remains open throughout preparation, and its complete identity is checked again before publication. Parsed names, shapes, dtypes, offsets and tensor counts must match the captured recipe; every source tensor must be accounted for exactly once.

The fixed limits are 256 KiB per recipe, 2 MiB per header prefix, 600 MiB for source files, 300 MiB for output files, and 256 source tensors. Identity transformations stream chunks of 8,192 elements with two fixed 32 KiB buffers. Transposes and QKV reordering read at most 16 MiB for a single source tensor. These are bounds on the preparer's admitted data and payload buffers, not a claim about total process memory or an arbitrary checkpoint conversion API. Parser nodes, text, depth, dimensions and reads also have explicit limits.

Checked arithmetic covers tensor sizes, transform indices, ranges, alignment and output byte accounting. Each source and transformed tensor payload is hashed as it is consumed or written. A private temporary output is flushed, synchronized, and hashed in full before hard-link publication. An existing or concurrently created destination causes an error without replacement; unsupported hard-link filesystems fail explicitly. Temporary cleanup is best effort on exit. The directory ownership assumption described above applies.

## Validation and licenses

The maintained evidence directory records both compiler profiles. Debug and release tests exercise corrupt input rejection, retained source rechecking, exact embedded layouts, output budget checks, and destination collision behavior. The complete file reconstruction and unchanged pretrained inference tests are then run on Windows/MSVC and GNU/Linux against the independent native expectations.

Model and tokenizer data remain governed by the included GPT2 MIT and Pythia Apache-2.0 notices. The frozen llama.cpp MIT notice is included as `LICENSE.llama.cpp.txt`. The repository's Rust implementation license does not replace these source notices.
