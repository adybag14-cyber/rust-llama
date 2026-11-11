# BLOOM strict CPU inference

This package adds a native Rust BLOOM decoder to `llama-model-runtime` and GGUF loading/generation to `llama-inference`. The implementation follows `src/models/bloom.cpp` in the supplied frozen llama.cpp archive, identified by SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. The archive has no inferred Git commit. Its BLOOM model file has SHA-256 `4323ced163de10039f8dbe35402e2cfb3e4b70e8489a3db886e6588a2928558f`.

The executable slice is the existing strict, causal, contiguous F32-cache `Context`. It supports real next-token logits, greedy generation, explicit embedding input, final hidden-state output, independent sequences, transactional batches, checkpoints, forks and exact suffix rewinds. The additive `Model::Bloom` arm uses the existing state format and state-reservation ledger. The model fingerprint includes BLOOM configuration and every active weight, including input normalization and affine biases.

The separate `NativeContext` compatibility cache still rejects BLOOM explicitly. This change makes no BLOOM C-ABI decoder, GPU, flash-attention, low-precision cache, noncausal, or per-layer-array execution claim. The registry marks only the defined execution slice; registry presence is not general llama.cpp parity.

## Graph and numerical order

Input tokens first select a row from `token_embd.weight`. Explicit embedding rows enter the same point. Both paths apply the required affine input LayerNorm before any transformer block. Each block performs:

1. Affine LayerNorm, then the fused Q/K/V projection and its required bias.
2. Causal attention with the source's fixed maximum ALiBi bias of `8.0`, followed by output projection and bias.
3. Residual addition, then affine FFN LayerNorm.
4. FFN up projection and bias, source GELU, down projection and bias, and the second residual addition.
5. Existing control-vector application.

A final affine LayerNorm produces the embedding output. Logits use `output.weight` when present and the token embedding matrix when it is absent. There is no learned position table, rotary transform or optional output bias in this source architecture.

ALiBi retains the original power-of-two head partition, including the second slope series for non-power-of-two head counts. The original mask is the negative integer distance between key and query positions, converted to F32; the diagonal is positive zero. Query/key products are rounded to F32 and accumulated in F64, then converted to F32. Attention scaling is a separate F32 multiplication before the F32 slope-times-mask addition. Softmax uses the existing native-profile implementation over the full row padded to a multiple of 256; probability/value products again round to F32 before F64 accumulation. The shared attention and normalization implementations were not changed for this port.

The native Windows MSVC and GNU/Linux paths have separate original references. The tests require exact F32 words on the measured x86-64 profiles. A predeclared `2e-4 + 2e-5 * abs(expected)` budget remains an additional finite-error check, and is the portability check for unmeasured targets. Token IDs and token-piece bytes are always exact. The string generation interface retains its existing UTF-8 replacement behavior.

## Loading and bounds

Required scalar metadata includes block count, context length, embedding width, feed-forward width, head count and LayerNorm epsilon. KV-head count defaults to head count. Metadata uses the frozen scalar types rather than silently accepting alternate integer/float encodings. Layer count is bounded to `1..=512`; dimensions, head grouping, tensor shapes, context limits, finite positive epsilon and size products are checked. The requested context is capped by the model's declared training context.

The loader checks its decoded F32-weight budget before reading weight payloads. That budget counts decoded model elements and does not describe total process RSS: retained packed weights, tokenizer storage, execution scratch and output rows are separate. The official 560m tests request a four-GiB decoded-weight budget explicitly.

| GGUF tensor | Required source shape |
| --- | --- |
| `token_embd.weight` | `[width, vocabulary]` |
| `token_embd_norm.weight`, `.bias` | `[width]` |
| `output_norm.weight`, `.bias` | `[width]` |
| `output.weight` | Optional `[width, vocabulary]`; absent means tied token weights |
| `blk.N.attn_norm.weight`, `.bias` | `[width]` |
| `blk.N.attn_qkv.weight`, `.bias` | `[width, width + 2 * kv_width]`, `[width + 2 * kv_width]` |
| `blk.N.attn_output.weight`, `.bias` | `[width, width]`, `[width]` |
| `blk.N.ffn_norm.weight`, `.bias` | `[width]` |
| `blk.N.ffn_up.weight`, `.bias` | `[width, feed_forward]`, `[feed_forward]` |
| `blk.N.ffn_down.weight`, `.bias` | `[feed_forward, width]`, `[width]` |

Active normalization and bias vectors require F32 storage, matching the original CPU ADD/MUL domain. Unsupported per-layer arrays, noncausal attention, nonstandard head lengths, sliding windows, experts, pooling, next-token-prediction extras and unconsumed tensors return checked errors. Invalid or nonfinite configuration, weights, tokens and embeddings also fail explicitly. All fallible strict-state growth uses the existing ledger before execution; cancellation or a late numerical error cannot commit partial sequence state.

## Independent numerical evidence

All original capture runs and fixture packing finished before the first maintained Rust BLOOM source file was written. External C++ callers invoke the unchanged original C API, original GGUF writer and original scalar quantizers; no Rust-generated numerical results are used as expected values. Source, compiler, library, converter, driver and input identities are retained in [the fixture provenance](../fixtures/inference/bloom/provenance.json). [The fixture manifest](bloom-inference-fixture-manifest.tsv) declares every maintained fixture byte count and SHA-256.

Each original compiler profile has 31 tiny cases: 18 executable variants, 8 invalid-input cases and 5 intentionally unsupported domains. Executable cases cover tied and untied output, F32/F16/BF16/Q4_0/Q8_0/Q4_K matrices, grouped/multi-query attention, 1/3/4/5 heads, odd head width, one/two/three layers, epsilon changes, implicit KV heads, ignored generic ALiBi metadata, multiple prompt lengths and arbitrary direct embedding rows. Original source assertions and rejected cases are retained as failures rather than converted into numerical successes.

The maintained tiny comparison checks 1,062 rows, 346,644 F32 values and 216 greedy IDs per profile. Separate tests exercise typed metadata rejection, decoded-weight budget preflight, token/context boundaries, state fingerprints, checkpoint/rewind/fork behavior, cancellation at multiple boundaries, finite-weight execution overflow, and injected state-allocation failures.

### Actual trained BLOOM checkpoint

The trained proof uses untouched official `bigscience/bloom-560m` at revision `ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971`. Its `model.safetensors` is 1,118,459,525 bytes with SHA-256 `a8702498162c95d68d2724e7f333c83d7be08de81cfc091455c38730682116d3`. It has 24 blocks, width 1024, 16 heads and a 250,880-token vocabulary. Acquisition URLs and full input identities are in [model-origin.json](../fixtures/inference/bloom/model-origin.json); the full [BigScience BLOOM RAIL 1.0 license](../fixtures/inference/bloom/LICENSE.BLOOM.txt) and [original model card](../fixtures/inference/bloom/MODEL_CARD.BLOOM.md) accompany derived numerical fixtures.

Original Windows and GNU converters independently produce these identical complete GGUF files:

| Conversion | Bytes | SHA-256 |
| --- | ---: | --- |
| F16 (also source auto) | 1,131,699,072 | `ac136500f37dbada4866b390123f14c9fbe14d215aabf8d93e96fc58772780bf` |
| Q8_0 | 607,738,752 | `97eec92fef8daac7afff0e0a0e71bb822e346ac267d84e05290e2b01f8821fdc` |

The two ignored Rust tests require these complete hashes and use `LLAMA_TEST_BLOOM_F16_MODEL` and `LLAMA_TEST_BLOOM_Q8_0_MODEL`. They compare full vocabulary logits, final embeddings, tokenizer bytes and four greedy steps for each of three prompts. They also compare original rows after direct embedding input, continuation, checkpoint/state restore, rewind and forks. Checkpoint payloads and raw large native result files remain external; only compact lossless reference outputs are maintained here.

Run the trained comparisons with the pinned paths set:

`cargo test --release --offline --locked -p llama-inference --test bloom official_bloom_560m -- --ignored --nocapture --test-threads=1`

The domain validation receipt in `evidence/2026-11-11/bloom-inference` retains successful checks and failed earlier attempts separately. The historical GNU capture failures were caused by `-llama` (requesting `liblama`) instead of the correct `-lllama` (requesting `libllama`); an initial symlink diagnosis was explicitly corrected. Original baseline/build files were not modified. Rust development failures include a test-only constructor error, an outdated registry-count assertion and strict Clippy iterator findings; none is represented as a successful attempt.

## Recorded domain validation

The final `full-v3` run succeeded on both hosts, with the same 558 selected source/fixture/Cargo inputs unchanged through execution and packaging. The selected-input manifest SHA-256 is `f80fb153b46fc7cfeba4407a717303c34dd971e7aa6b594d2740f30cc539d656`. [The retained summary](../evidence/2026-11-11/bloom-inference/summary.json) links to every successful and failed attempt and pins all raw command logs.

| Host | Debug | Release | Separate trained tests |
| --- | ---: | ---: | ---: |
| Windows/MSVC | 124 passed, 12 ignored | 124 passed, 12 ignored | 2 passed |
| GNU/Linux | 126 passed, 12 ignored | 126 passed, 12 ignored | 2 passed |

Both hosts passed strict all-target Clippy and rustfmt. Each host's trained comparisons checked 13,074,432 F32 words bit for bit, 80 output rows and 24 greedy IDs across F16 and Q8_0. The ordinary tiny corpus also matched every checked word in both debug and release. The two trained tests are ignored in ordinary runs and were explicitly run in release mode. Other architecture-specific ignored tests were not re-executed for this isolated package; integration gates remain the responsibility of the full combined tree.

The bounded package postflight passed the Rust-source audit, all fixture integrity declarations and git diff whitespace checks. A separate read-only [source review](../evidence/2026-11-11/bloom-inference/review/runtime-review-v1.json) found no blocking issue and records exact reviewed hashes; the final publication guard rechecked those bytes and the complete selected validation input set.
