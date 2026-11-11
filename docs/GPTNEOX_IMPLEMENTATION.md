# GPT-NeoX decoder implementation and evidence

The dense causal GPT-NeoX graph from the frozen `src/models/gptneox.cpp` now
executes through `llama-inference::Model` and the owned Rust decoder
`llama-model-runtime::Context`. It adds partial NeoX rotary positions and
both parallel and sequential residual calculation to the native Rust runtime.

The frozen ZIP SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`;
the original GPT-NeoX source SHA-256 is
`fa5f3d40e39d76a47e4d56f7eeaed5917f6f68be6953d63afa114a6e0b3a7a9b`.
[Fixture provenance](../fixtures/inference/gptneox/provenance.json) records the
reference programs, shared libraries, requests, generator, and packed data.
[Validation evidence](../evidence/2026-11-11/gptneox/summary.json) retains the
commands, results, source hashes, and original boundary failures.

## Graph and loader

The graph uses token embeddings, affine LayerNorm, fused Q/K/V with bias,
partial NeoX RoPE on Q and K, causal attention, a biased attention projection,
and a biased GELU feed-forward network. V remains unrotated. The rotary prefix
pairs its two halves; each head's remaining channels are unchanged.

The residual order follows the original floating-point associations:

| Configuration | FFN normalization input | Final layer calculation |
| --- | --- | --- |
| Parallel residual | Original layer input | `(ffn_output + input) + attention_output` |
| Sequential residual | `attention_output + input` | `ffn_output + (attention_output + input)` |

Final affine LayerNorm produces the last hidden embedding. A separate required
output matrix produces logits. This architecture has no tied-output fallback
or output-bias path in the frozen loader.

The loader checks mandatory BOOL residual metadata, exact UINT32/FLOAT32
hyperparameter types, required tensors, finite vector values, matrix shapes,
head divisibility, uniform GQA/MQA, and decoded weight limits. It admits positive
even rotary prefixes at most the even head width, with unscaled frequencies.
F32 active norm/bias vectors follow the source CPU operator requirements.
Requested context is capped at the trained context. Extra tensors cause an
explicit unsupported error.

The implementation reuses the existing Rust matrix, normalization, attention,
GELU, and NeoX rotation primitives. It does not change shared math or link an
original inference library. Its context uses the shared staged attention
memory and checked reservation logic. GPT-NeoX participates in the complete
allocation-failure suite. Additional tests cover state identity, fork,
truncate, restore, explicit embedding input, cancellation, and transactional
rollback when finite weights overflow in both residual modes.

## Independent reference corpus

Original native outputs were captured before the Rust graph implementation
and numerical comparison. The expected logits come exclusively from the frozen
implementation. The CPU captures use one worker, F32 KV, and flash attention
disabled. MSVC and GNU maintain separate original result bytes.

| Corpus | Executable coverage | Rows per profile | Logits per profile | Greedy IDs per profile |
| --- | --- | ---: | ---: | ---: |
| Synthetic original-writer models | 16 supported configurations | 496 | 253,952 | 192 |
| Official Pythia-14M F16 | Three prompts, eight generated tokens each | 33 | 1,660,032 | 24 |

The 24 synthetic cases cover partial/full rotary prefixes, two residual orders,
different frequency bases and normalization epsilon, explicit/default metadata,
GQA/MQA, one/two/three layers, and F32/F16/Q4_0/Q4_K matrices. Six cases cover
explicitly unsupported configurations; two cover absent required tensors.
The corpus compares full vocabulary logits, token IDs, ordinary and special
token pieces, detokenization, greedy IDs, and generated text.

All replayed logits matched the corresponding original profile bit for bit
on both measured x86-64 hosts, with zero maximum absolute error. The fixed
portable test bound is `2e-4 + 2e-5 * abs(reference)`; the measured Windows
MSVC and Linux GNU profiles additionally require exact bits. These results
do not establish other compiler/CPU profiles, other NeoX model sizes,
quantized pretrained Pythia, GPU execution, or long-context performance.

## Native boundary observations

The initial source-only capture attempted a zero rotary dimension. The frozen
graph aborted because its unused position input had no backend buffer.
That first capture log is retained. The original RoPE CPU tail copies channels
in pairs; odd head widths can copy past a head's boundary. Rust rejects both
zero rotation and odd head widths explicitly. The original odd-head fixture
happened to finish; its output does not establish a safe source domain.

This scope adjustment occurred before Rust implementation or evaluation and
did not change the fixture model bytes.
[scope-revision.json](../fixtures/inference/gptneox/scope-revision.json) records
the decision. Original F16 active norm/bias cases also abort, while missing
output/norm tensors fail loading. Rust returns checked errors for those inputs.
The original accepts noncausal and per-layer head-array fixtures; those modes
remain explicitly deferred in this Rust implementation.

## Official pretrained model

The input is `EleutherAI/pythia-14m`, pinned to revision
`cf967c0a9a04383db6f7b1108d86b2962634b4ac`.
The unchanged source `model.safetensors` has 28,143,920 bytes and SHA-256
`116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1`.
[Model origin](../fixtures/inference/gptneox/model-origin.json) records all
pinned download URLs and input hashes.

The frozen original converter consumed this official checkpoint without an
input adaptation. The F16 GGUF has 29,929,440 bytes and SHA-256
`4fefeed58ed27e1fc93b31145494ba87ce539cdf74cef809122b3c61bb0570a8`.
The model has six layers, width 128, four heads, a 32-channel head width,
and an 8-channel rotary prefix. The original conversion and both reference
executables use these same learned weights.

The model weights and C/C++/Python reference harnesses remain external to the
product repository. The repository contains the independently captured
expected results, input provenance, and the full upstream
[Apache-2.0 license](../fixtures/inference/gptneox/LICENSE.PYTHIA.txt).
Synthetic graph coverage and this separate pretrained replay are reported
separately.

## Reproduction and validation

The ordinary synthetic tests reconstruct their GGUF inputs from pinned fixture
bytes and require no model download:

```text
cargo test --offline --locked -p llama-inference --test gptneox -- --nocapture
cargo test --release --offline --locked -p llama-inference --test gptneox -- --nocapture
cargo test --offline --locked -p llama-model-runtime
cargo clippy --offline --locked -p llama-inference -p llama-model-runtime --all-targets -- -D warnings
```

Set `LLAMA_TEST_GPTNEOX_MODEL` to the exact F16 GGUF described above to run the
deliberately ignored real-model replay:

```text
cargo test --release --offline --locked -p llama-inference --test gptneox official_pythia -- --ignored --nocapture
```

The test checks the model's byte count and SHA-256 before loading it. The retained
Hermes model is
`C:\Users\adyba\rust-llama-reference\models\pythia-14m-f16.gguf`
or `/mnt/c/Users/adyba/rust-llama-reference/models/pythia-14m-f16.gguf` under WSL.

Both hosts passed all four ordinary GPT-NeoX tests in debug and release, the
separate pretrained replay, formatting, strict Clippy, and the full model-runtime
debug suite (82 tests on Windows and 83 on GNU). The unchanged GPT-2 synthetic
and official-model tests were rerun in release on both hosts and retained their
exact original-profile results.

## Remaining scope

The high-level complete-sequence embedding API still requires BERT.
Projection/output adapter hooks are present; token embedding adapter targets
are not provided. The native C-compatible context has a separate Llama graph
scope, so this work does not claim GPT-NeoX for that interface. Per-layer
arrays, noncausal attention, scaled/extended rotary modes, PRISM/NextN,
nonstandard key/value dimensions, pooling, sliding-window/expert modes,
and unknown tensors fail explicitly. All full-release gates in
`docs/plan.md` remain open.
