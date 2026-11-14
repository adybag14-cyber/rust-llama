# StarCoder and StarCoder2 CPU inference

The frozen source models `src/models/starcoder.cpp` and
`src/models/starcoder2.cpp` now execute through native Rust model graphs and
checked GGUF loaders. These are two independent model identities; state files
and checkpoints cannot be interchanged between them or with GPT-2.

## Graph and loading contract

| Feature | StarCoder | StarCoder2 |
| --- | --- | --- |
| GGUF architecture | `starcoder` | `starcoder2` |
| Position input | Required learned `position_embd.weight`; request context is capped to the training table | Full-head, half-pair NeoX RoPE; positive finite frequency base |
| Attention | Required fused Q/K/V matrix and bias | Separate or fused Q/K/V; native optional-bias precedence |
| Normalization | Required affine LayerNorm weight and bias at both sublayers and final output | Same affine LayerNorm contract |
| FFN | Required biased up and down matrices, sequential source GELU | Same sequential GELU contract |
| Output | Explicit output matrix or tied token embeddings | Explicit output matrix or tied token embeddings |
| KV storage | F32, causal sequential attention with MHA, GQA or MQA | F32, causal sequential attention with MHA, GQA or MQA |

The original StarCoder2 source comments call attention-output and FFN biases
optional, but its tensor-loading flags make those three biases required. The
Rust loader preserves the executable requirement, as confirmed by the original
missing-bias captures. Output biases are not loaded by either original graph and
are rejected as unconsumed tensors.

StarCoder2 fused QKV prefers the fused bias. With no fused bias, all three
separate biases are used together; an incomplete group is retained but inactive.
Separate Q/K/V matrices apply each available bias independently. Required
normalization and active bias tensors require F32 storage because the original
CPU ADD/MUL path requires F32 operands. An inactive incomplete fused bias group
does not gain an artificial active-vector restriction.

Both loaders accept the matrix formats supported by the existing frozen scalar
CPU dot implementation, retain packed storage for native activation conversion,
validate dimensions before execution and honor the decoded-weight budget.
The structural corpus exercises F32, F16, BF16, Q4_0, Q8_0 and Q4_K matrices.
This does not claim independent quality validation for every format/model pair.

The checked shared loader handles complete and sharded GGUF inputs, metadata
types, allocation budgets, tokenizer behavior and tensor-name uniqueness.
Tokenization follows the file's native GGUF vocabulary. Models can be used
through the existing CLI generation/logits commands and Rust inference facade.

## Context, state and error behavior

The strict Rust `Context` owns transactional sequence caches. Both architectures
participate in its reservation accounting, multi-sequence batches, fork,
truncate, checkpoint, serialized state, direct-embedding input and cancellation.
A failed batch leaves all sequences and their reserved state unchanged.
Fingerprints include architecture, configuration, packed representation,
normalization/bias vectors and position or rotary parameters.

Model weights remain immutable. Retained decoded rows, original packed matrix
bytes, execution scratch and KV allocations have the same separate resource
boundaries as other CPU models. The default decoded-weight budget is 2 GiB;
larger pretrained models require an explicit higher budget.

This addition does not add either architecture to the distinct C-compatible
dense-Llama `NativeContext` or to the CUDA runtime. Noncausal execution, per-layer
dimension arrays, scaled/partial rotary embeddings, nonstandard independent head
widths, nonzero sliding-window configuration, MoE, pooling and NextN/MTP remain
explicit unsupported cases.

## Independent structural acceptance

Before the Rust implementations were written, the frozen original GGUF writer
and quantizers produced 72 small model cases. The original model-loading and
C inference API were executed independently on Windows MSVC and Linux GNU.
The retained corpus contains:

- 43 successful model configurations, including tied/untied output, multiple
  layer/head counts, learned/rotary positions, six matrix storage formats,
  grouped-query attention, optional QKV biases and fused/separate QKV.
- 29 invalid or explicitly unsupported configurations, retaining original
  exits and original successful observations where Rust scope is narrower.
- Three text prompts and one explicit-embedding sequence per successful model,
  with full vocabulary logits, final token embeddings, token bytes, greedy IDs
  and generated output.

Each matching host/profile replay compares **826,444 unique output F32 words
and 516 greedy IDs** for the 43 admitted configurations. Windows MSVC and Linux
GNU debug/release passes are exact at the bit level. The original portable
numerical criterion remains `2e-4 + 2e-5 * abs(reference)` for unverified target
profiles; exact native equality is enforced on the two measured targets.

The ordinary suites also check typed metadata rejection, checked weight/context
limits, cross-model state identity, finite-weight numerical failure, rollback,
checkpoint/fork/truncate and strict state allocation. Expected numerical values
come only from the original implementation. Rust structural constructors supply
inputs for resource tests and never generate the original comparison outputs.

[Fixture provenance](../fixtures/inference/starcoder/provenance.json) records
source/library/executable hashes and capture order.
[The fixture manifest](starcoder-inference-fixture-manifest.tsv) pins all
committed structural inputs and outputs. The [component acceptance record](../evidence/2026-11-14/starcoder-inference/acceptance.json)
and command logs are stored under `evidence/2026-11-14/starcoder-inference`.

## Pretrained acceptance

The following independent pretrained comparisons pass in release mode on both
Windows MSVC and Linux GNU. Every compared F32 word and greedy token ID matches
its corresponding original host profile exactly. Each model uses three short
prompts, a 128-token context limit, one-token CPU decoding, F32 KV, complete
vocabulary logits and final prompt embeddings. The generation facade is also
run independently and checks the complete generated bytes and finish reason.

| Input | Compared F32 words per profile | Greedy IDs | Full GGUF bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Tiny StarCoder F16 | 1,041,408 | 12 | 342,950,656 | `abdd880356fe55d9e951e5da4b198639d15331d1e10c0015795d10caaa273602` |
| Tiny StarCoder Q8_0 | 1,041,408 | 12 | 195,126,016 | `b2620b0de962f582218530233849064c088c56e86344a7e49afb53c7d5957906` |
| StarCoder2 3B Q4_K_M | 1,069,056 | 12 | 1,848,976,448 | `d8fb39287a463549b80d97473b0a7595c3a5a6da3ae2604ca33906a1a43f7175` |

Tiny StarCoder comes from
[`bigcode/tiny_starcoder_py`](https://huggingface.co/bigcode/tiny_starcoder_py/tree/8547527bef0bc927268c1653cce6948c5c242dd1),
revision `8547527bef0bc927268c1653cce6948c5c242dd1`. The frozen original converter
produced the F16 and Q8_0 inputs; separate original Windows and GNU conversions
are equal as complete files. The original Safetensors input is pinned by size
and SHA-256 in the retained provenance.

The StarCoder2 quantized input comes from
[`second-state/StarCoder2-3B-GGUF`](https://huggingface.co/second-state/StarCoder2-3B-GGUF/tree/7fca3e2da2ce31df411461e2cb9cae2d2b492f35),
revision `7fca3e2da2ce31df411461e2cb9cae2d2b492f35`. This GGUF contains no
sliding-window metadata; its short-context result establishes parity with the
frozen original graph for this file. It does not establish long-context
sliding-window behavior or parity with every upstream Transformers option.

The 3B model uses an explicit 16 GiB decoded-weight allowance. The test also
checks that the default 2 GiB allowance rejects it before decoding weights.
Packed bytes, tokenizer, KV and execution scratch remain separate allocations.

[Pretrained provenance](../fixtures/inference/starcoder/pretrained-provenance.json)
retains publisher and derivative revisions, input hashes, both native inference
profiles, original converter reports and the pinned official license reference.
Both model publishers declare `bigcode-openrail-m`. Model weights and original
reference implementations remain outside the product repository. The model
licenses are distinct from the synthetic fixture MIT notice.

Pretrained original captures were made after architecture implementation and
before Rust pretrained comparison. They never derive expected values from Rust.
The earlier structural captures remain independently dated before implementation.

Set `LLAMA_TEST_STARCODER_F16_MODEL`, `LLAMA_TEST_STARCODER_Q8_MODEL` and
`LLAMA_TEST_STARCODER2_MODEL` to the exact pinned GGUF files and run:

```sh
cargo test --release --offline --locked -p llama-inference --test starcoder -- --ignored --nocapture --test-threads=1
```

The three tests are ignored by ordinary offline test runs because their large
external model inputs are not vendored. They check each complete model's size
and SHA-256 before loading. The ordinary structural suite runs without those
model downloads.

These nine short-prompt executions establish the measured pretrained arithmetic,
tokenization and generation behavior. They do not establish broad code
benchmark quality, performance, long-context, device, native C-context or full
architecture parity. The complete-port release gates remain open.
