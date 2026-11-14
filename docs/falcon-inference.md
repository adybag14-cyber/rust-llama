# Falcon runtime and GGUF contract

This module executes the frozen `src/models/falcon.cpp` graph in native Rust.
It is a distinct architecture from `falcon-h1`. The original graph uses full
NeoX rotary positions; it does not implement the ALiBi/nonparallel/bias
configuration used by the later Falcon-RW-1B checkpoint.

## Executable graph

A layer applies affine LayerNorm to its input. That primary normalized input
always feeds the GELU feed-forward network. Attention uses the same input for
the shared-norm variant, or a separately loaded `attn_norm_2.weight` and optional
`attn_norm_2.bias` for the separate-norm variant.

The fused projection stores all Q rows, then all K rows, then all V rows.
MHA, MQA and GQA are supported when query heads divide evenly into KV groups.
NeoX rotation pairs the two halves of each complete, even-width head. Attention
uses the source `1/sqrt(head_width)` scaling and its padded nonflash softmax
rows. Projection and feed-forward matrices have no biases in this graph.

The residual order is `(FFN_output + attention_output) + layer_input`, with
both additions rounded as F32. The primary normalization is reused for FFN;
attention output does not feed that normalization. Final affine LayerNorm
produces token embeddings and the output projection produces logits. When
`output.weight` is absent, token embedding weights supply the output matrix.

The source admits the second-norm weight and bias independently. A weight
without bias skips the addition entirely, including its signed-zero behavior.
A bias without the weight is retained but inactive. The Rust loader and typed
model preserve those cases. `falcon.attention.max_alibi_bias` is not read by
the frozen Falcon graph; its presence does not select ALiBi.

## Loading, state and resource boundaries

The shared GGUF reader handles ordinary files, discovered shard sets and
explicit ordered shard paths. Frozen scalar metadata uses its exact wire
types. Layer count is bounded to 1 through 512, and context length is limited by both
the request and model metadata. Matrix byte decoding and packed arithmetic use
the existing reviewed numeric implementations. The loader accounts for decoded
F32 weights before loading tensors; original packed matrix bytes, state and
temporary allocations are additional.

Required affine vectors have F32 storage, matching original CPU MUL/ADD
requirements. Unsupported low-precision active vectors return an error before
execution instead of entering original assertion paths. Vector dimensions,
finite values, matrix dimensions, full even RoPE width, head grouping, epsilon,
frequency base and arithmetic overflow are checked.

The strict `Context` owns F32 contiguous KV state and supports multiple
sequences, batches, direct embedding input, output selection, cancellation,
checkpoints, state save/restore, forking and rewind. Failed or cancelled batches
retain the prior state. Existing LoRA/control-vector handling applies through
the model's matrix and layer contracts. Model fingerprints include norm choices,
optional bias presence, output tying, configuration and actual packed storage.

Uniform scalar head/FFN metadata, causal attention and unscaled full-head RoPE
are admitted. Per-layer arrays, scaled RoPE or gain, noncausal attention,
nonstandard key/value/head widths, sliding windows, MoE, NextN, extra active
tensors and unsupported general transforms remain explicit errors. Partial
rotary dimensions and odd head widths are rejected because the frozen Falcon
graph asserts its complete-head contract. This addition does not enable Falcon
in the separate native C decoder context, flash attention, alternative KV
storage or a GPU model backend.

## Independent verification

The reference corpus was captured and packed before Rust Falcon implementation
using the frozen original GGUF writer, scalar quantizers and native C API on
Windows MSVC and Linux GNU. Original implementation and oracle code remain
outside the product; the fixture provenance contains their hashes.

There are 36 synthetic model cases. Twenty-two execute admitted variants:
shared/separate norms, omitted/inactive second bias, tied/untied output,
MHA/MQA/GQA, single and multiple layers, custom epsilon/base, omitted defaults,
and F32/F16/BF16/Q4_0/Q8_0/Q4_K matrices. Fourteen cases exercise original
load failures/assertions or explicitly unsupported execution modes.

Each profile's maintained inference comparison checks 1,277 output rows,
412,024 F32 words and 243 greedy token IDs, plus tokenizer pieces,
detokenization, emitted text, explicit embeddings and generation stop behavior.
Debug and release replays match every compared F32 word exactly on both measured
native profiles. The predeclared portable numerical budget remains
`2e-4 + 2e-5 * abs(reference)` for targets without measured bitwise parity.
These counts exclude the independently captured modes that Rust rejects.

Additional maintained tests check metadata types and limits, state/cancellation
rollback, independent sequences, rewind, direct embedding equivalence, optional
norm identity, malformed shapes and allocation accounting. The component
verification receipt records actual debug/release and lint results separately.

These synthetic fixtures establish graph structure and numeric behavior for the
documented configurations. They do not establish pretrained Falcon quality,
all Falcon variants, full native C ABI semantics or all-backend parity.
The original Falcon-RW-1B checkpoint has incompatible projection biases,
ALiBi and residual settings and is not relabeled as supported.

See [fixture provenance](../fixtures/inference/falcon/provenance.json),
[fixture inventory](falcon-inference-fixture-manifest.tsv), and
[component evidence](../evidence/2026-11-13/falcon-inference/README.md).
