# Native GPT-2 and GPT-NeoX HF conversion

The general Rust `llama-convert` pipeline admits the frozen
`GPT2LMHeadModel` and `GPTNeoXForCausalLM` conversion families. It reads
local single-file or indexed Safetensors checkpoints, prepares vocabulary
and metadata, transforms tensors and publishes GGUF through the existing
checked writer. These paths do not invoke a fixed model preparer, Python,
Transformers, Torch, NumPy or original C/C++ code.

The authority is the user-provided llama.cpp ZIP SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Pinned source identities and independent original outputs are recorded in
[synthetic provenance](../crates/convert/tests/fixtures/gpt-conversion/provenance.json)
and [pretrained provenance](../crates/convert/tests/fixtures/gpt-conversion/pretrained-reference.json).

## Family behavior

| Contract | GPT-2 | GPT-NeoX |
| --- | --- | --- |
| Source class | `conversion/gpt2.py::GPT2Model` | `conversion/gptneox.py::GPTNeoXModel` |
| GGUF architecture | `gpt2` | `gptneox` |
| Configuration | Transformers 4.57.6 GPT2Config defaults and attribute aliases | GPTNeoXConfig defaults and validity fallback |
| Weights | Swap dimensions 0 and 1 for Conv1D checkpoint names | Reorder fused Q/K/V from head-major to Q/K/V-major |
| Vocabulary | Frozen GPT2 BPE routine | Frozen GPT2 BPE routine with NeoX Fast class behavior |
| Output modes | `auto`, F32, F16, BF16, Q8_0, TQ1_0, TQ2_0 | Same |

GPT-2 transposes `.c_attn.weight`, `.c_proj.weight` and `.c_fc.weight`
before quantization. Higher-rank inputs swap their first two dimensions
while retaining trailing dimensions. Canonical aliases do not gain a
transpose unless their source name has one of these suffixes. The source
requires `n_ctx` and does not synthesize it from `n_positions`. It emits
an FFN dimension of `4*n_embd` even when `n_inner` differs, and emits
`general.file_type`.

GPT-NeoX reshapes the original
`gpt_neox.layers.<id>.attention.query_key_value` weight and bias names
using the source's `n_head` precedence and `hidden_size`, then groups
Q, K and V in that order. Other mapped aliases remain unpermuted.
Explicit `n_head:null` fails even without QKV tensors because the source
asserts its presence before name dispatch. Zero, negative and nonnumeric
`n_head` can remain unused for a checkpoint without matching QKV tensors;
the independent source accepts those cases.

The NeoX metadata override writes context, width, layers, FFN width,
truncated rotary-prefix dimension, heads, parallel-residual Boolean and
LayerNorm epsilon in the original order. It does **not** emit
`general.file_type`, generic RoPE-base fields or scaling metadata. These
omissions are preserved. Conversion admission does not imply that an
inference runtime accepts every emitted shape or rotary dimension.

Each tensor permutation acts on prepared scalar bytes before the shared
output codec, preserving F16 and exceptional scalar representations.
Checked dimensions and element counts precede allocation. Permutation
buffers are charged to the cumulative working budget, and empty buffers
do not cause loops over large zero-sized shapes. The generated
`gpt_tables.rs` contains 279 GPT-2 and 274 NeoX declarative alias rows,
plus pinned configuration defaults; execution is implemented in Rust.

## Tokenizer class contract

The sidecar names `GPTNeoXTokenizer` and `GPTNeoXTokenizerFast` use the
pinned Fast-class behavior. Root `gpt_neox` configuration can infer the
class. Defaults use `<|endoftext|>` for unknown/BOS/EOS tokens; BOS and EOS
insertion default to false.

The class recreates its BOS/EOS template, follows Python truth-value
semantics, applies special-token defaults and the generic Fast prefix
rules. Direct ByteLevel and Sequence roots preserve their different
prefix handling. Invalid template names and missing required BOS/EOS
tokens produce checked errors. This class contract is separate from the
BPE backend's executable schema boundary.

The official GPT-2 tokenizer has a legacy untagged BPE model. An absent
`model.type` now follows the native BPE deserializer; the BPE fields still
require validation. Explicit null and unsupported model types do not
become implicit BPE. Both GPT families always use their original GPT2
vocabulary routine even when an unrelated SentencePiece sidecar exists.

The [class corpus](../crates/convert/tests/fixtures/gpt-conversion/neox-class-reference.json)
has 46 independently captured configurations: 40 accepted, six rejected
and 960 exact encode/decode requests. Separate native profiles agree on
admission, IDs and text. Unordered word lists in native template-error
text are excluded from comparison and documented in adjacent provenance.

## Independent conversion evidence

The ordinary corpus combines synthetic tensors with unchanged official
GPT-2 or Pythia tokenizer sidecars. Synthetic embedding rows are deliberately
smaller than the vocabulary: this is conversion coverage, not miniature
inference.

| Family | Accepted files per host | Original rejections per host | Exact output bytes per host |
| --- | ---: | ---: | ---: |
| GPT-2 | 34 | 12 | 63,715,842 |
| GPT-NeoX | 43 | 12 | 80,094,888 |
| Total | 77 | 24 | 143,810,730 |

The 101 cases cover all seven output choices, source scalar types,
quantization fallback tails and full ternary blocks, fused-QKV requests,
vocabulary-only files, default/raw/nested configuration, class inference,
ignored SentencePiece sidecars, vocabulary bounds, shards, canonical and
short names, duplicate/unknown mappings, rank-three transforms, empty
inputs, missing configuration, causal masks and QKV dimension errors.

Each Rust replay verifies every input blob and the complete output length
and SHA-256. The 12,686,597-byte input pack contains 91 deduplicated blobs,
not pretrained weights. Only the original converter generated expected
files; the packer independently rehashes the retained complete outputs.
No Rust-generated expected output or numerical tolerance replaces them.

Native captures use Python 3.14.2/Torch 2.11.0/NumPy 2.4.6 on Windows and
Python 3.10.12/Torch 2.11.0/NumPy 2.2.6 on GNU/Linux, with Transformers
4.57.6 and Tokenizers 0.22.2 on both. Captures fix `PYTHONHASHSEED=0` and
one Torch thread.

## Pretrained inputs and the GPT-2 source failure

| Input | Immutable revision | Original Safetensors SHA-256 |
| --- | --- | --- |
| `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| `EleutherAI/pythia-14m` | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` | `116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1` |

The frozen GPT-2 converter rejects the unmodified checkpoint at
`h.0.attn.bias`: its subclass forwards static causal masks to the generic
tensor mapper. Both native profiles reject F16, Q8_0 and auto conversion.
Rust reproduces those rejections and publishes no destination.

As in the [GPT-2 inference qualification](GPT2_IMPLEMENTATION.md), a separate
validation input removes exactly twelve static causal-mask buffers. All
148 remaining names, dimensions, dtypes and tensor values are preserved.
All sidecars are unchanged.
[Derivation provenance](../crates/convert/tests/fixtures/gpt-conversion/gpt2-derived-input-provenance.json)
records every removed buffer, retained tensor hash and resulting file.
This adaptation is never an automatic product conversion step.

Direct Pythia-14M conversion needs no adaptation. Six accepted files cover
F16, Q8_0 and auto, totaling 718,209,760 complete bytes per profile:
641,600,320 for GPT-2 and 76,609,440 for Pythia. Exact per-mode hashes are in
the pretrained manifest. Full model weights and outputs remain external.

## Reproduction

Ordinary replay needs no external model or Python installation:

~~~console
cargo test -p llama-convert --offline --locked --test gpt_conversion -- --nocapture
cargo test -p llama-convert --offline --locked --lib native_gptneox_fast_class_semantics -- --nocapture
~~~

The two ignored pretrained tests use these local ordinary-file directories:

| Variable | Meaning |
| --- | --- |
| `LLAMA_TEST_GPT2_HF` | Unmodified pinned official GPT-2 input |
| `LLAMA_TEST_GPTNEOX_HF` | Unmodified pinned Pythia-14M input |
| `LLAMA_TEST_GPT_OUTPUT_ROOT` | Fresh external output root |
| `LLAMA_TEST_GPT2_FILTERED_HF` | Optional existing exact GPT-2 derivative |

~~~console
cargo test -p llama-convert --release --offline --locked --test gpt_pretrained -- --ignored --nocapture --test-threads=1
~~~

When `LLAMA_TEST_GPT2_FILTERED_HF` is absent, the test prepares its own
validation input in Rust under the fresh output root. It first verifies
the complete official checkpoint and every sidecar, removes exactly the
twelve named masks using retained tensor readers, copies all remaining
ranges unchanged, then checks the **complete derived Safetensors SHA-256**
against the independently captured input. Thus CI needs no Python
preparation script or transformation package.

Pretrained tests hash every input before and after conversion, retain
successful output files and refuse to reuse prior output subdirectories.
Frozen naming can depend on the input directory, so external input
basenames must retain the captured `official-hf` and
`gpt2-without-causal-mask-buffers` names.

## Boundaries and attribution

RuGPT3XL, other GPT architectures, mixture-of-experts, projectors, pickle,
remote acquisition, executable remote tokenizer code, prequantized source
layouts, big-endian output and output sharding remain outside this
conversion path. Repository-wide release gates remain open. The separate
inference qualifications are in `GPT2_IMPLEMENTATION.md` and
`GPTNEOX_IMPLEMENTATION.md`.

Frozen algorithms and alias data retain upstream MIT notices.
Transformers/Tokenizers configuration and behavior retain the Apache-2.0
notice in [METADATA_NOTICES.md](../crates/convert/METADATA_NOTICES.md).
Distributed official tokenizer data retain the
[GPT-2 Modified MIT license](../crates/convert/tests/fixtures/gpt-conversion/LICENSE.GPT2.txt)
and [Pythia Apache-2.0 license](../crates/convert/tests/fixtures/gpt-conversion/LICENSE.PYTHIA.txt).
No pretrained weights or external-language implementation are distributed.
