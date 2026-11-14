# StarCoder and StarCoder2 Hugging Face conversion

The native Rust converter accepts GPTBigCodeForCausalLM (StarCoder) and
Starcoder2ForCausalLM (StarCoder2), matching the two registered classes in the
frozen llama.cpp archive. The source archive has SHA-256
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.

Both paths use the existing bounded Safetensors reader, reviewed HF BPE
tokenizer implementation, scalar codecs and complete-file GGUF publication.
Python, Transformers, Torch, NumPy and the original converter are external
reference tools. Product conversion executes Rust only.

Conversion and model inference have separate contracts. The model and complete
port release gates are not closed by these converter results.

## Architecture-specific behavior

| Selection | StarCoder | StarCoder2 |
| --- | --- | --- |
| HF architecture | GPTBigCodeForCausalLM | Starcoder2ForCausalLM |
| AutoConfig type | gpt_bigcode | starcoder2 |
| GGUF architecture | starcoder | starcoder2 |
| Configuration defaults | Transformers 4.57.6 GPTBigCodeConfig | Transformers 4.57.6 Starcoder2Config |
| Tensor-map records | 279 | 339 |
| Attention tensors | Fused QKV or captured canonical aliases | Separate Q/K/V, with optional common QKV fusion |
| Weight orientation | Preserved | Preserved |
| Vocabulary path | GPT-2 BPE | GPT-2 BPE |
| Missing tokenizer class | Resolved root GPT2TokenizerFast | Resolved root GPT2TokenizerFast |

StarCoder writes its metadata in this order: context length from n_positions,
embedding length from n_embd, feed-forward length equal to four times n_embd,
block count, query heads from n_head, KV heads equal to one, layer norm epsilon
from layer_norm_epsilon and resolved output file type.

That metadata override ignores n_inner, multi_query, common RoPE, activation
and noncausal-attention fields. Changing these fields does not silently change
the source's StarCoder graph description. Head-count metadata retains the
source writer's scalar/sequence behavior; malformed metadata accepted by the
source is not thereby a supported inference graph.

The pinned GPTBigCodeConfig resolves the inherited aliases hidden_size,
max_position_embeddings, num_attention_heads and num_hidden_layers after its
direct n_* arguments. An explicitly supplied inherited alias therefore wins
over an explicitly supplied n_* argument. Raw configuration fallback and
nested-text merging preserve the source's different precedence.

StarCoder2 inherits the source TextModel metadata implementation. It writes
the common context, embedding, feed-forward, head, norm and RoPE fields with
the starcoder2 prefix and the same order and omission rules. In particular,
an explicit null num_key_value_heads remains null and the metadata key is
omitted; it is not replaced with the query-head count. The source does not
emit sliding-window metadata for this subclass. Writing supported scaling
metadata does not establish scaled-RoPE runtime support.

Neither class transposes matrices or performs a Llama Q/K row permutation.
StarCoder's c_attn and c_proj names refer to the checkpoint's linear tensors;
the GPT-2 Conv1D transform does not apply. Both families use the frozen
tensor-map declaration order, common filters, index/shard ordering, collision
checks and prefix selection.

## Tokenizer and output

The converter follows each class's source GPT-2 vocabulary path. An adjacent
SentencePiece tokenizer.model is ignored. Reviewed tokenizer.json pipelines,
legacy untagged BPE data, added tokens, special-token precedence and root
AutoTokenizer class resolution retain the shared converter behavior.

The retained official fixtures use unchanged tokenizer/configuration sidecars
from bigcode/tiny_starcoder_py and bigcode/starcoder2-3b. They test both explicit
and inferred tokenizer classes, complete conversion, QKV fusion,
vocabulary-only output and the unchanged official configuration. Tensor weights
in the ordinary fixture pack are synthetic; the StarCoder2 checkpoint was not
downloaded for converter acceptance.

The seven existing output selections remain available: auto, F32, F16, BF16,
Q8_0, TQ1_0 and TQ2_0, including source row-width-dependent F16 fallback. Dtype
casts, scalar profile differences, metadata order and filename generation
retain the shared contract. No new dependency or allocation limit is added.
The checked source, sidecar, layer, output and per-output-tensor working
budgets still apply.

Prequantized, MoE, recurrent, multimodal and other unregistered architectures,
pickle input, remote acquisition, output splitting and unsupported tokenizer
execution remain outside this crate's admitted path. GPT-J is not registered
in the frozen source and is not included in this change.

## Independent complete-file corpus

Original source execution on Windows MSVC and GNU produced these results
before the corresponding Rust comparisons:

| Result per native host | Count |
| --- | ---: |
| Complete conversion requests | 194 |
| Accepted complete files | 163 |
| Original rejections | 31 |
| Accepted output bytes | 357,723,535 |
| Tensor-map lookups | 1,248 |
| Deduplicated input blobs | 176 |
| Input-pack bytes | 38,284,433 |

The tests reconstruct each original input, run Rust conversion and compare the
entire output length and SHA-256. Source failures must remain failures. The
alias test independently compares every captured name lookup, including
canonical and suffix variants and out-of-range layers.

Coverage includes every admitted output type, source scalar input types,
partial and complete quantization blocks, special FP16 values, canonical
names, aliases, filtering, indexed shards, QKV fusion with and without biases,
root/nested/default configuration, null/present key distinctions, metadata
arrays, source-ignored settings, model variants, tokenizer class/prefix
behavior and malformed tensor/configuration inputs. There are 178 general
requests and 16 additional official-sidecar requests.

These are comparison counts, not a claim that every converted malformed
configuration forms a valid GGUF inference model. The original-compatible
writer can reproduce source metadata that the strict reader correctly rejects.

Run the focused corpus with:

~~~sh
cargo test -p llama-convert --offline --locked starcoder -- --nocapture
cargo test -p llama-convert --release --offline --locked starcoder -- --nocapture
~~~

Source/environment identities, immutable capture hashes and complete input
and output identities are retained beside the fixtures. The per-file fixture
declaration is docs/starcoder-conversion-fixture-manifest.tsv. Validation
receipts are in evidence/2026-11-14/starcoder-conversion.

## Unchanged pretrained StarCoder acceptance

The separate ignored test requires bigcode/tiny_starcoder_py revision
8547527bef0bc927268c1653cce6948c5c242dd1. Its original Safetensors file is
656,601,304 bytes with SHA-256
15fa942f055b618d5ca6283f5c27278a475ff12e53dc704b9658ffd5160d4021.
All nine downloaded source files are checksum-pinned; the checkpoint and
sidecars are unchanged.

The original Windows and GNU converters agree on these complete files:

| Output | Bytes | SHA-256 |
| --- | ---: | --- |
| F16 | 342,950,656 | abdd880356fe55d9e951e5da4b198639d15331d1e10c0015795d10caaa273602 |
| Q8_0 | 195,126,016 | b2620b0de962f582218530233849064c088c56e86344a7e49afb53c7d5957906 |
| auto (F16) | 342,950,656 | abdd880356fe55d9e951e5da4b198639d15331d1e10c0015795d10caaa273602 |

The Rust test checks all input hashes before and after conversion and all
881,027,328 output bytes through complete length/hash comparisons. Set
LLAMA_TEST_STARCODER_HF to the official checkpoint directory and
LLAMA_TEST_STARCODER_OUTPUT_ROOT to a fresh external output directory, then run:

~~~sh
cargo test -p llama-convert --release --test starcoder_pretrained --offline --locked -- --ignored --nocapture
~~~

The test preserves its output files and refuses to overwrite a prior model
output directory. Product conversion does not download weights or adapt
checkpoint inputs. No pretrained StarCoder2 HF weight conversion is claimed;
its official sidecars are pinned at
733247c55e3f73af49ce8e9c7949bf14af205928.

## Source and data notices

MIT notices cover the frozen llama.cpp converter and tensor-map data.
Apache-2.0 notices cover the pinned Transformers and Tokenizers reference
semantics. The GPT-2 tokenizer corpus retains its separate Modified MIT
notice. The official StarCoder/StarCoder2
tokenizer sidecars and model cards retain the complete BigCode Open RAIL-M v1
license, including its use restrictions. The license text was extracted from
the pinned official license Space without executing its application code.
See crates/convert/STARCODER_NOTICES.md and the adjacent provenance records.
