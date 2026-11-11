# BLOOM Hugging Face conversion

The native Rust converter admits the frozen `BloomForCausalLM` and `BloomModel` paths from `conversion/bloom.py`. It accepts local dense Safetensors checkpoints and uses the existing retained-input, checked planning and completed-file publication framework. It does not execute Python, Torch, Tokenizers or Transformers.

The authority is the supplied llama.cpp ZIP with SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
The original BLOOM subclass has SHA-256
`cf17a9c8717c45121c843112dfce7a2fd2aeab060158fb6349f50397f4963c1d`.
The [provenance record](../crates/convert/tests/fixtures/bloom-conversion/provenance.json)
seals source identities, external original files, capture scripts and both original environments before the first Rust implementation edit. The product was clean at commit
`37ba10da13aff8bc10bfe1ecb7b650b1c68029a7` when that seal was made.

## Configuration and metadata

The dispatch preserves the subclass's metadata order and its omissions. It writes:

| Order | GGUF key | Frozen input |
| ---: | --- | --- |
| 1 | `bloom.context_length` | `seq_length` when present, otherwise the embedding width |
| 2 | `bloom.embedding_length` | `hidden_size`, falling back to `n_embed` only when the first key is absent |
| 3 | `bloom.feed_forward_length` | Four times that embedding width |
| 4 | `bloom.block_count` | The base converter's selected layer count |
| 5 | `bloom.attention.head_count` | `n_head`, falling back to `num_attention_heads` only when absent |
| 6 | `bloom.attention.head_count_kv` | The same head input |
| 7 | `bloom.attention.layer_norm_epsilon` | Required `layer_norm_epsilon` |
| 8 | `general.file_type` | Resolved output mode |

The separately inherited architecture, naming, vocabulary and general metadata precede or follow these fields according to the shared frozen conversion pipeline. Explicit null dimensions fail; a null value does not enable the fallback alias. Unrelated common fields such as `is_causal`, `rope_theta`, `intermediate_size`, `n_inner` and `max_position_embeddings` do not add or replace the subclass's metadata.

A successfully selected Transformers `BloomConfig` normalizes the root before the base converter merges nested text configuration. Its captured defaults include hidden width 64, two layers, eight heads, vocabulary size 250,880 and LayerNorm epsilon 1e-5. A non-null `n_embed` overrides `hidden_size`; an explicit null `n_embed` is consumed without overriding it. The inherited aliases `num_attention_heads` and `num_hidden_layers` overwrite `n_head` and `n_layer`. The serialized normalized configuration no longer contains those aliases. This removal matters when a nested configuration supplies a different layer count or head count.

Unknown model types and unsuccessful AutoConfig resolution retain the raw configuration behavior. Absent, null and unknown `model_type` values are distinct. Architecture names and layers still use the frozen base converter's nested-wrapper order. Both known and raw configurations have independent complete-file cases.

The original writer accepts more than integer head counts. Integer and Boolean scalars become U32 values. Nonempty strings become arrays of Unicode character strings. Homogeneous JSON arrays preserve I32, F32, Boolean, string or nested-array wire types. Empty top-level strings and arrays omit both head keys; heterogeneous arrays, empty nested arrays, null values and out-of-range scalar integers fail. These are conversion semantics, not an assertion that the resulting metadata describes an executable inference graph. The explicitly named `write_source_conversion_header` reproduces the Python converter's nested-array wire extension. Nested arrays remain rejected by the normal Rust writers, the strict Rust readers and the strict native GGUF reader; emitting those original bytes does not make them generally valid GGUF. Nested array construction has a fixed 64-level ingestion bound in addition to the configurable JSON and GGUF limits.

## Tensor transformation

The base converter first selects and filters checkpoint contributions in source insertion order. The `text_model.` wrapper removal belongs only to the original Llama subclass; BLOOM, Qwen2/Qwen3 and GPT-2/GPT-NeoX retain that wrapper and reject names that do not map. The shared `e_score_correction_bias` rename tests the original suffix before removing `language_model.`. Both ordering rules have 1,308 independently captured filter decisions across the six families on each host. The BLOOM subclass then removes every literal `transformer.` occurrence from each selected name. It does this after the contribution dictionary has been formed. Names that become identical only after that removal cause the same duplicate canonical-name failure as the source; they are not silently merged.

The exact rooted names `h.N.self_attention.query_key_value.weight` and `h.N.self_attention.query_key_value.bias` select the fused-QKV transform. For heads `H` and embedding width `E`, the source interprets weights as `[H, 3, E // H, E]` and biases as `[H, 3, E // H]`. It concatenates the Q, K and V blocks in that order. The Rust permutation acts on the prepared scalar bytes, so F16 payloads and NaN bit patterns are preserved before the shared output codec runs.

The source determines reshape admission from element counts. Original rank-one, rank-three and singleton layouts are covered. A nondivisible head count can be accepted for a weight-only tensor when its element count matches the floor-divided source shape; the bias path requires the per-component width to equal `E`. An empty bias with zero embedding width is admitted. An empty weight with positive embedding width and zero per-head width is also admitted. A weight with zero embedding width fails its inferred reshape. Non-QKV tensors do not acquire extra numerical head validation beyond the subclass's non-null assertions and metadata writer behavior.

The frozen BLOOM tensor alias table is captured independently. Its 576 direct native lookup requests cover canonical names, checkpoint aliases, both weight and bias suffixes, and invalid block names. Canonical aliases do not accidentally trigger the checkpoint-specific QKV transform.

All seven existing output selections remain available: auto, F32, F16, BF16, Q8_0, TQ1_0 and TQ2_0. The shared numeric preparation, one-dimensional/norm F32 policy, ternary embedding/output F16 policy and shape-dependent F16 fallback remain in force. Native synthetic cases include small tail widths, complete ternary blocks, multiple input dtypes, special half payloads, indexed shards, vocabulary-only output and fusion requests.

## Tokenizer selection

BLOOM follows the frozen GPT-2 vocabulary routine and ignores a SentencePiece sidecar for this path. It admits the original legacy untagged BPE model representation. The local tokenizer class can be `BloomTokenizerFast`; Transformers also resolves `BloomTokenizer` to that fast class. An absent or null sidecar class falls back through the resolved root model configuration, not the merged nested text configuration. Generic or other already reviewed fast classes keep their own semantics when explicitly selected.

BLOOM's default special tokens are `<s>`, `</s>`, `<unk>` and `<pad>`. Normal configured and legacy-map overrides retain the shared precedence. The class preserves the saved postprocessor; `add_bos_token` and `add_eos_token` options do not recreate a Llama/NeoX template.

For a direct ByteLevel root, the generic fast constructor first defaults an absent prefix option to false, applies explicit Booleans, and leaves the saved setting for null or other values that cannot rebuild the stage. BLOOM then recursively enables every ByteLevel stage for any Python-truthy `add_prefix_space` value. That second step also reaches nested Sequences. False, zero, null, empty strings and empty containers do not request the recursive rewrite. Cleanup uses Python truth values, while `split_special_tokens` remains Boolean-only.

The exact official Split expression uses `Isolated` behavior with `invert=false`. Its nested Oniguruma character class excludes whitespace, the listed punctuation, literal parentheses and vertical bar; square brackets remain ordinary matched characters. The finite Rust matcher uses the existing captured Unicode whitespace table. Additional original captures before this matcher was implemented check all 1,112,064 Unicode scalars and 1,922 split/encoding/decoding requests with both ByteLevel prefix settings. A Sequence containing exactly one recursively selected ByteLevel decoder executes as that child; other unimplemented decoder sequences stay explicit errors.

The final vocabulary path still requires the exact recognized frozen tokenizer-probe hash. In particular, adding a truthy prefix to the official BLOOM tokenizer changes that probe and is rejected by the original converter. The Rust path preserves that rejection. A class label does not force a pretokenizer classification.

## Independent regression corpus

Both original host captures were completed before Rust implementation. The environments are the existing pinned Windows Python 3.14.2 / Torch 2.11.0+cpu / NumPy 2.4.6 and GNU Python 3.10.12 / Torch 2.11.0+cpu / NumPy 2.2.6 profiles, with Transformers 4.57.6, Tokenizers 0.22.2 and `PYTHONHASHSEED=0`. Reference imports disabled bytecode writes and used local-only model resolution.

| Cohort, per native host | Coverage |
| --- | --- |
| Complete conversion | 113 cases: 84 accepted files, 29 original rejections |
| Accepted conversion bytes | 1,066,141,612 bytes, each compared by complete length and independently captured SHA-256 |
| Tokenizer class | 87 configurations: 82 accepted, five rejected |
| Class encoding and decoding | 2,296 exact request pairs |
| Metadata writer | 41 calls: 20 accepted, 21 rejected; accepted complete unpadded headers include their original bytes |
| Native tensor name map | 576 exact lookups |
| Shared family filtering | 1,308 exact decisions across Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX and BLOOM, captured before the shared filter correction |
| Official Split pattern | All 1,112,064 Unicode scalars, plus 1,922 complete pretokenization/encoding/decoding requests with two prefix settings |
| Unchanged official pretrained checkpoint | F16, Q8_0 and auto complete-file comparison |

The [input pack](../crates/convert/tests/fixtures/bloom-conversion/pipeline.bin) contains 107 deduplicated original blobs totaling 43,093,090 bytes, SHA-256
`8bea6c8730470809a4926f65b7adf759e546b34d8827ebb1a9d62b0d5b590134`.
It contains synthetic tensor inputs and pinned official tokenizer data. It does not contain pretrained weights. The [pipeline manifest](../crates/convert/tests/fixtures/bloom-conversion/pipeline.json) records every input slice and every native result. Successful Rust output is checked against the complete original output hash; rejected cases must remain errors. No expected output is generated by the Rust implementation.

Full original GGUF files and native reports remain outside the repository under
`C:\Users\adyba\rust-llama-reference\continue-20261111-late\bloom-converter`.
Their identities and exact paths are retained in provenance. The pack preserves full coverage without committing gigabytes of repeated vocabulary metadata.

Ordinary tests exercise the class, metadata, aliases, family filters, exact Split behavior and complete synthetic pipeline. A QKV-specific publication check limits the working budget to the joined, raw and prepared buffers, forcing the additional permutation allocation to fail. It verifies that an existing destination remains byte-identical and that temporary output is removed. A separate source-byte limit rejects the same pinned input during planning.

## Genuine pretrained checkpoint

The test uses unchanged official `bigscience/bloom-560m` at revision
`ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971`.
All seven files, official URLs, lengths, SHA-256 digests, Git blob identities and applicable LFS identities are in the [pretrained manifest](../crates/convert/tests/fixtures/bloom-conversion/pretrained-reference.json).
The 1,118,459,525-byte `model.safetensors` has SHA-256
`a8702498162c95d68d2724e7f333c83d7be08de81cfc091455c38730682116d3`.
The source model and tokenizer data are distributed under BigScience BLOOM RAIL 1.0.

No weights, names, buffers or sidecars are removed or rewritten for this test. The original converter accepts the official input directly. Both captured native profiles produced identical complete hashes:

| Mode | Output bytes | SHA-256 |
| --- | ---: | --- |
| F16 | 1,131,699,072 | `ac136500f37dbada4866b390123f14c9fbe14d215aabf8d93e96fc58772780bf` |
| Q8_0 | 607,738,752 | `97eec92fef8daac7afff0e0a0e71bb822e346ac267d84e05290e2b01f8821fdc` |
| auto | 1,131,699,072 | Same as F16; the official source tensors select F16 |

Preparation consists of downloading the seven manifest-listed files into one ordinary local directory named `official-hf` and verifying their exact lengths and hashes. This basename is part of this exact-output reproduction: the frozen converter derives the default `general.name` value `Official Hf` from it. General conversion accepts other directory names, with the corresponding source-derived metadata. Set `LLAMA_TEST_BLOOM_HF` to the `official-hf` directory and `LLAMA_TEST_BLOOM_OUTPUT_ROOT` to a fresh external output root. Then run:

~~~console
cargo test -p llama-convert --release --offline --locked --test bloom_pretrained -- --ignored --nocapture
~~~

The test `pretrained_bloom_native_files` rechecks all seven inputs before and after conversion, retains three output files under `bloom-560m`, and refuses to reuse a prior output directory. The aggregate is 2,871,136,896 independently verified output bytes per profile. Ordinary test runs leave this test ignored because official model weights are external data.

## Scope and attribution

This addition implements the frozen dense BLOOM converter path. It does not close the full conversion/release gate, add an executable remote model loader, or expand the crate's general prequantized, multimodal, pickle or output-sharding boundaries. The separately developed BLOOM runtime has its own graph, resource and pretrained numerical evidence.

Retain [BLOOM_NOTICES.md](../crates/convert/BLOOM_NOTICES.md) and the complete [BigScience BLOOM RAIL license](../crates/convert/tests/fixtures/bloom-conversion/LICENSE.BLOOM-RAIL.txt). The synthetic model-card case exercises a declared `license: mit` string as metadata; that string does not relicense the accompanying official tokenizer data. Rust implementation and upstream library attribution remain covered by the crate's existing notices.
