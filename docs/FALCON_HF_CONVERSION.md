# Falcon Hugging Face conversion

The native Rust converter accepts the frozen FalconForCausalLM and
RWForCausalLM dispatch aliases. It reproduces the Falcon subclass in the supplied
llama.cpp archive, including its metadata override, grouped QKV transformation,
tensor aliases and GPT-2 vocabulary path. It uses the existing bounded
Safetensors reader and GGUF source-conversion writer.

This is conversion support. The current Falcon inference contract is separate,
and the complete port's model, converter and release gates remain open.

## Frozen behavior

The converter writes these architecture fields in the source's order:

| Field | Frozen value or selection |
| --- | --- |
| Context length | Always 2048, even if a different position limit is supplied |
| Tensor layout | jploski |
| Embedding length | hidden_size |
| Feed-forward length | Four times hidden_size; alternate FFN settings are ignored |
| Block count | The shared source layer-count precedence |
| Query heads | num_attention_heads unless absent/null, then n_head |
| KV heads | num_kv_heads unless absent/null, then n_head_kv or one |
| Layer norm epsilon | layer_norm_epsilon |
| File type | The resolved requested output type |

FalconConfig from the pinned Transformers 4.57.6 environment consumes the legacy
n_embed alias and resolves an absent/null num_kv_heads to its root query-head
count before a nested text configuration is merged. Its legacy head/layer keys
remain independent fields. The Rust implementation retains that distinction
between normalized configuration and the source's raw-config fallback.

The subclass does not emit the common Llama/Qwen RoPE, causal-attention,
activation or custom-FFN metadata. Passing those HF fields does not change the
frozen Falcon metadata contract.

The native head-count writer can emit arrays or character strings for
vocabulary-only metadata. Empty sequences produce no key. The separately named
source-conversion writer retains these original wire behaviors; they do not
establish valid inference models or relax the strict GGUF reader.

## Grouped QKV transformation

Every source tensor name containing query_key_value triggers the source
transformation. A canonical GGUF alias without that substring remains unchanged.

For query heads H, KV groups G and head width D = hidden_size // H, the source
views the tensor as [G, H // G + 2, D, D * H]. Each group contains its query
heads followed by one key and one value. The Rust path copies these ranges into
all queries, then all keys, then all values and restores the original source
shape. It preserves scalar bytes until the shared output codec runs, including
F16 NaN payloads and signed zeros.

The transform and metadata use different source fallback rules. Tensor
transformation selects the first *present key* from the modern/legacy names;
a false/zero/null KV value then follows Python's fallback to one. A true Boolean
KV group is rejected by the source's Torch view operation. Metadata instead
uses the explicit absent/null fallbacks above.

Bounds are checked before the additional permutation allocation. Floor-divided
hidden widths, zero-element tensors and shapes of rank one through four follow
the independently captured source results. Ordinary one-dimensional QKV biases
fail the source's weight-shaped view; a bias with the complete weight element
count can be transformed. Canonical aliases can bypass that reshape. These are
observed conversion behaviors, not a promise that every produced tensor is a
supported inference tensor.

Tensor alias lookup uses all 253 frozen map records in declaration order,
including later aliases that replace earlier entries. The shared filter, shard
ordering, collision checks and source-specific name handling remain in effect.
Falcon does not inherit BLOOM's global transformer-prefix removal.

## Tokenizer

The unchanged official Falcon 7B tokenizer has four stages: contiguous
punctuation, ByteLevel, grouped digits and isolated ASCII digit triples.
The Rust HF backend now executes all five source punctuation split behaviors
and the exact [0-9][0-9][0-9] expression. Unicode punctuation uses captured
Tokenizers 0.22.2 membership, rather than the host Rust compiler's Unicode
version. The original class defaults, added tokens, decoder, metadata probe and
special-token rules still pass through the shared reviewed BPE path.

All 1,112,064 Unicode scalar values were classified independently by the native
punctuation primitive on Windows and GNU. The two captures agree: 726 scalars
belong to the source punctuation set. An additional 8,745 native
split/encode/decode requests cover category boundaries, all five split modes,
prefix settings, contractions, symbols, mixed digit alphabets, byte boundaries
and official special tokens. The original tokenizer JSON and model card retain
their source and license identities.

## Output and resource limits

The seven existing output selections are supported: auto, F32, F16, BF16,
Q8_0, TQ1_0 and TQ2_0. Shared source quantization, F16 row-width fallback,
metadata ordering, model-card handling, vocabulary-only output and complete
temporary-file publication retain the existing converter contract.

Input, output, vocabulary and working-set limits are unchanged. In particular,
the extra QKV copy is charged to the per-output-tensor working budget before
allocation. Independent regression checks deliberately exhaust that allowance,
verify an existing destination survives, and verify temporary output cleanup.
The budget is not a total process-memory limit.

## Independent verification

The maintained fixture contains synthetic tensor data and unchanged official
Falcon 7B sidecars. Both native source environments captured 117 complete
converter requests before Rust comparison:

| Result per host | Count |
| --- | ---: |
| Accepted complete files | 86 |
| Source rejections | 31 |
| Compared accepted output bytes | 202,713,961 |
| Tensor alias lookups | 510 |

The input pack deduplicates 116 blobs into 9,498,011 bytes. It stores source
inputs; complete output lengths and SHA-256 identities come from the frozen
converter. Tests reconstruct the inputs, run the Rust converter, and compare
the complete output files. The ordinary suite covers GQA/MQA/MHA layouts,
one/two norm name sets, both architecture aliases, input scalar types, every
admitted output mode, metadata defaults and malformed configurations, QKV
shapes and bias behavior, canonical aliases, indexed shards, filtering,
class inference, model cards and official vocabulary-only conversion.

Run the focused checks with:

~~~sh
cargo test -p llama-convert --offline --locked falcon -- --nocapture
cargo test -p llama-convert --release --offline --locked falcon -- --nocapture
~~~

The exact validation receipts belong to
evidence/2026-11-13/falcon-conversion. Fixture hashes are declared in
docs/falcon-conversion-fixture-manifest.tsv.

## Pretrained boundary

No pretrained Falcon weight conversion or Falcon inference acceptance is claimed
by these converter fixtures. The pinned official Falcon 7B revision is
ec89142b67d748a1865ea4451372db8313ada0d8; only its sidecars were acquired.

The official configuration omits num_kv_heads. The captured pinned AutoConfig
therefore resolves that field to the query-head count even though multi_query
is true. The frozen converter's reshape then expects that resolved group count.
The corpus retains this source default behavior instead of silently repairing
the checkpoint configuration. An actual unchanged pretrained acceptance must
resolve and measure the source behavior with its weights.

Falcon-RW-1B is also not substituted as acceptance evidence: its official
checkpoint is a PyTorch pickle file and its model declares projection biases,
ALiBi and sequential attention. Those features require their own source and
runtime checks. Falcon-H1, Falcon-Mamba, Falcon3 and other related names are
different architecture/conversion families and are outside this Falcon path.
