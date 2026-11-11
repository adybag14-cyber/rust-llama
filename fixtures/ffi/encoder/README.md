# Independent native BERT C fixtures

These are data-only fixtures for the native BERT encoder and classifier
contract. The complete behavioral scope is documented in
[ENCODER.md](../../../crates/ffi/ENCODER.md).

The thirteen tiny models derive from the repository's independently tested
`fixtures/inference/tiny-bert-missing-pooling.gguf` (SHA-256
`1eeff594b8c9248833dafcdee68f7fbbe4d614261f7ff0f42258b40811b974f3`).
They have width four, two attention heads, one layer, a six-unit feed-forward
layer, eight learned positions and 108 vocabulary entries. Optional
classifier tensors use deterministic small rational values. These synthetic
weights are MIT-licensed repository test data, not pretrained-model evidence.

`rank-mean.gguf` intentionally retains the original INT32 classifier-pooling
type mistake as a source-rejected loader fixture. `rank-mean-u32.gguf` differs
in exactly one metadata type byte, preserving every tensor byte. It is the
separately captured valid mean-classifier case. F16/BF16 matrix variants retain
F32 token-type tables because both original libraries abort for half-typed
token-type addition; the earlier rejected capture is retained in evidence.

`windows-msvc.json` and `linux-gnu.json` each contain 192 original-header
C scenarios, with input requests and raw IEEE-754 output words. Thirteen
scenarios reject the negative model above. Four additional scenarios use the
external BGE F16 model, SHA-256
`f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999`,
67,308,128 bytes. Its weights are not included here. The existing pretrained
model provenance and license notices in `fixtures/inference` continue to
apply; the ignored test requires the exact external file.

`sequence-order-windows-msvc.json` and
`sequence-order-linux-gnu.json` each contain 569 independent standard-library
I32 set observations, including direct and copied order. They qualify the
container behavior observable through the frozen no-cache attention mask.

`provenance.json` retains the archive, caller, model, request-matrix,
preimplementation seal, original-output and addendum identities. No result
in these reference files was produced by the Rust encoder. Original C++,
Python generators and build artifacts remain outside the product.

All fixtures are hash registered by `docs/encoder-fixture-manifest.tsv`.
The ordinary `context::encoder` tests replay the tiny data and both set
profiles; the ignored pretrained test uses `LLAMA_TEST_BERT_MODEL`.

## Active vector storage addendum

The separate `vector-admission-*.json` corpora and seven additional tiny
GGUF variants record native CPU operand-type admission. Five original graph
failures use half-precision active norm/bias vectors; Rust returns a bounded
model-load error. Two ignored classifier biases retain F16/BF16 storage and
execute successfully when their paired matrix is absent. The existing plain
and full-classifier F32 models are the two controls. The primary 192-case
corpora and their numerical tolerances remain unchanged. Exact model/source,
caller and capture identities are in `vector-admission-provenance.json`.
