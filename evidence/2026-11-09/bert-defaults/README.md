# Missing BERT pooling metadata regression

The frozen original initializes the model pooling type to NONE when
`bert.pooling_type` is absent. An untouched default context requests
UNSPECIFIED (-1) and resolves to NONE (0). The prior Rust loader incorrectly
selected MEAN (1) in that case.

The independent native capture uses a **6,832-byte synthetic F32 BERT GGUF**:
one layer, width 4, feed-forward width 6, two heads, two token types, an
eight-position table, and 21 tensors. Its metadata deliberately omits the
pooling key. This fixture is a focused configuration regression and does not
represent pretrained-model parity.

The original C API records default, explicit None and explicit Mean contexts
for three prompts. The new ordinary test
`crates/inference/tests/bert_defaults.rs` verifies resolved pooling, exact
token IDs, 11 token embedding rows and three explicit Mean vectors. All
**56 compared floats are bit-identical** in the Windows debug and release
captures; the portable assertion uses the fixed bound
`2e-6 + 2e-6 * abs(reference)`.

| Capture stem | Result |
| --- | --- |
| `debug` | Ordinary regression passed; maximum absolute error 0 |
| `release` | Ordinary regression passed; maximum absolute error 0 |
| `strict-clippy` | New inference test passed with warnings denied |
| `fixture-inventory` | 113 fixture files, 113 declarations, four manifests |

The model, original request, native result and detailed provenance are in
`fixtures/inference/{tiny-bert-missing-pooling.gguf,bert-defaults-request.json,bert-defaults-reference.json,bert-defaults-provenance.json}`.
Their source/driver/compiler/library identities are recorded in the provenance.
The original comparison program and model generator remain outside the
maintained Rust product.

Run without network access or external model downloads:

```text
cargo test --locked --offline -p llama-inference --test bert_defaults
cargo test --release --locked --offline -p llama-inference --test bert_defaults
```
