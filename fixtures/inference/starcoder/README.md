# Independent StarCoder and StarCoder2 inference fixtures

The synthetic models and original C API outputs were captured on Windows MSVC
and Linux GNU before either Rust architecture implementation. The corpus uses
the frozen original GGUF writer and scalar quantizers and contains 72 model
cases: 43 executable variants and 29 malformed or explicitly unsupported cases.

StarCoder covers learned position embeddings, fused biased QKV and sequential
affine-LayerNorm/GELU blocks. StarCoder2 covers full-head NeoX RoPE, optional
QKV bias precedence and separate/fused projections. Both cover tied/untied
outputs, MHA/MQA/GQA, six float/packed storage types, text generation and direct
embedding inputs. Independent expected rows are stored losslessly as
little-endian F32 with complete-row deduplication.

`provenance.json` preserves source, oracle, executable and library identities
and capture times. Reference C/C++ sources and binaries remain external.
The synthetic corpus is MIT-licensed as recorded in `LICENSE.md`; any pretrained
acceptance uses separately attributed models and is documented separately.

No synthetic fixture establishes pretrained model quality, performance parity,
GPU execution, complete native C-context support or the full architecture gate.

## Pretrained references

`pretrained-*-reference.json` and `pretrained-*-values.f32le.bin` retain independent
original C API observations from pinned Tiny StarCoder F16 and Q8_0 files and a
pinned StarCoder2 3B Q4_K_M file. Each uses three short prompts, complete logits
and prompt embeddings, and four greedy predictions. These captures occurred
after architecture implementation and before Rust pretrained comparisons;
`pretrained-provenance.json` records that order explicitly.

Pretrained weights are absent from this directory and remain external test
inputs. The publishers declare `bigcode-openrail-m`; the provenance records
model and conversion hashes, publisher/derivative revisions, original reports,
license reference and the pinned official license document hash. The MIT notice
above applies to the synthetic fixture corpus, and does not relicense either
pretrained model. Short-prompt reference parity does not establish broad code
quality, long-context/SWA behavior, performance or universal model parity.
