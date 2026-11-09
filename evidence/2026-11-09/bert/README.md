# Dense BERT integration and arithmetic diagnosis

This evidence accompanies the dense `bert` GGUF loader, WordPiece integration and raw embedding API/CLI. It does not establish all BERT-derived architectures, classifier heads, GPU model execution or a complete upstream embedding server.

## Fixed reference

- Frozen source archive: `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
- Model: BGE-small-en-v1.5, 67,308,128 bytes, SHA-256 `f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999`.
- Reference: separately built original llama/ggml DLLs, one CPU thread, F32 KV parameter values, noncausal attention, flash attention disabled.
- Corpus: three prompts, 25 token vectors and nine pooled vectors, 384 dimensions; 13,056 unique expected floats.
- Predeclared model tolerance: `2e-4 + 2e-5 * abs(reference)`. This limit was not increased after the initial failure.

The request, native output and full provenance are committed under `fixtures/inference/bert-bge-*`. Weights and native executables are outside the product.

## Failure and correction

The initial Rust result exceeded the fixed limit: one value differed by `0.0002135932445526123` against a budget of `0.00020448090553283691`. The initial logs are preserved under their original `before-f16-accumulation-*` names. That filename reflects the first hypothesis, not the eventual diagnosis.

An additional independent F16 test ruled out F16 dot arithmetic: 68 original graphs produced 6,168 outputs exactly matching the existing Rust kernel, including 384/1536 input widths, multiple activation columns and cancellation around reduction boundaries. Changing to a hypothesized SSE reduction would have made 4,018 results disagree, so the F16 kernel was left unchanged.

A native evaluation callback then captured 457 graph tensors for the first BERT sequence. Its complete model output was exactly equal to the original run without observation. Comparing corresponding Rust stages showed:

1. Token/type/position embedding sums, embedding normalization and first-layer Q/K/V projections were exact.
2. Attention was the first difference. The source uses rounded F32 products accumulated in F64 for both QK and probability-times-V multiplication. Rust had used F32 sums.
3. Recomputing the first layer against native tensors confirmed the diagnosis: F32 summation differed on 844 QK entries and 2,066 weighted-V entries; the source reduction matched every entry.
4. Small attention differences crossed half-precision GELU table boundaries and amplified in later layers.
5. Matching the source reduction made all 122 corresponding captured stages exactly equal.
6. Matching mean pooling's repeated F32 reciprocal multiplication, followed by its F64 sum, removed the final pooled rounding difference.

The correction also applies to the decoded F32 matrix path. The earlier F32 Llama corpus now becomes bit-identical on Windows; Q4 remains exact. Original before/after stage comparison reports are retained here. Temporary runtime tracing was removed from production.

## Permanent checks

`crates/model-runtime/tests/fixtures/bert-attention-reference.bin` contains native Q/K/V and output tensors for all 12 layers of the first sequence. The ordinary runtime test compares 46,080 attention values. Expected inputs and outputs come from the original graph, so a Rust error in an earlier layer cannot contaminate this reference.

The external-model regression checks all raw vectors and mean/first/last pooling, default CLS pooling, exact token IDs, model capabilities, generation rejection, context bounds, weight limits, invalid token IDs and cancellation. It checks full input size and SHA-256 before using the model.

Windows final pretrained logs record zero absolute error for all 13,056 BERT values. Linux records maximum absolute error `1.348853111e-4`, within the same fixed tolerance; platform math functions can differ at GELU table boundaries. The full current Linux evidence is under `../linux-extension`.

## Endianness

A separately downloaded equivalent big-endian GGUF is 67,308,128 bytes with SHA-256 `0e68255af91c707227f4840603bcbd0555054b921f757d79efdb2652ae8bb952`. The original C reader rejects that opposite-endian file. The Rust reader loads it, and the same complete native corpus comparison passes: zero Windows error and the same Linux bound. This is explicitly an equivalent-encoding reader check, not native big-endian execution evidence.

## Application checks

The CLI exposes raw embeddings with explicit pooling and `normalized: false`. A generation request against a BERT-backed server fails during validation, before an SSE response begins. `cli-http-validation.json` records the executed command/HTTP checks and their observed results. General embedding HTTP routes remain outside this checkpoint.
