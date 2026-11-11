# BLOOM independent inference fixtures

Expected data was captured from the frozen original before Rust implementation. See provenance.json for full source/tool/library hashes, source-asserting and intentionally unsupported cases, declared numerical budgets, storage profiles and trained checkpoint origin. Tiny-inputs.bin contains synthetic GGUF files only. All four values files contain losslessly repacked original F32 outputs; pretrained weights remain external.

The official BLOOM560m checkpoint is unchanged. Frozen Windows and GNU converters independently produce identical F16 and Q8_0 GGUFs. Full BLOOM RAIL1.0 license and official model card are retained alongside the derived reference outputs. These fixtures do not establish full architecture/backend coverage.
