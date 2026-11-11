# Frozen BLOOM converter evidence

This folder contains independent original converter evidence for the two captured native profiles. The full source/original capture identities and clean-before-implementation seal are in [provenance.json](provenance.json).

- `pipeline.bin`: 43,093,090-byte pack of 107 deduplicated synthetic input and official tokenizer/sidecar blobs.
- `pipeline.json`: every original input range, 84 complete output size/hash pairs and 29 original failures per host.
- `class-reference.json`: 87 native tokenizer class configurations and 2,296 accepted encode/decode request pairs per host.
- `regex-reference.json` and `regex-membership.bin`: both-host exact official Split behavior, all Unicode scalars and 1,922 pretokenization/encoding/decoding requests, sealed before implementing the matcher.
- `metadata-reference.json`: 41 direct original metadata calls, including complete accepted header bytes.
- `aliases.json`: 576 original tensor-map lookups.
- `filter-reference.json`: 1,308 native filter decisions per host across all six converter families, sealed before the shared filter correction.
- `source-tables.json`: independently captured BLOOM configuration defaults and tensor aliases.
- `pretrained-reference.json`: unchanged official checkpoint identity and both native full-file references.
- `MODEL_CARD.txt`: byte-identical original official model card.
- `LICENSE.BLOOM-RAIL.txt`: complete original license for bundled official tokenizer data.

Expected files and requests come from frozen original code. Rust never generates reference answers. Successful pipeline comparisons hash the entire output, including metadata, descriptors, padding and tensor bytes. Full original files remain external at the paths recorded in provenance.

See [BLOOM_HF_CONVERSION.md](../../../../../docs/BLOOM_HF_CONVERSION.md) and [BLOOM_NOTICES.md](../../../BLOOM_NOTICES.md).
