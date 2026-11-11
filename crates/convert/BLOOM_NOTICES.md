# BLOOM conversion attribution

The native Rust BLOOM conversion path follows the immutable llama.cpp `conversion/bloom.py` under the upstream MIT license, and the pinned Transformers/Tokenizers behaviors under their existing Apache-2.0 attribution. Retain this crate's other third-party notices.

The ordinary BLOOM converter fixture pack includes unmodified official tokenizer data and sidecars from **bigscience/bloom-560m**, revision **ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971**, distributed by BigScience under **BigScience BLOOM RAIL 1.0**. These data are not represented as MIT-licensed data. The complete original license is [LICENSE.BLOOM-RAIL.txt](tests/fixtures/bloom-conversion/LICENSE.BLOOM-RAIL.txt). The complete [original model card](tests/fixtures/bloom-conversion/MODEL_CARD.txt) is retained. File URLs, lengths, hashes and acquisition identities are recorded in [pretrained-reference.json](tests/fixtures/bloom-conversion/pretrained-reference.json).

The tensor inputs bundled in `pipeline.bin` are independently generated synthetic data. The model-card regression case intentionally contains the literal metadata value `license: mit`; it tests metadata handling and does not change the license of tokenizer data bundled with that case.

The genuine pretrained model weights and complete native GGUF outputs remain external. The ignored test requires the exact unchanged official files and preserves their provenance. No original Python/C++ implementation, source environment or trained model weights are included in the product.

See [BLOOM_HF_CONVERSION.md](../../docs/BLOOM_HF_CONVERSION.md) for the captured source contracts, complete-file checks, declared limits and reproduction command.
