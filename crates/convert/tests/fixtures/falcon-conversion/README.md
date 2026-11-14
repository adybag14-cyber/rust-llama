# Independent Falcon conversion fixtures

These are reference data, not implementation code. The authority is the
supplied llama.cpp archive identified in provenance.json. Source captures
preceded the Rust comparisons.

- pipeline.bin and pipeline.json hold 116 deduplicated synthetic input blobs
  and 117 requests per native profile. Accepted output identities refer to
  complete original GGUF files: 86 files / 202,713,961 bytes per host. The
  other 31 requests are original rejections.
- aliases.json records 510 calls to the frozen Falcon tensor name map.
- punctuation.bin records native Tokenizers 0.22.2 membership for every
  Unicode codepoint slot; surrogate slots are zero and are excluded from the
  1,112,064 scalar comparisons.
- tokenizer-base.json contains the unchanged vocabulary/model/decoder fields
  of the official tokenizer, serialized as JSON data. The pretokenizer stages
  in tokenizer-reference.json reproduce seven native variants and 8,745
  exact split/encode/decode requests.
- official-sidecars.json records immutable source URLs, Git blob identities,
  byte counts and SHA-256 values for the official Falcon 7B sidecars.
- MODEL_CARD.txt is the unchanged official model card.
- LICENSE.MIT.txt and LICENSE.APACHE-2.0.txt preserve source/data licenses.
- provenance.json records frozen source and native dependency identities.

The sidecar source is tiiuae/falcon-7b at revision
ec89142b67d748a1865ea4451372db8313ada0d8. No pretrained tensor weights are
included. Vocabulary-only reference success does not establish pretrained
Falcon model conversion or inference.

See docs/FALCON_HF_CONVERSION.md and crates/convert/FALCON_NOTICES.md.
