# Independent StarCoder conversion fixtures

The authority is the frozen llama.cpp archive identified in provenance.json.
Original source execution produced the expected bytes before Rust comparison.

- pipeline.bin contains 176 deduplicated input blobs
  (38,284,433 bytes). Numerical tensor data is synthetic.
  Official GPT-2, StarCoder and StarCoder2 tokenizer data is included.
- pipeline.json describes 194 complete requests per native
  profile: 163 accepted files / 357,723,535
  bytes and 31 original failures. Every input blob and
  complete output has a length and SHA-256 identity.
- aliases.json records 1,248 original tensor-map queries. The independently
  captured Windows and GNU requests/results agree exactly after JSON newline
  normalization; provenance retains both raw capture hashes.
- pretrained-reference.json records complete original F16/Q8_0/auto hashes
  for the unchanged tiny StarCoder checkpoint on both native profiles.
  Its three files total 881,027,328 bytes per host.
- tiny-starcoder-acquisition.json and starcoder2-acquisition.json retain
  pinned official URLs, Git/LFS object identities and downloaded file hashes.
- TINY_MODEL_CARD.txt and STARCODER2_MODEL_CARD.txt retain official model cards.
- gpt2-sidecar-provenance.json and LICENSE.GPT2.txt retain the reused GPT-2
  tokenizer source identities and its separate Modified MIT notice.
- LICENSE.MIT.txt, LICENSE.APACHE-2.0.txt and
  LICENSE.BIGCODE-OPENRAIL-M.txt retain source/data licenses.
  license-provenance.json identifies the official BigCode license source.
- provenance.json records frozen source files, native package environments
  and original capture identities.

No pretrained tensor weights or generated GGUF files are bundled. The
ordinary StarCoder2 corpus uses the official sidecars with synthetic weights;
it does not establish unchanged pretrained HF weight conversion.

See docs/STARCODER_HF_CONVERSION.md and crates/convert/STARCODER_NOTICES.md.
