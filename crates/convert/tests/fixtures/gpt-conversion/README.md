# GPT conversion reference data

These files retain independent frozen GPT-2/GPT-NeoX converter expectations:

- `pipeline.bin`: 91 deduplicated input/tokenizer blobs, with synthetic tensors.
- `pipeline.json`: 101 cases per native host with complete expected file hashes.
- `provenance.json`: source identities, original captures and pinned environments.
- `neox-class-reference.json`: original Fast-class admission and 960 encode/decode expectations.
- `neox-class-provenance.json`: both native capture identities and the exact error-text exclusion.
- `pretrained-reference.json`: pinned source inputs, six accepted full output files and three source rejections per host.
- `gpt2-derived-input-provenance.json`: twelve removed masks and all 148 preserved learned tensors.
- `LICENSE.GPT2.txt` and `LICENSE.PYTHIA.txt`: original notices for official tokenizer bytes included in the pack.

The synthetic inputs deliberately combine small tensor row counts with full
official vocabularies. They test conversion, not miniature inference.
Expected complete-file hashes originate only from the original converter.
Rust never generates its own expected data. Source runtimes, Python capture
harnesses, pretrained checkpoints and full output files remain external.

See [GPT_HF_CONVERSION.md](../../../../../docs/GPT_HF_CONVERSION.md) for source
contracts, input adaptation and reproduction. The ordinary converter preserves
the GPT-2 static-mask failure. A separate ignored Rust test helper can prepare
the explicitly disclosed derivative from the pinned official input for CI;
its whole-file hash must match the independently prepared reference input.
