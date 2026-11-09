# Frozen Unicode and WordPiece validation

This checkpoint adds the complete frozen Unicode data/core functions and BERT
WordPiece to the Rust tokenizer. Expected data comes from original source
functions and original llama C API calls; the external C/C++ oracles are not
part of the product.

## Final Windows results

All captures use the pinned native Rust 1.99.1 toolchain. Each capture's JSON
records the exact command, exit code, elapsed time and stdout/stderr SHA-256.

| Capture stem | Result |
| --- | --- |
| `sentinel-debug` | 11 tokenizer tests passed: five existing unit tests and six independent native reference tests |
| `sentinel-release` | The same 11 tests passed in release |
| `sentinel-clippy` | All tokenizer targets passed with warnings denied |
| `spm-final` | Original SPM corpus and both explicit external pretrained model tests passed |
| `fixture-inventory-final` | 109 fixture files covered by 109 declarations in four manifests |
| `source-audit-final` | Source-role and foreign-implementation audit passed; unsafe candidates remain separately reported for review |

The independent tokenizer comparisons cover every one of 1,114,112
codepoints, 65,809 raw UTF-8 inputs, 15,957 normalization sequences, all 256
byte mappings, all 30,522 BERT vocabulary entries, 1,728 WordPiece cases
across nine variants, and 72 additional special-metadata boundary cases
across SPM and WordPiece. All IDs, byte pieces, attributes and detokenized
results use exact comparisons.

The preserved SPM corpus has 68 cases and 312 pieces in both special modes.
The actual stories260K model comparison covers 62,976 logits and 96 greedy
IDs; the actual stories15M Q4_0 comparison covers 640,000 logits and 16 greedy
IDs. Both have maximum absolute error **zero** in `spm-final`.
The portable tests still enforce their declared fixed absolute/relative
tolerance rather than requiring floating-point identity across all platforms.

These pretrained models remain external, optional test inputs supplied through
`LLAMA_TEST_MODEL` and `LLAMA_TEST_Q4_MODEL`. The ordinary sentinel regression
requires only the bundled original vocabulary-only fixtures.

## Corrections found by independent comparisons

Earlier captures are retained as development evidence:

- `initial-native-debug` found that the original BERT loader also treats
  `[PAD]` as inferred FIM_PAD/EOG.
- `wordpiece-after-special-attributes` exposed the frozen source's
  `seperator_token_id` spelling. The oracle case was strengthened to supply
  both spellings, showing that only the source spelling applies.
- Actual-context integration exposed UINT32_MAX special-ID sentinels in SPM
  metadata. Source-compatible fallback now applies uniformly to special IDs,
  with 72 independently captured ordinary regression cases.

The final captures supersede those intermediate implementation failures; none
were waived.

## Scope and source identity

The original ZIP SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Exact source, native DLL, driver, executable, vocabulary and fixture hashes
are in `fixtures/tokenizer/provenance.json` and
`fixtures/tokenizer/special-metadata-provenance.json`.

The maintained implementation is confined to Rust. The Unicode module retains
frozen category/case/NFD/NFC/UTF-8/byte-codec behavior and source quirks. This
checkpoint does not claim the separate BPE regex/pretokenizer dispatcher or
other tokenizer families. See `crates/tokenizer/README.md` for the public
API and detailed compatibility behavior.
