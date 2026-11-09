# llama-tokenizer

Rust implementation of the frozen source's `llama` SentencePiece-style scored
BPE and `bert` WordPiece vocabulary families. The crate forbids unsafe Rust,
uses no native library, and depends at runtime only on the workspace GGUF
metadata types. Unsupported vocabulary families and precompiled normalization
maps return explicit errors.

## Public interface

Construct a tokenizer from `Gguf::metadata` using
`Tokenizer::from_metadata(&metadata)`. Construction validates vocabulary
entries, scores, token types and relevant metadata.

| Method | Behavior |
| --- | --- |
| `tokenize(text, add_special, parse_special)` | Returns exact token IDs; the insertion and literal-special parsing flags are independent. |
| `token_piece(id, special)` | Returns bytes, preserving partial UTF-8 and source special-token visibility. |
| `detokenize_bytes(ids, remove_special, unparse_special)` | Joins pieces with the source's leading-space policy and optional ordered cleanup passes. |
| `token_info(id)` | Returns text, score and the normalized legacy token kind. |
| `token_attributes(id)` | Returns the original public attribute bitset, including combined control/unused and strip flags. |
| `vocab_type()` | Returns `VocabType::Spm = 1` or `VocabType::Wpm = 3`, matching frozen C ABI values. |
| `is_end_of_generation(id)` | Checks the source's configured and inferred end markers. |

The public BOS, EOS, UNK, SEP, PAD and MASK IDs use `-1` when unavailable.
Out-of-range metadata overrides, including UINT32_MAX, retain the source's
in-range family default; defaults outside the vocabulary are disabled.
The `add_bos`, `add_eos`, `add_sep`, `add_space_prefix` and `clean_spaces`
fields expose resolved metadata behavior. Input is limited to 4 MiB, vocabulary
size to one million entries, and detokenized output to 64 MiB.

### WordPiece details

WordPiece uses the complete frozen Unicode category, lowercase and NFD-base
tables. It handles whitespace, controls, accents, punctuation and Chinese
splitting before greedy vocabulary segmentation. If any segment of a word
cannot be matched, it rolls back the entire word and emits UNK. Long words
follow the source algorithm without an extra 100-character restriction.

Special literals are partitioned before normal word processing. USER_DEFINED
literals remain active when `parse_special` is false. Modern BERT mask tokens
retain their source left-strip attribute.

Source compatibility includes these observable behaviors:

- `add_special` inserts BOS and SEP for WordPiece even if its exposed
  `add_bos` or `add_sep` metadata flag is false.
- The separator metadata key is spelled
  `tokenizer.ggml.seperator_token_id` in the frozen source.
- The WordPiece Chinese range begins at U+2B920 for the relevant extension;
  the general Han helper uses its separate source range list.
- The source WordPiece path ignores the NFC metadata flag.
- `[PAD]` can be inferred as FIM_PAD and therefore an end-of-generation token.
- Detokenization preserves the exact ordered punctuation and apostrophe cleanup
  passes.

These behaviors have independent original-library cases, rather than being
inferred only from expected modern tokenizer behavior.

## Frozen Unicode core

The public `unicode` module contains the complete translated source tables
and implements:

- category and whitespace/case/NFD flags;
- single-codepoint lowercase and uppercase mappings;
- the source's one-codepoint NFD-base mapping;
- full canonical NFC decomposition, stable combining-class ordering and
  composition, including Hangul;
- both source Han/WordPiece Chinese predicates;
- UTF-8 leading lengths, encoding, decoding, replacement and offset behavior;
- all 256 byte-to-codepoint/UTF-8 mappings and inverse mappings.

The code uses frozen tables rather than the Rust toolchain's changing Unicode
data. Byte-oriented UTF-8 helpers intentionally retain the source's surrogate
encoding and decoding behavior; Rust string tokenization still accepts
`&str`. Out-of-range codepoints and malformed sequences return errors or use
the original per-byte replacement policy, according to the selected helper.

The BPE-specific regular-expression splitting dispatcher is a separate
unimplemented requirement. Unicode core support does not advertise the other
vocabulary families or their pretokenizers.

## Independent validation

Ordinary offline tests include the original BERT vocabulary-only GGUF and
data generated exclusively by original C/C++ functions outside the product:

| Comparison | Coverage |
| --- | --- |
| Unicode properties, case, NFD, Han and single-codepoint NFC | All 1,114,112 codepoints plus out-of-range cases |
| UTF-8 encoding | Every codepoint, including source surrogate behavior |
| Raw UTF-8 decoding | 65,809 cases, including every one-byte and two-byte input |
| Normalization | 15,957 sequences, including all Hangul compositions and source table pairs |
| Byte codecs | All 256 mappings and leading-byte lengths |
| WordPiece vocabulary | All 30,522 entries: text, score, attributes, EOG and both piece modes |
| WordPiece corpus | 1,728 cases across nine metadata variants, both piece modes and four detokenization flag combinations |
| Special metadata boundaries | 72 native SPM/WordPiece cases, including UINT32_MAX and exact vocabulary-size overrides |
| Preserved SPM corpus | 68 original cases and 312 pieces through the inference integration test |

Run:

```text
cargo test --locked --offline -p llama-tokenizer
cargo test --release --locked --offline -p llama-tokenizer
cargo clippy --locked --offline -p llama-tokenizer --all-targets -- -D warnings
cargo test --locked --offline -p llama-inference --test native_reference original_spm_68
cargo xtask fixtures
```

See `fixtures/tokenizer/README.md` and `fixtures/tokenizer/provenance.json`
for exact original source, library, driver, vocabulary and fixture identities.
The frozen source archive and retained `LICENSE-UPSTREAM` define the data
provenance.
