# llama-tokenizer

Rust implementation of every tokenizing family in the frozen source: llama/SPM,
bert/WordPiece, the metadata-selected BPE families, t5/Unigram, RWKV, PLaMo-2,
PLaMo-3 and the source test tokenizer, plus none/no_vocab metadata. The crate
forbids unsafe Rust, links no native library and depends at runtime only on the
workspace GGUF metadata types. Unknown families and profiles return explicit
errors; resource and malformed-input boundaries are checked.

## Public interface

Construct a tokenizer from `Gguf::metadata` using
`Tokenizer::from_metadata(&metadata)`. Construction validates vocabulary
entries, scores, token types and relevant metadata.

| Method | Behavior |
| --- | --- |
| `tokenize(text, add_special, parse_special)` | Returns exact token IDs; the insertion and literal-special parsing flags are independent. |
| `tokenize_bytes(bytes, add_special, parse_special)` | Preserves the source's raw-byte input behavior, including malformed UTF-8 and truncated sampler breakers. |
| `newline_id()` | Returns the source newline ID resolved before optional metadata overrides. |
| `metaspace_prepend_first()` | Exposes the source BPE first-rendered-piece space policy for C buffer sizing. |
| `token_piece(id, special)` | Returns bytes, preserving partial UTF-8 and source special-token visibility. |
| `detokenize_bytes(ids, remove_special, unparse_special)` | Joins pieces with the source's leading-space policy and optional ordered cleanup passes. |
| `token_info(id)` | Returns text, score and the normalized legacy token kind. |
| `token_attributes(id)` | Returns the original public attribute bitset, including combined control/unused and strip flags. |
| `vocab_type()` | Returns the frozen C ABI value: None 0, Spm 1, Bpe 2, Wpm 3, Ugm 4, Rwkv 5, Plamo2 6, Test 7 or Plamo3 8. |
| `suppressed_tokens()` | Returns configured IDs in source order, preserving duplicates and dropping out-of-range entries. |
| `token_type_count()` | Returns the typed source metadata count, including its zero default. |
| `is_end_of_generation(id)` | Checks the source's configured and inferred end markers. |

The public BOS, EOS, EOT, EOM, UNK, SEP, PAD, MASK and six FIM IDs use `-1` when unavailable.
GGUF token IDs require UINT32; token strings, scores and token-type arrays use
the source's exact element types. Out-of-range metadata overrides, including
UINT32_MAX, retain the source's
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
encoding and decoding behavior. Both Rust string and raw-byte tokenization are
available. Out-of-range codepoints and malformed sequences return errors or use
the original per-byte replacement policy, according to the selected helper.

## BPE pretokenizers and sessions

All 64 pretokenizer types and 99 metadata spellings from the frozen source are
explicitly dispatched. Merge ranks come from GGUF, equal-rank candidates use
the source leftmost tie break, duplicate merge pairs keep their first rank,
and profiles with ignore-merges retain their direct-token fast path.

The implementation includes byte-encoded GPT-2/Llama/Qwen-style profiles,
raw Gemma 4 and MMBERT, Nemotron normalization, Soprano normalization and
fallback, whitespace sessions, and HybridDNA six-base k-mers with source
tagging, padding and auto-detection. Metaspace first/never, forced Gemma 4 BOS,
normalizer options, EOG metadata and per-token stripping follow the source.
Unknown pretokenizer names and the source-rejected BPE metaspace `always`
setting return explicit errors.

Ambiguous inferred special tokens retain the frozen x64 native container order: MSVC and libstdc++ can select different IDs. Configured metadata takes precedence, with deprecated FIM keys applied in source order. The native map/hash corpus and a dedicated ambiguous-vocabulary corpus verify these accessors and their effects on token attributes.

The regular-expression engine implements the exact frozen expression set,
including lookahead and zero-length matches. It uses the frozen Unicode
properties, all eight source custom splitters, and measured default-locale
wide-regex classes. Windows and Linux retain their source differences:
Windows wchar values truncate to 16 bits, and its native digit class includes
additional BMP characters; Linux's default wide digit class is ASCII.

Public input is bounded to 4 MiB; derived normalized input can expand to
12 MiB plus one prefix. The regex VM has explicit instruction/work bounds
and returns a limit error if a pathological input exhausts its budget.
It does not silently switch to another tokenizer. Raw BPE/WPM inputs use
the source decoder's per-byte replacement and accepted-surrogate behavior;
SPM preserves source byte segmentation and fallback.

## Unigram, RWKV, PLaMo and vocabulary-free models

Unigram uses the complete precompiled charsmap supplied by GGUF. Its bounded
XCDA reader accepts the source INT8/UINT8 representations and preserves the
source normalization, whitespace and protected-prefix rules. Viterbi scores
accumulate as f64, use strict comparisons for ties, and merge consecutive UNK
results. The native corpus checks every mapping in the original Nomic map and
4,067,668 normalization outputs, including all Unicode codepoints.

RWKV unescapes source byte spellings and uses longest-prefix matching. Aliases
that decode to the same bytes retain the last ID. Its escape handling preserves
the source's unusual uppercase/malformed hexadecimal arithmetic. Unmatched
bytes return the configured UNK ID, including -1 when no UNK is available.
A nonterminal prefix that prevents native progress returns a checked error.

PLaMo-2 and PLaMo-3 share the original reverse-suffix table and backward
integer-scored dynamic program. Scores are promoted to f64 before scaling and
rounding; virtual suffixes and source score sentinels remain distinct. PLaMo-2
strips a leading BOM for each raw fragment. PLaMo-3 preserves it and applies
the source literal/run/space/U+EE00 presegmentation. All 256 byte fallback
tokens must exist at nonzero IDs.

The source test tokenizer retains its five-byte hashing and hexadecimal
token-piece behavior. The none/no_vocab family keeps dummy vocabulary counts
and special metadata getters. Its tokenization, text, score, attributes and
piece accessors report absence/unsupported behavior where the native code
asserts; detokenization returns empty.

These families use bounded tries, normalized buffers, suffix tables and work
budgets. The ordinary input limit remains 4 MiB. Unigram normalized output is
limited to 16 MiB, the charsmap to 64 MiB, tries/tables to eight million nodes
or rows, and Unigram/PLaMo traversal to 128 million steps. PLaMo suffix data
is limited to 256 MiB. Non-finite SPM merge scores, out-of-range PLaMo integer
scores and malformed byte labels retain explicit validation errors.

Adding a tokenizer does not implement its model architecture or hardware
backend in the inference loader.

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
| BPE pipelines and expressions | 126,896 native splits per platform, all 64 profiles / 99 aliases, and exhaustive source wide-regex classes |
| BPE sessions and options | 23,754 cases across 111 variants, plus 18,381 full vocabulary records |
| Original GPT-2 vocabulary | All 50,257 entries and 364 cases with the actual frozen merge table |
| Raw tokenization input | 22,518 native cases across SPM, WPM and seven BPE/session modes |
| Ambiguous special inference | All 51 inferred spellings, ten metadata variants, 5,680 complete records and native table ordering for 5,970 keys per platform |
| Unigram/RWKV/PLaMo/Test sessions | 91,476 native cases and 3,507 complete vocabulary records |
| Unigram normalization | 4,067,668 native outputs, every codepoint, all 224,711 XCDA prefixes and exhaustive one/two-byte inputs |
| Metadata and failure behavior | 176 native cases, 23,903 vocabulary records and 480 outputs; assertions/non-progress become checked errors |
| Explicit original Unigram/SPM corpus | 346,112 Nomic/Baichuan/Phi-3 records and 10,412 native cases using external original vocabularies |
| Preserved SPM corpus | 68 original cases and 312 pieces through the inference integration test |

Run:

```text
cargo test --locked --offline -p llama-tokenizer
cargo test --release --locked --offline -p llama-tokenizer
cargo clippy --locked --offline -p llama-tokenizer --all-targets -- -D warnings
cargo test --locked --offline -p llama-inference --test native_reference original_spm_68
cargo xtask fixtures
```

Additional explicit tests cover all 14 original BPE vocabularies and all 99
profiles using the original GPT-2 merge table. See `fixtures/tokenizer/bpe/README.md`
for the external-input commands and the exact framed-hash schema.
The remaining original vocabularies use LLAMA_TEST_TOKENIZER_VOCAB_DIR (or the
same BPE directory) and the ignored families_reference test; see
fixtures/tokenizer/families/README.md.

See `fixtures/tokenizer/README.md`, `fixtures/tokenizer/provenance.json` and
`fixtures/tokenizer/bpe/provenance.json` and
`fixtures/tokenizer/families/provenance.json`
for exact original source, library, driver, vocabulary and fixture identities.
The frozen source archive and retained `LICENSE-UPSTREAM` define the data
provenance.
