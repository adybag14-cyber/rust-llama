# Frozen remaining-tokenizer conformance fixtures

These fixtures compare the Rust tokenizer directly with the frozen original
implementation. They cover Unigram (t5), RWKV, PLaMo-2, PLaMo-3, the source
test family, none/no_vocab metadata, and the remaining original SPM
vocabularies. The source archive SHA-256 is
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.

Expected values were captured from the original Windows DLL and Linux shared
library, or from the exact original Unigram normalization functions compiled
outside the product. The compactor only frames and hashes those native values.
It does not execute Rust or use Rust results as expected outputs. All product
implementation and test code is Rust; no native implementation, native binary,
or foreign-language generator is included here.

## Coverage and inputs

| Fixture | Independent coverage |
| --- | --- |
| Six synthetic vocabulary-only GGUFs and synthetic-reference.json | 91,476 tokenization cases, both piece modes, all four detokenization flag combinations, 3,507 complete vocabulary records, family options and raw-byte input |
| original-vocabulary-reference.json | All 250,048 original Nomic Unigram entries, 64,000 Baichuan entries and 32,064 Phi-3 entries; 10,412 real-vocabulary cases |
| normalizer-inputs.bin and normalizer-reference.json | 4,067,668 exact original normalization outputs, including every codepoint and every original XCDA mapping prefix |
| metadata-models.bin and metadata-reference.json | 176 native metadata cases: 70 successful loads, 103 checked native errors, two native assertions and one native non-progress loop; 23,903 full vocabulary records and 480 tokenization outputs |

The synthetic GGUFs have zero tensors and contain no model weights.
ugm-synthetic.gguf includes the complete 237,539-byte precompiled charsmap from
the original models/ggml-vocab-nomic-bert-moe.gguf. Its byte identity is
0942789e0111452f1cc446f70036e26d24458c3c7858c53cc12ca4d87531279b.
The signed-charsmap fixture encodes the same bytes as INT8; the third Unigram
fixture omits the map. Other metadata, tokens and scores are deterministic test
inputs designed to exercise original behavior.

The complete original Nomic, Baichuan and Phi-3 vocabulary GGUFs remain external.
The explicit test checks their SHA-256 and byte size against the native corpus
before using them. Their compact expected records are included here; their
availability is not assumed by ordinary offline tests.

### Behavioral details

The Unigram corpus checks double-precision Viterbi accumulation, strict score
ties, unused/user-defined token handling, consecutive-UNK merging, original
whitespace normalization, NFC, all source charsmap data and malformed UTF-8.
The protected-prefix cases retain the source trie quirk that protects a
traversed prefix even when it stops before a terminal user-defined token.

RWKV cases include all raw bytes, escaped aliases, lowercase-only hexadecimal
escape arithmetic, incomplete escapes, longest matches, duplicate decoded
spellings and the metadata flags its source session ignores. An unmatched byte
emits the resolved UNK ID, including the source -1 sentinel. Newline resolution
uses the pre-override source default.

PLaMo uses the original reverse-suffix table and backward integer-scored
dynamic program. Cases cover virtual suffixes, double-promoted score rounding,
score-sentinel collisions, ties, byte fallback, BOM handling and PLaMo-3
presegmentation: special-looking literals, repeated runs, spaces and U+EE00
boundaries. NaN scores retain the original invalid-row behavior.

Metadata cases verify exact GGUF array/scalar types, out-of-range token-type
normalization, embedded NUL handling, UINT32 ID rules, ignored settings,
suppressed-token order/duplicates/range filtering, charsmap bounds, incomplete
RWKV vocabularies and PLaMo byte requirements. The none family keeps its source
count and special getters; Rust returns checked unsupported errors where the
original asserts on text, score, attributes, tokenization or token pieces.
Detokenization for that family remains empty.

The duplicate-vocabulary native assertions become Rust load errors. The
nonterminal RWKV prefix that loops in the original becomes a Rust error without
losing progress. Native assertion/timeout outcomes are recorded separately from
ordinary load errors; they are not advertised as matching native crashes.

## Exact framing

All integer framing uses little endian and all lengths are unsigned 32-bit byte
counts. blob(B) is u32(length(B)) followed by B.

Synthetic and original-vocabulary captures use the same schema as the BPE
fixtures. Each tokenization case hashes:

1. The token count as u32.
2. For each token: its i32 ID, u32 attributes, blob(plain_piece) and
   blob(special_piece).
3. Four detokenization blobs in the order (remove_special, unparse_special) =
   (false,false), (false,true), (true,false), (true,true).

Vocabulary hashes frame each entry as its UTF-8 text blob, f32 score bits,
u32 attributes, both piece blobs and one EOG byte. Chunks contain 256 entries.
Case hashes are deduplicated through an indexed hash table. The synthetic
matrix shares its 3,388 inputs across six models; any platform-specific
metadata, case or vocabulary hash remains explicitly represented.

The normalization input file starts with UINP0001, followed by a u32 count
and input blobs. Its 306,574 inputs comprise all 224,711 original XCDA mapping
keys, all 65,792 one/two-byte inputs, 15,957 NFC/NFD sequence inputs and 114
protected-prefix/boundary cases. Two additional groups generate every
codepoint from zero through U+10FFFF using the frozen UTF-8 codec, including
source surrogate behavior. Expected normalized output blobs are hashed in
groups of 1,024. The raw Windows and Linux captures are byte-identical:
42,550,364 bytes, SHA-256
6eadd59c0f823f4cdbb0501655ff040362fb9fcf161c4ab5493274144dacac14.

The metadata input pack starts with TMET0001, a u32 count, then a case-name
blob and exact-GGUF blob for each case. The companion JSON stores the native
outcome and, for successful loads, metadata, exact token IDs and one complete
vocabulary hash. That hash uses text, f32 score bits, attributes, both piece
blobs and the EOG byte in the same framing above. No-vocabulary models have
zero queryable vocabulary records even when their declared dummy count is nonzero.

## Running the checks

Ordinary checks require only this repository:

    cargo test --locked --offline -p llama-tokenizer
    cargo test --release --locked --offline -p llama-tokenizer
    cargo clippy --locked --offline -p llama-tokenizer --all-targets -- -D warnings
    cargo xtask fixtures

Set LLAMA_TEST_TOKENIZER_VOCAB_DIR to the frozen source models directory,
then run the complete original vocabularies explicitly:

    cargo test --release --locked --offline -p llama-tokenizer --test families_reference all_original_unigram_and_remaining_spm_vocabularies -- --ignored --nocapture

LLAMA_TEST_BPE_VOCAB_DIR is accepted as the same external-input location.
The original BPE corpus is still checked independently by bpe_reference.

The implementation has explicit input, output, trie, suffix-table and work
budgets. Out-of-range PLaMo integer scores, non-finite SPM merge scores and
malformed byte labels retain checked validation rather than native undefined
or permissive numeric-conversion behavior. These limits do not alter the
captured supported cases. Adding tokenizer families does not implement the
corresponding model architectures or hardware backends.

provenance.json identifies all original sources, native libraries, external
drivers, raw captures, input models and compact fixtures. The original archive
and retained LICENSE-UPSTREAM provide upstream provenance; generated synthetic
inputs contain no pretrained weights.
