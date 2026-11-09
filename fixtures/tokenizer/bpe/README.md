# Frozen BPE and byte-input reference data

These fixtures compare the Rust tokenizer with the original implementation from source archive SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
The original C++ sources, executables and libraries remain outside the product.
Both Windows/MSVC and Linux/GCC native captures are recorded in provenance.

## Coverage

| Fixture | Independent coverage |
| --- | --- |
| regex-reference.json | 40 pretokenizer pipelines and all 48 constituent expressions, 1,442 focused/random inputs, 126,896 exact splits per platform; exhaustive native wide-regex digit/word/space classification |
| synthetic-reference.json | All 64 source pretokenizer types and 99 metadata spellings, plus 12 model/session/metadata variants: 23,754 cases and 18,381 complete vocabulary records |
| vocabulary-reference.json | All 14 BPE vocabularies supplied by the frozen source, every vocabulary record, and 364 input/flag cases per model |
| gpt2-profiles-reference.json | The original 50,257-token GPT-2 vocabulary and actual merge table under all 99 metadata spellings: 18,018 cases |
| raw-bpe-reference.json | Seven BPE/session variants on 1,251 raw-byte inputs and both flag combinations: 17,514 cases |
| raw-spm-reference.json | The original 32,000-token SPM vocabulary on the same raw inputs: 2,502 cases |
| special-order-reference.json | Ten native metadata/precedence variants over all 51 inferred EOT/EOM/FIM spellings; 5,680 full records and 40 tokenization cases |
| map-reference.json | Exact native string hashes, reserve bucket counts and iteration order for 5,970 keys across 20 table sizes per platform; libstdc++ reserve sequence through one million entries |
| raw-wpm-reference.json | The original 30,522-token BERT WordPiece vocabulary on the same raw inputs: 2,502 cases |

The raw corpus includes every possible single byte, invalid and truncated sequences, accepted surrogate encodings, special-token boundaries, merge boundaries, random byte strings, and UTF-8 cut at the source DRY sampler's 40-byte boundary. The existing Unicode fixture separately checks all one- and two-byte decoder inputs.

The 128,192-byte synthetic GGUF contains 6,127 tokens, 124 merge entries, all byte fallbacks and DNA k-mers. It has no model tensors. Its deterministic input generator never invokes Rust. The original GPT-2 vocabulary GGUF is copied unchanged from the frozen source archive; its upstream notice is retained in LICENSE-UPSTREAM. Other full BPE vocabularies remain optional external inputs, pinned individually by SHA-256.

## Framing of expected hashes

Hashes are a compact representation of original native output, not tolerances or text-only summaries. All lengths and integer fields below are little endian. A blob is a u32 byte length followed by exactly those bytes.

A case record contains a u32 token count; then, for each token, its i32 ID, u32 attributes, a blob for its ordinary piece and a blob for its special piece; then four detokenization blobs in this order: remove-special false/true, with unparse-special false/true inside each group. Each case is SHA-256 hashed. Identical hashes share an index in the fixture's hash table. Linux overrides are recorded only where original native results differ.

A vocabulary record contains a UTF-8 text blob, the exact four f32 score bits, u32 attributes, both piece blobs, and one EOG byte. Consecutive groups of up to 256 records are SHA-256 hashed without separators.

Special getters and vocabulary chunks also carry Linux overrides when the original results differ. The source uses the first matching entry from its native unordered vocabulary map. The original Llama-BPE model therefore reports inferred EOT 128009 on MSVC and 128001 on libstdc++; tokenization and the EOG set remain separately compared. The 10,144-byte special-order GGUF contains 568 tokens, no tensors and all candidate spellings to test this ambiguity and deprecated metadata precedence.

A map-order frame contains u32 entry count and bucket count, one u64 native string hash per insertion, then the i32 IDs in native iteration order. Test keys are `key_{id}:é汉🦀`, with a NUL and `tail` appended when id is divisible by 37. Each complete group has a native SHA-256 digest. The recorded x64 MSVC and libstdc++ policies are explicit; other C++ standard-library profiles are not claimed as measured.

A regex case contains a u32 piece count followed by length-framed piece blobs. Groups of up to 64 consecutive cases are hashed. Separate Windows and Linux hashes preserve differences in the source's default-locale wide regular expressions. Empty pieces are retained. The wide-class fixture hash covers an eight-byte WCLS header and one classification byte for every native wchar value: digit bit 0, word bit 1, whitespace bit 2.

The compacting script reads only original native captures. Provenance records the full capture hashes, source/generator hashes, model identities and native libraries.

## Reproduction and validation

Run ordinary deterministic coverage with:

```text
cargo test --offline -p llama-tokenizer
cargo test --offline -p llama-tokenizer --release
```

The two additional corpus tests are explicit because one reloads the full original GPT-2 merge table under 99 profiles, and the other requires the original source vocabulary directory:

```text
cargo test --offline -p llama-tokenizer --release --test bpe_reference original_gpt2_merges_across_all_99_metadata_aliases -- --ignored
# Set LLAMA_TEST_BPE_VOCAB_DIR to the frozen source models directory.
cargo test --offline -p llama-tokenizer --release --test bpe_reference every_original_bpe_vocabulary_and_merge_corpus -- --ignored
```

No expected data is regenerated from the Rust implementation. The model hash is checked before any optional full-vocabulary comparison.
