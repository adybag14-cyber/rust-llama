# Frozen Unicode and tokenizer fixtures

The files in this directory are independent expected data from the exact
original source ZIP identified by SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

## Unicode core

The external C++ oracle compiles the original, unmodified `unicode.cpp` and
`unicode-data.cpp`. It evaluates every one of the **1,114,112 codepoints** from
U+000000 through U+10FFFF, including source surrogate behavior. The expected
properties are compressed into consecutive runs without dropping codepoints.
Single-codepoint NFC results are represented by 1,120 nonidentity exceptions
and explicit identity expectations for all other codepoints.

The same oracle supplies **15,957 normalization sequences** covering original
decomposition/composition pairs, all Hangul syllable compositions, combining
class reordering and blocking, leading marks and empty input. All 256 byte
encoding mappings and UTF-8 leading-byte lengths are included.

`utf8-reference.bin` contains **65,809 byte-input cases**: every one-byte and
two-byte input, plus longer boundary, surrogate, truncated and malformed
sequences. It records first-decode success, the exact offset, first codepoint
and complete source replacement behavior.

`unicode-utf8-hashes.bin` covers the original encoder's output for every
codepoint. For each consecutive block of 256 codepoints it stores SHA-256 over
the source records, each an encoded-length byte followed by the encoded bytes.
There are 4,352 consecutive 32-byte hashes. The original full binary's identity
and the exact lossless comparison layout are in `provenance.json`.

The source's `normalize_nfd` operation maps each codepoint to its frozen base
value; it is intentionally distinct from a general canonical decomposition
stream. Source NFC performs full recursive canonical decomposition, stable
combining-class ordering, and composition, including algorithmic Hangul.
The Rust tests exercise both against their independent expected results.

## WordPiece

`ggml-vocab-bert-bge.gguf` is the **unmodified original vocabulary-only GGUF**,
627,549 bytes and zero model tensors. Its SHA-256 is
`fbcbe22278fb302694d5f4a41bfe48c5f90e8e3554eab1c0435387dff654a854`.
The upstream notice is retained in `LICENSE-UPSTREAM`.

The original public llama C API captures all **30,522 vocabulary entries**,
including text, score, attribute bits, both token-piece modes and EOG state.
It also captures **192 input/flag cases across nine metadata variants**:
1,728 exact tokenization cases, both piece modes, and all four combinations of
the detokenization flags.

The inputs cover scripts, accents and combining marks, case behavior, all
source whitespace classes, controls, NUL/replacement characters, punctuation,
contractions, emoji, special tokens, long words, failed-word rollback and
Chinese-range boundaries. Variants exercise lowercase, accent stripping,
cleanup, ignored NFC behavior, special insertion flags, mask stripping and
special IDs. The SEP test supplies both spellings: only the frozen source's
`tokenizer.ggml.seperator_token_id` affects the separator; the correctly
spelled `separator_token_id` is unknown to the original.

The full-vocabulary comparison also checks original quirks such as `[PAD]`
being inferred as FIM_PAD/EOG and combined control/unused attributes.
WordPiece retains the source's documented U+2B920 Chinese-range boundary.

## Special metadata boundary regression

A separate unchanged original-library capture adds **72 SPM/WordPiece cases**
across default, UINT32_MAX and exact vocabulary-size overrides for all special
IDs. The original loader ignores out-of-range overrides and retains an in-range
family default, or disables an out-of-range default. These cases include BOS,
EOS, UNK, SEP, PAD, MASK, EOT, EOM and the three FIM end markers.

The combined original requests and results are in
`special-metadata-reference.json`, with separate
`special-metadata-provenance.json`. They compare resolved public IDs, exact
token IDs, both piece modes, attributes and all four detokenization flag modes.
This ordinary offline regression catches UINT32_MAX optional metadata in real
pretrained SPM models without requiring those external model weights.

## Reproduction and limits

Run the ordinary, offline Rust tests:

```text
cargo test --locked --offline -p llama-tokenizer
cargo test --release --locked --offline -p llama-tokenizer
cargo test --locked --offline -p llama-inference --test native_reference original_spm_68
cargo xtask fixtures
```

The native oracle source/executable identities, original library hashes,
original source hashes, request and fixture hashes are recorded in
`provenance.json`. Original C/C++ programs and libraries are external
comparison tools and are not maintained product implementations. No Rust
output is used to generate the expected results.

This checkpoint covers the complete frozen Unicode data and its property,
case, normalization and UTF-8/byte-codec functions, together with WordPiece.
The separate BPE-specific regex/pretokenizer dispatcher is not implemented by
these fixtures or by WordPiece support.
