# Raw model-name tokenizer fixtures

The frozen native vocabulary loader reads general.name as a first-NUL byte
string, performs C-locale lowercasing and searches for phi3, phi-3 and
modern-bert markers. These markers alter special-token RSTRIP/LSTRIP
attributes and whitespace tokenization.

The 38 synthetic models exercise missing/empty names, unrelated bytes,
ASCII case, byte 0xff before/after/inside markers, first-NUL truncation,
and Phi/ModernBERT precedence. Every non-ASCII name byte is 0xff: on the
measured signed-char platforms this promotes to EOF, a defined argument
to std::tolower. Bytes 0x80 through 0xfe are outside that source function's
defined domain and are not used to claim cross-platform conformance.

All models derive from fixtures/ffi/native-context-model.gguf. Three final
normal tokens are renamed to [MASK], <|endoftext|>, and [user], with token
types USER_DEFINED, CONTROL, and USER_DEFINED. Each model then changes only
the general.name value or omits it. Tensor directory and payload bytes stay
unchanged; the independent caller loads vocabulary-only.

## Independent observations

A separate C++17 caller compiled against the unchanged frozen llama.h loads
the original Windows/MSVC DLL and GNU/Linux SO directly. It records all
512 token attributes and 42 input texts with all four add/parse-special
flag combinations per model, including token IDs and every rendered piece.

The resulting 19,456 attributes and 6,384 tokenizations are identical across
both original platforms. They were sealed before changing Rust. The original
Rust regression failed on phi3-ff-prefix: token 2 had attributes 8 instead
of the native 264. Phi controls differ from a plain model in 62 tokenizations
per marker, and ModernBERT controls differ in 24, confirming real whitespace
behavior changes.

- models/: exact raw GGUF inputs, including non-UTF-8 name bytes.
- original.json: losslessly compacted native attribute/tokenization records.
- input-manifest.json: input identities, construction, and frozen source hashes.
- original-seal.json: pre-fix original result and candidate preimage identities.
- provenance.json: original DLL/SO, caller and external tool identities.
- fixture-manifest.tsv: checked file byte counts and SHA-256 hashes.

The Rust test replays the expected observations through both GgufSet's model
C-prefix metadata projection and the public tokenizer constructor with raw
unprojected name bytes. It also checks every model fixture hash. Existing
ASCII semantics are unchanged; no lossy decoding or name-sized allocation
is used for marker matching.

Native caller sources, complete metadata/buffer captures, request files and
build artifacts remain outside the product under the recorded reference
directory. No original library is linked or loaded by the Rust product or
ordinary Rust test suite. The upstream MIT notice applies to the source-derived
identifiers and synthetic fixture lineage; no external model weights or private
conversation data are introduced.

This is focused evidence for model-name tokenizer behavior in the C locale.
It does not close all tokenizer, metadata, locale or complete-port release gates.
