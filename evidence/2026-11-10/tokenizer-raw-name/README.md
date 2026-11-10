# Raw model-name tokenizer correction evidence

The new GGUF model view preserves non-UTF-8 STRING bytes. Review found that
the tokenizer still read general.name through the UTF-8 accessor and silently
substituted an empty string. A raw name containing an ASCII phi3/phi-3 or
modern-bert marker therefore skipped native RSTRIP/LSTRIP token attributes
and changed whitespace tokenization.

The fix borrows the first-NUL byte prefix and performs ASCII-insensitive
marker matching without allocating a lowercase name. It preserves existing
text behavior and raw bytes. The exact final source identities are recorded
in summary.json.

## Independent reference and regression

Both original MSVC and GNU libraries produced the same 38-model result before
the Rust change: 19,456 token attributes and 6,384 tokenizations, including
every token piece. The native model names contain only ASCII, NUL and 0xff;
0xff is the defined EOF argument to signed-char std::tolower. Other negative
char arguments are not used to claim native conformance.

The new test was run against the old implementation first. It failed on
phi3-ff-prefix, token 2: Rust attributes 8 versus native 264. The red log is
retained. Native controls show 62 changed tokenizations for each Phi marker
and 24 for ModernBERT compared with a plain model, so this is an observable
tokenization correction rather than only an internal attribute adjustment.

The frozen fixture data, input identities, original seal and DLL/SO/caller
identities are in fixtures/tokenizer/raw-name. Original full native captures,
requests and the standalone C++ caller remain outside the product under
rust-llama-reference/continue-20261110/tokenizer-raw-name-oracle.

## Final component validation

Windows and GNU each passed the new regression in debug and release. Every
test replays the entire original corpus through both the GgufSet C-prefix
model projection and the direct tokenizer constructor with unprojected raw
name bytes. Both hosts also passed formatting checks and strict tokenizer
Clippy with all targets and -D warnings. The final input maps are identical
across hosts and were rechecked unchanged afterward.

The component test command deliberately filters to this independent regression;
the broader suite is covered by the integration validation. No tests failed
or were ignored in the final component runs.

The first Windows component run had already passed both profiles, then Clippy
requested the test helper's constant chunks_exact call use as_chunks. The
first GNU run could not resolve the converter's locked yaml-rust2 version
from the default offline cache. The subsequent GNU run used the converter's
independently audited cache. Both attempt directories are retained; no native
expected output, model input or lockfile was changed to resolve these issues.

An external replay-final.py helper accepts HOST FINAL_LIBRARY NEW_OUTPUT_DIR
and repeats all 38 frozen C-header clients against a combined Rust DLL/SO,
checking complete native JSON results and immutable hashes. Final combined
library replays are part of the parent integration's evidence.

This addresses the source model-name marker behavior in the C locale.
It does not close complete tokenizer, metadata, locale or full-port gates.
