# HF Qwen tokenizer execution

This converter extension executes the exact Qwen Split expression and NFC
normalizer found in the pinned official Qwen2.5-0.5B-Instruct and Qwen3-0.6B
tokenizer JSON files. It adds no Cargo dependency and does not call a native
tokenizer, regular-expression engine, or host Unicode library.

## Exact execution domain

The accepted Split stage has `behavior = Isolated`, `invert = false`, and
the literal `QWEN_PATTERN` in `hf_schema.rs`. It preserves native alternative
order, case-insensitive contractions, an optional nonletter prefix before
letters, one Unicode number per match, punctuation/newline grouping, and
whitespace lookahead. Different patterns or Split modes remain explicitly
unsupported for execution. Existing reviewed schema admission remains separate
from executable support.

The casefold oracle scanned every valid Unicode scalar for the literal
contraction letters. Its only non-ASCII equivalence is U+017F for S/s. Letter,
Number and Space classification continues to use the existing exhaustive
Tokenizers 0.22.2 Oniguruma property capture, rather than Rust's Unicode tables.

NFC uses the canonical tables from Tokenizers 0.22.2's exact Cargo.lock
dependency, `unicode-normalization-alignments 0.1.12`. That dependency uses
**Unicode 9.0.0**, while the frozen llama runtime's normalization tables are
newer. Reusing the runtime tables would change combining-class behavior for
120 newer characters. The converter therefore retains its own 814 nonzero
combining classes, 2,060 fully decomposed canonical mappings, and 940
compositions, including all 12 astral pairs. Hangul decomposition and
composition are handled algorithmically.

The normalizer performs canonical decomposition, stable combining-class
ordering, blocked composition, and correct handling of leading nonstarters.
Sequences consisting of identity stages and NFC are supported; NFC is
idempotent. Other unimplemented normalization operations remain unsupported.
Added tokens are matched in the native order: raw patterns first, normalization
of remaining fragments next, then normalized patterns. The normalized pattern
strings use the same pinned NFC implementation, preserving original token IDs
and descriptor flags.

## Independent references

The Split, NFC and added-token reference outputs were captured before the implementation, using
Tokenizers 0.22.2 on both pinned Python reference environments. The compact
2,048-entry BPE fixture is an explicitly trimmed derivative of the official
Qwen vocabulary and merges; it is a primitive test fixture, not evidence of a
complete pretrained model conversion.

| Reference | Exact checks |
| --- | ---: |
| Split, ByteLevel, BPE IDs and decode, with/without NFC | 3,116 requests |
| NFC of every valid Unicode scalar | 1,112,064 scalars |
| Other NFC boundary, randomized, combining-mark and Hangul sequences | 5,645 sequences |
| Unicode 9 conformance inputs and later-CCC witnesses | 93,634 sequences |
| Normalized AddedToken flags and canonical collisions | 272 configurations / 4,896 requests |
| Unchanged official Qwen2.5 and Qwen3 tokenizers, both native profiles | 6,232 ID/decode requests |

The two profiles produced identical common fixture bytes. The unchanged
official tokenizer captures have identical request results and differ only in
their recorded source paths. The official checkpoints are pinned to
`7ae557604adf67be50417f59c2c2f167def9a775` (Qwen2.5) and
`c1899de289a04d12100db370d81485cdf75e47ca` (Qwen3).
Full tokenizer files remain in the external model cache and are hash-checked
before the ignored test runs.

A separate Windows Transformers 4.57.6 capture supplies 20 additional native
encodings for direct ByteLevel prefix, split-special-token and cleanup flags.
It was captured after the core helper implementation and before its own
regression test; the root converter integration applies those generic class
controls to Qwen as well.

The official replay is enabled with `LLAMA_TEST_QWEN_TOKENIZER_ROOT`, pointing
to the directory containing those two model directories:

```text
cargo test --release --offline --locked -p llama-convert --lib \
  pretrained_qwen_tokenizer_reference -- --ignored --nocapture
```

`crates/convert/tests/fixtures/hf-qwen/provenance.json` records exact input
hashes, capture reports, licenses, and fixture classification.
`evidence/2026-11-11/hf-qwen-tokenizer/` retains validation reports and logs,
including the initial Rust test-borrow compilation error. No expected native
outputs were altered during debugging.

## Resource and support boundaries

The NFC helper uses fallible allocation for its decomposition buffer, merge
scratch buffer, and UTF-8 output. A checked conservative bound covers its
concurrent buffers. Decomposition, ordering passes, composition and output
consume the existing tokenizer work budget; adversarial combining-mark runs
can fail with a checked limit error. The Split scan has linear bounded passes
over the current fragment and uses the same checked fragment allocation and
work accounting as the existing tokenizer backend.

These are limits on helper working storage and tokenizer work. They do not
claim a global process memory cap: the backend also retains vocabulary,
patterns and normalized added-token strings under the existing independent
input and descriptor limits.

Windows and GNU Linux passed the complete converter debug regression suite,
the focused reference suite in debug and release, the explicit official
tokenizer replay, formatting, and Clippy with warnings denied. This milestone
qualifies tokenizer execution; complete Qwen tensor conversion and broader
model/release gates are tracked separately.
