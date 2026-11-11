# Extended forward rotary and softmax attention

This package adds original-source extended forward rotary modes and masked,
sink-aware ALiBi softmax to the immutable Rust tensor executor. Both extend
existing frozen operation IDs; the executable operation count stays unchanged.
The full port and its twenty mandatory release gates remain open.

## Independent results

Each corresponding MSVC/GNU profile compares 3,303 admitted original graphs and
11,181,540 exact output bytes. The fixtures retain another 58 successful native
graphs at explicitly checked numerical boundaries (20,336 raw output bytes).
Four isolated original assertion probes per host establish constructor rejection
and the original CPU's nonzero backward-bias rejection.

The primary 892 rotary and 2,389 softmax graphs were captured before new Rust
implementation. Another 80 intermediate-divisor mask graphs were captured
independently after initial implementation and before adding their replay.
Their corrected v2 provenance describes that timing; it reproduces all v1 binary
bytes and both external captures are preserved. No expected output is generated
from the Rust implementation.

## Validation

The full tensor package passes ordinary debug and release tests, strict Clippy
and formatting on both platforms. The final postflight passes the same policy
checks and all new attention tests in debug and release after the supplemental
test/fixtures are added. Source implementation bytes are unchanged between
the full package run and final postflight.

Four negative controls per host reverse forward rotation, disable ALiBi, remove
the sink denominator, and use grouped rather than periodic mask indexing.
Every control fails an actually executed oracle test, and exact source bytes
are restored before final postflight.

The exact commands, counts, input hashes, unchanged selected-input checks,
compiler identities and retained logs are in summary.json and validation/.
The selected guard is not a whole-repository immutable snapshot. The first two
failed checks retain a test-only Clippy diagnostic and the temporary type error
in its correction; neither changes numerical expected data or tolerances.

The complete numerical/layout/resource contract is in
[ATTENTION_EXT.md](../../../crates/tensor/ATTENTION_EXT.md).
All maintained implementation is Rust. Original libraries and oracle programs
remain in the external attention-oracle directory; their hashes and original
source/library hashes are retained in fixture provenance.

## Integration

Register docs/attention-extended-fixture-manifest.tsv in the global fixture
audit. It declares 30 exact data/provenance files. Shared edits are confined to
Op/code/arity declarations, dispatch, exports and the shared rotary sign core.
No graph evaluator, root Cargo manifest, dependency lockfile, root status or
release-gate configuration is changed.

Exact original fixture and evidence bytes retain Windows CRLF and raw test-log
trailing blank lines. Maintained source and document whitespace checks pass;
`git-diff-check.json` records the expected raw-log whitespace diagnostics.
