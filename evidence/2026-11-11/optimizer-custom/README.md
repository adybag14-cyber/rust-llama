# Optimizer and custom callback continuation

The maintained Rust tensor engine adds six executable frozen operation IDs:
OPT_STEP_SGD, OPT_STEP_ADAMW, MAP_CUSTOM1/2/3 and CUSTOM. This is a bounded
CPU execution contract, with immutable state snapshots and owned Rust callbacks.
It does not close the complete tensor, optimizer, scheduler or full-port gates.

## Independent comparisons

Each original MSVC/GNU capture contains 300 optimizer cases and 440,704 exact
parameter/moment output bytes, plus 260 custom cases and 30,064 exact output
bytes. Both profiles have separate provenance and identical corpus bytes.
Reference constructors, CPU planners and native callback threads ran before the
corresponding Rust implementation. Oracle generators and original libraries
remain outside the product; their identities are in summary.json and fixture
provenance. Complete case ledgers are retained here.

The Rust tests replay both corpora in debug and release on both native hosts:
nine focused tests pass in each configuration. Strict tensor Clippy and formatting
also pass. The selected 282-file execution input set remained identical during
each focused run; reports and raw logs are retained by profile. The later
whole-workspace integration is a separate receipt.

An independent read-only source review found no blocking numerical, ownership,
thread joining or graph publication issue. Its allocation wording qualification
was applied to the domain documents after the focused runs.

## Preserved diagnoses

The first optimizer oracle tried to mark a GGML view as a parameter. The original
constructor asserted; capture v2 uses an owned leaf with public shape/stride
fields within its fully initialized backing allocation. The initial custom
oracle expected callbacks on empty outputs; both original profiles demonstrated
that the dispatcher skips them. Capture v2 and the Rust implementation preserve
zero invocations. Initial failure logs remain available here.

The first Rust optimizer reference comparison already passed. A separate graph
failure regression incorrectly used compatible matrix shapes and therefore
expected a spurious error; the test was corrected to genuinely incompatible
shapes. The first Clippy run reported three style issues, fixed without warning
allowances. Those failed logs remain distinct from subsequent passing evidence.

See [the optimizer contract](../../../../crates/tensor/OPTIMIZERS.md),
[the callback contract](../../../../crates/tensor/CUSTOM.md), and summary.json.
