# Quantize CLI verification

The native Rust quantize CLI and bounded importance loader pass formatting,
strict Clippy and complete package debug/release suites on Windows MSVC and
Linux GNU. Both hosts retain the same 226 guarded inputs unchanged; the
aggregate SHA-256 is d246a9035d53f5cf6b5d3ff6dcccf61c24de64974a4083f382a93a8c4a1c544a.

Each build mode replays 188 actual original executable requests and 517 direct
original parser/importance observations per host. Full-byte comparisons cover
117 MSVC output files (2,600,128 bytes) and 118 GNU files (2,658,656 bytes), with
no capability skips. Existing core quantizer regressions run as part of the
same suites. Six additional direct-core identity/dry-run cases per host verify
input preservation and one exact complete-file control.

The 392 accepted MSVC helper comparisons comprise 389 unchanged native private
results and three explicitly reviewed UCRT private F64 corrections. GNU has
398 unchanged native accepted helper results. Checked-domain exclusions are
15 MSVC and 8 GNU observations. They remain identified in the independent
fixtures and are required to return errors, not silently skipped.

The installed Windows SDK ABI and hard-link/symlink/different-file identities
were captured before the safe Rust OS helper. Original GNU C alias writes
that damage inputs or abort remain documented safety corrections. Original
MSVC write-time alias rejection and both hosts' harmless alias/directory
dry-run acceptance are preserved.

The source and full reference-tree seals, native aggregate reports, installed
SDK observations, focused/failed/final verification reports, input inventories
and independent read-only review are retained here. Final review has no open
findings. Original C/C++ programs, binaries and all full per-case raw console
logs remain external at the paths in the receipts; their hashes and producer
identities are recorded.

Read summary.json for exact commands, timings, counts and limitations; the
maintained contract is crates/model-quant/CLI.md. The separate integrity
manifest docs/model-quant-cli-fixture-manifest.tsv covers all 17 CLI/SDK fixture
files without changing the existing core quantizer manifest. No registry
dependency was added.

These results verify this bounded component. Full-port release gates remain
separate and open until independently satisfied.
