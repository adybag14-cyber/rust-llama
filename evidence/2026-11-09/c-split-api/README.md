# Explicit C split APIs: independent verification

This checkpoint adds llama_model_load_from_splits, llama_split_path and
llama_split_prefix in Rust. Both measured x64 hosts passed focused debug and
release suites, strict Clippy, source/fixture audits, external frozen-header
C callers and four deliberately wrong implementation controls. Symbol tables
contain 187 original C functions and the same 4 Rust diagnostic extensions.

The original cases were captured and sealed before Rust implementation.
The helper corpus checks 7,807 calls and 506,563 complete output-buffer bytes per
host/build profile. Model lists check 104 Windows / 106 GNU cases and their own
exact 172,032 / 180,224 F32 continuation words. Original assertions become checked
errors. Separate source profiles retain the original host differences.

The original compiled 968-file candidate remains unchanged. A separate final
candidate adds only five documentation changes to make 969 execution inputs.
This evidence archive is separate from those compiled inputs. Final package
audits are recorded outside this immutable archive and tied to its exact file
inventory; later hosted CI also audits the published commit.

Reports, original requests/outputs and nonempty logs are stored here. Empty
logs have their complete identities in empty-logs.json; zero bytes need no
duplicate physical file. The original reports/pools/matrix also exist unchanged
in the 14 committed fixtures, indexed by fixture-identities.json. No original
native code, external Python script or compiled oracle is part of product code.
external-driver-identities.json records their reproducible external identities.

See SPLITS.md and verification-summary.json for scope, pointer and byte-string
contracts, limits and retained corrections. Twenty mandatory full-port gates
remain open across twelve tracks. This is an intermediate capability checkpoint.
