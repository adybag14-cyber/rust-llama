# Final source, dependency and fixture audits

Both Windows MSVC and Linux GNU/WSL passed formatting, the source-language and
actual-cache dependency audit, 1,143 fixture files covered by
1,154 declarations across 30 manifests,
and both complete 4,082-file frozen source-tree checks. The audit inventories
403 Rust files. The Cargo graph still has 23 local packages and five
exactly reviewed registry packages with 132 materialized source files.
The unsafe-location inventory is a review aid, not a memory-safety proof.

The release check returned its expected exit code 1 with all twenty mandatory
full-port gates open. The recorded refusal is not release readiness.

## Controlled CI and documentation delta

After both runtime validation chains and their package guards finished, the reviewed
CI commit added Windows/Linux pinned tiny StarCoder conversion and inference
coverage. The only other selected-input change removed one extra trailing
newline from release/dependency-inventory.tsv. Exact before/after identities
are retained in ci-integration-delta.json; the unchanged preimage bytes remain
external at their recorded paths and hashes.
Rust, Cargo, fixtures, fixture manifests and numerical expectations are unchanged.

The final selected manifest has SHA-256
`cd79f8a092cd60c2e6c55121b5e2d6daac1b71020c78aedb4f8582f302735136` and covers 1,687 paths totaling
1,192,670,384 bytes. Both audit runs and packaging checked those
inputs before and after execution/copying. Markdown documentation and evidence
are outside that execution set and receive separate package checks.

## Packaging boundary

Complete reports, logs and selected input manifests are copied byte-for-byte
with a copy ledger. The manifest covers every packaged file except itself.
Audit file/byte totals describe the tree before these receipts, final Markdown
summaries and regenerated non-Rust allowlist were added. Final sealing verifies
the complete package separately and keeps its own logs external.
