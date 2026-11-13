# Final source, dependency and fixture audits

Both native Windows MSVC and Linux GNU/WSL profiles passed formatting, the
source-language and actual-cache dependency audit, all 1,094 fixture declarations
across 24 manifests, and both 4,082-file frozen source-tree verifications.
The default Cargo graph has 23 local packages and five exactly reviewed registry
packages with 132 materialized source files. Unsafe-location inventory is a
review aid, not blanket memory-safety proof.

The release-check command returned its expected exit code 1 because all twenty
mandatory full-port gates remain open. This expected refusal is preserved in
full and is not reported as release readiness.

## Controlled documentation delta

After the runtime evidence package's final current-file guard, only
release/dependency-inventory.tsv changed within the selected execution set.
Its documentation now records the implemented CUDA path, Falcon sidecar/synthetic
scope and platform CRT FILE ownership boundary. The exact before/after hashes,
old bytes and reason are retained here. All other 1,622 selected paths, including
all Rust, Cargo, fixtures and fixture manifests, remain identical to runtime
validation. Each audit verifies this delta and then its own unchanged input guard.

The final selected manifest has SHA-256
26eef0c4104501fde44ca730d22cbeaee9f3590057ef20c474d6d228af4f4bf4,
covering 1,623 paths / 1,048,385,877 bytes. It differs from the runtime manifest
only in that documented 765-byte inventory increase.

## Packaging boundary

The complete reports, command logs and selected input manifests are copied
byte-for-byte with a copy ledger. The manifest hashes every file here except
itself. Source-audit file/byte totals describe the tree before these receipts,
final Markdown/gate summaries and regenerated non-Rust allowlist were added.
Final sealing is run separately and its logs stay external to avoid an endless
self-reference cycle. No Rust implementation, fixture, expected output or
numerical tolerance changed during this audit phase.
