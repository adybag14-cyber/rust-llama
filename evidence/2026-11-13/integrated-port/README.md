# Final integrated port validation receipts — 2026-11-13

## Result and scope

The preserved Windows and GNU continuation chains satisfy all 13 requested integration stage groups and 22 successful unique command stages per host. This package combines selected receipts from those runs. Earlier failed reports retain their failed status. This is not a claim that every frozen-source feature and release gate is complete.

| Host | Valid chain | Debug ordinary tests | Release ordinary tests | Explicit pretrained tests | Native encoder test |
| --- | --- | --- | --- | --- | --- |
| Windows MSVC | v1 → v3 | 753 passed, 40 ignored | 753 passed, 40 ignored | 13 passed | 1 passed |
| Linux GNU | v1 → v2 → v3 | 757 passed, 40 ignored | 757 passed, 40 ignored | 13 passed | 1 passed |

Each ordinary profile contains 142 suites. Ignored tests are not counted as passes. These runs also passed format checks, Clippy with warnings denied, release workspace builds, debug FFI builds, and retained debug/release C client groups.

The stage groups cover format, ordinary debug/release tests, Clippy, release build, debug FFI build, pretrained inference/server tests, native encoder testing, historical chat/raw-name C clients, batch/sampler C clients, adapters, encoders/vector guards and public C quantization. The stage named new-c-clients is a retained batch/sampler replay. The seven newly added API clients have separate c-context-file-api component evidence and ordinary tests in the full suites.

The top-level report machine-checks each required group against successful receipts, parsed Rust test totals, command-log hashes, continuation identities, final success/input guards and debug/release library identity continuity.

## Preserved failure boundaries

Both v1 runs passed their first 14 stages, then the external historical batch/sampler helper rejected its old export inventory before client replay. The helper expected 232 frozen exports, 4 extensions and 23 absent; the copied helper explicitly adds the seven newly implemented APIs and requires 239, 4 and 16. Its review records unchanged pinned identities and AST-identical non-main functions.

GNU v2 passed six continuation stages, then the unchanged quantization helper refused a pre-existing historical output directory before invoking its client. GNU v3 uses a unique date-qualified label and passes both quantization profiles. Windows v3 passes all eight continuation stages.

All three failed reports and exact stderr remain under runs/. The helper review and byte-identical diff are under helper-review/; the diff uses a .diff.txt filename. Original and updated Python implementations remain external.

## Execution identity and current-file guard

All five execution manifests have SHA-256 e1f5c9469d0f5fd53691f832e727e3a729a6a1dc1bfe43a4d35cf930435e6f91: 1,623 paths totaling 1,048,385,112 bytes. Both final continuation reports record unchanged_execution_inputs as true. Every current file was checked by byte length and SHA-256 before packaging and again after copying/hashing the external outputs. The current-input-guard.json file records those checks; current-execution-inputs.json is the exact matched manifest.

The scope covers all Git-listed tracked/unignored execution inputs except evidence, Markdown documentation, release/gates.tsv and the derived non-Rust allowlist. It includes Rust source, Cargo/toolchain/configuration, dependency controls, fixtures and fixture manifests. The parent task's later documentation/inventory delta and final audits are separate.

## What is retained here

This package contains complete top-level reports, command logs and execution manifests for all five runs; nested client reports and summaries; export inventories; and helper review metadata. Current quantization reports are copied from their external replay directories. The copy ledger identifies each copied file's original absolute path, exact byte length and SHA-256.

Large client outputs, prepared models, executables and implementation scripts remain external. The external-output ledger freshly hashes every non-selected file under the named run and current quantization replay directories. Only selected receipts are copied here; the raw-result locations remain explicit.

The manifest hashes every packaged file except itself. No source, configuration, fixture, existing evidence or TSV file was modified, and no test or build was rerun for packaging.
