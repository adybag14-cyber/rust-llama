# Final integrated port validation receipts — 2026-11-14

The Windows v1 → v2 → v3 and GNU v1 → v2 chains satisfy all 18 requested
logical groups and 27 successful unique command stages per host. All three
failed reports retain their failed status and exact logs. Successful stage
coverage explicitly identifies its source run.

Test totals are parsed from the retained logs and checked against report fields.
Ignored tests are not counted as passes.

| Host | Ordinary debug tests (v1) | Ordinary release tests (v1) | Explicit pretrained tests (final run) | Native encoder test (final run) |
| --- | --- | --- | --- | --- |
| windows-msvc | 768 passed, 49 ignored | 768 passed, 49 ignored | 16 | 1 |
| linux-gnu | 772 passed, 49 ignored | 772 passed, 49 ignored | 16 | 1 |

## Preserved failures and corrections

Both v1 runs passed format, ordinary debug/release tests, strict Clippy and audit,
then fixture integrity rejected the missing declaration for the unchanged
backend-sampling README. The existing fixture manifest received exactly one row
with the README's 1,992-byte length and existing SHA-256. Both correction guards
checked all selected current inputs; all fixture contents, prior declarations,
Rust/Cargo/configuration and verifier policy remained unchanged.

The v2 runs reuse the four passed v1 format/test/Clippy stages. Windows v2 then
passed audit, fixtures, release build, debug FFI build and the first frozen-source
check, before its reference-copy check rejected ten added generated CPython cache
files. Root preserved all 946,128 bytes of those ten files outside the frozen
tree and rehashed them. No baseline source or product input changed. Windows v3
verifies the cleanup receipt and preserved identities, reuses the five passed
v2 stages, and reruns strict reference-copy verification before its remaining
eight logical groups. It has 18 command stages. GNU v2 completes its 23 commands
after cleanup without a third continuation.

The metadata correction and external cache correction receipts are retained.
Each failed command remains separate from successful stage coverage. Windows
v3 and GNU v2 use distinct date-qualified quantization output labels; historical
quantization output directories are preserved.

The groups cover format, ordinary tests, Clippy, audit, fixtures, both builds,
both frozen-source directory checks, release gates, pretrained inference, native
encoder tests and all retained C client families. Release-check deliberately
exits 1 and reports 20 mandatory gates open on both final hosts. These expected
nonzero receipts do not establish full-port or release completion.

## Identity and comparison guards

Both v1 input manifests are byte-identical. Every v2/v3 manifest is also identical
and differs from v1 only by the reviewed README declaration. All 1,687 current
corrected inputs are checked before packaging and after receipt copying/hashing.

All client families identify the same merged native debug/release library.
Current Windows DLLs and GNU shared libraries are rehashed before and after
packaging. The adjacent sampling/context/FILE C API receipt must identify those
same four libraries and matching runtime sources. The named inventory is exactly
247 frozen functions plus four extensions, with eight frozen functions absent.

Historical batch/sampler comparison functions and pinned identities are unchanged;
the eight explicit added exports and exact helper diff are recorded. The v2/v3
drivers also preserve the original comparison functions and suppress Python
bytecode generation.

## Retained material

All five run reports, their top-level command logs and execution manifests,
nested reports/summaries and export inventories are copied byte-for-byte.
Correction receipts and exact before/after manifest metadata are included.
The copy ledger records their original paths and hashes. Raw outputs, models,
preserved generated Python caches and auxiliary files remain external with fresh
hashes in the external-output ledger.

The package manifest hashes every packaged file except itself. Packaging does
not edit product source, configuration, fixtures or TSVs and performs no builds
or tests. Deferred CI and dependency-inventory formatting changes are recorded
separately by the parent afterward.
