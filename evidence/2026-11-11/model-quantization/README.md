# Model quantization component evidence

This package records a successful staged component proof for core commit 40c6573994019bbab9aadff0992988ea3b0c0dfd. It does not relabel either v5 validation attempt as successful and does not claim combined-branch or full-port completion.

## Executed runtime proof

| Platform | Debug | Release | Strict Clippy | Original-header C replay |
|---|---:|---:|---|---|
| Windows MSVC | 145 passed, 0 ignored, 15 suites | 145 passed, 0 ignored, 15 suites | Passed | Debug and release passed |
| Linux GNU | 144 passed, 0 ignored, 15 suites | 144 passed, 0 ignored, 15 suites | Passed | Debug and release passed |

Each public C replay invokes the real Rust exports through the unchanged frozen public header, exercises 88 file cases, agrees on 77 successes and 11 rejections, and verifies 76 complete output files by exact length and SHA-256. Two successful dry runs produce no file, and one split case produces two files. Both public exports and the default parameter layout are checked. Dependency receipts exclude linkage to an original llama/ggml runtime.

Independent original planner captures contain 462 type requests plus one defaults observation per host. They cover 16,315 MSVC and 16,417 GNU individual type assignments. The original file corpus contains 88 cases per host. All sealed original artifacts are rehashed before evidence publication.

Metadata admission has 346 original observations per host: 142 accepted, 188 rejected and 16 process aborts. The maintained eight-family file scope replays 140 exact complete copies and 202 checked rejections, while retaining four outside-family observations as explicit exclusions. Ten further original complete quantizations prove the NextN-dependent Llama classification, separately from only-copy validation. The general type planner's architecture registry does not expand model-file or inference support.

## Preserved attempts and packaging chronology

1. Early MSVC v1, v2 and v3 stopped at Clippy. Their original stdout/stderr logs are retained. No capture-time report or input guard was produced, so this package supplies a clearly labeled packaging-time log inventory without reconstructing an exit code or input snapshot.
2. MSVC v4 failed strict Clippy's manual checked-division lint. Its report, recorded log hashes and input manifest remain unchanged.
3. MSVC and GNU v5 passed formatting, strict Clippy, both affected-crate test profiles and both native library builds. Both reports remain failed overall: the audit rejected the then-unclassified crates/model-quant/fixture-manifest.tsv. Those attempts ended before their own final input guard.
4. Packaging correction v1 independently verifies the same 1,584-file runtime input manifests, moves the unchanged fixture declaration into docs, admits 11 already sealed GGUF inputs using a narrow ignore exception, and corrects the separate BLOOM CI input directory basename. All retained source and fixture bytes outside the declared configuration changes remain unchanged. Audit and all 1,002 fixtures across 16 manifests pass. The attempt then fails the ordinary whitespace check on CRLF endings; this failure is retained.
5. Postflight v2 verifies that v1's inputs remained unchanged, converts accidental CRLF endings to LF in only crates/gguf/src/lib.rs and writer.rs, proves all other selected inputs unchanged, and passes formatter plus CRLF-aware whitespace checks. Reconstruction of the two pre-change byte streams is checked again by this packager. The final archived selected set has 1,595 files and manifest SHA-256 d2f21a9d6242662806545c9aff0c32f57e78a4fc6e05ce2c02329f11ada565ee.
6. A recorded model-quant README-only continuation corrects the fixture path and explains metadata/C coverage. All 46 exact changed blobs of the resulting core commit are read from Git and verified against the staged manifest and archived inputs after that documentation change.
7. This evidence package is composed only after all four C replays and the packaging postflight succeeded. It performs no new runtime execution.

The parent is concurrently integrating independently validated domains. This package verifies immutable core commit blobs, not the current mutable root. It does not assert that the complete archived 1,595-file selected set equals the core commit: the original core receipt explicitly leaves CI integration separate. Final combined-branch validation remains a distinct gate.

## Artifact layout and integrity

- summary.json is the stable machine-readable staged summary.
- core-commit contains the original commit receipt, exact 46-file staged manifest, the verification receipt and the 26-file component fixture declaration.
- validation preserves every available v1-v5 report, input manifest and raw command log.
- development and launcher-failures preserve earlier diagnostic logs and non-runtime queue/launcher failures.
- packaging preserves both correction attempts, their manifests and raw logs, plus the README-only continuation.
- original contains unchanged native seals/provenance and request-result summaries. Complete foreign generator/build/model payloads remain external.
- metadata contains both independently sealed native admission/classification corpora and an explicitly attributed implementation/source-review resolution record.
- public-c contains all four successful reports, exact raw command logs and compact request-result summaries.
- integrity records every checked recorded link, all external artifact identities and the source-to-package copy map. Historical logs without a capture-time digest are individually identified.
- artifact-inventory.json covers every packaged file other than itself. An external completion receipt pins that inventory and the complete publication.

Native raw logs are preserved losslessly in JSONL bundles: each record carries the original path, normalized log name, exact byte length, SHA-256 and base64 payload. Long arrays in request-result summaries are described by canonical JSON digests; full original reports remain external and are still verified against their capture-time seals. Raw command logs for validation, packaging and C replays are copied byte-for-byte. No reference Python/C/C++ sources, native build products or large models are installed in the product.

Original aborts are translated to checked errors. Output conversion stages all bytes before per-file publication, but several shard renames are not a filesystem-wide transaction. The evidence supports the specified component behavior and retained corpus; it is not a claim that every model architecture, backend or release gate is complete.
