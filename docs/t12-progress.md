# T12 — validation and release checkpoint

> **Historical imported snapshot.** This report describes the standalone
> T09–T12 archive, including its Linux-only 123-test and 44-input validation.
> Those totals and missing-track statements do not describe the merged product.
> Current commands are in [xtask/README.md](../xtask/README.md); current Windows
> and Linux results are in
> [expanded runtime verification](../evidence/2026-11-09/expanded-runtime/README.md)
> and [PORT_STATUS.md](PORT_STATUS.md). The [import manifest](imports/manifest.json)
> retains the archive identity. The full validation and release requirements
> remain unchanged in the [active gates](../release/gates.tsv).

## Implemented

The `llama-conformance` crate and `xtask` binary are authored entirely in Rust 2024 with no Cargo dependencies outside this workspace. They provide streaming SHA-256, ZIP CRC-32, exact and budgeted numeric comparisons, source ownership/disposition proposals, registry extraction, drift detection, bounded process capture, source audits, package integrity validation and reproducible uncompressed ZIP output.

The independent SHA-256 tests use published empty/abc/multiblock/million-a vectors; CRC-32 uses the standard `123456789` check value. Mutation tests deliberately alter numeric bytes, truncate outputs, change same-length files and add/remove baseline entries. Process tests exercise pipe saturation and terminate a single subprocess on timeout. A shell is never involved in the `capture` task.

`xtask baseline` enforces the supplied plan's exact 4,082-file/183,804,212-byte ownership totals and primary model/operator registry counts before writing a ledger. The per-file SHA-256 identifies the reference content. The ledger's disposition and destination fields are proposed work instructions: they are not proof that those original files' capabilities have been replaced. Every baseline path retains `planned` until semantic closure is reviewed; domain capability ledgers report actual implemented and validated subsets.

The registry extractor records architectures, dtypes, operators, unary/GLU variants, model sources, multimodal builders, native templates, CLI flag literals, HTTP route registrations, CMake options/test registrations and public-function candidates. Candidate/discovery categories are intentionally named: this is lexical extraction, not a complete C++/CMake semantic evaluator. Conditional registrations and macro-generated APIs require later closure.

## Independent original reference

The supplied original `tests/test-sampling.cpp` was compiled separately with GCC/G++ 14.2.0, original GGML base, sampler, chat, vocabulary and Unicode sources. It passed with exit 0. `evidence/reference` records compiler commands, original-test stdout/stderr, executable hash and source archive identity. This is evidence that the selected original oracle runs, not evidence by itself that Rust sampling passes. T10 supplies the differential checks.

The original reference directory and its C/C++ object files/binaries are excluded from the Rust workspace and release package. Building or running the Rust product does not invoke them. Small audit adapters used to obtain reference fixtures reside with the temporary oracle, not the product. Golden fixtures carry source identity and domain documentation.

## Reproduce

With Rust 1.99.1 on PATH:

```text
cargo generate-lockfile --offline
cargo test --workspace --offline --locked
cargo build --workspace --release --offline --locked
cargo xtask baseline /absolute/path/to/llama.cpp-master
cargo xtask verify-source /absolute/path/to/llama.cpp-master
cargo xtask audit
cargo xtask fixtures
cargo xtask release-check
cargo xtask package /absolute/path/to/new-checkpoint.zip
```

`release-check` deliberately fails while any mandatory gate remains open. It is independent from `package`, which permits explicitly labelled progress snapshots. Packaging always applies the source/dependency audit, refuses existing output paths, uses fixed ZIP timestamps/permissions/order, and adds `PACKAGE_MANIFEST.tsv`. Repackaging an extracted archive regenerates that manifest. `cargo xtask verify-package` validates all extracted source files against it and rejects added/missing/changed files, excluding build caches and Git metadata.

The non-Rust allowlist identifies data, fixtures, notices, documentation and declarative manifests. Its own checksum is covered by the package manifest to avoid a self-hash cycle. Source-language scanning and Cargo metadata checks are useful evidence, not a proof against every possible embedded interpreter or opaque implementation. All delivered code still needs semantic review for the eventual all-feature release.

## Archive validation

The candidate archive was checked independently with Python's ZIP reader for CRC integrity, unique safe paths, and exact manifest membership, sizes and SHA-256 values. Its extracted source passed all 123 workspace tests, an offline locked release build, and the Rust package-integrity verifier from a fresh build cache. The final snapshot adds validation reports and documentation only; its 44 implementation/configuration file hashes must match `release/implementation-identity.json`. The candidate archive hash and executed clean-build results are recorded in `evidence/validation/clean-extracted-results.json`.

## Limits and next work

The checkpoint is validated only on Linux x86_64. Windows, Apple/Android/WASM and GPU/DSP builds have not been established. Miri, sanitizers and real model/hardware quality/performance gates remain open. The process runner bounds one directly launched process, its buffers and timeout; it is not a cross-platform process-tree supervisor, and a descendant that inherits pipes is outside its current contract. Compiler drivers such as Cargo and GCC can create such descendants; use the separate inherited-output build tasks, not `capture`, for those drivers. The capture deadline applies to its direct child and does not bound descendant-held pipe draining.

Next bounded task: integrate the actual T01–T08 commits and their reviewed shared contracts, add independent end-to-end tokenizer/model fixtures, and extend the release gates with target-specific commands, device identity and signed or hashed results.
