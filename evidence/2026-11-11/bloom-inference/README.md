# BLOOM inference validation

The final complete unchanged-input runs passed on Windows/MSVC and GNU/Linux. This receipt covers the strict causal Rust CPU BLOOM slice, including actual trained BLOOM560m F16 and Q8_0 models. It does not close the repository-wide full-port gates.

## Final results

| Platform | Debug passed / ignored | Release passed / ignored | Separate trained comparisons |
| --- | ---: | ---: | ---: |
| linux-gnu | 126 / 12 | 126 / 12 | 2 |
| windows-msvc | 124 / 12 | 124 / 12 | 2 |

Both hosts passed cargo fmt and strict all-target Clippy. The ordinary tiny BLOOM comparison checks 346,644 exact F32 words and 216 greedy IDs; each trained model compares full native vocabulary rows and final embeddings, exact generation, explicit embeddings, continuation, saved-state/checkpoint restoration, rewind and forks. Raw metrics are in summary.json and retained command logs.

## Input and provenance guard

The final runs used the same 558 selected inputs. The canonical path/size/SHA-256 manifest digest is f80fb153b46fc7cfeba4407a717303c34dd971e7aa6b594d2740f30cc539d656. Every selected file was rechecked after each run and again before packaging. The manifest is scoped to the listed runtime dependencies, inference fixtures and Cargo inputs; later documentation/evidence is outside that validation guard.

Original native captures and all expected bytes predate maintained BLOOM implementation. Official models have pinned complete input and GGUF hashes; full RAIL license and original model card are maintained with the numerical fixtures. Large checkpoints and raw native result JSON remain external.

## Retained attempts

- [full-v1 / linux-gnu](validation/full-v1/linux-gnu/report.json): failed.
- [full-v1 / windows-msvc](validation/full-v1/windows-msvc/report.json): failed.
- [full-v2 / linux-gnu](validation/full-v2/linux-gnu/report.json): failed.
- [full-v2 / windows-msvc](validation/full-v2/windows-msvc/report.json): failed.
- [full-v3 / linux-gnu](validation/full-v3/linux-gnu/report.json): succeeded.
- [full-v3 / windows-msvc](validation/full-v3/windows-msvc/report.json): succeeded.
- [quick-v1 / windows-msvc](validation/quick-v1/windows-msvc/report.json): failed.
- [quick-v2 / windows-msvc](validation/quick-v2/windows-msvc/report.json): succeeded.

## Qualifications

- Original captures and immutable fixture packing completed before Rust implementation; separate MSVC/GNU references.
- Initial original GNU linker attempts failed due to -llama rather than -lllama; the initial symlink diagnosis was explicitly corrected. Original baseline/build files were unchanged.
- quick-v1 MSVC failed compilation because the test assumed a Default implementation for NativeContextOptions; explicit limits fixed that test.
- full-v1 on both hosts failed only the stale seven-slice smoke assertion after BLOOM became the eighth slice; original BLOOM comparisons were exact.
- full-v2 on both hosts failed strict Clippy for two range loops; iterator traversal preserved numeric operation order.
- Only the final complete unchanged-input full runs establish the domain pass; every earlier attempt retains its actual status and scope.
- Official checkpoint payloads, native builds, raw large native result JSON and non-Rust generators remain external.

## Package postflight and independent review

The Rust source-package audit, complete fixture-manifest integrity verification and git diff whitespace check passed after the domain evidence was added. The final postflight receipts themselves are retained under postflight/v1. A separate source review found no blocking issue and pins the exact reviewed runtime, loader, helper and frozen reference bytes under review/runtime-review-v1.json. The final publication guard rechecks all 558 validated inputs and every reviewed source hash before commit.
