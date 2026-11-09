# Integrated checkpoint evidence: 9 November 2026

This directory records current execution of the merged Rust workspace. Historical archive evidence is retained separately under docs/imports/. This checkpoint does not satisfy the complete-release gate.

## Platform checks

| Check | Windows x64 | Linux x64 (Ubuntu under WSL2) |
| --- | --- | --- |
| Debug workspace tests | 285 passed, 9 explicitly ignored | 288 passed, 9 explicitly ignored |
| Release workspace tests | 285 passed, 9 explicitly ignored | 288 passed, 9 explicitly ignored |
| Strict all-targets Clippy | Passed | Passed |
| Release workspace build | Passed | Passed |
| Source and fixture audit | Passed | Passed |

Three Unix-only subprocess tests account for the count difference. Formatting passed on the shared source. Pinned compiler: Rust 1.99.1. Detailed commands, exit codes, elapsed time and output hashes are recorded in windows/results.json, linux/debug-results.json and linux/release-results.json.

The two pretrained-model tests were also invoked explicitly. Both platforms matched the F32 reference within the predeclared tolerance, and all 640,000 Q4 logits were bit-identical. The portable Q4 test retains the numerical tolerance rather than assuming every OS/ISA must produce identical bits.

## Domain evidence

- server/: real-model HTTP/SSE/replay/cancellation evidence, native F32/Q4 comparisons and reproduction commands.
- inference/: preserved failed all-F32 Q4 comparison and the corrected packed-arithmetic comparison.
- ffi/: actual frozen-header ABI/default-field and pretrained-model harness comparisons; exact export inventory.
- gpu/: real CUDA and OpenCL execution, device identities, numerical bounds and hashes.
- integration/: complete baseline/source/fixture validation and strict tool tests.
- checkpoint-summary.json: bounded machine-readable index of the exercised scope.

Ordinary ignored tests require external models, original native reference builds, a GLSL compiler or vendor libraries. Selected external tests were executed separately; the ordinary test summary alone does not establish their success. Frozen reference implementations and pretrained model weights remain outside the product.

The expected full-release gate result is nonzero because the requested complete port still has mandatory open requirements. See ../../release/gates.tsv and ../../docs/PORT_STATUS.md.
