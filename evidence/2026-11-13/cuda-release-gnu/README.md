# GNU release CUDA execution — 2026-11-13

## Result

All seven requested actual-GPU tests passed on the NVIDIA GeForce RTX 4090. The already-built GNU release binaries were invoked directly. No Cargo command, compiler invocation or product source edit was performed for this verification.

The three binaries came from the passed full integration `release-tests.stderr.log`. Each binary's `--list` and `--ignored --list` output was checked against the exact expected test names before execution. Each hardware invocation used:

`--ignored --test-threads=1 --nocapture`

| Suite | Hardware tests passed | Ordinary tests filtered out |
| --- | ---: | ---: |
| CUDA pretrained | 3 | 0 |
| CUDA runtime | 2 | 0 |
| Decoder kernels | 2 | 3 |
| Total | 7 | 3 |

These are opt-in hardware tests; no ignored test was counted as passed. The three ordinary decoder tests had already run in the full release workspace test stage.

## Observed numerical and behavioral results

| Corpus | Native logits compared | Exact greedy IDs | Maximum absolute error |
| --- | ---: | ---: | ---: |
| stories260K F32 | 62,976 | 96 | 6.020069122314453e-6 |
| Native stories260K F16 derivative | 59,904 | 96 | 6.556510925292969e-6 |
| stories15M mixed Q4_0/Q8_0 | 640,000 | 16 | 0 |
| Total | 762,880 | 208 | |

The existing pretrained threshold remains `2e-4 + 2e-5 * abs(expected)`. The F16 corpus is the separately captured original C++/C-API MSVC oracle, also replayed here on GNU. The pretrained suites additionally check shared generation/stop behavior and F32 cancellation, truncation, invalid-token rollback and retry against a clean prefix.

The runtime suite passed 44 independent original-GGML biased-GQA values with maximum absolute error 1.4901161e-7. An exact requested CUDA-buffer budget of 3,812 bytes admitted the model; 3,811 bytes was rejected. The numerical-failure regression retained the committed prefix and a clean retry after both adversarial finite-weight failures.

The decoder suite passed 135 analytical cases with two persistent device launches each, exact Q8 activation-bit checks, seven malformed-uniform guards, foreign stream/input/output rejection, short-input rejection and retained-context lifetime behavior.

## Binary identity

All paths are beneath `/home/tdamre/rust-llama-target-continue/release/deps/`.

| Binary | Bytes | SHA-256 |
| --- | ---: | --- |
| pretrained-7bc52d2050dadce0 | 6,449,824 | 9a09f89ab4e57e4f517de473c1d0ca1dbceeaa9e8babc4ec9ccee0eec1fdac08 |
| runtime-a1d616b6474b4bc3 | 1,846,280 | cacad77dbdaf65adf8d1bba41903247d9362b0bbe49e139f7624a34d4824264e |
| decoder_kernels-d6e2c7e553590f7a | 1,531,744 | f9320475d56ac28983d832489e072481afcc571c1328774ff26a68d8070eae09 |

Hashes were verified unchanged after each suite. The recorded build compiler is Rust 1.99.1, target x86_64-unknown-linux-gnu. The build-report snapshot also preserves the precise compiler identity and passed release test/build commands.

## Sources, inputs and execution receipts

All 1,623 execution inputs from the final integration manifest, totaling 1,048,385,112 bytes, matched by path, byte length and SHA-256 before and after this replay. The scope includes Rust source, Cargo/toolchain/configuration, dependency controls, fixtures and fixture manifests; it excludes evidence, Markdown documentation, `release/gates.tsv` and the derived non-Rust allowlist. The two current manifests are semantically identical to the integration manifest; their JSON key order differs, so raw serialized SHA-256 differs while every file entry matches.

All three pinned model environment variables were set for every command: `LLAMA_TEST_MODEL`, `LLAMA_TEST_F16_MODEL` and `LLAMA_TEST_Q4_MODEL`. Their absolute paths, byte lengths and hashes are in `report.json`. The test implementations also independently enforce each model hash.

`report.json` contains complete argument vectors, explicit environment values, expected/observed test names, binary identities, timing, exit status and stdout/stderr hashes. Each test command had a 180-second timeout and each stream a 2 MiB cap; list commands had a 30-second timeout. All output completed without truncation or timeout. The external driver serializes the suites; the Windows-to-WSL bridge has a separate 800-second GNU timeout and records its complete argument vector.

External driver: `C:\Users\adyba\rust-llama-reference\continue-20261113\gpu\validate-release-gnu.py`.

Successful Hermes job: `job-1794607075367-7a38bf2b-660`. The exact suite/list logs, copied integration receipts, bridge arguments, current-input manifests and job receipt are retained here. `manifest.tsv` hashes the evidence files other than itself.

## Legacy conformance example

No current non-test `cuda_conformance` executable was present. The only executable candidates were the older `cuda_conformance-83ebfd8e1933de22` and final `cuda_conformance-f3864ae2eb725a99`; both Cargo fingerprints identify test examples. The final log explicitly runs the latter as a unit-test harness, and the normal un-hashed executable is absent. Neither was represented as execution of the legacy example's main program.

The 138 legacy-kernel conformance replay is left for the parent task to build and execute separately after Cargo is available. This folder does not claim that replay.
