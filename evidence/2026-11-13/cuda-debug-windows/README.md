# Windows debug CUDA execution — 2026-11-13

## Result

All seven requested actual-GPU tests passed on the NVIDIA GeForce RTX 4090. The already-built Windows debug binaries were invoked directly. No Cargo command, compiler invocation or product source edit was performed for this verification.

The three binaries came from the passed full integration `debug-tests.stderr.log`. Each binary's `--list` and `--ignored --list` output was checked against the exact expected test names before execution. Each hardware invocation used:

`--ignored --test-threads=1 --nocapture`

| Suite | Hardware tests passed | Ordinary tests filtered out |
| --- | ---: | ---: |
| CUDA pretrained | 3 | 0 |
| CUDA runtime | 2 | 0 |
| Decoder kernels | 2 | 3 |
| Total | 7 | 3 |

These are opt-in hardware tests; no ignored test was counted as passed. The three ordinary decoder tests had already run in the full debug workspace test stage.

## Observed numerical and behavioral results

| Corpus | Native logits compared | Exact greedy IDs | Maximum absolute error |
| --- | ---: | ---: | ---: |
| stories260K F32 | 62,976 | 96 | 7.152557373046875e-6 |
| Native stories260K F16 derivative | 59,904 | 96 | 6.556510925292969e-6 |
| stories15M mixed Q4_0/Q8_0 | 640,000 | 16 | 0 |
| Total | 762,880 | 208 | |

The existing pretrained threshold remains `2e-4 + 2e-5 * abs(expected)`. The F16 corpus is the separately captured original C++/C-API MSVC oracle, replayed here on Windows. The pretrained suites additionally check shared generation/stop behavior and F32 cancellation, truncation, invalid-token rollback and retry against a clean prefix.

The runtime suite passed 44 independent original-GGML biased-GQA values with maximum absolute error 1.4901161e-7. An exact requested CUDA-buffer budget of 3,812 bytes admitted the model; 3,811 bytes was rejected. The numerical-failure regression retained the committed prefix and a clean retry after both adversarial finite-weight failures.

The decoder suite passed 135 analytical cases with two persistent device launches each, exact Q8 activation-bit checks, seven malformed-uniform guards, foreign stream/input/output rejection, short-input rejection and retained-context lifetime behavior.

## Binary identity

All paths are beneath `C:\Users\adyba\rust-llama\target\debug\deps\`.

| Binary | Bytes | SHA-256 |
| --- | ---: | --- |
| pretrained-03448b6b4409fbda.exe | 7,854,592 | 68e61f6676fea6030c0d84ab5f57cb5d7ab531d2e9597d42abf6d96163b6fa4e |
| runtime-6b1a1abe35f0c2b1.exe | 2,395,136 | 15348d2aaf39fa984ae8262ab3e8bd35fc3ae8c8ef41d7aea884392f880ab5fa |
| decoder_kernels-77588ce1ccf86f30.exe | 1,750,016 | cd8fdbde6a90ef2838bf36fcea2aac60b394027299407144b5bb7b89d6e3376a |

Hashes were verified unchanged after each suite. The build-report snapshot preserves the precise MSVC Rust compiler identity and the passed debug workspace test command.

## Sources, inputs and execution receipts

All 1,623 execution inputs from the final integration manifest, totaling 1,048,385,112 bytes, matched by path, byte length and SHA-256 before and after this replay. The scope includes Rust source, Cargo/toolchain/configuration, dependency controls, fixtures and fixture manifests; it excludes evidence, Markdown documentation, `release/gates.tsv` and the derived non-Rust allowlist. The two current manifests are semantically identical to the integration manifest; their JSON key order and line endings differ, so raw serialized SHA-256 differs while every file entry matches.

All three pinned model environment variables were set for every command: `LLAMA_TEST_MODEL`, `LLAMA_TEST_F16_MODEL` and `LLAMA_TEST_Q4_MODEL`. Their absolute paths, byte lengths and hashes are in `report.json`. The test implementations also independently enforce each model hash.

`report.json` contains complete argument vectors, explicit environment values, expected/observed test names, binary identities, timing, exit status and stdout/stderr hashes. Each test command had a 180-second timeout and each stream a 2 MiB cap; list commands had a 30-second timeout. All output completed without truncation or timeout. The external driver serializes the suites and uses one Rust test thread. The enclosing Hermes job has an 800-second timeout.

External driver: `C:\Users\adyba\rust-llama-reference\continue-20261113\gpu\validate-debug-windows.py`.

Successful Hermes job: `job-1794607183853-ba5b7592-16b`. The exact suite/list logs, copied integration receipts, current-input manifests and job receipt are retained here. `manifest.tsv` hashes the evidence files other than itself.
