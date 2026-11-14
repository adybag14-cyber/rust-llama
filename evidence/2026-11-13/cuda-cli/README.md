# CUDA CLI integration evidence — 2026-11-13

## Result

Both final CLI profiles succeeded on the NVIDIA GeForce RTX 4090:

| Profile | Binary | Commands | Native generated-text commands | Stop commands | Logits commands | Explicit rejections |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Windows release | MSVC release | 64 | 26 | 12 | 6 | 19 |
| GNU debug | WSL Ubuntu GNU debug | 64 | 26 | 12 | 6 | 19 |
| Total | | 128 | 52 | 24 | 12 | 38 |

Each profile also includes one help command. No product source edits or compiler work were performed by this CLI validation. The external driver ran each CLI command serially with a 120-second timeout, a 2 MiB stdout limit and a 128 KiB stderr limit. All successful-profile commands completed without timeout, truncation or reader error.

The combined `report.json` records binary identities, counts, numerical comparisons and an independent recheck of every completed-command stdout/stderr receipt's byte length and SHA-256. Each successful profile has its own detailed `report.json`, `driver-job.json` and 128 exact stdout/stderr receipts. `manifest.tsv` hashes all evidence files except itself.

## Coverage

Each profile exercises thirteen native generation cases: six stories260K F32 cases, six cases using the native F16 derivative, and one stories15M Q4 case. Each case runs through CPU and CUDA. Generated stdout bytes equal the corresponding independent native expectation exactly, and CPU and CUDA outputs equal each other. These cases contain 208 completion tokens per backend.

Six stop cases per profile check prefix and middle stops for all three models on both backends, including repeatable stop options with a nonmatching sentinel followed by the actual stop. Both emitted bytes and completion counts match the expected native prefix.

Generation statistics are parsed as JSON and checked for backend, device, buffer-byte count, kernel-dispatch count, prompt/completion counts, finish reason, weight-byte count and finite nonnegative timings. CPU reports no device, CUDA-buffer bytes or GPU dispatches; CUDA reports the RTX 4090 and positive buffer/dispatch counts. CUDA commands select device 0 and a 128 MiB requested-buffer budget. This budget describes requested CUDA buffers, not a ceiling on every driver or device allocation.

The logits command is checked for valid JSON, exact input tokens, row/vocabulary shape and finite values. Each backend is compared with the independent native values, then CPU and CUDA are compared:

| Model | Rows per backend | Vocabulary | Native values per backend, per profile | Windows CUDA maximum absolute error | GNU CUDA maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| stories260K F32 | 5 | 512 | 2,560 | 3.337860107421875e-6 | 3.814697265625e-6 |
| stories260K native F16 | 5 | 512 | 2,560 | 4.172325134277344e-6 | 4.172325134277344e-6 |
| stories15M Q4_0/Q8_0 | 1 | 32,000 | 32,000 | 0 | 0 |

There are 148,480 native float comparisons across both backends and profiles. The unchanged comparison threshold is `2e-4 + 2e-5 * abs(expected)`. Windows CPU values are exact for all three corpora. GNU CPU F32 and Q4 values are exact; GNU CPU F16 maximum absolute error is 3.814697265625e-6 against the independently captured MSVC native F16 oracle.

Nineteen negative commands per profile require exit code 1 and empty stdout: missing device value; nonnumeric, negative or unavailable device; zero, negative, overflowing or insufficient GPU budget; unsupported Qwen2 or BERT CUDA architecture; unknown backend; device/budget options with CPU; CUDA backend on inspect, tokenize or embed; device option on detokenize; and negative or out-of-vocabulary logits tokens. Detailed error receipts are retained rather than inferred from a successful process launch.

## Pinned inputs and independent expectations

The model directory is `C:\Users\adyba\rust-llama-reference\models`.

| Model file | Bytes | SHA-256 |
| --- | ---: | --- |
| stories260K.gguf | 1,185,376 | 270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d |
| stories260K-f16-native-20261113.gguf | 601,248 | 4cd29bbe9cb3b735750837816a43c3dd4b473dbd658e8612816f7c4d3926ac21 |
| stories15M-q4_0.gguf | 19,077,344 | 66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739 |

Native expectations are repository fixtures:

- F32: `fixtures/inference/stories260k-reference.json` on Windows and `fixtures/inference/stories260k-gnu-reference.json` on GNU.
- F16: `fixtures/inference/cuda/stories260k-f16-msvc-metadata.json` and `stories260k-f16-msvc-logits.f32le.bin`, captured with the original C++ quantizer and C API. Both platforms replay this MSVC native corpus; the GNU result is not represented as a separately captured GNU native F16 oracle.
- Q4: `fixtures/inference/stories15m-q4-metadata.json` and `stories15m-q4-logits.f32le.bin`.

Model hashes were verified before command execution. Binary hashes were verified before and after each completed matrix:

| Profile | Bytes | Binary SHA-256 |
| --- | ---: | --- |
| Windows release | 2,259,968 | 37eea761225d4ca897b6aefb72688fc4de551de44665310fc90af4cabbfe4ecf |
| GNU debug | 33,470,664 | 14263e04fef1d69dd013af2bdca69da7def1b63ff6ed2df9afbbc7654b20f4dc |

Windows binary: `C:\Users\adyba\rust-llama-reference\continue-20261113\gpu\target-msvc\release\rust-llama.exe`.

GNU binary: `/home/tdamre/rust-llama-target-continue/debug/rust-llama`.

The external drivers are `validate-cli.py` and `validate-cli-gnu-bridge.py` under `C:\Users\adyba\rust-llama-reference\continue-20261113\gpu`. They invoke the actual binaries and preserve receipts; they are not maintained product code.

## Scheduling history and excluded attempts

The successful jobs are `job-1794606338585-ba3e0056-665` (Windows) and `job-1794606512673-04122ff7-dc7` (GNU). Both used the light scheduling class for one serial bounded CLI worker.

Two prior heavy-scheduled jobs expired after the 300-second queue wait before any child process started. Their terminal status is preserved in `queue-heavy-windows.json` and `queue-heavy-gnu.json`. These were scheduler admission failures, not CLI results or CUDA prerequisite failures.

A prior GNU light run was interrupted for parent-agent coordination after 26 completed commands. Its partial report and receipts are retained under `gnu-debug-interrupted`, together with `interruption.json`, `driver-job.json` and `guest-process-check.json`. The guest process check found no remaining matching driver or CLI process before the serialized retry began. The historical partial report still reflects its state at interruption; it is excluded from the successful counts above.

## Scope

This is actual CLI integration evidence for the named binaries, device, models, cases and options. It supplements the kernel, model-runtime, pretrained and ordinary workspace tests recorded elsewhere. It does not claim CUDA support for every architecture, quantization format, GPU or command.
