# Continuation evidence — 14 November 2026

This checkpoint adds StarCoder and StarCoder2 CPU execution and HF conversion,
Qwen2/Qwen3 CUDA graphs, and eight context-attached sampling C APIs. It preserves
the earlier Llama CUDA, Falcon and context/FILE implementations. **The complete
frozen-source port remains unfinished, with all twenty mandatory gates open.**

## Integrated results

| Check | Windows MSVC | GNU/Linux WSL |
| --- | ---: | ---: |
| Ordinary debug tests | 768 passed | 772 passed |
| Ordinary release tests | 768 passed | 772 passed |
| Ignored prerequisites in each ordinary run | 49 | 49 |
| Result suites in each ordinary run | 147 | 147 |
| Separate pretrained inference/server tests | 16 passed | 16 passed |
| Separate native BERT encoder test | 1 passed | 1 passed |
| New release CLI commands | 15 passed | 15 passed |
| Final native libraries replayed | Debug and release | Debug and release |

The [integrated package](integrated-port/README.md) maps eighteen logical groups
and 27 successful unique command stages per host to their actual reports.
Windows v1 → v2 → v3 and GNU v1 → v2 preserve three failed attempts: two missing
README integrity declarations and one rejection of generated reference caches.
The single manifest-row correction and external cache preservation changed no
Rust implementation, fixture output, comparison bound or verifier policy.

The [final audits](final-audits/README.md) verify 1,143 fixture files covered by
1,154 integrity declarations across 30 manifests, both complete 4,082-file
frozen trees per host, 403 Rust
files and the same 23-local/five-registry dependency graph. The full-release
command returns its expected refusal with twenty mandatory gates open.

## Component and correction records

| Package | What it establishes |
| --- | --- |
| [StarCoder CPU](starcoder-inference/acceptance.json) | 72 configurations, including 43 executable variants and 29 checked rejections; 826,444 exact F32 values and 516 IDs per profile; three pretrained models add 3,151,872 values and 36 IDs per host. |
| [StarCoder conversion](starcoder-conversion/README.md) | 194 cases per host, with 163 exact complete files and 31 original failures; 1,248 alias queries; unchanged tiny StarCoder F16/Q8_0/auto outputs. StarCoder2 pretrained HF conversion remains unverified. |
| [Native sampling](c-backend-sampling/README.md) | Exact attach/accessor and CPU suffix behavior against original-header clients, with admitted prefix, ownership and transaction boundaries. |
| [Final C API replay](c-api-integration/README.md) | Sampling plus prior context/FILE checks against all four final library hashes; 247 frozen exports plus four extensions, eight absent. |
| [CUDA runtime](cuda-runtime/README.md) | Four actual GPU host/profile replays, 9,837,824 Qwen logits and 578 IDs per profile, existing Llama acceptance, 162 kernel cases and eight guards. |
| [CUDA numerical review](cuda-review/README.md) | Independent source review and the width-98 RMS division counterexample and correction. |
| [CUDA path correction](cuda-runtime-path-correction/receipt.json) | Four descriptive fixture-path corrections, with previous metadata preserved externally and unchanged raw/numerical data. |
| [CLI integration](cli-integration/README.md) | Fifteen actual commands per host across six models; 1,059,072 logit values, exact generation bytes/statistics and expected rejections. |
| [CI preparation](starcoder-ci/README.md) | Reviewed pinned tiny StarCoder acquisition/conversion/inference jobs and exact local syntax/selector/acquisition checks; hosted execution is recorded separately. |
| [Combined runtime](integrated-port/README.md) | Complete command coverage, failed-attempt history, metadata correction, source/library identities and retained native-client corpora. |
| [Final audits](final-audits/README.md) | Independent host audits after the reviewed CI workflow and dependency-documentation newline delta. |

## Identity and scope

Runtime packaging verifies 1,687 selected input paths with manifest SHA-256
`5de1123d3b7446c937ca5ee450a36a2bf1ebeae2115ff539fd82f4bc7fa578a3`. The final audit manifest is
`cd79f8a092cd60c2e6c55121b5e2d6daac1b71020c78aedb4f8582f302735136`. Between them, only the reviewed CI workflow and one trailing
newline in dependency-boundary documentation changed. All other selected
inputs, including every Rust, Cargo and fixture input, are identical.

The selected set excludes evidence, Markdown documentation, the progress gate
table and the derived non-Rust allowlist. Component manifests and copy ledgers
preserve exact raw receipts; final sealing verifies those packages and staged
bytes separately. External reference programs, model files and capture scripts
remain outside the native Rust product with recorded identities. Sealing logs
remain external to avoid recursive self-reference.

GPU runs share one RTX 4090 across Windows and WSL. GNU Qwen logits are bit exact;
the documented Windows F16/small-model values use unchanged numerical bounds.
These results do not establish unmeasured platforms, full architecture/backend
coverage, performance parity or release readiness. See
[current status](../../docs/PORT_STATUS.md) and
[remaining gates](../../release/gates.tsv).
