# Staged integration continuation, 11 November 2026

The required integration checks completed across the retained Windows MSVC and
Linux GNU attempts. This is a staged aggregate: three original attempts remain
failed in their unchanged reports. The full frozen-source port remains incomplete,
and all twenty mandatory full-port release gates remain open.

## Results and contributing attempts

| Check | Windows MSVC | Linux GNU |
| --- | ---: | ---: |
| Ordinary debug tests | 691 passed | 696 passed |
| Ordinary release tests | 691 passed | 696 passed |
| Ordinary failures per run | 0 | 0 |
| Prerequisite-dependent tests ignored per ordinary run | 29 | 29 |
| Test-result suites per ordinary run | 124 | 124 |
| Separate pretrained inference/server tests | 11 passed | 11 passed |
| Separate complete-file converter tests | 5 passed | 5 passed |
| Separate official Qwen tokenizer replay | 1 passed | 1 passed |
| Fixture files / declarations / manifests | 959 / 959 / 14 | 959 / 959 / 14 |
| Source files checked in each supplied/reference tree | 4,082 | 4,082 |
| Debug and release frozen C exports | 226 + 4 extensions | 226 + 4 extensions |
| Frozen exports still absent | 29 | 29 |

Formatting and strict all-target Clippy passed in integrated-v1 on both hosts.
Those attempts also completed the ordinary debug and release suites before their
packaging audit failed. The actual-cache source/dependency audit, fixture
integrity, release workspace build, debug C-library build and supplied-source
verification passed in postflight-v2 on each host. Windows completed the remaining
checks in postflight-v3; GNU completed them in postflight-v2.

## Chronology and qualifications

1. [Windows integrated-v1](stages/integrated-v1/windows-msvc/report.json) and
   [GNU integrated-v1](stages/integrated-v1/linux-gnu/report.json) ended with status
   failed at the strict audit. Each reported exactly 29 unclassified artifacts:
   27 evidence logs with bare stdout/stderr suffixes and two nested attributes
   files. All preceding test, format and Clippy commands passed. Neither whole
   attempt is relabeled successful.
2. [The packaging correction](packaging-correction/correction.json) renamed the
   complete set of 43 adapter logs after the final adapter evidence was merged.
   It removed two nested attributes files and the corresponding fixture-manifest
   row, preserving the explicit pipeline binary rule at root. The classifier,
   implementation, tests, dependency inputs and numerical fixture bytes did not
   change. The selected input sets contain 1,537 and 1,536 files respectively:
   1,534 retained files are byte-identical, two retained control files changed,
   and one selected attributes file was removed.
3. [Windows postflight-v2](stages/postflight-v2/windows-msvc/report.json) passed
   audit, fixture integrity, builds and supplied-source verification, then failed
   verification of the separate reference copy. The verifier found exactly ten
   generated Python bytecode additions. The [quarantine receipt](reference-bytecode-quarantine.json)
   preserves their identities and location. No pinned source file was changed.
   The quarantine itself is not recorded as a source-verification pass.
4. [GNU postflight-v2](stages/postflight-v2/linux-gnu/report.json) and
   [Windows postflight-v3](stages/postflight-v3/windows-msvc/report.json) completed
   their requested remaining stages with successful statuses and unchanged input
   guards. Both supplied and reference trees pass strict verification across the
   contributing stages. The pretrained and native C comparisons below passed.

Ordinary tests were executed against runtime commit
979b1f1901637eaaaae7042aa2304eb95734fdc6. The postflight runs used packaging
correction commit ebeb0637932774e6b07d989ab3f01259bc363715. They are joined by the
retained manifest comparison, not by a claim that a failed attempt completed.

The original selected-input SHA-256 is:

    9cc92f74debdb3381b9998f2639b91f8efbbd4c81e4d62b934a62831cf2b6662

The final selected-input SHA-256 is:

    f4d5d64039bd7013fae32b5469799db984e6ef67e2aeae9c7fd91aff36d74ee2

The guard includes every Git-listed tracked/unignored file except evidence and
the two progress documents README.md and docs/PORT_STATUS.md. It includes release
controls, fixtures, fixture manifests and Git attributes. Failed attempts lack
the runner's successful end guard; the separate correction receipts prove their
original start manifests remained identical before the declared correction.
Both final attempts have their own successful unchanged-input guards. Packaging
also checks the current selected files against the exact final manifest.

## Numerical and original-header coverage

The five converter tests cover Llama, Qwen2, Qwen3, GPT-2 and GPT-NeoX complete
outputs. The official GPT-2 input's original rejection remains covered; the
positive derived fixture removes twelve static mask buffers while preserving
148 other tensors and checking its complete hash.

Both debug and release libraries pass original-header chat, metadata, raw-name,
batch, sampler-performance and adapter callers. Each adapter replay covers
44 load cases, 107 scenarios, 333 decodes and 170,496 F32 words. Batch replay
covers 279 cases, 395 calls, 640,000 exact logit words and 86,694 exact state bytes.
Sampler replay checks its exact semantic projection and logger bytes; native
clock timings are qualified by the captured predicates rather than identical
numeric timings. Complete export inventories remain 226 frozen exports plus
four existing Rust extensions, with 29 frozen exports absent.

## Retained records and release boundary

Each attempt has a distinct stages directory preserving the original report,
full input manifest, command logs, compact native caller receipts and export
inventories. Artifact-paths.tsv records every copied source path, stored path,
byte count, SHA-256 and integrity basis. Bare stdout/stderr filenames append
.log without changing bytes. All recorded command-log hashes are checked.
Adapter replay stdout/stderr and some inspector/metadata logs do not have prior
log hashes in their producer reports; those files receive explicit
packaging-time identities rather than an invented historical seal.

Pretrained checkpoints, prepared models and native raw result payloads remain
external. Their producer reports retain result/library identities and comparison
outcomes. External driver identities are pinned in summary.json. No reference
implementation is called by the maintained Rust product.

The expected release-check exit code is one: twenty mandatory gates remain
open. Complete architecture/backend/application coverage, remaining API and
operator modes, scheduling, writable storage, autodiff/training, platform
execution, and performance/memory closure remain outside this success claim.
Live GitHub CI publication is verified separately.

See [current scope](../../../docs/PORT_STATUS.md),
[tensor coverage](../../../crates/tensor/CPU_SUPPORT.md),
[C ABI coverage](../../../crates/ffi/COVERAGE.md), and [summary.json](summary.json).
