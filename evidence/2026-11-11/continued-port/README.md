# Integrated native CPU continuation — 11 November 2026

This receipt qualifies the continued Rust implementation against the supplied
frozen llama.cpp source. It records selected native CPU execution, exact
independent comparisons and the limits of the integration evidence. **The full
port is not complete. All twenty mandatory release gates remain open.**
The baseline archive SHA-256 remains
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

The machine-readable [qualification](qualification.json) identifies the tested
hosts, stages, model outputs and snapshot review.
[Retained-file identities](retained-files.json) give the original external paths,
retained paths, lengths and hashes. Original failed reports are preserved;
their status has not been rewritten.

## Executed checks

Both Rust profiles used **Rust 1.99.1 and LLVM 23.1.1**: x86-64 Windows MSVC and
x86-64 GNU/Linux. Their corresponding frozen native references retain separate
compiler-specific expected bytes.

| Result per host | Windows MSVC | GNU/Linux |
| --- | ---: | ---: |
| Ordinary debug tests passed | 662 | 667 |
| Ordinary release tests passed | 662 | 667 |
| Failures in each ordinary run | 0 | 0 |
| Ignored prerequisite tests in each ordinary run | 27 | 27 |
| Test-result suites in each ordinary run | 117 | 117 |
| Explicit pretrained inference/server tests | 11 | 11 |
| Explicit complete-file converter tests | 3 | 3 |
| Explicit official Qwen tokenizer replay | 1 | 1 |

All 23 selected stages reached their declared exit codes. These include
formatting, strict all-target Clippy, release builds, debug C libraries,
dependency and fixture audits, source verification, native C clients and the
explicit model checks. The release check deliberately returned nonzero because
twenty mandatory gates remain open. Ignored ordinary tests keep their explicit
prerequisites; the separate model runs do not establish every device test.

Each host verified all **4,082 files** in both the user-pinned source tree and
the reference copy. The fixture audit verified **858 files with 858 declarations
in ten manifests**, including 138 additions in this continuation. Integrity
does not substitute for numerical or protocol comparison. The dependency audit
still admits 21 local packages and five reviewed registry packages, with 132
dependency source files and the pinned archive, feature and license identities.

The complete retained [Windows report](integrated/windows-msvc/report.json)
and [GNU report](integrated/linux-gnu/report.json) contain exact commands,
compiler identities, stage results and log hashes.

## Qualified implementation scope

The runtime has **seven admitted families**: Llama, Qwen2, Qwen3, GPT-2,
GPT-NeoX, Mamba-1 and BERT. GPT-2 and GPT-NeoX add independently captured
synthetic variants and official pretrained F16 replays. Their respective
pretrained comparisons cover 1,658,481 and 1,660,032 logit words plus 24 greedy
IDs per profile, with exact corresponding native bits.

Fresh Rust fixture preparation reconstructed both pinned GPT-2 and Pythia
GGUF inputs before the integrated inference tests. GPT-2's disclosed recipe
omits exactly twelve verified nonlearned static causal masks; retained learned
tensors are unchanged. Pythia needs no such omission. This fixed test-data
preparer does not establish general GPT-2 or GPT-NeoX checkpoint conversion.
Model weights and original executable reference tooling remain outside Git.

The tensor inventory reaches **94 executable operation IDs plus NONE input
leaves**, with seven IDs pending. New qualified operations cover vision/loss,
gated delta, hyper-connections, Lightning Indexer and backward RoPE.
Source-success domains excluded by the Rust adapters remain recorded as
explicit unsupported cases, with the original expected bytes retained.
This does not close graph scheduling, allocation planning, automatic
differentiation, optimizers or complete device execution.

Both debug and release libraries expose exactly **216 frozen C exports plus
four diagnostic extensions**; **39 frozen exports remain absent**.
Original-header chat, model-metadata and raw-name clients run alongside the
new batch and sampler-performance clients. Per library, the batch corpus has
279 scenarios, 395 process calls, 386 successful decodes, 640,000 exact logit
words, 2,688 embedding words and 86,694 state bytes. The performance corpus
has eight cases, 324 operations and 114 timing observations, with exact semantic
projection and default stderr. Timing values are measured rather than required
to match elapsed durations. Native C execution remains scoped to admitted
dense-Llama decode contexts; context/graph metrics, encoder processing,
backend-attached sampling and throughput equivalence are not established.

General checkpoint conversion covers **three dense families: Llama, Qwen2
and Qwen3**, through seven dispatch names. The new ordinary Qwen corpus contains
152 cases per native profile: 144 complete accepted files, eight rejections and
32,372,288 exact output bytes. The three explicit pretrained converter tests
cover SmolLM2 and both official Qwen checkpoints. The six Qwen outputs total
6,342,831,936 bytes per profile; the collector rehashed all twelve actual Qwen
outputs across both hosts. Pinned tokenizer, class-selection, prefix and NFC
evidence remains distinct from inference-architecture coverage.

## Execution-input change and postflight

The original runs began with **1,280 selected execution inputs** and manifest
SHA-256:

```text
fd4ef2611911a172fe483a6765f480aeec86703075a2ad8195ebe51e03a6231e
```

After launch, the CI workflow gained the reviewed isolated Qwen jobs. The
original final guards therefore failed with the exact recorded error:

```text
AssertionError('execution inputs changed during validation')
```

The [snapshot review](snapshot-review/review.json) proves that
`.github/workflows/ci.yml` is the only changed recorded input; all **1,279**
other recorded source, fixture, lockfile and build inputs are byte-identical.
The final 1,280-input manifest SHA-256 is:

```text
9da6d0bca8779e1691fbae7eca826155ef5d4ed23065bdbec0167ee9b5c37688
```

Both [Windows postflight](postflight/windows-msvc/report.json) and
[GNU postflight](postflight/linux-gnu/report.json) then completed formatting,
dependency/source audit, fixture integrity and the expected failing release
check with this final selected-input manifest unchanged. The postflight does
not pretend to rerun the complete numerical suites, and its success does not
erase the original guard failures. These manifests select execution inputs;
they are not whole-repository prose, license and release-policy guards.

## Publication and remaining work

The first published workflow was rejected before any job started in
[run 43005930716](https://github.com/adybag14-cyber/rust-llama/actions/runs/43005930716).
Its Qwen paths used the runner context at job-level environment scope. The
correction moves those same values to the consuming steps, preserving every
command, model pin and expected output. The [correction record](ci-workflow-context-fix/correction.json)
retains the failure and corrected workflow/input identities. All 1,279 non-CI
selected inputs remain unchanged. The earlier YAML/PowerShell static review did
not validate GitHub expression-context availability.

The [CI review](qwen-ci-review.json) checks the four isolated Windows/GNU by
Qwen-model jobs, pinned downloads, input hashes and explicit release commands.
The original CPU job, GPT-2/Pythia preparation and complete ignored
inference/server command remain unchanged. **Published-commit GitHub CI is recorded separately by GitHub Actions**;
static workflow review and local execution do not substitute for that result. The [publication review](publication-review.json)
records the read-only scope and object-size inspection.

The [release ledger](../../../release/gates.tsv) retains all existing statuses:
22 validated component rows and twenty open mandatory rows. Full architecture,
ABI semantics, tensor/backend, multimodal, browser/client, platform, performance,
quality, memory and distribution requirements remain open. No numerical
expectation or tolerance was weakened to obtain these results.
