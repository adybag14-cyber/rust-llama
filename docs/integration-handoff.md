# Historical T09–T12 integration handoff

> **Historical imported snapshot.** The sections below describe the standalone
> T09–T12 archive before integration. Statements about missing tracks, an
> unavailable model runtime and future provider wiring apply to that archive,
> not to the current repository. The current server loads supported GGUF models
> and executes CPU generation and BERT embeddings. See [current port status](PORT_STATUS.md),
> the [server contract](../crates/llama-server/README.md), the
> [target matrix](target-and-runtime-matrix.md) and
> [expanded runtime verification](../evidence/2026-11-09/expanded-runtime/README.md).
> The [import manifest](imports/manifest.json) preserves the archive identities.
> This historical handoff does not replace the complete port requirements or
> the active [release gates](../release/gates.tsv).

## Scope and source identity

This is a self-contained progress workspace for the user's T09, T010, T011 and T012 assignment (track IDs T09–T12 in the plan). The only code input was the supplied original archive; no Rust implementations from the other chats/tracks were present. The final package therefore contains the latest implementations of these four tracks and explicit interfaces for the missing producers.

Baseline: `llama.cpp-master.zip`, SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. Toolchain: Rust 1.99.1, commit `58cf8f9c35b355199727627d07ae34a4e2846d67`, uploaded toolchain archive SHA-256 `231fcae47ee950621374725b5fc3b5441d2fbda2955ac45dd7dc816d8876cfe7`.

## Workspace packages

| Package | Owner | Role | Integration boundary |
|---|---|---|---|
| `llama-media` | T09 | Native codecs and image/audio preprocessing | Takes owned RGB/audio buffers; typed sequence/position/embedding data. Does not execute a projector. |
| `llama-behavior` | T10 | Sampling, native chat formats, grammar, JSON, diagnostic CLI | Takes explicit logits and token IDs; future tokenizer/runtime adapters supply actual model outputs. |
| `llama-server` | T11 | HTTP/SSE transport, request/provider contract, diagnostics rendering | Embed a reviewed `InferenceProvider` implemented over T05; default executable remains unavailable for inference. |
| `llama-conformance` | T12 | Comparison, digest, inventory, bounded probes, benchmarks, ZIP packaging | Tooling only; never a production model dependency. |
| `xtask` | T12 | Offline Cargo tasks, registries, audits, provenance and packages | Owns integration release evidence and refuses full release with open gates. |

## Merge with other tracks

1. Preserve the destination integration repository and its shared contracts. Add these package directories and workspace members; merge Cargo metadata instead of overwriting another track's root manifest. All packages use Rust 2024, require Rust 1.99 and have local-only Cargo dependencies.
2. Reconcile local `TokenId`, sequence IDs, cancellation, embedding and provider shapes with the actual T01/T02/T04/T05 contracts. These APIs are reviewable local seams, not a claim that a cross-track contract was frozen with absent owners. Avoid duplicate model/graph implementations.
3. Implement a real provider that owns inference sequence state, validates requested parameters, honors cancellation and sink errors, and reports real token usage. Model loading, tokenizer IDs, prefill/decode, KV rollback, adapters and sampling integration need their owning tracks.
4. Connect T09 preprocessing tensors to the architecture-specific projector/encoder builders. Compare real embeddings and logits before exposing model support. The shape/position containers do not supply model weights or inference.
5. Retain the independent fixture corpus and re-run the offline workspace tests. Extend coverage for platform-specific sampling RNG/tie behavior, full Jinja/grammar/tokenizer integration and all server protocols.
6. Keep all backend/model/native/browser gates open until their target executions are recorded. `cargo xtask release-check` must not be made green by deleting or reclassifying missing features.

## Provenance rules

`ledger/baseline-files.tsv` identifies every original file and proposed disposition. `ledger/baseline-registries.tsv` records source locations for discovered capabilities. These baseline rows remain planned because filename ownership and a partial port cannot establish full replacement. Domain documents identify what actually ran. `release/gates.tsv` is the mandatory closure ledger.

Reference objects, original C/C++ executables, the 210 MB compiler archive, toolchain installation and build caches are excluded from the ZIP. The Rust source package builds offline with the supplied compiler. No inference speed or complete-library parity is claimed by this checkpoint.
