# Staged integration, 11 November 2026

Both original integrated-v1 attempts failed overall at the source-language
audit because the CLI fixture license had an unclassified filename. Each
had already passed formatting, ordinary debug tests, ordinary release tests
and strict all-target Clippy. Those original reports and logs are retained
unchanged, including the failed audit and the absent completion-time input
guard.

A separate post-attempt/pre-correction snapshot matches both original
starting inventories: 1,677 selected files, SHA-256
e1cdb7d8b8ea3a00052490f9d37f667c635c42a15f2c04da32e031e6f30a38f2. The correction renamed the license to LICENSE.md
without changing its bytes, updated exactly one fixture-manifest path and
raised only the CPU workflow timeout from 45 to 60 minutes. The selected
Rust sources, manifests for building Rust, and numerical/protocol fixture
bytes are unchanged. The exact before/after delta and all snapshots are
retained; 1,674 selected paths have
identical byte identities.

Both integrated-v2 continuations then passed every remaining required
stage against the final 1,677-file manifest, SHA-256
1ab73dd93269bcfc9b25f647ea06474568aaffc9e73c145b54d93a392c56acf0. Each continuation reached its complete unchanged-input
guard. The final manifest also matches the worktree before and after
evidence staging. Ordinary runtime suites were not repeated after this
packaging/CI-only correction.

This is an explicitly staged result. It does not describe either original
attempt as a successful full run or invent a completion guard for it.
summary.json assigns every required successful stage to its actual source
report and input manifest.

## Executed checks

| Check and source | Windows MSVC | Linux GNU |
| --- | ---: | ---: |
| Original ordinary debug tests | 728 passed; 0 failed | 732 passed; 0 failed |
| Original ordinary release tests | 728 passed; 0 failed | 732 passed; 0 failed |
| Original ignored prerequisites, debug / release | 33 / 33 | 33 / 33 |
| Original result suites, debug / release | 133 / 133 | 133 / 133 |
| Continuation pretrained inference/server tests | 13 passed | 13 passed |
| Continuation converter functions | 6 passed | 6 passed |
| Continuation native BGE encoder test | 1 passed | 1 passed |
| Continuation official Qwen tokenizer test | 1 passed | 1 passed |
| Continuation fixture files verified | 1062 | 1062 |
| Frozen C exports in each debug/release library | 232 plus 4 extensions | 232 plus 4 extensions |
| Frozen C exports still absent | 23 | 23 |

The remaining-stage continuations include the successful source/dependency
audit, fixture integrity, debug/release builds, both frozen-source copy
verifications, all explicit pretrained checks, and all retained original-header
native clients. The release-check command exits one as expected because
twenty mandatory full-port gates remain open.

## Native callers and component linkage

Each final Rust library passes the original-header chat, metadata, raw-name,
batch, sampler-performance and adapter clients. Exact exports include the
six encoder and quantizer additions. BERT runs 192 primary scenarios and
nine vector-admission scenarios. Quantization runs 88 file cases, including
77 successful cases and 11 rejections, with 76 complete output comparisons.
The actual numerical and bitwise comparisons remain in each native receipt;
bounded Rust errors are not mislabeled as identical original abort statuses.

All recorded parent logs and nested native artifacts are checked by byte
length and SHA-256 before copying. Embedded child reports must equal their
parent's entire recorded JSON value. Historical logs lacking capture-time
hashes retain explicitly labeled packaging-time identities. Reused native
caller build paths retain their older identities and require the latest
snapshot to match.

The sealed CLI component summary and all files in its artifact inventory are
unchanged. Its old fixture-manifest identity is checked against the retained
original manifest, then joined to the current manifest through the exact
license-path correction. The component evidence is never rewritten to point
at bytes that it did not originally verify.

Other conflicts, failed attempts, component receipts, static reviews and
compile-only preflights retain their actual statuses. They do not supply
additional successful stages. External implementation code, model payloads
and native executables remain outside this evidence package.

The full frozen-source port remains incomplete. Supported boundaries are in
[port status](../../../docs/PORT_STATUS.md) and
[C coverage](../../../crates/ffi/COVERAGE.md).
