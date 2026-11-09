# Expanded native runtime checkpoint

This checkpoint integrates Mamba-1 and BPE/Q4_K_M generation, the remaining
frozen tokenizer families, scalar/indexed tensor operations and the native
context/memory/session C API subset. It does not close the full-port contract.

## Immutable execution inputs

Both Windows and Linux checked the same 396 source/configuration/fixture inputs:
050cbc4b0c607a4be1bd7ade8d590f0b49b52b3cbbc284b12c8cfa5ed88413a9.

The fingerprints are checked before every validation step and after completion.
Prose and mutable evidence are excluded from that execution fingerprint. The
195 actual fixture files have 195 integrity declarations across four manifests;
all 4,082 frozen original source files retain their expected hashes.

## Results

| Gate | Windows | Linux |
| --- | --- | --- |
| Debug workspace tests | 384 passed; 17 explicit prerequisites ignored | 385 passed; 17 explicit prerequisites ignored |
| Release workspace tests | 384 passed; 17 explicit prerequisites ignored | 385 passed; 17 explicit prerequisites ignored |
| Formatting / strict Clippy / audit / release build | Passed | Passed |
| Explicit pretrained tests | 7 passed | 7 passed |
| Generation HTTP checks | 37 passed | 37 passed |
| Embedding CLI lifecycle | 11 scenarios; 12 native comparisons | 11 scenarios; 12 native comparisons |
| SmolLM2 / Mamba generation CLI | 2 exact text/usage comparisons | 2 exact text/usage comparisons |
| Full release gate | Correctly blocked (exit 1) | Correctly blocked (exit 1) |

The focused external-C ABI/state checks and tokenizer-family getter checks are
under ../native-context and ../tokenizer-families. The 62-op tensor slice has
1,252 distinct native graph cases; platform-specific sort ordering and the open
exact-ERF boundary are documented separately.

## Evidence sequence and retained failures

The first full run passed tests and Clippy but found three new tensor license
files absent from the explicit source-audit list. Only those specific license
paths were declared, with the existing unknown-license rejection test extended.
A fresh complete run is under revalidated.

That run then exposed an external CLI driver filename error. Runtime execution
inputs did not change. The continuation in revalidated/continued proves the same
fingerprint, retains the seven completed build/test/audit gates, and executes the
corrected CLI and remaining pretrained/HTTP/source/release gates. This is
explicitly recorded in verification.json rather than replacing the failed run.

The embedding CLI driver also reused old evidence output paths. The new reports
were preserved here, the prior committed bytes were restored from Git, and the
driver now writes only to this checkpoint. driver-corrections.json identifies
every restored file by its original hash.

The independent original fixture data and numerical thresholds were unchanged.

## Final incremental resource correction

After the full run, independent review found that NativeContext counted only the
first microbatch's additional visible KV rows. The final source now counts the
entire incoming batch (bounded by cache capacity) before staging any state.

Only crates/model-runtime/src/context/native.rs changed among the 396 execution inputs.
Substituting its recorded prior hash reproduces the full-run fingerprint exactly.
The final fingerprint is 26399fc8edf90a9c3871cea43e9212757d08094875bb046efe11898504ce2afa.
The unchanged full-run reports above are preserved as that base, not relabeled.

The patch passes 64 focused tests per debug/release profile on Windows and 62 on
Linux, strict Clippy on both, and 489,158 independent C float comparisons plus
native state/metadata per platform/profile. A negative-control build with only
the old bound restored fails the new near-budget regression. The corrected
implementation rejects atomically at the insufficient limit and executes all
512 tokens across 128 microbatches at the sufficient limit.

See final-checkpoint.json and ../native-context/budget-revalidation for exact
sources and evidence. Full CI will additionally validate the pushed commit.
