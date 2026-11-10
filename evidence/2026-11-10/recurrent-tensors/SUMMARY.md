# Recurrent tensor implementation and validation

Five original tensor operation IDs now execute in safe Rust:

| ID | Operation | Rust constructor |
|---:|---|---|
| 76 | SSM_CONV | `Op::SsmConv` |
| 77 | SSM_SCAN | `Op::SsmScan { snapshots }` |
| 82 | RWKV_WKV6 | `Op::RwkvWkv6` |
| 83 | GATED_LINEAR_ATTN | `Op::GatedLinearAttn { scale }` |
| 84 | RWKV_WKV7 | `Op::RwkvWkv7` |

The resulting constructor inventory is 81 executable IDs, plus the NONE input
leaf, with 20 IDs still lacking executable constructors. The complete llama.cpp
port and the full T02 tensor gate remain incomplete.

## Original source and independent expected results

The frozen archive SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
The original recurrent graph constructors, CPU kernels, SIMD mapping and
softplus helper were checked against the user's provided frozen source folder.
Reference builds, capture scripts and the initial candidate implementation are
retained outside the maintained Rust product.

The main native corpus contains 944 graphs per host and 1,147,172 output/state
bytes. The separate IEEE extension contains 450 graphs and 132,336 output/state
bytes per host. Every input word, shape, byte stride, parameter, worker count,
output descriptor and original output/state byte is preserved.

**All 1,394 graphs succeed in the original C implementation.** Rust executes
1,389 graphs with exact equality for 1,278,140 output/state bytes per host.
The remaining five graphs deliberately contain different NaN payloads that meet
at the same arithmetic operation. Rust returns an explicit Unsupported error
for this source-success domain. These five are not native error cases. Their
complete successful original outputs remain unchanged in the fixtures and the
test explicitly asserts each Rust rejection.

The explicit arithmetic boundary is:
`Unsupported("recurrent arithmetic with competing NaN payloads is unsupported")`.
It applies when an add/multiply meets two different quieted NaN encodings,
including one created by invalid arithmetic. Single quiet/signaling NaNs
preserve their quieted bits; distinct NaNs in independent rows remain allowed.
The initial MSVC/GNU debug/release payload discrepancies are summarized in
`initial-nan-observations.json`; that file is labeled as a transcription of
observed diagnostics, not a raw runner log.

No numerical tolerance, blanket NaN comparison exemption or replacement
expected output is used. All 450 raw IEEE extension graphs compare exactly.
The original extension's inherited descriptive provenance text was corrected by
a repeated independent capture whose binary was verified byte-identical; the
initial capture and generator remain outside the product.

## Preserved behavior and checked bounds

SSM_CONV preserves the source one-worker coefficient-row flattening and physical
input sequence stride, uses F32 reductions, and returns the original empty-output
shape without running a numerical kernel.

SSM_SCAN supports grouped heads, per-state and per-head A, remapped stored
sequence IDs, physical token/sequence padding and reverse-time state snapshots.
The output contains flattened Y followed by owned final/snapshot states.
It preserves the source `logf(1 + expf(x))` softplus expression.

RWKV6, gated linear attention and RWKV7 preserve original operand order,
equal nonempty sequence runs, first-three-axis token addressing, source
fourth-slab behavior and concatenated final state. RWKV7 keeps the two-pass
previous-state reduction/update order.

Unsupported dtypes, bad groups/shapes, invalid state IDs, out-of-view flattened
reads, unwritten snapshot domains, empty recurrent runs, unequal sequence runs,
insufficient time-first storage and ambiguous destination strides return checked
errors. Inputs remain immutable; cancellation does not publish partial state.

This is the measured scalar recurrent branch on the x86_64 MSVC and GNU native
profiles. Dense native three-worker controls match, but Rust general worker
scheduling and other ISA arithmetic are not implemented by this slice.
Floating exception flags and non-default rounding/FTZ/DAZ modes are outside the
validated contract.

## Validation results

| Gate | Windows MSVC | GNU/Linux |
|---|---|---|
| Full `llama-tensor` debug tests | 48 passed, 0 failed, 0 ignored | 48 passed, 0 failed, 0 ignored |
| Full `llama-tensor` release tests | 48 passed, 0 failed, 0 ignored | 48 passed, 0 failed, 0 ignored |
| `cargo clippy -p llama-tensor --all-targets --offline --locked -- -D warnings` | Passed | Passed |
| `cargo fmt -p llama-tensor --check` | Passed | Passed |
| `git diff --check` | Passed | Shared source worktree |

The 48 tests include nine new recurrent tests plus the existing tensor,
convolution, flash-attention, indexing, row-policy and numerical reference
suites. The reports retain exact command vectors, elapsed times, source-file
hashes and raw stdout/stderr log hashes. Rust 1.99.1 was used on both hosts.

After these recorded runs, one test-file documentation comment was clarified
to say that expected data comes only from original native graphs. No executable
code changed. `validation-summary.json` records the prior and current SHA-256
and verifies that replacing this single comment reproduces the validated file.
The parent integration run will test the final merged repository.

## Integration

The implementation worktree is
`C:\Users\adyba\rust-llama-reference\continue-20261110\tensor`.
The base is `b658976bd6200b8aa4b29f29f6a896b126dd756b`.

The external handoff directory contains:

- `recurrent-implementation-files.json`: 17 exact selected product paths, lengths and hashes.
- `fixture-rows.json`: eight fixture/provenance rows for the shared fixture manifest.
- `shared-tensor-edits.patch`: additive Op registration/dispatch/module changes for the three potentially shared Rust files.
- `evidence/recurrent-tensors/`: this summary, complete validation reports/logs and observed earlier NaN failures.

No code was committed or pushed by this agent. Shared root documentation,
source ledgers and manifests were left for parent integration.
