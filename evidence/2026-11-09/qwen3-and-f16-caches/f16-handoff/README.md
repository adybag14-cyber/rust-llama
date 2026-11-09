# Explicit F16/F32 nonflash native context handoff

This immutable follow-on contains 13 source/documentation files and six independently
captured fixture files. It is independently integrable before the separate per-layer
microbatch and flash-attention work. The original nonflash-checkpoint-v1 is unchanged.

## Supported and intentionally open settings

Dense Llama NativeContext accepts independent F32/F16 keys and values, explicit
FLASH_DISABLED, and exactly one generation worker and one batch worker. Existing
NativeContext::new and the safe strict Context remain F32/nonflash. The original
C default structure remains F16/F16, AUTO flash and four/four workers. Untouched
default context creation, AUTO/ENABLED, other worker counts and other cache types
are rejected with a diagnostic. This also closes the former silent manual
F32/one-worker/AUTO-to-nonflash fallback. No new C symbol is exported.

Raw F16 u16 storage preserves every bit during copies, pending-copy replay,
transposition, state/session I/O and CPU-owned snapshots. Evaluation alone converts
half values. Shared numerical operations retain checked nonfinite-result rejection:
raw NaN payload preservation does not claim parity with the original NaN logits.
No strict Context nonfinite check was removed.

## Independent proof

The ordinary FFI module uses losslessly deduplicated original tiny-model data on
both native compiler profiles. Per profile: all four K/V pairs; 160 memory cases,
160 sequence-state/flag cases, eight session suites, 384 short-buffer cases;
258,048 exact logit words and 1,048 exact buffer comparisons totaling 290,328 bytes.
This includes physical holes, deferred copy/rotation, restore/reserialize without
evaluation, full/sequence session failure side effects and sentinel boundaries.
The 520 original binary artifacts per profile reconstruct to their original hashes.

All focused runtime/FFI tests, build, format and strict all-target Clippy passed
in debug and release on Windows x64 and GNU x64. The external frozen-header C
adapter separately rechecked both trained and tiny models for all four type pairs.
See handoff.json for exact counts and evidence hashes. Deliberately forcing every
valid requested type to F32 storage fails the ordinary exact-state regression.

The preserved first-snapshot evidence also proves all 65,536 F16 patterns through
raw-state paths and 125 original strided KQ/PV graphs / 245,154 F32 outputs per host.
The graph fixture is common because both original output files are byte-identical.
The old trained captures and C sources remain external and are not product code.

Three initially queued release/negative jobs expired before a child process
started; their exact queue records remain in evidence. A single sequential retry
completed the remaining gates. These scheduling outcomes are not numerical failures.

## Integration

Apply only source.patch and the six declared fixtures after checking every
preimage. candidate contains the exact final bytes; preimages contains the
original root files. incremental-from-v1.patch contains only the ordinary
regression/docs and two typed-cache comment updates for consumers of v1.

The independent shard FFI handoff also changes crates/ffi/src/lib.rs and README.md.
Merge both test-module declarations and both documentation sections; do not
overwrite the shard adaptation with this older model-loader baseline. This
handoff does not touch crates/ffi/src/model.rs or any Cargo manifest. Preserve
root's intervening Qwen/shared-row/shard changes by applying narrow hunks.

Fixture manifest additions are in fixture-records.json. The existing
native-context-model.gguf is reused unchanged. Source-only tests need no trained
model download or original DLL. This does not close native flash/default worker,
recurrent cache, full C ABI or general performance coverage.
