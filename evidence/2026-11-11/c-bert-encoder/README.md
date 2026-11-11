# Native BERT encoder C validation

This evidence qualifies four frozen C exports: `llama_encode`,
`llama_get_embeddings_seq`, `llama_model_n_cls_out` and
`llama_model_cls_label`. The implementation executes BERT token/direct
embedding batches, compiler-native sequence grouping, all five pooling modes
and optional classifier heads. See `crates/ffi/ENCODER.md` for the bounded
compatibility contract.

The validation chain has two stages. The full model-runtime, inference and
FFI checks at v3 passed 180 ordinary tests, zero failed, with 11 deliberately
ignored pretrained tests across 25 suites in both debug and release on both
hosts. The pinned native and facade BGE tests were run explicitly and passed.
The final active-vector admission change then passed the affected inference
and FFI packages: 99 tests on Windows and 98 on GNU, zero failed, 11 ignored
across 11 suites in each mode, followed again by both explicit BGE checks. Strict Clippy
and formatting passed at both stages. The final change is confined to the
BERT storage guard, its new regression, domain documentation and fixtures;
the existing arithmetic and runtime source bytes are unchanged.

The independently compiled original-header C caller passes 192 primary
scenarios against each final debug/release Rust library on both platforms.
Each checks 37,384 floating-point values with zero bitwise differences and
maximum absolute error zero, plus exact statuses, model/context facts,
labels, output indices, offsets, nullness and pointer stability. The primary
corpus contains 188 tiny scenarios including 13 expected model-load
rejections, plus four pretrained scenarios accounting for 28,032 values.
Both independent 569-case integer-set profiles are exercised by the ordinary
tests.

A separate nine-case vector-storage replay runs against each final library.
Five graphs that abort in the original native binary dispatcher are bounded
model-load rejections in Rust. Two ignored classifier biases and two F32
controls execute and match 137 values with zero bitwise differences and
maximum absolute error zero. These nine cases are not included in the
primary 192-case count. Every library's exact inventory is 230 frozen exports
plus four Rust extensions; 25 frozen names remain absent at this package base.

## Independent inputs and retained failures

The primary original-library corpus was sealed at the clean base commit
before implementation edits. The original header clients, C++ sources,
Python reference tooling, original libraries and generated builds remain
external. Their identities are retained here and in fixture provenance;
none is a product implementation or runtime dependency.

All original, failed and superseded reports are preserved. Initial F16/BF16
token-type fixtures triggered source graph assertions; corrected fixtures
retain F32 type tables. The source-rejected INT32 classifier-pooling fixture
remains negative, with a separately captured UINT32 counterpart. The first
Rust shared-sequence comparison failed because it sorted memberships.
An independent original C++ set-copy corpus established MSVC/GNU behavior
before that path was corrected; no numerical tolerance was widened. A later
facade-only test selector included multi-sequence cases accidentally and was
corrected without changing model execution.

The first GNU validation driver stopped at a Windows worktree Git path
before compilation; its translated-path successor passed. The first Windows
run passed every runtime/C/fixture stage but failed its final input guard
after an owned README edit. Its documentation-only delta is retained in
`preflight-fix-receipt.json`, and the failed report is unchanged.

Independent review found a direct-embedding allocation preflight gap.
The corrected implementation checks shape/work bounds before every foreign
input slice or owned row copy, then validates row contents before changing
outputs. Its dedicated regression distinguishes early shape rejection from
missing-input rejection and checks that prior pooled outputs survive.
The complete v3 runtime validation covers this correction.

GNU v3 passed its runtime, pretrained, C ABI, exports, fixture and source
audits, then spent several minutes inside a redundant Git whitespace check
over the Windows filesystem. Only the identified Git child was terminated.
Its actual exit -15 and failed overall report are retained. The completed
Windows whitespace check covered identical source bytes, and a separate
Windows-native continuation manifest proved all 1,560 selected inputs
unchanged. No failed GNU receipt has been relabeled as successful.

The subsequent native vector-type addendum was captured on both original
libraries before the loader guard was added. The originals abort when active
F32 graph results use stored F16/BF16 norm/bias operands. Classifier biases
without their paired matrix remain ignored; the guard preserves that case.
The documented model-level restriction rejects a paired non-F32 classifier
bias even if a later context would choose ordinary embedding pooling.
The final 15-path delta and pre-guard seal preserve this ordering.

## Reading the final receipts

`summary.json` links the two validation stages, exact guard delta, final
1,570-file shared input manifest, native callers, fixture manifest, pretrained
identity and independent reviews. `validation-vector-v1` contains the final
compiler/test/C receipts. The `*-c-raw.jsonl` files preserve reconstructible
exact UTF-8 output bytes with per-file SHA-256 values. Nonempty logs and full
historical reports are retained; parent reports identify empty logs by hash.

Windows hashed the full selected shared-worktree manifest before and after
its incremental run and again after both hosts completed. GNU checked the
15-path change locally and inherits the final Windows-native whole-manifest
seal. This avoids repeating the completed slow GNU Git traversal. The final
package source-language, fixture and whitespace checks are recorded
separately in `package-postflight-v3/report.json` as a commit gate.
`copied-evidence-identities.json` records source/destination identities;
`evidence-files.tsv` inventories the final evidence package.

Before final copying, `receipt-chain-verification-v2.json` checks every recorded
log, native request/result, child C-report, manifest and continuation identity
in 43 reports: 8,226 recorded links across 5,738 distinct files. It also checks
inline source outputs and parent/child library/count agreement. Fifty earlier
log fields had no capture-time identity; they are explicitly listed rather
than assigned invented historical hashes. The initial package wrapper and
its passed audit are retained separately; product source and runtime evidence
did not change.

This milestone leaves the full-port release gates open. Architecture and CPU
profile limits, unsupported options, allocation/work bounds and pointer
lifetime requirements remain explicit in the domain documentation.
