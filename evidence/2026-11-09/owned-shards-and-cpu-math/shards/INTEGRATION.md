# Sharded GGUF handoff

This directory is an immutable external handoff, not a product commit.
The current root checkpoint remains unchanged.

## Merge boundary

Apply source.patch after checking each integration_preimage_sha256 in handoff.json.
It changes eight existing GGUF/inference files and adds two Rust files.
Do not replace the entire inference crate, runtime, tokenizer, quant, server,
FFI, Cargo files, CI or existing fixture manifests. The patch preserves root's
current /Qwen2 module comment. That comment is the only documented difference
between the validated inference/lib.rs and its prepared integration postimage.

Copy only the six candidate/fixtures/gguf data/provenance files after checking
fixture-records.json. Append the six reviewed fixture-declarations.tsv rows to
the shared integration manifest; existing reference rows remain unchanged.
No foreign implementation, native oracle or pretrained model binary is added.

documentation/GGUF.md is a complete proposed crate README/API/limits/provenance
document. Root owns final top-level status and inference README integration.
Update those scope statements to describe automatic and explicit Rust shard
loading while retaining the ordered writer/converter requirements.

## Required FFI integration

Read required-ffi-integration.json before promoting this slice.
The current C model loader performs a separate single-file metadata/statistics
pass. Its existing auto-load would now accept shard sets through the facade
and report incorrect first-shard-only size/parameter/guessed-format values.
Core/root must adapt that path to the same GgufSet ownership, or retain explicit
rejection until it is independently validated. This handoff does not edit FFI
or add native load_from_splits/split-path exports.

## Validated scope

The fixed candidate passed GGUF/inference ordinary debug and release suites,
strict Clippy and formatting on Windows x64 MSVC and Linux x64 GNU. Each host
then explicitly ran all seven hash-pinned pretrained inference model tests.
The source/fixture fingerprint stayed unchanged throughout both runs.
See handoff.json and evidence/validation/buffered-final-{windows,linux}.

This is focused validation of the loading slice. Parent integration still uses
the combined workspace, applications, audits and gates before its own commit.
Some runtime/NativeContext/tensor/server files in the root combined draft have
advanced after the source snapshot used here; none is replaced by this patch.

Native captures were completed before the first Rust model comparison. The
six new fixture bytes remain identical to their declared native-data identities.
Only provenance wording/model-origin hashes were clarified separately, with
the exact unchanged-data check retained. Both platform float corpora are kept.

## Retained failures and reviews

The two reservation preflight tests fail in the deliberately reverted external
negative control. The first replay attempt had a missing inherited workspace
lint section; that setup failure is preserved separately. The first Linux
validation driver needed a Python 3.10-compatible hash loop. Existing Mamba
and Qwen tests had single-file-only Unsupported expectations; those are replaced
with precise missing-split.no Invalid assertions, not removed.

Independent review found the allocation preflight gaps and later confirmed the
fixes, scalar/zero-dimension/layout semantics and bounded HeaderReader boundary.
UTF-8/NUL-name and other documented stricter domains are not misrepresented as
full raw-format parity.

The current reader had a measured WSL mounted-filesystem metadata bottleneck.
Before/after header-only Qwen measurements retain source/model/binary identities:
760,618 File reads / 100.5658 seconds versus 93 reads / 0.07945 seconds.
The single-run timing is qualified; deterministic read-count, byte-budget,
short-read/error and retained-position tests establish the change separately.

## External reproduction material

The original source/builds, C++ adapters, untouched original splitter,
capture scripts and raw MSVC/GNU outputs remain under the parent
sharded-gguf-port reference directory. shards-provenance.json pins those exact
identities. No native expected output is derived from Rust. Keep this material
outside the maintained product.
