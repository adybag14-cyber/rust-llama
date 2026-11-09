# C split-file loading and filename helpers

These three exports are implemented in Rust and use the shared GGUF loader.
The original llama/ggml library is an external oracle, never a product
dependency. The frozen interface is the uploaded `include/llama.h`, archive
SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

## Explicit model lists

`llama_model_load_from_splits(paths, n_paths, params)` retains an owned copy of
every supplied C-string path before invoking a progress callback, then passes
the complete ordered list to `GgufSet::open_splits`. It shares construction,
owned file handles, metadata, statistics and execution restrictions with
`llama_model_load_from_file`. The automatic entry point also owns its path
before callbacks.

The primary file supplies metadata and determines whether a split set is
required. Frozen single-file behavior can ignore extra supplied filenames when
the primary does not declare a multi-file set. A declared split set uses caller
order and validates count, indices, names and payload extents; arbitrary
filenames are permitted. Model/vocabulary restrictions and checked text metadata
remain those in README.md. This entry does not establish every architecture,
option or C ABI behavior.

On Unix the explicit list preserves raw OS path bytes, including invalid UTF-8.
On Windows it requires UTF-8; other encodings return a checked diagnostic.
Every path is NUL-terminated. Caller memory must remain live and correctly
aligned, and callbacks obey the crate's non-reentrancy and lifetime rules.
Callbacks may change the original path buffers or pointer array after the
loader has retained them. The existing 0/1 progress milestones remain;
complete source progress granularity is not claimed.

An empty list, null nonempty array, null path element, or more than 16,777,216
foreign elements returns NULL with a diagnostic. Path copies and the owned
pointer-list allocation use fallible reservation. The count ceiling is not a
total path-byte or process-memory bound. Invalid foreign allocations and
allocator aborts remain outside recoverable ABI guarantees.

## Byte-oriented filename helpers

Both helpers read bytes through the first NUL without UTF-8 conversion. They
preserve the zero-based `split_no + 1` rule and signed `%05d` minimum width,
including the minus sign. They do not apply the separate GGUF file-set
validator's unsigned-16-bit/count restrictions.

| Entry point | Output and return |
|---|---|
| `llama_split_path(dst, capacity, prefix, no, count)` | Forms `prefix-%05d-of-%05d.gguf`, with first value `no + 1`. It writes as much as fits and terminates any nonzero-capacity output. It returns the full length only if the complete string and terminator fit; otherwise 0. |
| `llama_split_prefix(dst, capacity, path, no, count)` | Matches the suffix formatted in the source's 32-byte local postfix. A mismatch or empty prefix returns 0 without writing. A match writes at most capacity minus one prefix bytes and returns the full prefix length, even for short or zero-capacity output. |

The source's 32-byte postfix can truncate a maximum-width signed suffix to
31 bytes; matching preserves that behavior. NULL output is valid at zero
capacity. A nonzero-capacity write needs valid writable memory. Outputs must
be exclusive and disjoint from input strings.

The source expression `split_no + 1` overflows for INT32_MAX. Rust rejects that
case before writing, with return 0 and a diagnostic. Null input strings,
null output when a write is required, and lengths outside the returned INT32
domain are checked errors. Arbitrary nonnull foreign pointers cannot be
validated. Source-defined returns of 0, such as mismatches or short path
buffers, do not set a diagnostic. Successful calls do not clear an older
thread-local diagnostic; callers can use `llama_rust_clear_error` first.

## Independent original corpora

Outputs were captured and sealed before Rust implementation. Helper results
are identical on MSVC/GNU. Models retain separate exact compiler profiles:
the original hosts differ in 58,590 shared F32 words, with maximum absolute
difference 5.960464477539063e-08.

| Corpus, per measured host/build profile | Windows MSVC | GNU/Linux |
|---|---:|---:|
| Filename-helper calls | 7,807 | 7,807 |
| Positive helper returns | 2,884 | 2,884 |
| Complete output bytes, including guards | 506,563 | 506,563 |
| Explicit model-list calls | 104 | 106 |
| Accepted loads | 55 | 57 |
| Checked load errors | 49 | 49 |
| Original assertions replaced by checked errors | 2 | 2 |
| Exact continuation logit words | 172,032 | 180,224 |

The extra GNU cases use raw non-UTF8 filenames. Helpers cover every input byte,
embedded NUL, Unicode/arbitrary bytes, signed extrema, wrong/truncated suffixes,
short/zero/null-zero outputs and sentinel guards. Of the compared output bytes,
256,739 lie within declared capacities; the rest are guard bytes.

Model lists cover arbitrary names, Unicode/raw Unix paths, 35 shards,
metadata-only primaries, single-file extras, missing/short/reordered/duplicate
lists, reserved key types, truncated payloads and empty lists. Accepted cases
compare getters, token/piece/detokenization outputs and greedy logits. The
logit totals include repeated model/list configurations, not that many unique
numerical inputs.

The FFI suite passes in debug/release on both measured hosts, alongside strict
Clippy and source/fixture audits. External callers built from the unchanged
original header replay the fixed requests against each Rust DLL/shared object.
Complete export tables contain the prior 184 functions plus these three,
with the same four Rust extensions.

An independent adapter verifier pins both original seals, the full capture
inventory, reports, matrix and every accepted output. It compares all 14
existing fixture projections and rechecks 756 original files before and after.
The earlier packer lacked explicit rehash guards; its source/output records
remain preserved with this separate verification correction. No Rust output
generates expected values.

Four deliberately incorrect isolated implementations must compile and fail
their selected regressions: omit the zero-based increment, return 0 for short
prefix output, replace the explicit list with discovery, or own only the first
shard before the callback. The last control snapshots raw pointer values before
the callback and keeps no shared foreign borrow; it intentionally copies the
remaining live modified strings too late. This tests ownership without reading
a mutated/null pointer array. Candidate and fixture hashes remain unchanged.

See `fixtures/ffi/split-helpers-provenance.json`,
`fixtures/ffi/split-list-provenance.json` and
`evidence/2026-11-09/c-split-api`. Twenty mandatory full-port release gates remain
open across twelve ownership tracks.
