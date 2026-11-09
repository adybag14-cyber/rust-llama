# Raw GGUF handoff

Base: published 40a4ee7deb59e5823dd6d242231592d0c9923168.
All 31 candidate files are guarded by exact base identities or required absence.
Append the 23 fixture declarations; no Cargo/dependency change is involved.
The six Rust source/test files and 23 new fixtures are the only executable-input
changes. The public text facade keeps its UTF-8/NUL key/name boundary; raw
reader/writer and split/merge APIs implement the independent native byte domain.

Final frozen input set: 744 files,
d5b0b9ed78260a8a2dc91b79b0a2d5f7277ffbf6fdbd942a1299cb6135a9401e.
Original native expected bytes were captured and sealed before Rust comparisons.

Windows v1 selected debug/release: 73 passed, 11 external prerequisites ignored.
Strict Clippy then rejected chunks_exact(2) in the test-only hex decoder. The
v2 completion changes exactly that helper to as_chunks::<2>(), reruns all 22
GGUF tests in debug/release, and passes fmt, strict Clippy, source audit and
488 fixture declarations. Production sources and fixtures are unchanged.
GNU v2 runs all focused checks on the exact final input set. Its test results,
logs and final statuses are retained without aggregation across repeated runs.

Three independent negative controls per host modify only an isolated code copy:
boolean-only-one truth, last-prefix tensor lookup and later-alias tool payload.
Each compiles and fails the unchanged native comparison through a test assertion.
Both the frozen source and fixture identities remain unchanged after controls.

All original assertions, raw failed observations and implementation/driver
failures remain evidence. The invalid direct WSL launcher failed before child
creation; its correction uses the same explicit Python-to-WSL execution route
as the other pinned Linux checks. It changes no oracle or product source.

When merging concurrent documentation changes, update the GGUF portions of the
root README and PORT_STATUS to point to docs/GGUF_BYTE_STRINGS.md. Replace the
old whole-reader non-UTF8/NUL restriction with the exact remaining high-level
text-model boundary. Nonzero boolean byte compatibility is now implemented.
Retain the published 103-case streaming corpus and all prior timing/deletion/
undefined-behavior qualifications. Do not imply T04/full-port completion.
All twelve release gates remain open.

Original binary pools/reports already stored as new fixtures are referenced by
exact identity. Other binary evidence uses lossless hex text with decoded and
encoded identities. Archive paths named fixtures become fixture-snapshots;
historical implementation/config sources receive .txt to keep them data.
No original reference file, expected value or tolerance is altered.
