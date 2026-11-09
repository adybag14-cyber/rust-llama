# Native Rust GGUF streaming split and merge

This checkpoint adds ordered borrowed-header writing and a bounded native Rust
llama-gguf-split executable. It is part of the incomplete full port.

Both integrated platform proofs cover all 17 checks against the same 648 source,
compiler/configuration and data inputs:
abd901487f73b01db4682a159c75b5a20c7e93b9e2c51cd81f2431923ca44a7c.
Windows passes 460 ordinary tests in both profiles; GNU passes 464. Nineteen
prerequisite tests remain ignored in ordinary execution. All 9 explicitly run
pretrained comparisons pass per host, with exact measured native model values.
CLI, HTTP/SSE, source integrity,465 fixture declarations and strict Clippy pass.
The release check still fails because all 12 full-port gates remain open.

The original CLI corpus has 103 cases per host: 63 defined successes and 40 checked
errors. It verifies 12,179,168 complete post-operation file bytes, source parser
rules and successful transcripts, while retaining explicit safety/UB differences.
Resource and alias regressions exercise pre-allocation header budgets,
cumulative merge budgets, one-byte copy buffers and output replacement after
planning. Raw original-only NUL/invalid-UTF8 cases remain an open boundary;
they are not described as implemented parity. A separate 512-case original-only
boolean corpus documents nonzero-byte normalization for subsequent work.

Windows source verification initially caught two generated reference Python
cache files. Only those files were relocated after the source module was checked
against the authoritative ZIP, and source/release checks then completed.
GNU completed its ordinary checks before an intentional resource pause; its
remaining checks run under a separate same-input continuation. Original status,
logs, failures, partial build output and correction identities are preserved.

Source-native output quirks, valid public writer semantics, destructive cleanup
requirements and remaining conversion work are documented in crates/gguf/README.md.
summary.json and source-provenance.json map the evidence to complete original
records. The previous published commit 1890f032 has independent successful
Windows/Ubuntu hosted CI logs below.

The final packaging audit rejected 344 binary evidence paths outside declared
fixture directories. Their evidence copies now use lossless hexadecimal text;
integration/evidence-encoding-v2/record.json records original and encoded hashes,
paths, lengths and the decoding rule. Raw originals remain in the external
reference archive. The audit failure is retained. Execution inputs, ordinary
fixture bytes and expected test results are unchanged.

A second final fixture audit identified four archived development documentation
snapshots under a directory named fixtures. Their archive directory is now
fixture-snapshots; integration/evidence-paths-v3/record.json preserves exact
before/after paths and unchanged hashes. This changes no fixture declaration.
