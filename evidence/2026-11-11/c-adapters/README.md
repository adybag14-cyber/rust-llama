# Native adapter evidence — initial integration handoff

This checkpoint adds ten implemented LoRA/control-vector C exports. The exact
fixture corpus and ownership/resource boundaries are described in
`crates/ffi/ADAPTERS.md`; `docs/adapter-fixture-manifest.tsv` pins all 53 retained
fixture files. No complete-port, performance or unqualified-platform gate is
claimed.

## Completed validation at the handoff

| Profile | Debug crate suites | Release crate suites | Strict Clippy / format | Original-header C caller | C export inventory |
|---|---:|---:|---|---|---|
| Windows x64 / MSVC | 140 passed | 140 passed | Passed | Debug and release exact | Both: 226 frozen + 4 existing extensions; 29 absent |
| Linux x64 / GNU | 138 passed | Pending | Passed | Debug exact | Debug: 226 frozen + 4 existing extensions; 29 absent |

The Linux optimized run launched in a detached guest was interrupted during
compilation. Its retained `linux-debug/report.json` explicitly records
`interrupted`; it provides no release result. Completed debug, Clippy and C
caller results remain individually recorded. A durable foreground GNU run and
the parent integration's whole-workspace release check will supply subsequent
release evidence. The Windows report is complete and passed.

Each C caller run checks all 44 load cases (34 accepted, 10 rejected), metadata
buffer writes, invocation-token bits, 107 execution scenarios, 333 decodes and
170,496 F32 logit words against independently captured original results.
There is no numerical tolerance or Rust-generated expected output. Two native
MSVC pointer-map orders are captured independently, so overlapping adapter
rounding is checked without requiring identical allocator addresses.

The compact inputs use synthetic dense Llama models with embedding width four,
two/four layers and vocabulary size 512, plus rank-two adapters. The corpus
covers every projection category and F32/F16/BF16 adapter payloads; it does not
measure pretrained-adapter quality or production throughput. Existing affected
crate suites additionally retain their broader model, state and kernel checks.

## Records and source identity

`windows/report.json` records the completed commands, debug/release library
hashes, original C output identity, exact export sets and tested source-file
hashes. `linux-debug/report.json` preserves the completed GNU debug evidence and
the explicit release interruption. The recorded Git HEAD is the base checkout;
source-file identities pin the actual changed code under test before its final
commit. Initial failure logs and all original builds stay external; the retained
original fixture provenance includes source, client, model and library hashes.

`independent-review.json` pins the formatted implementation reviewed by a
separate agent. It records the resolved address-reuse issue: an adapter selected
by a context retains owned data after manual free, and a newly allocated handle
at the same address must replace that old data rather than take the no-change
fast path. The final check compares data identity and has a deterministic
regression test. No remaining blocking source-review findings were reported.

## Commands

The affected crate checks are:

```text
cargo fmt -p llama-ffi -p llama-model-runtime -- --check
cargo test --locked --offline -p llama-ffi -p llama-model-runtime
cargo clippy --locked --offline -p llama-ffi -p llama-model-runtime --all-targets -- -D warnings
cargo test --locked --offline --release -p llama-ffi -p llama-model-runtime
cargo build --locked --offline -p llama-ffi
cargo build --locked --offline --release -p llama-ffi
```

The external client uses the untouched frozen `llama.h` and dynamically loads
the selected library. The integration replay driver is external at
`C:/Users/adyba/rust-llama-reference/continue-20261111-next/context-oracle/replay.py`.
It accepts `--repo`, `--debug-library`, `--release-library` and `--output` paths.
On GNU, use `/usr/bin/python3` and the corresponding `/mnt/c/...` path. It writes
reports/logs to a new output directory and does not modify the tested repository.

The inherited model-metadata file retains its existing CRLF convention to avoid
an unrelated whole-file rewrite. Whitespace review therefore uses
`git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check`.
