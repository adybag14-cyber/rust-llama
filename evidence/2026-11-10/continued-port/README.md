# Integrated continuation validation — 10 November 2026

The integrated Rust implementation passed the checks below on Windows MSVC and Linux GNU. The full frozen llama.cpp port is still incomplete: 20 mandatory full-port gates remain open.

| Check | Windows MSVC | Linux GNU |
| --- | --- | --- |
| Whole workspace, debug | 592 passed; 0 failed; 22 explicitly ignored external/device tests | 597 passed; 0 failed; 22 explicitly ignored external/device tests |
| Whole workspace, release | 592 passed; 0 failed; 22 explicitly ignored external/device tests | 597 passed; 0 failed; 22 explicitly ignored external/device tests |
| Formatting and strict Clippy | Passed | Passed |
| Source/dependency audit and release build | Passed | Passed |
| Fixture integrity | 720 files, 720 declarations, nine manifests | Same |
| Supplied frozen source identity | All 4,082 files matched | Same |
| Explicit pretrained inference/server tests | Nine passed | Nine passed |
| Explicit pretrained converter test | Three complete outputs, 686,581,536 exact bytes | Same |
| Original-header C clients, debug and release | 35,946 chat cases, 171 metadata requests and 38 raw-name models per build | Same |
| Complete exported symbol inventory, debug and release | 198 original C exports, four extensions; 57 original C exports absent | Same |
| Full-port release check | Expected failure: 20 open gates | Same |

The source/fixture/configuration manifest contains 1,047 files and has SHA-256
`1dfe31c3c3bd33924206b69111bfc60c7460f3f4a84f7c7540525ec0cf2ce1a6`.
Both hosts tested exactly those inputs and verified they remained unchanged.
The report's git_head is the pre-integration parent commit, not a claim that the
parent already contained the changes. The manifest identifies the tested worktree.

The metadata and chat clients compile against the supplied original headers and
load the actual debug/release Rust libraries. The full chat capture is identical
to the independent original output. Each complete metadata JSON result matches
its source-profile capture. Export inventories inspect actual PE/ELF symbols.
The raw-name clients compare all captured token attributes, tokenizations,
pieces, metadata and buffers for 38 independent models.

The ordinary suite's ignored entries retain their explicit external-data or
device prerequisites. Nine pretrained inference/server tests and one complete
converter test are executed separately here. This does not certify every
ignored hardware or full-vocabulary domain.

## Retained attempts and fixes

The initial run was cancelled for a changed line ending. The next full debug
run exposed a changed Mamba loader error label; raw lookup retained native bytes,
but its missing-key diagnostic had changed. The textual label was restored and
the entire debug/release suites passed on both hosts.

The following audit rejected sixteen retained evidence files solely because
their extensions were .stdout/.stderr. Their bytes were preserved under recognized
.log names. The final run checked that all execution inputs and previous passing
compiler/test logs were byte-identical before resuming the remaining gates.
Earlier reports and failure logs are retained under attempts/.

No test tolerance, expected native output or full-port requirement was relaxed.
The recurrent component still explicitly rejects five original-success graphs
with competing NaN payloads; its documented limit remains open.

## Evidence layout

Each host directory contains the original report, exact execution-input manifest,
raw stage logs, actual debug/release export inventories and raw-name replay
status. summary.json gives the combined result. retained-files.json maps each
original external evidence path to its byte-identical retained file.

Reference generators, foreign source, compiled libraries, pretrained weights and
generated pretrained model outputs remain outside this repository. Component
captures and contracts are retained separately in the metadata, chat, raw-name,
converter and recurrent-tensor evidence directories.
