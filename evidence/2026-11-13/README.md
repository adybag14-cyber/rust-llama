# 13 November 2026 implementation and verification

This checkpoint adds native dense Llama CUDA execution, Falcon CPU inference and
HF conversion, and seven frozen context/FILE APIs. The full port remains
incomplete: all twenty mandatory completion gates are open.

## Results

| Check | Windows MSVC | Linux GNU / WSL |
| --- | ---: | ---: |
| Ordinary debug tests | 753 passed, 40 ignored | 757 passed, 40 ignored |
| Ordinary release tests | 753 passed, 40 ignored | 757 passed, 40 ignored |
| Explicit pretrained inference/server tests | 13 passed | 13 passed |
| Explicit native BERT encoder test | 1 passed | 1 passed |
| Current frozen C exports | 239 plus 4 extensions | 239 plus 4 extensions |
| Fixture declarations | 1,094 in 24 manifests | 1,094 in 24 manifests |
| Complete frozen source trees checked | 2 × 4,082 files | 2 × 4,082 files |

Each ordinary run has 142 test-result suites and zero failed tests. Ignored
prerequisites are not counted as passes. Original-header C clients cover both
debug and release libraries on each host. Sixteen frozen exports remain absent.

Four actual CUDA host/profile replays each compare 762,880 original native logits
and 208 exact greedy IDs across three model files. Windows and WSL share one
physical RTX 4090; these are host/profile checks, not four independent devices.
The mixed Q4_0/Q8_0 model is bit exact. F32/F16 retain the existing numerical gate.
The separate CLI replay covers 128 actual commands.

## Evidence index

| Package | Purpose |
| --- | --- |
| [Combined runtime integration](integrated-port/README.md) | Both complete stage chains, original failed reports, exact library/input continuity and selected raw receipts |
| [Final audits](final-audits/README.md) | Source/dependency/fixture audits, frozen-tree verification, expected full-release refusal and the sole dependency-documentation delta |
| [CUDA runtime](cuda-runtime/README.md) | Four final GPU profiles, legacy kernels, provenance and retained earlier failures |
| [CUDA CLI](cuda-cli/README.md) | Text, stops, numerical outputs, device/budget statistics and explicit errors |
| [CUDA review](cuda-review/README.md) | Independent rounding, non-finite propagation and shared-generation corrections |
| [Falcon runtime](falcon-inference/README.md) | Original native graph/loader/state corpus; no pretrained Falcon acceptance claim |
| [Falcon conversion](falcon-conversion/README.md) | Exact complete conversions, original rejection behavior and exhaustive tokenizer stages |
| [Context and FILE APIs](c-context-file-api/README.md) | Original-header performance/warmup/system-info/owned-file clients and exact export inventories |

The runtime stage chain and package guard preserve a 1,623-file selected input
manifest. Afterwards, only dependency-boundary documentation changed within that
set; the final audit records its exact old/new hashes and verifies every other
selected input. Markdown, progress gates, evidence and the derived non-Rust
allowlist have their separately documented sealing boundary.

Reference C/C++/Python implementations, downloaded pretrained weights and large
raw client outputs stay external with recorded identities. The repository
contains the Rust implementation, independent fixtures, provenance, contracts
and selected verification receipts. The source audit's unsafe-location inventory
is a review aid, not blanket proof of memory safety.

See [current status](../../docs/PORT_STATUS.md) and
[completion gates](../../release/gates.tsv) for the remaining architecture,
backend, application, platform, performance and distribution work.
