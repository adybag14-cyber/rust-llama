# Gated delta and DeepSeek V4 tensor extension

Four additional original IDs (86, 88, 89 and 90) now execute through five
constructors in safe Rust. The inventory is 92 executable IDs plus NONE,
with nine pending IDs. The full T02 and full-port release gates remain open.

The original native capture was sealed before this implementation. Its 867
graphs per host contain 5,185,540 output bytes. A separate boundary capture
contains 89 native-success graphs and 40,244 output bytes. Sixteen of those
graphs execute in Rust with 32,096 exact bytes; 73 preserve successful native
nonfinite/overflow results for an explicit Rust Unsupported boundary.
All 8,148 output bytes of the rejected original graphs remain untouched.
These are source-success exclusions, not original failures.

In total, 883 graphs and 5,217,636 output bytes per host compare exactly.
No numerical tolerance, blanket NaN exemption or replacement expected value
is introduced. All inputs, parameters, shapes, byte strides, native worker
counts and expected output descriptors remain in the raw fixtures.
See [ATTENTION_STATE.md](../../../crates/tensor/ATTENTION_STATE.md) for the
precise API, state/snapshot order, cyclic heads, grouped sequences, physical
strides, iteration arithmetic, finite boundary and resource limits.

## Independent source and provenance

Baseline archive SHA-256:
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

The original source files and native-library hashes are in each sealed
provenance file. `reference-artifacts.json` records the shared primary capture
script, both attention generators, native case ledgers and raw output hashes.
All generators and native C/C++/library artifacts remain outside the product.
No Rust implementation is imported or executed by the generators. The
original raw and provenance files are copied without any byte changes.

## Validation

| Gate | Windows MSVC | GNU/Linux |
| --- | --- | --- |
| Complete tensor debug suite | 68 passed | 68 passed |
| Complete tensor release suite | 68 passed | 68 passed |
| All-target Clippy with warnings denied | Passed | Passed |
| Tensor rustfmt check | Passed | Passed |

The final reports' tested Rust source and fixture hashes match this commit's
files exactly. `final` retains raw stdout/stderr and the exact command vectors.
The initial focused native replay passed on both hosts. The first complete
Windows run also passed debug/release but exposed a new Clippy style lint in
test mismatch diagnostics. It was corrected with `as_chunks::<4>()`, without
changing test criteria, fixtures or implementation arithmetic. Both final
profiles then passed all four gates. Earlier logs remain in `development`.

Native one/three-worker controls have disjoint output work; this Rust adapter
remains serial. Other ISA/compiler arithmetic, device backends, exception flags,
general scheduling and complete model execution remain open.

## Allocation and span follow-up

The same change adds complete flat-address preflight before window, relative-
position and loss work. Unpartition's maximum read is computed across both the
last partial block and the previous complete block, preserving valid
nonmonotonic native addresses. This prevents large allocations before an
inevitable out-of-view read.

Window/relative and new attention outputs have an explicit 256 MiB payload
limit. Scalar loss work has an explicit 64-million-cell limit. A checked test
constructs tiny overlapping source storage that would otherwise request a
multi-GiB snapshot output and verifies rejection before allocation/work.
Every original vision/loss fixture still passes unchanged.

Maintained crate paths pass Git whitespace checking with CR-at-EOL accepted.
Raw evidence retains original Windows line endings and blank EOF lines;
those exact hashed bytes are intentionally preserved.

The implementation and native-derived fixtures preserve the GGML MIT
attribution in root `NOTICE` and `LICENSE-UPSTREAM`.
