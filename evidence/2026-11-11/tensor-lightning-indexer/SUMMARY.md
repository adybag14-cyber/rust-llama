# Lightning Indexer tensor extension

Operation 87 now executes through `Op::LightningIndexer` in safe Rust. The
inventory is 93 executable IDs plus NONE, with eight pending IDs. All broader
T02, complete-model and full-port release gates remain open.

## Independent original outputs

The native captures were sealed before the implementation. Every configuration
runs two actual original CPU graphs: its supplied F16 mask and a zero-mask
control recording the pre-mask score. The native control determines admission
for nonfinite-score cases. No Rust output supplies an expected value.

| Corpus per host | Configurations | Original graphs | Exact replay graphs | Exact bytes | Rejected native-success graphs | Retained excluded bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 652 | 1,304 | 1,304 | 2,802,864 | 0 | 0 |
| Empty / IEEE edges | 51 | 102 | 78 | 2,784 | 24 | 1,152 |
| Total | 703 | 1,406 | 1,382 | 2,805,648 | 24 | 1,152 |

Both Windows MSVC and GNU/Linux require every admitted output byte to match,
including signed zero, all half-mask NaN payloads and subnormals. The 24
excluded graphs succeeded in the original CPU and retain all 1,152 output
bytes. Rust returns Unsupported for their nonfinite pre-mask scores; the
exclusions are not reported as native failures.

Coverage includes all 29 native key storage types, original quantizer-encoded
blocks, padded outer rows, head/token/key/stream dimensions, modulo mask
streams, all 65,536 F16 masks at three different scores, all F16/BF16 key words
except positive infinity, all 256 F8 words, empty dispatch, zero heads and
zero embedding, and original one/three-worker controls. Independent original
graphs cover the exceptional-value boundary before Rust implementation.

The two separately captured profile binaries compare byte-for-byte equal.
Each binary is stored once with both unchanged native provenance files.
`source-and-fixture-verification.json` records direct equality checks against
both originals. There is no expected-value merge, numeric normalization or
replacement. `reference-artifacts.json` records hashes for the standalone
capture/validation scripts, both profiles' raw binaries, provenance and case
ledgers. Native C/C++ libraries, harnesses and build artifacts remain outside
the maintained product.

Baseline archive SHA-256:
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

See [LIGHTNING_INDEXER.md](../../../crates/tensor/LIGHTNING_INDEXER.md) for the
exact input order, types, source F32-product/F64-dot arithmetic, ReLU comparison,
flat rows, mask NaN words, original empty behavior and supported domains.

## Validation

| Gate | Windows MSVC | GNU/Linux |
| --- | --- | --- |
| Complete tensor debug suite | 75 passed | 75 passed |
| Complete tensor release suite | 75 passed | 75 passed |
| All-target Clippy with warnings denied | Passed | Passed |
| Tensor rustfmt check | Passed | Passed |

Both final reports verify the same 64 source/fixture hashes against the final
files. `final` preserves exact command vectors and raw stdout/stderr; the
initial two-host native replay is retained in `development/first-native-replay`.
The initial focused report's step is named "attention" by the reused external
runner, but its exact argv selects `--test lightning_indexer`.

Five checked tests cover ID/arity, seven missing native key decoders, incompatible
groups and dtypes, strict inner strides even on unit axes, scalar alignment,
and output/work amplification through overlapping immutable views. They also
check supported nonfinite masks, native NaN-suppressed dots, explicit overflow
rejection and input immutability.

Output payload and decoded-row scratch each have a 256 MiB limit. A checked
268,435,456-unit scalar-work estimate bounds key decoding, dots and weighted
reductions before allocation/work. Cancellation is checked at token/key work.
These local controls do not constitute a graph-wide memory planner.

The Rust adapter remains serial. Other ISA/compiler/CRT profiles, floating-point
exception or FTZ/DAZ settings, general graph scheduling, writable/device storage
and complete model inference parity remain open. The eight pending operators
are ROPE_BACK, FLASH_ATTN_BACK, four custom callback IDs, ADAMW and SGD.

Maintained crate paths pass Git whitespace checking with CR-at-EOL accepted.
Raw evidence keeps original Windows line endings and blank EOF lines because
its exact hashed bytes are intentionally preserved. The implementation and
native-derived fixtures preserve the GGML MIT attribution in root `NOTICE`
and `LICENSE-UPSTREAM`.
