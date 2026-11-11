# Backward RoPE tensor extension

Original operation ID 49 now executes through `RopeBack` with explicit
options for F32/F16 and all five frozen CPU rotary modes. The tensor inventory
is 94 executable IDs plus NONE, with seven pending IDs. Broader T02 and
full-port release gates remain open.

## Independent backward captures

The original generators called `ggml_rope_ext_back` and
`ggml_rope_multi_back`, then executed the actual frozen CPU graph before
this Rust implementation existed. Expected values are complete original
backward outputs, not a forward-rotation equivalence calculation.

| Corpus per host | Native graphs | Exact replay graphs | Exact output bytes | Native-success exclusions | Retained excluded bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main | 780 | 780 | 1,031,360 | 0 | 0 |
| Empty / IEEE / boundaries | 112 | 69 | 133,380 | 43 | 5,216 |
| Total | 892 | 849 | 1,164,740 | 43 | 5,216 |

The 43 excluded cases succeeded in the original CPU. Their complete raw
outputs remain unchanged, and Rust tests require explicit Unsupported
results for the stated finite parameter/input/coefficient/product boundary.
Every admitted byte compares exactly, including copied NaN payloads,
signed zeros and original stored infinities. No tolerance, blanket NaN
exception or Rust-generated expected output is introduced.

Coverage includes Normal, NeoX, MRoPE, vision and interleaved MRoPE; YaRN
correction/truncation, attention magnitudes, offsets and partial rotations;
source flat prefixes in strided positions/frequency factors; padded value
rows, grouped heads/batches, original one/three workers and empty dispatch.
Two graphs rotate every finite F16 input word. One graph copies all 65,536
half words outside its rotation. Inactive YaRN fields and unused frequency
cache entries have separate nonfinite controls.

MSVC and GNU main outputs differ and remain separate matching-profile
fixtures. All eight raw/provenance files are copied unchanged from the
independent originals. `source-and-fixture-verification.json` records exact
copy checks; `reference-artifacts.json` seals the external scripts, case
ledgers and raw captures. All native libraries and harnesses remain outside
the maintained product.

Baseline archive SHA-256:
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

The precise API, source arithmetic, physical spans, supported exceptional
values and resource limits are documented in
[ROPE_BACK.md](../../../crates/tensor/ROPE_BACK.md).

## Final validation

| Gate | Windows MSVC | GNU/Linux |
| --- | --- | --- |
| Complete tensor debug suite | 82 passed | 82 passed |
| Complete tensor release suite | 82 passed | 82 passed |
| All-target Clippy, warnings denied | Passed | Passed |
| Tensor rustfmt check | Passed | Passed |

Both final reports' 75 source/fixture hashes match the final files exactly.
`final` preserves raw logs and exact command vectors. The initial independent
native replay passed on both hosts. The first strict-Clippy run flagged the
source-style multiplication of sine by minus one; it was written as a unary
sign flip for the admitted finite coefficients. Both final profiles then
passed all numerical and checked tests. The initial successful replay and
the failed style gate remain in `development`; expected values were unchanged.

Five checked tests cover original ID/modes and optional-factor arity, dtype
and position groups, even dimensions/offsets and section bounds, signed I32
parameters, the source's full-width flat frequency read, physical scalar
alignment, small-view output amplification, cancellation and input immutability.

Output and cache payloads have separate 256 MiB limits and a checked scalar
work estimate. Allocation is fallible, every accessed span is bounded and
cancellation publishes no partial output. The adapter is serial. Other
ISA/compiler/CRT profiles, exception or FTZ/DAZ settings, general scheduling,
graph automatic differentiation and advanced forward RoPE remain open.

## Backward flash-attention source gap

Inspection of the next ID found that the frozen `ggml_flash_attn_back`
public constructor unconditionally aborts before constructing a tensor.
Both unmodified native profiles reproduced the same diagnostic with valid
F32 Q/K/V/d geometry. Windows exited with 3221226505 and GNU with SIGABRT (-6).
The raw outputs and provenance are in
`source-gap/flash-back-constructor/{msvc,gnu}`.

The dormant original CPU kernel does not supply public-API execution evidence.
ID 75 remains pending alongside four custom callback IDs and two optimizers.
[FLASH_BACK_STATUS.md](../../../crates/tensor/FLASH_BACK_STATUS.md) explains
this distinction. No disabled constructor was bypassed to claim completion.

GGML MIT attribution is preserved. The source's additional YaRN MIT copyright
for Jeffrey Quesnelle and Bowen Peng is retained with complete terms in
[ROPE_BACK_NOTICES.md](../../../crates/tensor/ROPE_BACK_NOTICES.md).

Maintained crate paths pass Git whitespace checking with CR-at-EOL accepted.
Raw evidence preserves original line endings and blank EOF lines so its
recorded hashes remain unchanged.
