# Shared row mathematics and the system ERF boundary

This handoff is frozen after independent Windows/MSVC and GNU/Linux validation.
It contains 31 product file candidates, including 16 new numerical fixture and
provenance files. The manifest records an exact preimage for each replacement
and an exact hash for every delivered file. No product or combined-draft file
was changed during preparation.

## Implementation

- One maintained safe `llama_quant::cpu_math` module serves tensor and inference.
  Its first 7,029 bytes are exactly the previously validated model-runtime row
  implementation, SHA-256
  `11952fcd2a620c2dd6402c7493a3d591395fd4977f4b6e58ee38df5c4d3133cc`.
  Model-runtime preserves its public `math::native_rows` path as a re-export.
- Whole F32 tensor rows use the source profile for SiLU, quick GELU and their
  GLUs, softmax exponentials/sums, and contiguous NORM variance. Scalar tails
  and F16/BF16 control paths keep their distinct source arithmetic.
- Odd packed GLU rows use floor-sized halves and ignore the last scalar.
  Measured NaN operand selection is explicit; finite arithmetic is unchanged.
- A small platform-loader boundary calls the OS `float erff(float)` on the
  two exercised dynamic-CRT profiles. Tensor retains its independently checked
  pure Rust mathematical implementation for other/static targets.
- Quant remains dependency-free. Tensor gains only a target-specific dependency
  on the already existing local platform-loader crate. There is no project C,
  vendor math, C shim, custom build script or new external Cargo dependency.

This slice improves existing operations. It does not add a new operation ID or
claim complete tensor/backend parity.

## Independent numerical proof

| Gate | Per matching host | Result |
| --- | ---: | --- |
| Actual original GGML row graphs | 935 cases; 2,894,848 output bytes | Exact bytes |
| Actual original odd packed GLU graphs | 72 cases; 14,208 output bytes | Exact bytes |
| Original scalar quick-GELU inline function | 502,058 raw F32 input/output words | Exact words |
| Independent call to OS erff | 502,058 raw F32 input/output words | Exact words |
| Retained 80-digit mathematical ERF oracle | 96,910 signed accuracy checks | Existing independent gate passes |

The new matching-host graph gate compares all output bytes, including NaN
payloads, without a tolerance. The old 297-case cross-platform numerical fixture
and its original criteria are unchanged. Its legacy diagnostic still describes
the historical portable ERF limitation; the new matching-provider gate is
separate and documented explicitly.

The scalar quick-GELU expected words were captured from each original host
after an explicit GNU-to-UCRT control failed by one ULP. That failed assertion,
the unchanged candidate at that stage, and both new same-host captures are
retained under the evidence and fixture directories. MathPolicy chooses source
grouping and vector-prefix behavior; scalar operations still use the current
host CRT. It does not simulate a foreign CRT.

## Exercised providers and actual Rust linkage

| Profile | Captured provider | Provider SHA-256 |
| --- | --- | --- |
| x86_64 Windows MSVC, dynamic CRT | C:\Windows\System32\ucrtbase.dll, 10.0.26100.9711 | dedb7611e1abc1ff3852dcac3d4c93a048d729e25b061973b9969f22698f14ca |
| x86_64 Linux GNU, dynamic CRT | /usr/lib/x86_64-linux-gnu/libm.so.6, glibc 2.35 | 3dd5511ae94785c9f921429b0f2b2f7aabb461b6f0e6de6dfdbef15f24bdfee6 |

The actual Rust release test PE imports erff from
`api-ms-win-crt-math-l1-1-0.dll`. The actual ELF imports
`erff@GLIBC_2.2.5` and has a `libm.so.6` dependency. Neither imports any
llama/ggml/oracle binary. Complete import tables, binary hashes and inspection
commands are under `evidence/linkage`.

These are tested provider versions, not a promise for every future UCRT/glibc
revision. CI runners with different CRT versions require an explicit
compatibility decision; old expected words must not silently be regenerated
from Rust. Other/static targets decline this provider and use the documented
mathematical fallback. The numerical environment is nearest-even rounding and
gradual underflow, with FTZ/DAZ disabled. Floating exception status flags are
outside the contract.

## Validation and integration

Job `job-1794245196435-daf16f16-466` ran the complete isolated four-crate
workspace through formatting, debug, release and strict all-target Clippy.
Windows passed 125 tests per profile and GNU passed 126, with zero failed or
ignored tests. All five numerical implementation hashes still match the final
reports. Later changes were documentation only.

Apply `shared-row-math.patch` / `files` with the guards in `manifest.json`.
The two tensor docs were rebased onto the root's accepted backward-pooling
documentation and preserve those additions. This handoff contains no
`cpu_conv.rs` replacement or backward-pooling fixture, so it cannot overwrite
the separately accepted numerical fix.

`preimages/Cargo.toml`, `preimages/Cargo.lock` and
`root-cargo-preimages.json` retain the root Cargo inputs. Never copy the
reduced isolated workspace manifests into the product. See
`cargo-integration.md` for the one local dependency and resulting Cargo.lock
edge. `dependency-inventory-delta.tsv` records the OS boundary.

`fixture-declarations.json` and its TSV form contain all 16 exact fixture
identities for the root-owned combined manifest. The old source fixtures remain
unchanged. `preserved-failures.json` inventories initial numeric, NaN-payload,
test-adapter, odd-width and foreign-CRT failures; available text logs/sources are
copied under `evidence/preserved-failures`.

Full combined-workspace and CI checks remain the integrator's next gate. This
handoff itself makes no claim that they have run.
