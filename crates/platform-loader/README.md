# Standard OS integration boundaries

This crate contains the narrow Rust boundaries for native driver loading and
system scalar math. It does not load llama/ggml or compile any project C/C++.

## System scalar ERF

The frozen CPU GELU_ERF uses the operating system's C-runtime
`float erff(float)`. `math::erf_f32` exposes the same function through a safe
`Option<f32>` API. The only unsafe operation is an audited call with one F32
argument and one F32 result; no pointers, ownership, callbacks or allocation
cross this boundary.

| Build target | Linkage | Independently exercised runtime |
| --- | --- | --- |
| x86_64 Windows MSVC, dynamic CRT | Standard `ucrt` import library | `ucrtbase.dll` 10.0.26100.9711 |
| x86_64 Linux GNU, dynamic CRT | Standard `-lm` system import | glibc libm 2.35 |
| Other targets or static CRT | No ERF import from this module | Returns `None`; caller chooses its explicit fallback |

Target selection identifies a linkage profile, not a guarantee for every future
version of its runtime. `math::erf_provider` reports the selected provider.
Tensor execution uses the existing independently bounded pure Rust
mathematical ERF when this module declines a provider.

The two original operating-system captures each contain **502,058 raw F32
inputs and outputs**, including all half/bfloat encodings represented in F32,
dense finite points, seeded raw bits, signed zero, infinities and NaN payloads.
Both debug and release regression tests require exact output words against
the matching host's independently captured reference. The fixtures and their
provider/source hashes are under `tests/fixtures`. An additional original
GGML graph corpus checks the enclosing GELU_ERF and GEGLU_ERF computations.

The numerical environment is nearest-even rounding with gradual underflow.
This API does not change the floating environment or promise floating exception
status-flag parity. Cross-CRT equality, unmeasured runtime versions and other
target profiles remain separate verification requirements.

This addition introduces no third-party Cargo dependency, build script, C shim,
vendor math library, or original-project binary into the Rust product.
