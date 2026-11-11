# Seven vision, relative-position, solve and loss operators

This extension adds safe Rust DAG implementations of original IDs 78, 79, 80,
81, 85, 97 and 98. The tensor inventory advances from 81 to 88 executable IDs,
plus NONE input leaves; 13 IDs and the wider T02 gate remain open.
See [the operator contract](../../../crates/tensor/VISION_LOSS.md).

## Independent expected results

The frozen archive SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Original constructors and kernels were checked against the user-pinned source.
Reference programs, native builds and the native case ledgers remain under the
external `continue-20261111/tensor-oracle` directory.

| Cohort | Graphs per host | Output bytes per host | Purpose |
| --- | ---: | ---: | --- |
| Primary | 920 | 1,135,798 | Shapes, strides, every half/bfloat word, native reduction order |
| IEEE | 111 | 25,992 | Subnormals, zeros, infinities, quiet/signaling and competing NaNs |
| Geometry and operands | 222 | 83,424 | Relative-position and solve sizes, batches and NaN placement |
| One-key widths | 168 | 26,208 | GNU specialized loop boundaries and signaling NaN selection |
| Total | 1,421 | 1,271,422 | Exact bytes on matching MSVC and GNU hosts |

Every captured graph is admitted by Rust and compared without tolerance, NaN
exemptions or regenerated expectations. Native output descriptors, input bytes,
parameters and thread counts are preserved. Inputs remain immutable and output
storage never aliases input storage.

The primary capture preceded the implementation. The later independent cohorts
preceded the corresponding arithmetic corrections and boundary tests. They call
only the original C graph API and never import or run Rust. The original sealed
provenance JSON and binary captures remain unchanged. Some later provenance
records inherit the primary generator's general descriptive text; their own
case counts, operation inventory, exact file hashes and the chronology above
identify their narrower actual scope. `reference-artifacts.json` also pins the
shared generator on which the later capture scripts depend.

## Native compiler policy

Initial IEEE replay exposed six competing-NaN payload differences on each host.
The original expected bytes and failed logs are in `development/ieee-before`.
Explicit quieting and operand selection now match both native profiles.

The larger geometry cohort exposed four further GNU cases with key width one
and multiple query columns. The frozen GNU library's specialized non-aliasing
loop loads height before the accumulator for its first addition; its width-one
fallback and general-key loop load the accumulator first. Both SIMD lanes and
scalar tail use the specialized order when query width exceeds one. The final
168-case cohort exercises query widths 1â€“17, 24, 31, 32 and 33, one/two patches
and four signaling-NaN operand masks. The unchanged failed replay is in
`development/policy-before`; the external disassembly's hash is retained.

## Verification and limitations

| Gate | Windows MSVC | GNU/Linux |
| --- | --- | --- |
| Complete tensor debug tests | 59 passed | 59 passed |
| Complete tensor release tests | 59 passed | 59 passed |
| Strict all-target Clippy | Passed | Passed |
| Tensor rustfmt check | Passed | Passed |

`final` contains raw reports/stdout/stderr and all tested source/fixture hashes.
The original complete corpus and every existing tensor test still run.
Three maintained Rust files were subsequently normalized from CRLF to LF only
to keep the patch additive. `post-validation-normalization.json` proves the
exact byte-only transformation; no Rust tokens or fixture/provenance bytes
changed. A subsequent rustfmt check covers these final line endings.
Git whitespace checking uses `core.whitespace=cr-at-eol` because independently
hashed native evidence deliberately retains Windows line endings. The maintained
crate paths pass this check. Raw runner stdout additionally retains its original
blank lines at EOF; the all-path check reports those evidence-only bytes, which
are deliberately not edited to satisfy whitespace style.

Explicit boundaries include positive signed-I32 geometry, supported native
dtypes/layouts, declared-view read bounds, nonzero solve diagonals and finite
cross-entropy inputs/gradient. Forward loss preserves one native worker's
reduction order. Competing-NaN policies on other targets remain unsupported.
Floating exception flags, other ISA/compiler math, general scheduling,
autodifferentiation, optimizer updates and complete pretrained-model execution
are not established by this extension.

The implementation and reference-derived fixtures preserve the GGML MIT
attribution in root `NOTICE` and `LICENSE-UPSTREAM`. No C/C++ source or native
binary is added to the product.
