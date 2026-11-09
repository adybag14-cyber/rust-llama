# Extended scalar tensor execution

The tensor crate now executes 53 named operation IDs plus NONE input leaves.
The registry still has 48 operation IDs without an executable path. The
CPU_SUPPORT.md table specifies dtype/layout/parameter boundaries.

The 297 new original GGML graph cases supplement the 65 existing cases.
They check 256,233 exact outputs and 170,260 transcendental outputs, including
exhaustive finite F16 GELU/GELU_QUICK and BF16 GELU tables. A separate 80-digit
independent reference checks 96,910 signed ERF inputs for correctly rounded F32.

The extended implementation covers all 22 scalar unary formulas and all seven
GLU formulas for the source-supported typed families, all 29 CPU matrix-weight
types and all 28 source out-product types. F32 row-grouped SUM and strided
normalization were corrected against independent source outputs.

| Observation against frozen Windows oracle | Windows MSVC | WSL Ubuntu |
| --- | ---: | ---: |
| Exact checked outputs | 256,233 | 256,233 |
| High-precision ERF checks | 96,910 | 96,910 |
| Non-identical transcendental bits | 1,815 | 1,838 |
| Largest F32 absolute difference | 4.76837158203125e-7 | 1.9073486328125e-6 |
| Largest all-dtype absolute difference | 0.0009765625 | 0.0009765625 |
| Comparisons needing final F16 rounding envelope | 4 | 4 |

The portable budget includes the fixed F32 numerical bound and explicitly
bounded final F16/BF16 conversion spacing. Exact UCRT erff parity remains open.
The original failed threshold checks and original expected data are retained
in the external transcendental-review record; expected outputs were not changed.
Linux used the same source and bounds without a Linux-specific adjustment.

Both platforms pass all 14 focused tensor tests in debug and release and strict
Clippy. Windows formatting also passed. validation-report.json records exact
source/fixture hashes and job identities; validation/ contains the complete
focused outputs and execution requests. The full workspace checkpoint
verification is a separate gate.
