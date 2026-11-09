# Tensor indexing and updates

The frozen original graphs independently exercise 890 new cases per platform,
including indexed addition/matmul, row writes, sorting/top-k, casts and overlapping
accumulation/set operations. All 599,152 checked output bytes match on Windows
and Linux. The source's 88 MSVC/GNU argsort differences retain distinct immutable
fixtures and precise toolchain provenance.

Together with the prior 65 graphs and the 297 scalar extension cases, this is
1,252 distinct graph cases (2,142 raw captures). The evaluator implements 62
operation IDs plus NONE input leaves; 39 operation IDs and additional modes remain
open. Original failure logs and platform numerical qualifications are retained.

The validation report records both product debug/release tests, strict Clippy and
format checks. The next whole-workspace report is a separate checkpoint.
