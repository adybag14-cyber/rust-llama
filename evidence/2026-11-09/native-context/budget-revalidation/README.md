# Native cache full-batch budget revalidation

The final budget estimate now includes the complete incoming batch. Earlier microbatches remain in the staged cache while subsequent microbatches gather visible K/V rows, so charging only the first microbatch could undercount the peak.

The private lower-limit entry point uses the same production preflight, capped by the production two-GiB ceiling. The regression submits 512 valid tokens in 128 microbatches: the constrained budget rejects before mutation, and a sufficient budget executes all tokens. Reinstating just the old first-microbatch calculation makes this regression fail; the expected failed negative-control logs are retained.

Only context/native.rs changed in product. Quantization independently reviewed the bound and the minimal patch. The 116 source-file comparison against the candidate snapshot and before/after hashes are recorded. No row-math draft code is included.

Fresh Windows and GNU/Linux debug/release runtime and FFI tests passed, as did strict all-target Clippy. Windows has 64 focused tests per profile; Linux has 62. The original native C corpus was rerun in both profiles and on both platforms: 415,857 trained and 73,301 tiny float comparisons per run. Windows is bitwise exact. GNU retains the unchanged predeclared 2e-4 absolute plus 2e-5 relative bound, with the same prior maximum differences (5.7220458984375e-6 trained, 1.1920928955078125e-7 tiny). The no-evaluation native state restore/reserialize regression is exact on both platforms. Windows also reran the original core layout/model/context harness.

Prior whole-workspace evidence remains unchanged. This directory records the explicit incremental resource fix and its focused revalidation; CI evaluates the final pushed commit.
