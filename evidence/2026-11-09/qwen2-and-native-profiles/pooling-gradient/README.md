# Overlapping pooling-gradient operand order

The original 961 convolution cases did not contain distinct NaN payloads in overlapping backward gradients. Twenty new actual GGML graphs on each frozen host retain six overlapping raw F32 gradients, including quiet/signaling NaNs, opposite infinities and signed zero. Max pooling directs every window into the same forward maximum; average pooling also exercises partially overlapping cells.

The original Windows results matched the former Rust debug implementation. Original GNU average-backward retained an earlier accumulator NaN, while Rust selected later gradients: ten cases and66 output words differed. Both original max-backward kernels and Windows average-backward select the incoming gradient payload. The correction spells out this source-profile operand policy using the existing ordered-add helper, leaving ordinary finite arithmetic unchanged.

The new ordinary regression compares every byte with each host's unchanged native file. Both hosts pass the complete existing tensor suite including all961 previous convolution/extraction/pooling/resize cases, plus20 new controls, in debug and release with strict Clippy and formatting. The pure numerical source handoff is independent of the pending shared-row/OS-erf work.

The first test adapter accidentally reversed PoolMode0/1; its source and failures are retained separately. The corrected pre-fix GNU mismatch is also retained. No raw reference value or tolerance was changed.

Exact source/fixture hashes and the original cpu_conv.rs preimage are in manifest.json. The patch applies only to cpu_conv.rs; copy the new regression/fixtures by their manifest paths. Compiler/library/archive provenance is retained in each native provenance file. This measured operand policy is qualified to the two exercised x86_64 MSVC/GNU builds; exceptional-value bit parity on other toolchains remains unverified.
