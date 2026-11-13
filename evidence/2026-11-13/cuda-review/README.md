# Independent CUDA review and numerical failure fixes

This evidence covers the focused review of the new persistent dense-Llama CUDA runtime. It does not close full backend, architecture, device, or pretrained-model parity gates.

## Findings and fixes

The first Q8 activation rounding implementation added 0.5 before truncation. Native UCRT `roundf`, used by the frozen GGML Q8 reference, returns zero for the binary32 value with bits `0x3effffff` and negative zero for `0xbeffffff`. The original GPU arithmetic instead returned 1 and -1 because its intermediate addition rounded to a tie. The parent replaced that arithmetic with integral/fractional comparison and added adjacent-value/tie tests. The independent capture records the exact input block, CRT DLL hash, and frozen source hash.

A second defect allowed numerical failure to become successful output. Two structurally valid models with finite weights reproduced it on the actual GPU. One overflowed RMS squares and obtained a zero normalization scale. The other overflowed a query projection, produced NaN attention, then erased that failure through a zero quantized output projection. The CPU runtime rejected both. Before the fix, CUDA returned success and advanced the committed position from 1 to 2.

The typed instruction tree now contains an explicitly validated finite predicate, emitted as PTX `testp.finite.f32` and interpreted with the same finite classification. This predicate accepts finite normal/subnormal values and signed zero, and rejects NaN and both infinities. The instruction is available within the emitter's existing target range; its semantics are documented in the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-testp).

Invalid RMS scales and activation blocks now produce NaN poison. Overflowing/nonfinite decoder outputs are normalized to NaN so they cannot be mistaken for attention masks. Softmax checks every visible score before a max operation can discard NaN. Invalid Q8 blocks overwrite all 33 output words, and packed dots preserve the poison even when weights are zero. The runtime's final output check then returns an error before committing the token. Failed cache rows remain outside the committed prefix and are overwritten on retry.

The review also identified a default-budget integer overflow on 32-bit targets and late validation of decoded embedding extents. The parent fixed both in the runtime admission code.

## Focused verification

Windows Rust 1.99.1 passed 24 ordinary device-family tests and both explicit hardware suites. The live kernel suite exercised 135 cases twice, including exact Q8 activation output words, adjacent half-integer boundaries, zero/subnormal scales, nonfinite/overflow cases, and propagation through zero packed weights. Seven kernels also received malformed device uniforms; none changed the buffers. Foreign stream/input/output ownership and short input buffers were rejected, and kernels remained usable after their original context handle was dropped.

The tiny Llama graph has grouped-query attention and every optional projection bias. Its four-token replay matched 44 values from the existing independent original-GGML graph fixture with maximum absolute error `1.4901161e-7`. The comparison bound is `2e-5 + 2e-5 * abs(expected)`. Exactly 3,812 requested CUDA-buffer bytes admitted this graph; 3,811 bytes were rejected. This is requested buffer length accounting, not a total VRAM ceiling.

Both previously failing models now return errors without advancing position 1 or the committed-token count. A subsequent valid token matches a clean recomputed prefix exactly.

The initial run passed all ordinary and live tests, then stopped on Clippy's test-only `chunks_exact` style warning. The one-line replacement with `as_chunks` did not alter kernel code. The Clippy continuation passed with `-D warnings`. Both the original run and continuation are retained, and `report.json` records the source hashes and this distinction.

## Files and limits

`activation-rounding-counterexample.json` contains the independent CRT observation. `before-rms-rejection.json` and `before-activation-poison.json` retain actual failing GPU runs. The validation logs contain passing ordinary, analytical/live-device, model/error, and budget checks. `validation-initial.json` preserves the original failure at Clippy, while `clippy-continuation.json` records the corrected gate.

Reference scripts and pre-fix source snapshots remain outside the product at `C:\\Users\\adyba\\rust-llama-reference\\continue-20261113\\gpu\\review`. The graph fixture is structural evidence; pretrained F32/Q4/F16 and final combined integration checks are recorded separately by the parent task.

## Shared generation validation order

The independent review compared the refactor with `f6d238c`. Sampler configuration, prompt acceptance, stop strings, UTF-8 flushing, token counting, and context-limit arithmetic were preserved. A concrete entry-order regression remained: constructing the CPU context before shared validation caused an immediately cancelled BERT generation request to return `Unsupported` without checking cancellation.

The private CPU session now borrows the model and creates its context only at the first evaluation. Shared validation runs first, preserving cancellation/architecture/option precedence and the zero-generation-limit return. The failing pre-fix regression is retained in `before-session-order.json`; the targeted follow-up is recorded in `session-report.json`. Its tests cover the original BERT error order, zero-token usage without emitted output, and complete native-derived Qwen2 generation/EOS behavior. No shared sampling or streaming algorithm was duplicated.
