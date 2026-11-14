# Context performance, warmup and runtime information

This native Rust surface implements `llama_perf_context`,
`llama_perf_context_print`, `llama_perf_context_reset`,
`llama_set_warmup` and `llama_print_system_info` for the admitted CPU Llama
decoder and BERT encoder contexts. It uses the same process monotonic clock as
`llama_time_us`. Timings measure this Rust implementation; they are never
copied from reference timings or presented as original-library throughput.

## ABI and reporting

The frozen `llama_perf_context_data` contains four doubles followed by five
32-bit integers. Independently compiled x64 MSVC/GNU callers measure size 56,
alignment 8, and offsets 0, 8, 16, 24, 32, 36, 40, 44 and 48. Rust uses the same
C layout. A null getter returns the all-zero structure, as the original does.

A live context's prompt and evaluation count getters clamp each raw count to a
minimum of one. Work is accounted when a synchronization boundary is reached:

| Boundary | Effect |
|---|---|
| Successful decode/encode submission | Queues the submitted token count and starts the elapsed interval if needed. |
| Explicit synchronize, output getters, state data transfers and session I/O | Completes pending accounting. |
| Performance getter or printer | Reads the existing counters without synchronizing. |
| State-size queries | Retains pending accounting. |
| BERT encode after a previous pooled sequence output | Completes the previous interval before clearing the pooled output. |
| Performance reset | Clears reported prompt/evaluation totals and refreshes start time, retaining pending work and its original start. |

An interval containing one queued token is evaluation work; an interval
containing more than one is prompt work. Thus two one-token decodes without an
intervening synchronization count as two prompt tokens, matching the source.
The CPU kernels themselves execute synchronously; only reporting remains
deferred. The elapsed interval includes time until the API synchronization
boundary, including intervening caller work, just as the source does.

`no_perf = true` disables prompt/evaluation elapsed accumulation, while token
counts still advance. Model load time is recorded from actual loading work.
At the first synchronization with queued work, it is updated from the model
start through that boundary. Later synchronizations retain it. Reset preserves
that load duration and whether first evaluation has occurred.

Input validation precedes queuing. A later computation cancellation or resource
failure may retain the submitted count, matching the original timing scope.
Checked arithmetic rejects counter/timer overflow instead of invoking the
original signed-overflow undefined behavior.

The direct Rust runtime does not build, reuse or resize GGML graphs.
Consequently `n_reused`, `n_reallocs_same` and `n_reallocs_diff` accurately
report zero. These values do not claim parity with the original graph scheduler.

## Printing and logger ownership

Printing emits six INFO messages through the existing logger: load, prompt,
evaluation and total time, followed by graph reuse and graph reallocation.
Field widths, floating-point division, labels and line endings match the
frozen source. On the measured MSVC/GNU profiles, bounded system-CRT formatting
preserves native rounding and NaN/infinity spelling, including null-context
printing. The platform CRT is the only foreign formatting dependency; no
original llama/ggml library is linked or loaded. On MSVC, snprintf uses the
toolchain's legacy_stdio_definitions compatibility library. Static C consumers
must include that library along with the usual Rust native static dependencies.

The independent corpus checks the default C locale. Total elapsed time remains
a live measurement and is checked structurally rather than against a fabricated
constant. Other targets use Rust formatting without a native formatting parity
claim. The callback lifetime and replacement requirements in PERFORMANCE.md
also apply here.

## Warmup and system information

`llama_set_warmup` stores the validated boolean and emits the frozen DEBUG
message. The original warmup flag changes expert routing in MoE graphs; it
does not change the admitted dense Llama or BERT graphs. Dense output values are
checked before and after warmup. This does not enable an unimplemented MoE path.

`llama_print_system_info` returns an immutable process-lifetime C string in the
source backend-feature grammar. It identifies the registered Rust CPU
implementation and its actual scalar, flash-attention and worker support.
It does not advertise an available GPU or CPU instruction set merely because
the host has one. The separate workspace GPU runtime is not registered as a
backend of this C facade.

## Independent evidence

Before the implementation was written, an external client compiled against the
unchanged frozen header captured six scenarios and 124 observations on each
original runtime. Scenarios cover Llama decoding and BERT with and without
pooled output, both no_perf settings, pending reset, explicit synchronization,
output/state boundaries, warmup and final numerical values. A second capture
includes native null-print formatting. The semantic projections agree across
hosts; elapsed values remain external observations rather than expected Rust
numbers.

The ordinary tests use only committed data in `fixtures/ffi/context-perf`.
They check exact counters/status/logits, ABI layout, print bytes, positive real
timing and the source reset relations without linking an oracle. Fixture
provenance identifies source, original libraries, client and input hashes;
`docs/context-perf-fixture-manifest.tsv` inventories the shipped corpus.
The frozen source archive SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

Final Windows/GNU debug/release tests, strict Clippy, native-header client
replays and exact PE/ELF export inventories are retained in
`evidence/2026-11-13/c-context-file-api`.
