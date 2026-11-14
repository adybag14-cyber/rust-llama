# Sampler performance and logging

The native Rust C facade implements llama_perf_sampler,
llama_perf_sampler_reset, llama_perf_sampler_print, llama_log_get and
llama_log_set. These exports are backed by actual sampler-chain work and global
callback state. Context/graph performance and backend-attached sampling remain
unimplemented.

## Counts and elapsed time

A chain with no_perf=false measures its CPU apply and accept scopes using the
same monotonic microsecond clock as llama_time_us. The getter converts the
accumulated integer microseconds to milliseconds. Callback execution is part of
that elapsed time. Nested chains accumulate inclusive time independently; these
per-chain totals must not be added as if they were disjoint wall-clock intervals.
This feature does not claim native throughput or hardware equivalence.

Every completed accept call increments that chain's n_sample, including with
no_perf=true and for empty chains. Applying candidates alone does not increment
it. llama_sampler_sample applies and accepts, so each successful sample adds one.
Disabling measurement leaves elapsed time at zero until state is copied from a
measured chain; it does not suppress the count.

The frozen state transitions are preserved:

| Operation | Timing/count effect | Child state |
|---|---|---|
| llama_sampler_reset | Retains every chain's performance fields | Resets sampler state recursively |
| llama_perf_sampler_reset | Clears only the selected chain's time and count | Unchanged, including nested-chain counters |
| llama_sampler_clone | Copies no_perf; starts fresh counters for each cloned chain | Deep-clones sampler state |
| llama_sampler_copy | Copies no_perf, accumulated time and count after child copy succeeds | Copies into existing compatible child handles |
| chain add/remove | Retains existing counters | Transfers ownership according to SAMPLERS.md |

Counters describe operations, not distinct token IDs or model-generated tokens.
An application that directly accepts a token twice records two accepts. A failed
checked accept does not increment the failing chain's count. Earlier child
effects remain incremental, as in the existing sampler API.

Only registered native chain handles are valid for these performance functions.
A null, foreign or non-chain handle returns the usual checked C-facade diagnostic;
the value-returning getter supplies its zero fallback. These errors replace
source assertions, not successful timing measurements. Count and timer arithmetic
rejects overflow outside the source's defined signed-integer domain. Existing
chain, candidate, history and allocation limits still apply; allocator aborts
are not generally recoverable.

## Logger contract

The process-wide logger stores the callback and opaque user pointer together.
llama_log_get returns both. Setting a null callback restores the default callback
while retaining the supplied user pointer, matching the frozen implementation.
The callback and its data remain caller-owned and must stay valid while installed.
A callback receives the exact log level and a transient NUL-terminated byte
string; it must copy any bytes it keeps after returning and must not unwind.

A mutex protects the pair, and logging snapshots it before calling foreign code.
No logger lock is held across a callback, so a callback can get or replace the
logger. The existing active-sampler/context ownership contract still applies.
The application must coordinate callback replacement with any concurrently
executing callback before freeing old user data.

Sampler printing sends one INFO message with the frozen field widths, newline
and label. Formatting is qualified in the default C locale. This does not claim
that the Rust runtime reproduces every original diagnostic message or global
locale change. Existing thread-local error strings remain a separate interface.

On verified GNU Linux and MSVC Windows targets, the default callback writes and
flushes the system C runtime's stderr stream. Small Rust extern declarations
preserve CRT redirection and text-mode behavior, including Windows CRLF
translation. They link only the platform C runtime already used by Rust, never
an original llama/ggml library. Other targets use Rust standard-error output and
have no native stream-interop parity claim.

## Independent verification

Before implementation, the unchanged original public header/library was used on
both MSVC and GNU to capture eight scenarios with 324 operations and 114 timing
observations per host. Inputs cover empty and nested chains, both no_perf flags,
CPU callbacks, skipped backend-only apply callbacks, actual-context sampling,
direct accept, reset distinctions, clone, cross-flag copy and child removal.
The two semantic reference projections are identical. Absolute duration values
remain in the external original captures; fixtures retain exact counts,
candidate float words, ABI, callback bytes, clock units and temporal relations.
No Rust result supplies an expected value.

Twelve chain log messages, callback replacement and reentrant logger getters are
checked. The external original-header client also verifies exact default stderr
bytes on both platforms. Ordinary Rust tests replay the committed reference and
check invalid chain/boolean/output arguments without linking a native oracle.
Raw log bytes include the terminator; stored default-stream bytes retain the
platform's actual newline encoding.

Fixtures and source/build identities are in fixtures/ffi/sampler-perf. Final
debug/release test, Clippy, native-client and export receipts are under
evidence/2026-11-11/c-sampler-perf. These checks qualify the described CPU subset;
they do not close backend or full release gates. Context performance is now
implemented and independently scoped in CONTEXT_PERFORMANCE.md.
