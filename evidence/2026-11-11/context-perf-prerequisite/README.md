# Context-performance prerequisite audit

No product code changed. The three context-performance exports and deprecated
warmup setter remain absent; the qualified inventory remains 216 frozen exports
plus four extensions, with 39 frozen exports absent.

An original-header C++ client dynamically loaded the unchanged frozen original
library on MSVC/Windows and GNU/Linux. Both four-case captures passed. They cover
no_perf true/false, flash attention disabled/enabled, empty-batch error, prompt
decode, separately synchronized and queued one-token calls, reset, warmup toggles,
and native logger output. Full original timings remain in original.json; no
fixed-duration expected values were invented.

## Concrete blocker

The public context-performance struct includes the number of reused compute
graphs and two lifetime graph-reallocation counters. Frozen source increments
these at actual graph reuse and scheduler allocation events. Current NativeContext
executes its microbatches directly; it owns no corresponding graph cache or graph
scheduler allocation lifecycle.

In every captured case, four separately synchronized one-token decodes yielded
n_eval=4 and n_reused=3. Two further one-token decodes without synchronization were
accounted as two prompt tokens when synchronized, producing n_p_eval=6,
n_eval=4 and n_reused=5. A performance reset cleared n_reused, but the retained
graph was reused by the very next decode, returning n_reused=1. These counts also
advance with no_perf=true. Reporting zero or counting repeated call shapes would
misrepresent the observable native event counters.

The reallocation counters stayed zero for these tiny contexts. Source nevertheless
defines them as scheduler callback events and deliberately preserves their lifetime
totals through performance reset.

## Additional lifecycle requirements

- Context creation inherits the model loading start and elapsed time.
- The first nonempty synchronization replaces load time with elapsed time from
  the current start timestamp. Thus resetting before first evaluation can make
  the subsequently reported load duration smaller; reset is not a new model load.
- Timing/token accounting occurs at synchronization. The performance getter and
  printer themselves do not synchronize pending work.
- no_perf suppresses prompt/evaluation duration accumulation, not token counts,
  graph counters or first-evaluation load timing.
- Getters clamp prompt/generated counts to at least one.
- Reset preserves load time, the evaluated-once flag, graph cache, lifetime
  reallocation counters and pending queued work.
- Warmup stores a flag and emits the exact native debug callback message.
  Its meaningful graph effect selects all MoE experts; dense Llama contexts have
  no expert-activation behavior to add as a separately useful implementation.

The source locations and complete capture identities are in blocker.json.
windows-v1/report.json and linux-v1/report.json retain build commands, immutable
source/header/library/model identities, and detailed summaries. Each directory
contains original.json plus exact build/capture stdout/stderr.

This is prerequisite evidence, not an implementation or expanded ABI claim.
