# Shared safe Rust FLASH_ATTN_EXT kernel

This module implements the standalone llama_quant::flash_attention CPU kernel. NativeContext and tensor graph dispatch are separate consumers and are not implemented by this module. The original archive and all C graph captures are external numerical references only; Rust runtime code neither links nor invokes the original project.

## Accepted execution contract

Q is an immutable F32 tensor [DK, queries, query heads, batches]. K is F32 or F16 [DK, physical KV columns, key heads, batches]. V is independently F32 or F16 [DV, physical KV columns, value heads, batches]. All scalar channel rows are contiguous; other strides and a checked byte offset remain explicit. F32 and raw u16 typed storage and little-endian byte storage share the same bounds-checked layout. Immutable overlapping input views are safe.

The optional mask is the complete original contiguous F16 [physical KV columns, padded queries, mask heads, mask batches] tensor. The complete query tile affects whether a masked KV tile is skipped. Callers must retain physical columns, holes and mask geometry rather than compacting visible KV values. In particular, masked nonfinite V values can still affect the original tiled computation. Sinks expose one contiguous F32 scalar per query head.

The output layout is [DV, query heads, queries, batches]. Selection::All returns the full output. Selection::Row returns [DV,1,1,1] while retaining original Q dimensions, query-chunk boundaries and the full physical mask. The API never pretends that a selected row was originally a one-query graph.

Key and value storage types may differ in the one-chunk branch. Matching F32/F32 or F16/F16 storage permits the source optimized branches. Quantized caches, kv_rows, noncontiguous scalar channels, empty tensors, dimensions outside the source signed-I32 CPU indexing domain, and NUMA dispatch are outside this first slice. The module rejects invalid checked domains instead of producing placeholder results.

## Source branches and precision

The frozen Q_TILE and KV_TILE constants are both 64. Without force_reference, query count one with one batch, matching K/V storage and at least 512 KV columns chooses split KV. Otherwise matching K/V storage and at least 64 queries chooses the tiled branch. Other inputs use the one-chunk branch. force_reference selects the one-chunk equations and original query partitioning.

One-chunk attention uses original F32 activation queries. F32/F16 K values form F32 products accumulated in F64, then cast to F32. F16 V accumulates and rescales through F16 after each source step; F32 V remains F32. The online maximum/sum, source scalar exp/tanh, sink contribution and final normalization preserve source evaluation order.

The tiled branch uses the frozen scalar F32 GEMM accumulation, 64-column score tiles and the shared maintained CPU row policy. Its softmax exponent/sum is added to the previous F32 sum through the source F64 expression. V tile storage is initialized once for the source query tile; untouched lanes of a final partial KV tile retain the preceding packed values. Full-tile mask skipping and that retained-lane behavior are covered by independent exceptional-input graphs.

Split KV assigns ceil(KV/workers) physical columns to each configured worker. Empty tail workers still execute and return their empty partial. The partial maximum, sum and V outputs are reduced in increasing worker index. The current coordinator performs the final reduction after joining workers; it reproduces source arithmetic, not the original threadpool scheduling topology.

The shared MathPolicy selects the measured scalar or SSE2 exponential/reduction policy. CompilerPolicy separately selects the independently measured MSVC/GNU NaN operand behavior from compiler vectorization. Foreign-runtime scalar exp/tanh is not emulated. Native defaults are qualified by the two captured x86_64 MSVC/GNU profiles; unverified target/compiler/runtime combinations do not inherit a native bit-parity claim.

Environment assumptions are round-to-nearest ties-to-even and gradual underflow. Floating exception flags, traps and altered FTZ/DAZ environments are not reproduced.

## Actual workers and resources

Plan::execute starts real scoped OS workers when more than one is requested. Source query chunks are assigned deterministically to actual workers. ExecutionStats distinguishes configured workers, completed partitions, unique actual worker threads and execution mode. Plan::execute_serial_partitions is explicitly diagnostic and reports one actual thread.

The accepted worker count is checked in 1..=256. Every worker requests a 512-KiB stack. Thread buffer/result/handle vectors are reserved fallibly; spawn errors and worker panic errors join every already-started worker before returning. Borrowed inputs cannot be mutated by a failed execution, and no caller-owned destination is partially overwritten.

MemoryRequirements gives mutually exclusive categories:
- owned returned output bytes;
- staged row results or split partials;
- worker scratch buffers;
- Vec/handle descriptors;
- requested worker stack reservations;
- a separate allowance for standard-library thread metadata.

Their sum is checked once before allocating. Borrowed Q/K/V/mask/sink buffers, caller-owned hidden/Q staging, and the caller's existing stack are excluded; consumers must charge those separately. OS bookkeeping, allocator rounding and runtime implementation overhead are not represented as an exact resident-memory guarantee.

The work preflight uses conservative scalar work units, not CPU-cycle estimates. It includes complete source tile-mask scans even for Selection::Row, score and V channel loops, scratch initialization/copies and worker bookkeeping. Both byte and work arithmetic use checked operations.

## Native evidence and immutable failures

Original graph cohorts are base246, split-boundary18, exceptional288, NaN-lanes390, paired NaN operands192, full-tile48, cross-tile2 and tiled-stage78 per host: 1,262 original graphs. Each host contributes 2,771,152 raw F32 words across the default and forced-reference outputs. An additional 24,066 selected-row words and 75,567 typed-view words are compared against the initial 552 graphs. Each includes unchanged default-policy expected output and a separately captured force_reference diagnostic.

The initial 552-graph implementation matched every finite output across 2,510,640 raw F32 comparisons per host. The first failures were exclusively exceptional NaN payload selection: 11,262 changed words on MSVC and 2,502 on GNU. That implementation, its hashes, source and failed logs are retained in first-native-comparison; no expected value or tolerance was changed.

A subsequent original-only 390-case width sweep covers widths 1..33, 63/64/65 and 127/128/129 for F32/F16 storage. It distinguishes MSVC four-element one-chunk normalization-scale prefixes, sixteen-element tiled and split-reducer prefixes, and GNU two-element prefixes plus scalar tails. These are compiler-profile observations, not properties guaranteed by the C or Rust languages.

The original 552 compact fixture files remain immutable. The NaN-lanes cohort is compacted into a separate directory. Each compaction reconstructs every original GGTR0002 and FAREF001 file and requires its exact original SHA-256 before accepting the encoded fixture. No Rust-produced values enter any fixture set. The paired-operand/tile controls occupy a third independently compacted fixture directory. It reconstructs all eight original host/cohort files and their forced-reference counterparts exactly.

The first complete debug run exposed Rust's unspecified NaN operand choice in unoptimized dot, MAD and scalar GEMM operations. Release had passed all 942 then-existing graphs, but full-validation-v1 retained debug failures on both hosts (36,594 changed words in the initial MSVC cohort and 37,944 in the GNU cohort, all NaN-only). The new 320-graph cohort was captured from original kernels before any Rust comparison. The implementation now explicitly quiets and selects operand payloads according to the measured compiler policy, preserving ordinary arithmetic and order for finite inputs. Original-only partial-buffer diagnostics also retain the split reduction's intermediate maxima, sums and values.

The first expanded focused run passed all 1,262 release graphs and all 710 debug NaN controls on both hosts; it then stopped at two Clippy collapsible-if errors. That failed validation run remains recorded in explicit-nan-v2. Final gates are recorded separately and never overwrite earlier evidence.

Independent resource review found an initialization-work undercount for tiled selected rows with many idle workers. Each worker's full KV_TILE * DV packed buffer is initialized even when it owns no selected row. The corrected work accounting charges every scratch element plus output/staged zeroing explicitly. A regression with 256 workers, DV 4096 and one selected row proves a minimum 67,108,864 packed-buffer stores, and rejects a lower work limit before allocating those buffers. The former expression is replayed only in a separate negative-control crate; the test's lower-bound assertion fails before execution, avoiding the large allocation.

Final validation and handoff status must be read from the later guarded reports; this design document itself is not a success claim.

## Consumer distinction

The original Llama flash graph may cast both physical K and V to F16 even when resident cache storage is F32. That consumer graph policy belongs to NativeContext and must include its own full physical staging budget. The generic graph kernel deliberately retains independently measured F32 and F16 inputs and does not silently cast F32 K/V.
