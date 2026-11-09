# Flash kernel integration handoff

The candidate adds one maintained safe Rust CPU implementation in llama_quant::flash_attention. It does not link or load llama/ggml, add a dependency, change the existing cpu_math implementation, or integrate a model/tensor consumer.

## Apply boundary

Only candidate/crates/quant belongs in integration. The standalone root Cargo.toml, Cargo.lock and rust-toolchain.toml are intentionally excluded. Existing lib.rs has only one semantic change: pub mod flash_attention after pub mod cpu_math. Its exact validated standalone copy uses CRLF; source.patch contains the normalized one-line semantic delta so the integrating workspace can retain its existing line endings. All new candidate files have exact byte counts and SHA-256 values in manifest.json.

fixture-declarations.tsv and fixture-declarations.json contain additions for the shared integration fixture inventory. Do not replace the shared manifest with either file.

## Consumer contract

Construct checked immutable View4 values over F32 slices, raw F16 u16 slices or explicit little-endian bytes. Plan takes the complete original Q geometry, physical K/V columns and optional full contiguous F16 mask. Selection::All is the intended complete microbatch/layer entry; Selection::Row preserves original query geometry and full-tile mask membership for diagnostic row evaluation. Owned output order is [DV, query heads, queries, batches].

Source query and KV tiles are 64. The kernel selects one-chunk, tiled or split-KV execution from original query/batch/storage geometry, while force_reference selects the captured one-chunk diagnostic. Matching K/V types enable tiled/split branches; mixed F32/F16 remains one-chunk. Quantized caches, kv_rows, NUMA, empty dimensions and noncontiguous channel axes are explicit exclusions. Parameters and the derived softcap scale must be finite; tensor NaN/Inf payloads are retained under the measured profiles.

MathPolicy selects source scalar versus SSE2 row math. CompilerPolicy independently selects the measured MSVC/GNU NaN operand choices. Default native policy is qualified only by the two captured x86_64 MSVC/GNU configurations and current fixed-corpus checks. Other toolchains, OS runtimes and targets need their own compatibility evidence. Round-to-nearest and gradual underflow are assumed; no floating exception/FTZ/DAZ emulation is claimed.

The original Llama consumer may cast both physical K/V views to F16 even for F32 resident storage. That source graph rule belongs in the model consumer, along with complete layer microbatch preparation and physical holes/future/mask columns. Do not compact visible columns or use a one-row replacement Q geometry.

## Workers and accounting

execute runs real scoped OS workers for counts greater than one, up to 256. Execution stats expose configured workers, completed partitions and unique actual worker threads. execute_serial_partitions is diagnostic and reports one actual thread. Split partial reduction is deterministic in ascending source worker order after joining; original threadpool scheduling topology is not reproduced.

requirements separates returned output, staged results/partials, worker scratch, descriptors, requested thread stacks and metadata allowances. Consumers should charge the reported peak once, then separately charge borrowed Q/K/V/mask/sink storage and model hidden/cast staging. Allocator/OS implementation overhead is not an exact resident-memory promise. The work limit charges all idle-worker packed initialization as well as selected/full-row arithmetic and full-mask scans.

## Evidence

manifest.json records both hosts' fresh debug/release test counts, exact source/fixture guards and strict all-target Clippy/fmt commands. All 1,262 original graph cases are compared in default and forced-reference modes: 2,771,152 F32 words per host and profile. Selected-row checks add 24,066 words; typed F32/F16 checks add 75,567 words against original outputs. Those are supplementary comparisons of existing graph cases, not new independent graphs.

The three compact fixture families losslessly reconstruct the original input/output streams and require matching SHA-256 values. They include finite, masked/nonfinite, signaling/quiet/distinct-payload NaNs, vector/scalar tails, tile transitions, uneven split partitions, empty workers, GQA and byte-strided views. Original graph captures were made before their Rust comparisons.

evidence retains the initial lifetime compile error; first, second and third numerical iterations; the first full debug failure; expanded-corpus Clippy failure; final full validation; and work-budget review/negative control. No expected output or numerical tolerance was changed to obtain a pass. The negative control restores only the old initialization-work formula in a separate crate and must fail its lower-bound assertion before any large packed scratch allocation.

NativeContext and tensor dispatch parity remain separate integration gates; this standalone handoff does not claim them complete.

The combined final validation driver returned failure only after all eight quant debug/release/Clippy/fmt stages passed: the first isolated negative-control crate omitted an existing IQ-sort fixture needed by lib tests. Both compiler errors remain under work-negative-control. The corrected work-negative-control-v2 copies that unchanged fixture and runs the intended old-formula assertion on both hosts; no kernel, test, fixture expected value or validation threshold changed for this harness correction.
