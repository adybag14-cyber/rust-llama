# FLASH_ATTN_EXT row-map kernel handoff v2

This is an additive revision of the sealed dense shared-kernel handoff at ../flash-kernel/handoff. It adds source I32 row maps and corrects two measured MSVC NaN operand-order cases in common tiled arithmetic. It is not a tensor or NativeContext integration.

## Apply boundary

Apply only candidate/crates/quant over the sealed dense-v1 candidate. manifest.json records exact v1 preimages and new bytes; preimage holds the two existing changed files. source.patch includes all text changes. The standalone root Cargo.toml, Cargo.lock and rust-toolchain.toml remain unchanged and are excluded. No dependency, cpu_math, lib.rs or product workspace file was changed in this additive slice.

The dense-v1 handoff remains immutable. Its manifest SHA-256 is ef9d404ba3f3a61e8c08b4e9cc12153bc6c549701fc4373a1c1267874c496281. Its initial kernel hash is 796620f9d39d3f5ed81b0064a7ee0b0ded69ab5dfee9afe3ee1a4f2def6930ed. The final row-map kernel hash is f58cc7552796ee1094dbe35278a7ee0988bded8327bae8c9df35b4350ae45ec7.

fixture-declarations.json and fixture-declarations.tsv contain only the 20 new fixture inventory rows. They are additions for the shared manifest, not a replacement. Existing fixture bytes are unchanged.

## Additive API

KvRows::from_i32 and KvRows::from_bytes borrow contiguous I32 [logical columns, query batches] storage with a checked byte offset. Plan::new_with_rows takes the existing Inputs and Params plus that map. Plan::new remains the dense entry point.

A row map requires a full contiguous F16 mask with exactly its logical column count. Its batch count matches Q, while K and V each retain physical batch count one. Every nonnegative map value must be inside both physical row extents. Validation scans the complete map after byte/work budget checks and before output allocation or worker creation. Negative source padding remains signed; one-chunk skips it, tiled packing uses physical row zero. The full physical K/V bytes and full source query/mask geometry are required, including masked nonfinite V values. A row map disables split KV; it does not disable eligible tiled execution.

Output remains [DV, query heads, queries, batches]. Selection::Row retains full source tile membership, mask geometry and map validation. The owned output and existing worker semantics are unchanged. Inputs, Params, Limits and MemoryRequirements do not gain fields, so the dense API remains source compatible.

## Arithmetic and native profiles

The frozen scalar GEMM C expression has observable compiled NaN operand choices. The MSVC fixed64 PV call prefers a new NaN product at offsets 0/16/32/48; the variable-DK score call does so at multiples of 4 below floor(DK/4)*4 and retains the accumulator through its tail. GNU retains the accumulator in both. Finite product/sum order is unchanged. These rules are independently captured for both dense and identity-map graphs and are explicitly selected by CompilerPolicy, separately from MathPolicy.

The exact original profiles are documented in evidence/native-build-profile.json and raw generated CMake/compiler settings. MSVC is 19.51.36260.0, v145, x64 Release with /O2 /Ob2 /arch:SSE4.2 and the recorded scalar source macro path. GNU is 11.4.0, -O3, scalar-source GEMM with the baseline SSE2 row-exp path. Native DLL/SO hashes are fixed in every original-capture provenance. Other compilers, flags, operating-system math runtimes, targets and altered floating environments require separate evidence; this handoff does not imply their bit parity.

## Checked resources and workers

The byte categories remain disjoint: returned output, staged results, scratch, descriptors, requested additional thread stacks and thread metadata allowance. Borrowed Q/K/V/mask/sink/map storage and consumer-owned hidden/cast buffers are excluded and must be charged by the caller. A full-map scan adds conservative scalar work even for one selected row. Existing idle-worker scratch-initialization charges remain in place.

execute starts actual scoped workers for configured counts above one. The accepted range remains 1..=256; N=1 runs on the caller. worker_threads counts distinct threads executing partitions, including the caller only in the one-thread or serial diagnostic path. execute_serial_partitions is explicitly diagnostic. Spawn failures and panics join every started worker; errors cannot partly mutate caller input or output. Accounting is a checked logical allocation/work envelope, not an exact allocator/OS resident-memory guarantee.

## Independent numerical evidence

The final gate compares 3,746 independently executed original graphs per host/profile and 7,060,000 default/forced raw F32 words:

- retained dense cohorts: 1,262 graphs and 2,771,152 words;
- row-map cohort: 576 graphs and 2,119,032 words;
- PV operand-position controls: 552 mapped plus 552 dense graphs and 1,449,432 words;
- QK position/tail controls: 402 mapped plus 402 dense graphs and 720,384 words.

The dense control calls are actual independent original graph executions. Their equality with identity-map outputs is required by lossless compaction and their original output-stream hashes. No Rust output is used to produce references. Another 211,118 selected-row/typed-view words are supplementary comparisons against those original cases.

The final all-quant test suite, including dense and row-map controls together, is run in Rust debug and release on both verified hosts. Strict all-target Clippy and formatting are also required. manifest.json records final counts, source guards and command outcomes.

## Retained failures and review

The original row-map failure was 6 MSVC default tiled cases / 16,128 NaN words, while finite values and all GNU words matched. The independent PV regression rejected the old rule in 80 comparisons / 52,644 words. The QK regression rejected the earlier score rule in 236 comparisons / 105,728 words. Each preimage, failed output log and native expected stream remains separately identified. The intermediate complete v2 validation is retained, while the final corrected source receives a fresh v3 full validation.

Independent core review covered layout/index proofs, complete-map work charging, allocation ordering, selected-row semantics and worker ownership. It then confirmed that both NaN corrections leave those paths unchanged. The reviewer record is preserved in evidence/independent-review.json.

Quantized caches, noncontiguous scalar channel rows, empty dimensions, out-of-domain signed CPU extents and NUMA remain checked exclusions. BF16/packed flash arithmetic, tensor dispatch, automatic model casts, per-layer microbatch execution and unified NativeContext state behavior are separate completion gates. This handoff does not claim them complete.
