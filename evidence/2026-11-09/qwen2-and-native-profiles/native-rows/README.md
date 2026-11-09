# Shared native row math and physical attention rows

This handoff targets the frozen llama.cpp archive and the Rust checkpoint 6ddeb75. Product source was not edited by this draft.

## Merge scope

Use the nine changed/new files in final-source-hashes-v2.json and the fifteen fixtures in new-fixtures.json. handoff/merge.patch contains the initial row-math changes against the exact checkpoint; handoff/descriptor-accounting.patch is the subsequent NativeContext allocation correction. The final source tree is rust-draft. The root-owned Qwen2 changes overlap math.rs and llama.rs, so merge the row functions and call sites while retaining the Qwen2 rotary implementation.

The changed runtime files are math.rs, math/native_rows.rs, llama.rs, mamba.rs and context/native.rs. BERT's source file is unchanged (SHA-256 fdf7faeadb86e40e779ae1cad3f4f9f034f96d533fbc7fec85e6ced941482c3a); its existing attention and layer-norm calls use the corrected shared functions. Three runtime integration tests and the FFI context regression file supply the remaining four source files. Do not copy the draft-only probe package or its root workspace member. No C/C++ product code or original library linkage is introduced.

The initial source snapshot and successful validation are retained under final-source-hashes.json and validation/final. The descriptor correction changes only context/native.rs, to SHA-256 6cf3e06725b79479e769cf698ba64f7b00ad5abee7d90a149c1cc9ed0a5e4da0. Its new profile checks are recorded separately under validation/final2, with an independent negative control under validation/descriptor-negative-control.

## Arithmetic contract

math::native_rows dispatches the measured original x86_64 Linux GNU profile to SSE2 and the measured Windows MSVC profile to scalar arithmetic. This is a frozen-source compiler profile, not runtime feature detection or a claim about all original ISA paths. The helper source is exactly the independently verified final quantization-agent handoff: 7,029 bytes, SHA-256 11952fcd2a620c2dd6402c7493a3d591395fd4977f4b6e58ee38df5c4d3133cc.

SiLU receives the complete original row. Softmax preserves the four-lane vector prefix, scalar tail, F32 pair/group sums and F64 total. Normalization takes a F64 reciprocal and casts once before F32 scaling. Layer norm preserves the source rounded mean and centered-variance reduction; the variance helper covers both finite values and measured exceptional payload ordering. The tests assume round-to-nearest-even and gradual underflow with FTZ/DAZ disabled. Floating-point exception flags are outside this claim.

Llama applies SiLU to its whole feed-forward gate row; Mamba does so to complete convolution and z rows before their subsequent products. Ordinary dense and BERT attention pads source rows to 256 columns. NativeContext uses the original common width across selected streams after allocating the whole microbatch, based on highest occupied physical cell plus one. It retains -Inf columns for holes, unrelated sequences and future positions. Shared physical cells appear once, using membership of the query's first sequence ID.

AttentionLayout validates nonempty widths, ordered unique in-bounds physical columns and shape agreement. Row/output allocations use checked sizes and fallible reserves. Masked softmax rejects NaN, positive infinity and an all-masked row. Tests exercise those error paths and padding overflow.

## Allocation review and correction

The committed full-batch KV bound from 63ef50ac remains: later microbatches can gather initial occupancy plus the entire batch, capped at capacity. The original 128-microbatch rejection/success regression remains intact.

One reused score row needs at most `4 * capacity` bytes. Its borrowed physical index map needs at most `8 * capacity` bytes on the measured x64 targets. Both fit the existing `16 * capacity` allowance, even with sparse high physical slots or common stream widths. The helper and AttentionLayout add no other row-sized allocation.

A subsequent review found that a single large microbatch also retains slot and stream-index buffers. These cannot reuse that same row allowance. The final correction separately charges four usize entries per microbatch token (slot pair, stream index and temporary find-slot index), the outer stream-index vector and gathered layer descriptors. Resident storage now includes the outer Stream vector, and constructor validation includes stream/layer descriptors. A fixed 256-entry validation table and a fallibly allocated vector of index vectors replace heap trees, making metadata payloads explicit while preserving ascending stream processing order.

A 512-token single-microbatch regression rejects a 32-KiB temporary allowance without changing serialized state, and accepts 64 KiB. Its negative control removes only the new chunk metadata charge and must fail the rejection assertion. The work limit still clamps to the production 2-GiB ceiling. Allocator-internal bookkeeping, thread stacks and shared immutable model weights are outside the owned heap payload budget.

## Independent evidence

The original adapters use the frozen header and original source libraries. Windows and GNU expected captures were generated independently, including separate GNU pretrained model outputs; no Rust output became expected data. Existing Windows corpora and predeclared portable tolerances remain unchanged.

The original numerical handoff includes 603,231 exact F32 primitive outputs and 417 F64 softmax sums, plus 198 actual GGML SiLU/softmax graph cases with 144,666 outputs per native profile. Centered variance adds 206,952 centered F32 values and 2,786 F64 variances across both explicit policies, plus 312 original NORM graphs with 368,208 outputs per host. These counters describe the independent datasets, not multiplied reruns.

The new C physical-layout corpus has 10 scenarios per host: dense 256-boundary crossings, holes, removed prefixes, shared cells, separate streams and their common row width. Each contains 31,744 exact logit values and 193,868 raw state bytes. The ordinary FFI regression replays this corpus. The broader existing native context harness compares 489,158 F32 values per host/profile across trained and tiny models, plus raw source state and metadata.

The final first-stage GNU release facade proof compares eight original model corpora and 4,459,272 F32 values with zero bit differences: stories260K, Mamba130M, BGE small, SmolLM2, stories15M Q4_0 and three synthetic Mamba variants. Those adapters retain the duplicate prompt-last/first-greedy rows; the root integration converts to the existing deduplicated fixture convention. Windows re-ran all six explicit pretrained regression tests, including big-endian BERT.

## Preserved failures and limits

The early formatting and Clippy failures remain in their job records. The first GNU BERT facade comparison failed before the centered-variance correction. Intermediate tensor traces established that embedding normalization, Q/K/V, padded attention and its output matmul matched; the first difference was the next source NORM. The failed report and trace are retained. Quantization-agent variance evidence also retains its initial exceptional-payload failure before the final 11952fcd helper. No expected values or tolerances were widened to pass these checks.

This slice does not add C exports, recurrent native-memory support, flash attention, new model architectures or accelerated ISA backends. The prior partial ABI coverage remains 184 of 255 original exports plus four documented Rust extensions. Root owns the combined model-reference selectors and final workspace integration.
