# Context-attached CPU sampling checkpoint

The Rust FFI now implements eight additional frozen exports: llama_set_sampler,
llama_get_sampled_token_ith, and the logits/probabilities/candidates pointer and
count accessors. Supported Rust prefix execution includes source-empty stages,
greedy, top-k and temperature, followed by the source-defined ordinary CPU suffix.
The detailed contract and remaining stage/lifetime boundaries are in
crates/ffi/BACKEND_SAMPLING.md. This checkpoint does not close a full release gate.

## Independent evidence

The authoritative frozen archive is SHA-256
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
The original MSVC and GNU libraries, untouched original-header clients and input
model are identified in fixtures/ffi/backend-sampling/provenance.json. The client
sources, executable files and original native binaries remain outside the Rust
product at C:/Users/adyba/rust-llama-reference/backend-sampler-20261114.

Each host's context corpus contains 26 scenarios, 390 accessor rows, 77,016 F32
words and 181,356 candidate IDs. It checks initial/attached/decoded/invalid/
detached/raw-after-detach states, mixed sequence output ordering, disabled and
absent samplers, greedy/top-k ties, positive/zero/negative temperatures, CPU
suffixes, constructor sampler configurations and output-limit rejection/recovery.
Every field is compared exactly, including pointer presence/alias and buffer
counts. This uses a deterministic tiny model; pretrained inference and throughput
are not accepted by this corpus.

The separate original-header CPU suffix corpus records four stages from two
contexts, with 2,500 F32 words and 1,250 candidate IDs. It independently checks
intermediate typical filtering and the following seeded distribution. A genuine
initial GNU failure selected token 315 where the original selected token 98. The
old non-MSVC native sort had retained stable ordering. The correction uses the
independently expressed GNU policy already verified by the tensor crate. The
new tests check every stage ID, logit, probability, selection and sorted flag.

No Rust output supplies the expected numerical values. The original context
captures preceded implementation; the diagnostic suffix captures preceded the
ordering correction. Invalid original client variants, failed builds, the real
GNU mismatch, formatting corrections and corrected runs remain preserved and
hashed in provenance. The native direct-greedy-as-chain case is excluded because
it aborts the original; Rust returns a checked rejection for that source-invalid
input.

## Acceptance artifacts

validation-msvc.json and validation-gnu.json record exact offline/locked Cargo
commands, per-step exits and final dynamic-library hashes. Both debug and release
libraries are replayed through clients compiled against the untouched original
header, including the complete intermediate CPU suffix corpus. All JSON values
must equal their original profile capture exactly.

validation-summary.json gives the final per-crate test totals and original-header
comparisons. The focused tests cover context attachment/lifetime, cloning, prefix
names, checked topology mutation, rejected unsupported prefix replacement,
mutable sampled buffers and rollback. The model-runtime test rejects a fallible
output consumer and also triggers cancellation after it, verifying preservation
of native state and execution statistics followed by successful retry. The
consumer executes before native context state is committed.

PE/ELF export lists are retained for every final library. Each contains exactly
247 frozen llama exports and four Rust diagnostic/layout extensions. The eight
remaining absent frozen functions are:

- llama_attach_threadpool
- llama_detach_threadpool
- llama_model_init_from_user
- llama_model_save_to_file
- llama_numa_init
- llama_opt_epoch
- llama_opt_init
- llama_opt_param_filter_all

The comparison also checks the complete named symbol sets against COVERAGE.md.
New symbol presence does not claim support for every parameter combination.

## Reproduction and scope

Use Rust 1.99.1 and the unchanged locked workspace dependencies. Run cargo test for
llama-ffi, llama-model-runtime and llama-behavior in debug and release; run strict
Clippy for all three packages and targets. The external validate.py driver and
its native client commands are retained with identities in the evidence manifest.
The only dependency change is a local llama-ffi to llama-tensor edge. The existing externally pinned BGE encoder test is the one pre-existing ignored
FFI test; this sampling checkpoint does not claim to rerun that model corpus.
Registry
packages and the lockfile's external package set are unchanged.

Backend distribution/top-p/min-p/dynamic-temperature/penalty/bias prefixes,
custom GGML backend callbacks, nested backend chains, nondefault output capacity,
GPU sampling and other native context architectures remain unsupported. The
admitted prefixes do not produce probability tensors, so their original sampled
probability getters return NULL and zero. Those getters do not imply backend
distribution support. The contiguous raw llama_get_logits buffer for fully
sampled rows is outside parity; the indexed and sampled getters are exact.

This evidence certifies the stated CPU C API subset on Windows x64 MSVC and GNU
x86_64. It does not certify the full C ABI, sampler backend catalog, all platforms,
pretrained numerical behavior or performance. All broader release gates remain
open.
