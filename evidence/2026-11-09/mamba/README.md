# Mamba-1 GGUF integration

This checkpoint adds the original Mamba-1 tensor layout to the Rust GGUF
inference facade. The executable path owns convolution and SSM state in Rust;
it does not load or invoke a llama/ggml reference library.

## Independent pretrained capture

The external model is mamba-130m-hf.f16.gguf, 261,787,232 bytes,
SHA-256 cde77cf92d0fbee2346f342faf7f4992622b28d163277a7e52eefa0331be7e32.
The recorded publisher revision is ecd441bd76c60ebf9c3e71bfc3f83ba64fa1edd1.
Model metadata, download identity, original request and oracle hashes are
retained in this directory and in the fixture provenance.

The frozen C API produced expected results before the Rust model was evaluated:
three prompts, 34 unique full-vocabulary rows, 1,709,520 logits and 24 greedy
predictions. The reference uses one CPU thread, F32 state and disabled flash
attention/offload. Only exact duplicate prompt/greedy overlap rows were removed
when packing the immutable F32 binary.

| Platform | Compared logits | Exact F32 bits | Maximum absolute error | Greedy IDs |
| --- | ---: | ---: | ---: | --- |
| Windows MSVC x64 | 1,709,520 | 1,709,520 | 0 | 24 exact |
| WSL Ubuntu x64 | 1,709,520 | 191,402 | 0.00008940696716308594 | 24 exact |

The numerical assertion is the unchanged pretrained-model budget
`2e-4 + 2e-5 * abs(reference)`. The largest Linux difference consumes about
42.805% of its allowed bound. Output bytes, token IDs, usage and finish reasons
are exact on both platforms.

## Recurrent arithmetic and loading

The source helper `ggml_compute_softplus_f32` evaluates log(1 + exp(x)) with
the addition rounded to F32 before the logarithm, for x at most 20. The former
Rust log1p(exp(x)) expression did not preserve that arithmetic. The helper now
matches the source's staged operations. The separate generic GGML SOFTPLUS
operator has a different source formula and remains separately implemented.

The loader checks metadata and dimension multiplication before allocating
weights, preserves packed matrix execution, and requires F32 convolution/A
tensors as the original CPU kernels do. Tied output, separate output and the
source boolean dt/B/C RMS flag are implemented. Unsupported additional tensors
are reported, not silently ignored.

## Ordinary synthetic tests

Three tiny deterministic models were independently executed by the original
runtime before the first Rust test. They exercise missing RMS metadata, true
RMS metadata, explicit false metadata, tied output and separate output.
Nine prompt/model combinations contribute 54 unique rows, 27,648 logits and
36 greedy predictions. The portable tiny-model bound was fixed beforehand at
`2e-6 + 2e-6 * abs(reference)`. Windows matches all values exactly.

Additional ordinary checks cover required tensor names and dimensions,
malformed/missing/zero/overflowing metadata, unsupported extra tensors, exact
decoded-weight limits, cancellation at token/layer boundaries, state restoration
and deterministic replay. Temporary malformed inputs never replace reference
fixtures.

## Focused product verification

The windows and linux subdirectories record debug/release tests for
llama-inference and llama-model-runtime, both explicit new pretrained tests,
and strict all-target Clippy. Every step passed. Source hashes for the root-owned
implementation/test inputs were identical before and after each run.
The complete workspace integration gate is recorded separately after all
concurrent implementation slices are stable.

The high-level strict Rust Context supports Mamba recurrence. Full native C
recurrent-memory/session compatibility, additional recurrent architectures,
accelerator model execution and performance parity remain open.
