# CUDA model comparison data

The F16 model is a native-original conversion of the existing hash-pinned
pretrained stories260K.gguf. The 601,248-byte converted model has SHA-256
4cd29bbe9cb3b735750837816a43c3dd4b473dbd658e8612816f7c4d3926ac21.
Weights remain external. The existing inference oracle and the frozen original
quantizer produced the model and expected values without invoking Rust.

stories260k-f16-msvc-logits.f32le.bin contains 59,904 unique F32 logits in
little-endian order, covering six prompts and 96 greedy predictions. The
metadata retains exact prompt and generated token IDs, pieces and bytes.
For each case, prompt rows precede greedy rows after the shared first row.
That shared row was verified bit-identical before packing it once.

Both Windows and GNU CUDA tests compare this same original MSVC CPU corpus.
They keep the pre-existing absolute 2e-4 plus relative 2e-5 bound; token IDs and
text must be exact. This is a CPU-reference comparison of native Rust GPU
execution, not a claim of original CUDA backend algorithm equivalence.

provenance.json records native executable/library, model, request and raw
capture identities. The C++ reference programs remain outside the Rust product.
The source archive's notice is retained in the repository LICENSE-UPSTREAM.
See docs/CUDA_RUNTIME.md for implemented behavior, tests and limitations.
