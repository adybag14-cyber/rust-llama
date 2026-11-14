# Native dense Llama CUDA execution

This crate executes a complete supported dense Llama graph through persistent
CUDA buffers and Rust-authored typed PTX kernels. It supports F32, F16, Q4_0
and Q8_0 matrices with one sequence and F32 KV. The original llama.cpp/GGML
runtime is neither linked nor launched.

Use `CudaLlama::from_inference_model` or `CudaLlama::new`, then `decode` or the
shared `Model::generate_with_session` pipeline. The CLI exposes the same path
through `--backend cuda` for `generate` and `logits`.

The [complete contract](../../docs/CUDA_RUNTIME.md) documents admission,
requested CUDA-buffer byte accounting, host memory, ownership, cancellation,
independent numerical references and actual hardware tests. Unsupported
architectures/formats and the remaining full-backend requirements are explicit.
