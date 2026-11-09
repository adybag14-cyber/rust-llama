# T06 peer review corrections

Reviewed by the independent T08 owner before integration.

## Pageable host uploads and nonblocking CUDA streams

**Finding:** `cuMemcpyHtoD_v2` can return after staging a pageable input, while the DMA remains pending. The generated-kernel path used a nonblocking stream without an explicit dependency. In particular, the final upload of the output buffer could race the kernel's writes.

**Correction:** `CudaBuffer::copy_from` now completes `cuCtxSynchronize` before returning. HIP/MUSA blocking uploads likewise explicitly synchronize their selected device. The context/device remains selected, and all host/device allocations remain borrowed through that wait. Asynchronous copies remain a separate future implementation requiring retained submission objects and events.

**Reference:** [NVIDIA CUDA Driver API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.2.1/cuda-driver-api/api-sync-behavior.html), synchronous-copy rules, accessed 8 November 2026.

## Pointer-only module image parsing

**Finding:** a public safe method accepted an arbitrary C string and passed it to `cuModuleLoadData`, which sniffs multiple image formats and takes no byte length. A C string alone cannot prove a binary image is complete.

**Correction:** the arbitrary-image `module` method is now `unsafe` with an explicit complete textual PTX contract. The safe `compile_generated` method accepts only the private-source artifact created by this crate's typed generator. The normal generated-kernel path uses that sealed artifact path.

## Validation after both corrections

- `cargo fmt -p llama-device-family`: exit 0.
- `cargo test --offline -p llama-device-family`: exit 0, 19 tests passed, none ignored.
- `cargo clippy --offline -p llama-device-family --all-targets -- -D warnings`: exit 0.
- `git diff --check`: exit 0.

These changes were compiled and reviewed against the vendor API contract. The absent CUDA/HIP/MUSA drivers still prevent hardware ordering and lifetime fault tests; the corrections do not convert the unavailable device tests into passes.
