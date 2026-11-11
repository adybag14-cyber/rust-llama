# Frozen backward flash-attention status

Original operation ID 75 remains without an executable Rust constructor.
The pinned original public API `ggml_flash_attn_back` unconditionally aborts
at the beginning of its constructor in `ggml/src/ggml.c`, with the diagnostic:

> TODO: adapt to ggml_flash_attn_ext() changes

That abort precedes its shape checks and tensor construction. The file also
contains a dormant CPU backward kernel, but calling the original public API
does not reach it. A numeric implementation created by bypassing the constructor
would require a separately stated compatibility target and independent kernel
evidence.

## Reproduced original rejection

An isolated native child created F32 Q/d tensors of shape `[2,1,1,1]` and
K/V tensors of shape `[2,2,1,1]`, then called the original constructor with
causal masking disabled. Both measured libraries rejected the call with the
same constructor diagnostic.

| Original profile | Child process exit | Result |
| --- | ---: | --- |
| Windows MSVC | 3221226505 | Native unconditional constructor abort |
| GNU/Linux | -6 | Native SIGABRT |

The raw stdout/stderr, original-library/source hashes and capture provenance
are preserved under
[evidence source-gap records](../../evidence/2026-11-11/tensor-rope-back/source-gap/flash-back-constructor).
The standalone reference harness and native libraries remain outside the
maintained product. No Rust implementation participates in this capture.

The support table keeps this ID pending. Forward flash attention and explicit
RoPE backward execution do not supply a graph automatic-differentiation system
or a working original backward flash-attention public API. Custom callbacks,
optimizers and general training-state integration are also separate open work.
