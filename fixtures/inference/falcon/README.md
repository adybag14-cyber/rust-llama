# Falcon independent inference fixtures

These synthetic models and original C API reference outputs were captured on Windows MSVC and Linux GNU before Rust Falcon implementation. The corpus covers shared or distinct attention/FFN affine normalization, inactive/omitted second-norm bias, tied/untied output, MHA/MQA/GQA, full-head NeoX RoPE, six packed/float storage formats, and checked malformed or unsupported modes. Expected rows are losslessly stored as little-endian F32 with complete-row deduplication. provenance.json records source, tool, library and input hashes. All original source and oracle implementation remain external.

These are structural conformance fixtures; pretrained weights and pretrained quality are separate requirements.
