# Backward RoPE and YaRN attribution

The backward rotary implementation in `src/cpu_rope_back.rs` follows the
frozen `ggml/src/ggml-cpu/ops.cpp` and `ggml/src/ggml.c` kernels, licensed MIT,
copyright (c) 2023-2026 The ggml authors. Their complete MIT license is
preserved in this crate's `LICENSE-LLAMA-CPP.txt` and root `LICENSE-UPSTREAM`.

The frozen source identifies its YaRN equations as derived from
`LlamaYaRNScaledRotaryEmbedding.py` in https://github.com/jquesnelle/yarn,
under MIT, copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
The Rust port retains their correction, interpolation and magnitude formulas
with checked scalar layouts, finite-domain boundaries and independent
matching-profile original CPU fixtures. The complete MIT terms follow.

MIT License

Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
