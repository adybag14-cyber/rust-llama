# Third-party algorithm attribution

The native_unstable_sort, median_three, sort_heap_repair, and msvc_sort_range
helpers in src/sampling.rs adapt the Microsoft STL sorting policy from the
MSVC 14.51.36231 algorithm header to checked Rust slices and indices. Their
tie ordering is observable through the frozen llama.cpp sampler C ABI.

Copyright (c) Microsoft Corporation.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

The full upstream license and LLVM exception are included in
LICENSE-MICROSOFT-STL.txt. The rest of this crate retains its MIT license; LICENSE-MIT.txt and
LICENSE-LLAMA-CPP.txt preserve the project and frozen upstream MIT notices.
The combined package license is MIT AND (Apache-2.0 WITH LLVM-exception).

Source: Microsoft STL, MSVC 14.51.36231 include/algorithm.
License source: https://raw.githubusercontent.com/microsoft/STL/main/LICENSE.txt
License SHA-256: 1fc6b6db121d4555d8f661937d7f6013693cd8fd67495d5e5d98c1d18fb83eda

No C++ runtime implementation or original llama/ggml object is linked by these
Rust helpers. The independent native harness is used only for verification.
