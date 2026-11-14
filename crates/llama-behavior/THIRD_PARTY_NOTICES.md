# Third-party algorithm attribution

The native_unstable_sort, median_three, sort_heap_repair, and msvc_sort_range
helpers in src/sampling.rs adapt the Microsoft STL sorting policy from the
MSVC 14.51.36231 algorithm header to checked Rust slices and indices. Their
tie ordering is observable through the frozen llama.cpp sampler C ABI.

src/native_regex.rs is an independently written bounded Rust byte matcher whose
parser/class/capture compatibility was informed by the same Microsoft STL regex
header and verified against std::regex. This notice and license also cover any
derived compatibility details in that module; its execution engine uses checked
Rust states and explicit allocation/work bounds.

Copyright (c) Microsoft Corporation.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

The full upstream license and LLVM exception are included in
LICENSE-MICROSOFT-STL.txt. The rest of this crate retains its MIT license; LICENSE-MIT.txt and
LICENSE-LLAMA-CPP.txt preserve the project and frozen upstream MIT notices.
The combined package license is MIT AND (Apache-2.0 WITH LLVM-exception).

Source: Microsoft STL, MSVC 14.51.36231 include/algorithm and include/regex.
License source: https://raw.githubusercontent.com/microsoft/STL/main/LICENSE.txt
License SHA-256: 1fc6b6db121d4555d8f661937d7f6013693cd8fd67495d5e5d98c1d18fb83eda

No C++ runtime implementation or original llama/ggml object is linked by these
Rust helpers. The independent native harness is used only for verification.

The GNU/Linux sampling/gnu_order.rs policy is independently expressed Rust,
shared in algorithm with tensor/src/native_sort.rs and verified against original
GNU 11.4.0 sampler outputs. No GNU header or library implementation is copied or
linked. The package license expression is unchanged.
