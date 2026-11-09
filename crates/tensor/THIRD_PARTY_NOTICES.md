# Third-party algorithm attribution

The MSVC sorting helpers in src/native_sort.rs adapt the behavior crate's checked Rust implementation of Microsoft STL 14.51.36231 std::sort. Tie ordering is observable in the frozen GGML ARGSORT and TOP_K outputs.

Copyright (c) Microsoft Corporation.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

The complete upstream license and LLVM exception are in LICENSE-MICROSOFT-STL.txt (SHA-256 1fc6b6db121d4555d8f661937d7f6013693cd8fd67495d5e5d98c1d18fb83eda). Project and frozen upstream MIT notices are preserved in LICENSE-MIT.txt and LICENSE-LLAMA-CPP.txt. The combined package license is MIT AND (Apache-2.0 WITH LLVM-exception).

The GNU/Linux sorting policy is independently expressed Rust, tested against the original GGML graphs compiled with GNU 11.4.0. No GNU library implementation is copied or linked. The product contains no original C/C++ object; native libraries are external verification oracles only.

Microsoft license source: https://raw.githubusercontent.com/microsoft/STL/main/LICENSE.txt
