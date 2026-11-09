# Automatic GGUF shard loading through the existing C ABI

Apply only source.patch and the four fixture files to the reviewed sharded-GGUF dependency. The candidate tree contains complete changed files; preimages and SHA-256 values are recorded in handoff.json. No product files or Git state were changed.

The loader opens one validated GgufSet, obtains primary metadata and complete tensor statistics, and consumes the same owner for full model loading. Vocabulary-only loading validates every declared shard. Automatic numbered-file discovery is supported; no explicit split-list C API is added.

Both Windows MSVC and GNU x86_64 pass focused debug/release, strict all-target Clippy, the ordinary original-native regression, and the same dynamically loaded frozen-header C harness. Each profile checks 96 fixed cases: 53 accepted, 43 rejected, 163,840 exact logit words. Two original assertion aborts become checked errors. Expected numeric storage is deduplicated to 16,384 words per platform. Restoring the previous FFI loader makes the regression fail at 17,008 reported bytes versus the original 17,648.

The export surface remains 184 frozen functions plus four diagnostic/layout extensions, verified in all four binaries. Initial harness and test compile failures, the deliberate negative control and original native reports are retained under evidence. Original C/C++ sources and full raw captures stay external.
