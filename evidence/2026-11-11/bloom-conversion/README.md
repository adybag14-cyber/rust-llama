# BLOOM conversion validation

This evidence records the native Rust BLOOM checkpoint converter, its original-reference captures, and the final Windows MSVC and Linux GNU validation. The frozen ZIP is the authority; native reference programs, Python libraries, build outputs and trained model weights remain external.

The [summary](summary.json) distinguishes Rust test counts from the independent native case counts. Each host completed ordinary converter and GGUF tests in debug and release, strict Clippy and formatting, then converted unchanged pinned bigscience/bloom-560m to F16, Q8_0 and auto. The three trained outputs total 2,871,136,896 bytes per host and match the original complete-file hashes. The test retained those files for downstream inference checks. For exact replay, the unchanged input directory must have basename `official-hf`: the source infers `general.name = Official Hf` from that basename. Arbitrary directories remain valid for general conversion, but their source-defined default metadata yields a different complete file. Set `LLAMA_TEST_BLOOM_HF` to the `official-hf` directory and `LLAMA_TEST_BLOOM_OUTPUT_ROOT` to a fresh output root.

The native synthetic pipeline contains 113 cases: 84 accepted complete files totaling 1,066,141,612 bytes and 29 original rejections per host. Separate captures cover metadata bytes, tokenizer classes, tensor aliases, 1,308 family filter decisions, all 1,112,064 Unicode scalars and 1,922 complete Split/encoding/decoding requests.

- [Native capture seal](capture-seal.json) records the clean product base before the initial captures and implementation.
- [Additional regex seal](regex-capture-seal.json) and native filter reports document later independent captures before their respective corrections.
- [Final source and fixture guard](rust-input-guard-v2.json) contains all 205 checked inputs. Both final runners verified the exact set before and after their commands.
- [Rust Windows report](rust/windows/report.json) and [Rust GNU report](rust/linux/report.json) retain exact commands and raw logs.
- [Initial development failures](initial-failures.json) are retained with their logs and explanations. Reference expectations were not rewritten to match Rust.
- [Official acquisition](acquisition.json) records the fixed revision, seven file identities, official URLs and RAIL license.
- [External tool identities](external-tool-identities.json) identifies reference and validation tooling without maintaining Python or C++ in the product.

The compact fixture pack contains synthetic tensor inputs and official tokenizer data; it does not contain trained model weights. Retain the [BLOOM notices](../../../crates/convert/BLOOM_NOTICES.md), complete original model card and BigScience BLOOM RAIL license. See [BLOOM conversion contracts](../../../docs/BLOOM_HF_CONVERSION.md) for exact metadata, tokenizer, error and publication boundaries. The explicitly named source-compatible nested-array writer can reproduce Python outputs that strict GGUF readers reject; strict Rust writers and readers retain that rejection.

This package is a bounded converter milestone. It does not close the full rewrite's release gates.
