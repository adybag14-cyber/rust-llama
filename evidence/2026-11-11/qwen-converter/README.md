# Dense Qwen converter qualification — 11 November 2026

This evidence records the native Rust Qwen2/Qwen3 converter extension against the supplied llama.cpp archive, SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. The product reads and writes model data in Rust; independent Python and native reference tools remain outside the repository.

## Independent expected data

The ordinary complete-file cohorts contain 152 cases per native profile: 144 accepted files, eight original rejections, and 32,372,288 exact successful output bytes. The three separately retained cohorts cover tensor/format/layout variants, configuration fallback edges, and explicit/absent/null/unknown model types.

Two unmodified official checkpoints produce six complete F16, Q8_0 and auto/BF16 files totaling 6,342,831,936 bytes per profile. All source and output files are hash-checked. The complete file identities and pinned revisions are in [the converter implementation contract](../../../docs/QWEN_HF_CONVERSION.md); model weights and generated outputs remain external.

Tokenizer preparation has additional native captures for 106 class-selection cases (82 accepted, 24 rejected), 48 direct-prefix cases (40 accepted, eight rejected), and 34 missing-sidecar/Sequence cases. These produce 5,708 accepted ID/decode requests per profile. Literal native paths, sidecar presence, source package identities and unchanged tokenizer inputs are retained with the fixture provenance.

## Validation history

| Retained phase | Result and scope |
| --- | --- |
| Windows `v7` | Converter debug/release, strict Clippy and all six complete official model outputs passed. |
| GNU `v7-core` / `v9-final` | Formatting/debug/Clippy followed by release and all six complete official model outputs passed. The final full-model run retained a foreground WSL client. |
| Both hosts `v10-flags` | The existing native generic-flag corpus also passed through actual Qwen class preparation. |
| Windows `v11-class-prefix` | The model-type and class/prefix corrections passed the complete converter debug/release suites and strict Clippy. |
| Windows `v12-boundaries` | The final missing-sidecar and Sequence regressions passed with the ordinary debug suite, formatting and strict Clippy. All 157 recorded converter inputs were rechecked unchanged during evidence collection. |
| Final integrated workspace | [The continuation evidence](../continued-port/README.md) records the final shared-source Windows/GNU suites, complete pretrained conversions and native clients. |

`receipt.json` retains the earlier complete-model qualification and its 144-case ordinary scope at that time. `late-resolution-receipt.json` records the subsequent 152-case scope and class/prefix fixes. Each report describes its actual source snapshot; earlier successful model runs are not represented as testing later edits.

The initial reference harness error, accidental Llama-only RoPE tensor emission for Qwen, unsupported Qwen tokenizer expression, and interrupted GNU orchestration attempts are retained alongside their corrections. The original expected files were not replaced with Rust output, and no tolerance was widened. The final boundary captures were independently checked on both original environments before replay.

This qualifies the documented dense path. Other architectures, specialized Qwen variants, full backend integration and the complete-port release gates remain open.
