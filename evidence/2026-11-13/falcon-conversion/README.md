# Falcon converter verification

The Falcon conversion component passed the checks below on the pinned native MSVC and GNU hosts. This is a selected-component record. Full-port gates and pretrained Falcon model acceptance remain open.

## Original-source comparisons

Each host independently captured 117 requests before Rust comparison: 86 accepted complete files (202,713,961 bytes) and 31 original rejections. Alias checks cover 510 native lookups. Tokenizer captures agree over all 1,112,064 Unicode scalars and 8,745 split/encode/decode requests.

## Rust execution

| Host | Ordinary converter debug | Falcon release | CLI |
| --- | --- | --- | --- |
| windows | 58 passed, 9 prerequisite tests ignored, 0 failed | 3 passed, 0 failed | Dry run publishes nothing; Q8_0 output exactly matches original |
| linux | 58 passed, 9 prerequisite tests ignored, 0 failed | 3 passed, 0 failed | Dry run publishes nothing; Q8_0 output exactly matches original |

Both hosts passed formatting and strict Clippy. Every final command has an argv, return code, elapsed time and capture-time stdout/stderr identity in its report. The CLI output identity is checked independently of exit status.

The guard covers 212 converter files, workspace Cargo/toolchain inputs and the dedicated documentation/fixture declarations. It matches before and after each host's run and during packaging. Other dependency crates and complete repository identity are deferred to integrated validation.

Guard SHA-256: 4959526a45130afcd1a558bfc9b34a7b1199e5a71677d4b41ee820a2786282de

The original pipeline reports and full external case records have separately labeled packaging-time identities. Complete GGUF output identities were recorded at capture time and rechecked during packaging. Large source output files and one-off foreign-language oracle programs remain outside the delivered repository; the independently captured input pack and complete output identities are maintained as test data.

## Retained earlier attempts

The first punctuation capture used the Python binding's wrong spelling for the enum value and failed before producing comparison data; the final v2 capture uses the accepted lowercase constructor spelling. An initial Linux validation wrapper omitted the script's tag argument and exited before running any Cargo command. Both actual failed receipts are retained. Short-lived WSL startup attempts were replaced by a durable Windows parent for successful native Linux execution. None of these failed attempts supplies final fixture or validation success.

The official model source is Falcon 7B revision ec89142b67d748a1865ea4451372db8313ada0d8. Only small sidecars were acquired; no pretrained Falcon tensor execution is claimed. The source-default and architecture boundaries are documented in docs/FALCON_HF_CONVERSION.md.
