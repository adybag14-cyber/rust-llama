# Dense Qwen converter evidence — 11 November 2026

The native Rust converter adds dense Qwen2/Qwen3 to the existing dense Llama path.
The authority is the supplied archive SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Original Python/native environments, capture programs and pretrained weights
remain outside the maintained Rust product.

## Compared behavior

| Independent cohort, per native host | Acceptance criterion |
| --- | --- |
| 112 main complete-file cases | 104 exact files / 25,723,552 bytes; eight native rejections |
| 32 configuration-fallback cases | 32 exact files / 5,318,912 bytes |
| Eight model-type presence cases | Eight exact files / 1,329,824 bytes |
| Official Qwen2.5 and Qwen3 checkpoints | Six complete F16/Q8_0/auto-BF16 files / 6,342,831,936 bytes |
| 106 tokenizer-class cases | 82 accepted, 24 rejected; 5,412 exact encoding/decode requests |
| 48 direct-prefix constructor cases | 40 accepted, eight rejected; 160 exact requests |
| 34 absent-sidecar and Sequence cases | 34 accepted; 136 exact requests |

All complete-file and tokenizer expected data comes from the original converter
or its pinned libraries. Windows and GNU profiles were captured separately.
Every input/output blob or retained tokenizer input has its size and SHA-256
recorded. No expected data was replaced by Rust results; no tolerance was widened.

## Reports and chronology

- `receipt.json` and `retained-files.json` record the earlier complete-file
  qualification, including all twelve retained official Rust output identities
  across the two profiles. Its 144-case total is the scope at that capture.
- `receipt-final-slice.json` and `retained-files-final.json` add the eight
  model-type cases and the class/prefix matrices. The final ordinary complete-file
  scope is 152 cases: 144 accepted files, eight rejections and 32,372,288 exact
  bytes per profile.
- `original/`, `config-type-original/` and `class-prefix-original/` retain
  original reports and independent capture receipts.
- `validation/` retains successful and failed development attempts with their
  own source manifests and raw logs.
- `fixture-registration-final.json` records the 27 root-integrated fixture
  additions; the workspace continuation manifest performs the final integrity
  check.

The successful earlier complete-model runs are Windows `v7` and GNU
`v9-final`; GNU ordinary/focused gates also ran in `v7-core` and
`v10-flags`. Windows `v11-class-prefix` passed ordinary debug/release and strict
Clippy after the class and prefix corrections; `v12-boundaries` passed the
expanded debug suite and strict Clippy. The [final integrated validation](../continued-port/README.md)
rechecks the complete latest workspace, source identity, fixture registration,
actual native C clients and official complete conversions on both hosts.

## Preserved findings

The logs distinguish an initial reference-harness omission, the fixed generation
of a Llama-only RoPE tensor in Qwen output, the initially unsupported Qwen Split
expression, a transient Clippy failure, and interrupted GNU orchestration.
The later class audit exposed the need to preserve absent versus null root
model type and successful root AutoConfig precedence. The prefix audit found
two additional bugs: missing-sidecar defaults were skipped and a flattened
one-child Sequence was mistaken for a direct ByteLevel root. These were fixed
against independently captured expectations.

A detached GNU attempt was interrupted by a WSL shutdown. Successful later runs
retain a foreground WSL client through a durable host job. That interruption
is not treated as a product failure or a successful test.

See [the implementation contract](../../../docs/QWEN_HF_CONVERSION.md) for exact
aliases, source profiles, resource bounds and unsupported variants. Complete
conversion, all-model, all-backend and full-release gates remain open.
