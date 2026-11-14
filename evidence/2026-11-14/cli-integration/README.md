# Native CLI continuation acceptance

This package records the actual release-binary CLI replay on Windows MSVC
and GNU/Linux. The external v2 driver reused the verified root binaries;
it did not rebuild product code. It preserves each complete command stdout,
stderr and report, together with the durable driver stdout/stderr.

Each host completed 15 commands: help, six generation commands, six full-logit
commands, and two explicit rejection commands. The comparisons cover Qwen2
F16/Q8_0 and Qwen3 Q8_0 on CUDA, plus tiny StarCoder F16/Q8_0 and StarCoder2
3B Q4_K_M on the CPU.

Generation matched the captured original bytes and expected token statistics.
The logit checks compared nine full vocabulary rows, 1,059,072 values per host,
using the fixed bound |actual - reference| <= 0.0002 + 0.00002 * |reference|.
The per-model observed maximum absolute errors are in acceptance.json.
The first StarCoder prompt is Hello with one token, so each StarCoder check
uses one row; each Qwen check uses two rows.

The two rejection commands verify an insufficient CUDA buffer budget and the
explicitly unsupported StarCoder2 CUDA path. CUDA generation statistics report
the selected device, owned allocation bytes, and actual kernel dispatches.
CPU generation reports no device allocation or dispatches. Each root binary
retained its complete SHA-256 identity throughout its own replay.

## Review and scope

The retained v1 review was read-only and v1 was never executed. The exact-text
v1-to-v2 delta permits only a fresh output directory, removal of an unused
stop field, and recorded per-command timeouts: 300 seconds for the three
StarCoder2 commands and 120 seconds for the other twelve commands.
No model, reference row, tolerance, or test case changed.

This 15-command continuation does not test stop sequences and makes no new
stop-coverage claim. It is local native CLI acceptance, not a hosted CI run.
StarCoder2 used its explicit 16 GiB decoded-weight allowance; this does not
measure peak process memory. These focused results do not close the full
Rust-port release gate.

## Integrity

The copy ledger records every original external receipt path, byte length,
source SHA-256 and matching destination SHA-256. Empty files are preserved.
The manifest lists hashes of all package payload files except itself.
The executable Python drivers remain outside the Rust product.

This package is adjacent to the separately sealed cuda-runtime evidence.
Packaging this continuation did not alter that earlier package.
