# Original split and merge with arbitrary byte strings

The unchanged original CLI ran 3,196 cases on each native profile before the
Rust split/merge byte-string migration: 3,176 successes and 20 failures. Inputs
reuse the independently sealed 1,605-case reader/boolean corpus, including all
byte values and first-prefix tensor aliases with distinct shapes/payloads.
Successful post-operation files agree byte-for-byte across both profiles and
cover 1,438,016 file bytes per profile. Each input is split one tensor per file;
multi-tensor aliases are also grouped; successful split products are merged.

The full original JSON reports and byte pools preserve all raw transcripts,
input/output files, nonzero process codes and failed partial files. The TSV/pool
adapter selects observations without running Rust or deriving expectations from
its output. Ordinary Rust tests run library file operations and compare every
successful file plus checked errors. The existing general CLI corpus separately
checks argument parsing, successful transcripts, dry runs and cleanup behavior.
