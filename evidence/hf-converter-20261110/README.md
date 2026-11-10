# Dense Llama converter validation, 10 November 2026

These immutable logs and JSON records validate the recovered converter in its isolated authoring workspace. They bind the exact selective handoff that was subsequently integrated into the main repository. They are component evidence; integrated whole-workspace checks are recorded separately.

The frozen source archive SHA-256 is 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.

- summary.json reports the final checks and exact pretrained output hashes.
- report.json records every final command, exit status, duration, log identity and complete before/after source identities.
- source-before.json records the initial source map separately.
- windows-debug-clippy.log preserves the preceding passing Windows debug tests, strict Clippy and actual dependency cache audit.
- The eight named stage logs preserve the complete final fmt, release, GNU debug/cache/Clippy, and Windows/GNU pretrained comparison output.
- integration-review.json records the later read-only review of the applied shared APIs, exact imported fixture bytes, lockfile, CI bootstrap and documentation.

Ordinary focused tests passed 79 cases on Windows and 81 on GNU, with three explicitly external tests ignored in ordinary runs. The complete pretrained conversion test was then explicitly run on both hosts. It matched all 686581536 output bytes per host across auto/F16/Q8_0 and checked the original input hashes. Model weights, generated GGUF outputs, executables, caches and original Python/C++ implementations are not included here.

The original handoff manifest SHA-256 is 044e6be5e63a26453cc8eeac66a7c09139e093e811d08ef19cc9d1fb0cc8ef0b. Every one of its 115 additions matched the final tested source; source remained unchanged throughout final validation.

Paths in the execution records name the actual original validation locations. They are provenance and are not required local paths for ordinary repository tests.
