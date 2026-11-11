# Frozen quantize CLI and importance fixtures

These observations come from the unchanged frozen llama.cpp source archive:
SHA-256 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
Original implementation and synthetic data are MIT-licensed; the exact source
LICENSE is retained here. No model weights or trained-model license is involved.

The original executables and direct helper ran on both native Windows MSVC and
Linux GNU profiles before the Rust CLI/scalar/imatrix modules existed.
Preimplementation seals verify the complete 4,082-file source tree, producer
binary identities and clean product base 40c6573994019bbab9aadff0992988ea3b0c0dfd.
Full native programs, builds, scripts and case directories remain outside the
repository under the external roots recorded in provenance.json.

Each executable manifest records 188 actual CLI requests, exit class,
usage/dry-run/importance-warning behavior, and complete input/output file hashes.
Each helper manifest records 517 direct calls to original scalar parsers and
prepare_imatrix, including exact F32/F64 bits, dataset order, count normalization
and error/abort observations. Source suffixes, aliases and metadata order are
retained. A source rejection is not relabeled as successful conversion.

The 99 blobs in files.bin contain every complete executable input/output needed
by the in-domain comparisons, deduplicated by SHA-256. files.json supplies
checked offsets, lengths and hashes. Its total size is 2,010,682 bytes.
Tests verify pack coverage, every blob hash and every accepted output byte.
There are 117 in-domain whole-output comparisons on MSVC and 118 on GNU.
The difference is the explicitly measured C long width for integer metadata.

The rust_boundary fields disclose undefined C atol/atoi overflow and invalid
lround domains. Rust requires bounded errors for those cases while retaining
the original saturated, wrapped or nonportable observation. The
rust_correction fields identify the one reviewed UCRT hexadecimal subnormal
carry defect and preserve its original private F64 result. Correct IEEE
rounding is used on both Rust hosts. The separate rounding-correction receipt
retains the native GNU result, four complete original files, and F32 signed-zero
equality. No other private F64 value is silently changed.

The initial missing-last-padding GGUF captures remain unchanged. Properly
padded cases form a separate subsequent native cohort; the Rust loader must
preserve the source's final-padding requirement.

The identity-native and identity-directory addenda call the original public C
model quantizer on new disposable files. Pinned inputs are verified unchanged.
Original GNU write-time alias corruption and SIGBUS observations are retained
as deliberate checked-error safety corrections. Both hosts accept harmless
alias/directory dry-run planning. The Windows receipt additionally records a
fresh program built against the installed SDK that verifies the real
FileIdInfo ABI and file-ID relationships before the Rust OS helper was added.

The complete contract, limits and reproduction commands are in
[CLI.md](../../../CLI.md). The separate root integrity manifest is
[model-quant-cli-fixture-manifest.tsv](../../../../../docs/model-quant-cli-fixture-manifest.tsv).
The existing core quantizer fixture manifest remains unchanged.
