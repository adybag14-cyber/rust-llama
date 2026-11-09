# Frozen GGUF split/merge command corpus

These are independent captures from the original `tools/gguf-split/gguf-split.cpp` in source archive SHA-256 `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. Each original host run contains 103 cases: 64 successful process exits, 39 nonzero exits and no timeouts. All successful after-file bytes agree across Windows MSVC and Linux GNU.

`native-data.bin` stores 245 complete unique blobs: initial GGUF files, original output files and raw stdout/stderr. Blobs are interned by SHA-256 only after exact byte equality. `native-reference.json` maps each case and profile to offsets, lengths and hashes. The two complete original reports and the original replayed input index are copied without modification. The original input constructor was replayed without executing either implementation; all 48 initial blobs, totaling 2,053,198 bytes, match the filenames, sizes and hashes captured on both hosts, including files the original later deleted or overwrote.

The black-box Rust test runs the actual `llama-gguf-split` binary in isolated temporary directories. It verifies complete after-file names and bytes for successful cases, including metadata order, every scalar/array GGUF type, NaN payload bits, all 36 storage IDs, version conversion, source tensor-shape canonicalization, custom alignment, split naming, C-string NUL truncation, overwrite behavior, dry-run behavior, merge metadata rules and final deletions. It does not require a stricter GGUF reader to accept an original-invalid output.

Seven successful output records per host are rejected by the original inspector: first shards for alignments 1, 8, 16, 64 and 256, and outputs for shapes `[2,0]` and `[2,1,0,1]`. Their bytes remain exact regression expectations.

## Deliberate checked-error and safety contracts

The Rust test expects 63 successful cases and 40 checked errors. One observed native success, `parse-junk-size`, depends on reading an uninitialized integer after a failed `sscanf`; Rust must reject it. `parse-empty-size` accesses `std::string::back()` on an empty string. Zero/junk tensor limits with multiple tensors trigger native integer division faults. These four undefined-behavior cases are recorded without treating the observed crash or accidental success as a required Rust behavior. Five original assertion failures also become checked errors. A panic or abort does not satisfy the normal Rust error-exit assertion.

Defined source quirks remain tested: negative tensor limits use the original divisor behavior, numeric prefixes such as `2tail` and `1.5M` are accepted as the original parses them, a zero tensor limit succeeds for zero or one tensor because modulo is short-circuited, and the advertised `-h` is rejected by the original option-loop condition. Split dry-run can succeed on a truncated payload; merge dry-run reads payloads and fails on the captured truncated secondary.

Failed partial output is not portable even between the two originals:

| Case | Windows partial output | GNU partial output |
| --- | ---: | ---: |
| `truncated-input` | 1,504 bytes | 1,312 bytes |
| `merge-truncated-secondary` | 1,504 bytes | 0 bytes |

Both versions are preserved in the byte pool. The Rust test requires a normal error, intact inputs and only bounded permitted partial-output names. It does not require matching those failed partial bytes.

Successful stdout/stderr are exact after executable-path and CRLF normalization, with three explicit exceptions:

- `--version` identifies the Rust implementation and its own build.
- The `--delete-splits` help line says: `delete split files only after the complete merge is flushed and synced`.
- Successful merge-deletion log lines may move after completion; their contents and relative order, and all other log contents/order, remain checked.

The Rust delete operation is intentionally stronger: input splits are deleted only after the complete output is flushed and synced. A separate regression adds `--delete-splits` to the unchanged original truncated-secondary inputs and requires every input to survive failure. This safety test is distinguished from the 103-case native corpus; the original incremental deletion behavior is not claimed as an input-preservation oracle.

All native sources, executables and capture/packing scripts remain outside the product. These fixtures carry the existing upstream MIT provenance; no pretrained model weights are included.
