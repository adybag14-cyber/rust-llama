# Native quantize CLI

The llama-model-quant package provides a native Rust llama-quantize binary.
It ports tools/quantize/quantize.cpp, common/imatrix-loader.cpp, and the scalar
metadata parser in common/common.cpp from the frozen source. It calls the Rust
model-file quantizer and scalar codecs. No original executable, Python program,
C++ library or model download is needed to run the binary or ordinary tests.

The source archive is pinned by SHA-256
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
The original source and independently generated synthetic data are MIT-licensed.

## Running the tool

~~~console
cargo run --release -p llama-model-quant --bin llama-quantize -- model-f16.gguf model-q4.gguf Q4_K_M 4
cargo run --release -p llama-model-quant --bin llama-quantize -- --dry-run model-f16.gguf Q4_K_M
cargo run --release -p llama-model-quant --bin llama-quantize -- --imatrix importance.gguf model-f16.gguf model-iq.gguf IQ2_XXS 4
~~~

Options precede the input model. The positional form is
INPUT [OUTPUT] TYPE [NTHREADS]. Omitted output names use
ggml-model-TYPE.gguf in the input directory. Type names are case-insensitive;
the default name retains the selected source-table spelling, so Q4_K and
Q4_K_M select the same mode but different default filenames. Numeric IDs
select the first matching table row. COPY preserves tensor types.

| Option | Result |
| --- | --- |
| --allow-requantize | Permit conversion of already quantized tensors. |
| --leave-output-tensor | Preserve output.weight. |
| --pure | Disable mixed tensor-type presets. |
| --output-tensor-type TYPE | Override the output-weight dtype. |
| --token-embedding-type TYPE | Override the embedding dtype. |
| --tensor-type PATTERN=TYPE | Append a tensor-name regex override. |
| --tensor-type-file FILE | Read whitespace-separated overrides. |
| --prune-layers L0,L1,... | Sort and deduplicate layer IDs, then prune and remap them. |
| --imatrix FILE | Read legacy or GGUF importance data. |
| --include-weights TEXT | Include matching importance names; repeatable. |
| --exclude-weights TEXT | Exclude matching importance names; repeatable. |
| --override-kv KEY=TYPE:VALUE | Append an int, float, bool or str metadata override. |
| --keep-split | Preserve source shards and source split-name behavior. |
| --dry-run | Calculate the layout without writing output. |
| --max-buffer-size MiB | Set the slab target, subject to the hard working-memory bound. |

Include and exclude filters cannot be combined in the CLI. They match
case-sensitive substrings, independently of tensor override regexes.
--help, unknown options and malformed option arguments print usage and exit
1, matching the original. Successful conversion and dry-run exit 0.
Diagnostics contain useful errors and sizes; compiler banners, timing and
progress text are not a byte-stable interface.

The file modes, admitted model families, type heuristics, payload restrictions
and staged publication contract are described in [README.md](README.md).
Requantization and pruning can reduce model quality. A successful metadata
override can produce a file whose metadata no longer describes its tensor
geometry, as in the original.

## Source parsing and ordering

The cli::parse_command API receives arguments without argv[0] and returns an
owned command. Its order matches the original: options and their files are
processed, incompatible filters are rejected, importance data is loaded and
normalized, generated importance metadata is appended, then positional
filenames, file type and thread count are interpreted. A later invalid type
therefore cannot hide an earlier invalid importance file.

The scalar helpers preserve C-string prefixes and decimal-prefix parsing,
including source acceptance of 15suffix where stoi reads a numeric prefix.
The type-name table has 39 rows for 35 file IDs plus aliases and COPY. Dtype
options accept ggml names, not numeric IDs. Tensor patterns are lowered with
ASCII byte rules. Pruning accepts nonnegative signed-32-bit prefixes; empty
components and overflowing IDs fail.

Metadata keys are shorter than 128 bytes, string values at most 127 bytes.
Empty keys are accepted by the scalar helper and terminate the source C
override array when preparing the command. Later user overrides and generated
importance overrides are consequently ignored. Integer metadata parsing uses
the measured native long width: 32 bits on MSVC and 64 bits on GNU. Integer
overflow has no portable C atol/atoi result and returns a checked Rust error;
native saturated/wrapped observations remain recorded in the fixtures.

Float parsing includes decimal and hexadecimal prefixes, signed zero, infinities
and the measured host NaN payload conventions. Subsequent model metadata and
payload admission still applies. One reviewed correction is explicit:
UCRT loses the carry into DBL_MIN for 0x1.fffffffffffffp-1023, returning a private
F64 zero. Rust uses correct IEEE rounding on both hosts; GNU independently
produces that result. The original MSVC observation is retained, and only that
private intermediate is excluded from a bit-exact claim. Four original complete
files cover the positive and negative boundary and its lower neighbor. Their
emitted F32 metadata is identical, including the sign bit of zero. See
[rounding-correction.json](tests/fixtures/cli/rounding-correction.json).

The source's unusual split-name rule is retained: if an explicit output
contains .gguf anywhere, --keep-split removes its last five bytes before
appending shard suffixes. A resulting incomplete UTF-8 path is outside the
portable Unicode path domain and fails explicitly. Options stop at the first
non-option; surplus positional arguments after the thread count are ignored.
Dry-run does not examine or change its destination.

## Importance data

The imatrix::prepare_imatrix API returns normalized owned F32 rows, dataset
byte strings and the source chunk count. An empty path returns an empty map and
chunk count -1. Legacy data stores signed-32-bit entry/name/value counts and
little-endian F32 sums. Positive call counts divide each F32 value; nonpositive
counts leave it unchanged. Repeated names replace earlier rows. Optional
trailing chunks and dataset bytes preserve the original partial-read behavior.

GGUF importance files need the metadata keys imatrix.datasets,
imatrix.chunk_count and imatrix.chunk_size. The latter two and optional
imatrix.n_layer_nextn use UINT32. The datasets key must be present, but a
non-array value contributes no dataset names, as measured in the original.
Tensor groups pair .in_sum2 and .counts; optional .in_sum must have the same
element count. Sum/count tensors are F32. Counts are rounded away from zero at
half-integers to the host native-long range. A positive count divides its expert
row; a nonpositive count fills that row with ones. Floor-sized rows leave any
remainder as zeros. Final tensor padding is required by the original importance
GGUF loader even when a generic GGUF reader can accept an unpadded last tensor.

Normalization and validation precede filtering. Excluding a malformed group
does not make the file valid. Empty count vectors, nonfinite sums/counts,
out-of-native-long counts, truncated data, missing group members and unsafe
activation reinterprets return checked errors. Original process-abort
observations remain explicit fixtures. Legacy and tensor-group names use
C-string prefixes and must be valid UTF-8 in the owned Rust map.

A nonempty filtered map appends generated metadata in source order: the
importance path, the first dataset if present, entry count, and positive chunk
count. Path/dataset values are truncated to 127 C-string bytes, including
possible partial UTF-8 metadata bytes. An empty filtered map retains a null
importance pointer and adds no importance metadata.

The dry-run importance warning comes from the original preliminary tensor
scan. An eligible IQ tensor still requires a matrix when its source and target
dtypes are identical. COPY and other excluded tensors do not. This package
corrects that report aggregation with captured same-IQ, COPY, explicit-F32,
with-matrix and no-matrix cases; it does not change type selection or codecs.

## Bounds

Defaults are explicit and adjustable through CliLimits and ImatrixLimits:

| Resource | Default |
| --- | ---: |
| Arguments | 4,096 |
| Total argument bytes | 1 MiB |
| Each scalar/path argument | 16 KiB |
| Tensor-type file bytes | 16 MiB |
| User/generated override entries | 4,096 |
| Importance file bytes | 1 GiB |
| Importance groups/legacy entries | 65,536 |
| Cumulative encoded sum/count values | 128 Mi values |
| Importance name/filter bytes | 4,096 |
| Dataset bytes | 16 MiB |
| Dataset entries | 4,096 |
| Importance GGUF metadata bytes | 16 MiB |
| Filter patterns | 4,096 |
| Conservative substring-filter work | 256 MiB |

Counts and byte arithmetic are checked before corresponding allocations.
Legacy duplicate rows still consume the cumulative value budget. The retained
tensor-type-file handle is sized once, its checked length bounds reservation,
and a capped read detects growth beyond the file limit. Filter work is charged
before filtering. These bounds are separate from the core quantizer's 256 MiB
live-slab bound, worker cap, scalar codec workspaces, allocator overhead and OS
cache; they do not claim a total resident-memory limit.

## Independent verification

Captures preceded Rust implementation on Windows MSVC and Linux GNU. The
original binaries and direct helper call the unchanged frozen source. The
preimplementation seals rehash the entire 4,082-file reference tree, producer
binaries and native report identities, and record a clean product base without
these Rust modules.

Each host has 188 actual executable cases and 517 direct helper observations.
Executable cases cover every source file mode, successful Q1 handling of an
otherwise unsupported odd-width tensor, aliases, default names, options and
error ordering, legacy/GGUF importance, filtering, metadata, pruning, split
files, same-file/hard-link/symlink paths, destination preservation and dry-run.
There are 117 complete in-domain MSVC outputs and 118 GNU outputs; the
difference is the documented native-long metadata range. The binary fixture
deduplicates 99 complete input/output blobs into 2,010,682 bytes. Tests compare
every output byte, not only a prefix, parsed summary or digest.

Helper cases compare accepted typed parser values, normalized F32 bits,
dataset ordering and chunk counts, and require checked errors for original
rejections or disclosed undefined domains. Original missing-final-padding
captures were retained when a later correctly padded GGUF cohort was added.
No expectations were generated by the Rust implementation.

~~~console
cargo test -p llama-model-quant --test cli_native_reference -- --nocapture
cargo test --release -p llama-model-quant
cargo clippy -p llama-model-quant --all-targets -- -D warnings
~~~

The compact fixture provenance is
[tests/fixtures/cli/provenance.json](tests/fixtures/cli/provenance.json).
Its separate integrity manifest is
[docs/model-quant-cli-fixture-manifest.tsv](../../docs/model-quant-cli-fixture-manifest.tsv).
Original producers, source libraries and full native logs remain external;
committed validation receipts are under
[evidence/2026-11-11/model-quant-cli](../../evidence/2026-11-11/model-quant-cli).
This CLI establishes the bounded file-quantization contract above. It does not
close full architecture, backend, ISA, performance or complete-port gates.

## File identity and checked safety corrections

Write-time input/output equivalence uses actual OS file identity. On Windows,
the small existing platform boundary calls GetFileInformationByHandleEx with
FileIdInfo, comparing the 64-bit volume number and complete 128-bit file ID.
The query asks for zero payload access, retains ordinary read/write/delete
sharing, follows symbolic links, and performs no content comparison or locking.
Unix compares device and inode. The model-quant crate retains forbid(unsafe_code);
the reviewed OS call stays in llama-platform-loader.

An independent program compiled against the installed Windows SDK verifies
the ABI: FILE_ID_INFO is 24 bytes with alignment 8 on the pinned x64 host,
volume and ID offsets are 0 and 8, the ID contains 16 bytes, BOOL is 4 bytes,
HANDLE is 8 bytes, and FileIdInfo is 18. Hard-linked and symlinked paths share
the same native identity, while a distinct file with identical bytes does not.
The native receipt also retains the installed header and producer hashes.
See Microsoft's [GetFileInformationByHandleEx documentation](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex)
and [FILE_ID_INFO definition](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_info).

The CLI performs its initial equivalence check in the source's position after
argument preparation and before conversion, and skips it for dry-run. The
library checks every output against every input shard before writing and
again before publication. These checks are point-in-time identity queries;
they are not a guarantee against a hostile filesystem path replacement after
the check. Existing source-handle retention and staged publication still apply.

Independent original public-C calls used fresh disposable copies only.
MSVC rejects hard-link, same-path and symlink writes and preserves the input.
GNU's hard-link write returns success while modifying the aliased input;
same-path and symlink writes modify it and terminate with SIGBUS. Rust retains
checked rejection for these unsafe GNU cases and extends real alias rejection
to Windows. This is an explicit input-safety correction, not a source-parity
claim for destructive behavior.

Both original hosts accept dry-run with an aliased destination and with an
existing directory as destination. The Rust library consequently skips alias
and directory rejection when no publication is requested. Direct core
regressions cover all six cases per host, preserve every input, and compare
the successful distinct-file control against the complete original bytes.
