# Owned Safetensors checkpoints

`llama-checkpoint` is the native Rust foundation for the frozen converter's local
Safetensors inputs. It parses headers, retains file owners, exposes checked raw
ranges, and assembles indexed checkpoints in the original contribution order.
It does not yet perform architecture conversion, PyTorch tensor construction,
numeric dtype decoding, remote downloads, or pickle loading.

## Compatibility authority

The primary oracle is the exact input ZIP
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`:

- `gguf-py/gguf/utility.py::SafetensorsLocal`, SHA-256
  `da920e2c62166ec9f30e407e9fb35ec29fda671310ffbf0f8ce856b62f1c70d7`.
- `conversion/base.py::ModelBase.index_local_tensors`,
  `get_model_part_names`, and `filter_tensors`, whole-file SHA-256
  `3ac8c8eb7e02b9fbfa594e887160b7c45a906273f672d286e0e1a10f245a7d60`.

The independent source harness imports the unchanged utility module and compiles
the three unchanged method ASTs without unrelated converter imports. It reads the
original lazy generators' captured LocalTensor objects without invoking Torch.
The pinned NumPy dependency is 2.4.6 for Python 3.14 on Windows and 2.2.6 for
Python 3.10 on GNU/Linux, matching the source requirements.

Safetensors 0.8.0 is a separately pinned comparison. Its stricter canonical-format
rules do not silently replace the source's more permissive raw-reader contract.
All reference Python, packages and source-extraction scripts remain outside the
product. No Python or reference implementation is linked by this crate.

## APIs and ownership

- `SafetensorsFile::open(path, limits)` opens an explicitly trusted single-file
  path. It may follow a caller-chosen symlink.
- `SafetensorsFile::from_file(file, limits)` accepts an already-opened file
  capability and applies the same parser and limits.
- `Checkpoint::open_directory(root, options, limits)` resolves discovered or
  index-selected files beneath a retained root directory with the checked path
  policy below.
- `Checkpoint::from_open_files(files, index_bytes, options, limits)` accepts
  labelled, already-opened capabilities. It performs the same header parsing,
  index selection, global-name validation, filtering and overwrite rules.
  Labels are caller-provided identifiers; this API does not resolve them as
  untrusted paths.
- `RawTensorInfo` exposes original dtype and shape values, source-computed
  absolute offset and length. `typed_layout()` separately validates a dense
  storage layout.
- `TensorReader` borrows its owner and supports bounded streaming Read/Seek.
  `read_tensor` applies a separate allocation-size limit.
- `Checkpoint::tensors()` follows first insertion order while selecting the
  last contributing value. `find` and `find_str` use a sorted lookup.

A parsed file retains the same File used to read its header. Raw reads never
reopen a tensor pathname. This deliberately strengthens the source's behavior:
the original closes its header owner and later reopens the stored filename.
A captured path-replacement control records that difference.

Windows paths opened by this crate allow read sharing and deny write/delete
sharing while retained. The explicit root directory allows sharing so ancestors
are not locked. Supplied File capabilities retain the caller's original sharing
configuration. On Unix, an owned descriptor does not prevent concurrent writes;
callers must keep file contents stable. Positioned reads detect short reads after
truncation, and no mmap operation can fault on truncated mapped memory. The API
does not claim an atomic snapshot under concurrent in-place modification.

## Preserve contribution order for converters

Every selected file remains available through `file_count()` and `file(i)`,
in sorted selected-filename order. Each file exposes its original tensors in
Python codepoint sort order. This is the full, uncollapsed contribution stream.

Family-specific filtering must run on this stream before the family overwrite
map. Filtering only the already-overwritten global map is incorrect. For example,
three shards can contain original A, a name that renames to A, then original A
again. The source selects the third contribution. Collapsing original names first
and then filtering can incorrectly select the second.

`NamePolicy::ConverterBase` applies the frozen ModelBase substitutions at the
correct point. `NamePolicy::Unchanged` leaves names unchanged but still resolves
duplicate original names in its global view; it is not a substitute for the full
per-file stream when adding Llama/TextModel or other family filters. Architecture
converters can read each retained file/tensor directly without reopening paths.

## Source behavior retained

| Input behavior | Raw-reader result |
| --- | --- |
| Duplicate JSON keys | Last value, first key insertion position |
| Header tensor order | Sorted by Python string codepoints |
| Unknown or non-string dtype | Retained; raw access can succeed |
| Array, string, or object shape | Python tuple/iteration semantics retained |
| Negative dimensions or mismatched shape/bytes | Retained; typed layout rejects invalid layouts |
| Gaps, overlaps and trailing payload bytes | Permitted for safe raw ranges |
| Negative relative data offsets | Permitted when the resulting absolute range is inside the file, including header bytes |
| Extra offset array entries | Ignored after the first two, as in the source |
| Float offsets | Metadata arithmetic retained; raw access returns a checked error |
| Nonfinite JSON floats and bounded arbitrary-size integers | Retained; impossible raw ranges fail when accessed |
| Escaped unpaired surrogates and NUL tensor names | Retained losslessly in Text |
| Invalid UTF-8 header | Rejected, matching the source's explicit UTF-8 decode |
| Arbitrary `__metadata__` | Ignored by loading and available as a separate raw value |
| Index filename assignments | Select a sorted unique filename set; do not enforce each name's particular assigned file |
| Index tensor names | Compared with the global original-name set before filtering |
| Empty weight_map | Selects no files, even if discoverable files exist |
| Later shard/name collision | Last value wins without moving first insertion order |
| Mistral mode | Ignores the index; use the source `consolidated` prefix |
| PyTorch pickle input | Explicit Unsupported error |

Text uses canonical extended UTF-8 to preserve Python's escaped surrogate
codepoints. `as_str()` returns None when scalar Unicode is required but unavailable;
`as_wtf8()` and `codepoints()` retain the original value. This is separate from
the remaining arbitrary-byte GGUF string work.

DType recognizes the 22 storage identifiers in the pinned secondary format.
`is_converter_dtype()` identifies only the frozen converter's 13 names:
F64, F32, BF16, F16, I64, I32, I16, U8, I8, BOOL, F8_E4M3, F8_E5M2 and F8_E8M0.
This classification does not claim numeric decoders for all formats.

## Checked directory policy

The explicitly supplied root is trusted. Below it, the default opener rejects
absolute paths, parent components, NUL path bytes, symbolic links/reparse points,
and Windows drive/alternate-stream names. Dot components and repeated separators
are normalized. Hard links are permitted. These are documented restrictions on
source-accepted paths, not rewritten native expectations.

On x86-64 Linux, each component is opened relative to a retained parent using
the verified Linux UAPI O_PATH/O_NOFOLLOW flags. A final regular file is reopened
through its retained procfs FD capability, with device/inode identity checked.
This avoids opening FIFOs or devices for data before checking their type.
Directory listing also uses the retained root's procfs capability. Procfs must
be available. Other Linux architectures return Unsupported for the directory
boundary; their constants are not assumed to equal the x86-64 ABI.
Already-opened-file assembly and standard positioned reads remain available.

On Windows, NtCreateFile resolves a relative Unicode name against the retained
directory HANDLE with OBJ_DONT_REPARSE, FILE_OPEN, synchronous read access and
FILE_SHARE_READ. NtQueryDirectoryFile enumerates that same retained directory.
Native status translation, ownership adoption and record bounds are confined to
the private platform module. The measured host is x86-64 Windows.

All reference configurations and results are bounded to the measured hosts.
These tests do not establish behavior on every operating system/filesystem.

## Explicit limits

Limits count cumulative input/parse work and owned descriptors across the
checkpoint. They are not a total-process or peak allocator budget.

| Limit | Default |
| --- | ---: |
| Selected/supplied file owners | 1,024 |
| Individual encoded header | 100,000,000 bytes |
| Cumulative encoded headers | 256 MiB |
| Encoded index | 16 MiB |
| JSON nodes | 4,000,000 |
| Cumulative decoded/copied text | 256 MiB |
| Individual string | 16 MiB |
| JSON depth | 256; configuration ceiling 512 |
| Decimal integer digits | 4,300 |
| Tensor descriptors | 100,000 |
| Shape items | 1,000,000 |
| Directory entries | 100,000 |
| Indexed path bytes | 64 KiB, plus Windows native UTF-16 ceiling |
| Path components | 256 |
| Allocating raw read | 256 MiB |

All explicit parser, header, descriptor and path allocations in this crate are
fallible. Standard-library and OS-internal overhead is outside the work counters.
Arithmetic for
ranges, dimensions, byte counts, cumulative limits and native UTF-16 lengths is
checked. Streaming payload access does not allocate a tensor-sized buffer.

## Independent validation

The immutable fixture cohort has 147 file cases and 48 index cases per host.
It contains 106 admitted headers, 98 exact native raw reads, six captured native
raw-read failures and six oversized/nonfinite ranges deliberately not probed by
the reference adapter. Those six are checked Rust errors, not claimed native
failure comparisons.

The source admits 32 indexed cases on Windows and 33 on GNU/Linux. The default
checked directory policy admits 28 on each host; trusted capability assembly
preserves all admitted source cases. One original Windows outside-symlink path
fails in its OS open, while GNU admits it; both observations remain intact.

Ordinary tests also cover cumulative limits, ignored unused files, separate typed
validation, stable raw ownership after replacement, truncation, stream bounds,
native-open unwind injection, directory/component/final-file swaps, and
FD/HANDLE cleanup after parsing and index errors. Linux additionally checks
non-UTF-8 filesystem names and rejection of a FIFO without blocking. Windows
checks write/delete sharing and the native UTF-16 path ceiling.

Windows passes 13 tests and GNU/Linux passes 14, each in debug and release,
with formatting, strict Clippy and a release library build. The extra GNU test
count comes from its filesystem/FIFO controls. Native expected bytes and source
inputs remain unchanged throughout the guarded runs.

Four isolated negative controls are required to fail: first-shard selection,
omitted global-name validation, following a substituted symlink/reparse path,
and accepting an empty range beyond EOF. Each fails through its intended ordinary
test on both hosts; no expected fixture is changed by these controls.

The `tests/fixtures` directory contains compact deterministic reference records,
complete original reports compressed losslessly with zlib, and provenance. The
Rust tests decode their small binary fixture format; no decompressor or Python
runtime is a product dependency. The reports retain secondary-library outcomes,
source failures, dependencies and capture identities. All expected records were
produced by original source before their corresponding Rust comparison.

The first reference import emitted two Python bytecode caches under the frozen
source tree. Source verification detected them; they were archived and relocated,
and utility.py was byte-compared with the original ZIP. The additive provenance
records the cleanup identity. Later captures disable bytecode before importing
the original module and assert those cache paths remain absent.

## License and scope

The port follows the frozen llama.cpp MIT-licensed implementation; the workspace
upstream license notice applies. Generated fixtures contain synthetic input
bytes and original behavioral observations, not pretrained model weights.
Reference NumPy/Safetensors distributions remain external and are identified
in provenance.

This foundation does not complete T04, dense Llama conversion, the Hugging Face
converter CLI, architecture registration, tokenizer export, remote loading,
pickle support, output quantization selection, or arbitrary-byte GGUF handling.
Those contracts remain explicit follow-on work.
