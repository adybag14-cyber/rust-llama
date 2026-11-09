# Raw GGUF strings and original tool semantics

This checkpoint adds a shared byte-preserving GGUF wire parser, a raw ordered
writer and native byte-string behavior in the split/merge application. It does
not close the format/converter track or any full-port release gate. All maintained
implementation is safe Rust; the original C/C++ code is used only as an external
independent oracle.

## Public representations

`MetadataValue<S>` holds the existing GGUF scalar/array variants with a string
storage parameter. `Value` remains the concrete `MetadataValue<String>` alias;
existing text constructors retain their inference behavior. `RawValue` uses
`Vec<u8>`. Numeric getters are shared. Raw scalar strings expose both complete
bytes and the prefix ending before the first NUL.

| API | Contract |
| --- | --- |
| `RawGguf::read` / `open` | Parse the header, validate the complete declared payload layout and retain the reader/file. |
| `RawGguf::read_header` | Match original no-allocation header admission; a later payload read still checks physical bounds. |
| `metadata_order` | Complete serialized keys in original order, filtered to keys still present in the public map. |
| `metadata_native` | First original-order key whose C-string prefix matches the query prefix. |
| `find_tensor`, `tensor`, `read_tensor`, `read_f32` | First descriptor with the matching C-string name prefix. |
| `read_tensor_by_index` | Select the exact retained descriptor, including distinct full names that share a native prefix. |
| `RawGguf::into_text` | Consume the raw owner and perform checked UTF-8 conversion, rejecting NUL keys/names. |
| `RawGguf::write_native_header` | Reproduce the frozen native metadata serializer's version, full metadata bytes, C-string tensor names and GGML rank. |
| `write_raw_header` | Write a v3 little-endian ordered header with complete supplied byte strings, names and dimensions. |

`RawGguf` stores complete names and values without lossy replacement. Diagnostic
formatting can use a readable representation without changing stored bytes.
Duplicate metadata keys and duplicate tensor names are checked against complete
serialized bytes. Distinct names such as `x\0a` and `x\0b` can therefore coexist
in the raw representation; original native lookup returns the first `x`.

Both raw and text APIs use one wire parser, shared layout validation and shared
payload readers. This does not create a second parser whose rules can diverge.
`Gguf`, `GgufSet` and the current high-level model loaders keep checked text
metadata. Extending every model/vocabulary consumer to arbitrary byte strings
remains separate work.

## Boolean bytes and serialization

The frozen original reads every nonzero boolean byte as true. The Rust parser
now follows that rule for scalar and array booleans across all 256 byte values.
The in-memory value is a boolean; serializers emit canonical 0/1 bytes.
Preserving the original noncanonical boolean byte itself is not promised.

The public raw writer preserves supplied full strings and dimensions, uses the
first native-prefix `general.alignment` entry to resolve alignment, and preflights
the complete header before its first write. The existing text writer API remains
available. Both use the same bounded encoder and payload layout machinery.

The native serializer retains full metadata keys and string values, but obtains
tensor names from their original C-string view and derives rank from GGML shape.
Its v2/v3 version and stored alignment come from the parsed file. It serializes
the original ordered keys still present in the map; newly inserted map keys have
no original order. A caller modifying a header should pass its chosen ordered
list to `write_raw_header`.

These contracts are distinct. A native reserialized header can intentionally
lose the suffix of a tensor name, and the frozen split/merge application has
additional setter semantics described below. Little-endian original captures
are the measured native serialization profiles. Existing opposite-endian scalar
reader support remains a separately documented Rust extension.

## Split and merge behavior

The tool now reads `RawGguf` directly. Its outputs preserve the frozen producer
contract, including behavior that is easy to lose when converting through text:

- Metadata keys are passed through C-string setters. Replacing an existing
  prefix removes its earlier position and appends the replacement. The final
  value appears at its final setter position.
- Every incoming alignment setter is checked, including a value that a later
  alias would overwrite. Empty native keys and original assertion conditions
  produce checked errors.
- Scalar and array string values use their C-string prefix in copied metadata.
  Reserved split keys are overwritten in the same order as the original.
- Tensor names use their native prefix. For each source descriptor, original
  tensor lookup selects the first matching prefix. Its shape, byte count and
  payload therefore also come from that first descriptor, even when a later
  alias has a different width or is placed in another output split.
- Duplicate names within an output that would assert in the original return
  checked errors. Merge resolves reserved keys through first-prefix lookup.

The previous streaming contracts remain: bounded copies of at most 64 KiB,
retained input handles, cumulative header budgets, output alias protection,
create-new merge outputs and deletion only after complete output flush/sync.
The original tool's fresh physical alignment of 32 and rank canonicalization
remain explicit producer quirks. Some native-success outputs are rejected by
the original reader; their bytes remain expected rather than being repaired.

## Independent source captures

All original captures were completed and sealed before the corresponding Rust
reader or tool implementation was compared. Both original MSVC x64 and GNU x64
profiles are retained, including raw exit codes, stdout/stderr, failures,
partial-file observations and source/compiler/library identities.

| Corpus | Cases per original host | Outcome / exact data |
| --- | ---: | --- |
| Byte-string reader and getters | 1,093 | 1,080 accepted; 13 rejected; all 256 values in scalar strings, string arrays, keys and names, plus typed/alignment/alias/version boundaries. |
| Boolean reader and getters | 512 | All accepted; every scalar byte and mixed boolean-array byte. |
| Native metadata serialization | 1,592 accepted reader cases | 269,184 exact serialized header bytes. |
| Public raw writer | 1,073 admitted canonical v3 inputs | Complete original input header bytes and independent complete-file layouts. Noncanonical boolean inputs are tested through original native normalization instead. |
| Byte-string split/merge | 3,196 | 3,176 successes, 20 checked failures; 1,438,016 exact successful post-operation file bytes. |
| Reserved-key/NUL supplement | 22 | 17 successes, 5 checked failures; 8,288 exact successful post-operation file bytes. |
| Existing command-line corpus | 103 | 63 defined successes, 40 checked errors; 12,179,168 complete post-operation file bytes and defined transcripts. |

The two original hosts produce identical successful bytes in the new raw
corpora. Their raw reports remain separate; cases are not doubled to inflate
coverage. Ordinary Rust tests use projections of the sealed native data and
unchanged original file bytes. Expected data is not generated from Rust output.

[Reader provenance](../fixtures/gguf-bytes/provenance.json),
[tool provenance](../fixtures/gguf-bytes/tools-provenance.json) and
[NUL supplement provenance](../fixtures/gguf-bytes/nul-tools-provenance.json)
identify the exact source, inputs and artifacts. The fixture manifest binds all
23 new fixture files, including full original reports and byte pools.

## Bounds and qualifications

Existing file/header/string/array/value/tensor limits remain active, including
the inherited maximum alignment of 1 MiB. Minimum encoded lengths are checked
before descriptor/value reservations. Payload offsets and dimension products use
checked arithmetic. Raw name indexing and ordered writer descriptor lists use
fallible reservations. Limits bound declared work and payload reads; they are
not a total-process memory budget.

The format slice does not claim full checkpoint conversion, arbitrary-byte
high-level model/vocabulary loading, remote loading, every file utility, all
architectures or every backend. Source undefined behavior and assertions remain
checked errors rather than portable success requirements. All twenty release
gates stay open.
