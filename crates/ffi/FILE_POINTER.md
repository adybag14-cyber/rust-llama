# Native FILE pointer loading

`llama_model_load_from_file_ptr` and
`llama_adapter_lora_init_from_file_ptr` load a GGUF stream from the caller's
current binary FILE position. The implementations use the same model,
vocabulary, metadata, LoRA validation and execution paths as their filename
counterparts. They accept anonymous and embedded streams on the measured
MSVC Windows and 64-bit GNU Linux CRT profiles.

## File ownership and offsets

The caller supplies a live, seekable, readable binary FILE from the platform
CRT and gives the call exclusive access to its cursor. The caller retains the
FILE and may close it after the call; successful model and adapter objects own
their decoded or packed weights independently.

The implementation obtains the public logical position through the CRT,
duplicates the underlying OS handle with the Rust standard library, and parses
and reads that retained duplicate. It never guesses the layout of FILE,
reopens a filename, or closes the caller's stream. Replacing or renaming the
path cannot substitute different payload data after header validation.

GGUF header/data alignment and tensor offsets are relative to the supplied
origin. Prefix bytes may have any length; trailing bytes are permitted.
Eager mode supports unaligned embedded streams. Frozen AUTO mode chooses mmap
and requires absolute CPU tensor-data alignment of 32 bytes, even for a
vocabulary-only load. The Rust entry point preserves that admission rule but
eagerly owns supported weights; `llama_supports_mmap` remains false.

The FILE entry point is a single anonymous stream. Split metadata remains
visible, but it never causes filename discovery or other streams to be opened.
Filename and explicit-split loaders retain their existing multi-file rules.

## Caller position after a call

Rust restores the caller's coherent CRT cursor to the input origin on normal
success and parsing failures. It also attempts restoration during unwinding.
Seek/duplication/restoration failures return the existing checked diagnostic.

This is an explicit boundary of compatibility. The frozen source does not
document a final cursor position, and the independently measured positions
differ between MSVC and GNU. In particular, the Windows source combines CRT
header buffering with direct HANDLE payload reads and can leave the CRT's
read-ahead state stale. Rust uses documented CRT seeks to make the subsequent
caller stream usable; it does not reproduce internal stale-buffer offsets.
Reference fixtures retain native final-position observations for audit, but
tests do not present them as equivalent Rust behavior.

The caller must not close, reposition or reenter the FILE during loading or a
progress callback. Arbitrary invalid non-null foreign pointers remain outside
the C facade's pointer contract. Null pointers, unsupported CRT targets,
non-seekable streams, malformed headers, payload truncation, resource limits,
invalid adapters and cancelled progress return checked errors. The same
architecture/weight/adapter limits as filename loading apply.

## Bounded retained GGUF support

The safe GGUF crate adds `GgufSet::from_file_at` and its limits variant.
Each consumes an owned File and an explicit origin, and retains that handle
for all lazy tensor reads. Construction uses the existing lossless parser,
metadata budget, payload validation and name index. Relative seeks reject
negative or overflowing positions, and origins beyond the file fail.
This safe API contains no foreign pointer or unsafe code.

Its tests cover unaligned prefixes, suffix bytes, renamed/replaced paths,
range bounds, malformed regions, parser budgets and anonymous split metadata.
Path loading uses the same retained-file wrapper at origin zero.

## Independent native corpus

An external client compiled with the frozen public header captured 48 model
and 24 adapter cases on each original platform before these exports were
implemented. Inputs cover prefixes 0, 1, 31, 32, 64 and 77; eager/AUTO modes;
full and vocabulary-only Llama/BERT; multi-target and single-target LoRA;
empty and metadata-only adapters; invocation tokens; incompatible architecture;
and payloads with trailing bytes. Every input FILE is closed before executing
the loaded model or applying the adapter.

Committed MSVC/GNU fixtures losslessly deduplicate output word arrays and retain
original status, metadata, invocation tokens and cursor observations. Ordinary
Rust tests compare the corresponding platform's exact output float bits,
ownership and alignment admission. They additionally establish CRT read-ahead
before loading and check coherent caller reads afterward.

Corpus files are in `fixtures/ffi/file-pointer`, with original client/library
and input identities in provenance.json.
`docs/file-pointer-fixture-manifest.tsv` inventories the shipped data.
The unchanged original source archive SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

Final Windows/GNU debug/release tests, strict Clippy, native-header client
replays and exact PE/ELF export inventories are retained in
`evidence/2026-11-13/c-context-file-api`.
