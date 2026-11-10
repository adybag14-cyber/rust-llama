# Frozen native C chat fixtures

These are independent original-library observations, not expected values
generated from Rust.

- `inputs.bin`: the deterministic input matrix.
- `original.bin`: the common byte-identical original Windows/GNU result.
- `provenance.json`: source hashes, input-generation identity, case families,
  native template names and source-derived detection markers.
- `original-seal.json`: immutable original capture identities and counts,
  created before Rust implementation.
- `fixture-manifest.tsv`: file lengths, hashes and purposes.

All integers are little-endian. A blob is a u32 byte count followed by bytes.

## Input format

The eight-byte magic is `CCHATIN1`, followed by a u32 case count.
Each case contains a u8 flags field, i32 requested capacity, template blob,
u32 message count and one role/content blob pair for each message.

Flag 0 is the assistant-prompt boolean; flag 1 selects a null template;
flag 2 selects a null output buffer; flag 3 derives actual capacity by adding
the requested offset to the original count-only result (or 17 on rejection).
Input blobs can contain NULs; the external C caller appends a final NUL and
lets the original API select the prefix.

## Original output format

The eight-byte magic is `CCHATOU1`, followed by the same u32 case count.
Each case records i32 count-only result, i32 buffered result, i32 actual
capacity, then a blob containing the entire buffer and both eight-byte guards.
The initial allocation is filled with `0xa5`; null/nonpositive output calls
still retain sixteen untouched guard bytes.

After cases, five u32 values describe native message size/alignment/role offset/
content offset and C bool size. A u32 builtin count precedes its byte-string
blobs in original enumeration order. Finally a u32 limit-case count precedes
each builtin return value, slot count, and per-slot pointer equality byte.
Limits are 0, 1, 2, count-1, count, count+1, count+8 and SIZE_MAX. Equality checks
compare written pointers with a prior complete call and remaining slots with
the untouched sentinel; process addresses are not stored.

The C caller and build artifacts remain outside the product under the Hermes
`rust-llama-reference/continue-20261110/chat-oracle` directory. It links no
Rust code while producing original results and uses the frozen header's
function pointer types. Reference symbols are loaded directly from the
identified original DLL/SO, under the C locale. It is reused unchanged against
the Rust DLL/SO for independent ABI replay.

The source is MIT licensed; the workspace upstream notice applies. Inputs are
synthetic message bytes and source-derived public template identifiers.
No model weights or private conversations are included.
