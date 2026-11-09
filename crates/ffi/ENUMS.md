# Frozen static C enum helpers

These four functions implement the frozen public header's descriptive interface
in native Rust. No model, quantizer, allocator or original llama/ggml library
is required. Their names describe source enumerators; they do not expand the
execution loader's accepted modes or any backend's capabilities.

| Function | Defined behavior | Checked invalid-input result |
|---|---|---|
| `llama_ftype_name` | Exact source file-type spelling; bit 1024 prefixes `(guessed) `; other unknown combinations retain the source unknown string. | Every i32 value has a source fallback string. |
| `llama_flash_attn_type_name` | -1 → `auto`, 0 → `disabled`, 1 → `enabled`. | Null pointer and thread-local diagnostic. |
| `llama_load_mode_name` | -1 → `auto`, 0 → `none`, 1 → `mmap`, 2 → `mlock`, 3 → `mmap+mlock`, 4 → `dio`. | Null pointer and thread-local diagnostic. |
| `llama_load_mode_from_str` | The six exact case-sensitive byte strings map to their source integer values. | INT32_MIN and thread-local diagnostic, including a null input pointer. |

## Ownership and bytes

Name pointers refer to immutable process-lifetime storage and must not be freed
or written. Repeated calls preserve their pointers. The plain file-type name
starts ten bytes into the same static string returned for its guessed variant;
the source relationship is retained even for unknown file types.

The string parser checks bytes without imposing UTF-8. It compares through
the required first NUL and ignores every later byte. It does not trim, fold
case or recognize aliases. The implementation reads at most eleven bytes of
a long input, and stops at an earlier terminator. The caller must still supply
a live NUL-terminated allocation; arbitrary invalid foreign memory cannot be
validated. Null is explicitly checked before any dereference.

Source aborts are retained in independent external evidence. The Rust boundary
returns a diagnostic instead of aborting the caller process. INT32_MIN is not
one of the six defined source mode values, so it does not alias AUTO (-1).
Successful calls follow the crate-wide diagnostic contract and do not clear a
previous diagnostic; callers can invoke `llama_rust_clear_error` first.

## Independent coverage

The original MSVC and GNU captures were sealed before the implementation.
Their defined output bytes agree. The fixture cohort contains:

- 73,057 file-type values, including all 65,536 unsigned 16-bit inputs, every
  signed value from -2048 through -1, signed extremes, source enum values with
  each 32-bit flag and guessed flag, and a deterministic signed 32-bit sample.
- All nine defined flash/load enum-name cases.
- 1,542 valid parser inputs: six exact names plus every byte value after their
  first NUL, including invalid UTF-8 tails.
- 43 exact plain/guessed byte and pointer-offset relationships.
- 30 separately captured source abort inputs: twelve invalid enum values and
  eighteen invalid load-mode strings. The null parser case is a separate Rust
  boundary check because the source pointer dereference is undefined.

Fixtures preserve original spelling, including the source's `IQ3_S mix`
description for the IQ3_M file type. Unknown enum values and guessed-bit
combinations are compared against the original data without canonicalizing
their outputs. Tests consume `fixtures/ffi/enums`; original libraries and
capture scripts remain outside the product.

The static helper surface is independent of the other 64 absent frozen C
exports and the remaining behavioral/platform restrictions on implemented
functions. Twenty full-port release gates remain open.
