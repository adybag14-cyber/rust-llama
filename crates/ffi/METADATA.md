# Model metadata and raw GGUF model loading

The native Rust library implements five additional frozen C exports:
`llama_model_meta_count`, `llama_model_meta_val_str`,
`llama_model_meta_key_by_index`, `llama_model_meta_val_str_by_index`, and
`llama_model_meta_key_str`. These operate on the metadata retained by an
admitted model. They do not add an architecture, execution backend, metadata
override list, or model conversion mode.

## Three distinct representations

The GGUF file owner retains the complete primary file's serialized keys, values
and order as bytes. Raw inspection and serialization therefore retain suffixes
after NUL and non-UTF-8 bytes.

Model getters use the original C-string view. The first serialized key whose
prefix matches wins, including when that first entry is an array. String values
and string-array elements also use their first-NUL prefix. Valid UTF-8 projects
to `Value::String`; other bytes project to `Value::ByteString`. The generic
`Value::as_bytes` accessor covers both variants. Invalid UTF-8 keys remain
available in the raw owner rather than being replaced by invented text keys.
Consumed architecture/tokenizer settings retain their checked semantic rules.
Tensor names still require valid text without NUL at the model boundary.

The C scalar metadata cache follows a different source rule: arrays are skipped
before scalar keys are inserted. The first remaining scalar with each native
prefix wins. Consequently an array followed by a scalar alias can be absent
from scalar-oriented model lookup while still appearing in the C metadata
enumeration. The raw owner, model projection and scalar cache intentionally
preserve their separate source contracts. Secondary shards supply their tensors
and split index; their metadata is not merged into the primary metadata.

## Exact native policies

The frozen original exposes `std::unordered_map` iteration and
`std::to_string` output through its C API. Those results differ between the
measured Windows/MSVC x64 and Linux/GNU x64 builds. Safe Rust implementations
reproduce the corresponding string hash, bucket growth, collision ordering,
rehashing and scalar formatting. There is no C++ runtime call in this policy.

Integer and boolean strings are exact. Floating-point output has six decimal
places for finite values, including negative zero, and preserves the measured
infinity and NaN spellings. Windows signaling and indefinite NaN spellings are
handled explicitly. The selected profile is based on the Rust target; changing
the source compiler or standard library would require new independent evidence.

The four model-dependent metadata queries return a checked unsupported error
on other targets. Existing model loading and byte-oriented chat-template
access retain a text fallback there. The enum key helper is target independent:
the twelve frozen sampling keys return stable static C strings, and unknown
enum values return null without setting an error.

## C buffers and ownership

Successful value/key calls return the full byte length, excluding NUL. A
nonempty output capacity copies at most `capacity - 1` bytes and writes one
terminating NUL; the rest of the caller's buffer remains untouched. A zero
capacity performs no output write and permits a null output pointer. Missing
keys and invalid indices return `-1`, write an empty string when capacity is
nonzero, and do not set a Rust diagnostic error.

Nonzero capacity with a null pointer, invalid model pointers supplied as null,
and arithmetic domains that cannot be represented safely return checked
errors. As with the other C functions, nonnull pointers must refer to live,
properly sized caller-owned memory; arbitrary dangling pointers cannot be
validated.

`llama_model_chat_template` reads the scalar cache, accepts raw byte names, and
returns model-owned, NUL-terminated storage. Its pointer remains valid until the
model is freed. The high-level Rust model exposes `chat_template_bytes()`;
the text renderer checks UTF-8 when rendering is requested.

## Bounds

`ShardLimits::max_metadata_storage_bytes` defaults to 512 MiB. It charges
conservative owned raw/projection storage and peak secondary-shard temporaries.
The C cache additionally charges borrowed-reference arrays, native-policy
nodes/buckets/rehash scratch, owned scalar strings and both lookup-sort arrays
against that same budget before the handle can escape. Metadata ordering and
lookup construction have a 128 Mi-unit work limit. Native map input is bounded
to one million keys, within the independently captured growth schedule.

These are checked allocation/work charges, not an RSS guarantee or a promise
that the host allocator cannot fail. Existing file, header, string, array,
value, tensor and payload bounds remain active. Reservations in the new cache
and ordering policy are fallible.

## Independent fixtures

The original-only captures were sealed before Rust comparison. The committed
fixture data preserves all original requests, output buffers and canaries,
metadata, tokens, token pieces, model getters, and continuation logit words.
A lossless adapter reconstructs the complete typed JSON before accepting its
compact representation.

| Corpus | Windows/MSVC x64 | Linux/GNU x64 |
| --- | ---: | ---: |
| Model requests | 171 | 171 |
| Accepted models | 151 | 151 |
| Metadata buffer calls | 334,369 | 334,341 |
| Enumerated scalar entries | 12,507 | 12,507 |
| Tokenization requests | 3,020 | 3,020 |
| Exact continuation logit words | 144,384 | 144,384 |
| Hash/growth/order maps | 282 | 282 |
| Hashed input keys | 1,063,864 | 1,063,864 |
| Raw f32 formatting patterns | 95,003 | 95,003 |
| Raw f64 formatting patterns | 127,259 | 127,259 |

Three original authored model files had truncated tensor payloads. Their six
requests remain as rejected cases, and corrected variants were added separately;
the original data was not silently replaced. Hash/growth tests compare complete
native record digests, including insertion snapshots, rather than selected keys.
Float digests bind every input bit pattern to its complete original output.

See [model provenance](../../fixtures/ffi/model-metadata/provenance.json),
[policy provenance](../../fixtures/ffi/model-metadata/policy-provenance.json),
and [fixture integrity declarations](../../docs/model-metadata-fixture-manifest.tsv).
The frozen source ZIP is identified by SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.

The full public ABI, arbitrary model features, remaining conversion paths and
other operating-system/compiler profiles remain tracked in the release gates.
