# Bounded GGUF files and model shards

The crate reads seekable GGUF v2/v3 files, validates their metadata and tensor
directories, and reads tensor payloads on demand. It also owns complete GGUF
shard sets using the frozen original model loader's rules. All maintained
implementation is Rust; no original llama/ggml binary or foreign executable is
called at runtime.

This is a formats and loading slice of T04. It does not implement the frozen
checkpoint converters, the ordered streaming split/merge applications, or every
format compatibility edge.

## Single files and shard sets

Gguf<R: Read + Seek> remains the single-file parser. Gguf<File>::open and
open_with_limits open local files. Public metadata and descriptors retain their
serialized value types, dimensions, dtype IDs and offsets. read_tensor returns
stored bytes, while read_f32 performs checked supported dtype decoding.

GgufSet opens one primary path and retains every file handle used during
validation. It has one primary metadata map and a unified tensor directory.
Each tensor is associated with its actual file, data offset and byte order.
Reading a tensor does not reopen its path. Replacing a directory entry after
opening cannot silently redirect a later read to a different file.

| API | Behavior |
| --- | --- |
| GgufSet::open(path) | Open a single file or discover its declared siblings. |
| GgufSet::open_with_limits(path, limits) | Same behavior with explicit cumulative parser and shard limits. |
| GgufSet::open_splits(paths) | Open a caller-supplied ordered list using default limits. |
| GgufSet::open_splits_with_limits(paths, limits) | Explicit list and explicit limits. |
| metadata() | Primary metadata only. Subsidiary model metadata is never merged. |
| tensors(), tensor(name) | Complete descriptor-order list and indexed lookup by unique name. |
| tensor_shard(name), tensor_endian(name) | Tensor's originating file and payload byte order. |
| shard_info(index) | File path, version, endian, alignment, data offset, length, count and header usage. |
| read_tensor(name), read_f32(name) | Fallible payload reads through the retained file handle. |
| file_bytes() | Sum of physical file sizes, including headers, padding and trailing bytes. |
| tensor_bytes(), tensor_elements() | Unified stored payload bytes and logical element count. |
| header_usage() | Cumulative charged header bytes, metadata entries, values and tensor descriptors. |

Construction is transactional: a malformed or missing shard releases all staged
file owners and returns an error. No partially indexed set is published.
Tensor payload decoding remains lazy; opening does not concatenate the model
or eagerly decode its weights. Header parsing uses one temporary read-ahead
buffer, bounded to 64 KiB and to the remaining file/header allowance. This can
prefetch a bounded payload prefix. The buffer is discarded and the reader is
returned to its consumed logical position before it is retained for later
payload reads.

## Frozen model loader rules

These rules follow src/llama-model-loader.cpp and the original llama C API.
They deliberately differ from some stricter checks performed by the original
standalone merge application.

1. Primary split.count is UINT16. An omitted value, zero or one selects the
   ordinary single-file path. For this case, additional explicit paths are
   ignored, including paths that do not exist, as in the original API.
2. When the primary count exceeds one, primary split.no must be UINT16 zero.
   An explicit path list must contain exactly the declared number of files.
   Oversized explicit lists are checked without reproducing the original
   narrowing cast to UINT16.
3. Automatic discovery requires the exact suffix
   -00001-of-000NN.gguf for the primary file. Siblings use the same prefix and
   one-based, minimum-width-five decimal numbers. The first split must be
   selected. Native path representations are preserved, including Unicode on
   Windows and non-UTF-8 names on Unix.
4. Subsidiary file i must have UINT16 split.no equal to i. This is the only
   subsidiary model metadata key used by the original file-set loader.
   Subsidiary architecture, tokenizer, split.count and total-count values do
   not replace or augment primary model metadata. Each file's GGUF header and
   general.alignment still undergo format validation.
5. Tensor names must be unique across all files. Primary split.tensors.count
   must be a nonnegative INT32 equal to the complete tensor count.
6. A metadata-only first shard is valid. Every later tensor offset uses its own
   file's data section, even when header lengths and alignments differ.

The original merge tool also compares subsidiary split.count and writes a
single-file output with split.count reset to zero. This loader does not apply
those merge-tool-only checks.

## Parser layout details

The frozen GGUF reader requires tensor offsets to equal the cumulative padded
payload size in descriptor order. Aligned gaps, overlaps and reordered payload
regions are rejected. Every absolute file offset, payload end, dimension product
and cumulative count uses checked arithmetic.

Zero-dimensional tensors represent one scalar. Zero-element dimensions and
empty tensor names are accepted where the original format reader accepts them;
names of 64 or more bytes are rejected. Dimensions and products remain inside
the signed GGML domain. Model vector/matrix shape checks pad serialized
dimensions with trailing ones, matching original GGML shape resolution.

When a file has no tensors, the original reader reports the unpadded end of
its header as data_offset. The original writer still pads its output.
Both padded and unpadded metadata-only inputs are therefore valid. A nonempty
tensor directory uses the aligned data-section offset.

The raw original no_alloc GGUF reader can accept a truncated tensor payload
because it only parses the header. The original model loader rejects that file.
The Rust parser checks payload bounds during opening and rejects it earlier;
the native test corpus records this distinction explicitly.

The existing Rust reader also supports opposite-endian scalar GGUF decoding.
The frozen original C reader rejects the opposite-endian file, so that extension
is not described as native reader acceptance. Packed matrix endian support
remains limited to the forms explicitly handled by the inference loader.

The metadata representation currently requires UTF-8 strings and rejects NUL
keys/names. Full arbitrary-byte metadata compatibility remains open.

## Resource limits

ShardLimits contains max_shards and the existing Limits value. The defaults are:

| Limit | Default | Accounting |
| --- | ---: | --- |
| Owned files | 1,024 | Complete shard set |
| Header bytes, including required alignment | 256 MiB | Sum across all shards |
| Metadata entries | 100,000 | Sum across all shards |
| Metadata values | 4,000,000 | Sum across all shards |
| Tensor descriptors | 100,000 | Sum across all shards |
| One string | 16 MiB | Each string |
| One array | 1,000,000 values | Each array |
| One stored/decoded tensor read | 2 GiB | Each requested read |

Limits are resource policy, not new GGUF field widths. Callers can select
different bounds explicitly. Each subsidiary parser receives the remaining
cumulative allowances, so a declared split count cannot multiply the default
header budget.

Before reserving a descriptor or metadata-value vector, the parser checks its
minimum possible encoded length against both the remaining file and the
remaining header allowance. This prevents an attacker-controlled count from
reserving a large vector before the first later byte charge fails. Owned vector
and file-set reservations use checked, fallible allocation APIs.

These limits bound format work and payload reads. They are not a total-process
memory budget. Metadata tree allocation overhead, temporary decode buffers,
model weights and runtime state have separate costs. The model facade's decoded
weight budget counts every tensor from every shard before decoding any weights.

## High-level model integration

The existing Model::load and Model::load_with_options paths automatically use
GgufSet. Dense Llama, dense Qwen2, Mamba-1 and dense BERT keep their existing
runtime graph and arithmetic paths. CLI and server calls that already use
Model::load inherit automatic sibling discovery.

The additional Rust facade APIs are:

- Model::load_from_splits(paths)
- Model::load_from_splits_with_options(paths, options)
- Model::load_from_gguf(files, options)

The last form consumes a previously validated GgufSet, allowing applications to
choose parser and file-count limits before model loading. No new C ABI exports
or native split-filename buffer conventions are claimed by this slice.

For example, a consumer can open explicit files with a 64-file limit:

~~~rust
use llama_gguf::{GgufSet, ShardLimits};
use llama_inference::{LoadOptions, Model};

let paths = ["part-a.gguf", "part-b.gguf"];
let limits = ShardLimits {
    max_shards: 64,
    ..ShardLimits::default()
};
let files = GgufSet::open_splits_with_limits(&paths, limits)?;
let model = Model::load_from_gguf(files, &LoadOptions::default())?;
~~~

Low-level malformed-file and resource errors remain distinct GGUF errors.
The high-level inference facade preserves its existing error mapping.

## Independent corpus

The committed shard inputs and expected outputs originate from the frozen
GGUF parser, original gguf-split executable, original writer/quantizers and
original model C APIs. Captures were completed on MSVC x64 and GNU x64 before
the first Rust shard-model comparison. Reference implementations, capture
programs and pretrained weights remain outside the product.

The six fixtures under fixtures/gguf contain:

| File | Contents |
| --- | --- |
| shards-reference.json | Case decisions, requests, native metadata, shapes, offsets, IDs, pieces and byte-pool references. |
| shards-inputs.bin | 132 distinct native/synthetic GGUF inputs, materialized at 187 case paths. |
| shards-payloads.bin | Exact independently extracted native tensor payload bytes. |
| shards-msvc-values.f32le.bin | Lossless MSVC model/embedding F32 expected values. |
| shards-gnu-values.f32le.bin | Independent GNU model/embedding F32 expected values. |
| shards-provenance.json | Source, generator, model-origin, library, raw capture and artifact identities. |

There are 90 model/index configurations: 65 complete file sets are accepted,
25 are rejected, and 64 supported model configurations execute. Ten additional
raw format cases cover scalar/empty/name/layout boundaries. The models include
F32 and packed Llama, BERT token embeddings and pooling, Mamba default/RMS/untied
graphs, and Qwen2 separate/fused-Q4/untied-output graphs. Ordinary and
metadata-only-first splits execute through both automatic and explicit APIs.

The ordinary regression verifies structural acceptance, primary-only metadata,
each file origin, exact tensor bytes, 464,272 model/embedding values and 590
greedy IDs over those executions. Repeated paths do not count as new models or
unique numerical rows. Expected pools remove only byte-identical duplicate
records and a verified final-prompt/first-greedy overlap.

Token IDs, bytes and structural decisions are exact. The two verified x86_64 MSVC/GNU targets require identical F32 bits.
Other targets use the existing 2e-4 + 2e-5 * abs(reference) portability budget
selected before the first Rust shard evaluation. Existing architecture-specific
suites and stricter criteria remain unchanged. Expected profile selection is independent
of runtime dispatch.

Additional tests verify cumulative budgets, payload-read laziness, aggregate
decoded-weight limits, retained file-handle identity, Unicode discovery and
Unix non-UTF-8 paths. The two early-reservation regressions were independently
replayed with only their preflight calls removed; both fail in that negative
control and pass with the fix. Validation logs retain that failure and the
earlier stale single-file-only Mamba and Qwen2 test assertions.

~~~sh
cargo test --offline --locked -p llama-gguf -p llama-inference --all-targets
cargo test --release --offline --locked -p llama-gguf -p llama-inference --all-targets
cargo clippy --offline --locked -p llama-gguf -p llama-inference --all-targets -- -D warnings
~~~

Existing hash-pinned pretrained inference comparisons run separately on both
verified targets. Their numerical data and tolerances are not regenerated.

## Bounded metadata I/O

The original stdio reader buffers small header reads. Passing a File directly
to the Rust scalar parser caused each length, scalar and token string to reach
the backing file separately. The corrected header adapter allocates at most
64 KiB through a fallible reservation. Refills cannot exceed the remaining
physical file or metadata allowance. It preserves short reads and I/O errors,
rewinds unused read-ahead on both parse success and failure, and leaves the
public Gguf<R> owner and payload seek path unchanged.

A header-only measurement on the pinned Qwen2.5-0.5B file through WSL /mnt/c
recorded 760,618 underlying File reads and 100.5658 seconds before buffering,
versus 93 reads and 0.07945 seconds afterward. Both parsed 26 metadata entries,
455,285 values, 291 tensors and 5,947,744 charged header bytes. The buffered run
read a further 16,043 bytes into its bounded temporary buffer and then restored
the logical cursor. It did not decode weights.

These are single-run measurements of this mounted filesystem and its current
cache/load conditions. They do not establish general CPU, disk or model
throughput. The deterministic read-count regression, budget checks, partial-read
and injected-I/O-error tests establish the implementation behavior separately
from timing. Existing decoder CLI evidence preserves the original 90-95 second
Qwen loads; native numerical references remain unchanged.

## Remaining format and conversion work

The existing writer emits little-endian GGUF from a sorted metadata map and
borrowed tensor payloads. It is not an ordered streaming source-equivalent
split/merge writer. An ordered metadata representation and a bounded streaming
payload writer are needed before those tools can preserve original output
ordering and handle large transformations faithfully.

Local Safetensors/index loading, architecture-specific checkpoint conversion,
tokenizer/config translation, quantization application policy, PyTorch inputs,
LoRA and multimodal conversion, remote range loading and remaining metadata
utilities are still separate full-port requirements. Sharded model execution
does not close T04 or the overall port.

Implementation follows the supplied llama.cpp/ggml MIT-licensed source. Preserve
LICENSE-UPSTREAM and NOTICE; fixture provenance identifies inherited model
vocabulary metadata and the deterministic synthetic weight generators.
