# Model-file quantization

This crate implements native Rust model-file quantization from the frozen
`src/llama-quant.cpp` contract. It uses the Rust GGUF reader and scalar
quantization codecs; it does not load or call the original llama/ggml library.

The authoritative source archive SHA-256 is
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
The original code is MIT-licensed; see the repository license and source ledger.

The package also provides a native [llama-quantize CLI](CLI.md) with bounded
legacy/GGUF importance ingestion, source argument rules and independent
complete-file fixtures.

## Library use

~~~rust,no_run
use llama_model_quant::{FileType, QuantizeParams, quantize_file};

let params = QuantizeParams {
    ftype: FileType::parse("Q4_K_M")?,
    nthread: 4,
    ..QuantizeParams::default()
};
let report = quantize_file("model-f16.gguf", "model-q4_k_m.gguf", &params)?;
assert!(report.written);
# Ok::<(), Box<dyn std::error::Error>>(())
~~~

`quantize_file_with_limits` additionally accepts explicit parser, shard,
working-buffer, importance-value, override-count and worker bounds. Input
handles are retained while reading tensors. Output payloads are streamed in
row-aligned slabs, without loading the model's inference graph or decoding all
weights at once.

`QuantizationReport` records input and output payload bytes, complete output
file bytes, output paths, and each tensor's original/remapped name, dimensions,
source/target dtype and importance requirement. With `dry_run`, it describes
the validated layout without creating output files. A dry run does not execute
codecs or validate tensor payload values.

## Admitted model files and type planning

The file driver admits the implemented dense/encoder metadata families
`llama`, `qwen2`, `qwen3`, `gpt2`, `gptneox`, `bert`, `t5` and `bloom`.
It validates the frozen loader's common and architecture-specific metadata
before assigning output types. File admission does not establish inference
support for every configuration or for T5. Other architecture names return a
checked unsupported error.

The separate `ModelDescriptor` / `assign_types` API accepts the frozen
architecture registry and applies its tensor type-selection heuristics. That
registry is not a claim that the file loader or runtime implements every
architecture. It requires explicit geometry and ordered tensor descriptors.

The 35 source-supported file selections are:

| Family | Selections |
| --- | --- |
| Floating point | F32, F16, BF16 |
| Legacy block formats | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 |
| K formats | Q2_K, Q2_K_S, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K |
| Importance-aware IQ formats | IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS |
| Other frozen formats | TQ1_0, TQ2_0, MXFP4_MOE, Q1_0, Q2_0 |

Names are case-insensitive. Q3_K, Q4_K and Q5_K select their M variants.
The frozen enum also declares NVFP4 and F8 file modes, but its file quantizer
rejects those IDs. This API preserves that rejection even though separate
low-level codecs exist.

Selection includes the original tensor-category precedence, layer counters,
GQA/MQA and expert heuristics, tied output handling, output/embedding overrides,
and block-width fallbacks. `pure` selects the requested dtype for eligible
weights subject to the original exclusions and explicit overrides. Tensor order
is significant: normal file conversion uses the source's leading numeric
`blk.` layer order followed by bytewise names. Split preservation uses shard
and physical-offset order.

The exported original type helper receives eligible tensors only and resets
its importance state. The real file driver counts all tensors, including
ineligible output weights. Independent tests cover both entry points without
conflating their different input contracts.

## Options and wire behavior

The owned parameter structure provides:

- Quantize or preserve output weights; independent token/output dtype overrides.
- Ordered tensor-name regex overrides, checked against the measured native
  MSVC or GNU regex profile.
- Optional importance rows, including the distinction between no importance
  pointer and a present empty map.
- Explicit permission for requantization of already quantized weights.
- Copy-only output, including the source's guessed file-type flag when the
  input has no file-type metadata.
- Typed metadata overrides with the source's separate lookup and output rules.
- Layer pruning and encounter-order layer remapping.
- Split preservation or merging, dry run, worker count and slab target.

The metadata writer preserves the original tool's C-prefix projection,
remove-and-append setter order, quantization version, file-type updates,
tensor-rank collapse and 32-byte physical layout. Split output names append the
source suffix even when only one output shard remains. Multi-shard metadata
uses the quantizer's own key order, which differs from the split utility.

Source metadata override lookup finds the first matching key and uses that
entry only when its tag matches the expected type. Integer hyperparameter overrides use the source's unsigned cast, whereas
integer output metadata uses the absolute magnitude as UINT32. INT64_MIN has
no defined source absolute-value result and is a checked error. Output setters
apply the full ordered override list. This can produce metadata inconsistent
with the geometry used during quantization, as in the original tool; a successful
quantization is not a guarantee that the resulting model will load for inference.

Importance is looked up by the original layer name after pruning. Per-expert
rows cover dimensions 2 and 3. Required IQ paths fail if a needed row is absent;
an incorrectly sized token-embedding importance row is ignored as in the source,
whereas other size mismatches fail. The scalar codecs admit finite,
nonnegative importance values. Negative or nonfinite weights are outside this
checked input domain.

Regex overrides preserve the measured byte-oriented MSVC/GNU differences,
including character escapes, blank classes, unmatched backreferences and
repeated quantifiers. The GNU extension is bounded: single-byte collating
elements and the named space element are implemented; unsupported named
collating forms fail explicitly. Existing callers of the default regex parser
retain the MSVC behavior.

## Resource and publication contract

Defaults allow 256 MiB of live source/decoded/output slab payloads and temporary
packed worker rows, at most 64 workers, 128 million owned importance values and
4,096 overrides. The source-compatible default slab target is 8 GiB, but the
hard working-buffer bound limits the actual allocation. One complete row must
fit; a smaller slab target still processes one row.

These are separate bounds, not a total process-memory guarantee. GGUF metadata,
owned importance maps, scalar codec internal workspaces, thread stacks, allocator
overhead and the operating-system cache are additional. Parser limits constrain
their applicable input structures. Worker rows are disjoint, and each worker
produces independent source-compatible scalar results.

The supported input boundary is little-endian GGUF accepted by the checked
model/shard reader, with one to four positive tensor dimensions and valid
UTF-8, NUL-free tensor names shorter than the frozen 64-byte name capacity.
Raw primary metadata retains its byte-string representation. Unsupported dtype
conversions and malformed geometry return errors. Copied and converted floating
payloads must remain finite; integer copy payloads retain their bytes.

Each destination gets a unique `create_new` temporary file in its own directory.
All tensors and all output shards are converted, flushed, synced and size-checked
before publication starts. Conversion failure preserves existing destinations
and removes unpublished temporary files. Write-time input/output aliases are
rejected using OS file identity, including hard links on Windows and Unix.
Dry-run plans a layout without examining or opening the output destination,
so an existing aliased file or directory is permitted. The original GNU C API
can modify an aliased input or abort; checked rejection preserves the Rust
input-safety contract. Independent disposable-copy observations retain that
explicit safety correction.

Each final rename is atomic on the supported local filesystems. Publication of
several shard files is not a filesystem-wide transaction: a later rename can
fail after an earlier shard was published. Callers needing atomic set replacement
must publish through a separate directory/version switch.

Some malformed source inputs abort the original process, including a captured
pruning case that creates duplicate tensor names. The Rust API returns a checked
error and preserves the destination for those cases; it does not reproduce a
native process abort.

## Frozen public C interface

The `llama-ffi` crate exposes
`llama_model_quantize_default_params` and `llama_model_quantize` with the
original public struct layout. On the measured x64 profiles the parameter
structure is 64 bytes with 8-byte alignment. Typed sentinel-terminated arrays
for importance data, metadata overrides, regex overrides and pruning are
copied into owned Rust values before conversion.

C callers must provide live pointers and correctly terminated arrays/strings.
The facade bounds entry counts and accepted values, catches Rust panics at its
existing error boundary, and returns the native success/error status. The
maintained Rust library never calls the original implementation.

## Independent reference coverage

The initial corpus was captured and sealed on the original Windows MSVC and
Linux GNU runtimes before this crate was written. Each profile contains 463
type/default scenarios and 88 file scenarios. It covers all 35 file modes,
mixed precision, regex differences, importance rows, experts, threads/slabs,
pruning, metadata overrides, copy/requantization and split/merge behavior.

The ordinary suite compares 16,315 MSVC and 16,417 GNU tensor assignments, and
the complete byte length and SHA-256 of all 76 successful native output files per
host. Original rejections and the isolated native pruning abort remain explicit.
A resource-bound regression repeats an independent complete-file comparison
with a one-row working budget and verifies that an insufficient budget preserves
an existing destination.

Metadata admission has a separate original corpus of 346 cases per host.
Within the eight admitted file families, the maintained tests compare 140
complete copy outputs and check 202 rejected or source-aborting inputs; four
observations outside that family boundary remain explicit exclusions. Ten
additional complete quantizations cover the source's NextN-aware Llama 70B
classification. These add 150 exact complete-file comparisons per host and
build mode to the main 76-file corpus.

The original public-header C caller separately checks defaults, struct layout,
both exports and all 88 file requests against each debug and release Rust
library on both hosts. Each replay produces 77 successes and 11 checked
rejections, including all 76 exact complete output files. The recorded library
imports contain no original llama/ggml runtime dependency.

The integrity manifest is `docs/model-quant-fixture-manifest.tsv`; fixture data,
preimplementation seals and producer identities are under `tests/fixtures`.
Original reference callers, builds and full raw output payloads remain outside
the product. The component receipts are under
`evidence/2026-11-11/model-quantization`. These comparisons establish this
documented slice; full architecture, ISA, performance and complete-port release
parity remain separate gates.
