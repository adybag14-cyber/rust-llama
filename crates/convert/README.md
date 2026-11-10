# Native dense Llama HF to GGUF conversion

`llama-convert` converts local Hugging Face dense Llama Safetensors checkpoints into the frozen converter's GGUF bytes. The library owns and bounds its inputs, plans the output before publication, and streams one output tensor at a time. The `convert-hf-to-gguf` binary exposes this implemented path.

The authority is the user-provided llama.cpp archive with SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`.
Python, Torch, NumPy, SentencePiece, Transformers, Tokenizers and original llama.cpp builds are external comparison tools. None is executed or linked by the Rust converter. The single registry dependency is the reviewed pure-Rust YAML parser and its four transitive packages.

## Run

After checksum-pinned dependency acquisition for a fresh checkout, normal build and test commands are offline:

~~~console
cargo fetch --locked
cargo build -p llama-convert --release --offline --locked
cargo run -p llama-convert --release --offline --locked --bin convert-hf-to-gguf -- \
  /path/to/model --outfile /path/to/model-f16.gguf --outtype f16
~~~

On Windows, use a single line or PowerShell continuation in place of the shell backslash. `--help` describes the supported options. Output may be an explicit file or an existing directory; directory naming follows the frozen metadata naming rules. The default destination is the model directory. A completed conversion replaces an existing destination file.

~~~console
convert-hf-to-gguf /path/to/model --outtype q8_0 --fuse-qkv --outfile /path/to/output
convert-hf-to-gguf /path/to/model --vocab-only --outtype f32 --outfile vocab.gguf
convert-hf-to-gguf /path/to/model --outtype f16 --dry-run
~~~

`--dry-run` parses inputs, validates the conversion plan, and reports the predicted file size without creating an output. It does not execute tensor preparation, so later tensor working-budget or read errors can still occur during a real conversion. Usage errors exit with status 2 and conversion errors with status 1.

## Admitted conversion contract

| Area | Implemented behavior |
| --- | --- |
| Architecture | `LlamaForCausalLM`, `LLaMAForCausalLM`, and `LlamaModel`; dense text checkpoints |
| Inputs | Local Safetensors files and indexed shards selected by the frozen prefix/index/global-name rules, with retained file owners and bounded JSON parsing |
| Output | Little-endian GGUF; `auto`, F32, F16, BF16, Q8_0, TQ1_0 and TQ2_0; source shape-dependent F16 fallback |
| Tensor mapping | Frozen Llama filtering, renaming, insertion order, Q/K row permutation, and complete compatible QKV fusion including biases |
| Metadata | Config defaults and supported nested text config; original metadata order and types; model-card YAML, generation configuration, special tokens, chat-template metadata and naming |
| Vocabulary | SentencePiece model vocabulary; reviewed Hugging Face BPE ingestion from local tokenizer JSON |
| RoPE conversion | Metadata for source-supported configurations; generated Llama 3 F32 `rope_freqs.weight` on the measured native source profiles |
| Vocabulary-only output | Original unpadded metadata-only GGUF header |
| Publication | Complete temporary file, flush, length verification and sync before rename into place |

The dtype codec accepts the frozen numeric input domain, including F16/F32/F64/BF16, signed integer types, U8, Boolean, and the measured FP8 scalar formats. These are casts into the dense conversion working representation; preserved prequantized FP8/ModelOpt layouts are a different contract. U16/U32/U64 and unsupported packed formats return errors. Exact cast and packing qualifications are in [CODEC.md](CODEC.md).

The HF JSON path admits LlamaTokenizer/LlamaTokenizerFast, GPT2Tokenizer/GPT2TokenizerFast and PreTrainedTokenizerFast with a locally executable reviewed BPE pipeline. It preserves added-token precedence, relevant decoder/schema admission, frozen pretokenizer probes and class-specific postprocessing. An accepted JSON schema is separate from executable conversion support: an unimplemented normalizer, pretokenizer or regex pipeline returns an explicit unsupported error when its behavior is required. It does not substitute a similarly named tokenizer family.

The HF boundary includes the Transformers Llama fast class's recreation of BOS/EOS templates, including Python truth-value semantics, unknown-token fallback and source template-name interpretation. Generic and GPT2 fast classes retain their distinct behavior. A native tokenizer panic or invalid template is a checked Rust error.

## Deliberate boundaries

This is one conversion path in the larger port. Other architecture families, mixture-of-experts conversion, multimodal projectors, adapters, PyTorch pickle input, remote model acquisition, source prequantization layouts, big-endian output, output sharding, executable remote tokenizer code, and general upstream CLI option compatibility remain outside this crate's admitted path. The converter rejects these when selected; completion of this crate does not close the project's full conversion or release gates.

The result reproduces conversion behavior. A model or metadata feature accepted for conversion is not thereby supported by the current Rust inference runtime. In particular, writing scaled-RoPE metadata or a Llama 3 factor tensor does not establish runtime scaled-RoPE inference parity.

The API rejects symlinks/reparse points, parent traversal, absolute relative-sidecar paths and Windows alternate streams beneath an untrusted input directory. Callers with trusted model-cache symlink layouts should materialize ordinary input files or use the checkpoint capability APIs as appropriate. Input configuration strings used as filenames must satisfy the retained-root policy.

## Source-build profiles

The output profile records the frozen dependency behavior; it does not inspect installed Python packages.

| Profile | Captured original environment | Rust host qualification |
| --- | --- | --- |
| WindowsTorch211Numpy246 | Python 3.14.2, Torch 2.11.0+cpu, NumPy 2.4.6 | x86_64 MSVC Windows |
| LinuxTorch211Numpy226 | Python 3.10.12, Torch 2.11.0+cpu, NumPy 2.2.6 | x86_64 GNU/Linux |
| PortableFinite | Format-defined finite scalar codec behavior | Narrow fallback; non-ASCII metadata and profile-specific helpers can be unsupported |

`SourceProfile::current()` selects the measured association on the two verified targets. The library allows an explicit `Options::source_profile`. Codecs reproduce both captured profiles in safe scalar Rust. Llama 3 factor generation additionally requires its matching target math runtime and rejects foreign/portable profiles; [ROPE.md](ROPE.md) explains this restriction and the measured scalar/vector tail behavior.

Named chat-template ordering reproduces the captured `PYTHONHASHSEED=0` policy (SipHash13/CPython 3.14 on Windows and SipHash24/CPython 3.10 on GNU). It does not promise byte identity with arbitrary random Python hash seeds. Unicode property and case-mapping tables preserve the two source interpreter versions, rather than borrowing the Rust host's current Unicode version.

## Resource and publication contract

Defaults are explicit and configurable through `Limits`:

| Resource | Default |
| --- | ---: |
| One source tensor | 1 GiB |
| Charged allocations for one output tensor | 2 GiB |
| Complete output file | 512 GiB |
| One sidecar | 64 MiB |
| All cached sidecar bytes | 256 MiB |
| Cached sidecar owners | 128 |
| Vocabulary entries | 1,000,000 |
| Parsed protobuf fields | 4,000,000 |
| Tokenizer work operations | 64,000,000 |
| Transformer layers | 512 |

Checkpoint headers, cumulative JSON nodes/text, path work, YAML events/depth/aliases, and GGUF metadata have additional independently configured limits. Size arithmetic is checked and heap reservation is fallible. Input tensors, prepared components, permutations and joined QKV buffers are charged before allocation. The working budget is conservative and cumulative for one output tensor; freeing an intermediate buffer does not refund its charge. It is not a process RSS, allocator, OS cache, wall-clock or total-file-system budget.

`Conversion::prepare` retains the root, source tensor owners and successfully read sidecar bytes. Repeated sidecar reads use the retained cache. Source owners prevent a later pathname replacement from redirecting an already opened tensor. Unix callers must keep tensor contents stable during conversion. The caller must also keep the input/output path namespace stable during publication.

`write_to` follows Rust's `Write` contract and can leave partial bytes in a caller-owned sink after an error. `write_file` creates a separate temporary file beside the destination, verifies and syncs it, then renames it. Conversion failures preserve an existing destination and clean up the temporary output. Known configuration, vocabulary, index and tensor source paths cannot be chosen as destinations. This is completed-file publication under a stable namespace, not a power-loss durability guarantee for the containing directory.

## Library use

~~~rust,no_run
use llama_convert::{Conversion, Limits, Options, OutputType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conversion = Conversion::prepare(
        "/path/to/model",
        Options {
            output_type: OutputType::F16,
            ..Options::default()
        },
        Limits::default(),
    )?;
    println!("{:?}", conversion.summary());
    conversion.write_file("/path/to/output.gguf")?;
    Ok(())
}
~~~

## Independent regression evidence

Ordinary `cargo test -p llama-convert --offline --locked` exercises the bundled native captures. The corpus bytes were produced by the original source and pinned libraries before their Rust comparisons. Adjacent provenance files retain source identities, package versions, immutable capture hashes and selection rules.

| Regression cohort | Coverage |
| --- | --- |
| Complete converter | 64 cases per native host: 57 accepted files totaling 8,091,032 exact bytes, and 7 original rejections |
| HF schema and postprocessor contract | 1,080 admission cases (313 accepted/767 rejected), plus 732 original postprocessor encoding requests |
| HF Transformers class postprocessing | 76 cases (64 accepted/12 rejected), with 1,024 original encoding requests on each captured profile |
| Converter dtype preparation and packing | 68,410 cases and 7,157,024 compared output bytes per captured profile; details in CODEC.md |
| Complete Llama 3 factors | 1,915 original calls per native host: 1,884 accepted tensors and 31 source errors, with 537,464 exact final bytes |
| Large-index RoPE diagnostics | 852 independent original Torch operator windows per host, 72,270 F32 stage words |
| Unicode | Exhaustive 1,114,112-codepoint source classification plus independent case expansions and contextual probes |
| Metadata/vocabulary | Independent SentencePiece, special-token, model-card YAML, authorship/naming, HF BPE ingestion and tokenizer behavior corpora |
| Publication and CLI | Exact converted file, dry-run nonpublication, limits, source-overwrite refusal and destination preservation after a write-stage failure |

The complete-pipeline binary is a deduplicated pack of 109 original input/output blobs (13,163,408 bytes). The manifest verifies each blob's offset, length and SHA-256. It is synthetic reference data, not pretrained model weights. Each host runs the profile that its scalar RoPE tail can reproduce.

Separate ignored tests accept a pinned external SmolLM2 checkpoint. They verify every input file and three complete auto/F16/Q8_0 output hashes, totaling 686,581,536 output bytes per host, without copying model weights into the repository. Their required environment variables and fresh output directories are declared on the tests. They are integration evidence, not a claim to support all pretrained checkpoints.

Numerical checks compare exact bytes and F32 words, including signs and payloads of NaNs where the captured profile defines them. No approximate self-generated expected outputs replace the original references.

## Attribution

Retain [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [METADATA_NOTICES.md](METADATA_NOTICES.md) and [ROPE_NOTICES.md](ROPE_NOTICES.md). The reviewed YAML dependency cohort, archive/source hashes, resolved features and complete license texts live under `docs/dependencies` and `release/reviewed-dependencies.json`. The dependency audit rejects unreviewed identities, source replacement, feature drift, source-cache changes, native links and build scripts.
