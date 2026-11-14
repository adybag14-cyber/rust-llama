# Native dense HF to GGUF conversion

`llama-convert` converts local Hugging Face dense Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX, BLOOM, Falcon, StarCoder and StarCoder2 Safetensors checkpoints into the frozen converter's GGUF bytes. The library owns and bounds its inputs, plans the output before publication, and streams one output tensor at a time. The `convert-hf-to-gguf` binary exposes this implemented path.

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
| Architecture | `LlamaForCausalLM`, `LLaMAForCausalLM`, `LlamaModel`, `Qwen2ForCausalLM`, `Qwen2Model`, `Qwen3ForCausalLM`, `Qwen3Model`, `GPT2LMHeadModel`, `GPTNeoXForCausalLM`, `BloomForCausalLM`, `BloomModel`, `FalconForCausalLM` and `RWForCausalLM`; `GPTBigCodeForCausalLM` and `Starcoder2ForCausalLM`; dense text checkpoints |
| Inputs | Local Safetensors files and indexed shards selected by the frozen prefix/index/global-name rules, with retained file owners and bounded JSON parsing |
| Output | Little-endian GGUF; `auto`, F32, F16, BF16, Q8_0, TQ1_0 and TQ2_0; source shape-dependent F16 fallback |
| Tensor mapping | Frozen architecture-specific aliases, filtering and insertion order; Llama Q/K row permutation, Qwen unpermuted rows, Qwen3 Q/K norm tensors, GPT-2 Conv1D transposes, NeoX/BLOOM fused-QKV reordering, grouped Falcon QKV reordering, BLOOM name normalization, and compatible QKV fusion including biases |
| Metadata | Family-specific config defaults and supported nested text config; original metadata order and types; model-card YAML, generation configuration, special tokens, chat templates, naming and Qwen pooling sidecars |
| Vocabulary | SentencePiece model vocabulary; reviewed Hugging Face BPE ingestion from local tokenizer JSON |
| RoPE conversion | Metadata for source-supported configurations; generated Llama 3 F32 `rope_freqs.weight` on the measured native source profiles |
| Vocabulary-only output | Original unpadded metadata-only GGUF header |
| Publication | Complete temporary file, flush, length verification and sync before rename into place |

The dtype codec accepts the frozen numeric input domain, including F16/F32/F64/BF16, signed integer types, U8, Boolean, and the measured FP8 scalar formats. These are casts into the dense conversion working representation; preserved prequantized FP8/ModelOpt layouts are a different contract. U16/U32/U64 and unsupported packed formats return errors. Exact cast and packing qualifications are in [CODEC.md](CODEC.md).

The HF JSON path admits LlamaTokenizer/LlamaTokenizerFast, GPT2Tokenizer/GPT2TokenizerFast, Qwen2Tokenizer/Qwen2TokenizerFast, GPTNeoXTokenizer/GPTNeoXTokenizerFast, BloomTokenizer/BloomTokenizerFast and PreTrainedTokenizerFast with a locally executable reviewed BPE pipeline. It preserves added-token precedence, relevant decoder/schema admission, frozen pretokenizer probes and class-specific postprocessing. An accepted JSON schema is separate from executable conversion support: an unimplemented normalizer, pretokenizer or regex pipeline returns an explicit unsupported error when its behavior is required. It does not substitute a similarly named tokenizer family.

The HF boundary includes the Transformers Llama fast class's recreation of BOS/EOS templates, including Python truth-value semantics, unknown-token fallback and source template-name interpretation. NeoX recreates the same template form with BOS/EOS insertion disabled by default. Generic, GPT2 and Qwen2 fast classes retain their distinct behavior. Qwen2 uses its own default special tokens. All admitted fast classes apply generic split-special and cleanup controls. For a direct ByteLevel root, Qwen2/GPT2/NeoX/BLOOM/generic Fast default an absent prefix setting to false, apply explicit booleans, and preserve saved settings for null/non-boolean inputs. This also applies when the whole sidecar is absent. Llama preserves its saved prefix for absent/null settings; explicit non-null prefix reconstruction requires its unsupported slow path. Qwen2 does not recreate Llama's BOS/EOS templates. BLOOM then applies its own truthy-prefix override to every ByteLevel stage, including nested Sequences, and preserves its saved postprocessor. Its cleanup option uses Python truth values. Other Sequence roots retain their nested ByteLevel settings, including a single or nested single child. A native tokenizer panic or invalid template is a checked Rust error.

A non-null sidecar tokenizer class takes precedence. An absent/null class uses the successfully resolved root model configuration, its root tokenizer class and the captured class mapping; nested text configuration does not override root tokenizer selection. Null and absent `model_type` values remain distinct. When class inference needs root `auto_map` or `configuration_files` indirection, it returns an explicit unsupported error. The native class, prefix and full-file evidence is documented in [QWEN_HF_CONVERSION.md](../../docs/QWEN_HF_CONVERSION.md).

## Deliberate boundaries

These are nine dense architecture conversion paths in the larger port. Other architecture families, mixture-of-experts conversion, multimodal projectors, adapters, PyTorch pickle input, remote model acquisition, source prequantization layouts, big-endian output, output sharding, executable remote tokenizer code, and general upstream CLI option compatibility remain outside this crate's admitted path. The converter rejects these when selected; completion of this crate does not close the project's full conversion or release gates.

Qwen HybridDNA tokenizers, InternS1 specialized vocabulary, reranker head extraction, the `linear.weight` embedding projection rewrite, Qwen mixture-of-experts and recurrent/Next/MTP variants remain outside the admitted dense path. Nested wrapper architectures retain their origin when checking these boundaries; a Qwen3 text config does not silently erase a reranker or InternS1 origin. Supported sentence-transformer sidecars select mean, CLS or last-token pooling with the frozen precedence. Conversion support for this metadata does not establish an embedding or reranking inference implementation.

The named source-conversion writer can reproduce original Python nested-array metadata. Strict GGUF writers and readers still reject nested arrays; those original-compatible outputs are not thereby valid inputs to the native GGUF reader.

The result reproduces conversion behavior. A model or metadata feature accepted for conversion is not thereby supported by the current Rust inference runtime. In particular, writing scaled-RoPE metadata or a Llama 3 factor tensor does not establish runtime scaled-RoPE inference parity.

The API rejects symlinks/reparse points, parent traversal, absolute relative-sidecar paths and Windows alternate streams beneath an untrusted input directory. Callers with trusted model-cache symlink layouts should materialize ordinary input files or use the checkpoint capability APIs as appropriate. Input configuration strings used as filenames must satisfy the retained-root policy.

GPT-2 and GPT-NeoX preserve their distinct metadata overrides, tokenizer class defaults and tensor permutations. The original official GPT-2 checkpoint fails on static causal-mask buffers; the converter preserves that source rejection. The disclosed external validation derivative, Rust-only ignored-test preparation, complete-file captures and exact boundaries are described in [GPT_HF_CONVERSION.md](../../docs/GPT_HF_CONVERSION.md).

BLOOM reproduces its separate metadata override, normalized configuration aliases, global `transformer.` removal, and head-major fused-QKV transform. The unchanged official BLOOM checkpoint is accepted by the frozen converter. Its independently captured malformed layouts, tokenizer classes, metadata sequences and complete pretrained files are described in [BLOOM_HF_CONVERSION.md](../../docs/BLOOM_HF_CONVERSION.md).

Falcon preserves its fixed metadata override, modern/legacy defaults, grouped QKV permutation and complete official tokenizer pipeline. The original-compatible malformed-input boundaries and the unverified pretrained-weight boundary are recorded in [FALCON_HF_CONVERSION.md](../../docs/FALCON_HF_CONVERSION.md).

StarCoder and StarCoder2 preserve the checkpoint's linear-weight orientation. StarCoder writes its fixed one-KV-head and four-times-hidden FFN metadata, while StarCoder2 inherits the common TextModel metadata and preserves an explicit null KV-head count. Both use the GPT-2 vocabulary path. Their independent complete-file corpus, unchanged tiny StarCoder pretrained acceptance, official sidecar provenance and exact boundaries are in [STARCODER_HF_CONVERSION.md](../../docs/STARCODER_HF_CONVERSION.md).

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
| Nested BLOOM/Falcon/StarCoder head-metadata arrays | 64 levels (fixed ingestion bound) |

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
| Complete Llama converter | 64 cases per native host: 57 accepted files totaling 8,091,032 exact bytes, and 7 original rejections |
| Complete GPT-2/GPT-NeoX converter | 101 cases per native host: 77 accepted files totaling 143,810,730 exact bytes and 24 source rejections |
| Complete StarCoder/StarCoder2 converter | 194 cases per native host: 163 accepted files totaling 357,723,535 exact bytes, 31 original rejections and 1,248 tensor-map lookups; synthetic weights and official sidecars |
| Complete Falcon converter | 117 cases per native host: 86 accepted files totaling 202,713,961 exact bytes, 31 source rejections and 510 alias lookups; synthetic weights plus unchanged official sidecars |
| Falcon tokenizer primitives | All 1,112,064 Unicode scalars and 8,745 native split/encode/decode requests across punctuation modes and prefix settings |
| Complete BLOOM converter | 113 cases per native host: 84 accepted files totaling 1,066,141,612 exact bytes, and 29 source rejections |
| BLOOM class / metadata / aliases | 87 tokenizer configurations (82 accepted/five rejected), 2,296 exact encoding/decoder requests, 41 metadata calls (20 accepted/21 rejected), and 576 native alias lookups per host |
| HF NeoX class behavior | 46 class configurations, 40 accepted/six rejected and 960 exact encoding/decoder requests per native host |
| BLOOM official tokenizer pattern | All 1,112,064 Unicode scalars and 1,922 exact split/encoding/decoding requests across both prefix settings |
| Complete Qwen2/Qwen3 converter | 112 cases per native host: 104 accepted files totaling 25,723,552 exact bytes, and eight original rejections |
| Qwen configuration fallback | 32 additional complete files per native host; 5,318,912 exact bytes, including default sliding windows, invalid thresholds and iterable layer types |
| Qwen root model-type presence | Eight additional complete files per native host; 1,329,824 exact bytes for explicit, absent, null and unknown types |
| HF class and prefix resolution | 106 class cases, 48 direct-prefix cases and 34 missing-sidecar/Sequence cases; 5,708 accepted encoding/decode requests per profile |
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

The Qwen ordinary corpus contains synthetic weights only. Separate ignored tests use unmodified official `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen3-0.6B` checkpoints. Set `LLAMA_TEST_QWEN_HF_ROOT` to the external directory containing those two named checkpoint folders and `LLAMA_TEST_QWEN_OUTPUT_ROOT` to a fresh external output root. `cargo test -p llama-convert --release --test qwen_pretrained --offline --locked -- --ignored --nocapture --test-threads=1` checks every source-file hash and the complete F16, Q8_0 and auto/BF16 output hashes. Six outputs total 6,342,831,936 bytes per profile. The tests retain outputs and refuse to reuse a prior model output directory.

The separate GPT ignored tests use the pinned official GPT-2 and Pythia-14M checkpoints. They reproduce three original GPT-2 failures, then compare six complete F16/Q8_0/auto files totaling 718,209,760 bytes per native profile. When an external filtered GPT-2 fixture is not supplied, the Rust test removes the twelve source-rejected mask buffers and verifies the entire derived input against the independently captured checksum. The product converter never performs that fixture adaptation. See the GPT conversion document for input variables and the two tests.

The separate BLOOM ignored test uses the unchanged official `bigscience/bloom-560m` checkpoint at revision `ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971`. Set `LLAMA_TEST_BLOOM_HF` to the directory holding all seven files in its [pretrained manifest](tests/fixtures/bloom-conversion/pretrained-reference.json) and `LLAMA_TEST_BLOOM_OUTPUT_ROOT` to a fresh output directory. `cargo test -p llama-convert --release --test bloom_pretrained --offline --locked -- --ignored --nocapture` compares three complete F16/Q8_0/auto files, totaling 2,871,136,896 bytes per profile, and verifies input hashes before and after. It makes no input adaptations. The bundled BLOOM tokenizer data retains its BigScience BLOOM RAIL 1.0 license.

The separate StarCoder ignored test uses the unchanged official bigcode/tiny_starcoder_py checkpoint at revision 8547527bef0bc927268c1653cce6948c5c242dd1. Set LLAMA_TEST_STARCODER_HF to its pinned checkpoint directory and LLAMA_TEST_STARCODER_OUTPUT_ROOT to a fresh output directory. The starcoder_pretrained test checks all nine input hashes before and after conversion and three complete F16/Q8_0/auto files, totaling 881,027,328 bytes per native profile. The bundled StarCoder tokenizer data retains its complete BigCode Open RAIL-M v1 license. StarCoder2 pretrained HF weight conversion is not claimed.

Numerical checks compare exact bytes and F32 words, including signs and payloads of NaNs where the captured profile defines them. No approximate self-generated expected outputs replace the original references.

## Attribution

Retain [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [METADATA_NOTICES.md](METADATA_NOTICES.md) [ROPE_NOTICES.md](ROPE_NOTICES.md) [BLOOM_NOTICES.md](BLOOM_NOTICES.md) [FALCON_NOTICES.md](FALCON_NOTICES.md) and [STARCODER_NOTICES.md](STARCODER_NOTICES.md). The reviewed YAML dependency cohort, archive/source hashes, resolved features and complete license texts live under `docs/dependencies` and `release/reviewed-dependencies.json`. The dependency audit rejects unreviewed identities, source replacement, feature drift, source-cache changes, native links and build scripts.
