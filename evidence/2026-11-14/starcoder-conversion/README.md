# StarCoder converter acceptance, 2026-11-14

This evidence qualifies native Rust Hugging Face conversion for the frozen
GPTBigCodeForCausalLM and Starcoder2ForCausalLM classes. It does not close a
repository-wide port or release gate.

## Verified result

Each Windows MSVC and GNU/Linux target passes the independently captured
194-case corpus in debug and release: 163 complete files / 357,723,535 exact
bytes, with all 31 original rejections preserved. The 1,248 tensor-map requests
also pass. Official GPT-2, tiny StarCoder and StarCoder2 tokenizer sidecars
exercise both the shared and architecture-specific paths.

Both hosts also pass unchanged pretrained tiny StarCoder F16/Q8_0/auto
conversion: three complete files / 881,027,328 bytes. All nine original input
files are checked before and after. The original converters agree on those
complete output hashes across both hosts. No input adaptation is used.

Both targets pass the full converter release suite and Clippy with warnings
denied. The focused debug run includes the new complete-file and alias tests.
The all-converter run provides regression coverage for the shared AutoConfig,
class-selection, vocabulary and dtype paths changed by this extension.
The ordinary ignored pretrained tests for unrelated existing families remain
ignored; no fresh execution of those model downloads is claimed here.

The repository selects Rust 1.99.1 on both targets. toolchains.json records
version/commit identities in the same repository context as validation and
the unchanged toolchain selection file. The separately retained probe of
Windows rustup's default outside the repository is diagnostic only; its
1.91.1 result was not the toolchain used to compile these tests.

## Evidence layout

- rust/<host>/<stage> stores actual argv, exit status, duration, raw stdout and
  stderr, and complete log identities for each successful Rust command.
- validation/<host> stores before/after input guards and the completed
  sequential run report. The tested source/fixtures remain unchanged during
  each run.
- validated-inputs.json rechecks the final executable Rust and numerical
  fixture/expected-file identities after documentation-only notice additions.
- original/<host>/pretrained.json records unchanged pretrained source files,
  original full GGUF output hashes and capture-program identity.
- launchers retains original process commands and capture output, including
  historical failed capture attempts where present.
- external-artifacts.json identifies retained complete original reports and
  external helper programs by full path, byte count and SHA-256.
- fixture-summary.json records the 176-blob, 38,284,433-byte deduplicated
  numerical/tokenizer input pack.
- historical/windows-178-debug retains the earlier, smaller successful
  pre-official-sidecar corpus. It is not substituted for the final 194 cases.

The reference environment uses the original archive and pinned native
Python/Torch/NumPy/Transformers/Tokenizers packages. Exact source and package
identities are in the fixture provenance. Expected outputs are captured
independently from that original implementation and compared by complete
length and SHA-256; no Rust-produced expected data or numerical tolerance
is used.

The Windows and GNU alias capture JSON files differ only in native CRLF/LF
serialization. Packaging proves equality after newline normalization and
retains both original raw capture hashes. The initial packaging assertion
that required identical serialization was corrected before validation; it did
not change any original lookup result or converter output.

A sidecar acquisition attempt encountered a connection reset before source
capture. The successful retry retains the same pinned revision and complete
Git/blob/SHA-256 verification. The original reference captures and Rust
validation run independently of network access.

## Boundaries

StarCoder2 unchanged pretrained HF weight conversion is not claimed. Its
sidecars are official and revision-pinned; its ordinary numerical tensors are
synthetic. Any separately measured StarCoder2 GGUF inference belongs to the
model implementation evidence. This conversion change introduces no new
registry dependency, foreign runtime or reference source inside the product.

The complete source/data notices include MIT, GPT-2 Modified MIT,
Apache-2.0 and BigCode Open RAIL-M v1. Fixture declarations are in
docs/starcoder-conversion-fixture-manifest.tsv. The detailed admitted
conversion contract is docs/STARCODER_HF_CONVERSION.md.

The fixture-local .gitattributes marks pipeline.bin as binary. Its leading
tokenizer JSON would otherwise make Git interpret later packed tensor bytes
as text. This review classification preserves every validated payload byte.
