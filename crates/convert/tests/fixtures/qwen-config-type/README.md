# Qwen config model-type resolution

This synthetic cohort records eight complete F32 conversions per source profile from the frozen original Qwen2/Qwen3 converter. It isolates the distinction between an absent `model_type`, explicit null, an explicit recognized type, and an explicit unknown type. It does not establish pretrained inference or AutoTokenizer parity.

Each architecture uses deterministic small tensors and a SentencePiece vocabulary. All cases omit `rope_theta` and `rms_norm_eps`. Qwen3 retains an explicit head dimension of 16, so a class-default head width does not obscure the model-type comparison.

## Observed original behavior

| Config field | Original AutoConfig selection | Converter hparams after its fallback | Default RoPE base and RMS epsilon |
| --- | --- | --- | --- |
| Explicit `qwen2` / `qwen3` | Qwen2Config / Qwen3Config | Normalized root config | Present: 10000 and 0.000001 |
| Field absent | LlamaConfig from the actual full path | Normalized root config | Present: 10000 and 0.000001 |
| Explicit null | ValueError | Original raw JSON | Absent |
| Explicit unknown string | ValueError | Original raw JSON | Absent |

The absent cases deliberately preserve what the source actually selected. Their literal paths contain both `llama` in `rust-llama-reference` and `qwen2` or `qwen3` in the case directory. The matching patterns have equal length; the pinned registration order selects `llama` first. The converter architecture remains Qwen2/Qwen3 because its architecture field is independently specified.

An absent model type therefore must not be interpreted as an automatic Qwen selection. The manifest records `source_input_path`, `independent_auto_config`, and a `replay_path_hint` for each case. Complete-file tests should create an input path containing the hint, such as `rust-llama-qwen2-type-absent`, so the source-relevant path markers remain present. A surrounding path containing another earlier/longer registered model pattern would require separate qualification.

Qwen2's explicit and absent cases have different complete output files because the selected LlamaConfig also supplies head-width metadata. The comparison preserves that difference. Null and unknown-type cases have identical output bytes for a given architecture.

## Files and integrity

- `cohort.bin` contains unchanged original input and complete GGUF output blobs. Only byte-identical blobs were deduplicated.
- `cohort.json` records both profile case lists, checked offsets, lengths and SHA-256 digests, conversion options, metadata probes, literal input paths and the independently observed AutoConfig result.
- `provenance.json` records the frozen archive identity, original converter and pinned Transformers source identities, environment reports, raw capture report/log hashes, and external harness identities.
- The included llama.cpp MIT and Transformers Apache-2.0 notices preserve the source licenses. Synthetic inputs and their harness authorship follow the repository's MIT license.

All eight original conversions succeeded on both Windows/MSVC and GNU/Linux. Corresponding inputs and complete outputs are byte-identical between the two profiles. Each profile produced 1,329,824 aggregate output bytes. The packed payload contains 17 distinct blobs totaling 1,158,353 bytes, with SHA-256:

```text
077262d99eadc7194c64a91a89dd8f84eb1ff01ddcc9783f411f6572a09638ea
```

The raw native reports remain outside the maintained repository under `continue-20261111/qwen-converter/config-type-oracle/native/{windows,linux}`. The existing 32-case Qwen config-edge corpus was not modified. No Rust implementation or Rust output participated in capture or expected-file construction.

## Capture isolation

The existing reference harness files execute cohorts at top level and have no import guard. The new external capture script therefore executes only the inspected function-definition prefix before `cases=[]`, redirecting its output-directory binding to this new cohort. The frozen original converter itself is unchanged. The harness captures its declared contract before conversion, verifies all input files remain unchanged, and records complete-file hashes. Independent AutoConfig calls then record the selected class or original exception without changing the inputs.
