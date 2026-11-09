# Native sampler and C ABI regression evidence — 2026-11-09

The new CPU sampler ABI and vocabulary-only WordPiece path passed the independent
frozen-header Windows x64 harness in both debug and release. The existing
model/context harness also passed again after the shared arithmetic and
tokenizer changes. This is evidence for the listed subset, not completion of the
full llama.cpp port.

## Recorded results per build profile

| Check | Result |
|---|---:|
| Frozen exports implemented | 137 / 255 |
| New frozen sampler functions | 30 |
| Frozen exports still absent | 118 |
| Rust diagnostic/layout extensions | 4 |
| New sampler size/alignment/offset checks | 36, all exact |
| Existing model/context/batch layout checks | 69, all exact |
| Original supported sampler fixture steps | 224 |
| Candidate-array apply comparisons | 1,401 |
| Candidate rows compared, including removed XTC prefixes | 45,949 |
| Sampler/WPM score floating-point comparisons | 122,388, maximum absolute error 0 |
| Counted lifecycle/error checks | 89 |
| Greedy and seeded-chain actual-context generated tokens | 192, all exact |
| WordPiece vocabulary entries | 30,522, all exact |
| WordPiece token pieces | Both special modes for every entry, all exact |
| Default WordPiece input cases | 192 |
| Detokenization | All four flag combinations, count-only and partial-buffer semantics |
| Existing core logit/embedding comparisons | 93,248, maximum absolute error 0 |
| Focused Rust lib tests | 31 behavior + 12 FFI, debug and release |
| Strict Clippy | Passed with warnings denied |

Each number above applies to each native build profile. The native sampler report
includes 30,522 vocabulary score comparisons in its floating-value count; the
remaining 91,866 values are candidate logits/probabilities. Core model/context
comparisons are recorded separately. All sampler constructors and state
transitions are exercised against the independently compiled original library.

The 192 generated tokens cover six trained stories260K prompts, 16 steps each,
through greedy sampling and through a seeded penalties/top-k/top-p/temperature/
distribution chain. llama_sampler_sample reads actual Rust-context logits and
accepts each selected token. The prior core harness additionally checks its
original greedy fixtures, sparse/negative output indexing, independent sequences,
embedding inputs/outputs, transactional batch rollback, cancellation and model
ownership after the loading handle is freed.

## Source-specific fixes established by the oracle

- Source candidate flags and selected-index semantics are preserved. Filters can
  leave a stale selected index, and XTC advances the data pointer.
- Native top-p's initial 256-item sort is retained. A full sort alone selected a
  different tied-token order for 4,097 candidates.
- MSVC heap, bucket and typical-score tie ordering is reproduced. Native captured
  cases are retained as Rust regression assertions.
- The MSVC MT19937 canonical mapping, including singleton discrete-sampling
  consumption, is preserved without changing the established high-level Rust
  sampler API's random stream.
- Logit bias preserves source duplicate and direct-versus-fallback lookup behavior.
- WordPiece uses normalized token attributes and special IDs, including inferred
  [PAD] end-of-generation behavior and the frozen seperator_token_id spelling.
- Detokenization returns a pre-cleanup required size for an insufficient buffer.
  Complete pieces that fitted before the first capacity failure remain written.
  Space cleanup can make a successful return shorter than the count-only result.
- Vocabulary-only model dimensions/descriptions retain the source's unloaded
  defaults. Native attention-head getters abort on those handles; the Rust
  implementation returns a diagnostic, and the harness does not invoke those
  invalid native calls. Public BERT architecture flags and RoPE enum follow
  the frozen source values even though BERT context execution is not yet exposed
  through this C layer.

## Artifacts and provenance

- verification.json records source, fixture, product, harness and reference hashes,
  exact results, job identifiers, OS DLL imports and scope restrictions.
- native-debug.json and native-release.json contain the new sampler/WPM results.
- core-debug.json and core-release.json retain the repeated original C ABI proof.
- exports.json and exports.tsv are generated from actual PE export tables and the
  frozen public header. The debug and release export sets are equal.
- Named log files record native runs, focused tests, builds and strict Clippy.

The frozen archive SHA-256 is
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
Its untouched include/llama.h SHA-256 is
a806cb05ff2f218945acefc0d5e49681f41d76af5d7ef7613d954fc18c81fe2a.

The native compiler is MSVC with _MSC_VER 1951; sorting/RNG headers are from
14.51.36231. Rust is 1.99.1 (58cf8f9c3, 2026-10-13), target
x86_64-pc-windows-msvc. The Microsoft STL adaptation and complete
Apache-2.0 WITH LLVM-exception text are preserved in the standalone
llama-behavior package. Its Cargo license expression is
MIT AND (Apache-2.0 WITH LLVM-exception); project and frozen upstream MIT
notices are also included. Root NOTICE identifies the separately licensed code.

The product DLL imports only OS/C-runtime libraries, including bcrypt.dll for
default random seeds. Neither llama.dll nor any ggml DLL is a product import.
Native source, C++ harnesses and original objects stay outside the product.

## Reproduction locations

The workspace is C:/Users/adyba/rust-llama. The original source and build are at
C:/Users/adyba/rust-llama-reference/llama.cpp-master and
C:/Users/adyba/rust-llama-reference/build/bin/Release.

Build the external sampler harness from
C:/Users/adyba/rust-llama-reference/sampler-oracle with CMake, the Visual Studio
18 2026 generator, x64 platform, and Release configuration. The executable
takes these five ordered arguments:

1. Original llama.dll.
2. Rust target/debug/llama_ffi.dll or target/release/llama_ffi.dll.
3. Repository directory.
4. External models/stories260K.gguf.
5. Output report JSON path.

The existing ffi-oracle executable uses a different order:

1. Rust DLL.
2. Original llama.dll.
3. External models/stories260K.gguf.
4. Repository fixtures/inference/stories260k-reference.json.
5. Output report JSON path.

The complete final PowerShell command sequence appears below.
It rebuilds both Rust DLLs, runs the two new sampler profiles and the two
original core profiles, and fails on any unsuccessful step.

Focused package checks are cargo test -p llama-ffi -p llama-behavior --lib,
the same command with --release, and cargo clippy -p llama-ffi
-p llama-behavior --all-targets -- -D warnings.

## Remaining limits

Exact native C++ tied-order and distribution parity is verified here only on
Windows/MSVC. The high-level Rust API and cross-platform compilation have
separate gates. This C context path remains scalar dense Llama with F32 KV.
BERT C execution, graph/backend sampling, performance counters, grammar, DRY
vocabulary breaker preprocessing, adaptive-p, infill, and the other 118 missing
public exports remain explicit work. No unsupported implementation returns
a fabricated success. See crates/ffi/SAMPLERS.md and COVERAGE.md.

## Exact final native command sequence

```powershell
$ErrorActionPreference='Stop'
cargo build -p llama-ffi 2>&1 | Tee-Object -FilePath 'evidence/2026-11-09/sampler-ffi/debug-build.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo build -p llama-ffi --release 2>&1 | Tee-Object -FilePath 'evidence/2026-11-09/sampler-ffi/release-build.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'C:\Users\adyba\rust-llama-reference\sampler-oracle\build\Release\sampler-oracle.exe' 'C:\Users\adyba\rust-llama-reference\build\bin\Release\llama.dll' 'C:\Users\adyba\rust-llama\target\debug\llama_ffi.dll' 'C:\Users\adyba\rust-llama' 'C:\Users\adyba\rust-llama-reference\models\stories260K.gguf' 'C:\Users\adyba\rust-llama\evidence\2026-11-09\sampler-ffi\native-debug.json' 2>&1 | Tee-Object -FilePath 'evidence/2026-11-09/sampler-ffi/native-debug.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'C:\Users\adyba\rust-llama-reference\sampler-oracle\build\Release\sampler-oracle.exe' 'C:\Users\adyba\rust-llama-reference\build\bin\Release\llama.dll' 'C:\Users\adyba\rust-llama\target\release\llama_ffi.dll' 'C:\Users\adyba\rust-llama' 'C:\Users\adyba\rust-llama-reference\models\stories260K.gguf' 'C:\Users\adyba\rust-llama\evidence\2026-11-09\sampler-ffi\native-release.json' 2>&1 | Tee-Object -FilePath 'evidence/2026-11-09/sampler-ffi/native-release.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'C:\Users\adyba\rust-llama-reference\ffi-oracle\build\Release\ffi-oracle.exe' 'C:\Users\adyba\rust-llama\target\debug\llama_ffi.dll' 'C:\Users\adyba\rust-llama-reference\build\bin\Release\llama.dll' 'C:\Users\adyba\rust-llama-reference\models\stories260K.gguf' 'C:\Users\adyba\rust-llama\fixtures\inference\stories260k-reference.json' 'C:\Users\adyba\rust-llama\evidence\2026-11-09\sampler-ffi\core-debug.json' 2>&1 | Tee-Object -FilePath 'evidence/2026-11-09/sampler-ffi/core-debug.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& 'C:\Users\adyba\rust-llama-reference\ffi-oracle\build\Release\ffi-oracle.exe' 'C:\Users\adyba\rust-llama\target\release\llama_ffi.dll' 'C:\Users\adyba\rust-llama-reference\build\bin\Release\llama.dll' 'C:\Users\adyba\rust-llama-reference\models\stories260K.gguf' 'C:\Users\adyba\rust-llama\fixtures\inference\stories260k-reference.json' 'C:\Users\adyba\rust-llama\evidence\2026-11-09\sampler-ffi\core-release.json' 2>&1 | Tee-Object -FilePath 'evidence/2026-11-09/sampler-ffi/core-release.log'
exit $LASTEXITCODE
```
