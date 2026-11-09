# Native context/state and remaining vocabulary C surface

The Rust FFI now exports 184 of the frozen 255 functions, with 71 still absent
and four inspection/diagnostic extensions. This cohort adds ten memory APIs,
18 state/session APIs and the vocabulary suppression getter.

Source ownership stayed within the approved FFI files, separate NativeContext
modules, a narrow dense-Llama physical-row insertion hook and module re-exports.
The existing strict Context implementation was left intact. The shared runtime
header was updated to describe current packed numerical paths and partial ABI
scope accurately.

## Verified behavior

- 40 original memory scenarios, 40 sequence-state scenarios, two session suites
  and 96 insufficient-buffer cases for both a trained model and a committed
  deterministic tiny model.
- 130 binary state/session artifacts per model: 1,329,186 trained bytes and
  58,890 tiny bytes. Original raw buffers restore and reserialize exactly.
- Windows debug/release generated data match bitwise, including 415,857 trained
  and 73,301 tiny float comparisons per profile across buffers and continuations.
- Linux metadata, IDs, boundaries and sentinels match exactly. Generated floats
  retain the predeclared 2e-4 absolute + 2e-5 relative model bound; maxima are
  5.7220458984375e-6 trained and 1.1920928955078125e-7 tiny in both profiles.
- The preserved Windows core harness passes 69 layout checks, every default
  field and 93,248 trained-model values. The complete prior sampler/context
  native capture remains exact.
- The extra C vocabulary adapter matches 347,066 entries / 27,508 tokenization
  cases across nine no-override corpora; a separate 176-case physical-metadata
  suite matches 23,903 entries / 480 cases and checks source failures safely.

Linux and Windows source numeric paths differ. Actual compiler macro probes show
GNU selects SSE2 vector exp for SiLU/softmax while the MSVC reference uses scalar
expf. Both F32 dot/sum branches use scalar products with F64 accumulation.
The evidence does not relabel those numerical differences as bitwise equality.

## Evidence layout

verification.json is the consolidated external-candidate result. windows/ and
linux/ preserve detailed native comparisons; vocabulary/ contains the independent
family reports. compiler/ records actual CMake flags and macro probes.
native-capture-manifest.json binds external expected captures by hash.
source-hashes.json binds the reviewed product source, and integration-copy.json
records the explicit file copy. jobs/ includes focused successful runs and
retained driver/output-directory failures without overwriting their logs.

The accepted driver report records two development failures transparently:
a Linux run preceded its output-directory creation; a Windows reader used
cp1252 for already completed UTF-8 native output. Their corrected runs retained
the old logs. The source partial cross-stream-copy assertion remains an explicit
unsupported input, not a passing test.

Full workspace integration checks are owned by the root release gate and are
separate from this focused evidence. No original C source, object, test harness
binary or library is distributed in the product runtime. Ordinary committed
fixtures were generated exclusively by the independent original source oracle.

Detailed support and lifetime contracts are in crates/ffi/STATE.md. Recurrent
native memory, other cache types/backends, duplicate positions in a batch,
separate coupled input, native partial-microbatch cancellation and the remaining
71 C functions still require further work.

## Integrated product verification

integrated-windows/ records a separate build directly from the product after the
typed-loader correction, using its own Cargo target. Both debug and release
reran every core, trained/tiny memory/state and full sampler/context native gate.
All passed with exact Windows values. The actual integrated release export
inventory remains 184/255 + four extensions. This fresh evidence supplements the
external-candidate reports above; it does not overwrite them.
