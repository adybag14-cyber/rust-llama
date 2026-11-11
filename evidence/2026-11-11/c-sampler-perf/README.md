# Native sampler-chain performance checkpoint

Five implemented C exports add real sampler-chain performance and global logging.
The original public-header harness was independently captured on MSVC and GNU
before implementation. It supplies 8 scenarios, 324 operations and 114 timing
observations per profile. Semantic projections are identical between hosts;
elapsed durations themselves are retained but never treated as exact constants.

Each platform folder retains original capture/build identity and raw timing JSON,
final debug/release native-client output, command logs, export sets and the full
source/fixture fingerprints. Both debug/release C clients match the independent
semantic projection and exact default stderr bytes. GNU passes all 49 FFI tests;
MSVC passes all 51. Formatting and Clippy with warnings denied pass on both.
Export tables contain exactly 216 frozen C names plus the four existing Rust
extensions, with 39 public functions still absent.

fixture-registration.json lists the five new inputs/reference/provenance/stream
files. Their expectations are authored or derived only from the original library.
Source, compiler, library and model hashes are retained with their captures.
The harness and build scripts remain external at the path in receipt.json.

GNU attempts v1 and v2 caught test-adapter compilation mistakes (an unavailable
serde_json import, then a partial move of callback test data). Final v3 uses the
existing native Rust JSON parser and preserves callback test-data ownership.
These adapter fixes changed no product behavior or native expectation. Earlier
attempt logs remain external.

See crates/ffi/PERFORMANCE.md for no_perf, nested timing, reset/clone/copy,
callback lifetime, checked errors, system CRT use and explicit remaining scope.
This is a CPU API checkpoint, not a full ABI or performance release closure.
