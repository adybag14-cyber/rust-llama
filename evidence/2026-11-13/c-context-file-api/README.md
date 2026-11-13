# Context performance and FILE pointer validation

This checkpoint implements seven frozen C exports and leaves 16 absent.
The unchanged native header declares 255 functions. Each final Windows/GNU
debug/release shared library exposes 239 of those functions and exactly four
existing Rust diagnostic/layout extensions.

## Completed checks

| Profile | FFI tests | GGUF tests | Strict Clippy | Original-header clients | Frozen exports |
|---|---:|---:|---|---|---:|
| Windows MSVC debug | 71 passed, 1 ignored | 30 passed | Passed | Both passed | 239 |
| Windows MSVC release | 71 passed, 1 ignored | 30 passed | Same source checked | Both passed | 239 |
| GNU Linux debug | 69 passed, 1 ignored | 31 passed | Passed | Both passed | 239 |
| GNU Linux release | 69 passed, 1 ignored | 31 passed | Same source checked | Both passed | 239 |

The existing ignored test requires the separately pinned pretrained BERT model.
Platform-specific test counts differ because some tests are conditionally
compiled. All commands use Rust 1.99.1 with offline, locked dependency resolution.
The product static and shared library targets build; the external native clients
load the shared libraries.

Each context client checks six scenarios and 124 observations. ABI layout, null
printing, status, counts and final numerical values are compared against the
original native library. Timings are real nonnegative measurements with source
synchronization/reset relations, not exact reference durations. Printing unit
tests separately compare frozen formatting vectors.

Each FILE client checks 48 model and 24 adapter cases, with exact status,
metadata, invocation tokens and output float words. The client closes every
input FILE before inference, so success also exercises owned weight lifetimes.
Three safe GGUF tests exercise retained file identity, relative offsets, parser
limits, malformed input and anonymous split handling.

## Explicit comparison boundaries

- The direct Rust executor reports zero GGML graph reuse/reallocation counts.
- System information identifies actual Rust CPU features rather than copying
  native host ISA/backend feature claims.
- Rust coherently restores FILE cursors to their input origins. Native final
  positions are retained in original captures, but no equivalence is claimed
  for the source's platform-dependent CRT/raw-handle buffering.

These are operational subset receipts, not closure of all ABI/backend/model
architecture/performance release gates. CONTEXT_PERFORMANCE.md and
FILE_POINTER.md specify the implemented contracts. COVERAGE.md also records the
complete prerequisites for the still-absent user-model constructor and saver.

## Retained records

The msvc and gnu directories contain complete command logs, raw original
observations, raw Rust client observations and report-all.json. Each report
includes exact commands, library/output SHA-256 identities and absent exports.
validation-snapshot.json records the base revision, Cargo.lock identity, owned
changed files, external client/driver identities and successful Hermes job IDs.
manifest.tsv hashes the evidence files.

The C++ header clients, build projects, validation driver, original libraries and
build artifacts remain outside the product tree under rust-llama-reference.
Committed fixtures are data only. No original llama/ggml library is linked or
loaded by the Rust product. The test driver dynamically loads a specified
library in a separate process and never supplies Rust-generated expected data.
