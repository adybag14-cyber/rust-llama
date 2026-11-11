# Extended C batch checkpoint

The independent frozen header client matches the original on both x86-64 MSVC
and GNU, in debug and release Rust builds. Each replay checks 279 scenarios,
395 process calls, 640,000 exact logit words and 86,694 exact serialized state
bytes, plus ABI layout, builder return values, embeddings and context outputs.
All 13 new exports are real builder/decode behavior within the admitted contexts.

The platform folders contain full focused command reports and logs, with source,
fixture and library identities. PE/ELF inventories confirm 211 frozen C exports,
four diagnostic/layout extensions and 44 still absent. Windows runs 49 FFI tests;
GNU runs 47 because two existing tests are Windows-specific. All four test runs
pass; format and Clippy checks pass on both platforms.

The external header harness and reference libraries are under the path recorded
in receipt.json. Original captures predate the Rust implementation. The current
GNU harness source was also rebuilt: its executable and original replay are
byte-identical to the initially captured versions. fixture-registration.json
lists the six authored/native fixture files for integration into global manifests.

Earlier GNU validation v1 passed debug/release tests but found the Rust 1.99
Clippy as_chunks lint in a test helper. Validation v2 follows that mechanical fix;
no native expectation or acceptance rule changed. Both final v3 runs follow LF
normalization of owned source and compact-reference text. Original input bytes,
numeric binary pools and JSON values remain unchanged. A prematurely submitted
Windows v2 run was cancelled before normalization; its log remains external.
Earlier build attempts and
full uncompressed original/replay JSON remain outside the product.

See crates/ffi/BATCH_EXT.md for builder side effects, ownership, input ordering,
checked foreign input boundaries and unresolved context/encode capabilities.
This checkpoint does not close full C ABI, runtime or release gates.
