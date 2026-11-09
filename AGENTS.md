# rust-llama implementation rules

- User objective: complete Rust port of the frozen llama.cpp ZIP, including all runtime/backend/application capabilities; no wrappers or renamed source.
- Baseline ZIP SHA-256: 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
- Read docs/plan.md for the 12-track contract. Every full release gate remains open until independently verified.
- Preserve source/fixture licenses and provenance. Keep reference C/C++ and build artifacts outside the product.
- All maintained implementation is Rust. No todo/unimplemented placeholders or false backend support claims.
- Use checked parser arithmetic, bounded inputs, explicit unsupported errors, and meaningful independent numerical fixtures.
- Hardware discovery is not execution parity. Synthetic inference is not pretrained-model parity.
- Never overwrite other workers changes; commit only after focused validation and integration.
- Keep generated builds/model downloads out of git. Do not change unrelated workspaces.
