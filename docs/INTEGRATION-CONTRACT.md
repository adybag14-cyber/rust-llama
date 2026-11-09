# T05–T08 integration checkpoint

This workspace is a standalone progress branch for the four requested tracks. T01–T04 implementations were not supplied. Each domain crate exposes safe, owned Rust types. Provisional internal descriptors must be adapted to the T01/T02 contracts on integration; they do not claim a compatible public llama ABI.

Coordinator owns workspace Cargo.toml, lockfile, platform loader, kernel IR and portable GPU packages. T05 owns crates/model-runtime; T06 owns crates/device-family; T08 owns crates/vendor-remote. No runtime dependency on the original source, no fake success on unavailable devices, no claiming registry entries are implemented models. External ABI calls must have explicit unsafe safety comments and retain the loaded library through object destruction.

All work must compile offline without crates.io dependencies with the supplied Rust 1.99.1. Track tests may use isolated reference programs as oracles, but the product is entirely Rust-authored.
