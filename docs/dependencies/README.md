# Reviewed YAML dependency cohort

The dense Llama converter uses yaml-rust2 0.13.0 with default features disabled. This is the only direct registry dependency. Its complete runtime closure is five packages:

| Package | Version | Resolved features |
| --- | --- | --- |
| yaml-rust2 | 0.13.0 | none |
| arraydeque | 0.5.1 | default, std |
| hashlink | 0.12.2 | none |
| hashbrown | 0.17.1 | default-hasher |
| foldhash | 0.2.0 | none |

The pinned review is in yaml-rust2-review.json. release/reviewed-dependencies.json binds every archive checksum, source-file path, byte length and SHA-256, plus version, registry identity, license and resolved default features. The review covers 132 materialized source files. Complete license texts are under licenses/.

`cargo run -p xtask --offline --locked -- dependencies` verifies the actual local graph and Cargo caches without network access. `xtask audit` performs this check together with the source-language audit. A fresh checkout must first acquire the checksum-pinned crates with `cargo fetch --locked`; offline gates cannot fetch missing packages.

The audit verifies policy/review hashes, lock identities, offline locked Cargo metadata, exact feature sets and licenses, absence of native links/build targets, extracted source inventories and actual bytes, and archive bytes/checksums. The locked package set must equal Cargo's resolved package set. Local path manifests must remain inside the audited workspace; registry manifests must be under the active Cargo source cache. Source replacement, additional registry packages, changed package versions/checksums, extra features, native source files, build scripts, and changed/untracked extracted files fail.

No registry development dependencies are built merely because these crates are used by the converter. This validates the default consumer graph. It does not establish every optional feature/backend combination, all dependency vulnerabilities, or full YAML/PyYAML semantic parity. The converter's separate independent model-card corpus validates its admitted YAML behavior.

Tests exercise unknown package, version, checksum, source, feature, license, native-link and build-target rejection, malformed policies/paths, exact source bytes, extra source files and Cargo extraction-marker drift. The audit is a diagnostic against a stable local dependency cache, not a capability mechanism against concurrent hostile cache mutation.

## Bootstrap and CI

1. Install the repository's pinned Rust toolchain.
2. Run `cargo fetch --locked` once in the clean checkout.
3. Run the normal offline locked builds/tests and `xtask audit`.
4. Preserve Cargo's registry archive cache as well as extracted source files when caching dependencies. Both are checked.

Changing the cohort requires a new explicit source/feature/license review, policy identity, tests and retained evidence. Do not broaden the audit to accept arbitrary registry packages or remove source-cache checks to make a build pass.
