# Rust validation and packaging commands

Run these commands from the repository root with the pinned Rust toolchain installed:

```text
cargo xtask audit
cargo xtask fixtures
cargo xtask verify-source /path/to/extracted/llama.cpp-master
cargo xtask test
cargo xtask build
```

The workspace uses local path dependencies. The `xtask` alias supplies
`--offline --locked`; a clean clone needs the pinned toolchain but does not need
a downloaded Cargo dependency registry. The upstream source directory and any
optional pretrained model are separate inputs. The ordinary workspace tests do
not silently treat a missing optional model as a successful model comparison;
the explicit external-model test is documented in `fixtures/inference/README.md`.

## Fixture integrity

`cargo xtask fixtures` checks each declared SHA-256 and, where present, exact byte
length. It also rejects missing declarations for every file under a
`fixtures/` directory. It recognizes only these two complete TSV headers:

| Header | Meaning |
| --- | --- |
| `path\tsha256\torigin\tsource_identity` | Existing T09/T10 and integration declarations with their source identity. |
| `path\tbytes\tsha256\tpurpose` | Original T05–T08 declarations, preserving their size and purpose fields. |

The table writes tab separators as `\t` for readability; the files contain real
tabs. Every row must contain four nonempty fields. Paths must be safe relative
paths, SHA-256 values must be 64 lowercase hexadecimal characters, and byte
counts must be unsigned decimal integers. Duplicate paths within a manifest,
unknown headers, changed bytes, incorrect declared sizes, and unrecorded
fixtures fail the command.

Passing an integrity check establishes that data matches its declaration. It
does not establish that every declaration is an independent numerical oracle.
Domain tests perform the numerical comparisons; provenance files distinguish
frozen-source reference outputs from synthetic inputs and other test data.

## Source audit boundary

`cargo xtask audit` classifies maintained Rust, documentation, declarative
configuration, and test/reference data. It fails on unclassified files and
foreign implementation include candidates. Cargo metadata is checked for
registry/git dependencies and custom build scripts. The output explicitly lists
unsafe review candidates; passing this inventory audit is not proof that every
unsafe block or backend has been independently reviewed.

The CI exception is the exact path `.github/workflows/ci.yml`. Upstream/imported
licenses and notices retain explicit roles. Vocabulary-only GGUF files are
accepted only as fixture data and must pass the fixture-integrity command. The
four tiny Hexagon ELF fixtures have exact path exceptions, with hashes in
`docs/fixture-manifest.tsv`; these are parser/rewriter test inputs and expected
outputs, not product implementations. Other ELF or executable paths remain
unclassified.

The scanner excludes the repository-root `.git` metadata and root Cargo build
cache directories named `target` or `target-*` only when they have Cargo's standard
`CACHEDIR.TAG` signature. A directory name alone never exempts source files.
Nested target directories, unmarked target directories, and arbitrary `dist`
directories remain subject to the audit. Symlinks and special source files are
rejected. The root `PACKAGE_MANIFEST.tsv` is generated package metadata and is
recomputed during packaging.

## Recording and packaging

The Rust capture command stores stdout, stderr, exit status, elapsed time, and
SHA-256 digests:

```text
cargo xtask capture evidence/example/audit cargo xtask audit
```

Capture has a 30-second direct-child timeout and a bounded output budget. It
fails on timeout, truncated output, or a nonzero exit. It does not certify an
entire child process tree.

`cargo xtask verify-source` compares all frozen baseline paths, hashes, sizes,
and ownership assignments against `ledger/baseline-files.tsv`. It detects
added, removed, and modified files. The reference source is an external oracle,
not maintained product source.

`cargo xtask package /outside/source/checkpoint.zip` audits and writes a
deterministic source archive without overwriting an existing destination.
Choose a destination outside the source tree. `cargo xtask verify-package`
checks the generated package manifest after extraction.
`cargo xtask release-check` remains a separate completeness gate: partial
implementation and host-only evidence do not satisfy full-port release criteria.
