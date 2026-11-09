# Frozen original GGUF byte-string and boolean observations

The two data pools and four complete source reports are captured from the original
C implementation on Windows and GNU before the corresponding Rust implementation.
The 1,605 cases include all byte values separately in strings, arrays, keys and
tensor names, invalid UTF-8, C-string prefix aliases, length limits, typed raw
metadata including NaNs, and every scalar/array boolean byte encoding. Both native
profiles have identical case records and pools; their identities remain separate.

The TSV files are a lossless selection of observations for dependency-free Rust
tests. Original reports and each pool slice retain complete SHA-256 provenance.
No expected value is derived from Rust. Tests compare header-only admission,
every captured getter and 269,184 exact serialized header bytes. Full-file payload
validation and the text facade have explicit separate contracts.
