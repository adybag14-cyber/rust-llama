# Native C chat interface evidence

This evidence covers the two frozen C chat exports and their shared native Rust
byte formatter at base commit b658976bd6200b8aa4b29f29f6a896b126dd756b.
The input map in each host directory records every executed Rust source,
workspace manifest/lockfile and chat fixture by byte length and SHA-256.
Both hosts used identical maps; the JSON files differ only in line endings.
The summary records their common normalized fingerprint.

## Independent reference and ABI replay

The original Windows/MSVC DLL and GNU/Linux SO each produced the same
3,937,927-byte capture before Rust implementation. The immutable original
seal and complete provenance are in fixtures/ffi/chat. The 35,946 cases
include 34,349 successful renders, 1,597 unknown-template rejections,
3,361,503 complete buffer bytes, all 54 builtin names, native message
layout, stable pointers and bounded writes. The matrix includes every
content byte, source-derived detection precedence, both assistant flags,
first-NUL string behavior and count-only/truncated/exact/zero-padded buffers.

The unchanged C++17 caller compiled against the frozen llama.h then
dynamically invoked each Rust debug and release DLL/SO. All four complete
captures matched the original byte for byte. The caller, its source, the
original binaries and actual replay binaries remain outside the product at
C:/Users/adyba/rust-llama-reference/continue-20261110/chat-oracle.
Hashes and invocation arguments are retained in each host status.json;
the replay result hash is the same as fixtures/ffi/chat/original.bin.

| Platform | Debug tests | Release tests | C replay cases per profile |
| --- | ---: | ---: | ---: |
| Windows MSVC | 95 | 95 | 35,946 |
| GNU/Linux | 93 | 93 | 35,946 |

Both profiles have zero failed or ignored tests. The two-test count difference
is existing platform-gated FFI coverage. Each host also passed focused
cargo fmt --check, strict Clippy with -D warnings, both library builds,
xtask audit and xtask fixtures; exact commands and logs are preserved.
The audit is inventory/accounting validation and does not close release gates.

## Validation history and limits

The first external Windows caller build exposed the standard Windows
min/max macro collision; its CMake configuration gained NOMINMAX before
any original output was captured. A preliminary strict Clippy run identified
a test-only signedness comparison, which was corrected before these final
runs. No expected output was regenerated to match Rust. The original seal,
original inputs, and executed Rust/fixture input hashes were checked unchanged
before and after final validation.

The measured locale is C. The C boundary explicitly rejects invalid pointers
where detectable, malformed boolean bytes, documented work/byte limits and
source-undefined signed-negative case-conversion arguments other than EOF.
Unterminated pointers and incorrectly sized allocations remain the caller's
C contract. The CHAT.md document specifies these boundaries.

The full Jinja language, parser/tool/reasoning extraction, other locales and
all broader T10/T01 requirements remain open. This directory documents a
completed native C formatting slice, not completion of the full Rust port.
