# Static C enum helper checkpoint

This archive binds the independently captured original bytes, checked native
Rust implementation, focused FFI regression, PE/ELF export tables and deliberate
fault tests for four frozen C helpers. All original libraries, Python drivers
and platform build tools remain external to the product. Their identities are
preserved; no original native implementation is linked or shipped here.

Both Windows/MSVC and GNU/Linux compare 73,057 file-type cases, nine defined
enum names, 1,542 valid byte-string parser cases and 43 static suffix-pointer
relationships in debug and release. Thirty original abort cases become checked
errors, with an additional null-string boundary. File-type outputs are already
in the exact fixture and are referenced rather than copied again.

The focused crate runs 37 Windows / 35 GNU tests per profile. Both hosts pass
formatting, checking, strict all-target Clippy, direct compiled-library checks,
the source/dependency audit and all 592 fixture declarations. Two isolated
production faults are compiled and caught on each host: skipping the guessed
bit mask and accepting string prefixes without the required NUL. Their selected
test must fail exactly once; compilation failures do not count as success.
Each mutation restores its clone, and the validated original source, expected
data and compiled libraries are rehashed unchanged.

Empty stdout/stderr files are recorded by exact zero-byte identities. Raw
non-UTF-8 diagnostic log bytes are preserved without text replacement. All
copied nonempty files, exact reused fixture outputs and external drivers are
listed separately. The pre-freeze test-decoder review correction and its
preimage are retained; production helper bytes did not change in that review.

The final historical archive is added after the successful source validation.
The external final-package receipt binds the whole final inventory and verifies
that the original 980 source/policy/license/prose/fixture files remain identical.
No recursive post-audit source edits are needed.

The current public surface contains 191 of 255 source C functions and four
diagnostic/layout extensions. Static descriptive helpers do not enable model
loading modes or backends. Twenty full-port release gates across twelve tracks
remain open.
