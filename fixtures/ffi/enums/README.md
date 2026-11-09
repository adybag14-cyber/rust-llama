# Frozen C enum helper corpus

Expected values come from the archived original MSVC and GNU libraries, captured
and sealed before the Rust implementation. No model is needed. The external
capture driver and every nonempty/empty assertion log are identified by the
sealed external inventory; provenance.json binds the exact input matrix.

The 73,057 file-type inputs cover every unsigned 16-bit value, signed boundaries,
all signed values from -2048 through -1, source enum values with every individual
32-bit flag and guessed flag, and a deterministic signed 32-bit sample. These
produce 76 different strings. Nine defined enum-name inputs and 1,542 load-mode
strings include all six names and every byte after a first NUL. The source
ignores bytes after that terminator. Forty-three pointer cases record the
10-byte guessed-prefix suffix relationship and returned string bytes.

Thirty separately executed inputs abort in both original hosts: 12 invalid
enum names and 18 unknown load-mode strings. Rust returns a null name pointer
or INT32_MIN and sets its thread-local diagnostic for these cases. A null
input string is an additional checked Rust boundary; the original has undefined
pointer behavior and supplies no expected numerical result for it.

This closes these four static helper functions for the documented C ABI.
It does not establish a model load mode, backend, quantizer, or complete port.
