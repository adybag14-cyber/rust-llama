# Independent C split filename helpers

7,807 unique original calls per MSVC/GNU host, captured before Rust comparison.
The hosts agree on every return and all 506,563 complete output-buffer bytes,
including guard bytes; 256,739 bytes lie within declared output capacities.

The corpus includes every input byte, embedded NUL C-string termination,
Unicode/invalid UTF-8 bytes, short/zero/null-zero buffers, signed limits and
minimum-width decimal formatting. The prefix helper preserves the original
32-byte local postfix buffer truncation. Its successful truncated/zero-capacity
return differs from the path formatter's zero-on-short-output return.

Source signed overflow for split_no==INT32_MAX is not a successful parity
requirement. Rust guards it and invalid pointer/count arguments with errors.
The fixtures are native expected data, never a product dependency on C/C++.
