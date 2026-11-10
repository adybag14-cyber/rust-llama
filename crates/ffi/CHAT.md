# Native C chat formatting

This module adds the frozen `llama_chat_apply_template` and
`llama_chat_builtin_templates` exports. Formatting is native Rust in
`llama_behavior::chat`; no original llama library, Jinja interpreter, Python
or C++ implementation is used by the product.

The compatibility authority is the frozen source archive with SHA-256
`2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`,
specifically `include/llama.h`, `src/llama.cpp` and `src/llama-chat.cpp`.
The byte fixture and its source/binary identities are under
`fixtures/ffi/chat`.

## Input and output contract

`LlamaChatMessage` is the frozen C layout: two borrowed pointers, `role`
and `content`. Strings are NUL-terminated byte sequences, with no UTF-8
requirement. The first NUL terminates each value. A null template selects
`chatml`. Empty roles, empty contents, empty chats, arbitrary non-NUL bytes,
source template aliases and ordered detection heuristics are supported.

All message strings and their message array remain borrowed for the call.
Caller-owned input allocations must stay live and unchanged, output buffers
must be exclusively writable and disjoint from inputs, and returned builtin
name pointers must never be freed or modified. As with the rest of this C ABI,
unreadable, unterminated or incorrectly sized allocations cannot be validated
from a pointer. Null required strings, malformed bool bytes, null/nonzero
message arrays and null/nonzero builtin output arrays return a checked
diagnostic with the `-1` sentinel.

The formatter first produces a bounded private byte vector. No output is
written on a checked error or for an unknown template. Success returns the
full formatted byte length, regardless of the caller's buffer capacity.

The frozen wrapper calls `strncpy`, so its behavior is more specific than
a generic terminated-string copy:

| Output argument | Behavior |
| --- | --- |
| Null buffer, any signed length | Return full length without writing. |
| Non-null buffer, zero or negative length | Return full length without writing. |
| Positive length shorter than the result | Copy exactly that many bytes; append no NUL. |
| Positive length equal to the result | Copy the complete result; append no NUL. |
| Positive length larger than the result | Copy the result and zero-fill every remaining byte. |

Truncation can split a UTF-8 character, matching the byte-oriented original.
The buffer's declared capacity is a caller contract, not the rendered-output
allocation limit.

`llama_chat_builtin_templates` returns 54 for the frozen inventory. It writes
`min(len, 54)` permanent C-string pointers in lexical order, preserving the
original `std::map` enumeration. Remaining caller slots stay unchanged.
A null output with zero length is supported. Even `SIZE_MAX` only requires
space for the 54 written pointers. The Rust convenience list retains its
existing declaration order; the C API order has independent native fixtures.

## Shared Rust APIs and checked boundaries

`ByteMessage`, `detect_template_bytes`, `render_bytes` and
`render_bytes_with_limit` provide byte-preserving Rust access. The byte renderer
allows empty roles and requires role/content slices without embedded NULs;
foreign C strings are truncated before reaching it. The pre-existing UTF-8
`Message`/`render` API uses this same renderer after retaining its checked
empty-role/NUL restrictions and ASCII case-conversion behavior.

The C interface admits at most 65,536 messages, 1 MiB of template bytes,
16 MiB of cumulative role/content input and 16 MiB of formatted output.
Every explicitly allocated message/output/scratch byte vector uses fallible
reservation. These are work and logical-byte limits, not an aggregate
allocator or process-memory budget.

The measured source uses the C locale. Whitespace trimming is the six
C-locale ASCII whitespace bytes and case conversion is ASCII uppercasing.
Other process locales remain unverified.

The frozen OpenChat formatter passes its first role `char` to
`::toupper`; the three Bailing formatters pass every role `char`.
On signed-char targets bytes `0x80..=0xfe` produce negative arguments
outside that C function's defined domain. The native Rust C boundary rejects
those cases explicitly, preserving the defined `0xff`/EOF case.
Other role bytes and every content byte remain byte-preserving. The safe Rust
API keeps its earlier ASCII-only casing policy even for Unicode roles.
Null-string rejection is a conservative foreign-input boundary; the port does
not claim that every such malformed message necessarily causes a native fault.

## Independent validation

Before changing the Rust formatter, a separate C++17 caller was compiled
against the unchanged frozen C header and dynamically invoked the original
MSVC and GNU libraries. It captured 35,946 cases per platform:

- 34,349 successful formats and 1,597 unknown-template returns.
- 3,361,503 complete output-buffer bytes, including untouched guards.
- Every registered name, the unregistered `dots1` detection path, all 256
  content bytes, raw roles/templates, empty/NUL-terminated inputs and both
  assistant-prompt modes.
- 75 source-derived detection markers and 2,775 marker-pair precedence cases.
- Exact-size/short/oversized/nonpositive/null output buffers.
- The frozen message layout, all 54 builtin strings, bounded writes and stable
  pointers, including a `SIZE_MAX` capacity.

Both original captures are byte-identical and were sealed before implementation.
Ordinary Rust tests replay their bytes and return values. Additional tests
exercise malformed foreign inputs, cumulative/output limits, source-undefined
case conversions and preservation of the safe UTF-8 API. The same frozen-header
caller is used to replay the built Rust libraries in debug and release on
Windows and GNU/Linux; retained validation results live under
`evidence/2026-11-10/c-chat-api`.

This implements the frozen native formatter interface. Full Jinja, chat parsing,
tool/reasoning extraction, all language locales and the broader T10/T01 release
requirements remain open.
