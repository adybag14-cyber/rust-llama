# Independent state resource review

The tensor/grammar reviewer inspected NativeContext, native_state and the C state
wrapper read-only before integration. The review found three concrete allocation
issues, all resolved before the accepted native/profile runs:

1. A metadata-only file-size check followed by read-to-EOF could exceed the limit
   if a file grew. The implementation now opens once, rejects nonregular files,
   and reads through a byte cap with fallible bounded buffering.
2. A tree per cache-cell sequence membership could exceed the KV-only budget.
   Fixed 256-ID bitsets remove those allocations. Cache/state transactions now
   preflight combined tensor, metadata, output and temporary storage, reserve
   large buffers fallibly, and compute size queries without serializing data.
3. Pending separate-stream copies allocated an extra full source stream during
   an already staged decode. They now select distinct streams with split_at_mut
   and copy into the existing destination buffers in queue order. Pending list
   growth uses exact fallible reservation.

The reviewer reread the final copy algorithm in source revision
e88df240b50571185d8ab8d9469a12565f1bb1a1142fdaadcd0cce26ed6dab2c,
before final formatting and the two module-local resource regressions were
appended. No subsequent algorithm change was made. The accepted product source
hashes are in source-hashes.json. Tests explicitly exercise fixed membership,
queued-copy ordering without buffer reallocation, checked budget overflow,
malformed-state atomicity and byte-capped reads.

The reviewer found no additional bounds error in the legacy length-less pointer
scanner under its documented complete-allocation foreign-caller contract.
That contract cannot make fabricated or truncated foreign allocations safe.

The review also identified a full-session failure side effect for verification:
trailing bytes are rejected after prompt outputs and restored memory have been
committed. The original library confirms this behavior on both platforms.
Independent captured fixtures and ordinary tests preserve it.

No approval exception or source assertion was treated as successful coverage.
Separate-stream partial copies, missing device-snapshot ownership, unsupported
cache types and duplicate batch positions remain explicit checked failures.
