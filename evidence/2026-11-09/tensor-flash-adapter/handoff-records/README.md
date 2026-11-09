# Frozen tensor FLASH_ATTN_EXT handoff

Apply candidate/ as a tensor-only overlay after the sealed quant flash v1 and
kv_rows v2 handoffs. Exact product-1890 preimages of all modified existing files
are included for a guarded three-way merge. No Cargo, lock, shared manifest,
quant arithmetic, runtime, FFI or product edits are included.

Final adapter: 314232268bd5614a42ab7d45601ff321e7705436bbfa592e9eb6de2a54a68d1f.
Required shared kernel: f58cc7552796ee1094dbe35278a7ee0988bded8327bae8c9df35b4350ae45ec7.

Both x86_64 MSVC and GNU hosts passed full tensor debug/release, strict all-target
Clippy and formatting with all execution inputs unchanged. This includes all
prior tensor regressions, 114 newly admitted original flash graph cases and
896 reused sealed original graphs: 3,032,980 exact default/forced F32 words per
host/profile, including NaN payloads. Five admitted graphs have empty output.
Nine successful original graphs remain explicitly unsupported, and 30 original
assertion contracts are retained. The repeated 27-case abort input capture is
ordinary regression data, not 27 additional distinct source cases.

The operator count becomes 76 executable IDs plus NONE; 25 IDs remain pending.
The fixture manifest owner should add the 12 provided declarations. Expected
bytes/provenance are copied unchanged. Existing shared flash fixture rows must
remain intact. The shared quant test readers are reused by a tensor regression;
its kernel corpus is not duplicated or recaptured.

The source/budget review by /root is archived in evidence/independent-review.json.
Operator-local memory/cancellation limits and all remaining domains are explicit
in candidate/crates/tensor/FLASH_ATTENTION.md. The first exact Windows debug
comparison, initial test-only Clippy lint and the external preimage-copy setup
failure are preserved in evidence/first-check. No numeric tolerance changed.

This handoff is immutable. Further zero-width or other flash-domain work must
use a fresh isolated revision and separate original captures.
