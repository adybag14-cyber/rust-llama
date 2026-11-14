# Independent context-sampling fixtures

The original frozen llama.cpp libraries produced all four expected JSON files.
The input is the existing deterministic native-context-model.gguf; this corpus
is not pretrained-model acceptance. No Rust output generates expected values.

Each original platform capture covers 26 scenarios, 390 accessor rows, 77,016
F32 logit/probability/query words and 181,356 candidate IDs. The separate CPU
suffix files record the complete typical and distribution stage arrays for two
contexts: 1,250 candidate IDs and 2,500 F32 words across four stages. Their
source/compiler identities are in provenance.json.
All numeric words, IDs, strings, counts and pointer-presence/alias fields compare
exactly. The platform-specific files preserve their measured tie order.

The external C++ clients include the untouched frozen llama.h and dynamically
load the independently built original library before Rust replay. Client source,
executables, library, model and output hashes are recorded in provenance.json;
the raw C++ clients and native binaries remain outside the product. The original
archive's authoritative SHA-256 is
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.

The corpus retains initial/attached/decoded/invalid/detached/raw-after-detach
accessor state, mixed sequence ordering, empty/absent samplers, tied greedy and
top-k, positive/zero/negative temperatures, a source CPU suffix, initial sampler
configuration and output-limit rejection/recovery. The suffix diagnostic exposed
and now guards the former GNU stable-sort mismatch. Earlier invalid native
harness inputs, failed Rust tests and correction logs remain identified in the
provenance file rather than replacing the accepted original outputs.

fixture-manifest.tsv records the committed JSON byte counts and SHA-256 values.
crates/ffi/BACKEND_SAMPLING.md gives the supported domain and explicit limits.
evidence/2026-11-14/c-backend-sampling records debug/release acceptance.
