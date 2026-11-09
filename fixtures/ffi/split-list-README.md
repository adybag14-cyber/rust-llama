# Independent explicit C model split-list corpus

The frozen original llama_model_load_from_splits was measured in 104 Windows
and 106 GNU configurations before the Rust entry point was implemented.
Windows accepts 55 and rejects 49; GNU accepts 57 and rejects 49. The two
additional GNU successes use raw non-UTF8 Unix filenames. The two source
assertions per host are checked Rust errors.

Accepted cases check the original model byte/parameter/type/description
getters, vocabulary, token/piece/detokenization results and corresponding-profile
continuation logits. Windows compares 172,032 logit words; GNU compares 180,224.
These totals include repeated model/list configurations; they are not a claim
of that many unique numerical inputs. The originals differ in 58,590 shared
words (maximum absolute difference 5.960464477539063e-08), so separate exact
pools are required.

Caller lists cover arbitrary filenames, Unicode, raw Unix names, 35 shards,
metadata-only primaries, ignored single-file extras, missing/short/reordered/
duplicate inputs, typed reserved keys, truncated payloads and empty lists.
The source's existing supported C model/vocabulary architecture domain,
explicit F32/nonflash single-worker test configuration and pointer contracts
remain recorded. This does not establish complete public C ABI parity.

All fixture expectations come from the original C library. The product
executes Rust only. Original reports, sources, compiler identities, raw failed
captures and output hashes are retained in provenance/evidence.
