# Final merged C API integration replay

Both debug and release shared libraries from the merged root checkout pass the
sealed original-header client replay on Windows x64 MSVC and GNU x86_64.

| Client family | Checked per library |
|---|---|
| Context-attached sampling | 26 scenarios, 390 accessor rows, 77,016 F32 words, 181,356 candidate IDs; every JSON value exact. |
| Ordinary CPU sampling suffix | Four intermediate stages, 2,500 F32 words, 1,250 candidate IDs; every JSON value exact. |
| Context performance and runtime information | Six scenarios, 124 observations; original ABI/null formatting, exact statuses/counters/logits, and existing structural live-timer checks. |
| FILE pointer model/adapter loading | 48 model and 24 adapter cases; exact statuses, metadata, invocation tokens and output words under the established coherent-cursor contract. |
| Named export inventory | Exactly 247 frozen functions plus four Rust extensions, compared with the independently built original 255-function library. |

No build or product implementation change is performed by the replay. The parent
declared the merged libraries ready before execution. Input model, expected
capture, client executable and client-source hashes were sealed beforehand.
Sampling expectations come from the independently captured original libraries.
The two older comparison functions are AST-identical to the 13 November driver;
their original expected outputs are preserved.

The runtime source fingerprint and library identities are checked before and
after each profile. The source fingerprints agree across profiles and are
rechecked before publication. summary.json and the profile reports identify the
exact libraries, commands and output hashes. Raw observations and logs remain
under C:/Users/adyba/rust-llama-reference/c-api-integration-20261114 and are listed
in external-artifact-manifest.tsv. Client sources and binaries remain outside the
Rust product.

Context timings remain live measurements; they are not compared with original
elapsed values. Direct-executor graph counters are zero. System information
describes the Rust CPU facade. FILE cursors restore their coherent input origins;
the original CRT/raw-handle post-positions remain outside parity. The detailed
contracts are in crates/ffi/{BACKEND_SAMPLING,CONTEXT_PERFORMANCE,FILE_POINTER}.md.
The sampler prefixes and backend/configuration limits documented there remain.

The remaining absent frozen functions are llama_attach_threadpool,
llama_detach_threadpool, llama_model_init_from_user, llama_model_save_to_file,
llama_numa_init, llama_opt_epoch, llama_opt_init and llama_opt_param_filter_all.
This integrated subset receipt does not close broader release gates.
