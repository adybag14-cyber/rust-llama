# Dense Qwen3 and explicit F16/F32 native caches

This checkpoint is part of the incomplete native Rust port. It does not close
any of the twelve global release gates.

The complete combined workspace passes debug/release tests, strict Clippy,
source/dependency and fixture audits, release builds, independent pretrained
comparisons, CLI checks and real HTTP/SSE checks on Windows MSVC and Linux GNU.
Each step rechecks the same 637 execution inputs, with SHA-256
89a863d2145bee98206987803404e1551a9edb45c5174a9ff550cae993e42343.

Windows has 451 ordinary passing tests per build profile; GNU has 454.
Nineteen prerequisite tests are ignored in the ordinary workspace run.
The separately invoked pretrained suite has nine passing tests per host,
including the new official Qwen3 Q8_0 model; all measured numeric comparisons
record zero observed error against the corresponding original profile.

Qwen3 supports the bounded dense causal LM graph with independent head width,
learned per-head Q/K RMS normalization, full-head NeoX RoPE, separate/fused QKV,
native bias precedence and tied/untied output. The ordinary original corpus
compares 294,912 logits and 400 greedy IDs per host; the pretrained corpus adds
5,469,696 unique logits and 24 IDs. Loader/state/cancellation/allocation behavior
remains explicit in docs/QWEN3_IMPLEMENTATION.md.

The Llama NativeContext independently stores keys and values as F32 or raw F16
words. The native ordinary corpus checks all four type pairs, 258,048 exact
logits and 1,048 memory/sequence/session buffers per host. Full external C
cohorts check 491,520 logits per host/build profile. Raw half state preservation
does not remove checked nonfinite-result errors during evaluation. This
checkpoint admits explicit disabled flash and one generation/batch worker.
Original defaults remain unchanged and rejected; AUTO, real flash/microbatch
execution and other worker counts are separate work.

Failed comparisons, negative controls, source identity and earlier integration
failures are retained. Handwritten C/C++ oracle builds and pretrained weights
remain outside the product. Historical driver/source/patch text is archived
with a .txt suffix and original byte hashes; it is not a product implementation.

The previous published checkpoint e0af03a also passed both hosted CI jobs on
the identical commit after GitHub's pre-run internal-server-error attempt.
The first attempt and successful raw logs are retained below.

See summary.json, source-provenance.json and the original handoff and complete
integration reports for precise scope, source hashes and test metrics.
