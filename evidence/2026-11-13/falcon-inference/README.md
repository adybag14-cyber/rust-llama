# Falcon native runtime verification

This component adds executable frozen Falcon graphs and checked GGUF loading. It does not claim complete llama.cpp parity or pretrained Falcon quality.

## Results

- **windows-msvc:** 132 ordinary debug tests and 132 ordinary release tests passed, zero failures, 12 explicitly ignored external-data tests per run, 26 reported test suites. Formatting and strict Clippy passed.
- **linux-gnu:** 134 ordinary debug tests and 134 ordinary release tests passed, zero failures, 12 explicitly ignored external-data tests per run, 26 reported test suites. Formatting and strict Clippy passed.

Both profiles compare **412,024 exact F32 words and 243 exact greedy IDs** per debug/release replay, across 22 executed tiny variants. Fourteen further native inputs cover original errors or explicit unsupported configurations. State, cancellation, malformed metadata, optional norms and memory accounting are separately tested.

The successful stages are split between the original and continuation reports. Both first release attempts stopped before compilation because root was adding another workspace member and Cargo.lock needed synchronization. The original reports and errors remain unchanged. After the root synchronized the lockfile, release continuations passed. Debug, formatting and Clippy were not repeated.

Each component report records input hashes before and after its run. This guard covers only Rust/Cargo source files in the two affected crates; it does not freeze the whole repository. Differences, if any, are listed explicitly in report.json. Root performs combined workspace integration separately.

## Source and evidence

The authoritative frozen ZIP checksum is recorded in report.json. Original C/C++ writers and C callers captured both native profiles before the Rust Falcon implementation. Packed inputs and expected output arrays, their complete hashes, licenses and tool provenance are in [the fixture corpus](../../../fixtures/inference/falcon/README.md). This evidence directory retains all native case requests and stdout/stderr, plus execution reports and hashes for each validation command. Oracle source and original libraries remain external.

The graph supports shared or separate affine normalization, independent optional second bias, fused QKV, MHA/MQA/GQA, full-head NeoX RoPE, parallel GELU FFN and tied/untied output. Six matrix storage formats have direct native corpus coverage. [The contract](../../../docs/falcon-inference.md) describes unsupported configuration modes and resource limits.

No full pretrained Falcon model was executed in this component. Falcon-RW-1B has incompatible bias/ALiBi/residual semantics; Falcon7B requires more simultaneously retained decoded/packed memory than was available for a safe concurrent run. Synthetic inputs are not relabeled as pretrained proof.

Validation driver hashes in report.json are packaging-time identities, not pre-execution captures. Per-stage stdout/stderr hashes were captured by the driver immediately after each process finished.
