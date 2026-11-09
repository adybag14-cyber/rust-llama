# Original GNU embedding HTTP reference

This cohort contains 283 HTTP requests captured from the unchanged original llama.cpp source on GNU x86_64. It complements the existing Windows MSVC captures. It contains no Rust-generated expected results.

## Inputs and outputs

The three drivers retain the original request lists, paths, JSON number types, model identities and predeclared comparison budgets. The only driver changes are host/output paths, shared-library discovery and provenance wording.

| Corpus | Requests | HTTP 200 | HTTP 400 | Embedding F32 values including base64 |
| --- | ---: | ---: | ---: | ---: |
| Tiny BERT | 226 | 180 | 46 | 1,168 |
| Pretrained BGE small F16 | 27 | 23 | 4 | 23,424 |
| JSON coercions | 30 | 28 | 2 | 112 |
| Total | 283 | 231 | 52 | 24,704 |

The six output/provenance files and their exact hashes are listed in fixture-manifest.json. validation-report.json (SHA-256 c7f2e360696b0ac542d4f01d8d160001d9869144dff3c5af33af7614cdf6882e) compares original GNU against original Windows. All response statuses and non-embedding metadata agree. Across the two source compiler profiles, 23,786 embedding F32 bit patterns differ.

The largest source-to-source difference is in BGE normalization mode 0, which scales values into a large range: the component at model/norm_0/embeddings/0/embedding/0/294 is 11734.919921875 on Windows and 11733.4990234375 on GNU. This is a difference of 1.4208984375. Tiny/coercion normalization mode 0 also amplifies a small arithmetic difference to 0.001953125. The independent GNU fixtures preserve those outcomes; the Windows files and existing budgets have not been altered.

## Build provenance

Build/capture job: job-1794241249751-1775cb9a-dae, succeeded.

The original server was built in a new build-server-gnu-profile-v1 directory using the complete GGML option set copied from build-ggml-linux-llama. Both pre- and post-build hashes confirm that the previous original library tree was unchanged. The build enables only the server/tool targets needed for capture and disables UI/network asset provisioning. No original source file was patched.

Compiler: Ubuntu GNU C++ 11.4.0. Actual CPU flags are recorded in ggml-cpu.dir.flags.make and compile_commands.json: release O3, no AVX/FMA flags, with SSE2 defined by the x86_64 compiler baseline. SSE3/AVX/FMA are absent from the macro probe. GGML_NATIVE, AVX, AVX2, AVX512, F16C, FMA, SSE42, BMI2, OpenMP and LLAMAFILE options remain off. The original GNU source consequently selects its SSE2 row exp/variance branches while retaining scalar F32/F64 dot accumulation in the relevant dense kernels.

The server executable SHA-256 is 9d652dad5fe7d1af374f1a4f74b06b1149db10bf0c98c935219dc8cfdc622ff5. It dynamically loads the independently built original libraries, whose identities are in build-status.json and each corpus provenance. linked-libraries.txt records resolution. Full configure/build logs and all original server process logs are retained.

## Replay and integration

build_and_capture.py records configuration, builds the original server, runs all three adapted capture drivers, and verifies the request corpus and fixed budgets against the preserved Windows files. It uses create-only cohort/build paths. The source-to-source analysis is finalize_capture.py.

Use these new expected responses only for the measured GNU x86_64 profile. Root owns the product fixture names, selector integration and comparison of Rust HTTP responses against these original captures. That Rust comparison is not claimed by this capture-only handoff.
