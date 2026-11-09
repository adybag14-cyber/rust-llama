# Rust model server

`llama-server --model PATH.gguf` now loads a supported model through the native
Rust `llama-inference` crate and serves actual CPU generation. Load failure stops
startup. Without `--model`, the executable remains a diagnostics-only service
whose health/generation report unavailable inference.

```text
cargo build --release --locked --offline -p llama-server
llama-server --model PATH.gguf --bind 127.0.0.1:8080
```

## Implemented integration

- Single text completion, native/OpenAI responses and SSE use actual model
  prefill, decode, tokenizer IDs and the Rust sampler.
- Text chat and `/apply-template` use the loaded model's supported native
  formatter. Models without a template and templates requiring unimplemented
  Jinja behavior return explicit errors.
- `/tokenize` and `/detokenize` use the model vocabulary. The exposed tokenizer
  interface parses special strings; `parse_special=false` and
  `with_pieces=true` currently return explicit unsupported errors.
- Stop strings are withheld across token boundaries. Native stop-word and
  end-token flags are distinct; OpenAI uses `finish_reason: "stop"` for both.
- Every request owns context and sampler state. Cancellation, sink errors and
  concurrent generations do not carry partial sequence state into later requests.
- Runtime options and prompt/context length are validated before streaming headers
  or replay-session allocation. Unknown sampling options are rejected.

Generation currently accepts a positive token limit, nonnegative f32-compatible
temperature, explicit u32 seed and stop strings. An omitted seed is **0**.
The original ambient-random sentinel is explicitly rejected. Token limits are
clamped to available model context by the runtime. Remaining sampling flags,
model routing, batch scheduling, TLS, advanced protocols, tools/MCP and browser/
native clients remain open requirements.

## Validation

```text
cargo test --locked --offline -p llama-server
cargo clippy --locked --offline -p llama-server --all-targets -- -D warnings
cargo run --locked --offline -p llama-server --example verify_model_server -- PATH/stories260K.gguf
```

The 9 November 2026 Windows run passed **38 ordinary tests** (18 unit and
20 real TCP cases). Six additional template transcripts come from the original
native formatter corpus. The explicit pretrained-model verifier passed
**37 HTTP requests** against six independently captured original model outputs,
including exact generation/SSE/replay, tokenizer round trips, seeded request
reset, cross-token stop handling, concurrent requests and real cancellation/
sink-failure recovery. The test model contains no chat template; unsupported
chat is therefore verified as an error, not a successful generation result.

A separate release executable check loaded the real model with `--model`,
returned health 200 and the exact original completion, reported usage 5 prompt
tokens plus 16 completion tokens, and exited cleanly on `quit`. A nonexistent
model stopped startup with exit code 1 before opening a listener.

See `fixtures/inference/README.md` for the external model identity and explicit
conformance commands. Original C/C++ libraries are comparison tools outside this
product; the Rust executable does not load or invoke them.
