# Rust model server

The executable loads supported GGUF models through the native Rust
llama-inference crate. It serves real CPU text generation and dense BERT
embeddings. A failed model load stops startup. Without --model, it runs a
diagnostics service whose inference requests return HTTP 503.

## Text generation

~~~sh
cargo build --release --locked --offline -p llama-server
llama-server --model /path/to/model.gguf --bind 127.0.0.1:8080
~~~

Native/OpenAI completions and SSE use actual model prefill, decode, tokenizer
IDs and the Rust sampler. Text chat and /apply-template use the loaded model's
supported native formatter. Models without a template, and templates requiring
unimplemented Jinja behavior, return explicit errors.

Generation accepts a positive token limit, nonnegative f32-compatible
temperature, an explicit u32 seed and stop strings. An omitted seed is **0**.
The original ambient-random seed sentinel is explicitly rejected. Token limits
are clamped to available model context. Unknown sampling options are rejected
before streaming headers or replay-session allocation.

Stop strings are withheld across token boundaries. Native stop-word and
end-token flags remain distinct; OpenAI reports finish_reason "stop" for
both. Each request owns its context and sampler state. Cancellation and sink
errors cannot carry partial sequence state into a later request.

The /tokenize and /detokenize routes use the loaded model vocabulary. The
transport currently parses special strings; parse_special=false and
with_pieces=true still return explicit unsupported errors.

## BERT embeddings

~~~sh
llama-server --model /path/to/bert.gguf --embeddings \
  --pooling model --bind 127.0.0.1:8080
~~~

The --embedding flag is an alias of --embeddings. Pooling accepts model, none,
mean, cls and last. The model setting uses the GGUF default; missing BERT
pooling metadata means none. The cls setting selects the first token.
Unsupported embedding architectures fail at startup when embedding execution
is requested.

| Route | Response |
| --- | --- |
| POST /embedding | Legacy alias of the native embedding route. |
| POST /embeddings | An array of objects with index and embedding fields. A pooled result contains one inner vector; none returns every token vector. |
| POST /v1/embeddings | OpenAI list/data/object fields, one flat vector per input, and prompt/total token usage. Requires pooling. |

For example, post this JSON to /v1/embeddings:

~~~json
{
  "input": ["A sentence to encode.", "Another sentence."],
  "encoding_format": "float"
}
~~~

Inputs preserve the frozen server's dispatch rules:

- A string is tokenized with model special-token insertion.
- An integer token array is used directly.
- A mixed text/token array is one sequence. Only a string in its first position
  receives automatic special tokens.
- Arrays without top-level integer elements are batches of input entries.
- An object with prompt_string accepts text or mixed text/token sequences.
- An object with a content array concatenates its text entries before
  tokenization. The source's text-only tool_reference behavior is also preserved.

The legacy top-level content key selects the native response shape, including
on /v1/embeddings; input takes precedence if both keys are present. The
OpenAI route still rejects none pooling before examining either key, as the
source does.

Pooled vectors use normalization 2 by default. The --embd-normalize N flag sets
the startup default, and request field embd_normalize overrides it. The source
modes are preserved: -1 leaves the vector unchanged; 0 scales its largest
absolute component to 32760; 1 uses the taxicab norm; 2 uses the Euclidean norm;
other integers use the source p-norm calculation. Unpooled token vectors remain
raw regardless of this option.

The base64 encoding_format returns the original native-endian F32 byte
representation in the OpenAI shape, with encoding_format "base64" on each
entry. The validated x64 targets use little-endian bytes. Native responses keep
float arrays even when this option is supplied, matching the original server.
Invalid encoding labels are rejected on either route.

The source JSON arithmetic conversions are retained, including boolean 0/1,
finite fractional truncation and the measured low-32-bit narrowing of integer
values. Token IDs are checked against the actual vocabulary after conversion.
Out-of-range floating-to-integer conversions are rejected instead of
reproducing source-undefined behavior.

A string model value is echoed as an output label, as in the original
single-model embedding handler. It does not load or select another model.
Unknown embedding fields are ignored where the frozen handler ignores them;
for example, dimensions does not project or shorten the output.

Media inputs require a multimodal runtime and currently return an explicit
unsupported error. Reranking and decoder-model embedding execution remain
separate open requirements.

## Resource and lifecycle behavior

Generation and embedding work share --max-generations admission capacity.
The --max-embedding-inputs limit defaults to 64 per request. Each input must fit
the loaded model context. The server bounds request bytes, response bytes,
connections, worker admission and deadlines through ServerConfig.

Embedding responses are assembled within the configured byte limit before any
success headers are sent. Oversized outputs, provider errors, invalid result
shapes, nonfinite values and provider panics return complete JSON errors.
A client disconnect cancels its work. A deadline returns HTTP 504, including
when the server monitor cancels the worker before the request handler's next
timer check. Shutdown cancels and joins active work and may close a response
before headers, matching the source's cancellation behavior.

## Validation

~~~sh
cargo test --locked --offline -p llama-server --all-targets
cargo test --release --locked --offline -p llama-server --all-targets
cargo clippy --locked --offline -p llama-server --all-targets -- -D warnings
cargo run --release --locked --offline -p llama-server \
  --example verify_model_server -- /path/to/stories260K.gguf
~~~

The committed embedding corpora contain **256 ordinary HTTP transcripts**
against a deterministic tiny BERT model and **27 explicit pretrained BGE HTTP
transcripts**. They come from the separately built original C++ server, not
from this Rust implementation. Coverage includes input shapes, ordering,
special-token insertion, all pooling modes, normalization, base64, model labels,
token usage, numeric conversions and typed errors.

Windows matches every compared output F32 bit, including seven base64 vectors.
Linux passes the fixed raw BERT error budget. Normalization 0 scales values into
an int16-sized range, so its portable end-to-end bound is derived from that
unchanged raw budget before comparison. Direct normalization of identical native
raw inputs is checked separately over 4,608 values, requiring exact Windows
bits and at most two F32 ULPs on portable targets. The formulas and observed
results are recorded in evidence/2026-11-09/server-embeddings.

Run the pretrained HTTP test explicitly after obtaining the hash-pinned model
listed in fixtures/inference/README.md:

~~~sh
LLAMA_TEST_BERT_MODEL=/path/to/bert-bge-small-f16.gguf \
cargo test --release --offline --locked -p llama-server \
  --test embedding_http -- --ignored --nocapture
~~~

The earlier generation verifier exercises 37 real-model HTTP requests against
six independently captured original outputs, including exact text/SSE/replay,
tokenizer round trips, seeded reset, cross-token stop handling, concurrency and
real cancellation/sink-failure recovery. The standalone executable is also
checked for successful model loading, failure before binding, and clean quit
shutdown.

Original C/C++ libraries are comparison tools outside this product; the Rust
executable does not load or invoke them. Remaining sampling options, routing,
batch scheduling, TLS, advanced protocols, tools/MCP and the complete browser
and native clients remain open full-port requirements.
