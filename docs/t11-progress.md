# T11: bounded Rust server transport checkpoint

> **Historical imported snapshot.** This report describes the transport-only
> T11 archive and its identified implementation checkpoint. Statements that the
> executable has no model runtime, embeddings return 501 or template rendering
> is disconnected apply to that earlier archive. The current implementation
> performs supported GGUF generation and BERT embeddings; see the
> [server contract](../crates/llama-server/README.md),
> [current port status](PORT_STATUS.md) and
> [expanded runtime verification](../evidence/2026-11-09/expanded-runtime/README.md).
> Historical limitations and route tables are retained below as provenance;
> the complete T11 requirements remain open under the [release gates](../release/gates.tsv).

## Status and scope

This is a working transport checkpoint, not completion of Track T11. The crate
`llama-server` contains a Rust HTTP executable, an embeddable inference boundary,
native/OpenAI text request handling, bounded streaming and replay, and a
Rust-authored diagnostic page. The default executable links
`UnavailableProvider`: **it has no model runtime and never fabricates generated
text**. Its health endpoint returns HTTP 503, its model list is empty, and valid
generation/tokenizer requests return typed HTTP 503 errors.

The complete CPU request → inference → stream → Rust application path remains
blocked on the T01–T08 runtime and model work. The deterministic provider used to
exercise request lifecycles lives only in `crates/llama-server/tests/tcp.rs`.
There is no production “demo model,” C/C++ wrapper, downloaded executable,
embedded legacy JavaScript, or third-party network dependency in this crate.
The only crate dependency is the workspace's Rust `llama-behavior`, used for
strict JSON parsing and exact integer serialization.

Implementation checkpoint: `c07eec6a44ee8088cd134064164ac473b23d9666`.

## Run and integrate

From the workspace, with the supplied Rust toolchain on `PATH`:

```sh
cargo run --offline -p llama-server -- --bind 127.0.0.1:8080
```

The command accepts `--bind`, repeatable `--api-key`, repeatable
`--allow-origin`, `--max-connections`, `--max-generations`, and `--help`.
The default binding is loopback. Enter `quit` on stdin for graceful shutdown;
stdin EOF leaves a daemonized listener running. This std-only checkpoint does
not install platform signal handlers. The host process may also embed the
library and call `RunningServer::shutdown` directly.

Minimal library startup:

```rust
use llama_server::{Server, ServerConfig, UnavailableProvider};
use std::sync::Arc;

let mut config = ServerConfig::default();
config.bind = "127.0.0.1:0".parse().unwrap();
let server = Server::new(config, Arc::new(UnavailableProvider))?
    .spawn()?;
println!("{}", server.local_addr());
server.shutdown()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

To connect the runtime, implement `InferenceProvider`:

| API | Contract |
| --- | --- |
| `status()` | Return `Ready` only when the real runtime can serve requests; otherwise return an explanatory `Unavailable(String)`. |
| `models()` | Return actual `ModelInfo { id, owned_by }` values. The default provider returns no models. |
| `generate(request, sink, cancel)` | Decode synchronously, emit valid UTF-8 through `TokenSink::send`, honor cancellation and sink errors, then return actual token counts and `FinishReason::{Stop, Length}`. |
| `tokenize(text, add_special)` | Optional model tokenizer implementation. The trait's default reports HTTP 501; `UnavailableProvider` reports HTTP 503. |
| `detokenize(tokens)` | Optional model detokenizer implementation with the same availability distinction. |

`GenerationRequest` contains `model`, `Prompt::{Text, Chat}`, `max_tokens`,
`temperature`, `seed`, `stop`, and the full parsed `parameters` object. Chat
messages have `role` and text `content`. Providers must validate any additional
parameters they support; transport preservation of a parameter is not a claim
that sampling, stop matching, grammar, or inference behavior has been wired up.

Public infrastructure is also usable independently:

- `http::{HttpLimits, Request, RequestParser, read_request, write_head}`.
- `stream::{CancellationToken, ByteQueue, QueueRead, QueueError}`.
- `stream::{ReplayStore, ReplaySession, ReplayInfo, ReplayRead, ReplayError}`.
- `stream::{sse_data, Utf8Decoder}`.
- `client::{escape_html, diagnostics}`.

`Server::new(config, Arc<dyn InferenceProvider>)` validates configuration;
`Server::spawn()` binds and starts the listener. `RunningServer::local_addr()`
reports the actual bound address, including an ephemeral port. A provider that
does not honor cancellation cannot be forcibly stopped safely by Rust threads.
Shutdown cancels all jobs, wakes request readers, and uses a bounded producer
grace period before detaching a noncooperating producer. Its admission slot
remains occupied until that producer actually exits.

## Route ledger

The source reference for ordinary registrations is
`tools/server/server.cpp:169–217`; replay registration is at 220–229, disabled
proxy/tools handlers at 241–303, and model-router registration at 529–533.
T12's source inventory contains the full lexical registration inventory.
The table below records T11's actual behavior, including partial schemas.

| Method and route(s) | Current Rust behavior |
| --- | --- |
| `GET /health`, `GET /v1/health` | Public health. Ready provider: 200 `{"status":"ok"}`. Missing runtime: 503 with `status`, `reason`, and `runtime_connected:false`. Not full source health schema parity. |
| `GET /models`, `GET /v1/models` | OpenAI-style list from real provider metadata. Default list is empty. Router state and source model metadata are not implemented. |
| `GET /props` | Honest checkpoint capabilities and resource limits, not the complete original properties schema. |
| `GET /metrics` | Rust request, handler-error, connection, and active-generation counters/gauges. Not the original inference metric set. |
| `POST /completion`, `POST /completions` | Native single text prompt. Both aliases are native in the source. Nonstream content/stop/token counters; streamed content events and a terminal native event. |
| `POST /v1/completions` | OpenAI single text completion with id/model/choices/usage; streaming text choices, finish reason, and `[DONE]`. |
| `POST /chat/completions`, `POST /v1/chat/completions` | Text-only chat. Nonstream assistant message/usage; streaming assistant role, content deltas, finish reason, and `[DONE]`. |
| `POST /tokenize`, `POST /detokenize` | Input validation followed by provider tokenizer/detokenizer; no byte-token approximation in production. |
| `GET /v1/stream` | Bounded replay from `conv_id` and raw-byte `from` offset, followed by live bytes. |
| `POST /v1/streams/lookup` | Known-id/prefix lookup with completion and byte metadata. Missing/non-array `conversation_ids` yields an empty list. Nonstring and empty items are ignored. |
| `DELETE /v1/stream` | Explicit cancellation and eviction. Idempotent 204 for an absent, valid id. |
| `GET /slots`, `GET/POST /lora-adapters`, `POST /props` | 403 `feature_disabled`. No synthetic slot/adapter/property mutation. |
| `GET/POST /tools`, `GET/POST /cors-proxy` | 403 `feature_disabled`, preserving the source's disabled default. No subprocess or proxy is launched. |
| `POST /slots/:id_slot` | 501 `not_implemented`. |
| `POST/DELETE /models`, `POST /models/load`, `POST /models/unload`, `GET /models/sse` | 501 `not_implemented`; no router, download, unload, or child lifecycle emulation. |
| `POST /responses`, `POST /v1/responses` | 501 `not_implemented`. |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | 501 `not_implemented`. |
| `POST /embedding`, `/embeddings`, `/v1/embeddings` | 501 `not_implemented`. |
| `POST /rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | 501 `not_implemented`. |
| `POST /audio/transcriptions`, `/v1/audio/transcriptions`, `/tts`, `/infill` | 501 `not_implemented`; T09 preprocessing is not an inference/audio generation runtime. |
| `POST /apply-template` | 501 `not_implemented`; T10 chat rendering is not yet connected to a model's selected template. |
| `POST /chat/completions/input_tokens`, `/v1/chat/completions/input_tokens`, `/responses/input_tokens`, `/v1/responses/input_tokens` | 501 `not_implemented`. |
| `POST /v1/chat/completions/control`, `POST /v1/systemone` | 501 `not_implemented`. |
| `WS /v1/realtime` | No WebSocket implementation; an HTTP request to the path receives 501. |
| Dynamic GCP routes | Not registered. `AIP_*` configuration and GCP protocol adaptation are pending; unknown routes receive 404. Source: `server-http.cpp:820–891`. |
| `GET /` | Rust-authored, HTML-escaped diagnostics with a restrictive CSP. This is not a WASM chat application. |
| `OPTIONS` | CORS preflight for the supported HTTP methods and explicit header allow-list. |

Supported generation input is deliberately narrow: one textual prompt or one
textual chat transcript, `n=1`, bounded positive token limit, nonnegative finite
temperature, unsigned seed, string/string-array stop values, and boolean stream.
The transport rejects tool/function transport, structured response formats,
logprobs, stream options, multimodal content, and unsupported chat roles with
typed errors. Negative native `n_predict` and batched prompts are not supported.
Runtime-owned sampler/stop behavior has no production validation until a real
provider is attached. This is route and lifecycle progress, not protocol parity.

## Resource, framing, and access behavior

| Default limit | Value |
| --- | --- |
| Header bytes | 16 KiB |
| Header count | 64 |
| Request target bytes | 4 KiB |
| Body bytes | 1 MiB |
| Concurrent accepted connections | 32 |
| Concurrent provider generations | 4 |
| Ordinary streaming queue | 64 KiB of raw bytes |
| Replay ring per session | 4 MiB |
| Replay session count | 32 |
| Completed replay retention | 300 seconds |
| Generated text per request | 8 MiB (`max_response_bytes`, before JSON framing/escaping) |
| Entire request read deadline | 10 seconds |
| Socket write / queue wait deadline | 5 seconds |
| Generation deadline | 300 seconds |

Configuration allows bounded overrides and rejects zero/invalid limits and
excessive individual sizes. The header parser uses checked length arithmetic,
incremental delimiter scanning, strict CRLF, ASCII header syntax, origin-form
targets, and a Host requirement for HTTP/1.1. It rejects duplicate headers,
ambiguous content lengths, folded headers, transfer encoding, and `Expect`.
Requests and responses use one connection each; keep-alive, chunked request
bodies, HTTP/2, and HTTP/3 are not implemented. Bytes following the first
complete request are not interpreted as another request.

SSE responses are close-delimited HTTP/1.1 bodies. Ordinary JSON responses have
Content-Length. Early connection-limit rejection writes a complete bounded 503
response without reading an arbitrary excess request body; clients should use
that Content-Length rather than assuming a clean EOF after an overload reply.

Request timeout covers the complete header and body, not each individual read.
The reader polls with at most 50 ms socket timeouts, checks shutdown, and refuses
slow-drip attempts to extend the deadline. The listener monitors absolute job
deadlines, including replay producers after every reader has disconnected.
Ordinary queue writes block only within the configured wait limit; network
backpressure has a separate real-TCP regression test.

Bearer authentication applies to all routes except GET health and preflight
when keys are configured. Token comparison processes the full supplied/key
length instead of returning at the first different byte; this is not a claim
of formally verified constant-time cryptography. CORS allows exact configured
origins or `*`; no credentialed wildcard mode is enabled. Allowed-origin CORS
headers also accompany authentication errors. The header allow-list covers
Authorization, Content-Type, and X-Conversation-Id. TLS is not implemented.
API keys share one process-wide model/replay namespace; tenant isolation is not
implemented.

Generated output bounds are text-byte bounds. JSON escaping and SSE metadata
add wire overhead. Queue and ring byte bounds are enforced on the actual wire
bytes. Provider model/error metadata and provider code itself remain a trusted
in-process boundary. There is no sandbox for a malicious provider.

## Replay and disconnect semantics

The source contract was inspected directly in `server-stream.cpp:17–214` and
`451–574` and `server-stream.h`. Replay stores the actual SSE byte stream, not
tokens or decoded characters. `from` can intentionally point into a UTF-8
codepoint or event; the bytes are returned exactly from that offset. A client
must retain its previous parser/decoder prefix across such a resume.

`X-Conversation-Id` enables replay on a streaming generation. With no such
header, generation uses the bounded queue and cancellation propagates when the
reader disconnects or writes stall. With the header, reader disconnect leaves
generation running into its ring so another reader can resume. An explicit
DELETE, id replacement, generation deadline, or shutdown cancels that producer.
Replacing an id finalizes and cancels its prior session. A provider must still
cooperate with cancellation.

Eviction retains the newest ring suffix. Reading an offset older than the
retained prefix gives HTTP 400 before headers; a reader overtaken by eviction
after headers is closed and must restart. A future offset waits for bytes or
completion. Missing/expired sessions give 404. Completed retention is checked
lazily on store operations; active sessions are not expired by completed-session
TTL. Capacity exhaustion returns 429. Lookup exposes only ids supplied by the
caller, plus matching `id::variant` sessions, and uses exact JSON integers for
byte counts and times. `completed_at` is zero until completion, matching source.

Two deliberate checkpoint differences are recorded: the Rust lookup de-duplicates
overlapping supplied ids, whereas the inspected C++ loop can repeat entries;
and offsets must contain only unsigned decimal digits, with duplicate query
parameters rejected. Rust lookup also adds `oldest_byte` so a client can inspect
the current retained prefix. Full transcript parity against a running original
model server has not been demonstrated.

`Utf8Decoder` is incremental across every byte boundary and remains poisoned
after invalid UTF-8. `sse_data` normalizes CR and CRLF before prefixing each data
line, preventing embedded carriage returns from becoming SSE field injection.
There is no browser EventSource/SSE application client in this checkpoint.

## Client, storage, native, and connect gaps

The diagnostic page and HTML escaping are implemented and tested as Rust
functions. No browser render, accessibility audit, WASM build, UI state model,
conversation branching, attachments, audio UI, settings UI, PWA, or native app
lifecycle has been verified. There is no new storage schema masquerading as a
migration of the original database.

The source storage contract was inspected in:

- `tools/ui/src/lib/services/migration.service.ts`: migrations copy values and
  preserve originals for downgrade compatibility; migration state is stored
  separately.
- `tools/ui/src/lib/constants/storage.constants.ts`: current `LlamaUi`, legacy
  localStorage `LlamaCppWebui`, legacy IndexedDB `LlamacppWebui`, and the
  `LlamaUi.streamResume.` key prefix.
- `tools/ui/src/lib/constants/database.constants.ts`: conversations and messages
  stores with their original indexed fields.

No browser storage is read, rewritten, or deleted by this implementation. A
future migration must preserve raw originals and unknown fields, stage writes
transactionally, validate success before recording completion, and retain exact
integer ids/timestamps. T10 now supplies exact signed/unsigned integer JSON
variants, but that alone does not establish IndexedDB/localStorage parity.

Remaining browser-library work includes PDF extraction/rendering, math and
diagram rendering, sanitization, arbitrary precision/symbolic math, and the
`run_javascript` tool's actual execution boundary. None is replaced with legacy
JavaScript hidden in a Rust string.

The Android reference remains `examples/llama.android`; the SwiftUI reference
remains `examples/llama.swiftui`. No Rust platform descriptors, Android APK,
Apple project, native launch/resume test, or Rust GGUF editor UI has been built
in this track.

`tools/connect/CMakeLists.txt` explicitly describes the connector as separately
developed and defaults `LLAMA_CONNECT_REPO` to `ggml-org/llama-connect` and
`LLAMA_CONNECT_VERSION` to `latest`, downloading a prebuilt binary. This
checkpoint neither downloads that binary nor treats it as a Rust rewrite.
Acquiring/pinning/auditing its separate source and verifying its tunnel protocol
remain pending.

## Validation evidence

The code checkpoint passes:

```text
cargo test -p llama-server --offline
  14 unit tests passed
  17 real local TCP integration tests passed
  0 failed, 0 ignored

cargo clippy -p llama-server --all-targets --offline -- -D warnings
  passed

cargo fmt -p llama-server
  applied
```

Unit cases cover every parser split boundary, malformed framing and smuggling,
size arithmetic overflow, all-or-nothing response header validation, UTF-8
split/error handling, bounded concurrent queue producers, blocked sender
cancellation, queue closure/timeouts, replay eviction/offsets/replacement/TTL,
and HTML/SSE escaping.

TCP cases exercise the unavailable default, both native aliases, OpenAI
completion/chat formats and usage, assistant role/deltas/final SSE framing,
auth/CORS/preflight, typed validation failures, provider token interfaces,
exact replay suffixes even inside UTF-8, lookup/delete/overflow, ordinary and
replay disconnect behavior, active-generation admission, producer errors and
panics, absolute slow-drip timeout, shutdown during a partial request,
disconnected replay deadlines, an actual nondraining socket's backpressure,
and connection admission.

These tests establish the implemented transport contract. They do not establish
model correctness, complete original protocol parity, throughput claims, TLS,
browser behavior, stored-data migration, subprocess crash recovery, or native
client certification. T12 performs the integrated workspace build, release
checks, default executable probe, inventory, and final package verification.
