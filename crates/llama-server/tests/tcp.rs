//! Real local TCP lifecycle tests. The deterministic provider is test code only.
use llama_behavior::json::{self, Value};
use llama_server::provider::FinishReason;
use llama_server::*;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum Mode {
    Finite,
    Gate,
    Endless,
    Fail,
    Panic,
    Flood,
}
struct FixtureProvider {
    mode: Mode,
    started: AtomicUsize,
    cancelled: AtomicUsize,
    release: AtomicBool,
}
impl FixtureProvider {
    fn new(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            started: AtomicUsize::new(0),
            cancelled: AtomicUsize::new(0),
            release: AtomicBool::new(false),
        })
    }
    fn send(&self, sink: &mut dyn TokenSink, text: &str) -> Result<(), ProviderError> {
        sink.send(text).inspect_err(|_| {
            self.cancelled.fetch_add(1, Ordering::SeqCst);
        })
    }
}
impl InferenceProvider for FixtureProvider {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::Ready
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "test-only".into(),
            owned_by: "test-fixture".into(),
        }]
    }
    fn generate(
        &self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
        cancel: &CancellationToken,
    ) -> Result<GenerationSummary, ProviderError> {
        self.started.fetch_add(1, Ordering::SeqCst);
        assert!(request.max_tokens > 0);
        if matches!(self.mode, Mode::Fail) {
            return Err(ProviderError::internal("test-only provider failure"));
        }
        if matches!(self.mode, Mode::Panic) {
            panic!("test-only provider panic");
        }
        self.send(sink, "Hi ")?;
        if matches!(self.mode, Mode::Flood) {
            let chunk = "x".repeat(64 * 1024);
            loop {
                if cancel.is_cancelled() {
                    self.cancelled.fetch_add(1, Ordering::SeqCst);
                    return Err(ProviderError::cancelled());
                }
                self.send(sink, &chunk)?;
            }
        }
        if matches!(self.mode, Mode::Gate | Mode::Endless) {
            while !self.release.load(Ordering::SeqCst) {
                if cancel.is_cancelled() {
                    self.cancelled.fetch_add(1, Ordering::SeqCst);
                    return Err(ProviderError::cancelled());
                }
                if matches!(self.mode, Mode::Endless) {
                    self.send(sink, "tick")?;
                }
                thread::sleep(Duration::from_millis(2));
            }
        }
        self.send(sink, "€")?;
        self.send(sink, "🦀")?;
        Ok(GenerationSummary {
            finish_reason: FinishReason::Stop,
            prompt_tokens: 2,
            completion_tokens: 3,
        })
    }
    fn apply_template(
        &self,
        messages: &[llama_server::provider::ChatMessage],
        add_assistant: bool,
    ) -> Result<String, ProviderError> {
        let messages: Vec<_> = messages
            .iter()
            .map(|message| llama_behavior::chat::Message::new(&message.role, &message.content))
            .collect();
        llama_behavior::chat::render("chatml", &messages, add_assistant)
            .map_err(|error| ProviderError::invalid(error.to_string()))
    }

    fn tokenize(&self, text: &str, _: bool) -> Result<Vec<i32>, ProviderError> {
        Ok(text.bytes().map(i32::from).collect())
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String, ProviderError> {
        let bytes = tokens
            .iter()
            .map(|n| u8::try_from(*n).map_err(|_| ProviderError::invalid("test byte out of range")))
            .collect::<Result<Vec<_>, _>>()?;
        String::from_utf8(bytes).map_err(|_| ProviderError::invalid("test bytes are not UTF-8"))
    }
}

fn config() -> ServerConfig {
    ServerConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        request_timeout: Duration::from_secs(2),
        write_timeout: Duration::from_millis(500),
        generation_timeout: Duration::from_secs(2),
        ..ServerConfig::default()
    }
}
fn spawn(provider: Arc<dyn InferenceProvider>) -> RunningServer {
    Server::new(config(), provider).unwrap().spawn().unwrap()
}
fn raw_request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: &str,
    extra: &str,
) -> TcpStream {
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}\r\n{body}",
        body.len()
    );
    socket.write_all(request.as_bytes()).unwrap();
    socket
}
struct Response {
    status: u16,
    head: String,
    body: String,
}
fn response(address: SocketAddr, method: &str, path: &str, body: &str, extra: &str) -> Response {
    let mut socket = raw_request(address, method, path, body, extra);
    let mut bytes = Vec::new();
    socket.read_to_end(&mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").expect(&text);
    Response {
        status: head.split(' ').nth(1).unwrap().parse().unwrap(),
        head: head.into(),
        body: body.into(),
    }
}
fn read_head(socket: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        socket.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        assert!(bytes.len() < 8192);
    }
    String::from_utf8(bytes).unwrap()
}
fn wait_for(mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while !predicate() {
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "condition timed out"
        );
        thread::sleep(Duration::from_millis(3));
    }
}
fn events(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|event| *event != "[DONE]")
        .map(|event| json::parse(event).unwrap())
        .collect()
}

#[test]
fn unavailable_binary_contract_over_tcp() {
    let server = spawn(Arc::new(UnavailableProvider));
    let health = response(server.local_addr(), "GET", "/health", "", "");
    assert_eq!(health.status, 503);
    assert!(health.body.contains("unavailable"));
    let models = response(server.local_addr(), "GET", "/v1/models", "", "");
    assert_eq!(models.status, 200);
    assert_eq!(
        json::parse(&models.body)
            .unwrap()
            .get("data")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        0
    );
    let generation = response(
        server.local_addr(),
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"Hi"}],"stream":true}"#,
        "",
    );
    assert_eq!(generation.status, 503);
    assert!(generation.body.contains("model_unavailable"));
    assert!(!generation.head.contains("text/event-stream"));
    let page = response(server.local_addr(), "GET", "/", "", "");
    assert_eq!(page.status, 200);
    assert!(page.body.contains("No Rust model runtime"));
    assert!(!page.body.contains("<script"));
    server.shutdown().unwrap();
}

#[test]
fn native_and_openai_completion_shapes_and_usage() {
    let provider = FixtureProvider::new(Mode::Finite);
    let server = spawn(provider);
    for path in ["/completion", "/completions"] {
        let result = response(server.local_addr(), "POST", path, r#"{"prompt":"hi"}"#, "");
        assert_eq!(result.status, 200);
        let value = json::parse(&result.body).unwrap();
        assert_eq!(value.get("content").and_then(Value::as_str), Some("Hi €🦀"));
        assert_eq!(
            value.get("tokens_predicted").and_then(Value::as_u64),
            Some(3)
        );
    }
    let result = response(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","model":"test-only"}"#,
        "",
    );
    assert_eq!(result.status, 200);
    let value = json::parse(&result.body).unwrap();
    assert_eq!(
        value.get("choices").unwrap().as_array().unwrap()[0]
            .get("text")
            .and_then(Value::as_str),
        Some("Hi €🦀")
    );
    assert_eq!(
        value
            .get("usage")
            .unwrap()
            .get("total_tokens")
            .and_then(Value::as_u64),
        Some(5)
    );
    let result = response(
        server.local_addr(),
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        "",
    );
    let value = json::parse(&result.body).unwrap();
    assert_eq!(
        value.get("choices").unwrap().as_array().unwrap()[0]
            .get("message")
            .unwrap()
            .get("role")
            .and_then(Value::as_str),
        Some("assistant")
    );
    server.shutdown().unwrap();
}

#[test]
fn streaming_chat_emits_role_deltas_finish_and_done() {
    let server = spawn(FixtureProvider::new(Mode::Finite));
    let result = response(
        server.local_addr(),
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        "",
    );
    assert_eq!(result.status, 200);
    assert!(result.head.contains("text/event-stream"));
    assert!(result.body.ends_with("data: [DONE]\n\n"));
    let events = events(&result.body);
    assert_eq!(events.len(), 5);
    let choice = |index: usize| &events[index].get("choices").unwrap().as_array().unwrap()[0];
    assert_eq!(
        choice(0)
            .get("delta")
            .unwrap()
            .get("role")
            .and_then(Value::as_str),
        Some("assistant")
    );
    let joined: String = events
        .iter()
        .filter_map(|event| {
            event.get("choices").unwrap().as_array().unwrap()[0]
                .get("delta")
                .unwrap()
                .get("content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(joined, "Hi €🦀");
    assert_eq!(
        choice(4).get("finish_reason").and_then(Value::as_str),
        Some("stop")
    );
    server.shutdown().unwrap();
}

#[test]
fn auth_cors_preflight_and_typed_failures() {
    let mut settings = config();
    settings.api_keys = vec!["secret-test-key".into()];
    settings.allowed_origins = vec!["https://client.example".into()];
    let server = Server::new(settings, FixtureProvider::new(Mode::Finite))
        .unwrap()
        .spawn()
        .unwrap();
    assert_eq!(
        response(server.local_addr(), "GET", "/health", "", "").status,
        200
    );
    let denied = response(server.local_addr(), "GET", "/v1/models", "", "");
    assert_eq!(denied.status, 401);
    assert!(denied.head.contains("WWW-Authenticate: Bearer"));
    let allowed = response(
        server.local_addr(),
        "GET",
        "/v1/models",
        "",
        "Authorization: Bearer secret-test-key\r\nOrigin: https://client.example\r\n",
    );
    assert_eq!(allowed.status, 200);
    assert!(
        allowed
            .head
            .contains("Access-Control-Allow-Origin: https://client.example")
    );
    assert_eq!(
        response(
            server.local_addr(),
            "GET",
            "/v1/models",
            "",
            "Authorization: Bearer secret-test-key\r\nOrigin: https://evil.example\r\n"
        )
        .status,
        403
    );
    let preflight = response(
        server.local_addr(),
        "OPTIONS",
        "/v1/chat/completions",
        "",
        "Origin: https://client.example\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: content-type, authorization\r\n",
    );
    assert_eq!(preflight.status, 204);
    assert!(preflight.body.is_empty());
    assert_eq!(
        response(
            server.local_addr(),
            "GET",
            "/missing",
            "",
            "Authorization: Bearer secret-test-key\r\n"
        )
        .status,
        404
    );
    server.shutdown().unwrap();
}

#[test]
fn invalid_json_unsupported_features_and_tokens_are_explicit() {
    let server = spawn(FixtureProvider::new(Mode::Finite));
    for body in [
        r#"{"prompt":"a","prompt":"b"}"#,
        r#"{"prompt":"a","stream":"yes"}"#,
        r#"{"prompt":"a","max_tokens":0}"#,
        r#"{"prompt":"a","seed":-1}"#,
    ] {
        assert_eq!(
            response(server.local_addr(), "POST", "/v1/completions", body, "").status,
            400
        );
    }
    assert_eq!(
        response(
            server.local_addr(),
            "POST",
            "/v1/completions",
            r#"{"prompt":"a","tools":[]}"#,
            ""
        )
        .status,
        501
    );
    assert_eq!(
        response(server.local_addr(), "POST", "/v1/responses", "{}", "").status,
        501
    );
    assert_eq!(
        response(server.local_addr(), "POST", "/lora-adapters", "{}", "").status,
        403
    );
    assert_eq!(
        response(server.local_addr(), "GET", "/tokenize", "", "").status,
        405
    );
    let tokenized = response(
        server.local_addr(),
        "POST",
        "/tokenize",
        r#"{"content":"ab"}"#,
        "",
    );
    assert_eq!(tokenized.status, 200);
    assert_eq!(
        json::parse(&tokenized.body)
            .unwrap()
            .get("tokens")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let detokenized = response(
        server.local_addr(),
        "POST",
        "/detokenize",
        r#"{"tokens":[97,98]}"#,
        "",
    );
    assert_eq!(detokenized.status, 200);
    assert_eq!(
        json::parse(&detokenized.body)
            .unwrap()
            .get("content")
            .and_then(Value::as_str),
        Some("ab")
    );
    assert_eq!(
        response(
            server.local_addr(),
            "POST",
            "/detokenize",
            r#"{"tokens":[2147483648]}"#,
            ""
        )
        .status,
        400
    );
    server.shutdown().unwrap();
}

#[test]
fn completed_replay_returns_exact_byte_suffix_and_lookup_metadata() {
    let server = spawn(FixtureProvider::new(Mode::Finite));
    let original = response(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "X-Conversation-Id: conversation::one\r\n",
    );
    assert_eq!(original.status, 200);
    let offset = original.body.find('€').unwrap() + 1; // deliberately inside UTF-8
    let mut socket = raw_request(
        server.local_addr(),
        "GET",
        &format!("/v1/stream?conv_id=conversation%3A%3Aone&from={offset}"),
        "",
        "",
    );
    assert!(read_head(&mut socket).starts_with("HTTP/1.1 200"));
    let mut bytes = Vec::new();
    socket.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, original.body.as_bytes()[offset..]);
    let lookup = response(
        server.local_addr(),
        "POST",
        "/v1/streams/lookup",
        r#"{"conversation_ids":["conversation","conversation::one",null,4,""]}"#,
        "",
    );
    let value = json::parse(&lookup.body).unwrap();
    let entries = value.as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].get("is_done").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        entries[0].get("total_bytes").and_then(Value::as_u64),
        Some(original.body.len() as u64)
    );
    assert_eq!(
        response(
            server.local_addr(),
            "DELETE",
            "/v1/stream?conv_id=conversation::one",
            "",
            ""
        )
        .status,
        204
    );
    assert_eq!(
        response(
            server.local_addr(),
            "GET",
            "/v1/stream?conv_id=conversation::one",
            "",
            ""
        )
        .status,
        404
    );
    server.shutdown().unwrap();
}

#[test]
fn replay_overflow_is_explicit_and_delete_is_idempotent() {
    let mut settings = config();
    settings.replay_bytes = 31;
    let server = Server::new(settings, FixtureProvider::new(Mode::Finite))
        .unwrap()
        .spawn()
        .unwrap();
    let _ = response(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "X-Conversation-Id: tiny\r\n",
    );
    let lost = response(
        server.local_addr(),
        "GET",
        "/v1/stream?conv_id=tiny&from=0",
        "",
        "",
    );
    assert_eq!(lost.status, 400);
    assert!(lost.body.contains("offset lost"));
    assert_eq!(
        response(
            server.local_addr(),
            "GET",
            "/v1/stream?conv_id=tiny&from=-1",
            "",
            ""
        )
        .status,
        400
    );
    assert_eq!(
        response(
            server.local_addr(),
            "GET",
            "/v1/stream?conv_id=tiny&from=1&from=2",
            "",
            ""
        )
        .status,
        400
    );
    for _ in 0..2 {
        assert_eq!(
            response(
                server.local_addr(),
                "DELETE",
                "/v1/stream?conv_id=tiny",
                "",
                ""
            )
            .status,
            204
        );
    }
    server.shutdown().unwrap();
}

#[test]
fn disconnect_cancels_ordinary_stream_and_releases_admission() {
    let provider = FixtureProvider::new(Mode::Endless);
    let server = spawn(provider.clone());
    let mut socket = raw_request(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "",
    );
    assert!(read_head(&mut socket).starts_with("HTTP/1.1 200"));
    drop(socket);
    wait_for(|| provider.cancelled.load(Ordering::SeqCst) > 0);
    server.shutdown().unwrap();
}

#[test]
fn replay_survives_disconnect_then_resumes_and_explicit_delete_cancels() {
    let provider = FixtureProvider::new(Mode::Gate);
    let server = spawn(provider.clone());
    let mut socket = raw_request(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "X-Conversation-Id: detached\r\n",
    );
    read_head(&mut socket);
    drop(socket);
    wait_for(|| provider.started.load(Ordering::SeqCst) == 1);
    let lookup = response(
        server.local_addr(),
        "POST",
        "/v1/streams/lookup",
        r#"{"conversation_ids":["detached"]}"#,
        "",
    );
    assert_eq!(
        json::parse(&lookup.body).unwrap().as_array().unwrap()[0]
            .get("is_done")
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(provider.cancelled.load(Ordering::SeqCst), 0);
    provider.release.store(true, Ordering::SeqCst);
    let replay = response(
        server.local_addr(),
        "GET",
        "/v1/stream?conv_id=detached&from=0",
        "",
        "",
    );
    assert_eq!(replay.status, 200);
    assert!(replay.body.contains('€'));
    assert!(replay.body.ends_with("data: [DONE]\n\n"));
    provider.release.store(false, Ordering::SeqCst);
    let mut socket = raw_request(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "X-Conversation-Id: delete-me\r\n",
    );
    read_head(&mut socket);
    drop(socket);
    assert_eq!(
        response(
            server.local_addr(),
            "DELETE",
            "/v1/stream?conv_id=delete-me",
            "",
            ""
        )
        .status,
        204
    );
    wait_for(|| provider.cancelled.load(Ordering::SeqCst) > 0);
    server.shutdown().unwrap();
}

#[test]
fn simultaneous_generation_admission_is_bounded() {
    let provider = FixtureProvider::new(Mode::Gate);
    let mut settings = config();
    settings.max_generations = 1;
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    let mut socket = raw_request(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "",
    );
    read_head(&mut socket);
    wait_for(|| provider.started.load(Ordering::SeqCst) == 1);
    let busy = response(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"other"}"#,
        "",
    );
    assert_eq!(busy.status, 429);
    assert!(busy.head.contains("Retry-After: 1"));
    provider.release.store(true, Ordering::SeqCst);
    let mut rest = Vec::new();
    socket.read_to_end(&mut rest).unwrap();
    assert_eq!(provider.started.load(Ordering::SeqCst), 1);
    server.shutdown().unwrap();
}

#[test]
fn provider_error_and_panic_are_typed_without_a_second_status_line() {
    for mode in [Mode::Fail, Mode::Panic] {
        let server = spawn(FixtureProvider::new(mode));
        let sync = response(
            server.local_addr(),
            "POST",
            "/v1/completions",
            r#"{"prompt":"hi"}"#,
            "",
        );
        assert_eq!(sync.status, 500);
        let stream = response(
            server.local_addr(),
            "POST",
            "/v1/completions",
            r#"{"prompt":"hi","stream":true}"#,
            "",
        );
        assert_eq!(stream.status, 200);
        assert!(stream.body.contains("server_error"));
        assert!(!stream.body.contains("HTTP/1.1"));
        assert!(stream.body.ends_with("data: [DONE]\n\n"));
        assert_eq!(
            response(server.local_addr(), "GET", "/health", "", "").status,
            200
        );
        server.shutdown().unwrap();
    }
}

#[test]
fn slow_drip_cannot_extend_absolute_request_deadline() {
    let mut settings = config();
    settings.request_timeout = Duration::from_millis(100);
    let server = Server::new(settings, Arc::new(UnavailableProvider))
        .unwrap()
        .spawn()
        .unwrap();
    let mut socket = TcpStream::connect(server.local_addr()).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut writer = socket.try_clone().unwrap();
    let started = Instant::now();
    let worker = thread::spawn(move || {
        for byte in b"GET /health HTTP/1.1\r\nHost: local\r\n\r\n" {
            if writer.write_all(&[*byte]).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
    });
    let head = read_head(&mut socket);
    assert!(head.starts_with("HTTP/1.1 408"));
    assert!(started.elapsed() < Duration::from_millis(700));
    worker.join().unwrap();
    server.shutdown().unwrap();
}

#[test]
fn shutdown_interrupts_incomplete_request_without_waiting_full_timeout() {
    let mut settings = config();
    settings.request_timeout = Duration::from_secs(60);
    let server = Server::new(settings, Arc::new(UnavailableProvider))
        .unwrap()
        .spawn()
        .unwrap();
    let mut socket = TcpStream::connect(server.local_addr()).unwrap();
    socket.write_all(b"GET /").unwrap();
    thread::sleep(Duration::from_millis(20));
    let started = Instant::now();
    server.shutdown().unwrap();
    assert!(started.elapsed() < Duration::from_millis(700));
}

#[test]
fn disconnected_replay_generation_still_has_an_absolute_deadline() {
    let provider = FixtureProvider::new(Mode::Gate);
    let mut settings = config();
    settings.generation_timeout = Duration::from_millis(100);
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    let mut socket = raw_request(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "X-Conversation-Id: deadline\r\n",
    );
    read_head(&mut socket);
    drop(socket);
    wait_for(|| provider.cancelled.load(Ordering::SeqCst) > 0);
    server.shutdown().unwrap();
}

#[test]
fn malformed_framing_is_rejected_on_the_wire() {
    let server = spawn(Arc::new(UnavailableProvider));
    let mut socket = TcpStream::connect(server.local_addr()).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    socket.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}").unwrap();
    assert!(read_head(&mut socket).starts_with("HTTP/1.1 400"));
    server.shutdown().unwrap();
}

#[test]
fn slow_network_reader_applies_backpressure_with_a_bounded_wait() {
    let provider = FixtureProvider::new(Mode::Flood);
    let mut settings = config();
    settings.queue_bytes = 1024;
    settings.write_timeout = Duration::from_millis(80);
    settings.generation_timeout = Duration::from_secs(30);
    settings.max_response_bytes = 64 * 1024 * 1024;
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    let mut socket = raw_request(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "",
    );
    read_head(&mut socket);
    // Keep the peer open and intentionally stop draining the response body.
    wait_for(|| provider.cancelled.load(Ordering::SeqCst) > 0);
    drop(socket);
    server.shutdown().unwrap();
}

#[test]
fn connection_admission_limit_rejects_excess_socket() {
    let mut settings = config();
    settings.max_connections = 1;
    let server = Server::new(settings, Arc::new(UnavailableProvider))
        .unwrap()
        .spawn()
        .unwrap();
    let mut first = TcpStream::connect(server.local_addr()).unwrap();
    first.write_all(b"GET /").unwrap();
    thread::sleep(Duration::from_millis(25));
    // Both headers and an in-flight body must not turn a complete 503 into a
    // Windows TCP reset when the server rejects admission before parsing.
    for unread_body in ["", "body deliberately unread by the request parser"] {
        let mut excess = raw_request(server.local_addr(), "GET", "/health", unread_body, "");
        let head = read_head(&mut excess);
        assert!(head.starts_with("HTTP/1.1 503"));
        let length: usize = head
            .lines()
            .find_map(|line| line.strip_prefix("Content-Length: "))
            .unwrap()
            .parse()
            .unwrap();
        let mut body = vec![0; length];
        excess.read_exact(&mut body).unwrap();
        assert!(
            String::from_utf8(body)
                .unwrap()
                .contains("Connection limit reached")
        );
    }
    drop(first);
    server.shutdown().unwrap();
}

struct PreflightRejectProvider;
impl InferenceProvider for PreflightRejectProvider {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::Ready
    }
    fn models(&self) -> Vec<ModelInfo> {
        Vec::new()
    }
    fn validate_generation(&self, _: &GenerationRequest) -> Result<(), ProviderError> {
        Err(ProviderError::unsupported(
            "test-only unimplemented runtime option",
        ))
    }
    fn generate(
        &self,
        _: &GenerationRequest,
        _: &mut dyn TokenSink,
        _: &CancellationToken,
    ) -> Result<GenerationSummary, ProviderError> {
        panic!("preflight-rejected request must never reach decoding");
    }
}

#[test]
fn runtime_preflight_rejects_before_stream_or_replay_creation() {
    let server = spawn(Arc::new(PreflightRejectProvider));
    let result = response(
        server.local_addr(),
        "POST",
        "/v1/completions",
        r#"{"prompt":"hi","stream":true}"#,
        "X-Conversation-Id: preflight-rejected\r\n",
    );
    assert_eq!(result.status, 501);
    assert!(!result.head.contains("text/event-stream"));
    assert!(result.body.contains("unimplemented runtime option"));
    let lookup = response(
        server.local_addr(),
        "POST",
        "/v1/streams/lookup",
        r#"{"conversation_ids":["preflight-rejected"]}"#,
        "",
    );
    assert_eq!(lookup.status, 200);
    assert_eq!(json::parse(&lookup.body).unwrap(), Value::Array(Vec::new()));
    server.shutdown().unwrap();
}

#[test]
fn template_endpoint_matches_independent_chatml_transcripts() {
    let server = spawn(FixtureProvider::new(Mode::Finite));
    let mut checked = 0;
    for line in include_str!("../../../fixtures/t10/chat-reference.jsonl").lines() {
        let fixture = json::parse(line).unwrap();
        if fixture.get("template").and_then(Value::as_str) != Some("chatml")
            || !matches!(fixture.get("case").and_then(Value::as_u64), Some(1 | 3 | 4))
        {
            continue;
        }
        let body = Value::object([
            ("messages", fixture.get("messages").unwrap().clone()),
            (
                "add_generation_prompt",
                fixture.get("add_assistant").unwrap().clone(),
            ),
        ]);
        let result = response(
            server.local_addr(),
            "POST",
            "/apply-template",
            &json::stringify(&body).unwrap(),
            "",
        );
        assert_eq!(result.status, 200);
        let actual = json::parse(&result.body).unwrap();
        assert_eq!(actual.get("prompt"), fixture.get("expected"));
        checked += 1;
    }
    assert_eq!(checked, 6);
    for body in [
        r#"{"messages":[{"role":"user","content":"Hi"}],"tools":[]}"#,
        r#"{"messages":[{"role":"user","content":"Hi","name":"Alice"}]}"#,
        r#"{"messages":[{"role":"user","content":"Hi"}],"chat_template":"custom"}"#,
    ] {
        let result = response(server.local_addr(), "POST", "/apply-template", body, "");
        assert_eq!(result.status, 501, "{body}");
    }
    server.shutdown().unwrap();
}

#[test]
fn tokenizer_never_ignores_unimplemented_modes() {
    let server = spawn(FixtureProvider::new(Mode::Finite));
    for body in [
        r#"{"content":"Hi","parse_special":false}"#,
        r#"{"content":"Hi","with_pieces":true}"#,
        r#"{"content":"Hi","unrecognized":true}"#,
    ] {
        let result = response(server.local_addr(), "POST", "/tokenize", body, "");
        assert_eq!(result.status, 501, "{body}");
    }
    let result = response(
        server.local_addr(),
        "POST",
        "/tokenize",
        r#"{"content":"Hi","parse_special":true,"with_pieces":false}"#,
        "",
    );
    assert_eq!(result.status, 200);
    server.shutdown().unwrap();
}
