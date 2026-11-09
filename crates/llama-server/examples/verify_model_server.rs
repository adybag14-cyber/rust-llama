//! Explicit real-model HTTP acceptance, independent of the normal unit suite.
//! Run with the hash-pinned external stories260K.gguf used by the native oracle:
//! cargo run -p llama-server --example verify_model_server -- PATH/stories260K.gguf
use llama_behavior::json::{self, Value};
use llama_server::{
    CancellationToken, GenerationRequest, GenerationSummary, InferenceProvider, Prompt,
    ProviderError, ProviderStatus, RuntimeProvider, Server, ServerConfig, TokenSink,
};
use std::{
    error::Error,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
static REQUESTS: AtomicUsize = AtomicUsize::new(0);

struct Response {
    status: u16,
    head: String,
    body: Vec<u8>,
}
impl Response {
    fn json(&self) -> Value {
        json::parse(std::str::from_utf8(&self.body).unwrap()).unwrap()
    }
}

fn http(
    address: SocketAddr,
    method: &str,
    route: &str,
    body: &Value,
    extra: &str,
) -> Result<Response> {
    let body = if method == "GET" {
        String::new()
    } else {
        json::stringify(body)?
    };
    let mut socket = TcpStream::connect(address)?;
    socket.set_read_timeout(Some(Duration::from_secs(10)))?;
    socket.set_write_timeout(Some(Duration::from_secs(10)))?;
    socket.write_all(
        format!(
            "{method} {route} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )?;
    let mut raw = Vec::new();
    socket.take(16 * 1024 * 1024).read_to_end(&mut raw)?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("missing HTTP delimiter")?;
    let head = std::str::from_utf8(&raw[..split])?.to_owned();
    let status = head
        .split(' ')
        .nth(1)
        .ok_or("missing HTTP status")?
        .parse()?;
    REQUESTS.fetch_add(1, Ordering::Relaxed);
    Ok(Response {
        status,
        head,
        body: raw[split + 4..].to_vec(),
    })
}

fn request(prompt: &str, maximum: u64, stream: bool) -> Value {
    Value::object([
        ("prompt", Value::from(prompt)),
        ("max_tokens", maximum.into()),
        ("temperature", 0u64.into()),
        ("seed", 0u64.into()),
        ("stream", stream.into()),
    ])
}

fn expected_bytes(text: &str) -> Vec<u8> {
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn completion_text(value: &Value) -> &str {
    value.get("choices").unwrap().as_array().unwrap()[0]
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
}

fn stream_text(response: &Response) -> (String, String) {
    assert_eq!(response.status, 200, "{:?}", response.body);
    assert!(response.head.contains("text/event-stream"));
    let mut text = String::new();
    let mut finish = None;
    let mut done = 0;
    for event in std::str::from_utf8(&response.body).unwrap().split("\n\n") {
        let Some(data) = event.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            done += 1;
            continue;
        }
        let value = json::parse(data).unwrap();
        assert!(value.get("error").is_none(), "{data}");
        let choice = &value.get("choices").unwrap().as_array().unwrap()[0];
        text.push_str(choice.get("text").unwrap().as_str().unwrap());
        if let Some(reason) = choice.get("finish_reason").unwrap().as_str() {
            assert!(finish.is_none());
            finish = Some(reason.to_owned());
        }
    }
    assert_eq!(done, 1);
    (text, finish.unwrap())
}

struct RejectSink;
impl TokenSink for RejectSink {
    fn send(&mut self, _: &str) -> std::result::Result<(), ProviderError> {
        Err(ProviderError::new(
            418,
            "acceptance_sink_error",
            "stop writing",
        ))
    }
}
struct CancelSink {
    cancel: CancellationToken,
    calls: usize,
}
impl TokenSink for CancelSink {
    fn send(&mut self, _: &str) -> std::result::Result<(), ProviderError> {
        self.calls += 1;
        self.cancel.cancel();
        Ok(())
    }
}
struct Collect(String);
impl TokenSink for Collect {
    fn send(&mut self, text: &str) -> std::result::Result<(), ProviderError> {
        self.0.push_str(text);
        Ok(())
    }
}

fn main() -> Result<()> {
    let model_path = std::env::args_os()
        .nth(1)
        .ok_or("supply the external stories260K.gguf path")?;
    let provider = Arc::new(RuntimeProvider::load(&model_path)?);
    assert_eq!(provider.status(), ProviderStatus::Ready);
    let original = if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        include_str!("../../../fixtures/inference/stories260k-gnu-reference.json")
    } else {
        include_str!("../../../fixtures/inference/stories260k-reference.json")
    };
    let fixture = json::parse(original)?;
    let cases = fixture.get("cases").unwrap().as_array().unwrap();
    let server = Server::new(
        ServerConfig {
            bind: "127.0.0.1:0".parse()?,
            generation_timeout: Duration::from_secs(10),
            ..ServerConfig::default()
        },
        provider.clone(),
    )?
    .spawn()?;
    let address = server.local_addr();
    let health = http(address, "GET", "/health", &Value::Null, "")?;
    assert_eq!(health.status, 200);
    let models = http(address, "GET", "/v1/models", &Value::Null, "")?;
    assert_eq!(models.status, 200);
    assert_eq!(
        models.json().get("data").unwrap().as_array().unwrap()[0]
            .get("id")
            .unwrap()
            .as_str(),
        Some(provider.model_id())
    );

    for (index, case) in cases.iter().enumerate() {
        let prompt = case.get("prompt").unwrap().as_str().unwrap();
        let expected = expected_bytes(case.get("generated_hex").unwrap().as_str().unwrap());
        let result = http(
            address,
            "POST",
            "/v1/completions",
            &request(prompt, 16, false),
            "",
        )?;
        assert_eq!(result.status, 200, "{}", std::str::from_utf8(&result.body)?);
        let value = result.json();
        assert_eq!(
            completion_text(&value).as_bytes(),
            expected,
            "case {index} nonstream"
        );
        let usage = value.get("usage").unwrap();
        assert_eq!(
            usage.get("prompt_tokens").and_then(Value::as_u64),
            Some(case.get("tokens").unwrap().as_array().unwrap().len() as u64)
        );
        assert_eq!(
            usage.get("completion_tokens").and_then(Value::as_u64),
            Some(16)
        );
        let replay_header = if index == 0 {
            "X-Conversation-Id: real-model-proof\r\n"
        } else {
            ""
        };
        let stream = http(
            address,
            "POST",
            "/v1/completions",
            &request(prompt, 16, true),
            replay_header,
        )?;
        let (text, finish) = stream_text(&stream);
        assert_eq!(text.as_bytes(), expected, "case {index} stream");
        assert_eq!(finish, "length");
        if index == 0 {
            let offset = 137;
            assert!(stream.body.len() > offset);
            let replay = http(
                address,
                "GET",
                &format!("/v1/stream?conv_id=real-model-proof&from={offset}"),
                &Value::Null,
                "",
            )?;
            assert_eq!(replay.status, 200);
            assert_eq!(
                replay.body,
                stream.body[offset..],
                "actual raw replay suffix"
            );
        }
        let tokenized = http(
            address,
            "POST",
            "/tokenize",
            &Value::object([("content", prompt.into()), ("add_special", true.into())]),
            "",
        )?;
        assert_eq!(tokenized.status, 200);
        assert_eq!(tokenized.json().get("tokens"), case.get("tokens"));
        let decoded = http(
            address,
            "POST",
            "/detokenize",
            &Value::object([("tokens", case.get("tokens").unwrap().clone())]),
            "",
        )?;
        assert_eq!(decoded.status, 200);
        assert_eq!(
            decoded
                .json()
                .get("content")
                .unwrap()
                .as_str()
                .unwrap()
                .as_bytes(),
            expected_bytes(case.get("detokenized_hex").unwrap().as_str().unwrap())
        );
    }

    let first = &cases[0];
    let prompt = first.get("prompt").unwrap().as_str().unwrap();
    let complete = expected_bytes(first.get("generated_hex").unwrap().as_str().unwrap());
    let stop = "little girl";
    let stop_index = complete
        .windows(stop.len())
        .position(|x| x == stop.as_bytes())
        .unwrap();
    let mut accumulated = Vec::new();
    let mut stop_tokens = 0;
    for piece in first.get("greedy_pieces_hex").unwrap().as_array().unwrap() {
        accumulated.extend(expected_bytes(piece.as_str().unwrap()));
        stop_tokens += 1;
        if accumulated
            .windows(stop.len())
            .any(|x| x == stop.as_bytes())
        {
            break;
        }
    }
    assert_eq!(stop_tokens, 8);
    let mut stop_request = request(prompt, 16, false);
    if let Value::Object(fields) = &mut stop_request {
        fields.insert("stop".into(), stop.into());
    }
    let native = http(address, "POST", "/completion", &stop_request, "")?;
    assert_eq!(native.status, 200);
    let native = native.json();
    assert_eq!(
        native.get("content").unwrap().as_str().unwrap().as_bytes(),
        &complete[..stop_index]
    );
    assert_eq!(
        native.get("tokens_predicted").and_then(Value::as_u64),
        Some(stop_tokens)
    );
    assert_eq!(
        native.get("stopped_eos").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        native.get("stopped_word").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        native.get("stopped_limit").and_then(Value::as_bool),
        Some(false)
    );
    if let Value::Object(fields) = &mut stop_request {
        fields.insert("stream".into(), true.into());
    }
    let stopped_stream = http(address, "POST", "/v1/completions", &stop_request, "")?;
    let (text, finish) = stream_text(&stopped_stream);
    assert_eq!(text.as_bytes(), &complete[..stop_index]);
    assert_eq!(finish, "stop");

    // A base model has no chat-template metadata. A successful-looking chat
    // response would conceal missing semantics, so rejection is the expected path.
    for route in ["/apply-template", "/v1/chat/completions"] {
        let response = http(
            address,
            "POST",
            route,
            &json::parse(r#"{"messages":[{"role":"user","content":"Hi"}]}"#)?,
            "",
        )?;
        assert_eq!(response.status, 501);
    }
    let mut unsupported = request(prompt, 16, true);
    if let Value::Object(fields) = &mut unsupported {
        fields.insert("top_p".into(), Value::Number(0.9));
    }
    let rejected = http(address, "POST", "/v1/completions", &unsupported, "")?;
    assert_eq!(rejected.status, 501);
    assert!(!rejected.head.contains("text/event-stream"));
    let oversized = http(
        address,
        "POST",
        "/v1/completions",
        &request(&"a ".repeat(2000), 16, true),
        "",
    )?;
    assert_eq!(oversized.status, 400);
    assert!(!oversized.head.contains("text/event-stream"));

    let mut sampling = request(prompt, 12, false);
    if let Value::Object(fields) = &mut sampling {
        fields.insert("temperature".into(), Value::Number(0.8));
        fields.insert("seed".into(), 777u64.into());
    }
    let a = http(address, "POST", "/v1/completions", &sampling, "")?;
    let b = http(address, "POST", "/v1/completions", &sampling, "")?;
    assert_eq!((a.status, b.status), (200, 200));
    assert_eq!(
        completion_text(&a.json()),
        completion_text(&b.json()),
        "fresh request RNG reset"
    );

    let typed = GenerationRequest {
        model: None,
        prompt: Prompt::Text(prompt.into()),
        max_tokens: 16,
        temperature: 0.0,
        seed: Some(0),
        stop: Vec::new(),
        parameters: Value::object(std::iter::empty::<(&str, Value)>()),
    };
    let error = provider
        .generate(&typed, &mut RejectSink, &CancellationToken::new())
        .unwrap_err();
    assert_eq!(error.status, 418);
    assert_eq!(error.kind, "acceptance_sink_error");
    let cancel = CancellationToken::new();
    let mut cancelling = CancelSink {
        cancel: cancel.clone(),
        calls: 0,
    };
    let error = provider
        .generate(&typed, &mut cancelling, &cancel)
        .unwrap_err();
    assert_eq!(error.status, 499);
    assert_eq!(cancelling.calls, 1);
    let mut collection = Collect(String::new());
    let recovered: GenerationSummary =
        provider.generate(&typed, &mut collection, &CancellationToken::new())?;
    assert_eq!(collection.0.as_bytes(), complete);
    assert_eq!(recovered.completion_tokens, 16);

    // Concurrent requests must not share a sequence or mutable sampler state.
    let request_a = request(prompt, 16, false);
    let request_b = request_a.clone();
    let one = std::thread::spawn(move || {
        let r = http(address, "POST", "/v1/completions", &request_a, "").unwrap();
        assert_eq!(r.status, 200);
        completion_text(&r.json()).to_owned()
    });
    let two = std::thread::spawn(move || {
        let r = http(address, "POST", "/v1/completions", &request_b, "").unwrap();
        assert_eq!(r.status, 200);
        completion_text(&r.json()).to_owned()
    });
    assert_eq!(one.join().unwrap().as_bytes(), complete);
    assert_eq!(two.join().unwrap().as_bytes(), complete);
    server.shutdown()?;

    println!(
        "{}",
        json::stringify(&Value::object([
            ("status", "passed".into()),
            ("reference_cases", (cases.len() as u64).into()),
            (
                "http_requests",
                (REQUESTS.load(Ordering::Relaxed) as u64).into()
            ),
            ("stop_completion_tokens", stop_tokens.into()),
            ("model_id", provider.model_id().into()),
            ("exact_native_generation", true.into()),
            ("exact_http_generation_and_sse", true.into()),
            ("exact_replay_suffix", true.into()),
            ("actual_model_cancellation_and_recovery", true.into()),
            ("concurrent_sequence_isolation", true.into()),
        ]))?
    );
    Ok(())
}
