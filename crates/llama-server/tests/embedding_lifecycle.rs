//! Transport failure and lifecycle tests for embedding work. The controlled
//! provider is test code; model/numeric parity is covered in embedding_http.rs.
use llama_behavior::json::{self, Value};
use llama_inference::Model;
use llama_server::*;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum Mode {
    Ready,
    Hold,
    Fail,
    Panic,
    WrongShape,
    Nonfinite,
}
struct Controlled {
    mode: Mode,
    started: AtomicUsize,
    cancelled: AtomicUsize,
    release: AtomicBool,
}
impl Controlled {
    fn new(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            started: AtomicUsize::new(0),
            cancelled: AtomicUsize::new(0),
            release: AtomicBool::new(false),
        })
    }
}
impl InferenceProvider for Controlled {
    fn status(&self) -> ProviderStatus {
        ProviderStatus::Ready
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "lifecycle-fixture".into(),
            owned_by: "test-only".into(),
        }]
    }
    fn embedding_info(&self) -> Result<EmbeddingInfo, ProviderError> {
        Ok(EmbeddingInfo {
            dimensions: 4,
            vocabulary_size: 256,
            context_length: 8,
            pooled: true,
            normalization: 2,
        })
    }
    fn tokenize(&self, text: &str, _: bool) -> Result<Vec<i32>, ProviderError> {
        Ok(text.bytes().map(i32::from).collect())
    }
    fn embed(
        &self,
        _: &[i32],
        _: i32,
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        self.started.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Fail => return Err(ProviderError::internal("test provider failure")),
            Mode::Panic => panic!("test embedding provider panic"),
            Mode::WrongShape => return Ok(vec![vec![1.0, 2.0]]),
            Mode::Nonfinite => return Ok(vec![vec![0.0, 1.0, f32::NAN, 2.0]]),
            Mode::Hold => {
                let started = Instant::now();
                while !self.release.load(Ordering::SeqCst) {
                    if cancel.is_cancelled() {
                        self.cancelled.fetch_add(1, Ordering::SeqCst);
                        return Err(ProviderError::cancelled());
                    }
                    if started.elapsed() > Duration::from_secs(5) {
                        return Err(ProviderError::internal("bounded test provider expired"));
                    }
                    thread::sleep(Duration::from_millis(2));
                }
            }
            Mode::Ready => {}
        }
        Ok(vec![vec![0.0, 1.0, -1.0, 0.5]])
    }
    fn generate(
        &self,
        _: &GenerationRequest,
        _: &mut dyn TokenSink,
        _: &CancellationToken,
    ) -> Result<GenerationSummary, ProviderError> {
        Err(ProviderError::unsupported(
            "test provider has no generation",
        ))
    }
}

fn config() -> ServerConfig {
    ServerConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        max_generations: 1,
        request_timeout: Duration::from_secs(2),
        generation_timeout: Duration::from_secs(2),
        ..ServerConfig::default()
    }
}
fn raw(address: SocketAddr, method: &str, path: &str, body: &str) -> TcpStream {
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    write!(
        socket,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    socket
}
fn read(mut socket: TcpStream) -> (u16, Value) {
    let mut output = String::new();
    socket.read_to_string(&mut output).unwrap();
    let (headers, body) = output.split_once("\r\n\r\n").unwrap();
    assert!(!headers.contains("text/event-stream"));
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    let length: usize = headers
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length: "))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(length, body.len());
    (status, json::parse(body).unwrap())
}
fn post(address: SocketAddr, path: &str, body: &str) -> (u16, Value) {
    read(raw(address, "POST", path, body))
}
fn wait_for(mut condition: impl FnMut() -> bool) {
    let started = Instant::now();
    while !condition() {
        assert!(started.elapsed() < Duration::from_secs(2));
        thread::sleep(Duration::from_millis(3));
    }
}

#[test]
fn disabled_embeddings_and_unavailable_models_fail_before_decoding() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/inference/tiny-bert-missing-pooling.gguf");
    let provider = RuntimeProvider::from_model(Model::load(path).unwrap());
    let server = Server::new(config(), Arc::new(provider))
        .unwrap()
        .spawn()
        .unwrap();
    for path in ["/embedding", "/embeddings", "/v1/embeddings"] {
        assert_eq!(post(server.local_addr(), path, "invalid json").0, 501);
        assert_eq!(read(raw(server.local_addr(), "GET", path, "")).0, 405);
    }
    server.shutdown().unwrap();
    let server = Server::new(config(), Arc::new(UnavailableProvider))
        .unwrap()
        .spawn()
        .unwrap();
    assert_eq!(
        post(server.local_addr(), "/v1/embeddings", r#"{"input":"ok"}"#).0,
        503
    );
    server.shutdown().unwrap();
}

#[test]
fn embedding_preflight_limits_do_not_admit_model_work() {
    let provider = Controlled::new(Mode::Ready);
    let mut settings = config();
    settings.max_embedding_inputs = 1;
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    for (body, status) in [
        (r#"{"input":["ok","ok"]}"#, 413),
        (r#"{"input":"123456789"}"#, 400),
        (r#"{"input":[]}"#, 400),
        (r#"{"input":[-1]}"#, 400),
        (r#"{"input":[256]}"#, 400),
        (r#"{"input":"ok","encoding_format":"invalid"}"#, 400),
        (r#"{"input":{"content":[{"type":"image_url"}]}}"#, 501),
    ] {
        assert_eq!(
            post(server.local_addr(), "/v1/embeddings", body).0,
            status,
            "{body}"
        );
    }
    assert_eq!(provider.started.load(Ordering::SeqCst), 0);
    assert_eq!(
        post(server.local_addr(), "/v1/embeddings", r#"{"input":"ok"}"#).0,
        200
    );
    assert_eq!(provider.started.load(Ordering::SeqCst), 1);
    server.shutdown().unwrap();
}

#[test]
fn embedding_response_limits_are_transactional_and_release_admission() {
    let provider = Controlled::new(Mode::Ready);
    let mut settings = config();
    settings.max_response_bytes = 64;
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    let (status, body) = post(server.local_addr(), "/v1/embeddings", r#"{"input":"ok"}"#);
    assert_eq!(status, 413);
    assert_eq!(
        body.get("error")
            .unwrap()
            .get("type")
            .and_then(Value::as_str),
        Some("response_too_large")
    );
    assert_eq!(provider.started.load(Ordering::SeqCst), 0);
    let (status, body) = post(server.local_addr(), "/embeddings", r#"{"input":"ok"}"#);
    assert_eq!(status, 200);
    assert_eq!(body.as_array().unwrap().len(), 1);
    assert_eq!(provider.started.load(Ordering::SeqCst), 1);
    server.shutdown().unwrap();

    let provider = Controlled::new(Mode::Ready);
    let mut settings = config();
    settings.max_response_bytes = 40;
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    assert_eq!(
        post(server.local_addr(), "/embeddings", r#"{"input":"ok"}"#).0,
        413
    );
    assert_eq!(provider.started.load(Ordering::SeqCst), 1);
    server.shutdown().unwrap();
}

#[test]
fn embedding_provider_failures_never_send_partial_success_json() {
    for mode in [Mode::Fail, Mode::Panic, Mode::WrongShape, Mode::Nonfinite] {
        let provider = Controlled::new(mode);
        let server = Server::new(config(), provider.clone())
            .unwrap()
            .spawn()
            .unwrap();
        for _ in 0..2 {
            let (status, body) = post(
                server.local_addr(),
                "/v1/embeddings",
                r#"{"input":"ok","encoding_format":"base64"}"#,
            );
            assert_eq!(status, 500);
            assert!(body.get("error").is_some());
            assert!(body.get("data").is_none());
        }
        assert_eq!(provider.started.load(Ordering::SeqCst), 2);
        server.shutdown().unwrap();
    }
}

#[test]
fn embedding_deadline_cancels_work_and_releases_capacity() {
    let provider = Controlled::new(Mode::Hold);
    let mut settings = config();
    settings.generation_timeout = Duration::from_millis(100);
    let server = Server::new(settings, provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    let (status, body) = post(server.local_addr(), "/v1/embeddings", r#"{"input":"ok"}"#);
    assert_eq!(status, 504);
    assert_eq!(
        body.get("error")
            .unwrap()
            .get("type")
            .and_then(Value::as_str),
        Some("embedding_timeout")
    );
    wait_for(|| provider.cancelled.load(Ordering::SeqCst) == 1);
    provider.release.store(true, Ordering::SeqCst);
    // Observe permit cleanup before submitting the next request.
    let started = Instant::now();
    loop {
        let (status, _) = post(server.local_addr(), "/v1/embeddings", r#"{"input":"ok"}"#);
        if status == 200 {
            break;
        }
        assert_eq!(status, 429);
        assert!(started.elapsed() < Duration::from_secs(2));
        thread::sleep(Duration::from_millis(3));
    }
    server.shutdown().unwrap();
}

#[test]
fn embedding_disconnect_and_shutdown_cancel_held_requests() {
    let provider = Controlled::new(Mode::Hold);
    let server = Server::new(config(), provider.clone())
        .unwrap()
        .spawn()
        .unwrap();
    let held = raw(
        server.local_addr(),
        "POST",
        "/v1/embeddings",
        r#"{"input":"ok"}"#,
    );
    wait_for(|| provider.started.load(Ordering::SeqCst) == 1);
    assert_eq!(
        post(server.local_addr(), "/v1/embeddings", r#"{"input":"ok"}"#).0,
        429
    );
    drop(held);
    wait_for(|| provider.cancelled.load(Ordering::SeqCst) == 1);
    let started = Instant::now();
    let mut held = 'admit: loop {
        let socket = raw(
            server.local_addr(),
            "POST",
            "/v1/embeddings",
            r#"{"input":"ok"}"#,
        );
        socket.set_nonblocking(true).unwrap();
        loop {
            if provider.started.load(Ordering::SeqCst) == 2 {
                socket.set_nonblocking(false).unwrap();
                break 'admit socket;
            }
            let mut byte = [0];
            match socket.peek(&mut byte) {
                Ok(n) if n != 0 => {
                    socket.set_nonblocking(false).unwrap();
                    let (status, _) = read(socket);
                    assert_eq!(status, 429);
                    thread::sleep(Duration::from_millis(3));
                    continue 'admit;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Ok(_) => panic!("held request closed unexpectedly"),
                Err(error) => panic!("held request failed: {error}"),
            }
            assert!(started.elapsed() < Duration::from_secs(2));
            thread::sleep(Duration::from_millis(2));
        }
    };
    server.shutdown().unwrap();
    // A server shutdown may close a cancelled response before headers, as the
    // original server does. It must still cancel and join the active work.
    let mut output = String::new();
    held.read_to_string(&mut output).unwrap();
    if !output.is_empty() {
        let (headers, body) = output.split_once("\r\n\r\n").unwrap();
        let status: u16 = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
        assert!(matches!(status, 499 | 503 | 504));
        assert!(json::parse(body).unwrap().get("error").is_some());
    }
    assert_eq!(provider.cancelled.load(Ordering::SeqCst), 2);
}
