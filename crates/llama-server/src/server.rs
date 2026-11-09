use crate::client;
use crate::http::{self, HttpLimits, Request};
use crate::provider::*;
use crate::stream::*;
use llama_behavior::json::{self, Value};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub http_limits: HttpLimits,
    pub api_keys: Vec<String>,
    /// Exact allowed origins, or `*`. Credentials are never enabled.
    pub allowed_origins: Vec<String>,
    pub max_connections: usize,
    pub max_generations: usize,
    pub queue_bytes: usize,
    pub replay_bytes: usize,
    pub replay_sessions: usize,
    pub replay_ttl: Duration,
    pub max_response_bytes: usize,
    pub request_timeout: Duration,
    pub write_timeout: Duration,
    pub generation_timeout: Duration,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".parse().unwrap(),
            http_limits: HttpLimits::default(),
            api_keys: Vec::new(),
            allowed_origins: Vec::new(),
            max_connections: 32,
            max_generations: 4,
            queue_bytes: 64 * 1024,
            replay_bytes: 4 * 1024 * 1024,
            replay_sessions: 32,
            replay_ttl: Duration::from_secs(300),
            max_response_bytes: 8 * 1024 * 1024,
            request_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(5),
            generation_timeout: Duration::from_secs(300),
        }
    }
}
impl ServerConfig {
    pub fn validate(&self) -> Result<(), ProviderError> {
        if self.max_connections == 0
            || self.max_connections > 4096
            || self.max_generations == 0
            || self.max_generations > 256
            || self.queue_bytes == 0
            || self.queue_bytes > 16 * 1024 * 1024
            || self.replay_bytes == 0
            || self.replay_bytes > 64 * 1024 * 1024
            || self.replay_sessions == 0
            || self.replay_sessions > 4096
            || self.max_response_bytes == 0
            || self.max_response_bytes > 64 * 1024 * 1024
            || self.http_limits.max_header_bytes < 16
            || self.http_limits.max_header_bytes > 1024 * 1024
            || self.http_limits.max_body_bytes > 64 * 1024 * 1024
            || self.http_limits.max_headers == 0
            || self.http_limits.max_headers > 4096
            || self.http_limits.max_target_bytes == 0
            || self.request_timeout.is_zero()
            || self.write_timeout.is_zero()
            || self.generation_timeout.is_zero()
        {
            return Err(ProviderError::invalid(
                "Invalid or excessive server resource limit",
            ));
        }
        if self.api_keys.iter().any(|key| {
            key.is_empty() || key.len() > 4096 || !key.bytes().all(|b| b > 32 && b < 127)
        }) || self
            .allowed_origins
            .iter()
            .any(|origin| origin.is_empty() || origin.bytes().any(|b| b <= 32 || b >= 127))
        {
            return Err(ProviderError::invalid("Invalid API key or CORS origin"));
        }
        Ok(())
    }
}

struct Shared {
    config: ServerConfig,
    provider: Arc<dyn InferenceProvider>,
    replay: ReplayStore,
    shutdown: CancellationToken,
    connections: AtomicUsize,
    generations: AtomicUsize,
    requests: AtomicU64,
    errors: AtomicU64,
    next_id: AtomicU64,
    active: Mutex<BTreeMap<u64, ActiveJob>>,
    producers: Mutex<Vec<JoinHandle<()>>>,
}
struct ActiveJob {
    cancel: CancellationToken,
    started: Instant,
}

pub struct Server {
    shared: Arc<Shared>,
}
impl Server {
    pub fn new(
        config: ServerConfig,
        provider: Arc<dyn InferenceProvider>,
    ) -> Result<Self, ProviderError> {
        config.validate()?;
        let replay = ReplayStore::new(
            config.replay_bytes,
            config.replay_sessions,
            config.replay_ttl,
        )
        .map_err(ProviderError::invalid)?;
        Ok(Self {
            shared: Arc::new(Shared {
                config,
                provider,
                replay,
                shutdown: CancellationToken::new(),
                connections: AtomicUsize::new(0),
                generations: AtomicUsize::new(0),
                requests: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                next_id: AtomicU64::new(1),
                active: Mutex::new(BTreeMap::new()),
                producers: Mutex::new(Vec::new()),
            }),
        })
    }
    pub fn spawn(self) -> io::Result<RunningServer> {
        let listener = TcpListener::bind(self.shared.config.bind)?;
        listener.set_nonblocking(true)?;
        let local_addr = listener.local_addr()?;
        let shared = self.shared.clone();
        let worker = thread::Builder::new()
            .name("llama-http-listener".into())
            .spawn(move || serve(listener, shared))?;
        Ok(RunningServer {
            local_addr,
            shared: self.shared,
            worker: Some(worker),
        })
    }
}

pub struct RunningServer {
    local_addr: SocketAddr,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<io::Result<()>>>,
}
impl RunningServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
    pub fn shutdown(mut self) -> io::Result<()> {
        self.stop();
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| io::Error::other("Listener panicked"))?
    }
    fn stop(&self) {
        self.shared.shutdown.cancel();
        self.shared.replay.cancel_all();
        for job in self.shared.active.lock().unwrap().values() {
            job.cancel.cancel();
        }
    }
}
impl Drop for RunningServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn acquire(counter: &AtomicUsize, limit: usize) -> bool {
    counter
        .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < limit).then_some(n + 1)
        })
        .is_ok()
}
struct ConnectionGuard(Arc<Shared>);
impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.connections.fetch_sub(1, Ordering::AcqRel);
    }
}
struct GenerationGuard {
    shared: Arc<Shared>,
    id: u64,
}
impl Drop for GenerationGuard {
    fn drop(&mut self) {
        self.shared.active.lock().unwrap().remove(&self.id);
        self.shared.generations.fetch_sub(1, Ordering::AcqRel);
    }
}
fn reap(handles: &mut Vec<JoinHandle<()>>) {
    let mut i = 0;
    while i < handles.len() {
        if handles[i].is_finished() {
            let _ = handles.swap_remove(i).join();
        } else {
            i += 1;
        }
    }
}

/// Finish a normal overload response before releasing the socket. In particular,
/// Winsock can replace an immediate close with a reset when request bytes remain
/// unread, preventing the peer from observing the already-written HTTP response.
fn reject_connection(socket: &mut TcpStream) -> io::Result<()> {
    socket.set_write_timeout(Some(Duration::from_millis(100)))?;
    write_error(
        socket,
        &ProviderError::new(503, "server_busy", "Connection limit reached"),
        &[],
    )?;
    socket.shutdown(Shutdown::Write)?;
    // This executes on the accept loop: neither a slow sender nor an oversized
    // upload may hold it indefinitely. Drain ordinary requests, then close.
    let deadline = Instant::now() + Duration::from_millis(50);
    let mut remaining = 8192usize;
    let mut buffer = [0u8; 1024];
    while remaining != 0 {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        if left.is_zero() {
            break;
        }
        socket.set_read_timeout(Some(left))?;
        let count = remaining.min(buffer.len());
        match socket.read(&mut buffer[..count]) {
            Ok(0) => break,
            Ok(n) => remaining -= n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    Ok(())
}

fn serve(listener: TcpListener, shared: Arc<Shared>) -> io::Result<()> {
    let mut workers = Vec::new();
    while !shared.shutdown.is_cancelled() {
        reap(&mut workers);
        reap(&mut shared.producers.lock().unwrap());
        for job in shared.active.lock().unwrap().values() {
            if job.started.elapsed() >= shared.config.generation_timeout {
                job.cancel.cancel();
            }
        }
        match listener.accept() {
            Ok((mut socket, _)) => {
                socket.set_read_timeout(Some(shared.config.request_timeout))?;
                socket.set_write_timeout(Some(shared.config.write_timeout))?;
                socket.set_nodelay(true)?;
                if !acquire(&shared.connections, shared.config.max_connections) {
                    let _ = reject_connection(&mut socket);
                    continue;
                }
                let state = shared.clone();
                let guard = ConnectionGuard(shared.clone());
                let worker = thread::Builder::new()
                    .name("llama-http-request".into())
                    .spawn(move || {
                        let _guard = guard;
                        let _ = handle_connection(&mut socket, state);
                    })?;
                workers.push(worker);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5))
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    shared.replay.cancel_all();
    for job in shared.active.lock().unwrap().values() {
        job.cancel.cancel();
    }
    for worker in workers {
        let _ = worker.join();
    }
    // Providers are required to cooperate. A broken external implementation must
    // not hold shutdown forever: after the bounded grace, detach its thread.
    let started = Instant::now();
    loop {
        let mut producers = shared.producers.lock().unwrap();
        reap(&mut producers);
        if producers.is_empty() || started.elapsed() >= shared.config.write_timeout {
            break;
        }
        drop(producers);
        thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}

fn handle_connection(socket: &mut TcpStream, shared: Arc<Shared>) -> io::Result<()> {
    let mut reader = DeadlineReader {
        socket,
        started: Instant::now(),
        timeout: shared.config.request_timeout,
        shutdown: &shared.shutdown,
    };
    let request = match http::read_request(&mut reader, shared.config.http_limits.clone()) {
        Ok(request) => request,
        Err(error) => {
            if shared.shutdown.is_cancelled() {
                return Ok(());
            }
            return write_error(
                socket,
                &ProviderError::new(error.status, "invalid_request_error", error.message),
                &[],
            );
        }
    };
    shared.requests.fetch_add(1, Ordering::Relaxed);
    let headers = match access_headers(&request, &shared.config) {
        Ok(headers) => headers,
        Err(error) => {
            shared.errors.fetch_add(1, Ordering::Relaxed);
            return write_error(socket, &error, &[]);
        }
    };
    if request.method == "OPTIONS" {
        return write_bytes(socket, 204, "application/json", &[], &headers);
    }
    if let Err(error) = authorize(&request, &shared.config) {
        shared.errors.fetch_add(1, Ordering::Relaxed);
        return write_error(socket, &error, &headers);
    }
    if let Err(error) = route(socket, &request, &shared, &headers) {
        shared.errors.fetch_add(1, Ordering::Relaxed);
        if error.status == 499 {
            return Ok(());
        }
        return write_error(socket, &error, &headers);
    }
    Ok(())
}

struct DeadlineReader<'a> {
    socket: &'a mut TcpStream,
    started: Instant,
    timeout: Duration,
    shutdown: &'a CancellationToken,
}
impl Read for DeadlineReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            let remaining = self.timeout.saturating_sub(self.started.elapsed());
            if remaining.is_zero() || self.shutdown.is_cancelled() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Request deadline exceeded or server stopped",
                ));
            }
            self.socket
                .set_read_timeout(Some(remaining.min(Duration::from_millis(50))))?;
            match self.socket.read(buffer) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::TimedOut
                            | io::ErrorKind::WouldBlock
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                result => {
                    if self.started.elapsed() >= self.timeout {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "Request deadline exceeded",
                        ));
                    }
                    return result;
                }
            }
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut difference = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        difference |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    difference == 0
}

fn access_headers(
    request: &Request,
    config: &ServerConfig,
) -> Result<Vec<(String, String)>, ProviderError> {
    let mut headers = Vec::new();
    if let Some(origin) = request.header("origin") {
        let wildcard = config
            .allowed_origins
            .iter()
            .any(|candidate| candidate == "*");
        if !wildcard
            && !config
                .allowed_origins
                .iter()
                .any(|candidate| candidate == origin)
        {
            return Err(ProviderError::new(
                403,
                "origin_forbidden",
                "Origin is not allowed",
            ));
        }
        headers.push((
            "Access-Control-Allow-Origin".into(),
            if wildcard { "*".into() } else { origin.into() },
        ));
        headers.push(("Vary".into(), "Origin".into()));
        headers.push((
            "Access-Control-Expose-Headers".into(),
            "X-Conversation-Id, X-Request-Id".into(),
        ));
    }
    if request.method == "OPTIONS" {
        if let Some(method) = request.header("access-control-request-method")
            && !matches!(method, "GET" | "POST" | "DELETE" | "OPTIONS")
        {
            return Err(ProviderError::new(
                405,
                "method_not_allowed",
                "Preflight method is not supported",
            ));
        }
        if let Some(requested) = request.header("access-control-request-headers")
            && requested.split(',').map(str::trim).any(|key| {
                !["authorization", "content-type", "x-conversation-id"]
                    .contains(&key.to_ascii_lowercase().as_str())
            })
        {
            return Err(ProviderError::new(
                403,
                "headers_forbidden",
                "Preflight header is not allowed",
            ));
        }
        headers.push((
            "Access-Control-Allow-Methods".into(),
            "GET, POST, DELETE, OPTIONS".into(),
        ));
        headers.push((
            "Access-Control-Allow-Headers".into(),
            "Authorization, Content-Type, X-Conversation-Id".into(),
        ));
        headers.push(("Access-Control-Max-Age".into(), "600".into()));
        return Ok(headers);
    }
    Ok(headers)
}

fn authorize(request: &Request, config: &ServerConfig) -> Result<(), ProviderError> {
    let public = request.method == "GET" && matches!(request.path(), "/health" | "/v1/health");
    if !public && !config.api_keys.is_empty() {
        let supplied = request
            .header("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
            .unwrap_or("");
        let mut valid = false;
        for key in &config.api_keys {
            valid |= constant_time_eq(supplied.as_bytes(), key.as_bytes());
        }
        if !valid {
            return Err(ProviderError::new(
                401,
                "authentication_error",
                "A valid Bearer API key is required",
            ));
        }
    }
    Ok(())
}

fn object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::object(entries)
}
fn number(n: u64) -> Value {
    Value::Unsigned(n)
}
fn encoded(value: &Value) -> Result<String, ProviderError> {
    json::stringify(value).map_err(|e| ProviderError::internal(e.to_string()))
}
fn error_value(error: &ProviderError) -> Value {
    object([(
        "error",
        object([
            ("message", error.message.clone().into()),
            ("type", error.kind.into()),
            ("code", Value::from(u32::from(error.status))),
        ]),
    )])
}
fn write_error(
    socket: &mut TcpStream,
    error: &ProviderError,
    headers: &[(String, String)],
) -> io::Result<()> {
    let body = encoded(&error_value(error))
        .unwrap_or_else(|_| "{\"error\":{\"type\":\"server_error\"}}".into());
    let mut headers = headers.to_vec();
    if error.status == 401 {
        headers.push(("WWW-Authenticate".into(), "Bearer".into()));
    }
    if error.status == 429 || error.status == 503 {
        headers.push(("Retry-After".into(), "1".into()));
    }
    write_bytes(
        socket,
        error.status,
        "application/json; charset=utf-8",
        body.as_bytes(),
        &headers,
    )
}
fn write_bytes(
    socket: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    headers: &[(String, String)],
) -> io::Result<()> {
    http::write_head(socket, status, content_type, Some(body.len()), headers)?;
    socket.write_all(body)
}
fn reply(
    socket: &mut TcpStream,
    status: u16,
    value: Value,
    headers: &[(String, String)],
) -> Result<(), ProviderError> {
    let body = encoded(&value)?;
    write_bytes(
        socket,
        status,
        "application/json; charset=utf-8",
        body.as_bytes(),
        headers,
    )
    .map_err(io_error)
}
fn io_error(error: io::Error) -> ProviderError {
    ProviderError::new(499, "connection_closed", error.to_string())
}
fn json_body(request: &Request) -> Result<Value, ProviderError> {
    if !request.header("content-type").is_some_and(|value| {
        value
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/json")
    }) {
        return Err(ProviderError::new(
            415,
            "invalid_content_type",
            "Content-Type must be application/json",
        ));
    }
    let text = std::str::from_utf8(&request.body)
        .map_err(|_| ProviderError::invalid("JSON body must be UTF-8"))?;
    json::parse_with_limits(
        text,
        json::Limits {
            max_bytes: request.body.len(),
            max_depth: 64,
            max_nodes: 100_000,
        },
    )
    .map_err(|e| ProviderError::invalid(e.to_string()))
}
fn ensure_ready(shared: &Shared) -> Result<(), ProviderError> {
    match shared.provider.status() {
        ProviderStatus::Ready => Ok(()),
        ProviderStatus::Unavailable(reason) => Err(ProviderError::unavailable(reason)),
    }
}

fn route(
    socket: &mut TcpStream,
    request: &Request,
    shared: &Arc<Shared>,
    headers: &[(String, String)],
) -> Result<(), ProviderError> {
    let path = request.path();
    match (request.method.as_str(), path) {
        ("GET", "/health" | "/v1/health") => match shared.provider.status() {
            ProviderStatus::Ready => reply(socket, 200, object([("status", "ok".into())]), headers),
            ProviderStatus::Unavailable(message) => reply(
                socket,
                503,
                object([
                    ("status", "unavailable".into()),
                    ("reason", message.into()),
                    ("runtime_connected", false.into()),
                ]),
                headers,
            ),
        },
        ("GET", "/") => {
            let status = match shared.provider.status() {
                ProviderStatus::Ready => "Inference provider is ready".into(),
                ProviderStatus::Unavailable(message) => message,
            };
            let page = client::diagnostics(&status, shared.provider.models().len());
            let mut headers = headers.to_vec();
            headers.push(("Content-Security-Policy".into(), "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'".into()));
            write_bytes(
                socket,
                200,
                "text/html; charset=utf-8",
                page.as_bytes(),
                &headers,
            )
            .map_err(io_error)
        }
        ("GET", "/models" | "/v1/models") => {
            let models = shared
                .provider
                .models()
                .into_iter()
                .map(|model| {
                    object([
                        ("id", model.id.into()),
                        ("object", "model".into()),
                        ("created", number(0)),
                        ("owned_by", model.owned_by.into()),
                    ])
                })
                .collect();
            reply(
                socket,
                200,
                object([("object", "list".into()), ("data", Value::Array(models))]),
                headers,
            )
        }
        ("GET", "/props") => reply(
            socket,
            200,
            object([
                ("implementation", "rust-transport-checkpoint".into()),
                (
                    "runtime_connected",
                    matches!(shared.provider.status(), ProviderStatus::Ready).into(),
                ),
                ("stream_replay", true.into()),
                ("replay_bytes", number(shared.config.replay_bytes as u64)),
                (
                    "replay_ttl_seconds",
                    number(shared.config.replay_ttl.as_secs()),
                ),
                (
                    "max_generations",
                    number(shared.config.max_generations as u64),
                ),
                ("ui", "rust-diagnostics-only".into()),
            ]),
            headers,
        ),
        ("GET", "/metrics") => {
            let metrics = format!(
                "# TYPE llama_http_requests_total counter\nllama_http_requests_total {}\n# TYPE llama_http_errors_total counter\nllama_http_errors_total {}\n# TYPE llama_http_connections gauge\nllama_http_connections {}\n# TYPE llama_generation_active gauge\nllama_generation_active {}\n",
                shared.requests.load(Ordering::Relaxed),
                shared.errors.load(Ordering::Relaxed),
                shared.connections.load(Ordering::Relaxed),
                shared.generations.load(Ordering::Relaxed)
            );
            write_bytes(
                socket,
                200,
                "text/plain; version=0.0.4",
                metrics.as_bytes(),
                headers,
            )
            .map_err(io_error)
        }
        (
            "POST",
            "/completion"
            | "/completions"
            | "/v1/completions"
            | "/chat/completions"
            | "/v1/chat/completions",
        ) => generation(socket, request, shared, headers),
        ("POST", "/apply-template") => {
            let body = json_body(request)?;
            let fields = body
                .as_object()
                .ok_or_else(|| ProviderError::invalid("Request body must be an object"))?;
            for (key, value) in fields {
                match key.as_str() {
                    "messages" | "model" | "add_generation_prompt" => {}
                    "tools" | "tool_choice" if value.is_null() => {}
                    _ => {
                        return Err(ProviderError::unsupported(format!(
                            "Template option {key} is not implemented"
                        )));
                    }
                }
            }
            let messages = parse_chat_messages(&body)?;
            let add_assistant = optional_bool(&body, "add_generation_prompt", true)?;
            ensure_ready(shared)?;
            if let Some(value) = body.get("model") {
                let id = value
                    .as_str()
                    .ok_or_else(|| ProviderError::invalid("model must be a string"))?;
                if !shared.provider.models().iter().any(|model| model.id == id) {
                    return Err(ProviderError::new(
                        404,
                        "model_not_found",
                        "Requested model is not available",
                    ));
                }
            }
            let prompt = shared.provider.apply_template(&messages, add_assistant)?;
            reply(socket, 200, object([("prompt", prompt.into())]), headers)
        }
        ("POST", "/tokenize") => {
            let body = json_body(request)?;
            let content = body
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::invalid("content must be a string"))?;
            if !optional_bool(&body, "parse_special", true)? {
                return Err(ProviderError::unsupported(
                    "parse_special=false is not implemented by this tokenizer interface",
                ));
            }
            if optional_bool(&body, "with_pieces", false)? {
                return Err(ProviderError::unsupported(
                    "with_pieces=true is not implemented by this tokenizer interface",
                ));
            }
            if let Some(fields) = body.as_object() {
                for key in fields.keys() {
                    if !matches!(
                        key.as_str(),
                        "content" | "add_special" | "parse_special" | "with_pieces"
                    ) {
                        return Err(ProviderError::unsupported(format!(
                            "Tokenizer option {key} is not implemented"
                        )));
                    }
                }
            }
            let add_special = optional_bool(&body, "add_special", false)?;
            ensure_ready(shared)?;
            let tokens = shared
                .provider
                .tokenize(content, add_special)?
                .into_iter()
                .map(Value::from)
                .collect();
            reply(
                socket,
                200,
                object([("tokens", Value::Array(tokens))]),
                headers,
            )
        }
        ("POST", "/detokenize") => {
            let body = json_body(request)?;
            let items = body
                .get("tokens")
                .and_then(Value::as_array)
                .ok_or_else(|| ProviderError::invalid("tokens must be an array"))?;
            let tokens: Result<Vec<i32>, _> = items
                .iter()
                .map(|v| {
                    v.as_i64()
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or_else(|| ProviderError::invalid("tokens must contain i32 integers"))
                })
                .collect();
            ensure_ready(shared)?;
            reply(
                socket,
                200,
                object([("content", shared.provider.detokenize(&tokens?)?.into())]),
                headers,
            )
        }
        ("GET", "/v1/stream") => {
            let id = query_id(request)?;
            let offset = request
                .query("from")
                .map_err(|e| ProviderError::invalid(e.message))?
                .unwrap_or_else(|| "0".into());
            if offset.is_empty() || !offset.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ProviderError::invalid(
                    "from must be an unsigned byte offset",
                ));
            }
            let offset = offset
                .parse()
                .map_err(|_| ProviderError::invalid("from exceeds u64"))?;
            let session = shared.replay.get(&id).ok_or_else(|| {
                ProviderError::new(404, "stream_not_found", "Stream not found or expired")
            })?;
            if offset < session.info().oldest_byte {
                return Err(ProviderError::invalid("Stream offset lost, please restart"));
            }
            let mut response_headers = stream_headers(headers);
            response_headers.push(("X-Conversation-Id".into(), id));
            http::write_head(
                socket,
                200,
                "text/event-stream; charset=utf-8",
                None,
                &response_headers,
            )
            .map_err(io_error)?;
            stream_replay(socket, &session, offset, shared);
            Ok(())
        }
        ("DELETE", "/v1/stream") => {
            let id = query_id(request)?;
            shared.replay.delete(&id);
            write_bytes(socket, 204, "application/json", &[], headers).map_err(io_error)
        }
        ("POST", "/v1/streams/lookup") => {
            let body = json_body(request)?;
            let ids = body
                .get("conversation_ids")
                .and_then(Value::as_array)
                .unwrap_or(&[]);
            let ids: Vec<String> = ids
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            let info = shared
                .replay
                .lookup(&ids)
                .into_iter()
                .map(|info| {
                    object([
                        ("conversation_id", info.conversation_id.into()),
                        ("is_done", info.is_done.into()),
                        ("total_bytes", number(info.total_bytes)),
                        ("oldest_byte", number(info.oldest_byte)),
                        ("started_at", number(info.started_at)),
                        ("completed_at", number(info.completed_at.unwrap_or(0))),
                    ])
                })
                .collect();
            reply(socket, 200, Value::Array(info), headers)
        }
        ("GET" | "POST", "/slots" | "/lora-adapters" | "/props" | "/tools" | "/cors-proxy")
            if request.method == "POST" || path != "/props" =>
        {
            Err(ProviderError::new(
                403,
                "feature_disabled",
                "this feature is disabled",
            ))
        }
        ("POST" | "DELETE", "/models") => Err(ProviderError::unsupported(
            "Model routing and lifecycle management are not implemented",
        )),
        _ if known_path(path) => Err(ProviderError::new(
            405,
            "method_not_allowed",
            "Method is not allowed for this route",
        )),
        _ if pending_path(path) => Err(ProviderError::unsupported(
            "This source route is not implemented in the Rust checkpoint",
        )),
        _ => Err(ProviderError::new(404, "not_found", "Route not found")),
    }
}
fn known_path(path: &str) -> bool {
    matches!(
        path,
        "/" | "/health"
            | "/v1/health"
            | "/models"
            | "/v1/models"
            | "/props"
            | "/metrics"
            | "/apply-template"
            | "/tokenize"
            | "/detokenize"
            | "/completion"
            | "/completions"
            | "/v1/completions"
            | "/chat/completions"
            | "/v1/chat/completions"
            | "/v1/stream"
            | "/v1/streams/lookup"
    )
}
fn pending_path(path: &str) -> bool {
    matches!(
        path,
        "/responses"
            | "/v1/responses"
            | "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/embedding"
            | "/embeddings"
            | "/v1/embeddings"
            | "/rerank"
            | "/reranking"
            | "/v1/rerank"
            | "/v1/reranking"
            | "/tts"
            | "/infill"
            | "/v1/audio/transcriptions"
            | "/audio/transcriptions"
            | "/v1/realtime"
            | "/v1/chat/completions/control"
            | "/v1/systemone"
            | "/chat/completions/input_tokens"
            | "/v1/chat/completions/input_tokens"
            | "/responses/input_tokens"
            | "/v1/responses/input_tokens"
            | "/models/load"
            | "/models/unload"
            | "/models/sse"
    ) || path.starts_with("/slots/")
}
fn query_id(request: &Request) -> Result<String, ProviderError> {
    let id = request
        .query("conv_id")
        .map_err(|e| ProviderError::invalid(e.message))?
        .ok_or_else(|| ProviderError::invalid("conv_id is required"))?;
    if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
        return Err(ProviderError::invalid("Invalid conv_id"));
    }
    Ok(id)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wire {
    Native,
    Completion,
    Chat,
}
fn optional_bool(body: &Value, name: &str, default: bool) -> Result<bool, ProviderError> {
    match body.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| ProviderError::invalid(format!("{name} must be a boolean"))),
    }
}
fn parse_chat_messages(body: &Value) -> Result<Vec<crate::provider::ChatMessage>, ProviderError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ProviderError::invalid("messages must be an array"))?;
    if messages.is_empty() {
        return Err(ProviderError::invalid("messages must not be empty"));
    }
    let mut result = Vec::with_capacity(messages.len());
    for message in messages {
        let fields = message
            .as_object()
            .ok_or_else(|| ProviderError::invalid("Each message must be an object"))?;
        for key in fields.keys() {
            if !matches!(key.as_str(), "role" | "content") {
                return Err(ProviderError::unsupported(format!(
                    "Message attribute {key} is not implemented by this text transport"
                )));
            }
        }
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| ProviderError::invalid("message role must be a string"))?;
        if !matches!(role, "system" | "developer" | "user" | "assistant") {
            return Err(ProviderError::unsupported(
                "This checkpoint accepts system, developer, user, and assistant roles",
            ));
        }
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProviderError::unsupported("This checkpoint requires text-only message content")
            })?;
        result.push(ChatMessage {
            role: role.into(),
            content: content.into(),
        });
    }
    Ok(result)
}

fn parse_generation(request: &Request) -> Result<(GenerationRequest, bool, Wire), ProviderError> {
    let body = json_body(request)?;
    if body.as_object().is_none() {
        return Err(ProviderError::invalid("Request body must be an object"));
    }
    let wire = if request.path().contains("chat/completions") {
        Wire::Chat
    } else if matches!(request.path(), "/completion" | "/completions") {
        Wire::Native
    } else {
        Wire::Completion
    };
    let prompt = if wire == Wire::Chat {
        Prompt::Chat(parse_chat_messages(&body)?)
    } else {
        Prompt::Text(
            body.get("prompt")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::invalid("prompt must be a string"))?
                .into(),
        )
    };
    for key in [
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "logprobs",
        "top_logprobs",
        "stream_options",
    ] {
        if body.get(key).is_some_and(|value| !value.is_null()) {
            return Err(ProviderError::unsupported(format!(
                "{key} is not implemented by this transport"
            )));
        }
    }
    if body.get("n").is_some_and(|n| n.as_u64() != Some(1)) {
        return Err(ProviderError::invalid("Only n=1 is supported"));
    }
    let model = match body.get("model") {
        None => None,
        Some(value) => Some(
            value
                .as_str()
                .ok_or_else(|| ProviderError::invalid("model must be a string"))?
                .to_owned(),
        ),
    };
    let max_value = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .or_else(|| body.get("n_predict"));
    let max_tokens = match max_value {
        None => 128,
        Some(value) => value
            .as_u64()
            .filter(|n| *n > 0 && *n <= 1_048_576)
            .ok_or_else(|| {
                ProviderError::invalid("Token limit must be an integer from 1 through 1048576")
            })? as u32,
    };
    let temperature = match body.get("temperature") {
        None => 0.8,
        Some(value) => value
            .as_f64()
            .filter(|n| n.is_finite() && *n >= 0.0)
            .ok_or_else(|| ProviderError::invalid("temperature must be finite and nonnegative"))?,
    };
    let seed = match body.get("seed") {
        None => None,
        Some(value) => Some(
            value
                .as_u64()
                .ok_or_else(|| ProviderError::invalid("seed must be an unsigned integer"))?,
        ),
    };
    let stop = match body.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ProviderError::invalid("stop entries must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(ProviderError::invalid(
                "stop must be a string or an array of strings",
            ));
        }
    };
    if stop.iter().any(String::is_empty) || stop.len() > 64 {
        return Err(ProviderError::invalid(
            "stop accepts at most 64 nonempty strings",
        ));
    }
    let stream = optional_bool(&body, "stream", false)?;
    Ok((
        GenerationRequest {
            model,
            prompt,
            max_tokens,
            temperature,
            seed,
            stop,
            parameters: body,
        },
        stream,
        wire,
    ))
}

fn begin_generation(
    shared: &Arc<Shared>,
    cancel: CancellationToken,
) -> Result<GenerationGuard, ProviderError> {
    if shared.shutdown.is_cancelled() {
        return Err(ProviderError::unavailable("Server is shutting down"));
    }
    if !acquire(&shared.generations, shared.config.max_generations) {
        return Err(ProviderError::new(
            429,
            "server_busy",
            "Generation limit reached",
        ));
    }
    let id = shared.next_id.fetch_add(1, Ordering::Relaxed);
    shared.active.lock().unwrap().insert(
        id,
        ActiveJob {
            cancel,
            started: Instant::now(),
        },
    );
    Ok(GenerationGuard {
        shared: shared.clone(),
        id,
    })
}
fn spawn_producer(
    shared: &Arc<Shared>,
    work: impl FnOnce() + Send + 'static,
) -> Result<(), ProviderError> {
    let handle = thread::Builder::new()
        .name("llama-inference-provider".into())
        .spawn(work)
        .map_err(|e| ProviderError::internal(e.to_string()))?;
    let mut producers = shared.producers.lock().unwrap();
    reap(&mut producers);
    producers.push(handle);
    Ok(())
}
fn generation(
    socket: &mut TcpStream,
    request: &Request,
    shared: &Arc<Shared>,
    headers: &[(String, String)],
) -> Result<(), ProviderError> {
    let (generation, stream, wire) = parse_generation(request)?;
    ensure_ready(shared)?;
    shared.provider.validate_generation(&generation)?;
    let models = shared.provider.models();
    if generation
        .model
        .as_ref()
        .is_some_and(|id| !models.is_empty() && !models.iter().any(|model| &model.id == id))
    {
        return Err(ProviderError::new(
            404,
            "model_not_found",
            "Requested model is not available",
        ));
    }
    let model = generation
        .model
        .clone()
        .or_else(|| models.first().map(|model| model.id.clone()))
        .unwrap_or_else(|| "provider".into());
    if stream {
        stream_generation(socket, request, generation, wire, model, shared, headers)
    } else {
        complete_generation(socket, generation, wire, model, shared, headers)
    }
}

struct CollectSink {
    text: String,
    limit: usize,
    cancel: CancellationToken,
    started: Instant,
    timeout: Duration,
}
impl TokenSink for CollectSink {
    fn send(&mut self, text: &str) -> Result<(), ProviderError> {
        if self.cancel.is_cancelled() {
            return Err(ProviderError::cancelled());
        }
        if self.started.elapsed() >= self.timeout {
            self.cancel.cancel();
            return Err(ProviderError::new(
                504,
                "generation_timeout",
                "Generation deadline exceeded",
            ));
        }
        if text.len() > self.limit.saturating_sub(self.text.len()) {
            self.cancel.cancel();
            return Err(ProviderError::new(
                413,
                "response_too_large",
                "Generated text exceeds response limit",
            ));
        }
        self.text.push_str(text);
        Ok(())
    }
}
fn complete_generation(
    socket: &mut TcpStream,
    generation: GenerationRequest,
    wire: Wire,
    model: String,
    shared: &Arc<Shared>,
    headers: &[(String, String)],
) -> Result<(), ProviderError> {
    let cancel = CancellationToken::new();
    let guard = begin_generation(shared, cancel.clone())?;
    let id = format!("cmpl-rust-{}", guard.id);
    let (tx, rx) = mpsc::sync_channel(1);
    let state = shared.clone();
    let producer_cancel = cancel.clone();
    spawn_producer(shared, move || {
        let _guard = guard;
        let mut sink = CollectSink {
            text: String::new(),
            limit: state.config.max_response_bytes,
            cancel: producer_cancel.clone(),
            started: Instant::now(),
            timeout: state.config.generation_timeout,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state
                .provider
                .generate(&generation, &mut sink, &producer_cancel)
        }))
        .unwrap_or_else(|_| Err(ProviderError::internal("Inference provider panicked")));
        let _ = tx.send(result.map(|summary| (sink.text, summary)));
    })?;
    let started = Instant::now();
    loop {
        match rx.recv_timeout(Duration::from_millis(25)) {
            Ok(result) => {
                let (text, summary) = result?;
                let value = completion_value(wire, &id, &model, &text, &summary)?;
                let mut headers = headers.to_vec();
                headers.push(("X-Request-Id".into(), id));
                return reply(socket, 200, value, &headers);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(ProviderError::internal("Inference worker stopped"));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if shared.shutdown.is_cancelled() || started.elapsed() >= shared.config.generation_timeout {
            cancel.cancel();
            return Err(ProviderError::new(
                504,
                "generation_timeout",
                "Generation deadline exceeded",
            ));
        }
        if disconnected(socket) {
            cancel.cancel();
            return Ok(());
        }
    }
}
fn usage(summary: &GenerationSummary) -> Result<Value, ProviderError> {
    let total = summary
        .prompt_tokens
        .checked_add(summary.completion_tokens)
        .ok_or_else(|| ProviderError::internal("Provider token counts overflow"))?;
    Ok(object([
        ("prompt_tokens", number(summary.prompt_tokens)),
        ("completion_tokens", number(summary.completion_tokens)),
        ("total_tokens", number(total)),
    ]))
}
fn completion_value(
    wire: Wire,
    id: &str,
    model: &str,
    text: &str,
    summary: &GenerationSummary,
) -> Result<Value, ProviderError> {
    if wire == Wire::Native {
        return Ok(object([
            ("content", text.into()),
            ("stop", true.into()),
            ("tokens_predicted", number(summary.completion_tokens)),
            ("tokens_evaluated", number(summary.prompt_tokens)),
            (
                "stopped_eos",
                (summary.finish_reason == FinishReason::Stop).into(),
            ),
            (
                "stopped_word",
                (summary.finish_reason == FinishReason::StopSequence).into(),
            ),
            (
                "stopped_limit",
                (summary.finish_reason == FinishReason::Length).into(),
            ),
        ]));
    }
    let choice = if wire == Wire::Chat {
        object([
            ("index", number(0)),
            (
                "message",
                object([("role", "assistant".into()), ("content", text.into())]),
            ),
            ("finish_reason", summary.finish_reason.as_str().into()),
        ])
    } else {
        object([
            ("index", number(0)),
            ("text", text.into()),
            ("logprobs", Value::Null),
            ("finish_reason", summary.finish_reason.as_str().into()),
        ])
    };
    Ok(object([
        ("id", id.into()),
        (
            "object",
            if wire == Wire::Chat {
                "chat.completion"
            } else {
                "text_completion"
            }
            .into(),
        ),
        ("created", number(epoch())),
        ("model", model.into()),
        ("choices", Value::Array(vec![choice])),
        ("usage", usage(summary)?),
    ]))
}
fn epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone)]
enum StreamTarget {
    Queue(ByteQueue),
    Replay(Arc<ReplaySession>),
}
impl StreamTarget {
    fn write(
        &self,
        bytes: &[u8],
        cancel: &CancellationToken,
        timeout: Duration,
    ) -> Result<(), ProviderError> {
        if cancel.is_cancelled() {
            return Err(ProviderError::cancelled());
        }
        match self {
            Self::Queue(queue) => {
                let started = Instant::now();
                for chunk in bytes.chunks(queue.capacity()) {
                    queue
                        .send(chunk, cancel, timeout.saturating_sub(started.elapsed()))
                        .map_err(|error| match error {
                            QueueError::Timeout => ProviderError::new(
                                504,
                                "backpressure_timeout",
                                "Client did not drain the bounded stream queue",
                            ),
                            _ => ProviderError::cancelled(),
                        })?;
                }
                Ok(())
            }
            Self::Replay(session) => session
                .append(bytes)
                .map_err(|_| ProviderError::cancelled()),
        }
    }
    fn finish(&self) {
        match self {
            Self::Queue(queue) => queue.finish(),
            Self::Replay(session) => session.finish(),
        }
    }
}
struct StreamSink {
    target: StreamTarget,
    cancel: CancellationToken,
    wire: Wire,
    id: String,
    model: String,
    created: u64,
    bytes: usize,
    limit: usize,
    started: Instant,
    deadline: Duration,
    wait: Duration,
}
impl StreamSink {
    fn event(&self, value: Value) -> Result<(), ProviderError> {
        self.target
            .write(&sse_data(&encoded(&value)?), &self.cancel, self.wait)
    }
    fn chunk(&self, text: &str, role: bool, finish: Option<FinishReason>) -> Value {
        if self.wire == Wire::Native {
            return object([("content", text.into()), ("stop", finish.is_some().into())]);
        }
        let finish = finish
            .map(|reason| Value::from(reason.as_str()))
            .unwrap_or(Value::Null);
        let choice = if self.wire == Wire::Chat {
            let delta = if role {
                object([("role", "assistant".into()), ("content", text.into())])
            } else if text.is_empty() {
                Value::Object(BTreeMap::new())
            } else {
                object([("content", text.into())])
            };
            object([
                ("index", number(0)),
                ("delta", delta),
                ("finish_reason", finish),
            ])
        } else {
            object([
                ("index", number(0)),
                ("text", text.into()),
                ("logprobs", Value::Null),
                ("finish_reason", finish),
            ])
        };
        object([
            ("id", self.id.clone().into()),
            (
                "object",
                if self.wire == Wire::Chat {
                    "chat.completion.chunk"
                } else {
                    "text_completion"
                }
                .into(),
            ),
            ("created", number(self.created)),
            ("model", self.model.clone().into()),
            ("choices", Value::Array(vec![choice])),
        ])
    }
}
impl TokenSink for StreamSink {
    fn send(&mut self, text: &str) -> Result<(), ProviderError> {
        if self.cancel.is_cancelled() {
            return Err(ProviderError::cancelled());
        }
        if self.started.elapsed() >= self.deadline {
            self.cancel.cancel();
            return Err(ProviderError::new(
                504,
                "generation_timeout",
                "Generation deadline exceeded",
            ));
        }
        if text.len() > self.limit.saturating_sub(self.bytes) {
            return Err(ProviderError::new(
                413,
                "response_too_large",
                "Generated text exceeds response limit",
            ));
        }
        self.bytes += text.len();
        self.event(self.chunk(text, false, None))
    }
}
fn stream_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let mut headers = headers.to_vec();
    headers.push(("Cache-Control".into(), "no-cache".into()));
    headers.push(("X-Accel-Buffering".into(), "no".into()));
    headers
}
fn stream_generation(
    socket: &mut TcpStream,
    request: &Request,
    generation: GenerationRequest,
    wire: Wire,
    model: String,
    shared: &Arc<Shared>,
    headers: &[(String, String)],
) -> Result<(), ProviderError> {
    // Reserve an admission slot before allocating/replacing a replay session.
    let preliminary_cancel = CancellationToken::new();
    let guard = begin_generation(shared, preliminary_cancel.clone())?;
    let id = format!("cmpl-rust-{}", guard.id);
    let target =
        if let Some(conversation) = request.header("x-conversation-id") {
            StreamTarget::Replay(shared.replay.create(conversation).map_err(
                |error| match error {
                    ReplayError::Full => {
                        ProviderError::new(429, "stream_limit", "Replay session limit reached")
                    }
                    _ => ProviderError::invalid("Invalid X-Conversation-Id"),
                },
            )?)
        } else {
            StreamTarget::Queue(
                ByteQueue::new(shared.config.queue_bytes).map_err(ProviderError::invalid)?,
            )
        };
    let cancel = match &target {
        StreamTarget::Queue(_) => preliminary_cancel,
        StreamTarget::Replay(session) => session.cancellation(),
    };
    shared.active.lock().unwrap().insert(
        guard.id,
        ActiveJob {
            cancel: cancel.clone(),
            started: Instant::now(),
        },
    );
    let mut response_headers = stream_headers(headers);
    response_headers.push(("X-Request-Id".into(), id.clone()));
    if let Some(conversation) = request.header("x-conversation-id") {
        response_headers.push(("X-Conversation-Id".into(), conversation.into()));
    }
    let mut sink = StreamSink {
        target: target.clone(),
        cancel: cancel.clone(),
        wire,
        id,
        model,
        created: epoch(),
        bytes: 0,
        limit: shared.config.max_response_bytes,
        started: Instant::now(),
        deadline: shared.config.generation_timeout,
        wait: shared.config.write_timeout,
    };
    let state = shared.clone();
    let producer_target = target.clone();
    let consumer_cancel = cancel.clone();
    spawn_producer(shared, move || {
        let _guard = guard;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if wire == Wire::Chat {
                sink.event(sink.chunk("", true, None))?;
            }
            state.provider.generate(&generation, &mut sink, &cancel)
        }))
        .unwrap_or_else(|_| Err(ProviderError::internal("Inference provider panicked")));
        match result {
            Ok(summary) => {
                let final_value = if wire == Wire::Native {
                    completion_value(wire, &sink.id, &sink.model, "", &summary)
                } else {
                    Ok(sink.chunk("", false, Some(summary.finish_reason)))
                };
                if let Ok(value) = final_value {
                    let _ = sink.event(value);
                }
            }
            Err(error) => {
                let _ = sink.event(error_value(&error));
            }
        }
        if wire != Wire::Native {
            let _ = producer_target.write(&sse_data("[DONE]"), &sink.cancel, sink.wait);
        }
        producer_target.finish();
    })?;
    if let Err(error) = http::write_head(
        socket,
        200,
        "text/event-stream; charset=utf-8",
        None,
        &response_headers,
    ) {
        if let StreamTarget::Queue(queue) = &target {
            consumer_cancel.cancel();
            queue.abort();
        }
        return Err(io_error(error));
    }
    match &target {
        StreamTarget::Queue(queue) => {
            let started = Instant::now();
            loop {
                match queue.recv(Duration::from_millis(25)) {
                    QueueRead::Data(bytes) => {
                        if socket.write_all(&bytes).is_err() {
                            break;
                        }
                    }
                    QueueRead::Closed => break,
                    QueueRead::Timeout => {
                        if disconnected(socket) {
                            break;
                        }
                    }
                }
                if shared.shutdown.is_cancelled()
                    || started.elapsed() >= shared.config.generation_timeout
                {
                    break;
                }
            }
            // Closing the queue wakes a blocked producer, which must stop.
            queue.abort();
            consumer_cancel.cancel();
        }
        StreamTarget::Replay(session) => stream_replay(socket, session, 0, shared),
    }
    Ok(())
}
fn stream_replay(
    socket: &mut TcpStream,
    session: &ReplaySession,
    mut offset: u64,
    shared: &Shared,
) {
    let started = Instant::now();
    loop {
        match session.read_from(offset, 16 * 1024, Duration::from_millis(25)) {
            Ok(ReplayRead::Data(bytes)) => {
                if socket.write_all(&bytes).is_err() {
                    break;
                }
                offset += bytes.len() as u64;
            }
            Ok(ReplayRead::Timeout) => {
                if disconnected(socket) {
                    break;
                }
            }
            Ok(ReplayRead::Done) | Err(_) => break,
        }
        if shared.shutdown.is_cancelled() || started.elapsed() >= shared.config.generation_timeout {
            break;
        }
    }
    // Replay is explicitly detached from readers. Disconnect keeps generation
    // running into its bounded ring; DELETE/replacement/shutdown cancels it.
}
fn disconnected(socket: &TcpStream) -> bool {
    if socket.set_nonblocking(true).is_err() {
        return false;
    }
    let result = socket.peek(&mut [0]);
    let _ = socket.set_nonblocking(false);
    matches!(result, Ok(0))
        || result.is_err_and(|e| {
            !matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            )
        })
}
