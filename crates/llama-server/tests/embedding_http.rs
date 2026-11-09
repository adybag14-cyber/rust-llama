//! Original C++ server transcripts executed through real local TCP and Rust
//! BERT inference. Expected vectors never come from the Rust implementation.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::hash_reader;
use llama_inference::Model;
use llama_server::{EmbeddingOptions, Pooling, RuntimeProvider, Server, ServerConfig};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

// Expected profiles are chosen from target cfg independently of the runtime
// dispatch, so a wrong runtime policy cannot also switch its expected output.
#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
const TINY: &str = include_str!("../../../fixtures/inference/embedding-http-tiny-gnu.json");
#[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
const TINY: &str = include_str!("../../../fixtures/inference/embedding-http-tiny.json");
#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
const BGE: &str = include_str!("../../../fixtures/inference/embedding-http-bge-gnu.json");
#[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
const BGE: &str = include_str!("../../../fixtures/inference/embedding-http-bge.json");
#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
const COERCIONS: &str =
    include_str!("../../../fixtures/inference/embedding-http-coercions-gnu.json");
#[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
const COERCIONS: &str = include_str!("../../../fixtures/inference/embedding-http-coercions.json");
const EXACT_NATIVE_PROFILE: bool = cfg!(any(
    all(
        target_os = "windows",
        target_arch = "x86_64",
        target_env = "msvc"
    ),
    all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ),
));

struct Reply {
    status: u16,
    body: Value,
}

fn post(address: SocketAddr, path: &str, body: &Value) -> Reply {
    let body = json::stringify(body).unwrap();
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    socket.read_to_end(&mut response).unwrap();
    let response = String::from_utf8(response).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    let length: usize = headers
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length: "))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(length, body.len());
    assert!(!headers.contains("text/event-stream"));
    Reply {
        status,
        body: json::parse(body).unwrap(),
    }
}

#[derive(Default)]
struct Comparisons {
    requests: usize,
    numbers: usize,
    base64_vectors: usize,
    maximum_error: f64,
}

fn base64_bytes(encoded: &str) -> Vec<u8> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    assert_eq!(encoded.len() % 4, 0);
    let mut decoded = Vec::new();
    for (index, chunk) in encoded.as_bytes().as_chunks::<4>().0.iter().enumerate() {
        let mut values = [0_u8; 4];
        let mut padding = 0;
        for (i, byte) in chunk.iter().enumerate() {
            if *byte == b'=' {
                assert!(i >= 2);
                padding += 1;
            } else {
                assert_eq!(padding, 0);
                values[i] = alphabet.iter().position(|x| x == byte).unwrap() as u8;
            }
        }
        assert!(padding <= 2);
        if padding != 0 {
            assert_eq!((index + 1) * 4, encoded.len());
        }
        decoded.push((values[0] << 2) | (values[1] >> 4));
        if padding < 2 {
            decoded.push((values[1] << 4) | (values[2] >> 2));
        }
        if padding == 0 {
            decoded.push((values[2] << 6) | values[3]);
        }
        if padding == 2 {
            assert_eq!(values[1] & 15, 0);
        } else if padding == 1 {
            assert_eq!(values[2] & 3, 0);
        }
    }
    decoded
}

fn compare_number(
    actual: f64,
    expected: f64,
    absolute: f64,
    relative: f64,
    location: &str,
    metrics: &mut Comparisons,
) {
    if !location.contains("/embedding/") && !location.ends_with("/embedding") {
        assert_eq!(actual, expected, "{location}: protocol integer differs");
    }
    let difference = (actual - expected).abs();
    assert!(
        difference <= absolute + relative * expected.abs(),
        "{location}: {actual} != {expected}; absolute error {difference}"
    );
    if EXACT_NATIVE_PROFILE {
        assert_eq!(
            (actual as f32).to_bits(),
            (expected as f32).to_bits(),
            "{location}: matching-profile native f32 bits differ"
        );
    }
    metrics.numbers += 1;
    metrics.maximum_error = metrics.maximum_error.max(difference);
}

fn compare(
    actual: &Value,
    expected: &Value,
    location: &str,
    model: &str,
    absolute: f64,
    relative: f64,
    metrics: &mut Comparisons,
) {
    if location == "/model" && expected.as_str() == Some("fixture-model") {
        // The native capture uses --alias fixture-model; the Rust test provider
        // uses its actual model identifier. Explicit request labels remain exact.
        assert_eq!(actual.as_str(), Some(model));
        return;
    }
    if let (Some(actual), Some(expected)) = (actual.as_f64(), expected.as_f64()) {
        return compare_number(actual, expected, absolute, relative, location, metrics);
    }
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>(),
                "{location}"
            );
            for (key, expected) in expected {
                compare(
                    &actual[key],
                    expected,
                    &format!("{location}/{key}"),
                    model,
                    absolute,
                    relative,
                    metrics,
                );
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{location}");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare(
                    actual,
                    expected,
                    &format!("{location}/{index}"),
                    model,
                    absolute,
                    relative,
                    metrics,
                );
            }
        }
        (Value::String(actual), Value::String(expected)) if location.ends_with("/embedding") => {
            let actual_bytes = base64_bytes(actual);
            let expected_bytes = base64_bytes(expected);
            assert_eq!(actual_bytes.len(), expected_bytes.len());
            assert_eq!(actual_bytes.len() % 4, 0);
            if EXACT_NATIVE_PROFILE {
                assert_eq!(actual, expected, "{location}: native base64 bytes differ");
            }
            for (actual, expected) in actual_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected_bytes.as_chunks::<4>().0.iter())
            {
                compare_number(
                    f64::from(f32::from_ne_bytes(*actual)),
                    f64::from(f32::from_le_bytes(*expected)),
                    absolute,
                    relative,
                    location,
                    metrics,
                );
            }
            metrics.base64_vectors += 1;
        }
        _ => assert_eq!(actual, expected, "{location}"),
    }
}

/// Propagate the unchanged raw BERT error envelope through max-absolute
/// normalization. This rule was recorded before the first Rust BGE HTTP run.
/// M is the native raw maximum and E bounds its possible change.
fn mode_zero_budget(group: &Value, absolute: f64, relative: f64) -> (f64, f64) {
    let raw = group
        .get("cases")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case.get("name").and_then(Value::as_str) == Some("raw_batch"))
        .unwrap();
    let row = raw
        .get("response")
        .unwrap()
        .get("body")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("embedding")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .as_array()
        .unwrap();
    let maximum = row
        .iter()
        .map(|value| value.as_f64().unwrap().abs())
        .fold(0.0_f64, f64::max);
    let error = absolute + relative * maximum;
    assert!(maximum > error);
    (
        32760.0 * absolute / (maximum - error),
        (relative * maximum + error) / (maximum - error) + 4.0 * f64::from(f32::EPSILON),
    )
}

fn verify(model_path: &Path, reference: &Value) -> Comparisons {
    let mut metrics = Comparisons::default();
    let absolute = reference
        .get("absolute_tolerance")
        .unwrap()
        .as_f64()
        .unwrap();
    let relative = reference
        .get("relative_tolerance")
        .unwrap()
        .as_f64()
        .unwrap();
    for group in reference
        .get("cases_by_pooling")
        .unwrap()
        .as_array()
        .unwrap()
    {
        let pooling = group.get("pooling").unwrap().as_str().unwrap();
        let options = EmbeddingOptions {
            pooling: match pooling {
                "model" => None,
                "none" => Some(Pooling::None),
                "mean" => Some(Pooling::Mean),
                "cls" | "cls_norm1" => Some(Pooling::First),
                "last" => Some(Pooling::Last),
                _ => panic!("Unknown fixture pooling mode"),
            },
            normalization: if pooling == "cls_norm1" { 1 } else { 2 },
        };
        let provider = RuntimeProvider::from_model(Model::load(model_path).unwrap())
            .with_embeddings(options)
            .unwrap();
        let model = provider.model_id().to_owned();
        let server = Server::new(
            ServerConfig {
                bind: "127.0.0.1:0".parse().unwrap(),
                ..ServerConfig::default()
            },
            Arc::new(provider),
        )
        .unwrap()
        .spawn()
        .unwrap();
        for case in group.get("cases").unwrap().as_array().unwrap() {
            let name = case.get("name").unwrap().as_str().unwrap();
            let path = case.get("path").unwrap().as_str().unwrap();
            let actual = post(server.local_addr(), path, case.get("request").unwrap());
            let expected = case.get("response").unwrap();
            assert_eq!(
                u64::from(actual.status),
                expected.get("status").unwrap().as_u64().unwrap(),
                "{pooling}/{name}/{path}: {:?}",
                actual.body
            );
            if actual.status == 200 {
                let (absolute, relative) = if name == "norm_0" {
                    mode_zero_budget(group, absolute, relative)
                } else {
                    (absolute, relative)
                };
                let mut case_metrics = Comparisons::default();
                compare(
                    &actual.body,
                    expected.get("body").unwrap(),
                    "",
                    &model,
                    absolute,
                    relative,
                    &mut case_metrics,
                );
                if case_metrics.maximum_error != 0.0 {
                    eprintln!(
                        "{pooling}/{name}: maximum absolute error {}, absolute budget {absolute}, relative budget {relative}",
                        case_metrics.maximum_error,
                    );
                }
                metrics.numbers += case_metrics.numbers;
                metrics.base64_vectors += case_metrics.base64_vectors;
                metrics.maximum_error = metrics.maximum_error.max(case_metrics.maximum_error);
            } else {
                let actual = actual.body.get("error").unwrap();
                let expected = expected.get("body").unwrap().get("error").unwrap();
                assert_eq!(actual.get("type"), expected.get("type"), "{pooling}/{name}");
                assert_eq!(actual.get("code"), expected.get("code"));
            }
            metrics.requests += 1;
        }
        server.shutdown().unwrap();
    }
    metrics
}

#[test]
fn tiny_bert_matches_original_embedding_http_transcripts() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/inference/tiny-bert-missing-pooling.gguf");
    let metrics = verify(&path, &json::parse(TINY).unwrap());
    assert_eq!(metrics.requests, 226);
    assert_eq!(metrics.base64_vectors, 4);
    eprintln!(
        "Original embedding HTTP: {} requests, {} numeric comparisons, {} base64 vectors, maximum absolute error {}",
        metrics.requests, metrics.numbers, metrics.base64_vectors, metrics.maximum_error,
    );
}

#[test]
#[ignore = "requires the hash-pinned BGE model; set LLAMA_TEST_BERT_MODEL and run --release --ignored --nocapture"]
fn pretrained_bert_matches_original_embedding_http_transcripts() {
    let path = std::path::PathBuf::from(
        std::env::var_os("LLAMA_TEST_BERT_MODEL").expect("set LLAMA_TEST_BERT_MODEL"),
    );
    let (bytes, hash) = hash_reader(std::fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(bytes, 67_308_128);
    assert_eq!(
        hash,
        "f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999"
    );
    let metrics = verify(&path, &json::parse(BGE).unwrap());
    assert_eq!(metrics.requests, 27);
    assert_eq!(metrics.base64_vectors, 3);
    eprintln!(
        "Original pretrained BGE embedding HTTP: {} requests, {} numeric comparisons, {} base64 vectors, maximum absolute error across output scales {}",
        metrics.requests, metrics.numbers, metrics.base64_vectors, metrics.maximum_error,
    );
}

#[test]
fn json_numeric_conversions_match_original_embedding_server() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/inference/tiny-bert-missing-pooling.gguf");
    let reference = json::parse(COERCIONS).unwrap();
    let metrics = verify(&path, &reference);
    assert_eq!(metrics.requests, 30);
    eprintln!(
        "Original embedding numeric conversions: {} requests, {} numeric comparisons, maximum absolute error {}",
        metrics.requests, metrics.numbers, metrics.maximum_error,
    );
}
