//! CUDA replays of original C-API corpora captured before this device port.
//! Model inputs, native float rows and their provenance remain in the shared
//! fixture set; no expected values are produced by a Rust execution path.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::{hash_reader, sha256};
use llama_cuda_runtime::{CudaDecoder, Error, Options};
use llama_inference::{FinishReason, GenerationOptions, LoadOptions, Model};
use std::{
    cell::Cell,
    fs,
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn string(v: &Value) -> &str {
    v.as_str().unwrap()
}
fn ids(v: &Value) -> Vec<u32> {
    array(v)
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn document(path: &Path) -> Value {
    json::parse(&fs::read_to_string(path).unwrap()).unwrap()
}
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/inference")
}
fn profile() -> &'static str {
    if cfg!(windows) { "msvc" } else { "gnu" }
}
fn hex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2));
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}

#[derive(Default)]
struct Metrics {
    models: usize,
    values: usize,
    exact: usize,
    greedy: usize,
    max_error: f64,
}
fn check_model(
    model: &Model,
    reference: &Value,
    data: &[u8],
    label: &str,
    tolerance: (f64, f64),
    metrics: &mut Metrics,
) {
    let mut gpu = CudaDecoder::from_inference_model(model, &Options::default()).unwrap();
    for case in array(field(reference, "cases")) {
        gpu.reset();
        let tokens = ids(field(case, "tokens"));
        let greedy = ids(field(case, "greedy_tokens"));
        assert!(!tokens.is_empty() && !greedy.is_empty());
        let mut sequence = tokens.clone();
        sequence.extend_from_slice(&greedy[..greedy.len() - 1]);
        let offsets: Vec<usize> = if let Some(rows) = case.get("prompt_rows") {
            array(rows)
                .iter()
                .chain(array(field(case, "greedy_rows")).iter().skip(1))
                .map(number)
                .collect()
        } else {
            let first = number(field(case, "first_row"));
            (first..first + sequence.len())
                .map(|r| r * model.vocabulary_size() * 4)
                .collect()
        };
        assert_eq!(offsets.len(), sequence.len());
        let mut first_logits = None;
        for (index, (&token, &offset)) in sequence.iter().zip(&offsets).enumerate() {
            let row = gpu.decode(token).unwrap().logits;
            if first_logits.is_none() {
                first_logits = Some(row.clone());
            }
            let expected = &data[offset..offset + row.len() * 4];
            for (column, (&actual, expected)) in
                row.iter().zip(expected.as_chunks::<4>().0).enumerate()
            {
                let expected = f32::from_le_bytes(*expected);
                let difference = (f64::from(actual) - f64::from(expected)).abs();
                let budget = tolerance.0 + tolerance.1 * f64::from(expected).abs();
                assert!(
                    actual.is_finite() && expected.is_finite() && difference <= budget,
                    "{label} row {index} col {column}: {actual} vs {expected}, error={difference:e}, budget={budget:e}"
                );
                metrics.values += 1;
                metrics.exact += usize::from(actual.to_bits() == expected.to_bits());
                metrics.max_error = metrics.max_error.max(difference);
            }
            if index + 1 >= tokens.len() {
                let mut best = 0;
                for i in 1..row.len() {
                    if row[i] > row[best] {
                        best = i;
                    }
                }
                assert_eq!(
                    best as u32,
                    greedy[index + 1 - tokens.len()],
                    "{label} greedy"
                );
                metrics.greedy += 1;
            }
        }
        // Failed entry, mid-graph cancellation, truncation and reset cannot
        // expose an uncommitted row to a later successful decode.
        let prefix = gpu.position();
        assert!(gpu.decode(u32::MAX).is_err());
        assert_eq!(gpu.position(), prefix);
        if prefix < gpu.context_length() {
            let count = Cell::new(0);
            assert!(matches!(
                gpu.decode_with_cancel(tokens[0], &|| {
                    count.set(count.get() + 1);
                    count.get() == 3
                }),
                Err(Error::Cancelled)
            ));
            assert_eq!(gpu.position(), prefix);
        }
        gpu.truncate(0).unwrap();
        assert_eq!(gpu.decode(tokens[0]).unwrap().logits, first_logits.unwrap());
        gpu.reset();
        let mut text = String::new();
        let stats = model
            .generate_with_session(
                string(field(case, "prompt")),
                &GenerationOptions {
                    max_tokens: number(field(reference, "max_tokens")),
                    ..GenerationOptions::default()
                },
                &|| false,
                &mut |part| {
                    text.push_str(part);
                    true
                },
                &mut gpu,
            )
            .unwrap();
        assert_eq!(
            text,
            String::from_utf8_lossy(&hex(string(field(case, "generated_hex")))),
            "{label} generated text"
        );
        assert_eq!(stats.completion_tokens, greedy.len());
        assert_eq!(
            stats.finish_reason,
            if field(case, "finished_eog").as_bool().unwrap() {
                FinishReason::Eos
            } else {
                FinishReason::Length
            }
        );
    }
    metrics.models += 1;
}
fn report(label: &str, m: &Metrics) {
    eprintln!(
        "CUDA {label}: models={} original_values={} bitwise_equal={} exact_greedy_ids={} maximum_absolute={:e}",
        m.models, m.values, m.exact, m.greedy, m.max_error
    );
}

#[test]
#[ignore = "requires an actual NVIDIA CUDA device; original Qwen2 C-API fixtures are included"]
fn cuda_qwen2_fused_biases_outputs_and_transactions_match_original() {
    let mut names: Vec<String> = [
        "separate",
        "fused-bias",
        "fused-q4",
        "no-bias",
        "fused-no-bias",
        "fused-separate-bias",
        "output-bias",
        "untied-output",
        "untied-output-bias",
    ]
    .iter()
    .map(|s| (*s).into())
    .collect();
    for mode in ["separate", "fused"] {
        for biases in ["q", "k", "v", "qk", "qv", "kv"] {
            names.push(format!("{mode}-partial-{biases}"));
        }
    }
    let root = fixtures();
    let mut metrics = Metrics::default();
    for name in names {
        let stem = format!("tiny-qwen2-{name}");
        let source = format!("{stem}{}", if cfg!(windows) { "" } else { "-gnu" });
        let provenance = document(&root.join(format!("{source}-provenance.json")));
        let model_provenance_bytes =
            fs::read(root.join(format!("{stem}-provenance.json"))).unwrap();
        let model_provenance =
            json::parse(std::str::from_utf8(&model_provenance_bytes).unwrap()).unwrap();
        if !cfg!(windows) {
            let shared = field(&provenance, "input_model_and_request_provenance");
            assert_eq!(
                string(field(shared, "path")),
                format!("fixtures/inference/{stem}-provenance.json")
            );
            assert_eq!(
                sha256(&model_provenance_bytes),
                string(field(shared, "sha256"))
            );
        }
        let model_path = root.join(format!("{stem}.gguf"));
        assert_eq!(
            hash_reader(fs::File::open(&model_path).unwrap()).unwrap().1,
            string(field(field(&model_provenance, "model"), "sha256"))
        );
        let metadata = fs::read(root.join(format!("{source}-metadata.json"))).unwrap();
        assert_eq!(
            sha256(&metadata),
            string(field(&provenance, "metadata_sha256"))
        );
        let reference = json::parse(std::str::from_utf8(&metadata).unwrap()).unwrap();
        let data = fs::read(root.join(format!("{source}-logits.f32le.bin"))).unwrap();
        assert_eq!(
            sha256(&data),
            string(field(field(&provenance, "logits"), "sha256"))
        );
        let model = Model::load_with_options(
            model_path,
            &LoadOptions {
                context_length: 32,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        // The tiny Qwen2 corpus fixed this tighter budget before the CPU port.
        check_model(&model, &reference, &data, &name, (2e-6, 2e-6), &mut metrics);
    }
    assert_eq!(metrics.models, 21);
    report("Qwen2", &metrics);
}

fn pretrained_qwen2(kind: &str, environment: &str) {
    let path = PathBuf::from(
        std::env::var(environment).unwrap_or_else(|_| panic!("{environment} is required")),
    );
    let root = fixtures().join("qwen2-cuda-pretrained");
    let stem = format!("{kind}-{}", profile());
    let provenance = document(&root.join(format!("{stem}-provenance.json")));
    let identity = field(&provenance, "model");
    let (length, hash) = hash_reader(fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(length as usize, number(field(identity, "bytes")));
    assert_eq!(hash, string(field(identity, "sha256")));
    let metadata = fs::read(root.join(format!("{stem}-metadata.json"))).unwrap();
    assert_eq!(
        sha256(&metadata),
        string(field(&provenance, "metadata_sha256"))
    );
    let reference = json::parse(std::str::from_utf8(&metadata).unwrap()).unwrap();
    let data_record = field(&provenance, "float_data");
    let data = fs::read(root.join(string(field(data_record, "path")))).unwrap();
    assert_eq!(data.len(), number(field(data_record, "bytes")));
    assert_eq!(sha256(&data), string(field(data_record, "sha256")));
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            context_length: 128,
            max_decoded_weight_bytes: 3 * 1024 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut metrics = Metrics::default();
    check_model(&model, &reference, &data, kind, (2e-4, 2e-5), &mut metrics);
    assert_eq!(
        metrics.values,
        number(field(&provenance, "original_values"))
    );
    assert_eq!(
        metrics.greedy,
        number(field(&provenance, "exact_greedy_ids"))
    );
    report(&format!("pretrained Qwen2 {kind}"), &metrics);
}

#[test]
#[ignore = "requires actual NVIDIA CUDA and the pinned LLAMA_TEST_QWEN2_F16_MODEL"]
fn cuda_pretrained_qwen2_f16_matches_original() {
    pretrained_qwen2("f16", "LLAMA_TEST_QWEN2_F16_MODEL");
}

#[test]
#[ignore = "requires actual NVIDIA CUDA and the pinned LLAMA_TEST_QWEN2_Q8_MODEL"]
fn cuda_pretrained_qwen2_q8_matches_original() {
    pretrained_qwen2("q8_0", "LLAMA_TEST_QWEN2_Q8_MODEL");
}

struct Corpus {
    root: PathBuf,
    cases: Value,
}
impl Corpus {
    fn new(kind: &str) -> Self {
        let source = fixtures().join("qwen3");
        let cases = document(&source.join(format!("{kind}-models.json")));
        let data = fs::read(source.join(format!("{kind}-inputs.bin"))).unwrap();
        let root = std::env::temp_dir().join(format!(
            "rust-llama-cuda-qwen3-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        for record in array(field(&cases, "files")) {
            let relative = Path::new(string(field(record, "path")));
            assert!(
                relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
            );
            let start = number(field(record, "offset"));
            let bytes = &data[start..start.checked_add(number(field(record, "bytes"))).unwrap()];
            assert_eq!(sha256(bytes), string(field(record, "sha256")));
            let target = root.join(relative);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::write(target, bytes).unwrap();
        }
        Self { root, cases }
    }
}
impl Drop for Corpus {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn reference_data(reference: &Value) -> Vec<u8> {
    let record = field(reference, "float_data");
    let bytes = fs::read(fixtures().join("qwen3").join(string(field(record, "path")))).unwrap();
    assert_eq!(sha256(&bytes), string(field(record, "sha256")));
    bytes
}

#[test]
#[ignore = "requires an actual NVIDIA CUDA device; original Qwen3 C-API fixtures are included"]
fn cuda_qwen3_heads_fusion_shards_and_storage_match_original() {
    let mut metrics = Metrics::default();
    let mut rejected_formats = 0;
    for kind in ["tiny", "vectors"] {
        let corpus = Corpus::new(kind);
        let reference = document(
            &fixtures()
                .join("qwen3")
                .join(format!("{kind}-{}-reference.json", profile())),
        );
        let data = reference_data(&reference);
        for case in array(field(&corpus.cases, "cases")) {
            if string(field(case, "rust_scope")) != "supported" {
                continue;
            }
            let name = string(field(case, "name"));
            let paths: Vec<_> = array(field(case, "paths"))
                .iter()
                .map(|p| corpus.root.join(string(p)))
                .collect();
            let options = LoadOptions {
                context_length: 64,
                ..LoadOptions::default()
            };
            let model = if field(case, "explicit").as_bool().unwrap() {
                Model::load_from_splits_with_options(&paths, &options)
            } else {
                Model::load_with_options(&paths[0], &options)
            }
            .unwrap();
            if let Err(error) = CudaDecoder::from_inference_model(&model, &Options::default()) {
                assert!(
                    name.contains("q4-k")
                        && matches!(error, Error::Unsupported(ref s) if s.contains("dtype")),
                    "{name}: {error}"
                );
                rejected_formats += 1;
                continue;
            }
            let native = array(field(&reference, "cases"))
                .iter()
                .find(|v| string(field(v, "name")) == name)
                .unwrap();
            assert_eq!(field(native, "exit_code").as_i64(), Some(0));
            check_model(
                &model,
                field(native, "reference"),
                &data,
                name,
                (2e-4, 2e-5),
                &mut metrics,
            );
        }
    }
    assert_eq!(rejected_formats, 6);
    assert!(metrics.models >= 33);
    report("Qwen3 tiny and vector-storage", &metrics);
    eprintln!(
        "CUDA Qwen3: {rejected_formats} Q4_K cases explicitly rejected before device execution"
    );
}

#[test]
#[ignore = "requires actual CUDA and pinned LLAMA_TEST_QWEN3_MODEL Qwen3-0.6B Q8_0"]
fn cuda_pretrained_qwen3_q8_matches_original() {
    let path = std::env::var_os("LLAMA_TEST_QWEN3_MODEL").expect("LLAMA_TEST_QWEN3_MODEL");
    let reference = document(
        &fixtures()
            .join("qwen3")
            .join(format!("pretrained-{}-reference.json", profile())),
    );
    let (size, hash) = hash_reader(fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(size, field(&reference, "model_bytes").as_u64().unwrap());
    assert_eq!(hash, string(field(&reference, "model_sha256")));
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            context_length: number(field(&reference, "n_ctx")),
            max_decoded_weight_bytes: number(field(&reference, "decoded_weight_budget")),
        },
    )
    .unwrap();
    let data = reference_data(&reference);
    let mut metrics = Metrics::default();
    check_model(
        &model,
        &reference,
        &data,
        "Qwen3-0.6B Q8_0",
        (2e-4, 2e-5),
        &mut metrics,
    );
    assert_eq!((metrics.values, metrics.greedy), (5_469_696, 24));
    report("pretrained Qwen3 Q8_0", &metrics);
}
