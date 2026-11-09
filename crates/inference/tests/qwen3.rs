//! Independent frozen Qwen3 graph, tokenizer and shard captures.
//! All expected model and numeric bytes were captured through original C/C++
//! before this Rust comparison; provenance retains both compiler profiles.
#[path = "support/native_profile.rs"]
mod native_profile;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::{hash_reader, sha256};
use llama_gguf::GgufSet;
use llama_inference::{Error, FinishReason, GenerationOptions, LoadOptions, Model};
use llama_model_runtime::{BatchToken, Context, ContextOptions};
use native_profile::source_profile;
use std::{
    fs::{self, File},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("missing fixture field {key}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture array")
}
fn number(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("fixture integer")).unwrap()
}
fn string(value: &Value) -> &str {
    value.as_str().expect("fixture string")
}
fn boolean(value: &Value) -> bool {
    value.as_bool().expect("fixture boolean")
}
fn ids(value: &Value) -> Vec<i32> {
    array(value)
        .iter()
        .map(|v| i32::try_from(v.as_i64().unwrap()).unwrap())
        .collect()
}
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/inference/qwen3")
        .join(name)
}
fn pinned(name: &str, digest: &str) -> Vec<u8> {
    let bytes = fs::read(fixture(name)).unwrap();
    assert_eq!(sha256(&bytes), digest, "{name}");
    bytes
}
fn document(name: &str, digest: &str) -> Value {
    json::parse(std::str::from_utf8(&pinned(name, digest)).unwrap()).unwrap()
}
fn decode_hex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2));
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn values(reference: &Value) -> Vec<u8> {
    let data = field(reference, "float_data");
    let bytes = pinned(string(field(data, "path")), string(field(data, "sha256")));
    assert_eq!(bytes.len(), number(field(data, "bytes")));
    bytes
}

struct Corpus {
    root: PathBuf,
    document: Value,
}
impl Corpus {
    fn new() -> Self {
        Self::from_fixture(
            "tiny-models.json",
            "898e84c2567d91f1a5ac8132ddb82ca5034fce399a6b0e4645e5774798220b61",
            "tiny-inputs.bin",
            "04c331fdf68d9cb2311631904099d674ce1f17def63abf93c908a856147c49bd",
        )
    }
    fn vectors() -> Self {
        Self::from_fixture(
            "vectors-models.json",
            "bfe0d5e18746ad103cc4aac5d75d90cce3eb01ad64e767f7f108f51d1ca6ee24",
            "vectors-inputs.bin",
            "69dc2175a31be9be8f28d8cfaa875ab99943071378740f9372079c1eff8184db",
        )
    }
    fn from_fixture(
        document_name: &str,
        document_hash: &str,
        input_name: &str,
        input_hash: &str,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let document = document(document_name, document_hash);
        let data = pinned(input_name, input_hash);
        let root = std::env::temp_dir().join(format!(
            "rust-llama-qwen3-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        for record in array(field(&document, "files")) {
            let relative = Path::new(string(field(record, "path")));
            assert!(
                relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
            );
            let start = number(field(record, "offset"));
            let end = start.checked_add(number(field(record, "bytes"))).unwrap();
            let bytes = &data[start..end];
            assert_eq!(sha256(bytes), string(field(record, "sha256")));
            let path = root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        Self { root, document }
    }
    fn cases(&self) -> &[Value] {
        array(field(&self.document, "cases"))
    }
    fn case(&self, name: &str) -> &Value {
        self.cases()
            .iter()
            .find(|c| string(field(c, "name")) == name)
            .unwrap()
    }
    fn paths(&self, case: &Value) -> Vec<PathBuf> {
        array(field(case, "paths"))
            .iter()
            .map(|p| self.root.join(string(p)))
            .collect()
    }
    fn load(&self, case: &Value) -> llama_inference::Result<Model> {
        let paths = self.paths(case);
        let options = LoadOptions {
            context_length: 64,
            ..LoadOptions::default()
        };
        if boolean(field(case, "explicit")) {
            Model::load_from_splits_with_options(&paths, &options)
        } else {
            Model::load_with_options(&paths[0], &options)
        }
    }
}
impl Drop for Corpus {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[derive(Default)]
struct Metrics {
    floats: usize,
    bitwise: usize,
    greedy: usize,
    rows: usize,
    maximum: f64,
    fraction: f64,
}
fn compare_row(row: &[f32], data: &[u8], offset: usize, label: &str, metrics: &mut Metrics) {
    let raw = &data[offset..offset.checked_add(row.len() * 4).unwrap()];
    for (index, (&actual, expected)) in row.iter().zip(raw.as_chunks::<4>().0).enumerate() {
        let expected = f32::from_le_bytes(*expected);
        let difference = (f64::from(actual) - f64::from(expected)).abs();
        let budget = 2e-4 + 2e-5 * f64::from(expected).abs();
        assert!(
            actual.is_finite() && expected.is_finite() && difference <= budget,
            "{label} col={index}: actual={actual:.9e} expected={expected:.9e} difference={difference:.9e} budget={budget:.9e}"
        );
        if cfg!(all(
            target_arch = "x86_64",
            any(
                all(target_os = "windows", target_env = "msvc"),
                all(target_os = "linux", target_env = "gnu")
            )
        )) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label} col={index}: measured native profile must match exactly"
            );
        }
        metrics.floats += 1;
        metrics.bitwise += usize::from(actual.to_bits() == expected.to_bits());
        metrics.maximum = metrics.maximum.max(difference);
        metrics.fraction = metrics.fraction.max(difference / budget);
    }
    metrics.rows += 1;
}
fn check_model(model: &Model, reference: &Value, data: &[u8], label: &str, metrics: &mut Metrics) {
    assert!(model.supports_generation());
    assert!(!model.supports_embeddings());
    assert_eq!(model.runtime_model().unwrap().architecture().name, "qwen3");
    assert_eq!(model.vocabulary_size(), number(field(reference, "n_vocab")));
    for case in array(field(reference, "cases")) {
        let tokens = ids(field(case, "tokens"));
        let prompt = string(field(case, "prompt"));
        assert_eq!(
            model
                .tokenizer()
                .tokenize(
                    prompt,
                    boolean(field(case, "add_special")),
                    boolean(field(case, "parse_special"))
                )
                .unwrap(),
            tokens,
            "{label}"
        );
        for (index, &id) in tokens.iter().enumerate() {
            for (special, name) in [(false, "pieces_hex"), (true, "pieces_special_hex")] {
                assert_eq!(
                    model.tokenizer().token_piece(id, special).unwrap(),
                    decode_hex(string(&array(field(case, name))[index])),
                    "{label} token {id}"
                );
            }
        }
        assert_eq!(
            model
                .tokenizer()
                .detokenize_bytes(
                    &tokens,
                    boolean(field(reference, "detokenize_remove_special")),
                    boolean(field(reference, "detokenize_unparse_special"))
                )
                .unwrap(),
            decode_hex(string(field(case, "detokenized_hex"))),
            "{label}"
        );
        let greedy = ids(field(case, "greedy_tokens"));
        assert!(!greedy.is_empty());
        let mut sequence = tokens.clone();
        sequence.extend_from_slice(&greedy[..greedy.len() - 1]);
        let actual = model.logits_for_tokens(&sequence).unwrap();
        let prompt_rows = array(field(case, "prompt_rows"));
        let greedy_rows = array(field(case, "greedy_rows"));
        assert_eq!(prompt_rows.len(), tokens.len());
        assert_eq!(greedy_rows.len(), greedy.len());
        assert_eq!(number(prompt_rows.last().unwrap()), number(&greedy_rows[0]));
        let expected_rows = prompt_rows.iter().chain(greedy_rows.iter().skip(1));
        assert_eq!(actual.len(), prompt_rows.len() + greedy_rows.len() - 1);
        for (row, offset) in actual.iter().zip(expected_rows) {
            compare_row(row, data, number(offset), label, metrics);
        }
        for (step, &expected) in greedy.iter().enumerate() {
            let row = &actual[tokens.len() - 1 + step];
            let mut best = 0;
            for i in 1..row.len() {
                if row[i] > row[best] {
                    best = i;
                }
            }
            assert_eq!(best as i32, expected, "{label} greedy {step}");
            assert_eq!(
                model.tokenizer().token_piece(expected, false).unwrap(),
                decode_hex(string(&array(field(case, "greedy_pieces_hex"))[step]))
            );
            metrics.greedy += 1;
        }
        let generated = decode_hex(string(field(case, "generated_hex")));
        // Raw byte comparisons above remain exact for any piece; this string API
        // deliberately uses its documented replacement semantics for invalid UTF-8.
        let mut emitted = String::new();
        let stats = model
            .generate(
                prompt,
                &GenerationOptions {
                    max_tokens: number(field(reference, "max_tokens")),
                    ..GenerationOptions::default()
                },
                &|| false,
                &mut |piece| {
                    emitted.push_str(piece);
                    true
                },
            )
            .unwrap();
        assert_eq!(
            emitted,
            String::from_utf8_lossy(&generated),
            "{label} emitted"
        );
        assert_eq!(stats.prompt_tokens, tokens.len());
        assert_eq!(stats.completion_tokens, greedy.len());
        assert_eq!(
            stats.finish_reason,
            if boolean(field(case, "finished_eog")) {
                FinishReason::Eos
            } else {
                FinishReason::Length
            }
        );
    }
}
fn print_metrics(label: &str, metrics: &Metrics) {
    eprintln!(
        "qwen3-native: {label} host={} rows={} floats={} bitwise_equal={} greedy_ids={} maximum_absolute={:.9e} maximum_budget_fraction={:.9e}",
        source_profile("msvc", "gnu"),
        metrics.rows,
        metrics.floats,
        metrics.bitwise,
        metrics.greedy,
        metrics.maximum,
        metrics.fraction
    );
}

#[test]
fn original_dense_qwen3_heads_norms_biases_storage_and_shards() {
    let corpus = Corpus::new();
    let reference = document(
        source_profile("tiny-msvc-reference.json", "tiny-gnu-reference.json"),
        source_profile(
            "a928e0c5e8c8f8cec7448590c8c085384320ba45cf31709d3a44084b171735f5",
            "6d484698728c52560dafab55714116e8885cff1b16c45ff5b309cd9a23417370",
        ),
    );
    let values = values(&reference);
    assert_eq!(corpus.cases().len(), 51);
    let mut supported = 0;
    let mut unsupported = 0;
    let mut invalid = 0;
    let mut numeric = 0;
    let mut metrics = Metrics::default();
    for case in corpus.cases() {
        let name = string(field(case, "name"));
        let native = array(field(&reference, "cases"))
            .iter()
            .find(|c| string(field(c, "name")) == name)
            .unwrap();
        let result = corpus.load(case);
        match string(field(case, "rust_scope")) {
            "supported" => {
                let model = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(field(native, "exit_code").as_i64(), Some(0));
                check_model(
                    &model,
                    field(native, "reference"),
                    &values,
                    name,
                    &mut metrics,
                );
                supported += 1;
            }
            "unsupported" => {
                assert!(
                    matches!(result, Err(Error::Unsupported(_))),
                    "{name}: {result:?}"
                );
                unsupported += 1;
            }
            "invalid" => {
                assert!(
                    matches!(result, Err(Error::Invalid(_))),
                    "{name}: {result:?}"
                );
                assert_ne!(field(native, "exit_code").as_i64(), Some(0));
                invalid += 1;
            }
            "numeric-failure" => {
                let model = result.unwrap();
                let mut context =
                    Context::new(model.runtime_model().unwrap(), ContextOptions::default())
                        .unwrap();
                let before = context.save_state().unwrap();
                let result = context.decode_batch(&[BatchToken::token(0, 0, 1)]);
                assert!(
                    matches!(result, Err(llama_model_runtime::Error::NonFinite(_))),
                    "{name}: {result:?}"
                );
                assert_eq!(context.save_state().unwrap(), before);
                assert_eq!(context.reserved_state_bytes(), 0);
                assert_ne!(field(native, "exit_code").as_i64(), Some(0));
                numeric += 1;
            }
            other => panic!("unknown scope {other}"),
        }
    }
    assert_eq!((supported, unsupported, invalid, numeric), (39, 8, 3, 1));
    print_metrics(
        "39 supported / 8 unsupported / 3 invalid / 1 nonfinite",
        &metrics,
    );
}

#[test]
fn native_qwen3_weight_and_context_boundaries_are_checked_before_execution() {
    let corpus = Corpus::new();
    let paths = corpus.paths(corpus.case("fused-q4-k-metadata-first-explicit"));
    let set = GgufSet::open_splits(&paths).unwrap();
    assert_eq!(set.shard_info(0).unwrap().tensor_count, 0);
    let decoded = usize::try_from(set.tensor_elements()).unwrap() * 4;
    let options = LoadOptions {
        max_decoded_weight_bytes: decoded - 1,
        context_length: 64,
    };
    assert!(
        matches!(Model::load_from_gguf(set, &options), Err(Error::Invalid(message)) if message.contains("decoded model requires"))
    );
    let mut options = options;
    options.max_decoded_weight_bytes = decoded;
    options.context_length = 0;
    assert!(matches!(
        Model::load_from_splits_with_options(&paths, &options),
        Err(Error::Invalid(_))
    ));
    options.context_length = 2;
    let model = Model::load_from_splits_with_options(&paths, &options).unwrap();
    assert_eq!(model.decoded_weight_bytes(), decoded);
    assert_eq!(model.context_length(), 2);
    assert!(matches!(
        model.logits_for_tokens(&[1, 2, 3]),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        model.logits_for_tokens(&[-1]),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        model.logits_for_tokens(&[512]),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        model.embed("Hello", None, &|| false),
        Err(Error::Unsupported(_))
    ));
}

#[test]
#[ignore = "requires pinned official Qwen3-0.6B Q8_0 model via LLAMA_TEST_QWEN3_MODEL and 3 GiB decoded weight budget"]
fn official_qwen3_q8_matches_original_native_profile() {
    let path = std::env::var_os("LLAMA_TEST_QWEN3_MODEL").expect("set LLAMA_TEST_QWEN3_MODEL");
    let reference = document(
        source_profile(
            "pretrained-msvc-reference.json",
            "pretrained-gnu-reference.json",
        ),
        source_profile(
            "e0a466acda2e6ab476112e6c300a2a6cc65cc680dd40babd5e730c966a2a3601",
            "29e14fc25d406e8cc40f80a6a3147f41b1cd9f2d3c862f9f2557f384632cf3fa",
        ),
    );
    let (bytes, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(bytes, field(&reference, "model_bytes").as_u64().unwrap());
    assert_eq!(hash, string(field(&reference, "model_sha256")));
    assert!(matches!(Model::load_with_options(&path, &LoadOptions {
        context_length: 128, ..LoadOptions::default()
    }), Err(Error::Invalid(message)) if message.contains("decoded model requires")));
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            max_decoded_weight_bytes: number(field(&reference, "decoded_weight_budget")),
            context_length: number(field(&reference, "n_ctx")),
        },
    )
    .unwrap();
    let values = values(&reference);
    let mut metrics = Metrics::default();
    check_model(&model, &reference, &values, "Qwen3-0.6B-Q8_0", &mut metrics);
    assert_eq!(metrics.floats, 5_469_696);
    assert_eq!(metrics.greedy, 24);
    print_metrics("official Qwen3-0.6B Q8_0", &metrics);
}

// Rewriting these files constructs malformed input, never expected numeric
// output. The unchanged baseline bytes are pinned original-writer fixtures.
#[test]
fn qwen3_typed_metadata_and_unsupported_arrays_are_explicit() {
    use llama_gguf::{Gguf, TensorData, Value as Metadata, write};
    use std::collections::BTreeMap;
    let corpus = Corpus::new();
    let original = corpus.paths(corpus.case("separate-bias-0")).remove(0);
    let mut gguf = Gguf::open(original).unwrap();
    let descriptors = gguf.tensors.clone();
    let payloads: Vec<_> = descriptors
        .iter()
        .map(|d| gguf.read_tensor(&d.name).unwrap())
        .collect();
    let tensors: Vec<_> = descriptors
        .iter()
        .zip(&payloads)
        .map(|(d, data)| TensorData {
            name: &d.name,
            dimensions: &d.dimensions,
            dtype: d.dtype,
            data,
        })
        .collect();
    let mut index = 0usize;
    let mut check = |key: &str, value: Option<Metadata>, unsupported: bool| {
        let mut metadata: BTreeMap<_, _> = gguf.metadata.clone();
        if let Some(value) = value {
            metadata.insert(key.into(), value);
        } else {
            assert!(metadata.remove(key).is_some(), "missing original {key}");
        }
        index += 1;
        let path = corpus.root.join(format!("modified-{index}.gguf"));
        write(File::create(&path).unwrap(), &metadata, &tensors).unwrap();
        let result = Model::load(path);
        if unsupported {
            assert!(
                matches!(result, Err(Error::Unsupported(_))),
                "{key}: {result:?}"
            );
        } else {
            assert!(
                matches!(result, Err(Error::Invalid(_))),
                "{key}: {result:?}"
            );
        }
    };
    for key in [
        "qwen3.block_count",
        "qwen3.context_length",
        "qwen3.embedding_length",
        "qwen3.feed_forward_length",
        "qwen3.attention.head_count",
    ] {
        check(key, None, false);
        for value in [
            Metadata::U32(0),
            Metadata::U8(4),
            Metadata::U16(4),
            Metadata::U64(4),
            Metadata::I32(4),
        ] {
            check(key, Some(value), false);
        }
    }
    check("qwen3.attention.layer_norm_rms_epsilon", None, false);
    for (key, value) in [
        ("qwen3.block_count", Metadata::U32(513)),
        ("qwen3.attention.head_count_kv", Metadata::U32(0)),
        ("qwen3.attention.head_count_kv", Metadata::U32(3)),
        ("qwen3.attention.key_length", Metadata::U32(0)),
        ("qwen3.attention.key_length", Metadata::U32(3)),
        ("qwen3.attention.key_length", Metadata::U64(32)),
        ("qwen3.attention.value_length", Metadata::U64(32)),
        ("qwen3.attention.layer_norm_rms_epsilon", Metadata::F32(0.0)),
        (
            "qwen3.attention.layer_norm_rms_epsilon",
            Metadata::F32(f32::NAN),
        ),
        (
            "qwen3.attention.layer_norm_rms_epsilon",
            Metadata::F64(1e-5),
        ),
        ("qwen3.attention.causal", Metadata::U32(1)),
        ("qwen3.rope.freq_base", Metadata::F32(0.0)),
        ("qwen3.rope.freq_base", Metadata::F64(10000.0)),
        ("qwen3.pooling_type", Metadata::F32(0.0)),
    ] {
        check(key, Some(value), false);
    }
    for key in [
        "qwen3.attention.head_count",
        "qwen3.attention.head_count_kv",
        "qwen3.feed_forward_length",
    ] {
        check(
            key,
            Some(Metadata::Array {
                element_type: 4,
                values: vec![Metadata::U32(4); 2],
            }),
            true,
        );
    }
    for (key, value) in [
        ("qwen3.pooling_type", Metadata::U32(1)),
        ("qwen3.attention.sliding_window", Metadata::U32(4)),
        ("qwen3.expert_count", Metadata::U32(4)),
        ("qwen3.rope.scaling.attn_factor", Metadata::F32(0.5)),
        ("general.type", Metadata::String("adapter".into())),
        ("qwen3.nextn_predict_layers", Metadata::U32(1)),
        ("prism.hadamard.version", Metadata::U32(1)),
    ] {
        check(key, Some(value), true);
    }
}

#[test]
fn original_active_vector_storage_and_ignored_fused_bias_types() {
    let corpus = Corpus::vectors();
    let reference = document(
        source_profile("vectors-msvc-reference.json", "vectors-gnu-reference.json"),
        source_profile(
            "744bb855ae30eb200c1fc7004dc4e4cc1c358dd784ee826a8f43ee18bfbee605",
            "605bee2d26918a939b4186c0b6cb21b777d810f164c7b261970980cd3b50b310",
        ),
    );
    let values = values(&reference);
    assert_eq!(corpus.cases().len(), 17);
    let mut supported = 0;
    let mut rejected = 0;
    let mut mismatches = Vec::new();
    let mut metrics = Metrics::default();
    for case in corpus.cases() {
        let name = string(field(case, "name"));
        let native = array(field(&reference, "cases"))
            .iter()
            .find(|c| string(field(c, "name")) == name)
            .unwrap();
        let result = corpus.load(case);
        if string(field(case, "rust_scope")) == "supported" {
            assert_eq!(field(native, "exit_code").as_i64(), Some(0));
            let model = result.unwrap_or_else(|error| panic!("{name}: {error}"));
            check_model(
                &model,
                field(native, "reference"),
                &values,
                name,
                &mut metrics,
            );
            supported += 1;
        } else {
            assert_ne!(field(native, "exit_code").as_i64(), Some(0));
            match result {
                Err(Error::Unsupported(_)) => rejected += 1,
                Ok(_) => mismatches.push(format!("{name}: incorrectly admitted")),
                Err(error) => mismatches.push(format!("{name}: {error}")),
            }
        }
    }
    print_metrics(
        "active vector storage / ignored partial fused biases",
        &metrics,
    );
    assert!(
        mismatches.is_empty(),
        "native vector storage boundaries: {mismatches:?}"
    );
    assert_eq!((supported, rejected), (5, 12));
    assert_eq!(metrics.floats, 15_360);
    assert_eq!(metrics.greedy, 10);
}
