//! Independent frozen GPT-NeoX graph and tokenizer captures.
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
        .join("../../fixtures/inference/gptneox")
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
            "ad6befa81f5595ed543108bba7372483f71cb741f9562728ee40e68548e2e353",
            "tiny-inputs.bin",
            "34e3afb0cca67ce6575c2df66f79f7601a3f7780ec3291ebe7529fede68934ce",
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
            "rust-llama-gptneox-{}-{}-{}",
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
    assert_eq!(
        model.runtime_model().unwrap().architecture().name,
        "gptneox"
    );
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
        "gptneox-native: {label} host={} rows={} floats={} bitwise_equal={} greedy_ids={} maximum_absolute={:.9e} maximum_budget_fraction={:.9e}",
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
fn original_gptneox_rope_residual_biases_storage_and_gqa() {
    let corpus = Corpus::new();
    let reference = document(
        source_profile("tiny-msvc-reference.json", "tiny-gnu-reference.json"),
        source_profile(
            "2ec59be5939e9eba3812fcc9034567896620395764adb88208898c65422cdf1b",
            "571f56e5272a41175535c91f68be7352d75ad21fdc0c2a17e0dacc757f93d0b4",
        ),
    );
    let values = values(&reference);
    assert_eq!(corpus.cases().len(), 24);
    let mut counts = [0usize; 3];
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
                assert_eq!(field(native, "exit_code").as_i64(), Some(0));
                check_model(
                    &result.unwrap_or_else(|e| panic!("{name}: {e}")),
                    field(native, "reference"),
                    &values,
                    name,
                    &mut metrics,
                );
                counts[0] += 1;
            }
            "unsupported" => {
                assert!(
                    matches!(result, Err(Error::Unsupported(_))),
                    "{name}: {result:?}"
                );
                counts[1] += 1;
            }
            "invalid" => {
                assert!(
                    matches!(result, Err(Error::Invalid(_))),
                    "{name}: {result:?}"
                );
                assert_ne!(field(native, "exit_code").as_i64(), Some(0));
                counts[2] += 1;
            }
            other => panic!("unexpected scope {other}"),
        }
    }
    assert_eq!(counts, [16, 6, 2]);
    print_metrics(
        "16 executable tiny variants / 8 checked boundaries",
        &metrics,
    );
}

#[test]
#[ignore = "requires pinned official Pythia-14M F16 conversion via LLAMA_TEST_GPTNEOX_MODEL"]
fn official_pythia_14m_f16_matches_original_native_profile() {
    let path = std::env::var_os("LLAMA_TEST_GPTNEOX_MODEL").expect("set LLAMA_TEST_GPTNEOX_MODEL");
    let reference = document(
        source_profile(
            "pretrained-msvc-reference.json",
            "pretrained-gnu-reference.json",
        ),
        source_profile(
            "fb34928fe06988cdbb756f99e3a8c5967aa8e72b8bebdc444a203223fe2be5d0",
            "3abacfa02f7f05291bad1f12632d972191f92fce3fd04da1fdb86f17a71d34fe",
        ),
    );
    let (bytes, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(bytes, field(&reference, "model_bytes").as_u64().unwrap());
    assert_eq!(hash, string(field(&reference, "model_sha256")));
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            max_decoded_weight_bytes: number(field(&reference, "decoded_weight_budget")),
            context_length: number(field(&reference, "n_ctx")),
        },
    )
    .unwrap();
    let mut metrics = Metrics::default();
    check_model(
        &model,
        &reference,
        &values(&reference),
        "official Pythia-14M F16",
        &mut metrics,
    );
    assert_eq!(metrics.greedy, 24);
    print_metrics("official Pythia-14M F16", &metrics);
}

#[test]
fn gptneox_weight_budget_and_context_boundaries() {
    let corpus = Corpus::new();
    let path = corpus.paths(corpus.case("f32-parallel")).remove(0);
    let set = GgufSet::open(&path).unwrap();
    let decoded = usize::try_from(set.tensor_elements()).unwrap() * 4;
    assert!(matches!(Model::load_from_gguf(set,&LoadOptions{
        max_decoded_weight_bytes:decoded-1,context_length:64,
    }),Err(Error::Invalid(message)) if message.contains("decoded model requires")));
    let options = LoadOptions {
        max_decoded_weight_bytes: decoded,
        context_length: 2,
    };
    let model = Model::load_with_options(&path, &options).unwrap();
    assert_eq!(model.context_length(), 2);
    assert_eq!(model.decoded_weight_bytes(), decoded);
    assert!(matches!(
        model.logits_for_tokens(&[1, 2, 3]),
        Err(Error::Invalid(_))
    ));
    for id in [-1, 512] {
        assert!(matches!(
            model.logits_for_tokens(&[id]),
            Err(Error::Invalid(_))
        ));
    }
    assert!(matches!(
        model.embed("hello", None, &|| false),
        Err(Error::Unsupported(_))
    ));
    let whole = Model::load_with_options(
        &path,
        &LoadOptions {
            context_length: 1024,
            ..options
        },
    )
    .unwrap();
    assert_eq!(whole.context_length(), 128);
    assert_eq!(
        model.logits_for_tokens(&[1, 2]).unwrap(),
        whole.logits_for_tokens(&[1, 2]).unwrap()
    );
}

#[test]
fn gptneox_typed_metadata_and_unsupported_domains_are_explicit() {
    use llama_gguf::{Gguf, TensorData, Value as Metadata, write};
    let corpus = Corpus::new();
    let path = corpus.paths(corpus.case("f32-parallel")).remove(0);
    let mut g = Gguf::open(path).unwrap();
    let desc = g.tensors.clone();
    let payload: Vec<_> = desc
        .iter()
        .map(|d| g.read_tensor(&d.name).unwrap())
        .collect();
    let tensors: Vec<_> = desc
        .iter()
        .zip(&payload)
        .map(|(d, data)| TensorData {
            name: &d.name,
            dimensions: &d.dimensions,
            dtype: d.dtype,
            data,
        })
        .collect();
    let mut count = 0;
    let mut check = |key: &str, value: Option<Metadata>, unsupported: bool| {
        let mut metadata = g.metadata.clone();
        if let Some(value) = value {
            metadata.insert(key.into(), value);
        } else {
            assert!(metadata.remove(key).is_some());
        }
        count += 1;
        let path = corpus.root.join(format!("modified-{count}.gguf"));
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
        "gptneox.block_count",
        "gptneox.context_length",
        "gptneox.embedding_length",
        "gptneox.feed_forward_length",
        "gptneox.attention.head_count",
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
    check("gptneox.use_parallel_residual", None, false);
    check(
        "gptneox.use_parallel_residual",
        Some(Metadata::U32(1)),
        false,
    );
    check("gptneox.rope.dimension_count", Some(Metadata::U32(0)), true);
    for dimension in [1, 17, 18] {
        check(
            "gptneox.rope.dimension_count",
            Some(Metadata::U32(dimension)),
            false,
        );
    }
    for base in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        check("gptneox.rope.freq_base", Some(Metadata::F32(base)), false);
    }
    check("gptneox.attention.layer_norm_epsilon", None, false);
    for value in [
        Metadata::F32(0.0),
        Metadata::F32(f32::NAN),
        Metadata::F64(1e-5),
    ] {
        check("gptneox.attention.layer_norm_epsilon", Some(value), false);
    }
    check("gptneox.block_count", Some(Metadata::U32(513)), false);
    check(
        "gptneox.attention.head_count_kv",
        Some(Metadata::U32(3)),
        false,
    );
    check("gptneox.attention.causal", Some(Metadata::U32(1)), false);
    for key in [
        "gptneox.attention.head_count",
        "gptneox.attention.head_count_kv",
        "gptneox.feed_forward_length",
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
        ("gptneox.attention.causal", Metadata::Bool(false)),
        ("gptneox.attention.key_length", Metadata::U32(8)),
        ("gptneox.attention.value_length", Metadata::U32(8)),
        ("gptneox.attention.sliding_window", Metadata::U32(8)),
        ("gptneox.pooling_type", Metadata::U32(1)),
        ("gptneox.rope.scaling.factor", Metadata::F32(2.0)),
        ("gptneox.rope.scaling.type", Metadata::String("yarn".into())),
        ("gptneox.rope.scaling.attn_factor", Metadata::F32(2.0)),
        ("gptneox.nextn_predict_layers", Metadata::U32(1)),
        ("general.type", Metadata::String("adapter".into())),
        ("prism.hadamard.version", Metadata::U32(1)),
    ] {
        check(key, Some(value), true);
    }
}

#[test]
fn gptneox_context_checkpoint_cancellation_and_embedding_input() {
    let corpus = Corpus::new();
    let model = corpus.load(corpus.case("grouped-query")).unwrap();
    let runtime = model.runtime_model().unwrap();
    let mut context = Context::new(
        runtime.clone(),
        ContextOptions {
            microbatch_tokens: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let first = context
        .decode_batch(&[BatchToken::token(0, 0, 1), BatchToken::token(0, 1, 2)])
        .unwrap();
    let state = context.save_state().unwrap();
    let reserved = context.reserved_state_bytes();
    let checkpoint = context.checkpoint().unwrap();
    context.fork_sequence(0, 8).unwrap();
    let fork = context
        .decode_batch(&[BatchToken::token(0, 2, 3), BatchToken::token(8, 2, 3)])
        .unwrap();
    assert_eq!(fork[0].logits, fork[1].logits);
    context.truncate_sequence(0, 1).unwrap();
    assert_eq!(
        context.decode_batch(&[BatchToken::token(0, 1, 2)]).unwrap()[0].logits,
        first[1].logits
    );
    context.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(context.save_state().unwrap(), state);
    drop(checkpoint);
    let mut cancelled = 0;
    for stop in 1..64 {
        let mut ctx = Context::new(runtime.clone(), ContextOptions::default()).unwrap();
        ctx.restore_state(&state).unwrap();
        let mut seen = 0;
        let result = ctx.decode_batch_with_cancel(
            &[BatchToken::token(0, 2, 3), BatchToken::token(7, 0, 4)],
            || {
                seen += 1;
                seen == stop
            },
        );
        match result {
            Err(llama_model_runtime::Error::Cancelled) => {
                cancelled += 1;
                assert_eq!(ctx.save_state().unwrap(), state);
                assert_eq!(ctx.reserved_state_bytes(), reserved);
            }
            Ok(_) => break,
            other => panic!("{other:?}"),
        }
    }
    assert!(cancelled >= 8);
    let llama_model_runtime::Model::GptNeoX(native) = runtime.as_ref() else {
        panic!("GPTNEOX runtime");
    };
    let inputs: [u32; 3] = [1, 2, 3];
    let mut tokens = Context::new(runtime.clone(), ContextOptions::default()).unwrap();
    let expected = tokens
        .decode_batch(
            &inputs
                .iter()
                .enumerate()
                .map(|(p, &t)| BatchToken::token(0, p, t))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut embeddings = Context::new(runtime.clone(), ContextOptions::default()).unwrap();
    let actual = embeddings
        .decode_batch(
            &inputs
                .iter()
                .enumerate()
                .map(|(position, &t)| BatchToken {
                    sequence: 0,
                    position,
                    input: llama_model_runtime::Input::Embedding(
                        native
                            .weights()
                            .token_embedding
                            .row(t as usize)
                            .unwrap()
                            .to_vec(),
                    ),
                    output: llama_model_runtime::OutputSelection::Logits,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.logits, expected.logits);
    }
}
