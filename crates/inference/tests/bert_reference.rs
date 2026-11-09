//! Pretrained BERT comparison against the external frozen C API oracle.
#[path = "support/native_profile.rs"]
mod native_profile;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::hash_reader;
use llama_inference::{Error, LoadOptions, Model, Pooling};
use native_profile::source_profile;
use std::{cell::Cell, fs::File, path::PathBuf};

fn native_reference() -> &'static str {
    source_profile(
        include_str!("../../../fixtures/inference/bert-bge-reference.json"),
        include_str!("../../../fixtures/inference/bert-bge-gnu-reference.json"),
    )
}
const MODEL_SHA256: &str = "f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999";

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("missing fixture field {key}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture array")
}
fn compare(actual: &[f32], expected: &Value, count: &mut usize, maximum: &mut f64) {
    let expected = array(expected);
    assert_eq!(actual.len(), expected.len());
    for (&a, e) in actual.iter().zip(expected) {
        let e = e.as_f64().expect("fixture float");
        let delta = (f64::from(a) - e).abs();
        let budget = 2e-4 + 2e-5 * e.abs();
        assert!(
            a.is_finite() && e.is_finite() && delta <= budget,
            "embedding value {a} differs from {e}: delta {delta}, budget {budget}"
        );
        *maximum = maximum.max(delta);
        *count += 1;
    }
}

#[test]
#[ignore = "requires hash-pinned external BERT model; set LLAMA_TEST_BERT_MODEL and run --release --ignored --nocapture"]
fn trained_bert_full_sequence_and_all_pooling_modes() {
    verify_model("LLAMA_TEST_BERT_MODEL", MODEL_SHA256);
}

#[test]
#[ignore = "requires hash-pinned big-endian BERT model; set LLAMA_TEST_BERT_BE_MODEL and run --release --ignored --nocapture"]
fn big_endian_bert_preserves_the_same_native_model_outputs() {
    // The frozen C reader rejects opposite-endian GGUF files. Compare this
    // equivalent encoding against the independent little-endian native corpus.
    verify_model(
        "LLAMA_TEST_BERT_BE_MODEL",
        "0e68255af91c707227f4840603bcbd0555054b921f757d79efdb2652ae8bb952",
    );
}

fn verify_model(variable: &str, model_sha256: &str) {
    let path = PathBuf::from(
        std::env::var_os(variable)
            .unwrap_or_else(|| panic!("set {variable} to the pinned BERT model")),
    );
    let (bytes, hash) =
        hash_reader(File::open(&path).expect("open pretrained BERT model")).unwrap();
    assert_eq!(bytes, 67_308_128);
    assert_eq!(hash, model_sha256, "wrong pretrained model");
    let reference = json::parse(native_reference()).unwrap();
    assert_eq!(
        field(&reference, "model_sha256").as_str().unwrap(),
        MODEL_SHA256
    );
    let model = Model::load(&path).unwrap();
    assert!(model.supports_embeddings());
    assert!(!model.supports_generation());
    assert_eq!(model.embedding_dimensions(), 384);
    assert_eq!(model.context_length(), 512);
    assert_eq!(model.default_embedding_pooling(), Some(Pooling::First));
    assert!(matches!(model.runtime_model(), Err(Error::Unsupported(_))));
    assert!(matches!(
        model.logits_for_tokens(&[101, 102]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        model.embed_tokens(&[101, 102], Pooling::None, &|| true),
        Err(Error::Cancelled)
    ));
    assert!(model.embed_tokens(&[], Pooling::None, &|| false).is_err());
    assert!(model.embed_tokens(&[-1], Pooling::None, &|| false).is_err());
    assert!(
        model
            .embed_tokens(&[30_522], Pooling::None, &|| false)
            .is_err()
    );
    let mut compared = 0usize;
    let mut maximum = 0.0f64;
    for case in array(field(&reference, "cases")) {
        let prompt = field(case, "prompt").as_str().unwrap();
        let expected_ids: Vec<i32> = array(field(case, "tokens"))
            .iter()
            .map(|v| i32::try_from(v.as_i64().unwrap()).unwrap())
            .collect();
        let tokens = model.tokenize(prompt, true).unwrap();
        assert_eq!(tokens, expected_ids, "{prompt:?}");
        let raw = model
            .embed_tokens(&tokens, Pooling::None, &|| false)
            .unwrap();
        assert!(raw.pooled.is_none());
        let expected_rows = array(field(case, "embeddings"));
        assert_eq!(raw.tokens.len(), expected_rows.len());
        for (a, e) in raw.tokens.iter().zip(expected_rows) {
            compare(a, e, &mut compared, &mut maximum);
        }
        for (name, pooling) in [
            ("mean", Pooling::Mean),
            ("first", Pooling::First),
            ("last", Pooling::Last),
        ] {
            let result = model.embed_tokens(&tokens, pooling, &|| false).unwrap();
            assert_eq!(
                result.tokens, raw.tokens,
                "pooling must not change token embeddings"
            );
            compare(
                result.pooled.as_ref().unwrap(),
                field(field(case, "pooling"), name),
                &mut compared,
                &mut maximum,
            );
        }
        let default = model.embed(prompt, None, &|| false).unwrap();
        assert_eq!(default.pooled.as_ref().unwrap(), &raw.tokens[0]);
        let calls = Cell::new(0usize);
        assert!(matches!(
            model.embed_tokens(&tokens, Pooling::None, &|| {
                calls.set(calls.get() + 1);
                calls.get() > tokens.len() + 2
            }),
            Err(Error::Cancelled)
        ));
    }
    let bounded = Model::load_with_options(
        &path,
        &LoadOptions {
            context_length: 2,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    assert_eq!(bounded.context_length(), 2);
    assert!(
        bounded
            .embed_tokens(&[101, 2000, 102], Pooling::None, &|| false)
            .is_err()
    );
    assert!(matches!(
        Model::load_with_options(
            &path,
            &LoadOptions {
                max_decoded_weight_bytes: 1024,
                ..LoadOptions::default()
            }
        ),
        Err(Error::Invalid(_))
    ));
    assert_eq!(compared, 13_056);
    eprintln!(
        "BERT conformance ({variable}): cases=3 token_vectors=25 pooled_vectors=9 compared_values={compared} max_abs={maximum:.9e}"
    );
}
