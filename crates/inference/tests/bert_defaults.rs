//! Ordinary loader regression with a tiny deterministic model and original C API evidence.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits};
use llama_inference::{Model, Pooling};
use std::{io::Cursor, path::PathBuf};

const MODEL: &[u8] = include_bytes!("../../../fixtures/inference/tiny-bert-missing-pooling.gguf");
const REFERENCE: &str = include_str!("../../../fixtures/inference/bert-defaults-reference.json");
const MODEL_SHA256: &str = "1eeff594b8c9248833dafcdee68f7fbbe4d614261f7ff0f42258b40811b974f3";

fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("missing native fixture field {key}"))
}
fn array(v: &Value) -> &[Value] {
    v.as_array().expect("native fixture array")
}
fn compare(actual: &[f32], expected: &Value, compared: &mut usize, maximum: &mut f64) {
    assert_eq!(actual.len(), array(expected).len());
    for (&actual, expected) in actual.iter().zip(array(expected)) {
        let expected = expected.as_f64().unwrap();
        let error = (f64::from(actual) - expected).abs();
        let bound = 2e-6 + 2e-6 * expected.abs();
        assert!(
            actual.is_finite() && error <= bound,
            "{actual} != {expected}; error {error}"
        );
        *maximum = maximum.max(error);
        *compared += 1;
    }
}

#[test]
fn missing_pooling_metadata_keeps_original_unpooled_default() {
    assert_eq!(sha256(MODEL), MODEL_SHA256);
    let gguf = Gguf::read(Cursor::new(MODEL), Limits::default()).unwrap();
    assert!(!gguf.metadata.contains_key("bert.pooling_type"));
    let reference = json::parse(REFERENCE).unwrap();
    assert_eq!(
        field(&reference, "model_sha256").as_str(),
        Some(MODEL_SHA256)
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/inference/tiny-bert-missing-pooling.gguf");
    let model = Model::load(path).unwrap();
    assert_eq!(model.default_embedding_pooling(), Some(Pooling::None));
    assert!(model.supports_embeddings());
    assert!(!model.supports_generation());
    assert_eq!(model.context_length(), 8);
    assert_eq!(model.embedding_dimensions(), 4);
    let mut compared = 0;
    let mut maximum = 0.0_f64;
    for case in array(field(&reference, "cases")) {
        let prompt = field(case, "prompt").as_str().unwrap();
        let ids: Vec<i32> = array(field(case, "tokens"))
            .iter()
            .map(|id| i32::try_from(id.as_i64().unwrap()).unwrap())
            .collect();
        assert_eq!(model.tokenize(prompt, true).unwrap(), ids);
        let contexts = array(field(case, "contexts"));
        assert_eq!(contexts.len(), 3);
        let native_default = &contexts[0];
        let native_none = &contexts[1];
        let native_mean = &contexts[2];
        assert_eq!(
            field(native_default, "default_parameter_pooling").as_i64(),
            Some(-1)
        );
        assert_eq!(
            field(native_default, "requested_pooling").as_i64(),
            Some(-1)
        );
        assert_eq!(field(native_default, "resolved_pooling").as_i64(), Some(0));
        assert_eq!(field(native_none, "requested_pooling").as_i64(), Some(0));
        assert_eq!(field(native_none, "resolved_pooling").as_i64(), Some(0));
        assert_eq!(field(native_mean, "resolved_pooling").as_i64(), Some(1));
        assert_eq!(field(native_default, "pooled"), &Value::Null);
        assert_eq!(field(native_none, "pooled"), &Value::Null);
        assert_eq!(
            field(native_default, "embeddings"),
            field(native_none, "embeddings")
        );

        let default = model.embed(prompt, None, &|| false).unwrap();
        let explicit_none = model.embed(prompt, Some(Pooling::None), &|| false).unwrap();
        assert_eq!(default, explicit_none);
        assert!(default.pooled.is_none());
        let expected = array(field(native_default, "embeddings"));
        assert_eq!(default.tokens.len(), expected.len());
        for (actual, expected) in default.tokens.iter().zip(expected) {
            compare(actual, expected, &mut compared, &mut maximum);
        }
        let mean = model.embed(prompt, Some(Pooling::Mean), &|| false).unwrap();
        assert_eq!(mean.tokens, default.tokens);
        compare(
            mean.pooled.as_ref().unwrap(),
            field(native_mean, "pooled"),
            &mut compared,
            &mut maximum,
        );
    }
    assert_eq!(compared, 56);
    eprintln!(
        "native missing-pooling default: 3 prompts, 11 token embeddings, 3 explicit mean vectors, {compared} floats, max_abs={maximum:.9e}"
    );
}
