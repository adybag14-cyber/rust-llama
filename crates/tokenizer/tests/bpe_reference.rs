//! Independent frozen C API results; shared framing also covers the remaining families.
#[path = "support/native_corpus.rs"]
mod native_corpus;
use llama_behavior::json;
use llama_gguf::{Gguf, Limits, Value as MetadataValue};
use llama_tokenizer::{Error, Tokenizer};
use native_corpus::{array, compare, field};
use std::{io::Cursor, path::PathBuf};
const SYNTHETIC: &[u8] = include_bytes!("../../../fixtures/tokenizer/bpe/synthetic-bpe.gguf");
const GPT2: &[u8] = include_bytes!("../../../fixtures/tokenizer/bpe/ggml-vocab-gpt-2.gguf");
#[test]
fn all_64_profiles_99_aliases_and_source_session_options() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/synthetic-reference.json"
    ))
    .unwrap();
    assert_eq!(array(field(&reference, "variants")).len(), 111);
    assert_eq!(compare(&reference, SYNTHETIC), (18_381, 23_754));
    eprintln!(
        "BPE: 64 profiles / 99 metadata aliases plus 12 configuration/session variants; 23,754 native cases and 18,381 exact vocabulary records"
    );
}
#[test]
fn complete_original_gpt2_vocabulary_and_real_merge_corpus() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/vocabulary-reference.json"
    ))
    .unwrap();
    let model = &array(field(&reference, "models"))[0];
    assert_eq!(field(model, "name").as_str(), Some("gpt-2"));
    assert_eq!(compare(model, GPT2), (50_257, 364));
    eprintln!(
        "frozen GPT-2: all 50,257 vocabulary records, 364 exact token/piece/detokenize cases"
    );
}
#[test]
fn malformed_utf8_and_byte_truncation_match_all_tokenizer_families() {
    let bpe = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/raw-bpe-reference.json"
    ))
    .unwrap();
    let spm = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/raw-spm-reference.json"
    ))
    .unwrap();
    let wpm = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/raw-wpm-reference.json"
    ))
    .unwrap();
    let (_, a) = compare(&bpe, SYNTHETIC);
    let (_, b) = compare(
        &spm,
        include_bytes!("../../../fixtures/inference/ggml-vocab-llama-spm.gguf"),
    );
    let (_, c) = compare(
        &wpm,
        include_bytes!("../../../fixtures/tokenizer/ggml-vocab-bert-bge.gguf"),
    );
    assert_eq!((a, b, c), (17_514, 2_502, 2_502));
    eprintln!(
        "raw-byte input: 22,518 native cases, including all 256 bytes, malformed/surrogate UTF-8, special fragments and 40-byte DRY cut points"
    );
}
#[test]
fn unsupported_profiles_and_source_rejected_settings_fail_explicitly() {
    let mut metadata = Gguf::read(Cursor::new(SYNTHETIC), Limits::default())
        .unwrap()
        .metadata;
    metadata.insert(
        "tokenizer.ggml.pre".into(),
        MetadataValue::String("unported_future_profile".into()),
    );
    assert!(matches!(
        Tokenizer::from_metadata(&metadata),
        Err(Error::Unsupported(_))
    ));
    metadata.insert(
        "tokenizer.ggml.pre".into(),
        MetadataValue::String("gpt-2".into()),
    );
    metadata.insert(
        "tokenizer.ggml.metaspace.prepend_scheme".into(),
        MetadataValue::String("always".into()),
    );
    assert!(matches!(
        Tokenizer::from_metadata(&metadata),
        Err(Error::Unsupported(_))
    ));
    metadata.remove("tokenizer.ggml.metaspace.prepend_scheme");
    metadata.remove("tokenizer.ggml.merges");
    assert!(matches!(
        Tokenizer::from_metadata(&metadata),
        Err(Error::Invalid(_))
    ));
    metadata.insert(
        "tokenizer.ggml.pre".into(),
        MetadataValue::String("kimi-k2".into()),
    );
    assert!(Tokenizer::from_metadata(&metadata).is_ok());
}
#[test]
#[ignore = "full 99-profile original GPT-2 merge corpus; run explicitly for tokenizer changes"]
fn original_gpt2_merges_across_all_99_metadata_aliases() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/gpt2-profiles-reference.json"
    ))
    .unwrap();
    assert_eq!(compare(&reference, GPT2), (0, 18_018));
}
#[test]
#[ignore = "requires LLAMA_TEST_BPE_VOCAB_DIR containing the original frozen source models directory"]
fn every_original_bpe_vocabulary_and_merge_corpus() {
    let path = PathBuf::from(
        std::env::var_os("LLAMA_TEST_BPE_VOCAB_DIR")
            .expect("set LLAMA_TEST_BPE_VOCAB_DIR to the frozen source models directory"),
    );
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/vocabulary-reference.json"
    ))
    .unwrap();
    let mut vocab = 0;
    let mut cases = 0;
    for model in array(field(&reference, "models")) {
        let input = std::fs::read(path.join(field(model, "file").as_str().unwrap())).unwrap();
        let (v, c) = compare(model, &input);
        vocab += v;
        cases += c;
    }
    assert_eq!(array(field(&reference, "models")).len(), 14);
    assert_eq!(cases, 5_096);
    eprintln!(
        "all 14 original BPE vocabularies: {vocab} exact vocabulary records and {cases} native token/piece/detokenization cases"
    );
}

#[test]
fn ambiguous_special_inference_and_metadata_override_precedence() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/special-order-reference.json"
    ))
    .unwrap();
    let model = include_bytes!("../../../fixtures/tokenizer/bpe/special-order.gguf");
    assert_eq!(compare(&reference, model), (5_680, 40));
    eprintln!(
        "native EOT/FIM inference: all 51 candidate spellings, 10 variants, 5680 attribute/piece/EOG records, metadata sentinel and deprecated-key precedence"
    );
}
