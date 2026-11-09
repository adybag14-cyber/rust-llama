//! Independent frozen C API results. Expected hashes frame exact IDs, attributes,
//! pieces and detokenization bytes; no Rust result was used to create them.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits, Value as MetadataValue};
use llama_tokenizer::{Error, Tokenizer};
use std::{io::Cursor, path::PathBuf};

const SYNTHETIC: &[u8] = include_bytes!("../../../fixtures/tokenizer/bpe/synthetic-bpe.gguf");
const GPT2: &[u8] = include_bytes!("../../../fixtures/tokenizer/bpe/ggml-vocab-gpt-2.gguf");
fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap()
}
fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
fn hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u8::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap())
        .collect()
}
fn blob(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend(u32::try_from(bytes.len()).unwrap().to_le_bytes());
    out.extend(bytes);
}
fn expected_case<'a>(reference: &'a Value, variant: &Value, index: usize) -> &'a str {
    let normal = array(field(variant, "case_hash_ids"))[index]
        .as_u64()
        .unwrap();
    let id = if cfg!(windows) {
        normal
    } else {
        field(variant, "linux_case_overrides")
            .get(&index.to_string())
            .and_then(Value::as_u64)
            .unwrap_or(normal)
    };
    array(field(reference, "hashes"))[id as usize]
        .as_str()
        .unwrap()
}
fn compare(reference: &Value, model: &[u8]) -> (usize, usize) {
    assert_eq!(field(reference, "schema").as_u64(), Some(1));
    assert_eq!(
        sha256(model),
        field(reference, "model_sha256").as_str().unwrap()
    );
    assert_eq!(
        model.len() as u64,
        field(reference, "model_bytes").as_u64().unwrap()
    );
    let parsed = Gguf::read(Cursor::new(model), Limits::default()).unwrap();
    let mut case_count = 0;
    let mut vocabulary_count = 0;
    for variant in array(field(reference, "variants")) {
        let name = field(variant, "name").as_str().unwrap();
        let mut metadata = parsed.metadata.clone();
        for (key, value) in field(variant, "overrides").as_object().unwrap() {
            let value = if let Some(value) = value.as_bool() {
                MetadataValue::Bool(value)
            } else if let Some(value) = value.as_str() {
                MetadataValue::String(value.into())
            } else {
                MetadataValue::I64(value.as_i64().unwrap())
            };
            metadata.insert(key.clone(), value);
        }
        let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
        let expected = field(variant, "metadata");
        for (key, value) in [
            ("vocab_type", tokenizer.vocab_type() as i32),
            ("n_vocab", tokenizer.vocabulary_size() as i32),
            ("bos", tokenizer.bos_id),
            ("eos", tokenizer.eos_id),
            ("sep", tokenizer.sep_id),
            ("pad", tokenizer.pad_id),
            ("mask", tokenizer.mask_id),
            ("nl", tokenizer.newline_id()),
        ] {
            assert_eq!(
                value as i64,
                (if cfg!(windows) {
                    None
                } else {
                    variant
                        .get("linux_metadata_overrides")
                        .and_then(|v| v.get(key))
                })
                .unwrap_or_else(|| field(expected, key))
                .as_i64()
                .unwrap(),
                "{name} metadata {key}"
            );
        }
        for (key, value) in [
            ("eot", tokenizer.eot_id),
            ("fim_pre", tokenizer.fim_pre_id),
            ("fim_suf", tokenizer.fim_suf_id),
            ("fim_mid", tokenizer.fim_mid_id),
            ("fim_pad", tokenizer.fim_pad_id),
            ("fim_rep", tokenizer.fim_rep_id),
            ("fim_sep", tokenizer.fim_sep_id),
        ] {
            if let Some(normal) = expected.get(key) {
                let native = (if cfg!(windows) {
                    None
                } else {
                    variant
                        .get("linux_metadata_overrides")
                        .and_then(|v| v.get(key))
                })
                .unwrap_or(normal);
                assert_eq!(
                    Some(value as i64),
                    native.as_i64(),
                    "{name} native getter {key}"
                );
            }
        }
        for (key, value) in [
            ("add_bos", tokenizer.add_bos),
            ("add_eos", tokenizer.add_eos),
            ("add_sep", tokenizer.add_sep),
        ] {
            assert_eq!(
                Some(value),
                field(expected, key).as_bool(),
                "{name} metadata {key}"
            );
        }
        if let Some(hashes) = variant.get("vocabulary_chunk_hashes") {
            let mut bytes = Vec::new();
            let mut chunk = 0;
            for id in 0..tokenizer.vocabulary_size() {
                let (text, score, _) = tokenizer.token_info(id as i32).unwrap();
                blob(&mut bytes, text.as_bytes());
                bytes.extend(score.to_bits().to_le_bytes());
                bytes.extend(tokenizer.token_attributes(id as i32).unwrap().to_le_bytes());
                blob(
                    &mut bytes,
                    &tokenizer.token_piece(id as i32, false).unwrap(),
                );
                blob(&mut bytes, &tokenizer.token_piece(id as i32, true).unwrap());
                bytes.push(u8::from(tokenizer.is_end_of_generation(id as i32)));
                vocabulary_count += 1;
                if (id + 1) % 256 == 0 || id + 1 == tokenizer.vocabulary_size() {
                    assert_eq!(
                        sha256(&bytes),
                        (if cfg!(windows) {
                            None
                        } else {
                            variant
                                .get("linux_vocabulary_chunk_overrides")
                                .and_then(|v| v.get(&chunk.to_string()))
                        })
                        .unwrap_or(&array(hashes)[chunk])
                        .as_str()
                        .unwrap(),
                        "{name} vocabulary chunk {chunk}"
                    );
                    bytes.clear();
                    chunk += 1;
                }
            }
            assert_eq!(chunk, array(hashes).len());
        }
        let cases = array(field(reference, "cases"));
        assert_eq!(cases.len(), array(field(variant, "case_hash_ids")).len());
        for (index, case) in cases.iter().enumerate() {
            let input = if let Some(raw) = case.get("raw_hex") {
                hex(raw.as_str().unwrap())
            } else {
                field(case, "prompt").as_str().unwrap().as_bytes().to_vec()
            };
            let add = field(case, "add_special").as_bool().unwrap();
            let parse = field(case, "parse_special").as_bool().unwrap();
            let tokens = tokenizer.tokenize_bytes(&input, add, parse).unwrap();
            if let Ok(text) = std::str::from_utf8(&input) {
                assert_eq!(tokenizer.tokenize(text, add, parse).unwrap(), tokens);
            }
            let mut bytes = u32::try_from(tokens.len()).unwrap().to_le_bytes().to_vec();
            for &token in &tokens {
                bytes.extend(token.to_le_bytes());
                bytes.extend(tokenizer.token_attributes(token).unwrap().to_le_bytes());
                blob(&mut bytes, &tokenizer.token_piece(token, false).unwrap());
                blob(&mut bytes, &tokenizer.token_piece(token, true).unwrap());
            }
            for remove in [false, true] {
                for unparse in [false, true] {
                    blob(
                        &mut bytes,
                        &tokenizer
                            .detokenize_bytes(&tokens, remove, unparse)
                            .unwrap(),
                    );
                }
            }
            assert_eq!(
                sha256(&bytes),
                expected_case(reference, variant, index),
                "{name} native case {} input={input:?}, tokens={tokens:?}",
                field(case, "name").as_str().unwrap()
            );
            case_count += 1;
        }
    }
    (vocabulary_count, case_count)
}
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
