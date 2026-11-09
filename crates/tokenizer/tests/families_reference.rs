//! The expected outputs are compact hashes and records from the frozen native
//! implementation. Input GGUFs contain vocabulary metadata and no model tensors.
#[path = "support/native_corpus.rs"]
mod native_corpus;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits};
use llama_tokenizer::{Error, Tokenizer, VocabType};
use native_corpus::{array, blob, compare, compare_with_cases, field, hex};
use std::{io::Cursor, path::PathBuf};

fn reference(text: &str) -> Value {
    json::parse_with_limits(
        text,
        json::Limits {
            max_bytes: 8 * 1024 * 1024,
            max_nodes: 1_000_000,
            max_depth: 32,
        },
    )
    .unwrap()
}
#[test]
fn unigram_rwkv_plamo_and_test_match_frozen_native_sessions() {
    let native = reference(include_str!(
        "../../../fixtures/tokenizer/families/synthetic-reference.json"
    ));
    let models: [(&str, &[u8]); 6] = [
        (
            "rwkv-synthetic.gguf",
            include_bytes!("../../../fixtures/tokenizer/families/rwkv-synthetic.gguf"),
        ),
        (
            "ugm-synthetic.gguf",
            include_bytes!("../../../fixtures/tokenizer/families/ugm-synthetic.gguf"),
        ),
        (
            "ugm-signed-charsmap.gguf",
            include_bytes!("../../../fixtures/tokenizer/families/ugm-signed-charsmap.gguf"),
        ),
        (
            "ugm-no-charsmap.gguf",
            include_bytes!("../../../fixtures/tokenizer/families/ugm-no-charsmap.gguf"),
        ),
        (
            "plamo-synthetic.gguf",
            include_bytes!("../../../fixtures/tokenizer/families/plamo-synthetic.gguf"),
        ),
        (
            "test-synthetic.gguf",
            include_bytes!("../../../fixtures/tokenizer/families/test-synthetic.gguf"),
        ),
    ];
    let mut counts = (0, 0);
    assert_eq!(array(field(&native, "models")).len(), models.len());
    for (native_model, (name, bytes)) in array(field(&native, "models")).iter().zip(models) {
        assert_eq!(field(native_model, "file").as_str(), Some(name));
        let count = compare_with_cases(native_model, bytes, field(&native, "cases"));
        counts.0 += count.0;
        counts.1 += count.1;
    }
    assert_eq!(counts, (3_507, 91_476));
    eprintln!(
        "remaining families: 91,476 native token/piece/detokenization cases and 3,507 vocabulary records"
    );
}
#[test]
#[ignore = "requires LLAMA_TEST_TOKENIZER_VOCAB_DIR (or LLAMA_TEST_BPE_VOCAB_DIR) containing the frozen source models directory"]
fn all_original_unigram_and_remaining_spm_vocabularies() {
    let directory = PathBuf::from(
        std::env::var_os("LLAMA_TEST_TOKENIZER_VOCAB_DIR")
            .or_else(|| std::env::var_os("LLAMA_TEST_BPE_VOCAB_DIR"))
            .expect("set LLAMA_TEST_TOKENIZER_VOCAB_DIR to the frozen source models directory"),
    );
    let native = reference(include_str!(
        "../../../fixtures/tokenizer/families/original-vocabulary-reference.json"
    ));
    let mut counts = (0, 0);
    for model in array(field(&native, "models")) {
        let bytes = std::fs::read(directory.join(field(model, "file").as_str().unwrap())).unwrap();
        let count = compare(model, &bytes);
        counts.0 += count.0;
        counts.1 += count.1;
    }
    assert_eq!(counts, (346_112, 10_412));
    eprintln!(
        "original Nomic Unigram, Baichuan and Phi-3 SPM: 346,112 vocabulary records and 10,412 exact native cases"
    );
}
fn take_blob<'a>(bytes: &'a [u8], position: &mut usize) -> &'a [u8] {
    let length = u32::from_le_bytes(bytes[*position..*position + 4].try_into().unwrap()) as usize;
    *position += 4;
    let start = *position;
    *position += length;
    &bytes[start..*position]
}
#[test]
fn typed_metadata_sentinels_and_native_failure_cases_are_bounded() {
    let pack = include_bytes!("../../../fixtures/tokenizer/families/metadata-models.bin");
    let native = reference(include_str!(
        "../../../fixtures/tokenizer/families/metadata-reference.json"
    ));
    assert_eq!(&pack[..8], b"TMET0001");
    assert_eq!(
        pack.len() as u64,
        field(&native, "model_pack_bytes").as_u64().unwrap()
    );
    assert_eq!(
        sha256(pack),
        field(&native, "model_pack_sha256").as_str().unwrap()
    );
    let cases = array(field(&native, "cases"));
    assert_eq!(
        u32::from_le_bytes(pack[8..12].try_into().unwrap()) as usize,
        cases.len()
    );
    assert_eq!(cases.len(), 176);
    let mut position = 12;
    let mut loaded = 0;
    let mut rejected = 0;
    let mut checked_tokens = 0;
    let mut vocabulary_records = 0;
    for case in cases {
        let name = field(case, "name").as_str().unwrap();
        assert_eq!(take_blob(pack, &mut position), name.as_bytes());
        let model = take_blob(pack, &mut position);
        assert_eq!(
            model.len() as u64,
            field(case, "model_bytes").as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            sha256(model),
            field(case, "model_sha256").as_str().unwrap(),
            "{name}"
        );
        let metadata = Gguf::read(Cursor::new(model), Limits::default())
            .unwrap()
            .metadata;
        let result = Tokenizer::from_metadata(&metadata);
        match field(case, "outcome").as_str().unwrap() {
            "load-error" | "native-assert" => {
                assert!(result.is_err(), "{name} unexpectedly loaded");
                rejected += 1;
            }
            "nonprogress" => {
                let tokenizer = result.unwrap();
                assert!(matches!(
                    tokenizer.tokenize_bytes(
                        &hex(field(case, "input_hex").as_str().unwrap()),
                        false,
                        false
                    ),
                    Err(Error::Invalid(_))
                ));
                rejected += 1;
            }
            "loaded" => {
                loaded += 1;
                let tokenizer = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    tokenizer.vocab_type() as i64,
                    field(case, "type").as_i64().unwrap(),
                    "{name}"
                );
                assert_eq!(
                    tokenizer.vocabulary_size() as u64,
                    field(case, "n_vocab").as_u64().unwrap(),
                    "{name}"
                );
                assert_eq!(
                    tokenizer.newline_id() as i64,
                    field(case, "newline").as_i64().unwrap(),
                    "{name}"
                );
                let suppressed: Vec<i32> = array(field(case, "suppressed_tokens"))
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect();
                assert_eq!(
                    tokenizer.suppressed_tokens(),
                    suppressed,
                    "{name} suppress metadata"
                );
                assert_eq!(field(case, "empty_detokenize").as_i64(), Some(0));
                assert!(
                    tokenizer
                        .detokenize_bytes(&[], true, true)
                        .unwrap()
                        .is_empty()
                );
                let records = field(case, "vocabulary_records").as_u64().unwrap() as usize;
                let mut data = Vec::new();
                if tokenizer.vocab_type() == VocabType::None {
                    assert_eq!(records, 0);
                    assert!(tokenizer.token_info(0).is_none());
                    assert!(tokenizer.token_attributes(0).is_none());
                    assert!(matches!(
                        tokenizer.tokenize("", false, false),
                        Err(Error::Unsupported(_))
                    ));
                    assert!(matches!(
                        tokenizer.token_piece(0, true),
                        Err(Error::Unsupported(_))
                    ));
                    assert!(
                        tokenizer
                            .detokenize_bytes(&[0, i32::MAX, -1], false, true)
                            .unwrap()
                            .is_empty()
                    );
                } else {
                    assert_eq!(records, tokenizer.vocabulary_size());
                    for id in 0..records {
                        let id = id as i32;
                        let (text, score, _) = tokenizer.token_info(id).unwrap();
                        blob(&mut data, text.as_bytes());
                        data.extend(score.to_bits().to_le_bytes());
                        data.extend(tokenizer.token_attributes(id).unwrap().to_le_bytes());
                        blob(&mut data, &tokenizer.token_piece(id, false).unwrap());
                        blob(&mut data, &tokenizer.token_piece(id, true).unwrap());
                        data.push(u8::from(tokenizer.is_end_of_generation(id)));
                    }
                }
                assert_eq!(
                    sha256(&data),
                    field(case, "vocabulary_sha256").as_str().unwrap(),
                    "{name} vocabulary"
                );
                vocabulary_records += records;
                for row in array(field(case, "cases")) {
                    let input = hex(field(row, "raw_hex").as_str().unwrap());
                    let expected: Vec<_> = array(field(row, "tokens"))
                        .iter()
                        .map(|v| v.as_i64().unwrap() as i32)
                        .collect();
                    assert_eq!(
                        tokenizer
                            .tokenize_bytes(
                                &input,
                                field(row, "add_special").as_bool().unwrap(),
                                field(row, "parse_special").as_bool().unwrap()
                            )
                            .unwrap(),
                        expected,
                        "{name}: {input:?}"
                    );
                    checked_tokens += 1;
                }
            }
            other => panic!("unrecognized native outcome {other}"),
        }
    }
    assert_eq!(position, pack.len());
    assert_eq!(
        (loaded, rejected, vocabulary_records, checked_tokens),
        (70, 106, 23_903, 480)
    );
    eprintln!(
        "176 independent metadata cases: 70 native loads, 103 native errors, 2 native assertions and 1 native non-progress loop; all Rust failure paths are checked"
    );
}
