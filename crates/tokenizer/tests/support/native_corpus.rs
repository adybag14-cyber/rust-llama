//! Shared framing of independent native token/piece/vocabulary captures.
use llama_behavior::json::Value;
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits, Value as MetadataValue};
use llama_tokenizer::Tokenizer;
use std::io::Cursor;
pub fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap()
}
pub fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
pub fn hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u8::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap())
        .collect()
}
pub fn blob(out: &mut Vec<u8>, bytes: &[u8]) {
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
pub fn compare_with_cases(reference: &Value, model: &[u8], cases: &Value) -> (usize, usize) {
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
                MetadataValue::U32(value.as_i64().unwrap() as u32)
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
        let cases = array(cases);
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
pub fn compare(reference: &Value, model: &[u8]) -> (usize, usize) {
    compare_with_cases(reference, model, field(reference, "cases"))
}
