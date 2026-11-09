//! Unmodified frozen unicode.cpp outputs, compiled independently on both hosts.
use super::*;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap()
}
fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
#[test]
fn all_frozen_regex_pipelines_and_expressions_match_native_platform() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/regex-reference.json"
    ))
    .unwrap();
    let cases = array(field(&reference, "cases"));
    assert_eq!(cases.len(), 1_442);
    assert_eq!(array(field(&reference, "groups")).len(), 88);
    for group in array(field(&reference, "groups")) {
        let patterns: Vec<_> = array(field(group, "regex"))
            .iter()
            .map(|expression| Pattern::compile(expression.as_str().unwrap()).unwrap())
            .collect();
        let encoded = field(group, "byte_encode").as_bool().unwrap();
        let expected = array(field(
            group,
            if cfg!(windows) {
                "windows_chunk_hashes"
            } else {
                "linux_chunk_hashes"
            },
        ));
        let mut bytes = Vec::new();
        let mut chunk = 0;
        for (index, case) in cases.iter().enumerate() {
            let pieces = split(
                field(case, "text").as_str().unwrap().as_bytes(),
                &patterns,
                encoded,
            )
            .unwrap();
            bytes.extend(u32::try_from(pieces.len()).unwrap().to_le_bytes());
            for piece in pieces {
                bytes.extend(u32::try_from(piece.len()).unwrap().to_le_bytes());
                bytes.extend(piece);
            }
            if (index + 1) % 64 == 0 || index + 1 == cases.len() {
                assert_eq!(
                    sha256(&bytes),
                    expected[chunk].as_str().unwrap(),
                    "native regex {:?}, chunk {chunk}",
                    field(group, "names")
                );
                bytes.clear();
                chunk += 1;
            }
        }
        assert_eq!(chunk, expected.len());
    }
    eprintln!(
        "native regex: 126,896 exact splits; 40 pipelines, 48 expressions, 1,442 focused/random inputs"
    );
}
#[test]
fn exhaustive_source_platform_wide_regex_classes() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/bpe/regex-reference.json"
    ))
    .unwrap();
    let platform = if cfg!(windows) { "windows" } else { "linux" };
    let expected = field(field(&reference, "wide_classes"), platform);
    let width = if cfg!(windows) { 2 } else { 4 };
    let count = if cfg!(windows) { 0x10000 } else { 0x110000 };
    let mut bytes = vec![b'W', b'C', b'L', b'S', 1, width, 0, 0];
    for cp in 0..count {
        bytes.push(
            u8::from(class(cp, 1, true))
                | (u8::from(class(cp, 2, true)) * 2)
                | (u8::from(class(cp, 4, true)) * 4),
        );
    }
    assert_eq!(
        bytes.len() as u64,
        field(expected, "bytes").as_u64().unwrap()
    );
    assert_eq!(sha256(&bytes), field(expected, "sha256").as_str().unwrap());
}
