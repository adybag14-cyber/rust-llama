//! Independently captured Falcon tokenizer primitives and complete pipelines.
use crate::Limits;
use crate::util::{JsonBudget, integer, object, text};
use llama_checkpoint::Value as Json;

fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    object(value).unwrap().get(key).unwrap()
}
fn array(value: &Json) -> &[Json] {
    value.as_array().unwrap()
}
fn parse(raw: &[u8]) -> Json {
    JsonBudget::new(llama_checkpoint::Limits::default())
        .parse(raw)
        .unwrap()
}
fn ids(value: &Json) -> Vec<u32> {
    array(value)
        .iter()
        .map(|v| integer(v).unwrap() as u32)
        .collect()
}

#[test]
fn native_falcon_punctuation_scalars_and_tokenizer_pipeline() {
    let root = parse(include_bytes!(
        "../tests/fixtures/falcon-conversion/tokenizer-reference.json"
    ));
    let membership = include_bytes!("../tests/fixtures/falcon-conversion/punctuation.bin");
    assert_eq!(membership.len(), 0x110000);
    assert_eq!(
        llama_conformance::digest::sha256(membership),
        text(field(field(&root, "membership"), "sha256")).unwrap()
    );
    let mut scalars = 0;
    for (cp, &expected) in membership.iter().enumerate() {
        if let Some(ch) = char::from_u32(cp as u32) {
            assert_eq!(
                crate::hf_backend::falcon_punctuation_member(ch),
                expected == 1,
                "U+{cp:04X}"
            );
            scalars += 1;
        }
    }
    assert_eq!(scalars, 1_112_064);
    assert_eq!(crate::hf_schema::FALCON_TRIPLES_PATTERN, "[0-9][0-9][0-9]");
    let base = include_str!("../tests/fixtures/falcon-conversion/tokenizer-base.json")
        .trim()
        .strip_prefix('{')
        .unwrap();
    let mut requests = 0;
    for profile in array(field(&root, "profiles")) {
        let name = text(field(profile, "name")).unwrap();
        let stage = text(field(profile, "pre_tokenizer")).unwrap();
        let tokenizer = parse(format!("{{\"pre_tokenizer\":{stage},{base}").as_bytes());
        let mut backend =
            crate::hf_backend::Backend::from_json(&tokenizer, &Limits::default()).unwrap();
        for case in array(field(profile, "cases")) {
            let input = text(field(case, "text")).unwrap();
            let pieces: Vec<_> = array(field(case, "pieces"))
                .iter()
                .map(|v| text(v).unwrap())
                .collect();
            assert_eq!(backend.pieces(input).unwrap(), pieces, "{name}: {input:?}");
            let actual = backend.encode(input).unwrap();
            assert_eq!(actual, ids(field(case, "ids")), "{name}: {input:?}");
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(case, "decoded")).unwrap(),
                "{name}: {input:?}"
            );
            requests += 1;
        }
    }
    assert_eq!(requests, 8745);
}
