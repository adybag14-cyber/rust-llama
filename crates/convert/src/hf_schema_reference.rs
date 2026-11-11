//! Independent Tokenizers 0.22.2 schema and postprocessor outcomes.
use super::*;
use crate::hf_backend::Backend;
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

#[test]
fn original_hf_schema_and_postprocessor_contracts() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-tokenizer/schema-reference.json"
    ));
    let mut failures = Vec::new();
    let (mut accepted, mut rejected, mut encodings) = (0usize, 0usize, 0usize);
    for case in array(field(&root, "cases")) {
        let name = text(field(case, "name")).unwrap();
        let input = parse(text(field(case, "tokenizer")).unwrap().as_bytes());
        let result = Backend::from_json(&input, &Limits::default());
        if text(field(case, "status")).unwrap() == "rejected" {
            rejected += 1;
            if result.is_ok() {
                failures.push(format!("{name}: Rust accepted a source rejection"));
            }
            continue;
        }
        accepted += 1;
        let mut backend = match result {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!("{name}: Rust rejected source admission: {error}"));
                continue;
            }
        };
        if let Some(requests) = object(case).unwrap().get("encodings") {
            for request in array(requests) {
                let text_input = text(field(request, "text")).unwrap();
                let special = matches!(field(request, "add_special"), Json::Bool(true));
                let result = backend.encode_with_special(text_input, special);
                encodings += 1;
                if let Some(ids) = object(request).unwrap().get("ids") {
                    let expected: Vec<_> = array(ids)
                        .iter()
                        .map(|v| integer(v).unwrap() as u32)
                        .collect();
                    match result {
                        Ok(actual) if actual == expected => {}
                        other => failures.push(format!("{name}: {text_input:?} special={special}: expected {expected:?}, actual {other:?}")),
                    }
                } else if result.is_ok() {
                    failures.push(format!("{name}: Rust accepted native postprocessor panic"));
                }
            }
        }
    }
    assert_eq!((accepted, rejected), (313, 767));
    eprintln!(
        "native schema cases: {}, postprocessor requests: {encodings}",
        accepted + rejected
    );
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn original_transformers_class_postprocessors() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-tokenizer/class-reference.json"
    ));
    for profile in [
        codec::SourceProfile::WindowsTorch211Numpy246,
        codec::SourceProfile::LinuxTorch211Numpy226,
    ] {
        let (mut accepted, mut rejected, mut count) = (0, 0, 0);
        for case in array(field(&root, "cases")) {
            let name = text(field(case, "name")).unwrap();
            let input = object(field(case, "files")).unwrap();
            let file = |name| input.get(name).map(|v| parse(text(v).unwrap().as_bytes()));
            let files = vocab::Files {
                tokenizer: file("tokenizer.json"),
                config: file("tokenizer_config.json"),
                added: file("added_tokens.json"),
            };
            let special = file("special_tokens_map.json");
            let result =
                hf_bpe::prepare(&files, special.as_ref(), &Limits::default(), profile, None);
            if text(field(case, "status")).unwrap() == "rejected" {
                assert!(result.is_err(), "source rejects {name}");
                rejected += 1;
                continue;
            }
            let (mut backend, _) = result.unwrap_or_else(|error| panic!("{name}: {error}"));
            accepted += 1;
            for request in array(field(case, "encodings")) {
                let input = text(field(request, "text")).unwrap();
                let special = matches!(field(request, "add_special"), Json::Bool(true));
                let result = backend.encode_with_special(input, special);
                if let Some(ids) = object(request).unwrap().get("ids") {
                    let ids: Vec<_> = array(ids)
                        .iter()
                        .map(|v| integer(v).unwrap() as u32)
                        .collect();
                    assert_eq!(
                        result.unwrap_or_else(|e| panic!("{name}: {e}")),
                        ids,
                        "{name} {input:?} {special}"
                    );
                } else {
                    assert!(result.is_err(), "source postprocessor rejects {name}");
                }
                count += 1;
            }
        }
        assert_eq!((accepted, rejected, count), (64, 12, 1024));
    }
}
