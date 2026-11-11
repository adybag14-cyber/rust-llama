//! Transformers GPTNeoXTokenizerFast captures, independent of product code.
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
fn native_gptneox_fast_class_semantics() {
    let root = parse(include_bytes!(
        "../tests/fixtures/gpt-conversion/neox-class-reference.json"
    ));
    let (mut accepted, mut rejected, mut encodings) = (0, 0, 0);
    for case in array(field(&root, "cases")) {
        let name = text(field(case, "name")).unwrap();
        let mut budget = JsonBudget::new(llama_checkpoint::Limits::default());
        let config = crate::config::Config::new(
            field(case, "model").clone(),
            "pinned-gpt-neox-model",
            &mut budget,
            &Limits::default(),
        )
        .unwrap();
        let sidecar = field(case, "config");
        let files = crate::vocab::Files {
            tokenizer: Some(field(case, "tokenizer").clone()),
            config: if matches!(sidecar, Json::Null) {
                None
            } else {
                Some(sidecar.clone())
            },
            added: None,
        };
        let result = crate::hf_bpe::prepare(
            &files,
            None,
            &Limits::default(),
            crate::codec::SourceProfile::current(),
            Some(&config),
        );
        if text(field(case, "status")).unwrap() == "rejected" {
            assert!(result.is_err(), "{name}: native class constructor rejected");
            rejected += 1;
            continue;
        }
        let (mut backend, _) = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        for request in array(field(case, "requests")) {
            let input = text(field(request, "text")).unwrap();
            let special = matches!(field(request, "special"), Json::Bool(true));
            let result = backend.encode_with_special(input, special);
            if text(field(request, "status")).unwrap() == "rejected" {
                assert!(
                    result.is_err(),
                    "{name}: {input:?}: native encoding rejected"
                );
                continue;
            }
            let actual = result.unwrap_or_else(|e| panic!("{name}: {input:?}: {e}"));
            assert_eq!(
                actual,
                ids(field(request, "ids")),
                "{name}: {input:?}, special={special}"
            );
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(request, "decoded")).unwrap(),
                "{name}: {input:?}"
            );
            encodings += 1;
        }
        accepted += 1;
    }
    assert_eq!((accepted, rejected, encodings), (40, 6, 960));
    eprintln!("GPTNeoXTokenizerFast: 46 class cases, 960 native encoding/decoder requests");
}
