//! Native attribute and tokenization observations for raw general.name bytes.
use crate::Tokenizer;
use llama_behavior::json::{self, Value as Json};
use llama_conformance::digest::sha256;
use llama_gguf::{GgufSet, Value};
use std::path::Path;

fn field<'a>(value: &'a Json, name: &str) -> &'a Json {
    value.get(name).unwrap()
}

fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2));
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("invalid hex fixture"),
            };
            digit(pair[0]) * 16 + digit(pair[1])
        })
        .collect()
}

fn check(tokenizer: &Tokenizer, case: &Json, route: &str) -> (usize, usize) {
    let name = field(case, "name").as_str().unwrap();
    let expected = field(case, "token_attributes").as_array().unwrap();
    assert_eq!(expected.len(), 512);
    for (id, value) in expected.iter().enumerate() {
        assert_eq!(
            tokenizer.token_attributes(id as i32),
            Some(value.as_u64().unwrap() as u32),
            "{name}/{route}: token {id} attributes"
        );
    }
    let calls = field(case, "tokenization").as_array().unwrap();
    assert_eq!(calls.len(), 168);
    for (index, call) in calls.iter().enumerate() {
        let input = unhex(field(call, "input_hex").as_str().unwrap());
        let ids: Vec<i32> = field(call, "tokens")
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(
            tokenizer
                .tokenize_bytes(
                    &input,
                    field(call, "add_special").as_bool().unwrap(),
                    field(call, "parse_special").as_bool().unwrap(),
                )
                .unwrap(),
            ids,
            "{name}/{route}: tokenization {index}"
        );
        let pieces = field(call, "pieces_hex").as_array().unwrap();
        assert_eq!(pieces.len(), ids.len());
        for (&id, piece) in ids.iter().zip(pieces) {
            assert_eq!(
                tokenizer.token_piece(id, true).unwrap(),
                unhex(piece.as_str().unwrap()),
                "{name}/{route}: tokenization {index} piece {id}"
            );
        }
    }
    (expected.len(), calls.len())
}

#[test]
fn raw_name_c_prefix_attributes_and_tokenization_match_original() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tokenizer/raw-name");
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/raw-name/original.json"
    ))
    .unwrap();
    let input = json::parse(include_str!(
        "../../../fixtures/tokenizer/raw-name/input-manifest.json"
    ))
    .unwrap();
    let cases = field(&reference, "cases").as_array().unwrap();
    let inputs = field(&input, "cases").as_array().unwrap();
    assert_eq!(cases.len(), 38);
    assert_eq!(inputs.len(), cases.len());
    let mut attributes = 0;
    let mut tokenizations = 0;
    for (case, source) in cases.iter().zip(inputs) {
        assert_eq!(field(case, "name"), field(source, "name"));
        let model = field(source, "model");
        let relative = field(model, "path").as_str().unwrap();
        assert_eq!(field(case, "model").as_str(), Some(relative));
        let path = root.join(relative);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, field(model, "bytes").as_u64().unwrap());
        assert_eq!(sha256(&bytes), field(model, "sha256").as_str().unwrap());
        let gguf = GgufSet::open(path).unwrap();
        let tokenizer = Tokenizer::from_metadata(gguf.metadata()).unwrap();
        let counts = check(&tokenizer, case, "model C-prefix view");
        attributes += counts.0;
        tokenizations += counts.1;

        // The public constructor also accepts unprojected metadata. It must
        // apply the same first-NUL view without converting raw bytes to text.
        let mut direct = gguf.metadata().clone();
        if let Some(raw_name) = field(case, "general_name_hex").as_str() {
            let name = unhex(raw_name);
            let value = match String::from_utf8(name) {
                Ok(text) => Value::String(text),
                Err(error) => Value::ByteString(error.into_bytes()),
            };
            direct.insert("general.name".into(), value);
        } else {
            direct.remove("general.name");
        }
        let tokenizer = Tokenizer::from_metadata(&direct).unwrap();
        assert_eq!(check(&tokenizer, case, "direct byte metadata"), counts);
    }
    assert_eq!(attributes, 19_456);
    assert_eq!(tokenizations, 6_384);
}
