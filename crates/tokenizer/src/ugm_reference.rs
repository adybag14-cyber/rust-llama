//! Exhaustive expected hashes from the independent original C++ Unigram
//! normalizer. Its complete raw captures remain in the external oracle archive.
use crate::{Tokenizer, unicode};
use llama_behavior::json::{self, Value as Json};
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits, Value};
use std::{borrow::Cow, collections::BTreeMap, io::Cursor};
fn number(bytes: &[u8], at: &mut usize) -> usize {
    let value = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap()) as usize;
    *at += 4;
    value
}
fn blob<'a>(bytes: &'a [u8], at: &mut usize) -> &'a [u8] {
    let length = number(bytes, at);
    let start = *at;
    *at += length;
    &bytes[start..*at]
}
fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    value.get(key).unwrap()
}
fn tokenizer(charsmap: Value, merge: bool, protected: bool) -> Tokenizer {
    let mut tokens = vec!["<pad>", "</s>", "<unk>"];
    let mut kinds = vec![3, 3, 2];
    if protected {
        tokens.extend(["Ａtoken", "<user>", "a\u{0301}special"]);
        kinds.extend([4, 4, 4]);
    }
    Tokenizer::from_metadata(&BTreeMap::from([
        ("tokenizer.ggml.model".into(), Value::String("t5".into())),
        (
            "tokenizer.ggml.tokens".into(),
            Value::Array {
                element_type: 8,
                values: tokens
                    .into_iter()
                    .map(|s| Value::String(s.into()))
                    .collect(),
            },
        ),
        (
            "tokenizer.ggml.token_type".into(),
            Value::Array {
                element_type: 5,
                values: kinds.into_iter().map(Value::I32).collect(),
            },
        ),
        ("tokenizer.ggml.precompiled_charsmap".into(), charsmap),
        (
            "tokenizer.ggml.remove_extra_whitespaces".into(),
            Value::Bool(merge),
        ),
    ]))
    .unwrap()
}
#[test]
fn all_codepoints_xcda_keys_raw_prefixes_and_nfc_match_original_unigram() {
    let bytes = include_bytes!("../../../fixtures/tokenizer/families/normalizer-inputs.bin");
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/families/normalizer-reference.json"
    ))
    .unwrap();
    assert_eq!(&bytes[..8], b"UINP0001");
    assert_eq!(
        bytes.len() as u64,
        field(&reference, "input_bytes").as_u64().unwrap()
    );
    assert_eq!(
        sha256(bytes),
        field(&reference, "input_sha256").as_str().unwrap()
    );
    assert_eq!(field(&reference, "chunk_size").as_u64(), Some(1024));
    let mut at = 8;
    let count = number(bytes, &mut at);
    let inputs: Vec<_> = (0..count).map(|_| blob(bytes, &mut at)).collect();
    assert_eq!(at, bytes.len());
    assert_eq!(count, 306_574);
    let model = Gguf::read(
        Cursor::new(include_bytes!(
            "../../../fixtures/tokenizer/families/ugm-synthetic.gguf"
        )),
        Limits::default(),
    )
    .unwrap();
    let charsmap = model.metadata["tokenizer.ggml.precompiled_charsmap"].clone();
    let groups = field(&reference, "groups").as_array().unwrap();
    assert_eq!(groups.len(), 8);
    let mut total = 0;
    for group in groups {
        let name = field(group, "name").as_str().unwrap();
        let count = field(group, "count").as_u64().unwrap() as usize;
        let all_codepoints = name.starts_with("all-codepoints");
        assert_eq!(
            count,
            if all_codepoints {
                0x110000
            } else {
                inputs.len()
            }
        );
        let prepend = !name.contains("prefix0");
        let merge = !name.contains("merge0");
        let nfc = name == "map-nfc";
        let protected = name.contains("protected");
        let tokenizer = tokenizer(charsmap.clone(), merge, protected);
        let normalizer = tokenizer.ugm.as_ref().unwrap();
        let expected = field(group, "chunk_hashes").as_array().unwrap();
        let mut buffer = Vec::new();
        let mut chunk = 0;
        let cases: Box<dyn Iterator<Item = Cow<'_, [u8]>> + '_> = if all_codepoints {
            Box::new(
                (0..count).map(|index| Cow::Owned(unicode::encode_utf8(index as u32).unwrap())),
            )
        } else {
            Box::new(inputs.iter().map(|input| Cow::Borrowed(*input)))
        };
        for (index, input) in cases.enumerate() {
            let output = normalizer.normalize(&input, prepend, nfc).unwrap();
            buffer.extend(u32::try_from(output.len()).unwrap().to_le_bytes());
            buffer.extend(output);
            if (index + 1) % 1024 == 0 || index + 1 == count {
                assert_eq!(
                    sha256(&buffer),
                    expected[chunk].as_str().unwrap(),
                    "{name}: source chunk {chunk}"
                );
                buffer.clear();
                chunk += 1;
            }
        }
        assert_eq!(chunk, expected.len());
        total += count;
    }
    assert_eq!(total, 4_067_668);
    eprintln!(
        "Unigram: 4,067,668 exact native normalization outputs, including every codepoint, every XCDA prefix, all one/two-byte strings and NFC/NFD sequence cases"
    );
}
