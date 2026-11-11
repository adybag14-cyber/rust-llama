//! Transformers BloomTokenizerFast captures, independent of product code.
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
fn native_bloom_fast_class_semantics() {
    let root = parse(include_bytes!(
        "../tests/fixtures/bloom-conversion/class-reference.json"
    ));
    let (mut accepted, mut rejected, mut encodings) = (0, 0, 0);
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    for case in array(field(field(&root, "profiles"), profile)) {
        let name = text(field(case, "name")).unwrap();
        let mut budget = JsonBudget::new(llama_checkpoint::Limits::default());
        let config = crate::config::Config::new(
            field(case, "model").clone(),
            "pinned-bloom-model",
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
                backend
                    .decode(&actual)
                    .unwrap_or_else(|e| panic!("{name}: {input:?}: decode: {e}")),
                text(field(request, "decoded")).unwrap(),
                "{name}: {input:?}"
            );
            encodings += 1;
        }
        accepted += 1;
    }
    assert_eq!((accepted, rejected, encodings), (82, 5, 2296));
    eprintln!("BloomTokenizerFast: 87 class cases, 2296 native encoding/decoder requests");
}

#[test]
fn native_bloom_metadata_wire_contract() {
    let root = parse(include_bytes!(
        "../tests/fixtures/bloom-conversion/metadata-reference.json"
    ));
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    let (mut accepted, mut rejected) = (0, 0);
    for case in array(field(field(&root, "profiles"), profile)) {
        let name = text(field(case, "name")).unwrap();
        let limits = Limits::default();
        let result = (|| -> crate::Result<Vec<u8>> {
            let mut budget = JsonBudget::new(limits.json.clone());
            let config = crate::config::Config::new(
                field(case, "config").clone(),
                "bloom-metadata",
                &mut budget,
                &limits,
            )?;
            let mut metadata = crate::util::Metadata::default();
            metadata.string("general.architecture", "bloom")?;
            config.metadata(&mut metadata, crate::OutputType::F32)?;
            let refs: Vec<_> = metadata.0.iter().map(|(k, v)| (k.as_bytes(), v)).collect();
            let mut bytes = Vec::new();
            let layout =
                llama_gguf::write_source_conversion_header(&mut bytes, &refs, &[], &limits.gguf)?;
            bytes.truncate(layout.encoded_header_bytes as usize);
            Ok(bytes)
        })();
        if text(field(case, "status")).unwrap() == "accepted" {
            let bytes = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                bytes.len() as i64,
                integer(field(case, "bytes")).unwrap(),
                "{name}"
            );
            assert_eq!(
                llama_conformance::digest::sha256(&bytes),
                text(field(case, "sha256")).unwrap(),
                "{name}"
            );
            let encoded: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(encoded, text(field(case, "hex")).unwrap(), "{name}");
            accepted += 1;
        } else {
            assert!(result.is_err(), "{name}: original metadata writer rejected");
            rejected += 1;
        }
    }
    assert_eq!((accepted, rejected), (20, 21));
}

#[test]
fn native_bloom_regex_scalar_and_pipeline_contract() {
    let root = parse(include_bytes!(
        "../tests/fixtures/bloom-conversion/regex-reference.json"
    ));
    let membership = include_bytes!("../tests/fixtures/bloom-conversion/regex-membership.bin");
    assert_eq!(membership.len(), 0x110000);
    assert_eq!(
        llama_conformance::digest::sha256(membership),
        text(field(field(&root, "membership"), "sha256")).unwrap()
    );
    let mut scalars = 0;
    for (cp, &expected) in membership.iter().enumerate() {
        if let Some(ch) = char::from_u32(cp as u32) {
            assert_eq!(
                crate::hf_backend::bloom_regex_member(ch),
                expected == 1,
                "U+{cp:04X}"
            );
            scalars += 1;
        }
    }
    assert_eq!(scalars, 1_112_064);
    assert_eq!(
        crate::hf_schema::BLOOM_PATTERN,
        text(field(&root, "pattern")).unwrap()
    );
    let mut requests = 0;
    for profile in array(field(&root, "profiles")) {
        let mut backend =
            crate::hf_backend::Backend::from_json(field(profile, "tokenizer"), &Limits::default())
                .unwrap();
        for case in array(field(profile, "cases")) {
            let input = text(field(case, "text")).unwrap();
            let pieces: Vec<_> = array(field(case, "pieces"))
                .iter()
                .map(|v| text(v).unwrap())
                .collect();
            assert_eq!(backend.pieces(input).unwrap(), pieces, "{input:?}");
            let actual = backend.encode(input).unwrap();
            assert_eq!(actual, ids(field(case, "ids")), "{input:?}");
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(case, "decoded")).unwrap(),
                "{input:?}"
            );
            requests += 1;
        }
    }
    assert_eq!(requests, 1922);
}
