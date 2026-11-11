//! Independent pinned HF/Tokenizers fixtures; expected data is native-only.
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
fn opt_file(files: &Json, name: &str) -> Option<Json> {
    object(files)
        .unwrap()
        .get(name)
        .map(|v| parse(text(v).unwrap().as_bytes()))
}
fn files(value: &Json) -> vocab::Files {
    vocab::Files {
        tokenizer: opt_file(value, "tokenizer.json"),
        config: opt_file(value, "tokenizer_config.json"),
        added: opt_file(value, "added_tokens.json"),
    }
}
fn assert_metadata(backend: &Backend<'_>, specials: &[&str], expected: &Json, name: &str) {
    assert_eq!(
        backend.base_size,
        integer(field(expected, "vocab_size")).unwrap() as usize,
        "{name}"
    );
    let actual: Vec<_> = backend
        .by_id
        .iter()
        .enumerate()
        .filter_map(|(id, token)| token.map(|token| (token, id as u32)))
        .collect();
    let reference: Vec<_> = array(field(expected, "vocab"))
        .iter()
        .map(|p| {
            let p = array(p);
            (text(&p[0]).unwrap(), integer(&p[1]).unwrap() as u32)
        })
        .collect();
    assert_eq!(actual, reference, "vocabulary {name}");
    let reference = array(field(expected, "decoder"));
    assert_eq!(backend.added.len(), reference.len(), "added count {name}");
    for ((id, token), value) in backend.added.iter().zip(reference) {
        let pair = array(value);
        assert_eq!(*id, integer(&pair[0]).unwrap() as u32, "{name}");
        let expected = &pair[1];
        assert_eq!(
            token.content,
            text(field(expected, "content")).unwrap(),
            "{name}"
        );
        for (key, actual) in [
            ("single_word", token.single_word),
            ("lstrip", token.lstrip),
            ("rstrip", token.rstrip),
            ("normalized", token.normalized),
            ("special", token.special),
        ] {
            assert_eq!(
                actual,
                matches!(field(expected, key), Json::Bool(true)),
                "{name} token {} flag {key}",
                token.content
            );
        }
    }
    let reference: Vec<_> = array(field(expected, "special_tokens"))
        .iter()
        .map(|v| text(v).unwrap())
        .collect();
    assert_eq!(specials, reference, "special names {name}");
}
#[test]
fn native_hf_ingestion_and_saved_token_flags() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-tokenizer/ingest-reference.json"
    ));
    for profile in [
        codec::SourceProfile::WindowsTorch211Numpy246,
        codec::SourceProfile::LinuxTorch211Numpy226,
    ] {
        let mut accepted = 0;
        for case in array(field(&root, "cases")) {
            let name = text(field(case, "name")).unwrap();
            let input = field(case, "files");
            let files = files(input);
            let map = opt_file(input, "special_tokens_map.json");
            let result = hf_bpe::prepare(&files, map.as_ref(), &Limits::default(), profile);
            if text(field(case, "status")).unwrap() == "accepted" {
                let (backend, specials) = result.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_metadata(&backend, &specials, field(case, "metadata"), name);
                accepted += 1;
            } else {
                assert!(result.is_err(), "source rejected {name}");
            }
        }
        assert_eq!(accepted, 41);
    }
}
#[test]
fn native_bytelevel_digits_bpe_and_decoder() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-tokenizer/primitive-reference.json"
    ));
    let mut requests = 0;
    for group in array(field(&root, "cases")) {
        let name = text(field(group, "name")).unwrap();
        let mut backend =
            Backend::from_json(field(group, "tokenizer"), &Limits::default()).unwrap();
        for request in array(field(group, "requests")) {
            let input = text(field(request, "text")).unwrap();
            let pieces = backend.pieces(input).unwrap();
            let expected: Vec<_> = array(field(request, "pieces"))
                .iter()
                .map(|v| text(v).unwrap())
                .collect();
            assert_eq!(pieces, expected, "pieces {name} {input:?}");
            let ids = backend.encode(input).unwrap();
            let expected: Vec<_> = array(field(request, "ids"))
                .iter()
                .map(|v| integer(v).unwrap() as u32)
                .collect();
            assert_eq!(ids, expected, "IDs {name} {input:?}");
            assert_eq!(
                backend.decode(&ids).unwrap(),
                text(field(request, "decode")).unwrap(),
                "decode {name} {input:?}"
            );
            requests += 1;
        }
    }
    assert_eq!(requests, 300);
}
#[test]
fn exhaustive_hf_unicode_and_added_boundaries() {
    let expected = include_bytes!("../tests/fixtures/hf-unicode/flags.bin");
    assert_eq!(expected.len(), 0x110000);
    for (cp, &reference) in expected.iter().enumerate() {
        assert_eq!(hf_unicode::flags(cp as u32), reference, "U+{cp:04X}");
    }
    assert_eq!(
        llama_conformance::digest::sha256(expected),
        "471fd91d0bae4376af9804da70189aa1e6b0ee0ac887287da72f15dc766421eb"
    );
}
fn metadata_frame(backend: &Backend<'_>, specials: &[&str]) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend((value.len() as u64).to_le_bytes());
        out.extend(value.as_bytes());
    }
    let mut out = (backend.base_size as u64).to_le_bytes().to_vec();
    out.extend((backend.by_id.iter().filter(|v| v.is_some()).count() as u64).to_le_bytes());
    for (id, token) in backend.by_id.iter().enumerate() {
        if let Some(token) = token {
            out.extend((id as u32).to_le_bytes());
            string(&mut out, token);
        }
    }
    out.extend((backend.added.len() as u64).to_le_bytes());
    for (id, token) in &backend.added {
        out.extend(id.to_le_bytes());
        string(&mut out, token.content);
        out.extend([
            token.single_word as u8,
            token.lstrip as u8,
            token.rstrip as u8,
            token.normalized as u8,
            token.special as u8,
        ]);
    }
    out.extend((specials.len() as u64).to_le_bytes());
    for value in specials {
        string(&mut out, value);
    }
    out
}
#[test]
#[ignore = "requires the pinned HF SmolLM2 input directory in LLAMA_TEST_HF_MODEL"]
fn pretrained_smollm2_hf_vocabulary_and_probe() {
    let input = std::env::var_os("LLAMA_TEST_HF_MODEL").expect("set LLAMA_TEST_HF_MODEL");
    let reference = parse(include_bytes!(
        "../tests/fixtures/hf-tokenizer/smollm2-reference.json"
    ));
    let mut source =
        llama_checkpoint::SourceDirectory::open(&input, llama_checkpoint::SourceLimits::default())
            .unwrap();
    for expected in array(field(&reference, "inputs")) {
        let path = text(field(expected, "path")).unwrap();
        let bytes = source.read_required(path).unwrap();
        assert_eq!(
            bytes.len(),
            integer(field(expected, "bytes")).unwrap() as usize,
            "{path}"
        );
        assert_eq!(
            llama_conformance::digest::sha256(bytes),
            text(field(expected, "sha256")).unwrap(),
            "{path}"
        );
    }
    let mut budget = JsonBudget::new(llama_checkpoint::Limits::default());
    let files = vocab::Files::load(&mut source, &mut budget).unwrap();
    let map = source
        .read_optional("special_tokens_map.json")
        .unwrap()
        .map(parse);
    let (mut backend, specials) = hf_bpe::prepare(
        &files,
        map.as_ref(),
        &Limits::default(),
        codec::SourceProfile::current(),
    )
    .unwrap();
    let frame = metadata_frame(&backend, &specials);
    assert_eq!(
        frame.len(),
        integer(field(&reference, "metadata_frame_bytes")).unwrap() as usize
    );
    assert_eq!(
        llama_conformance::digest::sha256(&frame),
        text(field(&reference, "metadata_frame_sha256")).unwrap()
    );
    let ids = backend.encode(hf_probe::PROBE).unwrap();
    let expected: Vec<_> = array(field(&reference, "probe_ids"))
        .iter()
        .map(|v| integer(v).unwrap() as u32)
        .collect();
    assert_eq!(ids, expected);
}

#[path = "hf_qwen_reference.rs"]
mod qwen;
