//! Captured Tokenizers 0.22.2 execution; expectations precede the port.
use crate::hf_backend::{Added, Backend, nfc, qwen_regex_end};
use crate::util::{JsonBudget, integer, object, text};
use crate::{Error, Limits};
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
fn strings(value: &Json) -> Vec<&str> {
    array(value).iter().map(|v| text(v).unwrap()).collect()
}
fn ids(value: &Json) -> Vec<u32> {
    array(value)
        .iter()
        .map(|v| integer(v).unwrap() as u32)
        .collect()
}
#[test]
fn native_qwen_split_nfc_and_bpe() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-qwen/split-reference.json"
    ));
    let mut count = 0;
    for group in array(field(&root, "groups")) {
        let mut backend =
            Backend::from_json(field(group, "tokenizer"), &Limits::default()).unwrap();
        for request in array(field(group, "requests")) {
            let input = text(field(request, "input")).unwrap();
            let label = text(field(request, "label")).unwrap();
            let mut split = Vec::new();
            let mut start = 0;
            while start < input.len() {
                let end = qwen_regex_end(input, start);
                assert!(end > start && input.is_char_boundary(end));
                split.push(&input[start..end]);
                start = end;
            }
            assert_eq!(
                split,
                strings(field(request, "split")),
                "{label}: {input:?}"
            );
            assert_eq!(
                backend.pieces(input).unwrap(),
                strings(field(request, "pieces")),
                "{label}: {input:?}"
            );
            let actual = backend.encode(input).unwrap();
            assert_eq!(actual, ids(field(request, "ids")), "{label}: {input:?}");
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(request, "decode")).unwrap(),
                "{label}: {input:?}"
            );
            count += 1;
        }
    }
    assert_eq!(count, 3116);
    eprintln!("Qwen Split/NFC: {count} native piece/ID/decoder cases on both captured profiles");
}
#[test]
fn native_nfc_every_scalar_and_sequences() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-qwen/nfc-reference.json"
    ));
    let changes = array(field(&root, "single_scalar_changes"));
    let mut changed = 0;
    let mut work = Limits::default().max_tokenizer_operations;
    let mut stream = Vec::new();
    let mut scalars = 0;
    for cp in 0..0x110000 {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let mut encoded = [0; 4];
        let input: &str = ch.encode_utf8(&mut encoded);
        let actual = nfc::normalize(input, 1024, &mut work).unwrap();
        if changed < changes.len() && integer(&array(&changes[changed])[0]).unwrap() as u32 == cp {
            let expected: String = ids(&array(&changes[changed])[1])
                .into_iter()
                .map(|cp| char::from_u32(cp).unwrap())
                .collect();
            assert_eq!(actual, expected, "NFC U+{cp:06X}");
            changed += 1;
        } else {
            assert_eq!(actual, input, "NFC identity U+{cp:06X}");
        }
        stream.extend_from_slice(&cp.to_le_bytes());
        stream.extend_from_slice(&(actual.chars().count() as u32).to_le_bytes());
        for ch in actual.chars() {
            stream.extend_from_slice(&(ch as u32).to_le_bytes());
        }
        scalars += 1;
    }
    assert_eq!(changed, changes.len());
    assert_eq!(scalars, 1_112_064);
    assert_eq!(
        llama_conformance::digest::sha256(&stream),
        text(field(&root, "single_scalar_sha256")).unwrap()
    );
    for case in array(field(&root, "sequences")) {
        let input = text(field(case, "input")).unwrap();
        assert_eq!(
            nfc::normalize(input, 1 << 20, &mut work).unwrap(),
            text(field(case, "output")).unwrap(),
            "{input:?}"
        );
    }
    assert_eq!(array(field(&root, "sequences")).len(), 5645);
    eprintln!("NFC: all {scalars} Unicode scalars and 5645 native sequences, Unicode9 tables");
}
#[test]
fn native_nfc_unicode9_conformance() {
    let raw = include_bytes!("../tests/fixtures/hf-qwen/nfc-conformance.bin");
    assert_eq!(&raw[..8], b"HFNFCV01");
    let mut at = 8;
    let u32_at = |at: &mut usize| {
        let value = u32::from_le_bytes(raw[*at..*at + 4].try_into().unwrap()) as usize;
        *at += 4;
        value
    };
    let count = u32_at(&mut at);
    let mut work = Limits::default().max_tokenizer_operations;
    for index in 0..count {
        let a = u32_at(&mut at);
        let b = u32_at(&mut at);
        let input = std::str::from_utf8(&raw[at..at + a]).unwrap();
        at += a;
        let expected = std::str::from_utf8(&raw[at..at + b]).unwrap();
        at += b;
        assert_eq!(
            nfc::normalize(input, 1 << 20, &mut work).unwrap(),
            expected,
            "native conformance {index}: {input:?}"
        );
    }
    assert_eq!(at, raw.len());
    assert_eq!(count, 93634);
    eprintln!(
        "NFC: {count} native outputs from Unicode9 conformance inputs and newer-CCC witnesses"
    );
}
#[test]
fn native_nfc_added_token_flags_and_collisions() {
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-qwen/added-reference.json"
    ));
    let mut count = 0;
    for group in array(field(&root, "groups")) {
        let mut backend =
            Backend::from_json(field(&root, "tokenizer"), &Limits::default()).unwrap();
        let added: Vec<_> = array(field(group, "added"))
            .iter()
            .map(|v| Added::from_json(v, true, false).unwrap())
            .collect();
        backend.add_tokens(&added).unwrap();
        for request in array(field(group, "requests")) {
            let input = text(field(request, "input")).unwrap();
            let actual = backend.encode(input).unwrap();
            assert_eq!(actual, ids(field(request, "ids")), "{added:?}: {input:?}");
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(request, "decode")).unwrap(),
                "{added:?}: {input:?}"
            );
            count += 1;
        }
    }
    assert_eq!(array(field(&root, "groups")).len(), 272);
    assert_eq!(count, 4896);
    eprintln!(
        "NFC added-token patterns: 272 flag/collision configurations, {count} native requests"
    );
}
fn small_tokenizer(behavior: &str, invert: bool, normalizer: &str) -> Json {
    let pattern = crate::hf_schema::QWEN_PATTERN.replace('\\', "\\\\");
    let input = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":{normalizer},"pre_tokenizer":{{"type":"Split","pattern":{{"Regex":"{pattern}"}},"behavior":"{behavior}","invert":{invert}}},"post_processor":null,"decoder":null,"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{"a":0,"b":1,"ab":2}},"merges":[["a","b"]]}}}}"#
    );
    parse(input.as_bytes())
}
#[test]
fn qwen_split_scope_and_normalization_budgets() {
    for behavior in [
        "Removed",
        "MergedWithPrevious",
        "MergedWithNext",
        "Contiguous",
    ] {
        let root = small_tokenizer(behavior, false, "null");
        let mut backend = Backend::from_json(&root, &Limits::default()).unwrap();
        assert!(matches!(backend.encode("ab"), Err(Error::Unsupported(_))));
    }
    let root = small_tokenizer("Isolated", true, "null");
    assert!(matches!(
        Backend::from_json(&root, &Limits::default())
            .unwrap()
            .encode("ab"),
        Err(Error::Unsupported(_))
    ));
    for normalizer in [
        "NFD",
        "NFKC",
        "NFKD",
        "Lowercase",
        "StripAccents",
        "Nmt",
        "ByteLevel",
    ] {
        let root = small_tokenizer("Isolated", false, &format!(r#"{{"type":"{normalizer}"}}"#));
        assert!(matches!(
            Backend::from_json(&root, &Limits::default())
                .unwrap()
                .encode("ab"),
            Err(Error::Unsupported(_))
        ));
    }
    let root = small_tokenizer(
        "Isolated",
        false,
        r#"{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"NFC"}]}"#,
    );
    assert_eq!(
        Backend::from_json(&root, &Limits::default())
            .unwrap()
            .encode("ab")
            .unwrap(),
        [2]
    );
    let root = small_tokenizer(
        "Isolated",
        false,
        r#"{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Lowercase"}]}"#,
    );
    assert!(matches!(
        Backend::from_json(&root, &Limits::default())
            .unwrap()
            .encode("ab"),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        nfc::normalize("e\u{301}", 20, &mut 1000),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        nfc::normalize("e\u{301}", 1024, &mut 1),
        Err(Error::Limit(_))
    ));
    let adversarial = "a".to_owned() + &"\u{315}\u{300}\u{323}".repeat(128);
    assert!(matches!(
        nfc::normalize(&adversarial, 1 << 20, &mut 2000),
        Err(Error::Limit(_))
    ));
    let root = small_tokenizer("Isolated", false, "null");
    let limits = Limits {
        max_tokenizer_operations: 1,
        ..Limits::default()
    };
    assert!(matches!(
        Backend::from_json(&root, &limits).unwrap().pieces("abc"),
        Err(Error::Limit(_))
    ));
}
#[test]
#[ignore = "requires unchanged official Qwen tokenizer sidecars via LLAMA_TEST_QWEN_TOKENIZER_ROOT"]
fn pretrained_qwen_tokenizer_reference() {
    let base = std::path::PathBuf::from(
        std::env::var_os("LLAMA_TEST_QWEN_TOKENIZER_ROOT")
            .expect("set official tokenizer cache root"),
    );
    let captures: &[(&str, &[u8])] = &[
        (
            "Qwen2.5-0.5B-Instruct",
            include_bytes!(
                "../tests/fixtures/hf-qwen/windows/Qwen2.5-0.5B-Instruct-reference.json"
            ),
        ),
        (
            "Qwen3-0.6B",
            include_bytes!("../tests/fixtures/hf-qwen/windows/Qwen3-0.6B-reference.json"),
        ),
        (
            "Qwen2.5-0.5B-Instruct",
            include_bytes!("../tests/fixtures/hf-qwen/linux/Qwen2.5-0.5B-Instruct-reference.json"),
        ),
        (
            "Qwen3-0.6B",
            include_bytes!("../tests/fixtures/hf-qwen/linux/Qwen3-0.6B-reference.json"),
        ),
    ];
    let mut count = 0;
    for &(name, reference) in captures {
        let root = parse(reference);
        let raw = std::fs::read(base.join(name).join("tokenizer.json")).unwrap();
        assert_eq!(
            llama_conformance::digest::sha256(&raw),
            text(field(field(&root, "tokenizer"), "sha256")).unwrap()
        );
        let tokenizer = parse(&raw);
        let mut backend = Backend::from_json(&tokenizer, &Limits::default()).unwrap();
        for request in array(field(&root, "requests")) {
            let input = text(field(request, "input")).unwrap();
            let actual = backend.encode(input).unwrap();
            assert_eq!(actual, ids(field(request, "ids")), "{name}: {input:?}");
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(request, "decode")).unwrap(),
                "{name}: {input:?}"
            );
            count += 1;
        }
    }
    assert_eq!(count, 6232);
    eprintln!("Official unchanged Qwen2.5/Qwen3 tokenizers: {count} captured ID/decoder requests");
}

#[test]
fn native_qwen_fast_generic_flags() {
    let tokenizer = parse(include_bytes!(
        "../tests/fixtures/hf-qwen/qwen-fast-input.json"
    ));
    let root = parse(include_bytes!(
        "../tests/fixtures/hf-qwen/qwen-fast-generic-flags-reference.json"
    ));
    let mut count = 0;
    for case in array(field(&root, "cases")) {
        let config = object(field(case, "kwargs")).unwrap();
        let mut backend = Backend::from_json(&tokenizer, &Limits::default()).unwrap();
        backend.override_direct_prefix(config, false).unwrap();
        backend
            .add_tokens(&[Added::plain("<|endoftext|>", true)])
            .unwrap();
        for request in array(field(case, "requests")) {
            let input = text(field(request, "input")).unwrap();
            let actual = backend.encode(input).unwrap();
            assert_eq!(actual, ids(field(request, "ids")), "{config:?}: {input:?}");
            assert_eq!(
                backend.decode(&actual).unwrap(),
                text(field(request, "decoded")).unwrap(),
                "{config:?}: {input:?}"
            );
            count += 1;
        }
    }
    assert_eq!(count, 20);
}
