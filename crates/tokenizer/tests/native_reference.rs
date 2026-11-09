//! Independent original C++ expected data; no Rust output defines the oracle.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits, Value as GgufValue};
use llama_tokenizer::{Tokenizer, VocabType, unicode};
use std::{collections::BTreeMap, io::Cursor};

fn numbers(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
fn unicode_reference() -> Value {
    json::parse(include_str!(
        "../../../fixtures/tokenizer/unicode-reference.json"
    ))
    .unwrap()
}
#[test]
fn exhaustive_frozen_properties_case_nfd_and_single_nfc() {
    let reference = unicode_reference();
    assert_eq!(
        reference.get("codepoints").unwrap().as_u64(),
        Some(0x110000)
    );
    let uppercase: BTreeMap<u32, u32> = reference
        .get("uppercase_map")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let row = numbers(row);
            (row[0], row[1])
        })
        .collect();
    let exceptions: BTreeMap<u32, Vec<u32>> = reference
        .get("nfc_single_exceptions")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let row = row.as_array().unwrap();
            (row[0].as_u64().unwrap() as u32, numbers(&row[1]))
        })
        .collect();
    assert_eq!(exceptions.len(), 1120);
    let mut next = 0;
    for run in reference.get("property_runs").unwrap().as_array().unwrap() {
        let run = run.as_array().unwrap();
        let first = run[0].as_u64().unwrap() as u32;
        let end = run[1].as_u64().unwrap() as u32;
        assert_eq!(first, next);
        assert!(end > first && end <= unicode::MAX_CODEPOINTS);
        let expected_flags = run[2].as_u64().unwrap() as u16;
        let lower_delta = run[3].as_i64().unwrap();
        let nfd_delta = run[4].as_i64().unwrap();
        let han = run[5].as_bool().unwrap();
        for cp in first..end {
            assert_eq!(unicode::flags(cp).0, expected_flags, "flags U+{cp:06X}");
            assert_eq!(
                i64::from(unicode::lowercase(cp)),
                i64::from(cp) + lower_delta,
                "lower U+{cp:06X}"
            );
            assert_eq!(
                unicode::uppercase(cp),
                uppercase.get(&cp).copied().unwrap_or(cp),
                "upper U+{cp:06X}"
            );
            assert_eq!(
                i64::from(unicode::nfd_base(cp)),
                i64::from(cp) + nfd_delta,
                "NFD U+{cp:06X}"
            );
            assert_eq!(unicode::is_han(cp), han, "Han U+{cp:06X}");
            let normalized = unicode::normalize_nfc(&[cp]);
            if let Some(expected) = exceptions.get(&cp) {
                assert_eq!(&normalized, expected, "NFC U+{cp:06X}");
            } else {
                assert_eq!(normalized, [cp], "NFC identity U+{cp:06X}");
            }
        }
        next = end;
    }
    assert_eq!(next, 0x110000);
    for row in reference
        .get("outside_codepoints")
        .unwrap()
        .as_array()
        .unwrap()
    {
        let row = row.as_array().unwrap();
        let cp = row[0].as_u64().unwrap() as u32;
        assert_eq!(u64::from(unicode::flags(cp).0), row[1].as_u64().unwrap());
        assert_eq!(u64::from(unicode::lowercase(cp)), row[2].as_u64().unwrap());
        assert_eq!(unicode::normalize_nfd(&[cp]), numbers(&row[3]));
        assert_eq!(unicode::normalize_nfc(&[cp]), numbers(&row[4]));
        assert_eq!(unicode::is_han(cp), row[5].as_bool().unwrap());
        assert_eq!(unicode::encode_utf8(cp).is_err(), row[6].as_bool().unwrap());
    }
    eprintln!(
        "frozen Unicode: all 1,114,112 codepoints, flags/case/NFD/Han and single-codepoint NFC; four out-of-range inputs"
    );
}
#[test]
fn complete_native_normalization_pairs_hangul_and_combining_order() {
    let reference = unicode_reference();
    let cases = reference
        .get("normalization_cases")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), 15_957);
    for (index, case) in cases.iter().enumerate() {
        let case = case.as_array().unwrap();
        let input = numbers(&case[0]);
        assert_eq!(
            unicode::normalize_nfd(&input),
            numbers(&case[1]),
            "NFD case {index}: {input:x?}"
        );
        assert_eq!(
            unicode::normalize_nfc(&input),
            numbers(&case[2]),
            "NFC case {index}: {input:x?}"
        );
    }
    for (byte, case) in reference
        .get("byte_encoding")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let case = case.as_array().unwrap();
        let cp = case[0].as_u64().unwrap() as u32;
        let encoded = hex(case[1].as_str().unwrap());
        assert_eq!(unicode::byte_to_codepoint(byte as u8), cp);
        assert_eq!(unicode::byte_to_utf8(byte as u8), encoded);
        assert_eq!(unicode::codepoint_to_byte(cp), Ok(byte as u8));
        assert_eq!(unicode::utf8_to_byte(&encoded), Ok(byte as u8));
        assert_eq!(
            unicode::len_utf8(byte as u8) as u64,
            reference
                .get("utf8_lead_lengths")
                .unwrap()
                .as_array()
                .unwrap()[byte]
                .as_u64()
                .unwrap()
        );
    }
    assert!(unicode::codepoint_to_byte(0).is_err());
    assert!(unicode::utf8_to_byte(b"ab").is_err());
    eprintln!(
        "frozen Unicode: 15,957 normalization sequences and all 256 byte mappings/lead lengths"
    );
}
#[test]
fn every_codepoint_utf8_encoding_matches_native_chunk_hashes() {
    let expected = include_bytes!("../../../fixtures/tokenizer/unicode-utf8-hashes.bin");
    assert_eq!(expected.len(), 4352 * 32);
    for (block, digest) in expected.chunks(32).enumerate() {
        let mut encoded = Vec::new();
        for cp in block as u32 * 256..(block as u32 + 1) * 256 {
            let bytes = unicode::encode_utf8(cp).unwrap();
            encoded.push(bytes.len() as u8);
            encoded.extend(&bytes);
            let mut offset = 0;
            assert_eq!(unicode::decode_next(&bytes, &mut offset), Ok(cp));
            assert_eq!(offset, bytes.len());
            assert_eq!(unicode::from_utf8(&bytes), [cp]);
        }
        assert_eq!(hex(&sha256(&encoded)), digest, "native UTF-8 block {block}");
    }
    eprintln!("frozen UTF-8: all 1,114,112 codepoints, including source surrogate encodings");
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, size: usize) -> &'a [u8] {
        let end = self.offset.checked_add(size).unwrap();
        let result = self
            .bytes
            .get(self.offset..end)
            .expect("truncated native fixture");
        self.offset = end;
        result
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn bytes(&mut self) -> &'a [u8] {
        let len = self.u32() as usize;
        self.take(len)
    }
}
#[test]
fn native_utf8_malformed_and_valid_sequences_preserve_offsets() {
    let bytes = include_bytes!("../../../fixtures/tokenizer/utf8-reference.bin");
    let mut reader = Reader { bytes, offset: 0 };
    assert_eq!(reader.take(8), b"UUTF0001");
    let count = reader.u32();
    assert_eq!(count, 65_809);
    for case in 0..count {
        let input = reader.bytes();
        let valid = reader.take(1)[0] != 0;
        let expected_offset = reader.u32() as usize;
        let first = reader.u32();
        let length = reader.u32();
        let expected: Vec<_> = (0..length).map(|_| reader.u32()).collect();
        let mut offset = 0;
        let result = unicode::decode_next(input, &mut offset);
        assert_eq!(result.is_ok(), valid, "case {case} validity: {input:x?}");
        if valid {
            assert_eq!(result.unwrap(), first, "case {case} first codepoint");
        }
        assert_eq!(offset, expected_offset, "case {case} offset");
        assert_eq!(
            unicode::from_utf8(input),
            expected,
            "case {case} replacement policy"
        );
        if input.is_empty() {
            assert_eq!(
                unicode::flags_from_utf8(input).unwrap().0,
                unicode::Flags::UNDEFINED
            );
        } else {
            assert_eq!(unicode::flags_from_utf8(input).is_ok(), valid);
        }
    }
    assert_eq!(reader.offset, bytes.len());
    let mut bad_offset = usize::MAX;
    assert!(unicode::decode_next(b"a", &mut bad_offset).is_err());
    assert_eq!(bad_offset, usize::MAX);
    eprintln!(
        "frozen UTF-8: 65,809 cases, every one/two-byte input plus boundary, surrogate and malformed sequences"
    );
}
#[test]
fn original_bert_vocabulary_and_all_wordpiece_variants() {
    let vocabulary = include_bytes!("../../../fixtures/tokenizer/ggml-vocab-bert-bge.gguf");
    assert_eq!(
        sha256(vocabulary),
        "fbcbe22278fb302694d5f4a41bfe48c5f90e8e3554eab1c0435387dff654a854"
    );
    let gguf = Gguf::read(Cursor::new(vocabulary), Limits::default()).unwrap();
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/wordpiece-reference.json"
    ))
    .unwrap();
    let variants = reference.get("variants").unwrap().as_array().unwrap();
    assert_eq!(variants.len(), 9);
    let mut case_count = 0;
    let mut piece_count = 0;
    let mut vocab_count = 0;
    for variant in variants {
        let name = variant.get("name").unwrap().as_str().unwrap();
        let mut metadata = gguf.metadata.clone();
        for (key, value) in variant.get("overrides").unwrap().as_object().unwrap() {
            let value = if let Some(value) = value.as_bool() {
                GgufValue::Bool(value)
            } else if let Some(value) = value.as_str() {
                GgufValue::String(value.to_owned())
            } else {
                GgufValue::U32(u32::try_from(value.as_u64().unwrap()).unwrap())
            };
            metadata.insert(key.clone(), value);
        }
        let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
        assert_eq!(tokenizer.vocab_type(), VocabType::Wpm);
        assert_eq!(
            tokenizer.vocabulary_size() as u64,
            variant.get("n_vocab").unwrap().as_u64().unwrap()
        );
        for (field, actual) in [
            ("bos", tokenizer.bos_id),
            ("eos", tokenizer.eos_id),
            ("sep", tokenizer.sep_id),
            ("pad", tokenizer.pad_id),
            ("mask", tokenizer.mask_id),
        ] {
            assert_eq!(
                i64::from(actual),
                variant.get(field).unwrap().as_i64().unwrap(),
                "{name} {field}"
            );
        }
        for (field, actual) in [
            ("add_bos", tokenizer.add_bos),
            ("add_eos", tokenizer.add_eos),
            ("add_sep", tokenizer.add_sep),
        ] {
            assert_eq!(
                actual,
                variant.get(field).unwrap().as_bool().unwrap(),
                "{name} {field}"
            );
        }
        if let Some(vocab) = variant.get("vocabulary") {
            let vocab = vocab.as_array().unwrap();
            assert_eq!(vocab.len(), tokenizer.vocabulary_size());
            for (id, item) in vocab.iter().enumerate() {
                let item = item.as_array().unwrap();
                let (text, score, _) = tokenizer.token_info(id as i32).unwrap();
                assert_eq!(text, item[0].as_str().unwrap(), "vocab text {id}");
                assert_eq!(
                    score.to_bits(),
                    (item[1].as_f64().unwrap() as f32).to_bits(),
                    "vocab score {id}"
                );
                assert_eq!(
                    tokenizer.token_attributes(id as i32).map(u64::from),
                    item[2].as_u64(),
                    "vocab attrs {id}"
                );
                assert_eq!(
                    tokenizer.token_piece(id as i32, false).unwrap(),
                    hex(item[3].as_str().unwrap()),
                    "vocab piece {id}"
                );
                assert_eq!(
                    tokenizer.token_piece(id as i32, true).unwrap(),
                    hex(item[4].as_str().unwrap()),
                    "vocab special piece {id}"
                );
                assert_eq!(
                    tokenizer.is_end_of_generation(id as i32),
                    item[5].as_bool().unwrap(),
                    "vocab EOG {id}"
                );
                vocab_count += 1;
            }
        }
        for case in variant.get("cases").unwrap().as_array().unwrap() {
            let label = case.get("name").unwrap().as_str().unwrap();
            let expected: Vec<i32> = numbers(case.get("tokens").unwrap())
                .into_iter()
                .map(|id| id as i32)
                .collect();
            let actual = tokenizer
                .tokenize(
                    case.get("prompt").unwrap().as_str().unwrap(),
                    case.get("add_special").unwrap().as_bool().unwrap(),
                    case.get("parse_special").unwrap().as_bool().unwrap(),
                )
                .unwrap();
            assert_eq!(actual, expected, "{name}/{label} token IDs");
            for (index, &id) in actual.iter().enumerate() {
                for (special, field) in [(false, "pieces"), (true, "pieces_special")] {
                    let expected = hex(case.get(field).unwrap().as_array().unwrap()[index]
                        .as_str()
                        .unwrap());
                    assert_eq!(
                        tokenizer.token_piece(id, special).unwrap(),
                        expected,
                        "{name}/{label} piece {index}"
                    );
                }
                assert_eq!(
                    tokenizer.token_attributes(id).map(u64::from),
                    case.get("attributes").unwrap().as_array().unwrap()[index].as_u64()
                );
                piece_count += 1;
            }
            for remove in [false, true] {
                for unparse in [false, true] {
                    let index = usize::from(remove) * 2 + usize::from(unparse);
                    let expected = hex(case.get("detokenized").unwrap().as_array().unwrap()[index]
                        .as_str()
                        .unwrap());
                    assert_eq!(
                        tokenizer
                            .detokenize_bytes(&actual, remove, unparse)
                            .unwrap(),
                        expected,
                        "{name}/{label} detokenize remove={remove} unparse={unparse}"
                    );
                }
            }
            case_count += 1;
        }
    }
    assert_eq!(case_count, 1728);
    assert_eq!(vocab_count, 30522);
    eprintln!(
        "native WordPiece: {case_count} cases / nine variants; {vocab_count} complete vocabulary entries; {piece_count} case tokens with both piece modes and four detokenization modes"
    );
}

#[test]
fn original_special_metadata_sentinels_retain_defaults() {
    let reference = json::parse(include_str!(
        "../../../fixtures/tokenizer/special-metadata-reference.json"
    ))
    .unwrap();
    let mut count = 0;
    for family in reference.get("families").unwrap().as_array().unwrap() {
        let name = family.get("family").unwrap().as_str().unwrap();
        let (vocabulary, kind): (&[u8], VocabType) = match name {
            "spm" => (
                include_bytes!("../../../fixtures/inference/ggml-vocab-llama-spm.gguf"),
                VocabType::Spm,
            ),
            "wpm" => (
                include_bytes!("../../../fixtures/tokenizer/ggml-vocab-bert-bge.gguf"),
                VocabType::Wpm,
            ),
            other => panic!("unknown native fixture family {other}"),
        };
        let gguf = Gguf::read(Cursor::new(vocabulary), Limits::default()).unwrap();
        for variant in family
            .get("reference")
            .unwrap()
            .get("variants")
            .unwrap()
            .as_array()
            .unwrap()
        {
            let variant_name = variant.get("name").unwrap().as_str().unwrap();
            let mut metadata = gguf.metadata.clone();
            for (key, value) in variant.get("overrides").unwrap().as_object().unwrap() {
                metadata.insert(
                    key.clone(),
                    GgufValue::U32(u32::try_from(value.as_u64().unwrap()).unwrap()),
                );
            }
            let tokenizer = Tokenizer::from_metadata(&metadata).unwrap();
            assert_eq!(tokenizer.vocab_type(), kind);
            for (key, actual) in [
                ("bos", tokenizer.bos_id),
                ("eos", tokenizer.eos_id),
                ("sep", tokenizer.sep_id),
                ("pad", tokenizer.pad_id),
                ("mask", tokenizer.mask_id),
            ] {
                assert_eq!(
                    i64::from(actual),
                    variant.get(key).unwrap().as_i64().unwrap(),
                    "{name}/{variant_name}/{key}"
                );
            }
            for case in variant.get("cases").unwrap().as_array().unwrap() {
                let label = case.get("name").unwrap().as_str().unwrap();
                let expected: Vec<i32> = numbers(case.get("tokens").unwrap())
                    .into_iter()
                    .map(|id| id as i32)
                    .collect();
                let actual = tokenizer
                    .tokenize(
                        case.get("prompt").unwrap().as_str().unwrap(),
                        case.get("add_special").unwrap().as_bool().unwrap(),
                        case.get("parse_special").unwrap().as_bool().unwrap(),
                    )
                    .unwrap();
                assert_eq!(actual, expected, "{name}/{variant_name}/{label}");
                for (index, &id) in actual.iter().enumerate() {
                    for (special, field) in [(false, "pieces"), (true, "pieces_special")] {
                        assert_eq!(
                            tokenizer.token_piece(id, special).unwrap(),
                            hex(case.get(field).unwrap().as_array().unwrap()[index]
                                .as_str()
                                .unwrap()),
                            "{name}/{variant_name}/{label} piece {index}"
                        );
                    }
                    assert_eq!(
                        tokenizer.token_attributes(id).map(u64::from),
                        case.get("attributes").unwrap().as_array().unwrap()[index].as_u64()
                    );
                }
                for remove in [false, true] {
                    for unparse in [false, true] {
                        let index = usize::from(remove) * 2 + usize::from(unparse);
                        assert_eq!(
                            tokenizer
                                .detokenize_bytes(&actual, remove, unparse)
                                .unwrap(),
                            hex(case.get("detokenized").unwrap().as_array().unwrap()[index]
                                .as_str()
                                .unwrap()),
                            "{name}/{variant_name}/{label} detokenize {index}"
                        );
                    }
                }
                count += 1;
            }
        }
    }
    assert_eq!(count, 72);
    eprintln!(
        "native special metadata: 72 SPM/WPM cases; all special ID UINT32_MAX and vocabulary-boundary overrides retain original defaults"
    );
}
