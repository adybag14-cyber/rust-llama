//! Independent complete-file converter oracle. The large input model corpus
//! lives outside the repository; every file is checked before it is consumed.
use super::*;
use crate::util::{JsonBudget, integer, object, text};
use llama_checkpoint::Value as Json;
use std::path::PathBuf;
fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    object(value).unwrap().get(key).unwrap()
}
fn array(value: &Json) -> &[Json] {
    value.as_array().unwrap()
}
#[test]
#[ignore = "requires frozen original corpus in LLAMA_TEST_HF_CORPUS and a fresh output directory"]
fn native_dense_llama_converter_full_bytes() {
    let root =
        PathBuf::from(std::env::var_os("LLAMA_TEST_HF_CORPUS").expect("set LLAMA_TEST_HF_CORPUS"));
    let output =
        PathBuf::from(std::env::var_os("LLAMA_TEST_HF_OUTPUT").expect("set LLAMA_TEST_HF_OUTPUT"));
    std::fs::create_dir(&output).expect("output directory must be fresh");
    let raw = std::fs::read(root.join("report.json")).unwrap();
    let report = JsonBudget::new(llama_checkpoint::Limits::default())
        .parse(&raw)
        .unwrap();
    let mut failures = Vec::new();
    let mut accepted = 0;
    let mut rejected = 0;
    let mut bytes = 0;
    for case in array(field(&report, "cases")) {
        let name = text(field(case, "name")).unwrap();
        let input = root.join("cases").join(name).join("model");
        for file in array(field(case, "input_files")) {
            let path = text(field(file, "path")).unwrap();
            let raw = std::fs::read(input.join(path)).unwrap();
            assert_eq!(
                raw.len(),
                integer(field(file, "bytes")).unwrap() as usize,
                "input {name}/{path}"
            );
            assert_eq!(
                llama_conformance::digest::sha256(&raw),
                text(field(file, "sha256")).unwrap(),
                "input {name}/{path}"
            );
        }
        let options = field(case, "options");
        let settings = Options {
            output_type: OutputType::parse(text(field(case, "ftype")).unwrap()).unwrap(),
            model_name: object(options)
                .unwrap()
                .get("model_name")
                .filter(|v| !matches!(v, Json::Null))
                .map(|v| text(v).unwrap().to_owned()),
            fuse_qkv: matches!(field(case, "fuse_qkv"), Json::Bool(true)),
            vocab_only: matches!(field(case, "vocab_only"), Json::Bool(true)),
            ..Options::default()
        };
        let result = Conversion::prepare(&input, settings, Limits::default()).and_then(|value| {
            let mut raw = Vec::new();
            value.write_to(&mut raw)?;
            Ok((raw, value.summary().clone()))
        });
        match (result, text(field(case, "status")).unwrap()) {
            (Ok((actual, summary)), "accepted") => {
                let expected =
                    std::fs::read(root.join("cases").join(name).join("converted.gguf")).unwrap();
                assert_eq!(
                    expected.len(),
                    integer(field(field(case, "output"), "bytes")).unwrap() as usize
                );
                assert_eq!(
                    llama_conformance::digest::sha256(&expected),
                    text(field(field(case, "output"), "sha256")).unwrap(),
                    "native output {name}"
                );
                std::fs::write(output.join(format!("{name}.gguf")), &actual).unwrap();
                if actual == expected {
                    accepted += 1;
                    bytes += actual.len();
                    println!(
                        "PASS {name}: {} bytes, {} tensors",
                        actual.len(),
                        summary.tensors
                    );
                } else {
                    let first = actual.iter().zip(&expected).position(|(a, b)| a != b);
                    failures.push(format!(
                        "{name}: actual {} bytes, expected {} bytes, first difference {first:?}",
                        actual.len(),
                        expected.len()
                    ));
                }
            }
            (Err(error), "rejected") => {
                rejected += 1;
                println!("PASS rejected {name}: {error}");
            }
            (Err(error), "accepted") => {
                failures.push(format!("{name}: unexpectedly rejected: {error}"))
            }
            (Ok((actual, _)), "rejected") => {
                std::fs::write(output.join(format!("{name}-unexpected.gguf")), actual).unwrap();
                failures.push(format!("{name}: unexpectedly accepted"));
            }
            (_, status) => panic!("unrecognized native status {status}"),
        }
    }
    println!("exact accepted={accepted}, checked rejected={rejected}, exact bytes={bytes}");
    assert!(
        failures.is_empty(),
        "native converter mismatches:\n{}",
        failures.join("\n")
    );
    assert_eq!((accepted, rejected), (57, 7));
    assert_eq!(bytes, 8_091_032);
}

#[test]
#[ignore = "requires pinned SmolLM2 HF weights in LLAMA_TEST_HF_MODEL and a fresh output directory"]
fn pretrained_smollm2_complete_conversions() {
    let input =
        PathBuf::from(std::env::var_os("LLAMA_TEST_HF_MODEL").expect("set LLAMA_TEST_HF_MODEL"));
    let output =
        PathBuf::from(std::env::var_os("LLAMA_TEST_HF_OUTPUT").expect("set LLAMA_TEST_HF_OUTPUT"));
    std::fs::create_dir(&output).expect("output directory must be fresh");
    let reference = JsonBudget::new(llama_checkpoint::Limits::default())
        .parse(include_bytes!(
            "../tests/fixtures/dense-conversion/smollm2-reference.json"
        ))
        .unwrap();
    for file in array(field(&reference, "input_files")) {
        let path = text(field(file, "path")).unwrap();
        let (length, hash) =
            llama_conformance::digest::hash_reader(std::fs::File::open(input.join(path)).unwrap())
                .unwrap();
        assert_eq!(
            length,
            integer(field(file, "bytes")).unwrap() as u64,
            "{path}"
        );
        assert_eq!(hash, text(field(file, "sha256")).unwrap(), "{path}");
    }
    let mut failures = Vec::new();
    let mut total = 0;
    for case in array(field(&reference, "outputs")) {
        let mode = text(field(case, "mode")).unwrap();
        let start = std::time::Instant::now();
        let conversion = Conversion::prepare(
            &input,
            Options {
                output_type: OutputType::parse(mode).unwrap(),
                ..Options::default()
            },
            Limits::default(),
        )
        .unwrap_or_else(|e| panic!("{mode}: {e}"));
        let path = output.join(format!("{mode}.gguf"));
        std::fs::write(&path, b"pre-publication sentinel").unwrap();
        let result = conversion
            .write_file(&path)
            .unwrap_or_else(|e| panic!("{mode}: {e}"));
        assert_eq!(result.canonicalize().unwrap(), path.canonicalize().unwrap());
        let seconds = start.elapsed().as_secs_f64();
        let (length, hash) =
            llama_conformance::digest::hash_reader(std::fs::File::open(&path).unwrap()).unwrap();
        let expected_len = integer(field(case, "bytes")).unwrap() as u64;
        let expected_hash = text(field(case, "sha256")).unwrap();
        if length != expected_len || hash != expected_hash {
            failures.push(format!(
                "{mode}: {length} bytes {hash}; expected {expected_len} bytes {expected_hash}"
            ));
        } else {
            println!(
                "PASS pretrained {mode}: {length} exact bytes, {} tensors, {seconds:.3}s conversion",
                conversion.summary().tensors
            );
            total += length;
        }
    }
    assert!(
        std::fs::read_dir(&output).unwrap().all(|v| !v
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".rust-llama-convert-")),
        "temporary outputs leaked"
    );
    assert!(
        failures.is_empty(),
        "pretrained converter mismatches:\n{}",
        failures.join("\n")
    );
    assert_eq!(total, 686_581_536);
}
