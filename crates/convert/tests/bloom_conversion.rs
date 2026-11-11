//! Complete files from the frozen BLOOM converters.
//! The input pack stores only deduplicated independent source data.
use llama_checkpoint::{Value as Json, json};
use llama_conformance::digest;
use llama_convert::{Conversion, Limits, Options, OutputType, codec::SourceProfile};
use std::fs;
use std::path::PathBuf;
fn field<'a>(j: &'a Json, key: &str) -> &'a Json {
    j.as_object().unwrap().get(key).unwrap()
}
fn text(j: &Json) -> &str {
    j.as_text().unwrap().as_str().unwrap()
}
fn integer(j: &Json) -> u64 {
    let Json::Number(n) = j else {
        panic!("integer")
    };
    n.as_integer().unwrap().as_u64().unwrap()
}
fn boolean(j: &Json) -> bool {
    let Json::Bool(b) = j else { panic!("bool") };
    *b
}
fn array(j: &Json) -> &[Json] {
    j.as_array().unwrap()
}
fn blob<'a>(payload: &'a [u8], record: &Json) -> &'a [u8] {
    let at = integer(field(record, "offset")) as usize;
    let n = integer(field(record, "bytes")) as usize;
    let bytes = &payload[at..at + n];
    assert_eq!(digest::sha256(bytes), text(field(record, "sha256")));
    bytes
}
fn run() {
    let fixtures =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bloom-conversion");
    let manifest = json::parse(
        &fs::read(fixtures.join("pipeline.json")).unwrap(),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    let payload = fs::read(fixtures.join("pipeline.bin")).unwrap();
    assert_eq!(
        payload.len() as u64,
        integer(field(field(&manifest, "payload"), "bytes"))
    );
    assert_eq!(
        digest::sha256(&payload),
        text(field(field(&manifest, "payload"), "sha256"))
    );
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    let root = std::env::temp_dir().join(format!(
        "rust-llama-bloom-conversion-{}",
        std::process::id()
    ));
    fs::create_dir(&root).expect("fresh test folder");
    let cases = array(field(field(&manifest, "profiles"), profile));
    let (mut accepted, mut rejected, mut total) = (0, 0, 0u64);
    for case in cases {
        let name = text(field(case, "name"));
        let folder = root.join(name).join("model");
        fs::create_dir_all(&folder).unwrap();
        for file in array(field(case, "input_files")) {
            let path = folder.join(text(field(file, "path")));
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, blob(&payload, file)).unwrap();
        }
        let result = Conversion::prepare(
            &folder,
            Options {
                output_type: OutputType::parse(text(field(case, "ftype"))).unwrap(),
                source_profile: Some(SourceProfile::current()),
                vocab_only: boolean(field(case, "vocab_only")),
                fuse_qkv: boolean(field(case, "fuse_qkv")),
                model_name: match field(case, "model_name") {
                    Json::Null => None,
                    value => Some(text(value).to_owned()),
                },
                ..Options::default()
            },
            Limits::default(),
        )
        .and_then(|c| {
            let mut bytes = Vec::new();
            c.write_to(&mut bytes)?;
            assert_eq!(bytes.len() as u64, c.summary().bytes, "{name}");
            Ok(bytes)
        });
        if text(field(case, "status")) == "accepted" {
            let output = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let expected = field(case, "output");
            assert_eq!(
                output.len() as u64,
                integer(field(expected, "bytes")),
                "{name} length"
            );
            let hash = digest::sha256(&output);
            if hash != text(field(expected, "sha256")) {
                fs::write(folder.parent().unwrap().join("rust-mismatch.gguf"), &output).unwrap();
            }
            assert_eq!(
                hash,
                text(field(expected, "sha256")),
                "{name} complete original file"
            );
            accepted += 1;
            total += output.len() as u64;
        } else {
            assert!(result.is_err(), "{name}: original rejected, Rust admitted");
            rejected += 1;
        }
        if name == "bloom-nondivisible-weight-only" {
            let options = Options {
                output_type: OutputType::F32,
                model_name: Some("Tiny BLOOM Converter Fixture".to_owned()),
                ..Options::default()
            };
            // This independently captured QKV has 90*32 F32 elements. The
            // allowance covers joined/raw/prepared buffers exactly, so the
            // new permutation's buffer must fail its charge before allocation.
            let limited = Limits {
                max_working_bytes: 3 * 90 * 32 * 4,
                ..Limits::default()
            };
            let conversion = Conversion::prepare(&folder, options.clone(), limited).unwrap();
            let target = folder.parent().unwrap().join("preserved.gguf");
            fs::write(&target, b"existing destination").unwrap();
            let error = conversion.write_file(&target).unwrap_err();
            assert!(matches!(
                error,
                llama_convert::Error::Limit("per-output tensor working allocations")
            ));
            assert_eq!(fs::read(&target).unwrap(), b"existing destination");
            assert!(fs::read_dir(folder.parent().unwrap()).unwrap().all(|e| {
                !e.unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".rust-llama-convert-")
            }));
            assert!(
                Conversion::prepare(
                    &folder,
                    options,
                    Limits {
                        max_tensor_input_bytes: 90 * 32 * 4 - 1,
                        ..Limits::default()
                    },
                )
                .is_err()
            );
        }
        println!("{profile} {name}: original result reproduced");
    }
    let expected = (84, 29, 1_066_141_612);
    assert_eq!((accepted, rejected, total), expected);
    fs::remove_dir_all(root).unwrap();
    println!("{profile} bloom: {accepted} complete files, {rejected} errors, {total} bytes");
}
#[test]
fn frozen_bloom_complete_files() {
    run()
}
