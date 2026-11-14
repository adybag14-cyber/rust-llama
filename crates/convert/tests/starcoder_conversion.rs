//! Complete files from the frozen StarCoder and StarCoder2 converters.
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
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/starcoder-conversion");
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
        "rust-llama-starcoder-conversion-{}",
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
        println!("{profile} {name}: original result reproduced");
    }
    let expected = (163, 31, 357723535);
    assert_eq!((accepted, rejected, total), expected);
    fs::remove_dir_all(root).unwrap();
    println!("{profile} starcoder: {accepted} complete files, {rejected} errors, {total} bytes");
}
#[test]
fn frozen_starcoder_complete_files() {
    run()
}
