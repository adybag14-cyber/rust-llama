//! Complete-file equality against the frozen BLOOM converter on the genuine
//! unchanged, revision-pinned bigscience/bloom-560m checkpoint.
use llama_checkpoint::{Value as Json, json};
use llama_conformance::digest;
use llama_convert::{Conversion, Limits, Options, OutputType, codec::SourceProfile};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    value.as_object().unwrap().get(key).unwrap()
}
fn text(value: &Json) -> &str {
    value.as_text().unwrap().as_str().unwrap()
}
fn integer(value: &Json) -> u64 {
    let Json::Number(n) = value else {
        panic!("expected integer")
    };
    n.as_integer().unwrap().as_u64().unwrap()
}
fn array(value: &Json) -> &[Json] {
    value.as_array().unwrap()
}
fn verify(path: &Path, expected: &Json) -> u64 {
    let (bytes, hash) = digest::hash_reader(File::open(path).unwrap()).unwrap();
    assert_eq!(
        bytes,
        integer(field(expected, "bytes")),
        "length: {}",
        path.display()
    );
    assert_eq!(
        hash,
        text(field(expected, "sha256")),
        "SHA-256: {}",
        path.display()
    );
    bytes
}
fn run(input: &Path) {
    let name = "bloom-560m";
    let output = PathBuf::from(
        std::env::var_os("LLAMA_TEST_BLOOM_OUTPUT_ROOT")
            .expect("set LLAMA_TEST_BLOOM_OUTPUT_ROOT to a fresh external output root"),
    )
    .join(name);
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::create_dir(&output).expect("preserve prior runs: choose a fresh output root");
    let manifest = json::parse(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/bloom-conversion/pretrained-reference.json"),
        )
        .unwrap(),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    let model = field(field(&manifest, "profiles"), profile);
    for file in array(field(model, "inputs")) {
        verify(&input.join(text(field(file, "path"))), file);
    }
    let outputs = array(field(model, "outputs"));
    assert_eq!(outputs.len(), 3);
    let (mut total, mut accepted, mut rejected) = (0, 0, 0);
    for reference in outputs {
        let mode = text(field(reference, "mode"));
        let target = output.join(format!("{mode}.gguf"));
        let result = Conversion::prepare(
            input,
            Options {
                output_type: OutputType::parse(mode).unwrap(),
                source_profile: Some(SourceProfile::current()),
                ..Options::default()
            },
            Limits::default(),
        )
        .and_then(|conversion| {
            conversion.write_file(&target)?;
            Ok(conversion.summary().bytes)
        });
        if text(field(reference, "status")) == "rejected" {
            assert!(
                result.is_err(),
                "{name}/{mode}: original rejected the pinned input"
            );
            assert!(
                !target.exists(),
                "{name}/{mode}: failed conversion published a file"
            );
            rejected += 1;
            println!("{profile} {name} {mode}: original rejection reproduced");
        } else {
            assert_eq!(
                result.unwrap_or_else(|e| panic!("{name}/{mode}: {e}")),
                integer(field(reference, "bytes"))
            );
            total += verify(&target, reference);
            accepted += 1;
            println!(
                "{profile} {name} {mode}: {} exact bytes, SHA-256 {}",
                integer(field(reference, "bytes")),
                text(field(reference, "sha256"))
            );
        }
    }
    assert_eq!((accepted, rejected, total), (3, 0, 2_871_136_896));
    for file in array(field(model, "inputs")) {
        verify(&input.join(text(field(file, "path"))), file);
    }
    println!(
        "{profile} {name}: {accepted} complete files, {rejected} rejections, {total} exact bytes"
    );
}

#[test]
#[ignore = "requires unchanged pinned official BLOOM checkpoint and a fresh output root; run in release mode"]
fn pretrained_bloom_native_files() {
    let input = PathBuf::from(
        std::env::var_os("LLAMA_TEST_BLOOM_HF")
            .expect("set LLAMA_TEST_BLOOM_HF to the unchanged checksum-pinned official checkpoint"),
    );
    run(&input);
}
