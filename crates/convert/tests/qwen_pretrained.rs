//! Full pretrained files captured independently from the frozen converter.
//! Model weights remain in the external, checksum-pinned cache.
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
fn number(value: &Json) -> u64 {
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
        number(field(expected, "bytes")),
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
fn run(name: &str) {
    let root = PathBuf::from(std::env::var_os("LLAMA_TEST_QWEN_HF_ROOT").expect(
        "set LLAMA_TEST_QWEN_HF_ROOT to the external directory containing the pinned Qwen checkpoints"
    ));
    let output = PathBuf::from(std::env::var_os("LLAMA_TEST_QWEN_OUTPUT_ROOT").expect(
        "set LLAMA_TEST_QWEN_OUTPUT_ROOT to an external directory with no prior model output subdirectory"
    )).join(name);
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::create_dir(&output)
        .expect("pretrained comparisons preserve prior runs: use a fresh output root");
    let manifest = json::parse(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/qwen-conversion/pretrained-reference.json"),
        )
        .unwrap(),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    let model = array(field(&manifest, "models"))
        .iter()
        .find(|m| text(field(m, "name")) == name)
        .unwrap();
    let input = root.join(name);
    for file in array(field(model, "inputs")) {
        verify(&input.join(text(field(file, "path"))), file);
    }
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    let outputs = array(field(field(model, "outputs"), profile));
    assert_eq!(outputs.len(), 3);
    let mut total = 0;
    for reference in outputs {
        let mode = text(field(reference, "mode"));
        let target = output.join(format!("{mode}.gguf"));
        let conversion = Conversion::prepare(
            &input,
            Options {
                output_type: OutputType::parse(mode).unwrap(),
                source_profile: Some(SourceProfile::current()),
                ..Options::default()
            },
            Limits::default(),
        )
        .unwrap_or_else(|e| panic!("{name}/{mode}: {e}"));
        assert_eq!(
            conversion.summary().bytes,
            number(field(reference, "bytes"))
        );
        conversion
            .write_file(&target)
            .unwrap_or_else(|e| panic!("{name}/{mode}: {e}"));
        total += verify(&target, reference);
        println!(
            "{profile} {name} {mode}: {} exact bytes, SHA-256 {}",
            number(field(reference, "bytes")),
            text(field(reference, "sha256"))
        );
    }
    println!("{profile} {name}: 3 full pretrained outputs, {total} exact bytes");
}
#[test]
#[ignore = "requires pinned external Qwen checkpoint and fresh output root; run in release mode"]
fn pretrained_qwen2_native_files() {
    run("Qwen2.5-0.5B-Instruct");
}
#[test]
#[ignore = "requires pinned external Qwen checkpoint and fresh output root; run in release mode"]
fn pretrained_qwen3_native_files() {
    run("Qwen3-0.6B");
}
