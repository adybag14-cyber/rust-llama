//! Complete-file equality with the frozen converter for the unchanged,
//! revision-pinned bigcode/tiny_starcoder_py pretrained checkpoint.
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
fn verify(path: &Path, expected: &Json) -> u64 {
    let (bytes, hash) = digest::hash_reader(File::open(path).unwrap()).unwrap();
    assert_eq!(
        bytes,
        integer(field(expected, "bytes")),
        "{}",
        path.display()
    );
    assert_eq!(hash, text(field(expected, "sha256")), "{}", path.display());
    bytes
}

#[test]
#[ignore = "requires unchanged pinned tiny_starcoder_py checkpoint and a fresh output root; run in release mode"]
fn pretrained_starcoder_native_files() {
    let input = PathBuf::from(
        std::env::var_os("LLAMA_TEST_STARCODER_HF")
            .expect("set LLAMA_TEST_STARCODER_HF to the checksum-pinned official checkpoint"),
    );
    let output = PathBuf::from(
        std::env::var_os("LLAMA_TEST_STARCODER_OUTPUT_ROOT")
            .expect("set LLAMA_TEST_STARCODER_OUTPUT_ROOT to a fresh external output root"),
    )
    .join("tiny_starcoder_py");
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::create_dir(&output).expect("preserve prior runs: choose a fresh output root");
    let manifest = json::parse(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/starcoder-conversion/pretrained-reference.json"),
        )
        .unwrap(),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    let model = field(field(&manifest, "profiles"), profile);
    let inputs = field(model, "inputs").as_array().unwrap();
    for file in inputs {
        verify(&input.join(text(field(file, "path"))), file);
    }
    let outputs = field(model, "outputs").as_array().unwrap();
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
                model_name: Some("tiny_starcoder_py".to_owned()),
                ..Options::default()
            },
            Limits::default(),
        )
        .unwrap_or_else(|e| panic!("{profile}/{mode}: {e}"));
        conversion.write_file(&target).unwrap();
        assert_eq!(
            conversion.summary().bytes,
            integer(field(reference, "bytes"))
        );
        total += verify(&target, reference);
        println!(
            "{profile} tiny_starcoder_py {mode}: {} exact bytes, SHA-256 {}",
            integer(field(reference, "bytes")),
            text(field(reference, "sha256"))
        );
    }
    assert_eq!(total, 881_027_328);
    for file in inputs {
        verify(&input.join(text(field(file, "path"))), file);
    }
    println!(
        "{profile} tiny_starcoder_py: three exact complete files, {total} bytes; inputs unchanged"
    );
}
