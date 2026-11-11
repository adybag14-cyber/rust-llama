//! Actual pinned pretrained HF checkpoints and the frozen GPT-2 buffer rejection.
//! The derived GPT-2 fixture differs only by removal of causal-mask buffers;
//! all learned tensor names, shapes, dtypes and bytes remain unchanged.
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
fn run(name: &str, input: &Path) {
    let output = PathBuf::from(
        std::env::var_os("LLAMA_TEST_GPT_OUTPUT_ROOT")
            .expect("set LLAMA_TEST_GPT_OUTPUT_ROOT to a fresh external output root"),
    )
    .join(name);
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::create_dir(&output).expect("preserve prior runs: choose a fresh output root");
    let manifest = json::parse(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gpt-conversion/pretrained-reference.json"),
        )
        .unwrap(),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    let model = array(field(&manifest, "models"))
        .iter()
        .find(|m| text(field(m, "name")) == name)
        .unwrap();
    for file in array(field(model, "inputs")) {
        verify(&input.join(text(field(file, "path"))), file);
    }
    let profile = if cfg!(windows) { "windows" } else { "linux" };
    let outputs = array(field(field(model, "outputs"), profile));
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
    if name == "gpt2-original" {
        assert_eq!((accepted, rejected), (0, 3));
    } else {
        assert_eq!((accepted, rejected), (3, 0));
    }
    for file in array(field(model, "inputs")) {
        verify(&input.join(text(field(file, "path"))), file);
    }
    println!(
        "{profile} {name}: {accepted} complete files, {rejected} rejections, {total} exact bytes"
    );
}
#[test]
#[ignore = "requires pinned official GPT-2 input and fresh output root; prepares exact filtered fixture in Rust"]
fn pretrained_gpt2_native_files_and_original_rejection() {
    let original = input_from_env("LLAMA_TEST_GPT2_HF");
    run("gpt2-original", &original);
    let filtered = std::env::var_os("LLAMA_TEST_GPT2_FILTERED_HF")
        .map(PathBuf::from)
        .unwrap_or_else(|| prepare_gpt2_fixture(&original));
    run("gpt2", &filtered);
}
#[test]
#[ignore = "requires pinned external Pythia checkpoint and fresh output root; run in release mode"]
fn pretrained_gptneox_native_files() {
    run("gptneox", &input_from_env("LLAMA_TEST_GPTNEOX_HF"));
}

fn input_from_env(variable: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(variable).unwrap_or_else(|| {
            panic!("set {variable} to the unchanged checksum-pinned checkpoint")
        }),
    )
}

/// Test-data preparation for the one pinned official checkpoint. This is not
/// a product converter option. It recreates the independently captured exact
/// Safetensors derivative using retained source ranges and no external runtime.
fn prepare_gpt2_fixture(original: &Path) -> PathBuf {
    use std::io::{BufWriter, Write};
    let manifest = json::parse(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gpt-conversion/pretrained-reference.json"),
        )
        .unwrap(),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    let source = array(field(&manifest, "models"))
        .iter()
        .find(|m| text(field(m, "name")) == "gpt2-original")
        .unwrap();
    let expected = array(field(&manifest, "models"))
        .iter()
        .find(|m| text(field(m, "name")) == "gpt2")
        .unwrap();
    for file in array(field(source, "inputs")) {
        verify(&original.join(text(field(file, "path"))), file);
    }
    let output_root = input_from_env("LLAMA_TEST_GPT_OUTPUT_ROOT");
    let target = output_root
        .join("_inputs")
        .join("gpt2-without-causal-mask-buffers");
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::create_dir(&target).expect("use a fresh output root for the exact fixture derivative");
    let checkpoint = llama_checkpoint::SafetensorsFile::open(
        original.join("model.safetensors"),
        llama_checkpoint::Limits::default(),
    )
    .unwrap();
    assert_eq!(text(field(checkpoint.metadata().unwrap(), "format")), "pt");
    assert_eq!(checkpoint.tensors().len(), 160);
    let removed: Vec<_> = (0..12).map(|i| format!("h.{i}.attn.bias")).collect();
    let mut removed_count = 0;
    let mut kept = Vec::new();
    let mut header = String::from(r#"{"__metadata__":{"format":"pt"}"#);
    let mut offset = 0u64;
    for (index, tensor) in checkpoint.tensors().iter().enumerate() {
        let name = tensor.name().as_str().unwrap();
        let layout = tensor.typed_layout().unwrap();
        assert_eq!(layout.dtype, llama_checkpoint::DType::F32, "{name}");
        if removed.iter().any(|n| n == name) {
            assert_eq!(layout.shape, [1, 1, 1024, 1024], "{name}");
            removed_count += 1;
            continue;
        }
        // All names and dimensions come from the hash-verified pinned input.
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_')
        );
        let shape = layout
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let end = offset.checked_add(layout.bytes).unwrap();
        header.push_str(&format!(
            r#",{name:?}:{{"dtype":"F32","shape":[{shape}],"data_offsets":[{offset},{end}]}}"#
        ));
        offset = end;
        kept.push(index);
    }
    assert_eq!((removed_count, kept.len()), (12, 148));
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let path = target.join("model.safetensors");
    let mut file = BufWriter::new(File::create_new(&path).unwrap());
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for index in kept {
        std::io::copy(&mut checkpoint.reader(index).unwrap(), &mut file).unwrap();
    }
    file.flush().unwrap();
    drop(file);
    for descriptor in array(field(expected, "inputs")) {
        let name = text(field(descriptor, "path"));
        if name != "model.safetensors" {
            fs::copy(original.join(name), target.join(name)).unwrap();
        }
        verify(&target.join(name), descriptor);
    }
    for descriptor in array(field(source, "inputs")) {
        verify(&original.join(text(field(descriptor, "path"))), descriptor);
    }
    println!(
        "Prepared exact pinned GPT-2 test input in Rust: twelve masks removed, 148 tensors retained"
    );
    target
}
