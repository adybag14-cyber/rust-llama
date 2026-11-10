//! Full original-converter files are ordinary offline regression inputs.
//! The binary pack only deduplicates identical bytes; see pipeline-provenance.json.
use llama_checkpoint::{Value as Json, json};
use llama_conformance::digest;
use llama_convert::{Conversion, Limits, Options, OutputType, codec::SourceProfile};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{fs, process::Command};

fn field<'a>(value: &'a Json, name: &str) -> &'a Json {
    value.as_object().unwrap().get(name).unwrap()
}
fn text(value: &Json) -> &str {
    value.as_text().unwrap().as_str().unwrap()
}
fn number(value: &Json) -> usize {
    let Json::Number(n) = value else {
        panic!("expected integer")
    };
    usize::try_from(n.as_integer().unwrap().as_u64().unwrap()).unwrap()
}
fn array(value: &Json) -> &[Json] {
    value.as_array().unwrap()
}
struct Corpus {
    manifest: Json,
    payload: Vec<u8>,
}
impl Corpus {
    fn read() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dense-conversion");
        let manifest = json::parse(
            &fs::read(root.join("pipeline.json")).unwrap(),
            llama_checkpoint::Limits::default(),
        )
        .unwrap();
        let payload = fs::read(root.join("pipeline.bin")).unwrap();
        let identity = field(&manifest, "payload");
        assert_eq!(payload.len(), number(field(identity, "bytes")));
        assert_eq!(digest::sha256(&payload), text(field(identity, "sha256")));
        Self { manifest, payload }
    }
    fn bytes<'a>(&'a self, blob: &Json) -> &'a [u8] {
        let offset = number(field(blob, "offset"));
        let length = number(field(blob, "bytes"));
        let raw = &self.payload[offset..offset.checked_add(length).unwrap()];
        assert_eq!(digest::sha256(raw), text(field(blob, "sha256")));
        raw
    }
    fn cases(&self, host: &str) -> &[Json] {
        array(field(field(&self.manifest, "profiles"), host))
    }
    fn base(&self) -> &Json {
        self.cases(if cfg!(windows) { "windows" } else { "linux" })
            .iter()
            .find(|c| text(field(c, "name")) == "base-f32")
            .unwrap()
    }
    fn materialize(&self, case: &Json, folder: &Path) {
        fs::create_dir_all(folder).unwrap();
        for file in array(field(case, "input_files")) {
            let name = text(field(file, "path"));
            assert!(!name.contains(['\\', ':']) && !name.split('/').any(|p| p == ".."));
            let path = folder.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, self.bytes(file)).unwrap();
        }
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rust-llama-converter-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn settings(case: &Json, profile: SourceProfile) -> Options {
    Options {
        output_type: OutputType::parse(text(field(case, "ftype"))).unwrap(),
        source_profile: Some(profile),
        model_name: match field(case, "model_name") {
            Json::Null => None,
            other => Some(text(other).to_owned()),
        },
        vocab_only: matches!(field(case, "vocab_only"), Json::Bool(true)),
        fuse_qkv: matches!(field(case, "fuse_qkv"), Json::Bool(true)),
        ..Options::default()
    }
}
#[test]
fn complete_files_match_original_converter_for_native_source_profile() {
    let corpus = Corpus::read();
    let temp = Temp::new();
    let (host, profile) = if cfg!(windows) {
        ("windows", SourceProfile::WindowsTorch211Numpy246)
    } else {
        ("linux", SourceProfile::LinuxTorch211Numpy226)
    };
    {
        let (mut accepted, mut rejected, mut bytes) = (0, 0, 0);
        for case in corpus.cases(host) {
            let name = text(field(case, "name"));
            let input = temp.0.join(host).join(name).join("model");
            corpus.materialize(case, &input);
            let result = Conversion::prepare(&input, settings(case, profile), Limits::default())
                .and_then(|conversion| {
                    let mut output = Vec::new();
                    conversion.write_to(&mut output)?;
                    assert_eq!(conversion.summary().bytes, output.len() as u64);
                    Ok(output)
                });
            match (text(field(case, "status")), result) {
                ("accepted", Ok(actual)) => {
                    let expected = corpus.bytes(field(case, "output"));
                    if actual != expected {
                        let first = actual.iter().zip(expected).position(|(a, b)| a != b);
                        panic!(
                            "{host}/{name}: actual {} bytes, expected {} bytes; first difference {first:?}",
                            actual.len(),
                            expected.len()
                        );
                    }
                    accepted += 1;
                    bytes += actual.len();
                }
                ("rejected", Err(_)) => rejected += 1,
                ("accepted", Err(error)) => panic!("{host}/{name}: unexpectedly rejected: {error}"),
                ("rejected", Ok(_)) => panic!("{host}/{name}: unexpectedly accepted"),
                _ => panic!("invalid fixture status"),
            }
        }
        assert_eq!((accepted, rejected, bytes), (57, 7, 8_091_032), "{host}");
    }
}
#[test]
fn atomic_publication_preserves_destination_and_source_on_failure() {
    let corpus = Corpus::read();
    let case = corpus.base();
    let temp = Temp::new();
    let input = temp.0.join("model");
    corpus.materialize(case, &input);
    let options = settings(case, SourceProfile::current());
    let destination = temp.0.join("existing.gguf");
    let sentinel = b"pre-existing output must survive failed publication";
    fs::write(&destination, sentinel).unwrap();
    let conversion = Conversion::prepare(
        &input,
        options.clone(),
        Limits {
            max_working_bytes: 64 * 1024,
            ..Limits::default()
        },
    )
    .unwrap();
    assert!(conversion.write_file(&destination).is_err());
    assert_eq!(fs::read(&destination).unwrap(), sentinel);
    assert!(fs::read_dir(&temp.0).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".rust-llama-convert-")
    }));
    drop(conversion);

    let conversion = Conversion::prepare(&input, options.clone(), Limits::default()).unwrap();
    for file in ["config.json", "model.safetensors", "tokenizer.model"] {
        let source = input.join(file);
        let before = fs::read(&source).unwrap();
        assert!(conversion.write_file(&source).is_err(), "{file}");
        assert_eq!(fs::read(source).unwrap(), before, "{file}");
    }
    assert!(
        Conversion::prepare(
            &input,
            options,
            Limits {
                max_output_bytes: 1024,
                ..Limits::default()
            }
        )
        .is_err()
    );
    conversion.write_file(&destination).unwrap();
    assert_eq!(
        fs::read(&destination).unwrap(),
        corpus.bytes(field(case, "output"))
    );
}
#[test]
fn command_line_converts_exact_bytes_and_dry_run_does_not_publish() {
    let corpus = Corpus::read();
    let case = corpus.base();
    let temp = Temp::new();
    let input = temp.0.join("model");
    corpus.materialize(case, &input);
    let output = temp.0.join("cli.gguf");
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_convert-hf-to-gguf"))
            .arg(&input)
            .arg("--outfile")
            .arg(&output)
            .args([
                "--outtype=f32",
                "--model-name",
                "Tiny Llama Converter Fixture",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let dry = run(&["--dry-run"]);
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert!(!output.exists());
    let result = run(&[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read(&output).unwrap(),
        corpus.bytes(field(case, "output"))
    );
    let before = fs::read(&output).unwrap();
    let bad = run(&["--max-working-mib=0"]);
    assert!(!bad.status.success());
    assert_eq!(fs::read(&output).unwrap(), before);
    let help = Command::new(env!("CARGO_BIN_EXE_convert-hf-to-gguf"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(
        String::from_utf8(help.stdout)
            .unwrap()
            .contains("LlamaForCausalLM")
    );
}
