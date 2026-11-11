//! Complete-file differential tests from the frozen Qwen converters.
use llama_checkpoint::{Value as Json, json};
use llama_conformance::digest;
use llama_convert::{Conversion, Limits, Options, OutputType, codec::SourceProfile};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    value.as_object().unwrap().get(key).unwrap()
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
    bytes: Vec<u8>,
}
impl Corpus {
    fn read(cohort: &str) -> Self {
        let (folder, stem) = if cohort == "config-type" {
            ("qwen-config-type", "cohort")
        } else {
            ("qwen-conversion", cohort)
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(folder);
        let manifest = json::parse(
            &fs::read(root.join(format!("{stem}.json"))).unwrap(),
            llama_checkpoint::Limits::default(),
        )
        .unwrap();
        let bytes = fs::read(root.join(format!("{stem}.bin"))).unwrap();
        assert_eq!(
            bytes.len(),
            number(field(field(&manifest, "payload"), "bytes"))
        );
        assert_eq!(
            digest::sha256(&bytes),
            text(field(field(&manifest, "payload"), "sha256"))
        );
        Self { manifest, bytes }
    }
    fn blob<'a>(&'a self, value: &Json) -> &'a [u8] {
        let offset = number(field(value, "offset"));
        let length = number(field(value, "bytes"));
        let raw = &self.bytes[offset..offset.checked_add(length).unwrap()];
        assert_eq!(digest::sha256(raw), text(field(value, "sha256")));
        raw
    }
    fn materialize(&self, case: &Json, dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        for file in array(field(case, "input_files")) {
            let name = text(field(file, "path"));
            assert!(!name.contains(['\\', ':']) && !name.split('/').any(|p| p == ".."));
            let path = dir.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, self.blob(file)).unwrap();
        }
    }
    fn cases(&self) -> &[Json] {
        array(field(
            field(&self.manifest, "profiles"),
            if cfg!(windows) { "windows" } else { "linux" },
        ))
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "rust-llama-qwen-convert-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn options(case: &Json) -> Options {
    Options {
        output_type: OutputType::parse(text(field(case, "ftype"))).unwrap(),
        source_profile: Some(SourceProfile::current()),
        model_name: match field(case, "model_name") {
            Json::Null => None,
            v => Some(text(v).to_owned()),
        },
        fuse_qkv: matches!(field(case, "fuse_qkv"), Json::Bool(true)),
        vocab_only: matches!(field(case, "vocab_only"), Json::Bool(true)),
        ..Options::default()
    }
}
fn compare_native_files(cohort: &str, expected_totals: (usize, usize, usize)) {
    let corpus = Corpus::read(cohort);
    let temp = Temp::new();
    let actual_dir = std::env::var_os("LLAMA_QWEN_ACTUAL_DIR").map(PathBuf::from);
    if let Some(dir) = &actual_dir {
        fs::create_dir_all(dir).unwrap();
    }
    let mut failures = Vec::new();
    let (mut accepted, mut rejected, mut bytes) = (0, 0, 0);
    assert_eq!(corpus.cases().len(), expected_totals.0 + expected_totals.1);
    for case in corpus.cases() {
        let name = text(field(case, "name"));
        let input = temp.0.join(name).join("model");
        corpus.materialize(case, &input);
        let result =
            Conversion::prepare(&input, options(case), Limits::default()).and_then(|conversion| {
                let mut output = Vec::new();
                conversion.write_to(&mut output)?;
                assert_eq!(conversion.summary().bytes, output.len() as u64);
                Ok(output)
            });
        match (text(field(case, "status")), result) {
            ("accepted", Ok(actual)) => {
                let expected = corpus.blob(field(case, "output"));
                if actual == expected {
                    accepted += 1;
                    bytes += actual.len();
                } else {
                    let first = actual.iter().zip(expected).position(|(a, b)| a != b);
                    failures.push(format!(
                        "{name}: {} versus {} bytes; first {first:?}",
                        actual.len(),
                        expected.len()
                    ));
                    if let Some(dir) = &actual_dir {
                        fs::write(dir.join(format!("{name}.gguf")), actual).unwrap();
                    }
                }
            }
            ("rejected", Err(_)) => rejected += 1,
            ("accepted", Err(e)) => failures.push(format!("{name}: rejected: {e}")),
            ("rejected", Ok(_)) => failures.push(format!("{name}: unexpectedly accepted")),
            _ => panic!("invalid captured status"),
        }
    }
    println!(
        "Qwen complete converter: {accepted} accepted, {rejected} rejected, {bytes} exact bytes"
    );
    assert!(failures.is_empty(), "{}\n", failures.join("\n"));
    assert_eq!((accepted, rejected, bytes), expected_totals);
}
#[test]
fn qwen2_and_qwen3_complete_native_files() {
    compare_native_files("pipeline", (104, 8, 25_723_552));
}
#[test]
fn qwen_autoconfig_fallbacks_match_original_files() {
    compare_native_files("edges", (32, 0, 5_318_912));
}
#[test]
fn qwen_model_type_presence_matches_original_files() {
    compare_native_files("config-type", (8, 0, 1_329_824));
}
#[test]
fn qwen_publication_limits_and_sidecar_paths() {
    let corpus = Corpus::read("pipeline");
    let temp = Temp::new();
    for arch in ["qwen2", "qwen3"] {
        let name = format!("{arch}-base-f32");
        let case = corpus
            .cases()
            .iter()
            .find(|c| text(field(c, "name")) == name)
            .unwrap();
        let input = temp.0.join(arch);
        corpus.materialize(case, &input);
        let output = temp.0.join(format!("{arch}.gguf"));
        fs::write(&output, b"existing complete file").unwrap();
        let conversion = Conversion::prepare(
            &input,
            options(case),
            Limits {
                max_working_bytes: 16,
                ..Limits::default()
            },
        )
        .unwrap();
        assert!(conversion.write_file(&output).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"existing complete file");
        assert!(
            Conversion::prepare(
                &input,
                options(case),
                Limits {
                    max_output_bytes: 64,
                    ..Limits::default()
                }
            )
            .is_err()
        );
        let conversion = Conversion::prepare(&input, options(case), Limits::default()).unwrap();
        conversion.write_file(&output).unwrap();
        assert_eq!(
            fs::read(&output).unwrap(),
            corpus.blob(field(case, "output"))
        );
        fs::write(
            input.join("modules.json"),
            br#"[{"type":"Pooling","path":"../outside"}]"#,
        )
        .unwrap();
        assert!(Conversion::prepare(&input, options(case), Limits::default()).is_err());
        assert_eq!(
            fs::read(&output).unwrap(),
            corpus.blob(field(case, "output"))
        );
    }
}

#[test]
fn specialized_qwen_variants_do_not_silently_become_dense_models() {
    let corpus = Corpus::read("pipeline");
    let temp = Temp::new();
    for arch in ["qwen2", "qwen3"] {
        let name = format!("{arch}-base-f32");
        let case = corpus
            .cases()
            .iter()
            .find(|c| text(field(c, "name")) == name)
            .unwrap();
        let input = temp.0.join(arch);
        corpus.materialize(case, &input);
        let original = fs::read(input.join("config.json")).unwrap();
        let original = std::str::from_utf8(&original).unwrap();
        if arch == "qwen3" {
            for origin in [
                "InternS1ForConditionalGeneration",
                "Qwen3ForSequenceClassification",
                "QWEN3-RERANKER-WRAPPER",
            ] {
                let wrapper = format!(
                    r#"{{"architectures":["{origin}"],"model_type":"unregistered_wrapper","text_config":{original}}}"#
                );
                fs::write(input.join("config.json"), wrapper).unwrap();
                assert!(
                    matches!(
                        Conversion::prepare(&input, options(case), Limits::default()),
                        Err(llama_convert::Error::Unsupported(_))
                    ),
                    "{origin}"
                );
            }
            fs::write(input.join("config.json"), original).unwrap();
            fs::write(input.join("README.md"), "# QWEN3-RERANKER").unwrap();
            assert!(matches!(
                Conversion::prepare(&input, options(case), Limits::default()),
                Err(llama_convert::Error::Unsupported(_))
            ));
            fs::remove_file(input.join("README.md")).unwrap();
        }
        fs::write(
            input.join("tokenizer_config.json"),
            r#"{"tokenizer_class":"HybridDNATokenizer"}"#,
        )
        .unwrap();
        assert!(matches!(
            Conversion::prepare(&input, options(case), Limits::default()),
            Err(llama_convert::Error::Unsupported(_))
        ));
    }
}
