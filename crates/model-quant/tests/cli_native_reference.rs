//! Independent original-program captures precede every Rust CLI implementation.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use llama_model_quant::{
    Error, OverrideValue, QuantizeLimits, cli,
    imatrix::{ImatrixLimits, PreparedImatrix, prepare_imatrix},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn text(v: &Value) -> &str {
    v.as_str().unwrap()
}
fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn number(v: &Value) -> i64 {
    v.as_i64().unwrap()
}
fn boolean(v: &Value) -> bool {
    v.as_bool().unwrap()
}
fn profile() -> &'static str {
    if cfg!(windows) {
        "windows-msvc"
    } else {
        "linux-gnu"
    }
}
fn fixture(name: &str) -> Value {
    json::parse(
        &fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/cli")
                .join(name),
        )
        .unwrap(),
    )
    .unwrap()
}
fn cohort(name: &str) -> Value {
    fixture(&format!("{name}-{}.json", profile()))
}
static TEMP: AtomicUsize = AtomicUsize::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "llama-quant-cli-{}-{}",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| {
            let digit = |b: u8| (b as char).to_digit(16).unwrap() as u8;
            digit(v[0]) * 16 + digit(v[1])
        })
        .collect()
}
fn requests(v: &Value, key: &str) -> Vec<String> {
    v.get(key).map_or_else(Vec::new, |v| {
        array(v).iter().map(|v| text(v).to_owned()).collect()
    })
}
#[derive(Debug, PartialEq)]
enum ScalarValue {
    Integer(i64),
    Float(u64),
    Bool(bool),
    String(Vec<u8>),
}
#[derive(Debug, PartialEq)]
enum Observation {
    Ftype(u32, String),
    Dtype(u32),
    Tensor(Vec<u8>, u32),
    Layers(Vec<i32>),
    Kv(Vec<u8>, ScalarValue),
    Imatrix(i32, Vec<Vec<u8>>, BTreeMap<String, Vec<u32>>),
}
fn matrix_observation(matrix: PreparedImatrix) -> Observation {
    Observation::Imatrix(
        matrix.chunk_count,
        matrix.datasets,
        matrix
            .entries
            .into_iter()
            .map(|(name, v)| (name, v.into_iter().map(f32::to_bits).collect()))
            .collect(),
    )
}
fn rust_helper(case: &Value, dir: &Path) -> llama_model_quant::Result<Observation> {
    let request = field(case, "request");
    let scalar = request.get("text").map_or("", text);
    Ok(match text(field(request, "mode")) {
        "ftype" => {
            let (value, name) = cli::parse_file_type(scalar)?;
            Observation::Ftype(value.id(), name.into())
        }
        "dtype" => Observation::Dtype(cli::parse_dtype(scalar)?.id()),
        "tensor" => {
            let value = cli::parse_tensor_type(scalar)?;
            Observation::Tensor(value.pattern, value.dtype.id())
        }
        "layers" => Observation::Layers(cli::parse_prune_layers(scalar)?),
        "kv" => {
            let value = cli::parse_metadata_override(scalar)?;
            let data = match value.value {
                OverrideValue::Integer(n) => ScalarValue::Integer(n),
                OverrideValue::Float(n) => ScalarValue::Float(n.to_bits()),
                OverrideValue::Bool(n) => ScalarValue::Bool(n),
                OverrideValue::String(n) => ScalarValue::String(n),
            };
            Observation::Kv(value.key, data)
        }
        "imatrix" => {
            if let Some(file) = case.get("file") {
                let bytes = unhex(text(field(file, "hex")));
                assert_eq!(sha256(&bytes), text(field(file, "sha256")));
                fs::write(dir.join(text(field(file, "path"))), bytes)?;
            }
            let name = text(field(request, "file"));
            let path = if name.is_empty() {
                PathBuf::new()
            } else {
                dir.join(name)
            };
            matrix_observation(prepare_imatrix(
                path,
                &requests(request, "include"),
                &requests(request, "exclude"),
                &ImatrixLimits::default(),
            )?)
        }
        mode => panic!("unknown native helper {mode}"),
    })
}
fn original_helper(case: &Value) -> Option<Observation> {
    if case.get("rust_boundary").is_some() {
        return None;
    }
    let result = case.get("result")?;
    if result.get("accepted").is_some_and(|v| !boolean(v)) {
        return None;
    }
    Some(match text(field(field(case, "request"), "mode")) {
        "ftype" => Observation::Ftype(
            number(field(result, "id")) as u32,
            text(field(result, "name")).into(),
        ),
        "dtype" => Observation::Dtype(number(field(result, "id")) as u32),
        "tensor" => {
            let v = &array(field(result, "options"))[0];
            Observation::Tensor(
                text(field(v, "name")).as_bytes().to_vec(),
                number(field(v, "id")) as u32,
            )
        }
        "layers" => Observation::Layers(
            array(field(result, "layers"))
                .iter()
                .map(|v| number(v) as i32)
                .collect(),
        ),
        "kv" => {
            let v = &array(field(result, "options"))[0];
            let data = match number(field(v, "tag")) {
                0 => ScalarValue::Integer(number(field(v, "value"))),
                1 => {
                    let bits = case.get("rust_correction").map_or_else(
                        || text(field(v, "bits")),
                        |c| text(field(c, "correct_f64_bits")),
                    );
                    ScalarValue::Float(u64::from_str_radix(bits, 16).unwrap())
                }
                2 => ScalarValue::Bool(boolean(field(v, "value"))),
                3 => ScalarValue::String(text(field(v, "value")).as_bytes().to_vec()),
                tag => panic!("unknown override tag {tag}"),
            };
            Observation::Kv(text(field(v, "key")).as_bytes().to_vec(), data)
        }
        "imatrix" => Observation::Imatrix(
            number(field(result, "chunks")) as i32,
            array(field(result, "datasets"))
                .iter()
                .map(|v| text(v).as_bytes().to_vec())
                .collect(),
            array(field(result, "entries"))
                .iter()
                .map(|v| {
                    (
                        text(field(v, "name")).to_owned(),
                        array(field(v, "bits"))
                            .iter()
                            .map(|n| number(n) as u32)
                            .collect(),
                    )
                })
                .collect(),
        ),
        mode => panic!("unknown helper mode {mode}"),
    })
}
#[test]
fn original_scalar_and_imatrix_observations() {
    let data = cohort("helper");
    let cases = array(field(&data, "cases"));
    assert_eq!(cases.len(), 517);
    let mut failures = Vec::new();
    let mut accepted = 0;
    let mut checked = 0;
    let mut corrections = 0;
    for case in cases {
        let dir = Directory::new();
        let expected = original_helper(case);
        let actual = rust_helper(case, &dir.0);
        if case.get("rust_boundary").is_some() {
            checked += 1;
        }
        if case.get("rust_correction").is_some() {
            corrections += 1;
        }
        match (expected, actual) {
            (Some(want), Ok(got)) if want == got => accepted += 1,
            (None, Err(_)) => {}
            (want, got) => failures.push(format!(
                "{} {}: expected {want:?}, got {got:?}",
                text(field(case, "id")),
                text(field(case, "name"))
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!(
        "517 original helper cases: {accepted} accepted exact results, {checked} checked-domain exclusions, {corrections} explicitly reviewed private F64 corrections"
    );
}
fn blobs() -> BTreeMap<String, &'static [u8]> {
    let packed = include_bytes!("fixtures/cli/files.bin");
    let mut previous = 0;
    let mut result = BTreeMap::new();
    for value in array(&fixture("files.json")) {
        let start = number(field(value, "offset")) as usize;
        let size = number(field(value, "bytes")) as usize;
        let hash = text(field(value, "sha256"));
        assert_eq!(start, previous);
        let bytes = &packed[start..start + size];
        assert_eq!(sha256(bytes), hash);
        assert!(result.insert(hash.to_owned(), bytes).is_none());
        previous += size;
    }
    assert_eq!(previous, packed.len());
    assert_eq!(result.len(), 99);
    result
}
fn write_inputs(case: &Value, dir: &Path, blobs: &BTreeMap<String, &[u8]>) {
    for input in array(field(case, "inputs")) {
        let path = dir.join(text(field(input, "path")));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = blobs[text(field(input, "sha256"))];
        assert_eq!(bytes.len(), number(field(input, "bytes")) as usize);
        fs::write(path, bytes).unwrap();
    }
}
fn output_files(dir: &Path, root: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            output_files(&entry.path(), root, out);
        } else {
            let name = entry
                .path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(name, fs::read(entry.path()).unwrap());
        }
    }
}
fn symlink(input: &Path, alias: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(input, alias)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(input, alias)
    }
}
fn native_cli_cases() -> (usize, usize) {
    let data = cohort("executable");
    let cases = array(field(&data, "cases"));
    assert_eq!(cases.len(), 188);
    let blobs = blobs();
    let mut failures = Vec::new();
    let mut exact_files = 0;
    let mut exact_bytes = 0;
    #[cfg(windows)]
    let mut symlink_skips = 0;
    #[cfg(not(windows))]
    let symlink_skips = 0;
    for case in cases {
        let id = text(field(case, "id"));
        let dir = Directory::new();
        write_inputs(case, &dir.0, &blobs);
        if let Some(setup) = field(case, "setup").as_str() {
            let input = dir.0.join("model.gguf");
            let alias = dir.0.join("alias.gguf");
            fs::remove_file(&alias).unwrap();
            match setup {
                "hardlink" => fs::hard_link(input, alias).unwrap(),
                "symlink" => {
                    if let Err(error) = symlink(&input, &alias) {
                        #[cfg(windows)]
                        if error.raw_os_error() == Some(1314) {
                            symlink_skips += 1;
                            println!(
                                "{id}: Windows symlink privilege unavailable; canonical and hard-link cases remain required"
                            );
                            continue;
                        }
                        panic!("cannot reproduce native symlink case: {error}");
                    }
                }
                value => panic!("unknown setup {value}"),
            }
        }
        let args = requests(case, "argv");
        let output = Command::new(env!("CARGO_BIN_EXE_llama-quantize"))
            .args(&args)
            .current_dir(&dir.0)
            .env_remove("LLAMA_TRACE")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let bounded = case.get("rust_boundary").is_some();
        let success = text(field(case, "status")) == "accepted" && !bounded;
        if output.status.code() != Some(if success { 0 } else { 1 }) {
            failures.push(format!(
                "{id}: exit {:?}, expected success={success}\n{stdout}\n{stderr}",
                output.status.code()
            ));
            continue;
        }
        if !bounded && stdout.contains("usage:") != boolean(field(case, "usage")) {
            failures.push(format!("{id}: usage class differs\n{stdout}\n{stderr}"));
        }
        if !bounded
            && stderr.contains("calculating quantization size")
                != boolean(field(case, "dry_run_message"))
        {
            failures.push(format!("{id}: dry-run message class differs"));
        }
        if !bounded {
            let warnings: Vec<_> = stderr
                .lines()
                .filter(|line| {
                    line.contains("require an imatrix!") || line.contains("requires an imatrix!")
                })
                .collect();
            let expected: Vec<_> = array(field(case, "imatrix_warnings"))
                .iter()
                .map(text)
                .collect();
            if warnings != expected {
                failures.push(format!(
                    "{id}: imatrix warnings {warnings:?}, expected {expected:?}"
                ));
            }
        }
        let expected_outputs: BTreeMap<_, _> = if success {
            array(field(case, "outputs"))
                .iter()
                .map(|v| (text(field(v, "path")).to_owned(), v))
                .collect()
        } else {
            BTreeMap::new()
        };
        let mut files = BTreeMap::new();
        output_files(&dir.0, &dir.0, &mut files);
        for (name, want) in &expected_outputs {
            let Some(actual) = files.get(name) else {
                failures.push(format!("{id}: missing {name}"));
                continue;
            };
            let expected = blobs[text(field(want, "sha256"))];
            if actual.as_slice() != expected {
                failures.push(format!(
                    "{id}/{name}: complete bytes differ (Rust {} {}, original {} {})",
                    actual.len(),
                    sha256(actual),
                    expected.len(),
                    text(field(want, "sha256"))
                ));
            } else {
                exact_files += 1;
                exact_bytes += actual.len();
            }
        }
        let inputs: BTreeMap<_, _> = array(field(case, "inputs"))
            .iter()
            .map(|v| {
                (
                    text(field(v, "path")).replace('\\', "/"),
                    text(field(v, "sha256")),
                )
            })
            .collect();
        for (name, hash) in &inputs {
            if expected_outputs.contains_key(name) {
                continue;
            }
            if files.get(name).is_none_or(|v| sha256(v) != *hash) {
                failures.push(format!("{id}: input/existing destination changed: {name}"));
            }
        }
        for name in files.keys() {
            if !inputs.contains_key(name) && !expected_outputs.contains_key(name) {
                failures.push(format!(
                    "{id}: unexpected file or leaked staging file {name}"
                ));
            }
        }
    }
    assert!(symlink_skips <= 1);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(exact_files, if cfg!(windows) { 117 } else { 118 });
    println!(
        "188 original executable cases: {exact_files} complete outputs, {exact_bytes} exact bytes; {symlink_skips} explicitly reported platform-privilege skips"
    );
    (exact_files, exact_bytes)
}
#[test]
fn original_executable_complete_files_and_errors() {
    native_cli_cases();
}

fn helper_case<'a>(data: &'a Value, id: &str) -> &'a Value {
    array(field(data, "cases"))
        .iter()
        .find(|v| text(field(v, "id")) == id)
        .unwrap()
}
fn matrix_file(data: &Value, id: &str, dir: &Path) -> PathBuf {
    let case = helper_case(data, id);
    let path = dir.join("limit-matrix.dat");
    fs::write(&path, unhex(text(field(field(case, "file"), "hex")))).unwrap();
    path
}
#[test]
fn bounded_matrix_admission_and_filtering() {
    let data = cohort("helper");
    let dir = Directory::new();
    let legacy = matrix_file(&data, "helper-native-v2/362", &dir.0);
    let base = ImatrixLimits::default();
    assert!(prepare_imatrix(&legacy, &[], &[], &base).is_ok());
    let checks = [
        ImatrixLimits {
            max_file_bytes: 16,
            ..base.clone()
        },
        ImatrixLimits {
            max_entries: 0,
            ..base.clone()
        },
        ImatrixLimits {
            max_name_bytes: 0,
            ..base.clone()
        },
        ImatrixLimits {
            max_values: 7,
            ..base.clone()
        },
    ];
    for limits in checks {
        assert!(matches!(
            prepare_imatrix(&legacy, &[], &[], &limits),
            Err(Error::Limit(_))
        ));
    }
    assert!(matches!(
        prepare_imatrix(
            &legacy,
            &["x".into()],
            &[],
            &ImatrixLimits {
                max_filter_patterns: 0,
                ..base.clone()
            }
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        prepare_imatrix(
            &legacy,
            &["x".into()],
            &[],
            &ImatrixLimits {
                max_filter_work: 1,
                ..base.clone()
            }
        ),
        Err(Error::Limit(_))
    ));
    let gguf = matrix_file(&data, "helper-gguf-v3/000", &dir.0);
    assert!(prepare_imatrix(&gguf, &[], &[], &base).is_ok());
    for limits in [
        ImatrixLimits {
            max_metadata_bytes: 16,
            ..base.clone()
        },
        ImatrixLimits {
            max_dataset_bytes: 4,
            ..base.clone()
        },
        ImatrixLimits {
            max_datasets: 1,
            ..base.clone()
        },
        ImatrixLimits {
            max_values: 9,
            ..base.clone()
        },
    ] {
        assert!(prepare_imatrix(&gguf, &[], &[], &limits).is_err());
    }
    // An exclusion cannot hide a malformed entry: normalize before filtering.
    let malformed = matrix_file(&data, "helper-gguf-v3/011", &dir.0);
    assert!(prepare_imatrix(&malformed, &[], &[String::new()], &base).is_err());
    let before = fs::read(&malformed).unwrap();
    assert!(prepare_imatrix(&malformed, &[], &[], &base).is_err());
    assert_eq!(fs::read(malformed).unwrap(), before);
}
#[test]
fn bounded_cli_and_publication_failures_preserve_destination() {
    let data = cohort("executable");
    let c = helper_case(&data, "executable-native-v1/002");
    let dir = Directory::new();
    let blobs = blobs();
    write_inputs(c, &dir.0, &blobs);
    let input = dir.0.join("model.gguf");
    let output = dir.0.join("out.gguf");
    fs::write(&output, b"existing destination").unwrap();
    let args = vec![
        input.to_string_lossy().into_owned(),
        output.to_string_lossy().into_owned(),
        "F16".into(),
        "1".into(),
    ];
    let limits = cli::CliLimits::default();
    let command = cli::parse_command(&args, &limits).unwrap();
    assert!(matches!(
        command.execute(&QuantizeLimits {
            max_work_bytes: 1,
            ..QuantizeLimits::default()
        }),
        Err(Error::Limit(_))
    ));
    assert_eq!(fs::read(&output).unwrap(), b"existing destination");
    assert!(
        cli::parse_command(
            &args,
            &cli::CliLimits {
                max_arguments: 3,
                ..limits.clone()
            }
        )
        .is_err()
    );
    assert!(
        cli::parse_command(
            &args,
            &cli::CliLimits {
                max_argument_bytes: 8,
                ..limits.clone()
            }
        )
        .is_err()
    );
    let file = dir.0.join("types.txt");
    fs::write(&file, b"a=f16 b=f32").unwrap();
    let args = vec![
        "--tensor-type-file".into(),
        file.to_string_lossy().into_owned(),
        input.to_string_lossy().into_owned(),
        "F16".into(),
    ];
    for bounded in [
        cli::CliLimits {
            max_tensor_type_file_bytes: 4,
            ..limits.clone()
        },
        cli::CliLimits {
            max_overrides: 1,
            ..limits.clone()
        },
    ] {
        assert!(cli::parse_command(&args, &bounded).is_err());
    }
    let huge = "x".repeat(16385);
    assert!(matches!(cli::parse_file_type(&huge), Err(Error::Limit(_))));
    assert_eq!(fs::read(&output).unwrap(), b"existing destination");
    assert!(fs::read_dir(&dir.0).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
}
#[test]
fn reviewed_hex_carry_keeps_public_f32_signed_zero() {
    let data = cohort("helper");
    for (index, expected) in [
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        0x000f_ffff_ffff_ffff,
        0x800f_ffff_ffff_ffff,
    ]
    .into_iter()
    .enumerate()
    {
        let case = helper_case(&data, &format!("helper-rounding-correction-v4/{index:03}"));
        let value =
            cli::parse_metadata_override(text(field(field(case, "request"), "text"))).unwrap();
        let OverrideValue::Float(value) = value.value else {
            panic!("not float")
        };
        assert_eq!(value.to_bits(), expected);
        assert_eq!(
            (value as f32).to_bits(),
            if index % 2 == 0 { 0 } else { 0x8000_0000 }
        );
    }
    // The executable suite above compares all four public complete output files,
    // independently captured on both hosts before implementation.
    let executable = cohort("executable");
    let mut signs = BTreeSet::new();
    for case in array(field(&executable, "cases")) {
        if text(field(case, "id")).starts_with("executable-rounding-correction-v4/") {
            let want = &array(field(case, "outputs"))[0];
            signs.insert(text(field(want, "sha256")).to_owned());
        }
    }
    assert_eq!(signs.len(), 2);
}

#[test]
fn core_identity_guards_preserve_native_dry_runs_and_inputs() {
    let captured = cohort("identity-native");
    let extra = cohort("identity-directory");
    let blobs = blobs();
    let source = blobs["694867b88f5718d56508c8a255a2a342d72688913010f4ab943e8ea511135390"];
    for case in array(field(&captured, "cases"))
        .iter()
        .chain(array(field(&extra, "cases")))
    {
        let id = text(field(case, "id"));
        let dir = Directory::new();
        let input = dir.0.join("input.gguf");
        let mut output = dir.0.join("output.gguf");
        fs::write(&input, source).unwrap();
        let dry = boolean(field(field(case, "params"), "dry_run"));
        match id {
            "hardlink-write" | "hardlink-dry" => fs::hard_link(&input, &output).unwrap(),
            "samepath-write" => output = input.clone(),
            "symlink-write" => {
                if let Err(error) = symlink(&input, &output) {
                    #[cfg(windows)]
                    if error.raw_os_error() == Some(1314) {
                        println!(
                            "core {id}: Windows symlink privilege unavailable; hard-link check remains required"
                        );
                        continue;
                    }
                    panic!("native symlink setup: {error}");
                }
            }
            "distinct-copy-write" => {
                fs::write(&output, source).unwrap();
            }
            "directory-dry" => fs::create_dir(&output).unwrap(),
            _ => panic!("unknown original C identity case {id}"),
        }
        let params = llama_model_quant::QuantizeParams {
            ftype: llama_model_quant::FileType::F16,
            nthread: 1,
            dry_run: dry,
            ..llama_model_quant::QuantizeParams::default()
        };
        let result = llama_model_quant::quantize_file(&input, &output, &params);
        if dry {
            assert_eq!(
                number(field(
                    field(&array(field(case, "result"))[0], "result"),
                    "return_code"
                )),
                0
            );
            assert!(!result.unwrap().written, "{id}");
        } else if id == "distinct-copy-write" {
            assert!(result.unwrap().written);
            let native = text(field(field(case, "output_after"), "sha256"));
            assert_eq!(fs::read(&output).unwrap().as_slice(), blobs[native]);
        } else {
            // GNU's native hard-link write mutates the input; same-path/symlink
            // writes abort with SIGBUS. These retained unsafe observations are
            // intentionally replaced by checked rejection, not imitated.
            assert!(result.is_err(), "{id}");
            if cfg!(windows) {
                assert_eq!(
                    number(field(
                        field(&array(field(case, "result"))[0], "result"),
                        "return_code"
                    )),
                    1
                );
                assert!(boolean(field(case, "input_unchanged")));
            } else {
                assert!(!boolean(field(case, "input_unchanged")));
            }
        }
        assert_eq!(fs::read(&input).unwrap().as_slice(), source, "{id}");
        assert!(fs::read_dir(&dir.0).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
    }
}
