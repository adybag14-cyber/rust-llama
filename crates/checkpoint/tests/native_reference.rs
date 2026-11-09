use llama_checkpoint::{
    Checkpoint, CheckpointOptions, Limits, Number, RawTensorInfo, SafetensorsFile, Text, Value,
};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SERIAL: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "llama-checkpoint-native-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("unique test directory");
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], magic: &[u8]) -> Self {
        assert!(bytes.starts_with(magic));
        Self {
            bytes,
            at: magic.len(),
        }
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let end = self.at.checked_add(n).expect("fixture cursor");
        let bytes = self
            .bytes
            .get(self.at..end)
            .expect("complete immutable fixture");
        self.at = end;
        bytes
    }
    fn byte(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn count(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
    fn blob(&mut self) -> &'a [u8] {
        let count = self.count();
        self.take(count)
    }
    fn text(&mut self) -> String {
        String::from_utf8(self.blob().to_vec()).expect("test path is Unicode")
    }
    fn finish(self) {
        assert_eq!(self.at, self.bytes.len(), "trailing fixture data");
    }
}
fn length(bytes: &mut Vec<u8>, n: usize) {
    bytes.extend_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
}
fn blob(bytes: &mut Vec<u8>, input: &[u8]) {
    length(bytes, input.len());
    bytes.extend_from_slice(input);
}
fn encode_number(bytes: &mut Vec<u8>, number: &Number) {
    match number {
        Number::Integer(value) => {
            bytes.push(b'I');
            bytes.push(u8::from(value.is_negative()));
            blob(bytes, value.digits().as_bytes());
        }
        Number::Float(value) => {
            bytes.push(b'F');
            let bits = if value.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                value.to_bits()
            };
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
}
fn encode_text(bytes: &mut Vec<u8>, value: &Text) {
    bytes.push(b'S');
    blob(bytes, value.as_wtf8());
}
fn encode(bytes: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => bytes.push(b'N'),
        Value::Bool(value) => {
            bytes.push(b'B');
            bytes.push(u8::from(*value));
        }
        Value::Number(value) => encode_number(bytes, value),
        Value::Text(value) => encode_text(bytes, value),
        Value::Array(values) => {
            bytes.push(b'A');
            length(bytes, values.len());
            for value in values {
                encode(bytes, value);
            }
        }
        Value::Object(object) => {
            bytes.push(b'O');
            length(bytes, object.len());
            for (key, value) in object.entries() {
                blob(bytes, key.as_wtf8());
                encode(bytes, value);
            }
        }
    }
}
fn info(name: &Text, tensor: &RawTensorInfo) -> Vec<u8> {
    let mut result = Vec::new();
    encode_text(&mut result, name);
    encode(&mut result, tensor.dtype_value());
    result.push(b'A');
    length(&mut result, tensor.shape_values().len());
    for value in tensor.shape_values() {
        encode(&mut result, value);
    }
    encode_number(&mut result, tensor.absolute_offset());
    encode_number(&mut result, tensor.byte_length());
    result
}
struct ExpectedTensor<'a> {
    info: &'a [u8],
    raw: Option<&'a [u8]>,
    state: u8,
}
fn expected_tensor<'a>(cursor: &mut Cursor<'a>) -> ExpectedTensor<'a> {
    let info = cursor.blob();
    let state = cursor.byte();
    assert!(state <= 2);
    let raw = (state == 1).then(|| cursor.blob());
    ExpectedTensor { info, raw, state }
}
fn compare_raw(
    case: &str,
    expected: &ExpectedTensor<'_>,
    result: llama_checkpoint::Result<Vec<u8>>,
) {
    if let Some(raw) = expected.raw {
        assert_eq!(result.unwrap(), raw, "{case}: original raw bytes");
    } else {
        assert!(
            result.is_err(),
            "{case}: expected checked raw error (native state {})",
            expected.state
        );
    }
}
#[test]
fn frozen_source_file_metadata_and_ranges() {
    run_files(
        include_bytes!("fixtures/source-files-reference.bin"),
        137,
        97,
        false,
    );
}
#[test]
fn frozen_source_additive_metadata_and_numeric_edges() {
    run_files(
        include_bytes!("fixtures/source-files-edges-reference.bin"),
        10,
        9,
        true,
    );
}
fn run_files(fixture: &[u8], cases: usize, expected_admissions: usize, edges: bool) {
    let mut cursor = Cursor::new(fixture, if edges { b"STF2" } else { b"STF1" });
    assert_eq!(cursor.count(), cases);
    let temp = Temp::new();
    let mut admitted = 0;
    let mut native_raw_success = 0;
    let mut native_raw_error = 0;
    let mut unprobed_checked_error = 0;
    for index in 0..cases {
        let case = cursor.text();
        let input = cursor.blob();
        let source_admits = cursor.byte() != 0;
        let path = temp.0.join(format!("{index}.safetensors"));
        fs::write(&path, input).unwrap();
        let result = SafetensorsFile::open(&path, Limits::default());
        if !source_admits {
            assert!(result.is_err(), "{case}: source rejected header");
            continue;
        }
        admitted += 1;
        let file = result.unwrap_or_else(|e| panic!("{case}: original admitted header, Rust: {e}"));
        let count = cursor.count();
        assert_eq!(file.tensors().len(), count, "{case}");
        for tensor_index in 0..count {
            let expected = expected_tensor(&mut cursor);
            let tensor = &file.tensors()[tensor_index];
            assert_eq!(
                info(tensor.name(), tensor),
                expected.info,
                "{case}: exact original metadata"
            );
            assert_eq!(file.find(tensor.name()), Some(tensor_index));
            compare_raw(&case, &expected, file.read_tensor(tensor_index));
            match expected.state {
                1 => native_raw_success += 1,
                0 => native_raw_error += 1,
                2 => unprobed_checked_error += 1,
                _ => unreachable!(),
            }
        }
        if edges {
            let present = cursor.byte() != 0;
            assert_eq!(
                file.metadata().is_some(),
                present,
                "{case}: metadata presence"
            );
            if present {
                let expected_metadata = cursor.blob();
                let mut actual_metadata = Vec::new();
                encode(&mut actual_metadata, file.metadata().unwrap());
                assert_eq!(
                    actual_metadata, expected_metadata,
                    "{case}: original JSON metadata view"
                );
            }
        }
    }
    cursor.finish();
    assert_eq!(admitted, expected_admissions);
    eprintln!(
        "source files: {cases}, admitted: {admitted}, raw successes: {native_raw_success}, raw failures: {native_raw_error}, unprobed checked errors: {unprobed_checked_error}"
    );
}

#[derive(Debug)]
struct Initial {
    label: String,
    bytes: Vec<u8>,
}
fn create_link(kind: u8, target: &str, name: &str, model: &Path) {
    match kind {
        1 => {
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, model.join(name)).unwrap();
            #[cfg(windows)]
            std::os::windows::fs::symlink_file(target, model.join(name)).unwrap();
        }
        2 => fs::hard_link(model.join(target), model.join(name)).unwrap(),
        _ => unreachable!(),
    }
}
fn compare_checkpoint(
    case: &str,
    checkpoint: &Checkpoint,
    tensors: &[ExpectedTensor<'_>],
    names: &[(Vec<u8>, Vec<u8>)],
) {
    assert_eq!(checkpoint.len(), tensors.len(), "{case}: final map length");
    for (actual, expected) in checkpoint.tensors().zip(tensors) {
        assert_eq!(
            info(actual.name(), actual.info()),
            expected.info,
            "{case}: final tensor metadata/order"
        );
        assert_eq!(
            checkpoint.find(actual.name()).unwrap().file_name(),
            actual.file_name()
        );
        compare_raw(case, expected, actual.read());
    }
    let actual: Vec<_> = checkpoint
        .source_names()
        .map(|(a, b)| (a.as_wtf8().to_vec(), b.as_wtf8().to_vec()))
        .collect();
    assert_eq!(actual, names, "{case}: original source-name dict order");
}
#[test]
fn frozen_source_index_selection_and_overwrites() {
    #[cfg(windows)]
    let fixture = include_bytes!("fixtures/source-index-windows-reference.bin").as_slice();
    #[cfg(not(windows))]
    let fixture = include_bytes!("fixtures/source-index-linux-reference.bin").as_slice();
    run_indexes(fixture, 43, if cfg!(windows) { 27 } else { 28 }, false);
}
#[test]
fn frozen_source_additive_index_edges() {
    run_indexes(
        include_bytes!("fixtures/source-index-edges-reference.bin"),
        5,
        5,
        true,
    );
}
fn run_indexes(fixture: &[u8], cases: usize, expected_admissions: usize, edges: bool) {
    let mut cursor = Cursor::new(fixture, if edges { b"STI2" } else { b"STI1" });
    assert_eq!(cursor.count(), cases);
    let temp = Temp::new();
    let mut source_admitted = 0;
    let mut strict_admitted = 0;
    let mut capability_admitted = 0;
    for index in 0..cases {
        let case = cursor.text();
        let prefix = cursor.text();
        let mistral = cursor.byte() != 0;
        let policy_difference = cursor.byte() != 0;
        let count = cursor.count();
        let model = temp.0.join(index.to_string()).join("model");
        fs::create_dir_all(&model).unwrap();
        let mut initial = Vec::new();
        for _ in 0..count {
            let label = cursor.text();
            let bytes = cursor.blob().to_vec();
            let path = model.join(&label);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, &bytes).unwrap();
            initial.push(Initial { label, bytes });
        }
        if edges {
            let count = cursor.count();
            for _ in 0..count {
                fs::create_dir_all(model.join(cursor.text())).unwrap();
            }
        }
        let link = cursor.byte();
        if link != 0 {
            let target = cursor.text();
            let name = cursor.text();
            create_link(link, &target, &name, &model);
        }
        let count = cursor.count();
        let mut labels = Vec::new();
        for _ in 0..count {
            labels.push((cursor.text(), cursor.count()));
        }
        let source_accepts = cursor.byte() != 0;
        let mut expected = Vec::new();
        let mut source_names = Vec::new();
        if source_accepts {
            source_admitted += 1;
            let count = cursor.count();
            for _ in 0..count {
                expected.push(expected_tensor(&mut cursor));
            }
            let count = cursor.count();
            for _ in 0..count {
                source_names.push((cursor.blob().to_vec(), cursor.blob().to_vec()));
            }
        }
        let options = CheckpointOptions {
            prefix,
            mistral,
            ..CheckpointOptions::default()
        };
        let strict = Checkpoint::open_directory(&model, options.clone(), Limits::default());
        if source_accepts && !policy_difference {
            let checkpoint =
                strict.unwrap_or_else(|e| panic!("{case}: original accepts safe directory: {e}"));
            strict_admitted += 1;
            compare_checkpoint(&case, &checkpoint, &expected, &source_names);
        } else {
            assert!(
                strict.is_err(),
                "{case}: source error or documented checked path policy"
            );
        }

        // The Windows original outside-symlink was invalid at the OS boundary.
        // Its separate trusted-capability case is tested by the GNU observation;
        // we do not relabel this recorded native failure as an admitted case.
        if cfg!(windows) && case == "indexed-symlink-outside" {
            continue;
        }
        let physical: Vec<_> = initial
            .iter()
            .filter(|file| !file.label.ends_with(".index.json"))
            .collect();
        let mut supplied = Vec::new();
        for (label, position) in labels {
            let owner = File::open(model.join(&physical[position].label)).unwrap();
            supplied.push((Text::try_from_str(&label).unwrap(), owner));
        }
        let index_name = format!("{}.safetensors.index.json", options.prefix);
        let index_bytes = initial
            .iter()
            .find(|file| file.label == index_name)
            .map(|file| file.bytes.as_slice());
        let capability =
            Checkpoint::from_open_files(supplied, index_bytes, options, Limits::default());
        if source_accepts {
            let checkpoint =
                capability.unwrap_or_else(|e| panic!("{case}: original accepts capabilities: {e}"));
            capability_admitted += 1;
            compare_checkpoint(&case, &checkpoint, &expected, &source_names);
        } else {
            assert!(capability.is_err(), "{case}: original index rejects");
        }
    }
    cursor.finish();
    assert_eq!(source_admitted, expected_admissions);
    assert_eq!(capability_admitted, source_admitted);
    eprintln!(
        "source index cases: {cases}, original admissions: {source_admitted}, default path admissions: {strict_admitted}, capability admissions: {capability_admitted}"
    );
}
