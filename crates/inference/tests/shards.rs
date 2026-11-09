//! Independent original C/C++ shard reader, model loader and inference corpus.
//! Source captures and lossless packing are documented in shards-provenance.json.
#[path = "support/native_profile.rs"]
mod native_profile;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, GgufSet, Limits, ShardLimits, Value as Metadata};
use llama_inference::{LoadOptions, Model, Pooling};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const INPUTS: &[u8] = include_bytes!("../../../fixtures/gguf/shards-inputs.bin");
const PAYLOADS: &[u8] = include_bytes!("../../../fixtures/gguf/shards-payloads.bin");
const REFERENCE: &str = include_str!("../../../fixtures/gguf/shards-reference.json");

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("missing fixture field {key}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture array")
}
fn number(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("fixture unsigned integer")).unwrap()
}
fn string(value: &Value) -> &str {
    value.as_str().expect("fixture string")
}
fn boolean(value: &Value) -> bool {
    value.as_bool().expect("fixture boolean")
}
fn block<'a>(data: &'a [u8], description: &Value) -> &'a [u8] {
    let start = number(field(description, "offset"));
    let bytes = number(field(description, "bytes"));
    let result = &data[start..start.checked_add(bytes).unwrap()];
    assert_eq!(sha256(result), string(field(description, "sha256")));
    result
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn decode_hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

struct Corpus {
    root: PathBuf,
    document: Value,
}
impl Corpus {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rust-llama-shards-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let document = json::parse(REFERENCE).unwrap();
        let inputs = array(field(&document, "input_blobs"));
        for entry in array(field(&document, "paths")) {
            let relative = Path::new(string(field(entry, "path")));
            assert!(
                relative
                    .components()
                    .all(|p| matches!(p, Component::Normal(_)))
            );
            let path = root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, block(INPUTS, &inputs[number(field(entry, "blob"))])).unwrap();
        }
        Self { root, document }
    }
    fn cases(&self) -> &[Value] {
        array(field(&self.document, "cases"))
    }
    fn case(&self, name: &str) -> &Value {
        self.cases()
            .iter()
            .find(|c| string(field(c, "name")) == name)
            .unwrap()
    }
    fn paths(&self, case: &Value) -> Vec<PathBuf> {
        array(field(case, "paths"))
            .iter()
            .map(|p| self.root.join(string(p)))
            .collect()
    }
    fn inspection(&self, relative: &str) -> &Value {
        let path = array(field(&self.document, "paths"))
            .iter()
            .find(|p| string(field(p, "path")) == relative)
            .unwrap();
        let input = &array(field(&self.document, "input_blobs"))[number(field(path, "blob"))];
        &array(field(&self.document, "inspections"))[number(field(input, "inspection"))]
    }
    fn open(&self, case: &Value) -> llama_gguf::Result<GgufSet> {
        let paths = self.paths(case);
        if boolean(field(case, "explicit")) {
            GgufSet::open_splits(&paths)
        } else {
            GgufSet::open(&paths[0])
        }
    }
}
impl Drop for Corpus {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn scalar_bytes(value: &Metadata, bytes: &mut Vec<u8>) {
    match value {
        Metadata::U8(v) => bytes.push(*v),
        Metadata::I8(v) => bytes.extend(v.to_le_bytes()),
        Metadata::U16(v) => bytes.extend(v.to_le_bytes()),
        Metadata::I16(v) => bytes.extend(v.to_le_bytes()),
        Metadata::U32(v) => bytes.extend(v.to_le_bytes()),
        Metadata::I32(v) => bytes.extend(v.to_le_bytes()),
        Metadata::U64(v) => bytes.extend(v.to_le_bytes()),
        Metadata::I64(v) => bytes.extend(v.to_le_bytes()),
        Metadata::F32(v) => bytes.extend(v.to_le_bytes()),
        Metadata::F64(v) => bytes.extend(v.to_le_bytes()),
        Metadata::Bool(v) => bytes.push(u8::from(*v)),
        _ => panic!("fixture scalar array"),
    }
}
fn compare_metadata(actual: &BTreeMap<String, Metadata>, expected: &Value) {
    let entries = array(field(expected, "metadata"));
    assert_eq!(
        actual.len(),
        entries.len(),
        "metadata must not be merged across shards"
    );
    for item in entries {
        let key = string(field(item, "key"));
        let a = actual
            .get(key)
            .unwrap_or_else(|| panic!("missing metadata {key}"));
        assert_eq!(a.type_id() as usize, number(field(item, "type")), "{key}");
        let e = field(item, "value");
        match a {
            Metadata::String(v) => assert_eq!(v, string(e), "{key}"),
            Metadata::Bool(v) => assert_eq!(*v, boolean(e), "{key}"),
            Metadata::F32(v) => assert_eq!(
                hex(&v.to_le_bytes()),
                string(field(e, "little_endian_bits_hex")),
                "{key}"
            ),
            Metadata::F64(v) => assert_eq!(
                hex(&v.to_le_bytes()),
                string(field(e, "little_endian_bits_hex")),
                "{key}"
            ),
            Metadata::Array {
                element_type,
                values,
            } => {
                assert_eq!(*element_type as usize, number(field(e, "element_type")));
                assert_eq!(values.len(), number(field(e, "length")));
                if *element_type == 8 {
                    let strings = array(field(e, "values"));
                    for (value, expected) in values.iter().zip(strings) {
                        assert_eq!(value.as_str().unwrap(), string(expected), "{key}");
                    }
                } else {
                    let mut bytes = Vec::new();
                    for value in values {
                        scalar_bytes(value, &mut bytes);
                    }
                    assert_eq!(hex(&bytes), string(field(e, "native_bytes_hex")), "{key}");
                }
            }
            Metadata::I8(_) | Metadata::I16(_) | Metadata::I32(_) | Metadata::I64(_) => {
                assert_eq!(a.as_i64(), e.as_i64(), "{key}");
            }
            Metadata::U8(_) | Metadata::U16(_) | Metadata::U32(_) | Metadata::U64(_) => {
                assert_eq!(a.as_u64(), e.as_u64(), "{key}");
            }
        }
    }
}
fn compare_tensor(actual: &llama_gguf::TensorInfo, expected: &Value) {
    assert_eq!(actual.name, string(field(expected, "name")));
    assert_eq!(actual.dtype.id() as usize, number(field(expected, "dtype")));
    assert_eq!(actual.offset, field(expected, "offset").as_u64().unwrap());
    assert_eq!(
        actual.byte_len,
        field(expected, "byte_len").as_u64().unwrap()
    );
    let dimensions: Vec<u64> = (0..4)
        .map(|i| actual.dimensions.get(i).copied().unwrap_or(1))
        .collect();
    let expected: Vec<u64> = array(field(expected, "dimensions"))
        .iter()
        .map(|n| n.as_u64().unwrap())
        .collect();
    assert_eq!(dimensions, expected);
}

#[test]
fn original_reader_headers_shapes_offsets_and_payloads() {
    let corpus = Corpus::new();
    let mut checked = 0usize;
    let mut tensor_bytes = 0usize;
    for path in array(field(&corpus.document, "paths")) {
        let name = string(field(path, "path"));
        let expected = corpus.inspection(name);
        let valid = boolean(field(expected, "accepted"))
            && array(field(expected, "tensors"))
                .iter()
                .all(|t| boolean(field(t, "payload_complete")));
        let actual = Gguf::open(corpus.root.join(name));
        assert_eq!(actual.is_ok(), valid, "{name}: {:?}", actual.as_ref().err());
        if !valid {
            continue;
        }
        let mut actual = actual.unwrap();
        assert_eq!(
            actual.version as usize,
            number(field(expected, "version")),
            "{name}"
        );
        assert_eq!(
            actual.alignment as usize,
            number(field(expected, "alignment")),
            "{name}"
        );
        assert_eq!(
            actual.data_offset as usize,
            number(field(expected, "data_offset")),
            "{name}"
        );
        assert_eq!(
            actual.file_len as usize,
            number(field(expected, "file_len")),
            "{name}"
        );
        compare_metadata(&actual.metadata, expected);
        let tensors = array(field(expected, "tensors"));
        assert_eq!(actual.tensors.len(), tensors.len());
        for tensor in tensors {
            let name = string(field(tensor, "name"));
            compare_tensor(actual.tensor(name).unwrap(), tensor);
            let expected_bytes = block(PAYLOADS, field(tensor, "payload"));
            assert_eq!(actual.read_tensor(name).unwrap(), expected_bytes);
            if number(field(tensor, "dtype")) == 0 {
                let decoded = actual.read_f32(name).unwrap();
                let source: Vec<u32> = expected_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                assert_eq!(
                    decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    source
                );
            }
            tensor_bytes += expected_bytes.len();
        }
        checked += 1;
    }
    assert_eq!(array(field(&corpus.document, "format_cases")).len(), 10);
    eprintln!("shard-native-format: valid_files={checked} compared_payload_bytes={tensor_bytes}");
}

#[test]
fn original_model_shard_admission_primary_metadata_and_file_origins() {
    let corpus = Corpus::new();
    assert_eq!(corpus.cases().len(), 90);
    let mut accepted = 0usize;
    let mut tensors = 0usize;
    for case in corpus.cases() {
        let name = string(field(case, "name"));
        let result = corpus.open(case);
        assert_eq!(
            result.is_ok(),
            boolean(field(case, "index_accepted")),
            "{name}: {:?}",
            result.as_ref().err()
        );
        if !boolean(field(case, "index_accepted")) {
            continue;
        }
        let mut set = result.unwrap();
        let paths = array(field(case, "paths"));
        let first = corpus.inspection(string(&paths[0]));
        compare_metadata(set.metadata(), first);
        let count = set
            .metadata()
            .get("split.count")
            .and_then(Metadata::as_u64)
            .unwrap_or(0)
            .max(1) as usize;
        assert_eq!(set.shard_count(), count, "{name}");
        let mut file_bytes = 0u64;
        let mut tensor_bytes = 0u64;
        let mut elements = 0u64;
        let mut file_tensors = 0usize;
        for (index, path) in paths.iter().take(count).enumerate() {
            let reference = corpus.inspection(string(path));
            let info = set.shard_info(index).unwrap();
            assert_eq!(
                info.data_offset as usize,
                number(field(reference, "data_offset")),
                "{name} shard {index}"
            );
            assert_eq!(
                info.alignment as usize,
                number(field(reference, "alignment"))
            );
            assert_eq!(info.tensor_count, array(field(reference, "tensors")).len());
            file_bytes += field(reference, "file_len").as_u64().unwrap();
            for tensor in array(field(reference, "tensors")) {
                let key = string(field(tensor, "name"));
                compare_tensor(set.tensor(key).unwrap(), tensor);
                assert_eq!(set.tensor_shard(key), Some(index), "{name} {key}");
                assert_eq!(
                    set.read_tensor(key).unwrap(),
                    block(PAYLOADS, field(tensor, "payload"))
                );
                tensor_bytes += field(tensor, "byte_len").as_u64().unwrap();
                elements += array(field(tensor, "dimensions"))
                    .iter()
                    .map(|n| n.as_u64().unwrap())
                    .product::<u64>();
                file_tensors += 1;
            }
        }
        assert_eq!(set.tensors().len(), file_tensors);
        assert_eq!(set.file_bytes(), file_bytes);
        assert_eq!(set.tensor_bytes(), tensor_bytes);
        assert_eq!(set.tensor_elements(), elements);
        assert!(set.tensor("does.not.exist").is_none());
        assert!(set.read_tensor("does.not.exist").is_err());
        accepted += 1;
        tensors += file_tensors;
    }
    assert_eq!(accepted, 65);
    eprintln!(
        "shard-native-index: cases=90 accepted={accepted} rejected={} compared_tensors={tensors}",
        90 - accepted
    );
}

#[derive(Default)]
struct Metrics {
    values: usize,
    bitwise: usize,
    maximum: f64,
    largest_fraction: f64,
    greedy: usize,
}
fn compare_values(actual: &[f32], expected: &[u8], label: &str, metrics: &mut Metrics) {
    assert_eq!(expected.len(), actual.len() * 4, "{label}");
    for (&a, e) in actual.iter().zip(expected.as_chunks::<4>().0) {
        let e = f32::from_le_bytes(*e);
        let difference = (f64::from(a) - f64::from(e)).abs();
        let budget = 2e-4 + 2e-5 * f64::from(e).abs();
        assert!(
            a.is_finite() && e.is_finite() && difference <= budget,
            "{label}: {a:.9e} != {e:.9e}; difference={difference:.9e} budget={budget:.9e}"
        );
        if cfg!(any(
            all(
                target_arch = "x86_64",
                target_os = "windows",
                target_env = "msvc"
            ),
            all(
                target_arch = "x86_64",
                target_os = "linux",
                target_env = "gnu"
            )
        )) {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: measured native profile requires exact F32 bits"
            );
        }
        metrics.values += 1;
        metrics.bitwise += usize::from(a.to_bits() == e.to_bits());
        metrics.maximum = metrics.maximum.max(difference);
        metrics.largest_fraction = metrics.largest_fraction.max(difference / budget);
    }
}
fn ids(value: &Value) -> Vec<i32> {
    array(value)
        .iter()
        .map(|v| i32::try_from(v.as_i64().unwrap()).unwrap())
        .collect()
}
fn check_model(
    model: &Model,
    reference: &Value,
    values: &[u8],
    label: &str,
    metrics: &mut Metrics,
) {
    assert_eq!(model.vocabulary_size(), number(field(reference, "n_vocab")));
    let embedding = string(field(reference, "family")) == "embedding";
    assert_eq!(model.supports_embeddings(), embedding);
    assert_eq!(model.supports_generation(), !embedding);
    for case in array(field(reference, "cases")) {
        let tokens = ids(field(case, "tokens"));
        let prompt = string(field(case, "prompt"));
        assert_eq!(
            model.tokenize(prompt, true).unwrap(),
            tokens,
            "{label} {prompt:?}"
        );
        if embedding {
            assert_eq!(
                model.embedding_dimensions(),
                number(field(reference, "n_embd"))
            );
            let actual = model
                .embed_tokens(&tokens, Pooling::None, &|| false)
                .unwrap();
            assert!(actual.pooled.is_none());
            let expected = field(case, "embeddings");
            assert_eq!(actual.tokens.len(), number(field(expected, "rows")));
            let bytes = block(values, expected);
            let width = number(field(expected, "columns"));
            for (index, row) in actual.tokens.iter().enumerate() {
                compare_values(
                    row,
                    &bytes[index * width * 4..(index + 1) * width * 4],
                    label,
                    metrics,
                );
            }
            for (name, pooling) in [
                ("mean", Pooling::Mean),
                ("first", Pooling::First),
                ("last", Pooling::Last),
            ] {
                let result = model.embed_tokens(&tokens, pooling, &|| false).unwrap();
                compare_values(
                    result.pooled.as_ref().unwrap(),
                    block(values, field(field(case, "pooling"), name)),
                    label,
                    metrics,
                );
            }
        } else {
            let greedy = ids(field(case, "greedy_tokens"));
            let mut sequence = tokens.clone();
            sequence.extend_from_slice(&greedy[..greedy.len() - 1]);
            let actual = model.logits_for_tokens(&sequence).unwrap();
            let expected = field(case, "logits");
            assert_eq!(actual.len(), number(field(expected, "rows")));
            let bytes = block(values, expected);
            let width = number(field(expected, "columns"));
            for (index, row) in actual.iter().enumerate() {
                compare_values(
                    row,
                    &bytes[index * width * 4..(index + 1) * width * 4],
                    label,
                    metrics,
                );
            }
            for (step, &expected) in greedy.iter().enumerate() {
                let row = &actual[tokens.len() - 1 + step];
                let mut best = 0;
                for i in 1..row.len() {
                    if row[i] > row[best] {
                        best = i;
                    }
                }
                assert_eq!(best as i32, expected, "{label} greedy step {step}");
                assert_eq!(
                    model.tokenizer().token_piece(expected, false).unwrap(),
                    decode_hex(string(&array(field(case, "greedy_pieces_hex"))[step]))
                );
                metrics.greedy += 1;
            }
        }
    }
}

#[test]
fn all_four_architectures_execute_original_sharded_native_corpora() {
    let corpus = Corpus::new();
    let host = native_profile::source_profile("msvc", "gnu");
    let values = native_profile::source_profile::<&[u8]>(
        include_bytes!("../../../fixtures/gguf/shards-msvc-values.f32le.bin"),
        include_bytes!("../../../fixtures/gguf/shards-gnu-values.f32le.bin"),
    );
    let references = array(field(field(&corpus.document, "references"), host));
    let options = LoadOptions {
        context_length: 64,
        ..LoadOptions::default()
    };
    let mut metrics = Metrics::default();
    let mut models = 0usize;
    for case in corpus.cases() {
        let name = string(field(case, "name"));
        let paths = corpus.paths(case);
        let actual = if boolean(field(case, "explicit")) {
            Model::load_from_splits_with_options(&paths, &options)
        } else {
            Model::load_with_options(&paths[0], &options)
        };
        let expected = case.get("inference_accepted").is_some_and(boolean);
        assert_eq!(
            actual.is_ok(),
            expected,
            "{name}: {:?}",
            actual.as_ref().err()
        );
        if !expected {
            continue;
        }
        let reference = &references[number(field(field(case, "reference"), host))];
        check_model(&actual.unwrap(), reference, values, name, &mut metrics);
        models += 1;
    }
    assert_eq!(models, 64);
    assert_eq!(metrics.values, 464_272);
    assert_eq!(metrics.greedy, 590);
    eprintln!(
        "shard-native-models: host={host} configurations=90 executed={models} logits_and_embeddings={} greedy_ids={} bitwise_equal={} maximum_absolute={:.9e} maximum_budget_fraction={:.9e}",
        metrics.values, metrics.greedy, metrics.bitwise, metrics.maximum, metrics.largest_fraction
    );
}

fn require_limit(result: llama_gguf::Result<GgufSet>) {
    match result {
        Err(llama_gguf::Error::Limit(_)) => {}
        Err(error) => panic!("expected cumulative resource limit, got {error}"),
        Ok(_) => panic!("expected cumulative resource limit"),
    }
}
#[test]
fn cumulative_limits_include_every_shard_and_reads_remain_lazy() {
    let corpus = Corpus::new();
    let case = corpus.case("three-explicit");
    let paths = corpus.paths(case);
    let original = GgufSet::open_splits(&paths).unwrap();
    let usage = original.header_usage();
    let exact = ShardLimits {
        max_shards: paths.len(),
        gguf: Limits {
            max_metadata_bytes: usage.bytes,
            max_metadata_entries: usage.metadata_entries,
            max_total_values: usage.metadata_values,
            max_tensors: usage.tensors,
            ..Limits::default()
        },
    };
    let loaded = GgufSet::open_splits_with_limits(&paths, exact.clone()).unwrap();
    assert_eq!(loaded.header_usage(), usage);
    for budget in 0..5 {
        let mut limits = exact.clone();
        match budget {
            0 => limits.max_shards -= 1,
            1 => limits.gguf.max_metadata_bytes -= 1,
            2 => limits.gguf.max_metadata_entries -= 1,
            3 => limits.gguf.max_total_values -= 1,
            4 => limits.gguf.max_tensors -= 1,
            _ => unreachable!(),
        }
        require_limit(GgufSet::open_splits_with_limits(&paths, limits));
    }
    let mut tiny_read = exact;
    tiny_read.gguf.max_tensor_read_bytes = 1;
    let mut lazy = GgufSet::open_splits_with_limits(&paths, tiny_read).unwrap();
    assert!(matches!(
        lazy.read_tensor("token_embd.weight"),
        Err(llama_gguf::Error::Limit(_))
    ));
    assert!(matches!(
        lazy.read_f32("token_embd.weight"),
        Err(llama_gguf::Error::Limit(_))
    ));
    let model = Model::load_from_gguf(loaded, &LoadOptions::default()).unwrap();
    assert_eq!(model.vocabulary_size(), 512);
}

#[test]
fn model_weight_budget_counts_all_shards_before_decoding() {
    let corpus = Corpus::new();
    let paths = corpus.paths(corpus.case("two-explicit"));
    let set = GgufSet::open_splits(&paths).unwrap();
    let primary = Gguf::open(&paths[0]).unwrap();
    let first_bytes: usize = primary
        .tensors
        .iter()
        .map(|t| t.elements().unwrap() * 4)
        .sum();
    assert!((first_bytes as u64) < set.tensor_elements() * 4);
    let error = Model::load_from_splits_with_options(
        &paths,
        &LoadOptions {
            max_decoded_weight_bytes: first_bytes + 1,
            ..LoadOptions::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, llama_inference::Error::Invalid(ref message) if message.contains("decoded model requires"))
    );
}

#[test]
fn reads_keep_the_validated_file_handles_after_path_replacement() {
    let corpus = Corpus::new();
    let paths = corpus.paths(corpus.case("two-explicit"));
    let mut set = GgufSet::open_splits(&paths).unwrap();
    let name = set
        .tensors()
        .iter()
        .find(|t| set.tensor_shard(&t.name) == Some(1))
        .unwrap()
        .name
        .clone();
    let expected = set.read_tensor(&name).unwrap();
    let backup = paths[1].with_extension("opened-original");
    fs::rename(&paths[1], &backup).unwrap();
    fs::write(&paths[1], b"not a GGUF").unwrap();
    assert_eq!(set.read_tensor(&name).unwrap(), expected);
    assert!(GgufSet::open_splits(&paths).is_err());
}

fn rename_with_prefix(corpus: &Corpus, prefix: &std::ffi::OsStr) -> Vec<PathBuf> {
    let source = corpus.paths(corpus.case("two-explicit"));
    source
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let mut name = prefix.to_owned();
            name.push(format!("-{:05}-of-{:05}.gguf", index + 1, source.len()));
            let path = corpus.root.join(name);
            fs::copy(input, &path).unwrap();
            path
        })
        .collect()
}
#[test]
fn automatic_discovery_preserves_unicode_file_paths() {
    let corpus = Corpus::new();
    let paths = rename_with_prefix(&corpus, std::ffi::OsStr::new("模型-😺"));
    let mut set = GgufSet::open(&paths[0]).unwrap();
    assert_eq!(set.shard_count(), 2);
    assert_eq!(set.shard_info(1).unwrap().path, paths[1]);
    assert_eq!(set.read_tensor("token_embd.weight").unwrap().len(), 8192);
}
#[cfg(unix)]
#[test]
fn automatic_discovery_preserves_non_utf8_unix_file_paths() {
    use std::os::unix::ffi::OsStringExt;
    let corpus = Corpus::new();
    let prefix = std::ffi::OsString::from_vec(b"model-\xff".to_vec());
    let paths = rename_with_prefix(&corpus, &prefix);
    let set = GgufSet::open(&paths[0]).unwrap();
    assert_eq!(set.shard_info(1).unwrap().path, paths[1]);
    assert_eq!(set.tensors().len(), 35);
}
