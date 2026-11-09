//! Original frozen-header microbatch replay, independent of Rust expectations.
//! Both measured compiler profiles, native default parameters, all four K/V
//! storage pairs, tiled/split/row-map flash, worker setters and exact state bytes.
use crate::{batch::*, context::*, memory::*, model::*, state::*, vocab::*, *};
use llama_behavior::json::{self, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const DATA: &[u8] = include_bytes!("../../../fixtures/ffi/native-microbatch-data.bin");
#[cfg(target_env = "msvc")]
const PROFILE: &str = "msvc";
#[cfg(target_env = "gnu")]
const PROFILE: &str = "gnu";

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value.get(name).unwrap_or_else(|| panic!("missing {name}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
fn integer(value: &Value) -> i64 {
    value.as_i64().unwrap()
}
fn number(value: &Value, name: &str) -> i64 {
    integer(field(value, name))
}
fn text<'a>(value: &'a Value, name: &str) -> &'a str {
    field(value, name).as_str().unwrap()
}
fn flag(value: &Value, name: &str) -> bool {
    field(value, name).as_bool().unwrap()
}
fn number_or(value: &Value, name: &str, default: i64) -> i64 {
    value.get(name).map_or(default, integer)
}
fn flag_or(value: &Value, name: &str, default: bool) -> bool {
    value
        .get(name)
        .map_or(default, |entry| entry.as_bool().unwrap())
}
fn packed(value: &Value) -> &'static [u8] {
    let offset = usize::try_from(number(value, "offset")).unwrap();
    let bytes = usize::try_from(number(value, "bytes")).unwrap();
    &DATA[offset..offset.checked_add(bytes).unwrap()]
}
fn clean() {
    unsafe {
        llama_rust_clear_error();
    }
}
fn okay() {
    let message = unsafe { CStr::from_ptr(llama_rust_last_error()) };
    assert!(
        message.to_bytes().is_empty(),
        "{}",
        message.to_string_lossy()
    );
}
fn cpath(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rust-llama-native-microbatch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
struct ModelOwner(*mut ModelHandle);
impl ModelOwner {
    fn load(path: &Path) -> Self {
        let path = cpath(path);
        let params = LlamaModelParams {
            n_gpu_layers: 0,
            load_mode: 0,
            use_extra_bufts: 0,
            ..LlamaModelParams::default()
        };
        clean();
        let model = unsafe { llama_model_load_from_file(path.as_ptr(), params) };
        okay();
        assert!(!model.is_null());
        Self(model)
    }
}
impl Drop for ModelOwner {
    fn drop(&mut self) {
        unsafe {
            llama_model_free(self.0);
        }
    }
}
struct ContextOwner(*mut ContextHandle);
impl Drop for ContextOwner {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.0);
        }
    }
}

fn check_parameters(params: &LlamaContextParams, expected: &Value) {
    for (key, value) in [
        ("n_ctx", i64::from(params.n_ctx)),
        ("n_batch", i64::from(params.n_batch)),
        ("n_ubatch", i64::from(params.n_ubatch)),
        ("n_seq_max", i64::from(params.n_seq_max)),
        ("threads", i64::from(params.n_threads)),
        ("threads_batch", i64::from(params.n_threads_batch)),
        ("type_k", i64::from(params.type_k)),
        ("type_v", i64::from(params.type_v)),
        ("flash", i64::from(params.flash_attn_type)),
    ] {
        assert_eq!(value, number(expected, key), "parameter {key}");
    }
    assert_eq!(params.kv_unified != 0, flag(expected, "unified"));
    assert_eq!(params.embeddings != 0, flag(expected, "embeddings"));
}
fn parameters(request: &Value, report: &Value) -> LlamaContextParams {
    let mut params = unsafe { llama_context_default_params() };
    check_parameters(&params, field(report, "default_parameters"));
    if !flag_or(request, "literal_defaults", false) {
        let settings = field(request, "params");
        params.n_ctx = number_or(settings, "n_ctx", 1024).try_into().unwrap();
        params.n_batch = number_or(settings, "n_batch", 512).try_into().unwrap();
        params.n_ubatch = number_or(settings, "n_ubatch", 128).try_into().unwrap();
        params.n_seq_max = number_or(settings, "n_seq_max", 4).try_into().unwrap();
        params.n_threads = number_or(settings, "threads", 1).try_into().unwrap();
        params.n_threads_batch = number_or(settings, "threads_batch", 1).try_into().unwrap();
        params.type_k = number_or(settings, "type_k", 1).try_into().unwrap();
        params.type_v = number_or(settings, "type_v", 1).try_into().unwrap();
        params.flash_attn_type = number_or(settings, "flash", 1).try_into().unwrap();
        params.kv_unified = u8::from(flag_or(settings, "unified", true));
        params.embeddings = u8::from(flag_or(settings, "embeddings", false));
        params.offload_kqv = 0;
        params.op_offload = 0;
        params.no_perf = 1;
    }
    check_parameters(&params, field(report, "requested_parameters"));
    params
}

#[derive(Default, Debug)]
struct Counts {
    cases: usize,
    decode_steps: usize,
    logit_words: usize,
    embedding_words: usize,
    binary_files: usize,
    binary_bytes: usize,
    state_bytes: usize,
    state_files: usize,
}
impl Counts {
    fn check(&self, expected: &Value) {
        for (key, actual) in [
            ("cases", self.cases),
            ("decode_steps", self.decode_steps),
            ("logit_words", self.logit_words),
            ("embedding_words", self.embedding_words),
            ("binary_files", self.binary_files),
            ("binary_bytes", self.binary_bytes),
            ("state_bytes", self.state_bytes),
            ("state_files", self.state_files),
        ] {
            assert_eq!(actual, number(expected, key) as usize, "{key}");
        }
    }
}

fn check_words(pointer: *const f32, expected: &[u8]) {
    assert!(expected.len().is_multiple_of(4));
    if expected.is_empty() {
        return;
    }
    assert!(!pointer.is_null());
    let actual = unsafe { slice::from_raw_parts(pointer, expected.len() / 4) };
    for (index, (actual, word)) in actual.iter().zip(expected.as_chunks::<4>().0).enumerate() {
        assert_eq!(
            actual.to_bits(),
            u32::from_le_bytes(*word),
            "native word {index}"
        );
    }
}

#[allow(clippy::too_many_arguments)] // Explicit immutable fixture and observed counters.
fn decode(
    context: &ContextOwner,
    input: &Value,
    expected: &Value,
    files: &BTreeMap<&str, &'static [u8]>,
    vocabulary: usize,
    width: usize,
    embeddings: bool,
    counts: &mut Counts,
) {
    let entries = array(field(input, "tokens"));
    let mut output_indices = Vec::new();
    clean();
    unsafe {
        let mut batch = llama_batch_init(entries.len() as i32, 0, 16);
        okay();
        assert!(!batch.token.is_null());
        batch.n_tokens = entries.len() as i32;
        for (index, token) in entries.iter().enumerate() {
            let seq = array(field(token, "seq"));
            *batch.token.add(index) = number(token, "id") as i32;
            *batch.pos.add(index) = number(token, "pos") as i32;
            *batch.n_seq_id.add(index) = seq.len() as i32;
            for (part, value) in seq.iter().enumerate() {
                *(*batch.seq_id.add(index)).add(part) = integer(value) as i32;
            }
            let requested = flag_or(token, "output", true);
            *batch.logits.add(index) = i8::from(requested);
            if requested || embeddings {
                output_indices.push(index);
            }
        }
        let code = llama_decode(context.0, batch);
        llama_batch_free(batch);
        assert_eq!(code, number(expected, "code") as i32);
        if code == 0 {
            okay();
        }
        assert_eq!(
            output_indices,
            array(field(expected, "output_indices"))
                .iter()
                .map(|value| integer(value) as usize)
                .collect::<Vec<_>>()
        );
        counts.decode_steps += 1;
        if code != 0 {
            assert!(
                !CStr::from_ptr(llama_rust_last_error())
                    .to_bytes()
                    .is_empty()
            );
            clean();
            if let Some(available) = expected.get("retained_logits_available") {
                let retained = llama_get_logits_ith(context.0, -1);
                assert_eq!(!retained.is_null(), available.as_bool().unwrap());
                if !retained.is_null() {
                    let bytes = files[text(expected, "retained_logits_file")];
                    assert_eq!(
                        bytes.len() / 4,
                        number(expected, "retained_logit_words") as usize
                    );
                    check_words(retained, bytes);
                    counts.logit_words += bytes.len() / 4;
                    counts.binary_files += 1;
                    counts.binary_bytes += bytes.len();
                }
                clean();
            }
            return;
        }
        let logits = files[text(expected, "logits_file")];
        assert_eq!(logits.len(), output_indices.len() * vocabulary * 4);
        assert_eq!(logits.len() / 4, number(expected, "logit_words") as usize);
        for (row, &index) in output_indices.iter().enumerate() {
            let expected_row = &logits[row * vocabulary * 4..(row + 1) * vocabulary * 4];
            check_words(llama_get_logits_ith(context.0, index as i32), expected_row);
            // Negative indices alias the native output order after sequence-oriented
            // microbatch execution; the positive original rows are the oracle bytes.
            check_words(
                llama_get_logits_ith(context.0, row as i32 - output_indices.len() as i32),
                expected_row,
            );
        }
        counts.logit_words += logits.len() / 4;
        counts.binary_files += 1;
        counts.binary_bytes += logits.len();
        if embeddings {
            let values = files[text(expected, "embedding_file")];
            assert_eq!(values.len(), output_indices.len() * width * 4);
            assert_eq!(
                values.len() / 4,
                number(expected, "embedding_words") as usize
            );
            for (row, &index) in output_indices.iter().enumerate() {
                let expected_row = &values[row * width * 4..(row + 1) * width * 4];
                check_words(
                    llama_get_embeddings_ith(context.0, index as i32),
                    expected_row,
                );
                check_words(
                    llama_get_embeddings_ith(context.0, row as i32 - output_indices.len() as i32),
                    expected_row,
                );
            }
            counts.embedding_words += values.len() / 4;
            counts.binary_files += 1;
            counts.binary_bytes += values.len();
        }
        okay();
    }
}

fn replay(model: &ModelOwner, case: &Value, expected: &Value, counts: &mut Counts) {
    let input = field(case, "request");
    let report = field(expected, "report");
    let mut params = parameters(input, report);
    let files: BTreeMap<_, _> = array(field(expected, "files"))
        .iter()
        .map(|file| (text(file, "name"), packed(file)))
        .collect();
    assert_eq!(files.len(), array(field(expected, "files")).len());
    clean();
    let context = ContextOwner(unsafe { llama_init_from_model(model.0, params) });
    assert_eq!(!context.0.is_null(), flag(report, "created"));
    counts.cases += 1;
    if context.0.is_null() {
        return;
    }
    okay();
    let vocabulary = number(report, "vocabulary") as usize;
    let width = number(report, "embedding") as usize;
    unsafe {
        assert_eq!(
            llama_vocab_n_tokens(llama_model_get_vocab(model.0)) as usize,
            vocabulary
        );
        assert_eq!(llama_model_n_embd(model.0) as usize, width);
        let actual = field(report, "actual");
        for (key, value) in [
            ("n_ctx", i64::from(llama_n_ctx(context.0))),
            ("n_ctx_seq", i64::from(llama_n_ctx_seq(context.0))),
            ("n_batch", i64::from(llama_n_batch(context.0))),
            ("n_ubatch", i64::from(llama_n_ubatch(context.0))),
            ("threads", i64::from(llama_n_threads(context.0))),
            ("threads_batch", i64::from(llama_n_threads_batch(context.0))),
        ] {
            assert_eq!(value, number(actual, key), "getter {key}");
        }
    }
    let steps = array(field(input, "steps"));
    let outcomes = array(field(report, "steps"));
    assert_eq!(steps.len(), outcomes.len());
    for (step_index, (step, result)) in steps.iter().zip(outcomes).enumerate() {
        let operation = text(step, "op");
        assert_eq!(operation, text(result, "op"));
        clean();
        unsafe {
            let memory = llama_get_memory(context.0);
            let begin = number_or(step, "begin", -1) as i32;
            let end = number_or(step, "end", -1) as i32;
            match operation {
                "decode" => decode(
                    &context,
                    step,
                    result,
                    &files,
                    vocabulary,
                    width,
                    params.embeddings != 0,
                    counts,
                ),
                "threads" => {
                    llama_set_n_threads(
                        context.0,
                        number(step, "generation") as i32,
                        number(step, "batch") as i32,
                    );
                    assert_eq!(
                        llama_n_threads(context.0),
                        number(result, "generation") as i32
                    );
                    assert_eq!(
                        llama_n_threads_batch(context.0),
                        number(result, "batch") as i32
                    );
                }
                "embeddings" => {
                    params.embeddings = u8::from(flag(step, "enabled"));
                    llama_set_embeddings(context.0, params.embeddings);
                }
                "remove" => assert_eq!(
                    llama_memory_seq_rm(memory, number(step, "seq") as i32, begin, end),
                    flag(result, "result")
                ),
                "copy" => llama_memory_seq_cp(
                    memory,
                    number(step, "source") as i32,
                    number(step, "destination") as i32,
                    begin,
                    end,
                ),
                "shift" => llama_memory_seq_add(
                    memory,
                    number(step, "seq") as i32,
                    begin,
                    end,
                    number(step, "delta") as i32,
                ),
                "clear" => llama_memory_clear(memory, u8::from(flag_or(step, "data", false))),
                other => panic!("unknown original operation {other}"),
            }
            okay();
            let expected_positions = array(field(result, "positions"));
            assert_eq!(expected_positions.len(), params.n_seq_max as usize);
            for (sequence, expected_position) in expected_positions.iter().enumerate() {
                let position = array(expected_position);
                assert_eq!(
                    llama_memory_seq_pos_min(memory, sequence as i32),
                    integer(&position[0]) as i32
                );
                assert_eq!(
                    llama_memory_seq_pos_max(memory, sequence as i32),
                    integer(&position[1]) as i32
                );
            }
            let expected_state = files[text(result, "state_file")];
            assert_eq!(expected_state.len(), number(result, "state_bytes") as usize);
            let length = llama_state_get_size(context.0);
            assert_eq!(length, expected_state.len());
            let mut state = vec![0xa5; length + 16];
            assert_eq!(
                llama_state_get_data(context.0, state.as_mut_ptr(), length),
                length
            );
            okay();
            assert!(state[length..].iter().all(|&byte| byte == 0xa5));
            assert_eq!(
                &state[..length],
                expected_state,
                "native state after step {step_index}"
            );
            counts.binary_files += 1;
            counts.binary_bytes += length;
            counts.state_files += 1;
            counts.state_bytes += length;
        }
    }
}

#[test]
fn original_native_microbatches_defaults_workers_masks_and_exact_state() {
    let document = json::parse(include_str!(
        "../../../fixtures/ffi/native-microbatch-reference.json"
    ))
    .unwrap();
    let directory = Directory::new();
    let models: Vec<_> = array(field(&document, "models"))
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let path = directory.0.join(format!("model-{index}.gguf"));
            assert_eq!(text(model, "kind"), "gguf");
            std::fs::write(&path, packed(model)).unwrap();
            ModelOwner::load(&path)
        })
        .collect();
    assert_eq!(models.len(), 5);
    let cases = array(field(&document, "cases"));
    let expected = array(field(field(&document, "profiles"), PROFILE));
    assert_eq!(cases.len(), 205);
    assert_eq!(cases.len(), expected.len());
    let mut counts = Counts::default();
    for (index, (case, expected)) in cases.iter().zip(expected).enumerate() {
        assert_eq!(number(expected, "case") as usize, index);
        eprintln!("original microbatch {PROFILE}: {}", text(case, "name"));
        replay(
            &models[number(case, "model") as usize],
            case,
            expected,
            &mut counts,
        );
    }
    counts.check(field(field(&document, "counts"), PROFILE));
    assert_eq!(counts.logit_words, 2_398_720);
    assert_eq!(counts.embedding_words, 588);
    assert_eq!(counts.state_files, 319);
    assert_eq!(counts.state_bytes, 1_098_603);
    let checked_errors = array(field(&document, "checked_errors"));
    assert_eq!(checked_errors.len(), 2);
    for case in checked_errors {
        let input = field(field(case, "request"), "params");
        let params = LlamaContextParams {
            n_ctx: number(input, "n_ctx") as u32,
            n_batch: number(input, "n_batch") as u32,
            n_ubatch: number(input, "n_ubatch") as u32,
            n_seq_max: number(input, "n_seq_max") as u32,
            n_threads: number(input, "threads") as i32,
            n_threads_batch: number(input, "threads_batch") as i32,
            kv_unified: u8::from(flag(input, "unified")),
            flash_attn_type: number(input, "flash") as i32,
            offload_kqv: 0,
            op_offload: 0,
            no_perf: 1,
            ..LlamaContextParams::default()
        };
        clean();
        let context =
            unsafe { llama_init_from_model(models[number(case, "model") as usize].0, params) };
        assert!(
            context.is_null(),
            "source assertion must become a checked initialization error"
        );
        let message = unsafe { CStr::from_ptr(llama_rust_last_error()) }.to_string_lossy();
        assert!(message.contains(text(case, "required_error")), "{message}");
        clean();
    }
}
