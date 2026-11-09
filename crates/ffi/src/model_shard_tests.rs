#![cfg(any(
    all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]

//! Frozen-header original C API regression for automatic shard discovery.
//! Expected getters, token bytes and logits are independent native captures.
//! See fixtures/ffi/shards-provenance.json and split-list-provenance.json.
use crate::{batch::*, context::*, memory::*, model::*, vocab::*, *};
use llama_behavior::json::{self, Value};
use std::fs;
use std::path::{Component, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const INPUTS: &[u8] = include_bytes!("../../../fixtures/gguf/shards-inputs.bin");
const REFERENCE: &str = include_str!("../../../fixtures/ffi/shards-reference.json");
#[cfg(target_env = "msvc")]
const PROFILE: &str = "msvc";
#[cfg(not(target_env = "msvc"))]
const PROFILE: &str = "gnu";
#[cfg(target_env = "msvc")]
const NUMBERS: &[u8] = include_bytes!("../../../fixtures/ffi/shards-msvc-values.f32le.bin");
#[cfg(not(target_env = "msvc"))]
const NUMBERS: &[u8] = include_bytes!("../../../fixtures/ffi/shards-gnu-values.f32le.bin");

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture array")
}
fn number(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("fixture unsigned value")).unwrap()
}
fn integer(value: &Value) -> i64 {
    value.as_i64().expect("fixture integer")
}
fn text(value: &Value) -> &str {
    value.as_str().expect("fixture string")
}
fn flag(value: &Value) -> bool {
    value.as_bool().expect("fixture bool")
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(value: &Value) -> Vec<u8> {
    let bytes = text(value).as_bytes();
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn clean() {
    unsafe { llama_rust_clear_error() }
}
fn okay() {
    let error = unsafe { CStr::from_ptr(llama_rust_last_error()) };
    assert!(error.to_bytes().is_empty(), "{}", error.to_string_lossy());
}

struct Corpus {
    root: PathBuf,
    document: Value,
}
impl Corpus {
    fn new() -> Self {
        Self::from_data(REFERENCE, INPUTS)
    }
    fn from_data(reference: &str, inputs: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rust-llama-ffi-shards-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let document = json::parse(reference).unwrap();
        for item in array(field(&document, "materializations")) {
            if let Some(profiles) = item.get("profiles")
                && !array(profiles).iter().any(|value| text(value) == PROFILE)
            {
                continue;
            }
            let relative = if let Some(encoded) = item.get("path_hex") {
                let bytes = unhex(encoded);
                #[cfg(unix)]
                {
                    use std::os::unix::ffi::OsStringExt;
                    PathBuf::from(std::ffi::OsString::from_vec(bytes))
                }
                #[cfg(not(unix))]
                {
                    PathBuf::from(String::from_utf8(bytes).unwrap())
                }
            } else {
                PathBuf::from(text(field(item, "path")))
            };
            assert!(
                relative
                    .components()
                    .all(|p| matches!(p, Component::Normal(_)))
            );
            let offset = number(field(item, "offset"));
            let bytes = number(field(item, "bytes"));
            let mut payload = inputs[offset..offset.checked_add(bytes).unwrap()].to_vec();
            if let Some(patch) = item.get("patch").filter(|v| !matches!(v, Value::Null)) {
                let before = unhex(field(patch, "from_hex"));
                let after = unhex(field(patch, "to_hex"));
                assert_eq!(before.len(), after.len());
                let positions: Vec<_> = payload
                    .windows(before.len())
                    .enumerate()
                    .filter_map(|(index, part)| (part == before).then_some(index))
                    .collect();
                assert_eq!(positions.len(), number(field(patch, "occurrences")));
                for position in positions {
                    payload[position..position + before.len()].copy_from_slice(&after);
                }
            }
            let path = root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, payload).unwrap();
        }
        Self { root, document }
    }
}
impl Drop for Corpus {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
struct ModelOwner(*mut ModelHandle);
impl Drop for ModelOwner {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.0) }
    }
}
struct ContextOwner(*mut ContextHandle);
impl Drop for ContextOwner {
    fn drop(&mut self) {
        unsafe { llama_free(self.0) }
    }
}
struct BatchOwner(LlamaBatch);
impl BatchOwner {
    fn new() -> Self {
        let batch = unsafe { llama_batch_init(1, 0, 1) };
        assert!(!batch.token.is_null());
        assert!(!batch.pos.is_null());
        assert!(!batch.n_seq_id.is_null());
        assert!(!batch.seq_id.is_null());
        assert!(!batch.logits.is_null());
        Self(batch)
    }
    fn decode(
        &mut self,
        context: &ContextOwner,
        token: i32,
        position: i32,
        vocab: usize,
    ) -> Vec<f32> {
        unsafe {
            self.0.n_tokens = 1;
            *self.0.token = token;
            *self.0.pos = position;
            *self.0.n_seq_id = 1;
            **self.0.seq_id = 0;
            *self.0.logits = 1;
            assert_eq!(llama_decode(context.0, self.0), 0);
            okay();
            let logits = llama_get_logits_ith(context.0, 0);
            assert!(!logits.is_null());
            slice::from_raw_parts(logits, vocab).to_vec()
        }
    }
}
impl Drop for BatchOwner {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.0) }
    }
}
fn compare_row(actual: &[f32], block: &Value, row: usize, numbers: &[u8]) {
    assert!(row < number(field(block, "rows")));
    assert_eq!(actual.len(), number(field(block, "columns")));
    let start = number(field(block, "offset")) + row * actual.len() * 4;
    for (column, (actual, expected)) in actual
        .iter()
        .zip(numbers[start..start + actual.len() * 4].as_chunks::<4>().0)
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            u32::from_le_bytes(*expected),
            "row {row} column {column}"
        );
    }
}
fn piece(vocab: *const VocabHandle, token: i32, special: bool) -> Vec<u8> {
    unsafe {
        let count = llama_token_to_piece(vocab, token, ptr::null_mut(), 0, 0, u8::from(special));
        assert!(count <= 0);
        let mut output = vec![0u8; count.unsigned_abs() as usize];
        assert_eq!(
            llama_token_to_piece(
                vocab,
                token,
                output.as_mut_ptr().cast(),
                output.len() as i32,
                0,
                u8::from(special)
            ),
            output.len() as i32
        );
        output
    }
}
fn compare_model(model: &ModelOwner, expected: &Value, vocab_only: bool, numbers: &[u8]) -> usize {
    unsafe {
        assert_eq!(
            llama_model_size(model.0),
            number(field(expected, "model_tensor_bytes")) as u64
        );
        assert_eq!(
            llama_model_n_params(model.0),
            number(field(expected, "model_parameters")) as u64
        );
        assert_eq!(
            llama_model_ftype(model.0) as i64,
            integer(field(expected, "model_ftype"))
        );
        assert_eq!(
            llama_model_n_ctx_train(model.0) as i64,
            integer(field(expected, "n_ctx_train"))
        );
        assert_eq!(
            llama_model_n_embd(model.0) as i64,
            integer(field(expected, "n_embd"))
        );
        assert_eq!(
            llama_model_n_layer(model.0) as i64,
            integer(field(expected, "n_layer"))
        );
        assert_eq!(
            llama_model_rope_type(model.0) as i64,
            integer(field(expected, "rope_type"))
        );
        let mut description = [0 as c_char; 512];
        let count = llama_model_desc(model.0, description.as_mut_ptr(), description.len());
        assert_eq!(
            count as usize,
            text(field(expected, "model_description")).len()
        );
        assert_eq!(
            CStr::from_ptr(description.as_ptr()).to_bytes(),
            text(field(expected, "model_description")).as_bytes()
        );
        let template = llama_model_chat_template(model.0, ptr::null());
        match field(expected, "chat_template") {
            Value::Null => assert!(template.is_null()),
            value => {
                assert!(!template.is_null());
                assert_eq!(CStr::from_ptr(template).to_bytes(), text(value).as_bytes());
            }
        }
        let vocab = llama_model_get_vocab(model.0);
        assert!(!vocab.is_null());
        let n_vocab = number(field(expected, "n_vocab"));
        assert_eq!(llama_vocab_n_tokens(vocab) as usize, n_vocab);
        assert_eq!(
            llama_vocab_type(vocab) as i64,
            integer(field(expected, "vocab_type"))
        );
        let context = if vocab_only {
            None
        } else {
            let n_ctx = number(field(expected, "n_ctx")) as u32;
            let params = LlamaContextParams {
                n_ctx,
                n_batch: n_ctx,
                n_ubatch: n_ctx,
                n_threads: 1,
                n_threads_batch: 1,
                type_k: 0,
                type_v: 0,
                flash_attn_type: 0,
                offload_kqv: 0,
                op_offload: 0,
                no_perf: 1,
                ..LlamaContextParams::default()
            };
            let context = llama_init_from_model(model.0, params);
            okay();
            assert!(!context.is_null());
            Some(ContextOwner(context))
        };
        let mut words = 0;
        let mut batch = BatchOwner::new();
        for item in array(field(expected, "cases")) {
            let prompt = text(field(item, "prompt"));
            let count = llama_tokenize(
                vocab,
                prompt.as_ptr().cast(),
                prompt.len() as i32,
                ptr::null_mut(),
                0,
                u8::from(flag(field(item, "add_special"))),
                u8::from(flag(field(item, "parse_special"))),
            );
            assert!(count <= 0);
            let mut tokens = vec![0; count.unsigned_abs() as usize];
            assert_eq!(
                llama_tokenize(
                    vocab,
                    prompt.as_ptr().cast(),
                    prompt.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    u8::from(flag(field(item, "add_special"))),
                    u8::from(flag(field(item, "parse_special")))
                ),
                tokens.len() as i32
            );
            let expected_tokens: Vec<i32> = array(field(item, "tokens"))
                .iter()
                .map(|v| integer(v) as i32)
                .collect();
            assert_eq!(tokens, expected_tokens);
            for (special, key) in [(false, "pieces_hex"), (true, "pieces_special_hex")] {
                for (&token, expected_piece) in tokens.iter().zip(array(field(item, key))) {
                    assert_eq!(hex(&piece(vocab, token, special)), text(expected_piece));
                }
            }
            let count = llama_detokenize(
                vocab,
                tokens.as_ptr(),
                tokens.len() as i32,
                ptr::null_mut(),
                0,
                1,
                0,
            );
            assert!(count <= 0);
            let mut decoded = vec![0u8; count.unsigned_abs() as usize];
            let count = llama_detokenize(
                vocab,
                tokens.as_ptr(),
                tokens.len() as i32,
                decoded.as_mut_ptr().cast(),
                decoded.len() as i32,
                1,
                0,
            );
            assert!(count >= 0);
            decoded.truncate(count as usize);
            assert_eq!(hex(&decoded), text(field(item, "detokenized_hex")));
            if let Some(context) = &context {
                llama_memory_clear(llama_get_memory(context.0), 1);
                let prompt_rows = field(item, "prompt_logits");
                assert_eq!(tokens.len(), number(field(prompt_rows, "rows")));
                let mut logits = Vec::new();
                for (position, &token) in tokens.iter().enumerate() {
                    logits = batch.decode(context, token, position as i32, n_vocab);
                    compare_row(&logits, prompt_rows, position, numbers);
                    words += n_vocab;
                }
                let greedy = array(field(item, "greedy_tokens"));
                let rows = field(item, "greedy_logits");
                assert_eq!(greedy.len(), number(field(rows, "rows")));
                let mut generated = Vec::new();
                let mut ended = false;
                for (index, expected_token) in greedy.iter().enumerate() {
                    compare_row(&logits, rows, index, numbers);
                    words += n_vocab;
                    let mut selected = 0;
                    for id in 1..logits.len() {
                        if logits[id] > logits[selected] {
                            selected = id;
                        }
                    }
                    assert_eq!(selected as i64, integer(expected_token));
                    let bytes = piece(vocab, selected as i32, false);
                    assert_eq!(
                        hex(&bytes),
                        text(&array(field(item, "greedy_pieces_hex"))[index])
                    );
                    if llama_vocab_is_eog(vocab, selected as i32) {
                        ended = true;
                        break;
                    }
                    generated.extend(bytes);
                    if index + 1 < greedy.len() {
                        logits = batch.decode(
                            context,
                            selected as i32,
                            (tokens.len() + index) as i32,
                            n_vocab,
                        );
                    }
                }
                assert_eq!(ended, flag(field(item, "finished_eog")));
                assert_eq!(hex(&generated), text(field(item, "generated_hex")));
            }
            okay();
        }
        words
    }
}
#[test]
fn original_auto_shard_loading_getters_and_inference() {
    let corpus = Corpus::new();
    let mut accepted = 0;
    let mut rejected = 0;
    let mut words = 0;
    for case in array(field(&corpus.document, "cases")) {
        let name = text(field(case, "name"));
        let path = CString::new(
            corpus
                .root
                .join(text(field(case, "path")))
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let vocab_only = flag(field(case, "vocab_only"));
        let params = LlamaModelParams {
            n_gpu_layers: 0,
            load_mode: 0,
            vocab_only: u8::from(vocab_only),
            use_extra_bufts: 0,
            check_tensors: 1,
            ..LlamaModelParams::default()
        };
        clean();
        let pointer = unsafe { llama_model_load_from_file(path.as_ptr(), params) };
        let should_load = flag(field(field(case, "accepted"), PROFILE));
        assert_eq!(
            !pointer.is_null(),
            should_load,
            "{name}: {}",
            unsafe { CStr::from_ptr(llama_rust_last_error()) }.to_string_lossy()
        );
        if !should_load {
            assert!(
                !unsafe { CStr::from_ptr(llama_rust_last_error()) }
                    .to_bytes()
                    .is_empty(),
                "{name}"
            );
            rejected += 1;
            continue;
        }
        okay();
        let model = ModelOwner(pointer);
        let index = number(field(field(case, "reference"), PROFILE));
        let expected = &array(field(field(&corpus.document, "references"), PROFILE))[index];
        words += compare_model(&model, expected, vocab_only, NUMBERS);
        accepted += 1;
    }
    assert_eq!((accepted, rejected, words), (53, 43, 163_840));
}

// Reuse the same getter/token/logit assertions for the explicit-list entry point.
#[path = "model_split_list_tests.rs"]
mod split_list_tests;
