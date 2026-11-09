//! Exact typed-cache replay of independently captured original native C outputs.
//! Four K/V dtype pairs, unified/separate physical buffers, state/session bytes,
//! continuations and failure side effects; see f16-context-provenance.json.
#![cfg(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    )
))]

use crate::{batch::*, context::*, memory::*, model::*, state::*, *};
use llama_behavior::json::{self, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

const PROMPT: [i32; 5] = [1, 403, 407, 261, 378];
const PACKED_DATA: &[u8] = include_bytes!("../../../fixtures/ffi/f16-context-data.bin");
#[cfg(target_env = "msvc")]
const PROFILE: &str = "msvc";
#[cfg(not(target_env = "msvc"))]
const PROFILE: &str = "gnu";
static CHECKED_LOGIT_WORDS: AtomicUsize = AtomicUsize::new(0);
static CHECKED_STATE_BYTES: AtomicUsize = AtomicUsize::new(0);
static CHECKED_STATE_BLOCKS: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    types: (i32, i32),
    report: Value,
    files: BTreeMap<String, Vec<u8>>,
}
fn packed_bytes(value: &Value) -> &'static [u8] {
    let start = usize::try_from(number(value, "offset")).unwrap();
    let bytes = usize::try_from(number(value, "bytes")).unwrap();
    &PACKED_DATA[start..start.checked_add(bytes).unwrap()]
}
fn fixtures() -> &'static [Fixture] {
    static FIXTURE: OnceLock<Vec<Fixture>> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let document = json::parse(include_str!(
            "../../../fixtures/ffi/f16-context-reference.json"
        ))
        .unwrap();
        let entries = array(field(field(&document, "profiles"), PROFILE));
        assert_eq!(entries.len(), 4);
        entries
            .iter()
            .map(|entry| {
                let mut files = BTreeMap::new();
                for file in array(field(entry, "files")) {
                    assert!(
                        files
                            .insert(text(file, "name").to_owned(), packed_bytes(file).to_vec())
                            .is_none()
                    );
                }
                assert_eq!(files.len(), 130);
                Fixture {
                    types: (
                        number(entry, "type_k") as i32,
                        number(entry, "type_v") as i32,
                    ),
                    report: field(entry, "report").clone(),
                    files,
                }
            })
            .collect()
    })
}
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
fn clean() {
    unsafe {
        llama_rust_clear_error();
    }
}
fn okay() {
    let error = unsafe { CStr::from_ptr(llama_rust_last_error()) };
    assert!(error.to_bytes().is_empty(), "{}", error.to_string_lossy());
}
fn failed() {
    assert!(
        !unsafe { CStr::from_ptr(llama_rust_last_error()) }
            .to_bytes()
            .is_empty()
    );
}
fn c_path(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}
struct ModelOwner(*mut ModelHandle);
impl ModelOwner {
    fn load() -> Self {
        let path = c_path(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/ffi/native-context-model.gguf"
        )));
        let params = LlamaModelParams {
            n_gpu_layers: 0,
            load_mode: 0,
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
impl ContextOwner {
    fn new(model: &ModelOwner, unified: bool, types: (i32, i32)) -> Self {
        let n_ctx = 512;
        let p = LlamaContextParams {
            n_ctx,
            n_seq_max: 4,
            n_batch: 128,
            n_ubatch: 32,
            kv_unified: u8::from(unified),
            n_threads: 1,
            n_threads_batch: 1,
            type_k: types.0,
            type_v: types.1,
            flash_attn_type: 0,
            offload_kqv: 0,
            op_offload: 0,
            no_perf: 1,
            ..LlamaContextParams::default()
        };
        clean();
        let context = unsafe { llama_init_from_model(model.0, p) };
        okay();
        assert!(!context.is_null());
        Self(context)
    }
    fn decode(&self, tokens: &[i32], sequence: i32, start: i32) -> Vec<u32> {
        clean();
        unsafe {
            let mut batch = llama_batch_init(tokens.len() as i32, 0, 1);
            okay();
            batch.n_tokens = tokens.len() as i32;
            for (index, token) in tokens.iter().enumerate() {
                *batch.token.add(index) = *token;
                *batch.pos.add(index) = start + index as i32;
                *batch.n_seq_id.add(index) = 1;
                **batch.seq_id.add(index) = sequence;
                *batch.logits.add(index) = i8::from(index + 1 == tokens.len());
            }
            let result = llama_decode(self.0, batch);
            llama_batch_free(batch);
            okay();
            assert_eq!(result, 0);
            let row = llama_get_logits_ith(self.0, -1);
            okay();
            assert!(!row.is_null());
            slice::from_raw_parts(row, 512)
                .iter()
                .map(|f| f.to_bits())
                .collect()
        }
    }
    fn seed(&self) {
        self.decode(&PROMPT, 0, 0);
        self.decode(&PROMPT.iter().rev().copied().collect::<Vec<_>>(), 1, 0);
    }
    fn positions(&self) -> Vec<Vec<i32>> {
        unsafe {
            let mem = llama_get_memory(self.0);
            (0..4)
                .map(|id| {
                    vec![
                        llama_memory_seq_pos_min(mem, id),
                        llama_memory_seq_pos_max(mem, id),
                    ]
                })
                .collect()
        }
    }
    fn assert_positions(&self, expected: &Value) {
        let expected: Vec<Vec<i32>> = array(expected)
            .iter()
            .map(|row| array(row).iter().map(|v| integer(v) as i32).collect())
            .collect();
        assert_eq!(self.positions(), expected);
    }
    fn state(&self) -> Vec<u8> {
        clean();
        let size = unsafe { llama_state_get_size(self.0) };
        okay();
        assert!(size > 0);
        let mut bytes = vec![0xA5; size + 16];
        let result = unsafe { llama_state_get_data(self.0, bytes.as_mut_ptr(), size) };
        okay();
        assert_eq!(result, size);
        assert!(bytes[size..].iter().all(|b| *b == 0xA5));
        bytes.truncate(size);
        bytes
    }
}
impl Drop for ContextOwner {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.0);
        }
    }
}
fn check_logits(actual: &[u32], expected: &Value) {
    assert_eq!(text(expected, "kind"), "u32");
    let bytes = packed_bytes(expected);
    assert_eq!(actual.len() * 4, bytes.len());
    for (index, (&actual, expected)) in actual.iter().zip(bytes.as_chunks::<4>().0).enumerate() {
        assert_eq!(
            actual,
            u32::from_le_bytes(*expected),
            "native logit word {index}"
        );
    }
    CHECKED_LOGIT_WORDS.fetch_add(actual.len(), Ordering::Relaxed);
}
fn raw_u32(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
}
fn check_encoded(actual: &[u8], expected: &[u8], _template: &[u8], _flags: u32) {
    assert_eq!(actual, expected, "exact native typed state bytes");
    CHECKED_STATE_BYTES.fetch_add(expected.len(), Ordering::Relaxed);
    CHECKED_STATE_BLOCKS.fetch_add(1, Ordering::Relaxed);
}
fn check_state(actual: &[u8], expected: &[u8], flags: u32) {
    check_encoded(actual, expected, expected, flags)
}

fn mutate(context: &ContextOwner, operation: &str, case: &Value) {
    clean();
    unsafe {
        let memory = llama_get_memory(context.0);
        let result = match operation {
            "unchanged" => None,
            "remove_suffix" => Some(llama_memory_seq_rm(memory, 0, 2, -1)),
            "remove_middle" => Some(llama_memory_seq_rm(memory, 0, 1, 3)),
            "remove_prefix" => Some(llama_memory_seq_rm(memory, 0, -1, 2)),
            "remove_all_sequence" => Some(llama_memory_seq_rm(memory, 0, -1, -1)),
            "remove_all_sequences" => Some(llama_memory_seq_rm(memory, -1, 1, 3)),
            "remove_reversed" => Some(llama_memory_seq_rm(memory, 0, 3, 1)),
            "copy_full" => {
                llama_memory_seq_cp(memory, 0, 2, -1, -1);
                None
            }
            "copy_middle" => {
                llama_memory_seq_cp(memory, 0, 2, 1, 3);
                None
            }
            "copy_into_existing" => {
                llama_memory_seq_cp(memory, 0, 1, 1, 3);
                None
            }
            "copy_self" => {
                llama_memory_seq_cp(memory, 0, 0, 1, 3);
                None
            }
            "keep_sequence" => {
                llama_memory_seq_cp(memory, 0, 2, -1, -1);
                llama_memory_seq_keep(memory, 2);
                None
            }
            "keep_missing" => {
                llama_memory_seq_keep(memory, 3);
                None
            }
            "clear_metadata" => {
                llama_memory_clear(memory, 0);
                None
            }
            "clear_data" => {
                llama_memory_clear(memory, 1);
                None
            }
            "shift_all" => {
                llama_memory_seq_add(memory, 0, -1, -1, 3);
                None
            }
            "shift_suffix" => {
                llama_memory_seq_add(memory, 0, 2, -1, 2);
                None
            }
            "shift_negative" => {
                llama_memory_seq_add(memory, 0, -1, -1, -2);
                None
            }
            "shift_shared" => {
                llama_memory_seq_cp(memory, 0, 2, -1, -1);
                llama_memory_seq_add(memory, 0, -1, -1, 2);
                None
            }
            "divide_all" => {
                llama_memory_seq_div(memory, 0, -1, -1, 2);
                None
            }
            "divide_suffix" => {
                llama_memory_seq_div(memory, 0, 2, -1, 2);
                None
            }
            other => panic!("unknown fixture operation {other}"),
        };
        if let Some(result) = result {
            assert_eq!(result, flag(case, "result"));
        }
    }
    okay();
}
fn replay_memory_and_sequence_state(expected: &Fixture) {
    let model = ModelOwner::load();
    let cases = array(field(&expected.report, "cases"));
    assert_eq!(cases.len(), 40);
    for case in cases {
        let name = text(case, "name");
        let unified = flag(case, "kv_unified");
        let context = ContextOwner::new(&model, unified, expected.types);
        context.seed();
        context.assert_positions(field(case, "before"));
        assert_eq!(
            unsafe { llama_memory_can_shift(llama_get_memory(context.0)) },
            flag(case, "can_shift")
        );
        mutate(&context, name.split_once('_').unwrap().1, case);
        context.assert_positions(field(case, "after"));
        let bytes = context.state();
        let native = &expected.files[&format!("{name}.state.bin")];
        check_state(&bytes, native, 0);
        let restored = ContextOwner::new(&model, unified, expected.types);
        assert_eq!(
            unsafe { llama_state_set_data(restored.0, native.as_ptr(), native.len()) },
            bytes.len()
        );
        okay();
        restored.assert_positions(field(case, "after"));
        check_state(&restored.state(), native, 0);
        let token = number(case, "follow_token") as i32;
        let seq = number(case, "follow_sequence") as i32;
        let pos = number(case, "follow_position") as i32;
        check_logits(
            &context.decode(&[token], seq, pos),
            field(case, "continuation_logit_bits"),
        );
        check_logits(
            &restored.decode(&[token], seq, pos),
            field(case, "restored_continuation_logit_bits"),
        );
        context.assert_positions(field(case, "after_follow"));
    }
    let cases = array(field(&expected.report, "sequence_cases"));
    assert_eq!(cases.len(), 40);
    for case in cases {
        let name = text(case, "name");
        let context = ContextOwner::new(&model, flag(case, "kv_unified"), expected.types);
        context.seed();
        context.decode(&PROMPT[..2], 3, 0);
        let source = number(case, "source") as i32;
        let destination = number(case, "destination") as i32;
        let flags = number(case, "flags") as u32;
        context.assert_positions(field(case, "before"));
        clean();
        let size = unsafe { llama_state_seq_get_size_ext(context.0, source, flags) };
        okay();
        assert_eq!(size, number(case, "state_bytes") as usize);
        let mut bytes = vec![0xA5; size + 16];
        assert_eq!(
            unsafe {
                llama_state_seq_get_data_ext(context.0, bytes.as_mut_ptr(), size, source, flags)
            },
            size
        );
        okay();
        assert!(bytes[size..].iter().all(|b| *b == 0xA5));
        bytes.truncate(size);
        check_state(&bytes, &expected.files[&format!("{name}.state.bin")], flags);
        assert_eq!(
            unsafe {
                llama_state_seq_set_data_ext(context.0, bytes.as_ptr(), size, destination, flags)
            },
            number(case, "read_bytes") as usize
        );
        okay();
        context.assert_positions(field(case, "after"));
        check_state(
            &context.state(),
            &expected.files[&format!("{name}.after.bin")],
            0,
        );
        check_logits(
            &context.decode(
                &[number(case, "follow_token") as i32],
                number(case, "follow_sequence") as i32,
                number(case, "follow_position") as i32,
            ),
            field(case, "continuation_logit_bits"),
        );
    }
}
struct TempDirectory(PathBuf);
impl TempDirectory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rust-llama-typed-native-state-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn cpath(&self, name: &str) -> CString {
        c_path(&self.path(name))
    }
}
impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn replay_session_files_and_short_buffers(expected: &Fixture) {
    let model = ModelOwner::load();
    let directory = TempDirectory::new();
    for (name, data) in &expected.files {
        std::fs::write(directory.path(name), data).unwrap();
    }
    let cases = array(field(&expected.report, "file_cases"));
    assert_eq!(cases.len(), 2);
    for case in cases {
        let unified = flag(case, "kv_unified");
        let label = text(case, "name");
        let context = ContextOwner::new(&model, unified, expected.types);
        context.seed();
        assert_eq!(
            unsafe { llama_n_ctx(context.0) },
            number(case, "n_ctx") as u32
        );
        assert_eq!(
            unsafe { llama_n_ctx_seq(context.0) },
            number(case, "n_ctx_seq") as u32
        );
        assert_eq!(
            unsafe { llama_n_batch(context.0) },
            number(case, "n_batch") as u32
        );
        assert_eq!(
            unsafe { llama_n_ubatch(context.0) },
            number(case, "n_ubatch") as u32
        );
        let session = directory.cpath(&format!("{label}.written"));
        assert!(unsafe {
            llama_state_save_file(context.0, session.as_ptr(), PROMPT.as_ptr(), PROMPT.len())
        });
        okay();
        check_state(
            &std::fs::read(session.to_str().unwrap()).unwrap(),
            &expected.files[&format!("{label}.session.bin")],
            0,
        );
        let alias = directory.cpath(&format!("{label}.written.alias"));
        assert!(unsafe {
            llama_save_session_file(context.0, alias.as_ptr(), PROMPT.as_ptr(), PROMPT.len())
        });
        okay();
        check_state(
            &std::fs::read(alias.to_str().unwrap()).unwrap(),
            &expected.files[&format!("{label}.session.bin.alias")],
            0,
        );
        let state = context.state();
        assert_eq!(unsafe { llama_get_state_size(context.0) }, state.len());
        let mut copied = vec![0; state.len()];
        assert_eq!(
            unsafe { llama_copy_state_data(context.0, copied.as_mut_ptr()) },
            state.len()
        );
        assert_eq!(copied, state);
        let restored = ContextOwner::new(&model, unified, expected.types);
        assert_eq!(
            unsafe { llama_set_state_data(restored.0, copied.as_ptr()) },
            state.len()
        );
        okay();
        assert_eq!(restored.positions(), context.positions());
        let mut tokens = [-77; 9];
        let mut count = 999;
        let restored = ContextOwner::new(&model, unified, expected.types);
        assert!(unsafe {
            llama_state_load_file(
                restored.0,
                session.as_ptr(),
                tokens.as_mut_ptr(),
                5,
                &mut count,
            )
        });
        okay();
        assert_eq!(count, 5);
        assert_eq!(tokens[..5], PROMPT);
        assert_eq!(tokens[5..], [-77; 4]);
        restored.assert_positions(field(case, "loaded_positions"));
        check_logits(
            &restored.decode(&[PROMPT[2]], 0, 5),
            field(case, "session_continuation"),
        );
        let restored = ContextOwner::new(&model, unified, expected.types);
        assert!(unsafe {
            llama_load_session_file(
                restored.0,
                session.as_ptr(),
                tokens.as_mut_ptr(),
                5,
                &mut count,
            )
        });
        okay();
        restored.assert_positions(field(case, "alias_positions"));
        let sequence = directory.cpath(&format!("{label}.written.sequence"));
        assert_eq!(
            unsafe {
                llama_state_seq_save_file(context.0, sequence.as_ptr(), 0, PROMPT.as_ptr(), 5)
            },
            number(case, "sequence_file_bytes") as usize
        );
        okay();
        check_state(
            &std::fs::read(sequence.to_str().unwrap()).unwrap(),
            &expected.files[&format!("{label}.sequence.bin")],
            0,
        );
        let restored = ContextOwner::new(&model, unified, expected.types);
        restored.decode(&PROMPT.iter().rev().copied().collect::<Vec<_>>(), 1, 0);
        let prior = restored.positions();
        count = 999;
        assert_eq!(
            unsafe {
                llama_state_seq_load_file(
                    restored.0,
                    sequence.as_ptr(),
                    3,
                    ptr::null_mut(),
                    0,
                    &mut count,
                )
            },
            12
        );
        okay();
        assert_eq!(count, 5);
        assert_eq!(restored.positions(), prior);
        assert_eq!(
            unsafe {
                llama_state_seq_load_file(
                    restored.0,
                    sequence.as_ptr(),
                    3,
                    tokens.as_mut_ptr(),
                    5,
                    &mut count,
                )
            },
            number(case, "sequence_file_bytes") as usize
        );
        okay();
        restored.assert_positions(field(case, "sequence_loaded_positions"));
        check_logits(
            &restored.decode(&[PROMPT[2]], 3, 5),
            field(case, "sequence_continuation"),
        );
        let restored = ContextOwner::new(&model, unified, expected.types);
        restored.decode(&PROMPT[..1], 2, 0);
        count = 999;
        clean();
        assert_eq!(
            unsafe {
                llama_state_load_file(
                    restored.0,
                    session.as_ptr(),
                    tokens.as_mut_ptr(),
                    1,
                    &mut count,
                )
            },
            flag(case, "small_prompt_result")
        );
        failed();
        assert_eq!(count, number(case, "small_prompt_count") as usize);
        restored.assert_positions(field(case, "small_prompt_positions"));
        let restored = ContextOwner::new(&model, unified, expected.types);
        restored.decode(&PROMPT[..1], 2, 0);
        tokens.fill(-77);
        count = 999;
        let trailing = directory.cpath(&format!("{label}.session.bin.trailing"));
        clean();
        assert_eq!(
            unsafe {
                llama_state_load_file(
                    restored.0,
                    trailing.as_ptr(),
                    tokens.as_mut_ptr(),
                    5,
                    &mut count,
                )
            },
            flag(case, "trailing_full_result")
        );
        failed();
        assert_eq!(count, number(case, "trailing_full_count") as usize);
        assert_eq!(
            tokens.to_vec(),
            array(field(case, "trailing_full_tokens"))
                .iter()
                .map(|v| integer(v) as i32)
                .collect::<Vec<_>>()
        );
        restored.assert_positions(field(case, "trailing_full_positions"));
        check_logits(
            &restored.decode(&[PROMPT[2]], 0, 5),
            field(case, "trailing_full_continuation"),
        );
        let restored = ContextOwner::new(&model, unified, expected.types);
        let trailing = directory.cpath(&format!("{label}.sequence.bin.trailing"));
        clean();
        assert_eq!(
            unsafe {
                llama_state_seq_load_file(
                    restored.0,
                    trailing.as_ptr(),
                    3,
                    tokens.as_mut_ptr(),
                    5,
                    &mut count,
                )
            },
            number(case, "trailing_sequence_result") as usize
        );
        okay();
        assert_eq!(count, number(case, "trailing_sequence_count") as usize);
        restored.assert_positions(field(case, "trailing_sequence_positions"));
        for row in array(field(&expected.report, "buffer_cases"))
            .iter()
            .filter(|row| text(row, "name").starts_with(label))
        {
            let capacity = number(row, "capacity") as usize;
            let mut bytes = vec![0xA5; capacity + 16];
            clean();
            let result = unsafe {
                if let Some((_, parts)) = text(row, "name").split_once("_seq_") {
                    let parts: Vec<_> = parts.split('_').collect();
                    llama_state_seq_get_data_ext(
                        context.0,
                        bytes.as_mut_ptr(),
                        capacity,
                        parts[0].parse().unwrap(),
                        parts[2].parse().unwrap(),
                    )
                } else {
                    llama_state_get_data(context.0, bytes.as_mut_ptr(), capacity)
                }
            };
            assert_eq!(result, number(row, "result") as usize);
            let golden = packed_bytes(field(row, "hex"));
            if let Some((_, parts)) = text(row, "name").split_once("_seq_") {
                let parts: Vec<_> = parts.split('_').collect();
                let mode = if unified { "unified" } else { "separate" };
                let template = &expected.files
                    [&format!("{mode}_sequence_{}_flags_{}.state.bin", parts[0], parts[2])];
                check_encoded(&bytes, golden, template, parts[2].parse().unwrap());
            } else {
                let file = &expected.files[&format!("{label}.session.bin")];
                let template = &file[12 + raw_u32(file, 8) * 4..];
                check_encoded(&bytes, golden, template, 0);
            }
            if result == 0 { failed() } else { okay() }
        }
    }
    assert_eq!(array(field(&expected.report, "buffer_cases")).len(), 96);
}

#[test]
fn original_four_dtype_pairs_memory_sessions_and_exact_continuations() {
    CHECKED_LOGIT_WORDS.store(0, Ordering::Relaxed);
    CHECKED_STATE_BYTES.store(0, Ordering::Relaxed);
    CHECKED_STATE_BLOCKS.store(0, Ordering::Relaxed);
    let fixtures = fixtures();
    assert_eq!(
        fixtures.iter().map(|f| f.types).collect::<Vec<_>>(),
        [(0, 0), (1, 1), (1, 0), (0, 1)]
    );
    for expected in fixtures {
        replay_memory_and_sequence_state(expected);
        replay_session_files_and_short_buffers(expected);
    }
    assert_eq!(CHECKED_LOGIT_WORDS.load(Ordering::Relaxed), 258_048);
    assert_eq!(CHECKED_STATE_BLOCKS.load(Ordering::Relaxed), 1_048);
    assert_eq!(CHECKED_STATE_BYTES.load(Ordering::Relaxed), 290_328);
}

#[test]
fn original_defaults_remain_explicitly_outside_nonflash_single_worker_support() {
    let model = ModelOwner::load();
    let defaults = unsafe { llama_context_default_params() };
    assert_eq!((defaults.type_k, defaults.type_v), (1, 1));
    assert_eq!((defaults.n_threads, defaults.n_threads_batch), (4, 4));
    assert_eq!(defaults.flash_attn_type, -1);
    for (flash, workers, batch_workers, key, value) in [
        (-1, 4, 4, 1, 1),
        (-1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (-1, 1, 1, 0, 0),
        (0, 2, 1, 1, 1),
        (0, 1, 2, 1, 1),
        (0, 4, 4, 1, 1),
        (0, 1, 1, 2, 1),
        (0, 1, 1, 1, 2),
    ] {
        let params = LlamaContextParams {
            flash_attn_type: flash,
            n_threads: workers,
            n_threads_batch: batch_workers,
            type_k: key,
            type_v: value,
            no_perf: 1,
            ..defaults
        };
        clean();
        let context = unsafe { llama_init_from_model(model.0, params) };
        assert!(context.is_null());
        failed();
    }
}
