//! Frozen-header C oracle replay. Expected bytes were captured from the original
//! scalar library, not from this implementation; see fixtures/ffi provenance.
use crate::{batch::*, context::*, memory::*, model::*, state::*, *};
use llama_behavior::json::{self, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

const PROMPT: [i32; 5] = [1, 403, 407, 261, 378];
struct Fixture {
    report: Value,
    files: BTreeMap<String, Vec<u8>>,
}
fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        #[cfg(target_env = "msvc")]
        let data = include_bytes!("../../../fixtures/ffi/native-context-reference.bin");
        #[cfg(not(target_env = "msvc"))]
        let data = include_bytes!("../../../fixtures/ffi/native-context-reference-linux.bin");
        let mut at = 8;
        assert_eq!(&data[..8], b"NCTXREF1");
        fn take<'a>(data: &'a [u8], at: &mut usize, len: usize) -> &'a [u8] {
            let result = &data[*at..*at + len];
            *at += len;
            result
        }
        fn length(data: &[u8], at: &mut usize) -> usize {
            u32::from_le_bytes(take(data, at, 4).try_into().unwrap()) as usize
        }
        let n = length(data, &mut at);
        let report = json::parse(std::str::from_utf8(take(data, &mut at, n)).unwrap()).unwrap();
        let n = length(data, &mut at);
        let mut files = BTreeMap::new();
        for _ in 0..n {
            let len = length(data, &mut at);
            let name = std::str::from_utf8(take(data, &mut at, len))
                .unwrap()
                .to_owned();
            let len = length(data, &mut at);
            assert!(
                files
                    .insert(name, take(data, &mut at, len).to_vec())
                    .is_none()
            );
        }
        assert_eq!(at, data.len());
        Fixture { report, files }
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
    fn new(model: &ModelOwner, unified: bool) -> Self {
        Self::with_context_tokens(model, unified, 512)
    }
    fn with_context_tokens(model: &ModelOwner, unified: bool, n_ctx: u32) -> Self {
        let p = LlamaContextParams {
            n_ctx,
            n_seq_max: 4,
            n_batch: 128,
            n_ubatch: 32,
            kv_unified: u8::from(unified),
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
    let expected: Vec<u32> = array(expected).iter().map(|v| integer(v) as u32).collect();
    assert_eq!(actual.len(), expected.len());
    for (&actual, expected) in actual.iter().zip(expected) {
        check_float_bits(actual, expected);
    }
}

// GNU source kernels select an SSE2 vector exponential; the frozen MSVC scalar
// reference does not. Keep structural bytes exact and use the existing model
// conformance bound only for generated floating-point values on non-MSVC hosts.
const ABSOLUTE_ERROR: f32 = 2e-4;
const RELATIVE_ERROR: f32 = 2e-5;
fn check_float_bits(actual: u32, expected: u32) {
    if cfg!(target_env = "msvc") {
        assert_eq!(actual, expected, "frozen MSVC float bits");
        return;
    }
    let a = f32::from_bits(actual);
    let e = f32::from_bits(expected);
    assert!(
        a.is_finite()
            && e.is_finite()
            && (a - e).abs() <= ABSOLUTE_ERROR + RELATIVE_ERROR * e.abs(),
        "native float {a} versus {e}: bits {actual:08x}/{expected:08x}"
    );
}
fn raw_u32(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
}
fn float_offsets(raw: &[u8], mut flags: u32) -> Vec<usize> {
    let magic = raw_u32(raw, 0);
    let (mut at, full) = match magic {
        0xaf143cd8 => {
            if flags & 1 != 0 {
                return Vec::new();
            }
            (8, false)
        }
        0x6767736e => {
            flags = 0;
            (12 + raw_u32(raw, 8) * 4, true)
        }
        0x67677371 => {
            flags = 0;
            (12 + raw_u32(raw, 8) * 4, false)
        }
        _ => {
            flags = 0;
            (0, true)
        }
    };
    if full {
        let n = raw_u32(raw, at);
        at += 4;
        assert_eq!(&raw[at..at + n], b"llama");
        at += n;
    }
    let streams = raw_u32(raw, at);
    at += 4;
    let mut offsets = Vec::new();
    for _ in 0..streams {
        let cells = raw_u32(raw, at);
        at += 4;
        if cells == 0 {
            continue;
        }
        for _ in 0..cells {
            let ids = raw_u32(raw, at + 4);
            at += 8 + ids * 4;
        }
        assert_eq!(raw_u32(raw, at), 1);
        let layers = raw_u32(raw, at + 4);
        at += 16;
        for _ in 0..layers {
            assert_eq!(raw_u32(raw, at), 0);
            let row = u64::from_le_bytes(raw[at + 4..at + 12].try_into().unwrap()) as usize;
            at += 12;
            if flags & 2 == 0 {
                offsets.extend((at..at + cells * row).step_by(4));
                at += cells * row;
            }
        }
        for _ in 0..layers {
            assert_eq!(raw_u32(raw, at), 0);
            assert_eq!(raw_u32(raw, at + 4), 4);
            let width = raw_u32(raw, at + 8);
            at += 12;
            if flags & 2 == 0 {
                offsets.extend((at..at + cells * width * 4).step_by(4));
                at += cells * width * 4;
            }
        }
    }
    assert!(at == raw.len() || at + 3 == raw.len());
    offsets
}
fn check_encoded(actual: &[u8], expected: &[u8], template: &[u8], flags: u32) {
    assert_eq!(actual.len(), expected.len(), "native byte length");
    if cfg!(target_env = "msvc") {
        assert_eq!(actual, expected, "native state bytes");
        return;
    }
    let mut numerical = vec![false; expected.len()];
    for offset in float_offsets(template, flags) {
        if offset + 4 > expected.len()
            || expected[offset..offset + 4] != template[offset..offset + 4]
        {
            continue;
        }
        check_float_bits(
            raw_u32(actual, offset) as u32,
            raw_u32(expected, offset) as u32,
        );
        numerical[offset..offset + 4].fill(true);
    }
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !numerical[i] {
            assert_eq!(a, e, "metadata/prefix/sentinel byte {i}");
        }
    }
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
#[test]
fn native_memory_and_sequence_state_match_original_bytes_and_continuations() {
    let expected = fixture();
    let model = ModelOwner::load();
    let cases = array(field(&expected.report, "cases"));
    assert_eq!(cases.len(), 40);
    for case in cases {
        let name = text(case, "name");
        let unified = flag(case, "kv_unified");
        let context = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
        assert_eq!(
            unsafe { llama_state_set_data(restored.0, native.as_ptr(), native.len()) },
            bytes.len()
        );
        okay();
        restored.assert_positions(field(case, "after"));
        assert_eq!(
            restored.state(),
            *native,
            "native raw state must roundtrip bit-exactly without evaluation"
        );
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
        let context = ContextOwner::new(&model, flag(case, "kv_unified"));
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
            "rust-llama-native-state-{}-{}",
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
fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn native_session_files_aliases_short_buffers_and_failure_side_effects() {
    let expected = fixture();
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
        let context = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
        assert_eq!(
            unsafe { llama_set_state_data(restored.0, copied.as_ptr()) },
            state.len()
        );
        okay();
        assert_eq!(restored.positions(), context.positions());
        let mut tokens = [-77; 9];
        let mut count = 999;
        let restored = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
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
        let restored = ContextOwner::new(&model, unified);
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
            let golden = unhex(text(row, "hex"));
            if let Some((_, parts)) = text(row, "name").split_once("_seq_") {
                let parts: Vec<_> = parts.split('_').collect();
                let mode = if unified { "unified" } else { "separate" };
                let template = &expected.files
                    [&format!("{mode}_sequence_{}_flags_{}.state.bin", parts[0], parts[2])];
                check_encoded(&bytes, &golden, template, parts[2].parse().unwrap());
            } else {
                let file = &expected.files[&format!("{label}.session.bin")];
                let template = &file[12 + raw_u32(file, 8) * 4..];
                check_encoded(&bytes, &golden, template, 0);
            }
            if result == 0 { failed() } else { okay() }
        }
    }
    assert_eq!(array(field(&expected.report, "buffer_cases")).len(), 96);
}

#[test]
fn malformed_state_is_atomic_and_device_state_requires_its_owner() {
    let model = ModelOwner::load();
    let context = ContextOwner::new(&model, true);
    context.seed();
    let original = context.state();
    // Every truncated prefix is rejected and cannot commit partial metadata/KV.
    for end in 0..original.len() {
        clean();
        assert_eq!(
            unsafe { llama_state_set_data(context.0, original.as_ptr(), end) },
            0
        );
        failed();
        assert_eq!(context.state(), original, "truncation {end}");
    }
    for offset in [0usize, 9, 13, 21] {
        let mut malformed = original.clone();
        malformed[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        clean();
        assert_eq!(
            unsafe { llama_state_set_data(context.0, malformed.as_ptr(), malformed.len()) },
            0
        );
        failed();
        assert_eq!(context.state(), original, "oversized field {offset}");
    }
    let size = unsafe { llama_state_seq_get_size_ext(context.0, 0, 2) };
    let mut device = vec![0; size];
    clean();
    assert_eq!(
        unsafe { llama_state_seq_get_data_ext(context.0, device.as_mut_ptr(), size, 0, 2) },
        size
    );
    okay();
    let other = ContextOwner::new(&model, true);
    let other_before = other.state();
    clean();
    assert_eq!(
        unsafe { llama_state_seq_set_data_ext(other.0, device.as_ptr(), size, 3, 2) },
        0
    );
    failed();
    assert_eq!(other.state(), other_before);
    let mut host = original.clone();
    host.extend_from_slice(&[0x13, 0x37]);
    clean();
    assert_eq!(
        unsafe { llama_state_set_data(context.0, host.as_ptr(), host.len()) },
        original.len()
    );
    okay();
    assert_eq!(context.state(), original);
    clean();
    assert_eq!(
        unsafe { llama_state_get_data(context.0, ptr::null_mut(), original.len()) },
        0
    );
    failed();
    assert_eq!(context.state(), original);
    let separate = ContextOwner::new(&model, false);
    separate.seed();
    let before = separate.state();
    clean();
    unsafe { llama_memory_seq_cp(llama_get_memory(separate.0), 0, 2, 1, 3) };
    failed();
    assert_eq!(separate.state(), before);
}

#[test]
#[cfg(any(
    target_env = "msvc",
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
fn original_physical_attention_rows_match_boundary_holes_and_shared_masks_bitwise() {
    #[cfg(target_env = "msvc")]
    let data = include_bytes!("../../../fixtures/ffi/native-attention-layout-msvc.bin");
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    let data = include_bytes!("../../../fixtures/ffi/native-attention-layout-gnu.bin");
    assert_eq!(&data[..8], b"ATTNROW1");
    let metadata_bytes = raw_u32(data, 8);
    let metadata =
        json::parse(std::str::from_utf8(&data[12..12 + metadata_bytes]).unwrap()).unwrap();
    let payload = &data[12 + metadata_bytes..];
    fn expected<'a>(descriptor: &Value, payload: &'a [u8]) -> &'a [u8] {
        let offset = number(descriptor, "offset") as usize;
        let length = number(descriptor, "length") as usize;
        &payload[offset..offset + length]
    }
    fn bytes_equal(actual: &[u8], expected: &[u8], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} byte count");
        let difference = actual.iter().zip(expected).position(|(a, b)| a != b);
        assert!(
            difference.is_none(),
            "{label} first differing byte {difference:?}"
        );
    }
    fn seed_long(context: &ContextOwner, sequence: i32, count: usize) {
        for first in (0..count).step_by(32) {
            let tokens = (first..(first + 32).min(count))
                .map(|i| PROMPT[(i + sequence as usize) % PROMPT.len()])
                .collect::<Vec<_>>();
            context.decode(&tokens, sequence, first as i32);
        }
    }
    let model = ModelOwner::load();
    let cases = array(field(&metadata, "cases"));
    assert_eq!(cases.len(), 10);
    let mut checked_logits = 0;
    let mut checked_state_bytes = 0;
    for case in cases {
        let name = text(case, "name");
        let unified = flag(case, "unified");
        let operation = name.split_once('_').unwrap().1;
        let context =
            ContextOwner::with_context_tokens(&model, unified, if unified { 512 } else { 2048 });
        let count = if operation == "dense_boundary" {
            257
        } else {
            260
        };
        seed_long(&context, 0, count);
        if matches!(operation, "physical_holes" | "shared_membership_mask") {
            seed_long(&context, 1, 17);
        }
        if operation == "common_stream_width" {
            seed_long(&context, 1, 3);
        }
        clean();
        let memory = unsafe { llama_get_memory(context.0) };
        match operation {
            "physical_holes" => {
                for (start, end) in [(1, 2), (3, 4), (64, 66), (127, 130), (254, 257)] {
                    assert!(unsafe { llama_memory_seq_rm(memory, 0, start, end) });
                }
            }
            "high_used_prefix_removed" => {
                assert!(unsafe { llama_memory_seq_rm(memory, 0, 0, 253) });
            }
            "shared_cells" => unsafe { llama_memory_seq_cp(memory, 0, 2, -1, -1) },
            "shared_membership_mask" => unsafe {
                llama_memory_seq_cp(memory, 0, 2, -1, -1);
                assert!(llama_memory_seq_rm(memory, 0, 0, 253));
            },
            "dense_boundary" | "common_stream_width" => {}
            _ => panic!("unknown frozen layout case {operation}"),
        }
        okay();
        context.assert_positions(field(case, "positions_before"));
        let reference = expected(field(case, "state_before"), payload);
        bytes_equal(&context.state(), reference, name);
        checked_state_bytes += reference.len();
        let items: Vec<(i32, i32, Vec<i32>)> = match operation {
            "common_stream_width" => vec![(403, count as i32, vec![0]), (407, 3, vec![1])],
            "shared_membership_mask" => {
                vec![(403, count as i32, vec![0]), (407, count as i32, vec![2])]
            }
            _ => (0..8)
                .map(|i| {
                    (
                        PROMPT[i % PROMPT.len()],
                        (count + i) as i32,
                        if operation == "shared_cells" {
                            vec![0, 2]
                        } else {
                            vec![0]
                        },
                    )
                })
                .collect(),
        };
        let rows = array(field(case, "logits"));
        assert_eq!(rows.len(), items.len());
        clean();
        unsafe {
            let mut batch = llama_batch_init(items.len() as i32, 0, 2);
            okay();
            batch.n_tokens = items.len() as i32;
            for (index, (token, position, sequences)) in items.iter().enumerate() {
                *batch.token.add(index) = *token;
                *batch.pos.add(index) = *position;
                *batch.n_seq_id.add(index) = sequences.len() as i32;
                for (offset, sequence) in sequences.iter().enumerate() {
                    *(*batch.seq_id.add(index)).add(offset) = *sequence;
                }
                *batch.logits.add(index) = 1;
            }
            let code = llama_decode(context.0, batch);
            llama_batch_free(batch);
            okay();
            assert_eq!(code, 0, "{name}");
            for (index, descriptor) in rows.iter().enumerate() {
                let row = llama_get_logits_ith(context.0, index as i32);
                okay();
                assert!(!row.is_null());
                let actual = slice::from_raw_parts(row, 512)
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<_>>();
                bytes_equal(&actual, expected(descriptor, payload), name);
                checked_logits += 512;
            }
        }
        context.assert_positions(field(case, "positions_after"));
        let reference = expected(field(case, "state_after"), payload);
        bytes_equal(&context.state(), reference, name);
        checked_state_bytes += reference.len();
    }
    assert_eq!(checked_logits, 31_744);
    assert_eq!(checked_state_bytes, 193_868);
}
