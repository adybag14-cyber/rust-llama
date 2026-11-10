//! Complete replay of the independently captured frozen-header metadata corpus.
use crate::{batch::*, context::*, model::*, model_metadata::*, vocab::*, *};
use llama_behavior::json::{self, Value};
use std::fs;
use std::path::{Component, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MODELS: &[u8] = include_bytes!("../../../fixtures/ffi/model-metadata/models.bin");
#[cfg(target_env = "msvc")]
const REFERENCE: &[u8] = include_bytes!("../../../fixtures/ffi/model-metadata/reference-msvc.bin");
#[cfg(not(target_env = "msvc"))]
const REFERENCE: &[u8] = include_bytes!("../../../fixtures/ffi/model-metadata/reference-gnu.bin");

struct Reader<'a> {
    data: &'a [u8],
    pool: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn plain(data: &'a [u8]) -> Self {
        Self {
            data,
            pool: &[],
            position: 0,
        }
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let end = self.position.checked_add(n).unwrap();
        let value = &self.data[self.position..end];
        self.position = end;
        value
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn blob(&mut self) -> &'a [u8] {
        let start = self.u32() as usize;
        let n = self.u32() as usize;
        &self.pool[start..start.checked_add(n).unwrap()]
    }
    fn corpus(data: &'a [u8]) -> Self {
        let mut outer = Self::plain(data);
        assert_eq!(outer.take(8), b"MMEX0001");
        let length = outer.u64() as usize;
        let records = outer.take(length);
        let length = outer.u64() as usize;
        let pool = outer.take(length);
        assert_eq!(outer.position, data.len());
        Self {
            data: records,
            pool,
            position: 0,
        }
    }
}
fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap()
}
fn number(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}
fn text(value: &Value) -> &str {
    value.as_str().unwrap()
}
fn clear() {
    unsafe { llama_rust_clear_error() }
}
fn error() -> String {
    unsafe { CStr::from_ptr(llama_rust_last_error()) }
        .to_string_lossy()
        .into_owned()
}
fn okay() {
    assert_eq!(error(), "");
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
impl Drop for BatchOwner {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.0) }
    }
}
struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rust-llama-metadata-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
unsafe fn call(
    model: *const ModelHandle,
    operation: u8,
    index: i32,
    key: *const c_char,
    output: *mut c_char,
    capacity: usize,
) -> i32 {
    match operation {
        0 => unsafe { llama_model_meta_key_by_index(model, index, output, capacity) },
        1 => unsafe { llama_model_meta_val_str_by_index(model, index, output, capacity) },
        2 => unsafe { llama_model_meta_val_str(model, key, output, capacity) },
        _ => panic!("unknown metadata operation"),
    }
}
fn full(model: &ModelOwner, operation: u8, index: i32) -> Vec<u8> {
    clear();
    let n = unsafe { call(model.0, operation, index, ptr::null(), ptr::null_mut(), 0) };
    assert!(n >= 0, "{}", error());
    let mut output = vec![0xa5u8; n as usize + 1];
    assert_eq!(
        unsafe {
            call(
                model.0,
                operation,
                index,
                ptr::null(),
                output.as_mut_ptr().cast(),
                output.len(),
            )
        },
        n
    );
    assert_eq!(output.pop(), Some(0));
    okay();
    output
}
#[test]
fn original_metadata_bytes_order_buffers_tokenization_and_continuation_logits() {
    let inputs = json::parse(include_str!(
        "../../../fixtures/ffi/model-metadata/inputs.json"
    ))
    .unwrap();
    let provenance = json::parse(include_str!(
        "../../../fixtures/ffi/model-metadata/provenance.json"
    ))
    .unwrap();
    let input_cases = field(&inputs, "cases").as_array().unwrap();
    let profile = if cfg!(target_env = "msvc") {
        "windows-msvc"
    } else {
        "linux-gnu"
    };
    let statistics = field(field(&provenance, "profiles"), profile);
    let mut reader = Reader::corpus(REFERENCE);
    assert_eq!(reader.u32() as usize, input_cases.len());
    let temp = Temporary::new();
    let (mut admitted, mut calls, mut entries, mut tokens, mut words) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for (case_index, input) in input_cases.iter().enumerate() {
        let name = std::str::from_utf8(reader.blob()).unwrap();
        assert_eq!(name, text(field(input, "name")));
        let vocab_only = reader.u8() != 0;
        let accepted = reader.u8() != 0;
        assert_eq!(vocab_only, field(input, "vocab_only").as_bool().unwrap());
        for _ in 0..reader.u32() {
            let id = reader.i32();
            let present = reader.u8() != 0;
            clear();
            let actual = unsafe { llama_model_meta_key_str(id) };
            assert_eq!(!actual.is_null(), present, "{name} key {id}");
            if present {
                assert_eq!(unsafe { CStr::from_ptr(actual) }.to_bytes(), reader.blob());
                assert_eq!(actual, unsafe { llama_model_meta_key_str(id) });
            }
            okay();
        }
        let directory = temp.0.join(case_index.to_string());
        fs::create_dir(&directory).unwrap();
        for file in field(input, "files").as_array().unwrap() {
            let path = std::path::Path::new(text(field(file, "name")));
            assert_eq!(path.components().count(), 1);
            assert!(path.components().all(|p| matches!(p, Component::Normal(_))));
            let offset = number(field(file, "offset"));
            let size = number(field(file, "bytes"));
            fs::write(directory.join(path), &MODELS[offset..offset + size]).unwrap();
        }
        let path = CString::new(
            directory
                .join(text(field(input, "primary")))
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let params = LlamaModelParams {
            n_gpu_layers: 0,
            load_mode: 0,
            vocab_only: u8::from(vocab_only),
            use_extra_bufts: 0,
            check_tensors: 1,
            ..LlamaModelParams::default()
        };
        clear();
        let pointer = unsafe { llama_model_load_from_file(path.as_ptr(), params) };
        assert_eq!(!pointer.is_null(), accepted, "{name}: {}", error());
        if !accepted {
            assert!(!error().is_empty(), "{name}");
            continue;
        }
        let model = ModelOwner(pointer);
        okay();
        admitted += 1;
        let count = reader.i32();
        assert_eq!(unsafe { llama_model_meta_count(model.0) }, count, "{name}");
        for i in 0..count {
            assert_eq!(full(&model, 0, i), reader.blob(), "{name} key {i}");
            assert_eq!(full(&model, 1, i), reader.blob(), "{name} value {i}");
            entries += 1;
        }
        for _ in 0..reader.u32() {
            let operation = reader.u8();
            let index = reader.i32();
            let capacity = reader.u32() as usize;
            let null = reader.u8() != 0;
            let expected_return = reader.i32();
            let key = reader.blob();
            let expected_buffer = reader.blob();
            let key: Vec<u8> = key.iter().copied().chain([0]).collect();
            let mut output = vec![0xa5u8; if null { 0 } else { capacity + 64 }];
            let target = if null {
                ptr::null_mut()
            } else {
                unsafe { output.as_mut_ptr().add(32).cast() }
            };
            clear();
            let actual = unsafe {
                call(
                    model.0,
                    operation,
                    index,
                    key.as_ptr().cast(),
                    target,
                    capacity,
                )
            };
            assert_eq!(
                actual,
                expected_return,
                "{name} op {operation} index {index} capacity {capacity}: {}",
                error()
            );
            assert_eq!(
                output, expected_buffer,
                "{name} buffer op {operation} index {index} capacity {capacity}"
            );
            okay();
            calls += 1;
        }
        let n_vocab = reader.i32();
        let vocab_type = reader.i32();
        let tensor_bytes = reader.u64();
        let parameters = reader.u64();
        let vocab = unsafe { llama_model_get_vocab(model.0) };
        assert_eq!(unsafe { llama_vocab_n_tokens(vocab) }, n_vocab, "{name}");
        assert_eq!(unsafe { llama_vocab_type(vocab) }, vocab_type, "{name}");
        assert_eq!(unsafe { llama_model_size(model.0) }, tensor_bytes, "{name}");
        assert_eq!(
            unsafe { llama_model_n_params(model.0) },
            parameters,
            "{name}"
        );
        for _ in 0..reader.u32() {
            let input = reader.blob();
            let add = reader.u8();
            let parse = reader.u8();
            let expected: Vec<_> = (0..reader.u32()).map(|_| reader.i32()).collect();
            clear();
            let needed = unsafe {
                llama_tokenize(
                    vocab,
                    input.as_ptr().cast(),
                    input.len() as i32,
                    ptr::null_mut(),
                    0,
                    add,
                    parse,
                )
            };
            assert!(needed <= 0, "{name}");
            let mut actual = vec![0i32; needed.unsigned_abs() as usize];
            let filled = unsafe {
                llama_tokenize(
                    vocab,
                    input.as_ptr().cast(),
                    input.len() as i32,
                    actual.as_mut_ptr(),
                    actual.len() as i32,
                    add,
                    parse,
                )
            };
            assert!(filled >= 0, "{name}: {}", error());
            actual.truncate(filled as usize);
            assert_eq!(actual, expected, "{name} tokenization");
            okay();
            assert_eq!(reader.u32() as usize, actual.len());
            for token in actual {
                let expected = reader.blob();
                let needed =
                    unsafe { llama_token_to_piece(vocab, token, ptr::null_mut(), 0, 0, 1) };
                assert!(needed <= 0);
                let mut actual = vec![0u8; needed.unsigned_abs() as usize];
                let size = unsafe {
                    llama_token_to_piece(
                        vocab,
                        token,
                        actual.as_mut_ptr().cast(),
                        actual.len() as i32,
                        0,
                        1,
                    )
                };
                assert!(size >= 0);
                actual.truncate(size as usize);
                assert_eq!(actual, expected, "{name} piece {token}");
            }
            okay();
            tokens += 1;
        }
        let context = reader.u8();
        if context != 0 {
            let params = LlamaContextParams {
                n_ctx: 64,
                n_batch: 64,
                n_ubatch: 64,
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
            clear();
            let pointer = unsafe { llama_init_from_model(model.0, params) };
            assert_eq!(!pointer.is_null(), context == 1, "{name}: {}", error());
            if context == 1 {
                let context = ContextOwner(pointer);
                let mut batch = BatchOwner(unsafe { llama_batch_init(1, 0, 1) });
                assert!(!batch.0.token.is_null());
                assert!(!batch.0.pos.is_null());
                for position in 0..reader.u32() {
                    let token = reader.i32();
                    let expected_return = reader.i32();
                    let expected = reader.blob();
                    unsafe {
                        batch.0.n_tokens = 1;
                        *batch.0.token = token;
                        *batch.0.pos = position as i32;
                        *batch.0.n_seq_id = 1;
                        **batch.0.seq_id = 0;
                        *batch.0.logits = 1;
                        assert_eq!(llama_decode(context.0, batch.0), expected_return, "{name}");
                        let logits = llama_get_logits_ith(context.0, 0);
                        assert!(!logits.is_null());
                        assert_eq!(expected.len(), n_vocab as usize * 4);
                        for (index, (value, word)) in
                            slice::from_raw_parts(logits, n_vocab as usize)
                                .iter()
                                .zip(expected.as_chunks::<4>().0)
                                .enumerate()
                        {
                            assert_eq!(
                                value.to_bits(),
                                u32::from_le_bytes(*word),
                                "{name} position {position} logit {index}"
                            );
                        }
                    }
                    okay();
                    words += n_vocab as usize;
                }
            }
        }
    }
    assert_eq!(reader.position, reader.data.len());
    assert_eq!(input_cases.len(), 171);
    assert_eq!(admitted, number(field(statistics, "admitted")));
    assert_eq!(calls, number(field(statistics, "calls")));
    assert_eq!(entries, number(field(statistics, "metadata_entries")));
    assert_eq!(tokens, number(field(statistics, "tokenization_cases")));
    assert_eq!(words, number(field(statistics, "logit_words")));
}
