use super::*;
use crate::adapter::*;
use crate::batch::llama_batch_get_one;
use crate::context::encoder::llama_encode;
use crate::context::*;
use crate::model::*;
use crate::model_metadata::llama_model_meta_count;
use crate::vocab::llama_vocab_n_tokens;
use crate::*;
use llama_behavior::json::{self, Value};
use std::path::{Path, PathBuf};

unsafe extern "C" {
    fn tmpfile() -> *mut c_void;
    fn fclose(stream: *mut c_void) -> i32;
    fn fwrite(data: *const c_void, size: usize, count: usize, stream: *mut c_void) -> usize;
    fn fread(data: *mut c_void, size: usize, count: usize, stream: *mut c_void) -> usize;
}
struct Stream(*mut c_void);
impl Stream {
    fn embedded(data: &[u8], prefix: usize, suffix: bool) -> Self {
        let pointer = unsafe { tmpfile() };
        assert!(!pointer.is_null());
        let stream = Self(pointer);
        let pad = vec![0xa7_u8; prefix];
        for bytes in [&pad[..], data] {
            assert_eq!(
                unsafe { fwrite(bytes.as_ptr().cast(), 1, bytes.len(), stream.0) },
                bytes.len()
            );
        }
        if suffix {
            let bytes: Vec<u8> = (0..64).collect();
            assert_eq!(
                unsafe { fwrite(bytes.as_ptr().cast(), 1, bytes.len(), stream.0) },
                bytes.len()
            );
        }
        unsafe { platform::restore(stream.0, 0).unwrap() };
        // Deliberately establish CRT read-ahead before offset loading.
        let mut prefix_bytes = vec![0_u8; prefix];
        assert_eq!(
            unsafe { fread(prefix_bytes.as_mut_ptr().cast(), 1, prefix, stream.0) },
            prefix
        );
        assert_eq!(prefix_bytes, pad);
        assert_eq!(
            unsafe { platform::position(stream.0).unwrap() },
            prefix as i64
        );
        stream
    }
    fn check_live_at_origin(&self, origin: usize) {
        assert_eq!(
            unsafe { platform::position(self.0).unwrap() },
            origin as i64
        );
        let mut magic = [0_u8; 4];
        assert_eq!(unsafe { fread(magic.as_mut_ptr().cast(), 1, 4, self.0) }, 4);
        assert_eq!(&magic, b"GGUF");
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        assert_eq!(unsafe { fclose(self.0) }, 0);
    }
}
struct ModelOwner(*mut ModelHandle);
impl Drop for ModelOwner {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.0) };
    }
}
struct ContextOwner(*mut ContextHandle);
impl Drop for ContextOwner {
    fn drop(&mut self) {
        unsafe { llama_free(self.0) };
    }
}
struct AdapterOwner(*mut AdapterHandle);
impl Drop for AdapterOwner {
    fn drop(&mut self) {
        unsafe { llama_adapter_lora_free(self.0) };
    }
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value.get(name).unwrap_or_else(|| panic!("missing {name}"))
}
fn integer(value: &Value, name: &str) -> i32 {
    field(value, name).as_i64().unwrap().try_into().unwrap()
}
fn boolean(value: &Value, name: &str) -> bool {
    field(value, name).as_bool().unwrap()
}
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/ffi")
        .join(name)
}
fn error() -> String {
    unsafe { CStr::from_ptr(llama_rust_last_error()) }
        .to_string_lossy()
        .into_owned()
}
fn params() -> LlamaModelParams {
    LlamaModelParams {
        n_gpu_layers: 0,
        load_mode: 0,
        use_extra_bufts: 0,
        ..LlamaModelParams::default()
    }
}
fn context(model: &ModelOwner, encoder: bool) -> ContextOwner {
    let result = ContextOwner(unsafe {
        llama_init_from_model(
            model.0,
            LlamaContextParams {
                n_ctx: 32,
                n_batch: 32,
                n_ubatch: 32,
                n_threads: 1,
                n_threads_batch: 1,
                type_k: 0,
                type_v: 0,
                flash_attn_type: 0,
                no_perf: 1,
                embeddings: u8::from(encoder),
                pooling_type: 0,
                ..LlamaContextParams::default()
            },
        )
    });
    assert!(!result.0.is_null(), "{}", error());
    result
}
fn execute(context: &ContextOwner, encoder: bool, expected: &Value, corpus: &Value) {
    let mut tokens = [1, 2, 3];
    let batch = unsafe { llama_batch_get_one(tokens.as_mut_ptr(), 3) };
    let status = unsafe {
        if encoder {
            llama_encode(context.0, batch)
        } else {
            llama_decode(context.0, batch)
        }
    };
    assert_eq!(status, integer(expected, "status"), "{}", error());
    let values =
        &field(corpus, "outputs").as_array().unwrap()[integer(expected, "output") as usize];
    let values = values.as_array().unwrap();
    let output = unsafe {
        if encoder {
            llama_get_embeddings(context.0)
        } else {
            llama_get_logits_ith(context.0, -1)
        }
    };
    assert!(!output.is_null(), "{}", error());
    for (i, (value, expected)) in unsafe { slice::from_raw_parts(output, values.len()) }
        .iter()
        .zip(values)
        .enumerate()
    {
        assert_eq!(
            value.to_bits(),
            expected.as_u64().unwrap() as u32,
            "output word {i}"
        );
    }
}

#[test]
fn original_embedded_model_and_lora_file_pointer_cases() {
    let reference = json::parse(if cfg!(target_env = "msvc") {
        include_str!("../../../../fixtures/ffi/file-pointer/msvc-reference.json")
    } else {
        include_str!("../../../../fixtures/ffi/file-pointer/gnu-reference.json")
    })
    .unwrap();
    for case in field(&reference, "models").as_array().unwrap() {
        let encoder = field(case, "family").as_str().unwrap() == "bert";
        let path = fixture(if encoder {
            "encoder/plain.gguf"
        } else {
            "adapters/inputs/model-four.gguf"
        });
        let prefix = integer(case, "prefix") as usize;
        let file = Stream::embedded(&std::fs::read(path).unwrap(), prefix, true);
        let model = ModelOwner(unsafe {
            llama_model_load_from_file_ptr(
                file.0,
                LlamaModelParams {
                    load_mode: integer(case, "mode"),
                    vocab_only: u8::from(boolean(case, "vocab_only")),
                    ..params()
                },
            )
        });
        assert_eq!(
            !model.0.is_null(),
            boolean(case, "loaded"),
            "{case:?}: {}",
            error()
        );
        file.check_live_at_origin(prefix);
        drop(file);
        if model.0.is_null() {
            continue;
        }
        assert_eq!(
            unsafe { llama_model_n_embd(model.0) },
            integer(case, "width")
        );
        assert_eq!(
            unsafe { llama_model_meta_count(model.0) },
            integer(case, "metadata")
        );
        assert_eq!(
            unsafe { llama_vocab_n_tokens(llama_model_get_vocab(model.0)) },
            integer(case, "vocabulary")
        );
        if !boolean(case, "vocab_only") {
            let context = context(&model, encoder);
            execute(&context, encoder, case, &reference);
        }
    }
    let path = CString::new(fixture("adapters/inputs/model-four.gguf").to_str().unwrap()).unwrap();
    let model = ModelOwner(unsafe { llama_model_load_from_file(path.as_ptr(), params()) });
    assert!(!model.0.is_null(), "{}", error());
    assert_eq!(
        unsafe { llama_model_load_from_file_ptr(ptr::null_mut(), params()) }.is_null(),
        boolean(&reference, "null_model")
    );
    assert_eq!(
        unsafe { llama_adapter_lora_init_from_file_ptr(model.0, ptr::null_mut()) }.is_null(),
        boolean(&reference, "null_adapter")
    );
    for case in field(&reference, "adapters").as_array().unwrap() {
        let name = field(case, "name").as_str().unwrap();
        let prefix = integer(case, "prefix") as usize;
        let file = Stream::embedded(
            &std::fs::read(fixture(&format!("adapters/inputs/{name}"))).unwrap(),
            prefix,
            true,
        );
        let adapter =
            AdapterOwner(unsafe { llama_adapter_lora_init_from_file_ptr(model.0, file.0) });
        assert_eq!(
            !adapter.0.is_null(),
            boolean(case, "loaded"),
            "{case:?}: {}",
            error()
        );
        file.check_live_at_origin(prefix);
        drop(file);
        if adapter.0.is_null() {
            continue;
        }
        assert_eq!(
            unsafe { llama_adapter_meta_count(adapter.0) },
            integer(case, "metadata")
        );
        let expected = field(case, "invocation").as_array().unwrap();
        let count = unsafe { llama_adapter_get_alora_n_invocation_tokens(adapter.0) };
        assert_eq!(count as usize, expected.len());
        if count > 0 {
            let values = unsafe {
                slice::from_raw_parts(
                    llama_adapter_get_alora_invocation_tokens(adapter.0),
                    count as usize,
                )
            };
            for (a, b) in values.iter().zip(expected) {
                assert_eq!(i64::from(*a), b.as_i64().unwrap());
            }
        }
        let context = context(&model, false);
        let mut pointer = adapter.0;
        let mut scale = 0.7;
        assert_eq!(
            unsafe { llama_set_adapters_lora(context.0, &mut pointer, 1, &mut scale) },
            integer(case, "set_status")
        );
        execute(&context, false, case, &reference);
    }
}

#[test]
fn file_errors_preserve_caller_streams_and_unowned_handles() {
    let data = std::fs::read(fixture("adapters/inputs/model-four.gguf")).unwrap();
    for bytes in [&b"GGUF"[..], &data[..data.len() - 4096]] {
        let file = Stream::embedded(bytes, 7, false);
        assert!(unsafe { llama_model_load_from_file_ptr(file.0, params()) }.is_null());
        file.check_live_at_origin(7);
        assert!(!error().is_empty());
    }
    unsafe extern "C" fn cancel(_: f32, _: *mut c_void) -> bool {
        false
    }
    let file = Stream::embedded(&data, 1, true);
    assert!(
        unsafe {
            llama_model_load_from_file_ptr(
                file.0,
                LlamaModelParams {
                    progress_callback: Some(cancel),
                    ..params()
                },
            )
        }
        .is_null()
    );
    file.check_live_at_origin(1);

    let file = Stream::embedded(&data, 32, true);
    assert!(unsafe { llama_adapter_lora_init_from_file_ptr(ptr::null_mut(), file.0) }.is_null());
    file.check_live_at_origin(32);
}
