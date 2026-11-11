use super::*;
use crate::{context::*, memory::*, model::*, state::*};
use llama_behavior::json::{self, Value};
use std::mem::{align_of, offset_of, size_of};
use std::path::Path;

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("missing field {key}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
fn integer(value: &Value) -> i32 {
    value.as_i64().unwrap().try_into().unwrap()
}
fn usize_value(value: &Value) -> usize {
    value.as_i64().unwrap().try_into().unwrap()
}
fn number(value: &Value, key: &str, default: i32) -> i32 {
    value.get(key).map(integer).unwrap_or(default)
}
fn flag(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}
fn assert_return(actual: i32, expected: &Value, location: &str) {
    assert_eq!(actual, integer(expected), "{location}");
}
fn assert_flag(actual: bool, expected: &Value, location: &str) {
    assert_eq!(actual, expected.as_bool().unwrap(), "{location}");
}
unsafe fn compare_words(pointer: *const f32, spec: &Value, data: &[u8], location: &str) {
    if matches!(spec, Value::Null) {
        assert!(pointer.is_null(), "{location}: expected unavailable output");
        return;
    }
    assert!(!pointer.is_null(), "{location}: missing output");
    let region = array(spec);
    let start = usize_value(&region[0]);
    let count = usize_value(&region[1]);
    let expected = &data[start..start + count * 4];
    for (index, (actual, raw)) in unsafe { slice::from_raw_parts(pointer, count) }
        .iter()
        .zip(expected.as_chunks::<4>().0)
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            u32::from_le_bytes(*raw),
            "{location}: word {index}"
        );
    }
}
struct ModelOwner(*mut ModelHandle);
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
struct BatchOwner(*mut BatchExtHandle);
impl Drop for BatchOwner {
    fn drop(&mut self) {
        unsafe {
            llama_batch_ext_free(self.0);
        }
    }
}

#[test]
fn original_extended_batch_builders_and_decode_are_exact() {
    #[cfg(target_env = "msvc")]
    let (reference, data): (&str, &[u8]) = (
        include_str!("../../../../fixtures/ffi/batch-ext/windows-reference.json"),
        include_bytes!("../../../../fixtures/ffi/batch-ext/windows-data.bin"),
    );
    #[cfg(not(target_env = "msvc"))]
    let (reference, data): (&str, &[u8]) = (
        include_str!("../../../../fixtures/ffi/batch-ext/linux-reference.json"),
        include_bytes!("../../../../fixtures/ffi/batch-ext/linux-data.bin"),
    );
    let inputs = json::parse(include_str!(
        "../../../../fixtures/ffi/batch-ext/inputs.json"
    ))
    .unwrap();
    let expected = json::parse(reference).unwrap();
    let actual_layout = [
        size_of::<LlamaEmbd>(),
        align_of::<LlamaEmbd>(),
        offset_of!(LlamaEmbd, data),
        offset_of!(LlamaEmbd, n_rows),
        offset_of!(LlamaEmbd, n_embd),
    ];
    assert_eq!(
        actual_layout.as_slice(),
        array(field(&expected, "embedding_layout"))
            .iter()
            .map(usize_value)
            .collect::<Vec<_>>()
    );
    let path = CString::new(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/ffi/native-context-model.gguf"
        ))
        .to_str()
        .unwrap(),
    )
    .unwrap();
    let model = ModelOwner(unsafe {
        llama_model_load_from_file(
            path.as_ptr(),
            LlamaModelParams {
                n_gpu_layers: 0,
                load_mode: 0,
                ..LlamaModelParams::default()
            },
        )
    });
    assert!(!model.0.is_null());
    let cases = array(field(&inputs, "cases"));
    let originals = array(field(&expected, "cases"));
    assert_eq!(cases.len(), 279);
    assert_eq!(cases.len(), originals.len());
    let mut calls = 0;
    for (case, original) in cases.iter().zip(originals) {
        let id = field(case, "id").as_str().unwrap();
        assert_eq!(field(original, "id").as_str().unwrap(), id);
        let params = LlamaContextParams {
            n_ctx: 512,
            n_seq_max: 4,
            n_batch: number(case, "n_batch", 16) as u32,
            n_ubatch: number(case, "n_ubatch", 4) as u32,
            n_threads: 1,
            n_threads_batch: 1,
            type_k: number(case, "type_k", 0),
            type_v: number(case, "type_v", 0),
            flash_attn_type: number(case, "flash", 0),
            kv_unified: u8::from(flag(case, "unified", true)),
            embeddings: u8::from(flag(case, "embeddings", false)),
            offload_kqv: 0,
            op_offload: 0,
            no_perf: 1,
            ..LlamaContextParams::default()
        };
        let ctx = ContextOwner(unsafe { llama_init_from_model(model.0, params) });
        assert!(!ctx.0.is_null(), "{id}: context creation failed");
        let batch = BatchOwner(unsafe { llama_batch_ext_init(ctx.0) });
        assert!(!batch.0.is_null());
        let limits = unsafe {
            [
                llama_n_batch(ctx.0),
                llama_n_ctx(ctx.0),
                llama_n_ctx_seq(ctx.0),
                llama_n_seq_max(ctx.0),
            ]
        };
        assert_eq!(
            limits.iter().map(|v| *v as usize).collect::<Vec<_>>(),
            array(field(original, "limits"))
                .iter()
                .map(usize_value)
                .collect::<Vec<_>>(),
            "{id}"
        );
        let ops = array(field(case, "ops"));
        let outputs = array(field(original, "events"));
        assert_eq!(ops.len(), outputs.len());
        for (step, (op, result)) in ops.iter().zip(outputs).enumerate() {
            let op = array(op);
            let name = op[0].as_str().unwrap();
            let loc = format!("{id}/{step}/{name}");
            unsafe {
                match name {
                    "clear" => llama_batch_ext_clear(batch.0),
                    "clear_memory" => llama_memory_clear(llama_get_memory(ctx.0), 1),
                    "bare" => {
                        assert_return(llama_batch_ext_add(batch.0, integer(&op[1])), result, &loc)
                    }
                    "token" => assert_return(
                        llama_batch_ext_add_token(batch.0, integer(&op[1]), integer(&op[2])),
                        result,
                        &loc,
                    ),
                    "seq" => assert_flag(
                        llama_batch_ext_add_seq(batch.0, integer(&op[1]), integer(&op[2])),
                        result,
                        &loc,
                    ),
                    "pos" => {
                        let p = integer(&op[2]);
                        let pointer = if op.get(3).and_then(Value::as_bool).unwrap_or(false) {
                            ptr::null()
                        } else {
                            &p
                        };
                        assert_flag(
                            llama_batch_ext_set_pos(batch.0, integer(&op[1]), pointer),
                            result,
                            &loc,
                        );
                    }
                    "out" | "out_embd" => {
                        let value = u8::from(op[2].as_bool().unwrap());
                        let actual = if name == "out" {
                            llama_batch_ext_set_output_logits(batch.0, integer(&op[1]), value)
                        } else {
                            llama_batch_ext_set_output_embd(batch.0, integer(&op[1]), value)
                        };
                        assert_flag(actual, result, &loc);
                    }
                    "embd" | "set_embd" | "state" => {
                        let mut values: Vec<_> = array(&op[4])
                            .iter()
                            .map(|v| f32::from_bits(v.as_i64().unwrap() as u32))
                            .collect();
                        let descriptor = LlamaEmbd {
                            data: if op.get(5).and_then(Value::as_bool).unwrap_or(false) {
                                ptr::null()
                            } else {
                                values.as_ptr()
                            },
                            n_rows: usize_value(&op[2]),
                            n_embd: usize_value(&op[3]),
                        };
                        if name == "embd" {
                            assert_return(
                                llama_batch_ext_add_embd(batch.0, integer(&op[1]), descriptor),
                                result,
                                &loc,
                            );
                        } else {
                            let actual = if name == "set_embd" {
                                llama_batch_ext_set_embd_token(batch.0, integer(&op[1]), descriptor)
                            } else {
                                llama_batch_ext_set_embd_state(batch.0, integer(&op[1]), descriptor)
                            };
                            assert_flag(actual, result, &loc);
                        }
                        values.fill(123.0);
                    }
                    "process" => {
                        calls += 1;
                        let actual = llama_process(ctx.0, integer(&op[1]), batch.0);
                        assert_eq!(actual, integer(field(result, "code")), "{loc}");
                        if actual == 0 {
                            let query = array(&op[2]);
                            for (index, spec) in query.iter().zip(array(field(result, "logits"))) {
                                compare_words(
                                    llama_get_logits_ith(ctx.0, integer(index)),
                                    spec,
                                    data,
                                    &loc,
                                );
                            }
                            for (index, spec) in
                                query.iter().zip(array(field(result, "embeddings")))
                            {
                                compare_words(
                                    llama_get_embeddings_ith(ctx.0, integer(index)),
                                    spec,
                                    data,
                                    &loc,
                                );
                            }
                            let mem = llama_get_memory(ctx.0);
                            for (seq, pos) in array(field(result, "positions")).iter().enumerate() {
                                let pos = array(pos);
                                assert_eq!(
                                    [
                                        llama_memory_seq_pos_min(mem, seq as i32),
                                        llama_memory_seq_pos_max(mem, seq as i32)
                                    ],
                                    [integer(&pos[0]), integer(&pos[1])],
                                    "{loc}"
                                );
                            }
                            let mut state = vec![0u8; llama_state_get_size(ctx.0)];
                            let written =
                                llama_state_get_data(ctx.0, state.as_mut_ptr(), state.len());
                            state.truncate(written);
                            let spec = array(field(result, "state"));
                            let start = usize_value(&spec[0]);
                            let count = usize_value(&spec[1]);
                            assert_eq!(state, data[start..start + count], "{loc}: native state");
                        }
                    }
                    _ => panic!("unknown fixture op {name}"),
                }
            }
        }
    }
    assert_eq!(calls, 395);
}

#[test]
fn checked_descriptor_overflow_and_invalid_bool_preserve_owned_rows() {
    let mut batch = BatchExtHandle {
        maximum_tokens: 4,
        maximum_sequences: 2,
        vocabulary: 512,
        width: 4,
        embedding_width: 0,
        entries: Vec::new(),
        embeddings: Vec::new(),
    };
    let handle = &mut batch as *mut BatchExtHandle;
    unsafe {
        assert_eq!(llama_batch_ext_add(handle, 0), 0);
        let row = [1.0, 2.0, 3.0, 4.0];
        assert!(!llama_batch_ext_set_embd_token(
            handle,
            0,
            LlamaEmbd {
                data: row.as_ptr(),
                n_rows: usize::MAX,
                n_embd: 2,
            }
        ));
        assert!(batch.embeddings.is_empty());
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .windows(8)
                .any(|v| v == b"overflow")
        );
        assert!(llama_batch_ext_set_embd_token(
            handle,
            0,
            LlamaEmbd {
                data: row.as_ptr(),
                n_rows: 2,
                n_embd: 2,
            }
        ));
        assert!(!llama_batch_ext_set_output_logits(handle, 0, 2));
        assert!(!batch.entries[0].output);
        assert_eq!(batch.embeddings, row);
        llama_batch_ext_clear(handle);
        assert!(batch.entries.is_empty());
        assert!(batch.embeddings.is_empty());
        assert_eq!(batch.embedding_width, 0);
        assert!(llama_batch_ext_init(ptr::null_mut()).is_null());
        assert_eq!(llama_batch_ext_add(ptr::null_mut(), 0), -1);
        assert_eq!(llama_process(ptr::null_mut(), 1, handle), -1);
        llama_batch_ext_free(ptr::null_mut());
    }
}
