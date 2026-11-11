use super::*;
use crate::{batch::llama_batch_get_one, context::*, model::*};
use llama_behavior::json::{self, Value};
use std::path::{Path, PathBuf};

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}
fn integer(value: &Value) -> i32 {
    value.as_i64().unwrap().try_into().unwrap()
}
fn size(value: &Value) -> usize {
    value.as_u64().unwrap().try_into().unwrap()
}
fn text(value: &Value) -> &str {
    value.as_str().unwrap()
}
fn bytes(value: &Value) -> Vec<u8> {
    array(value).iter().map(|v| size(v) as u8).collect()
}
fn profile_name() -> &'static str {
    if cfg!(target_env = "msvc") {
        "windows"
    } else {
        "linux"
    }
}
fn references(name: &str) -> (Value, &'static [u8]) {
    let (json, data): (&str, &[u8]) = match name {
        "windows" => (
            include_str!("../../../fixtures/ffi/adapters/windows-reference.json"),
            include_bytes!("../../../fixtures/ffi/adapters/windows-data.bin"),
        ),
        "linux" => (
            include_str!("../../../fixtures/ffi/adapters/linux-reference.json"),
            include_bytes!("../../../fixtures/ffi/adapters/linux-data.bin"),
        ),
        _ => unreachable!(),
    };
    (json::parse(json).unwrap(), data)
}
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/ffi/adapters/inputs")
        .join(name)
}
fn filename(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}
fn error() -> String {
    unsafe { CStr::from_ptr(llama_rust_last_error()) }
        .to_string_lossy()
        .into_owned()
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
fn load_model(name: &str) -> ModelOwner {
    let path = filename(&fixture(name));
    let pointer = unsafe {
        llama_model_load_from_file(
            path.as_ptr(),
            LlamaModelParams {
                n_gpu_layers: 0,
                load_mode: 0,
                ..LlamaModelParams::default()
            },
        )
    };
    assert!(!pointer.is_null(), "model {name}: {}", error());
    ModelOwner(pointer)
}
fn load_adapter(model: &ModelOwner, name: &str) -> AdapterOwner {
    let pointer = unsafe { llama_adapter_lora_init(model.0, filename(&fixture(name)).as_ptr()) };
    assert!(!pointer.is_null(), "adapter {name}: {}", error());
    AdapterOwner(pointer)
}
fn context(model: &ModelOwner) -> ContextOwner {
    let params = LlamaContextParams {
        n_ctx: 512,
        n_batch: 32,
        n_ubatch: 8,
        n_seq_max: 1,
        n_threads: 1,
        n_threads_batch: 1,
        type_k: 0,
        type_v: 0,
        flash_attn_type: 0,
        kv_unified: 1,
        offload_kqv: 0,
        op_offload: 0,
        no_perf: 1,
        ..LlamaContextParams::default()
    };
    let pointer = unsafe { llama_init_from_model(model.0, params) };
    assert!(!pointer.is_null(), "context: {}", error());
    ContextOwner(pointer)
}
fn region<'a>(data: &'a [u8], value: &Value, unit: usize) -> &'a [u8] {
    let spec = array(value);
    let offset = size(&spec[0]);
    let length = size(&spec[1]) * unit;
    &data[offset..offset + length]
}
fn check_buffer(
    expected: &Value,
    data: &[u8],
    mut call: impl FnMut(*mut c_char, usize) -> i32,
    location: &str,
) {
    let mut output = [0x5a_u8; 160];
    let rc = call(
        output.as_mut_ptr().cast(),
        size(field(expected, "capacity")),
    );
    assert_eq!(rc, integer(field(expected, "rc")), "{location}: return");
    assert_eq!(
        output.as_slice(),
        region(data, field(expected, "buffer"), 1),
        "{location}: bytes"
    );
}

#[test]
fn original_lora_load_metadata_buffers_and_invocation_tokens_are_exact() {
    let (expected, data) = references(profile_name());
    let model = load_model("model-four.gguf");
    let loads = array(field(&expected, "loads"));
    assert_eq!(loads.len(), 44);
    let mut accepted = 0;
    for row in loads {
        let file = text(field(row, "file"));
        let pointer =
            unsafe { llama_adapter_lora_init(model.0, filename(&fixture(file)).as_ptr()) };
        assert_eq!(
            !pointer.is_null(),
            field(row, "loaded").as_bool().unwrap(),
            "{file}: {}",
            error()
        );
        if pointer.is_null() {
            continue;
        }
        let adapter = AdapterOwner(pointer);
        accepted += 1;
        let meta = field(row, "metadata");
        assert_eq!(
            unsafe { llama_adapter_meta_count(adapter.0) },
            integer(field(meta, "count")),
            "{file}"
        );
        for row in array(field(meta, "indices")) {
            let index = integer(field(row, "index"));
            for output in array(field(row, "keys")) {
                check_buffer(
                    output,
                    data,
                    |p, n| unsafe { llama_adapter_meta_key_by_index(adapter.0, index, p, n) },
                    file,
                );
            }
            for output in array(field(row, "values")) {
                check_buffer(
                    output,
                    data,
                    |p, n| unsafe { llama_adapter_meta_val_str_by_index(adapter.0, index, p, n) },
                    file,
                );
            }
        }
        for row in array(field(meta, "lookup")) {
            let key = CString::new(bytes(field(row, "key"))).unwrap();
            for output in array(field(row, "values")) {
                check_buffer(
                    output,
                    data,
                    |p, n| unsafe { llama_adapter_meta_val_str(adapter.0, key.as_ptr(), p, n) },
                    file,
                );
            }
        }
        let expected: Vec<i32> = array(field(meta, "invocations"))
            .iter()
            .map(integer)
            .collect();
        assert_eq!(
            unsafe { llama_adapter_get_alora_n_invocation_tokens(adapter.0) },
            expected.len() as u64,
            "{file}"
        );
        let tokens = unsafe { llama_adapter_get_alora_invocation_tokens(adapter.0) };
        if !expected.is_empty() {
            assert!(!tokens.is_null());
            assert_eq!(
                unsafe { slice::from_raw_parts(tokens, expected.len()) },
                expected,
                "{file}"
            );
        }
    }
    assert_eq!(accepted, 34);
    assert_eq!(
        unsafe { llama_adapter_get_alora_n_invocation_tokens(ptr::null()) },
        size(field(&expected, "null_invocation_count")) as u64
    );
    unsafe {
        llama_adapter_lora_free(ptr::null_mut());
    }
}

#[test]
fn native_pointer_map_order_is_exact_for_both_compilers() {
    for (name, profile) in [("windows", Profile::Msvc), ("linux", Profile::Gnu)] {
        let (expected, _) = references(name);
        for row in array(field(&expected, "pointer_order")) {
            let keys: Vec<usize> = array(field(row, "keys")).iter().map(size).collect();
            let expected: Vec<usize> = array(field(row, "order")).iter().map(size).collect();
            assert_eq!(
                pointer_map_order(&keys, profile).unwrap(),
                expected,
                "{name} {} entries",
                keys.len()
            );
        }
    }
}

#[test]
fn original_lora_control_vectors_and_retained_kv_execution_are_bit_exact() {
    let inputs = json::parse(include_str!("../../../fixtures/ffi/adapters/inputs.json")).unwrap();
    let (expected, data) = references(profile_name());
    let cases = array(field(&inputs, "scenarios"));
    let results = array(field(&expected, "scenarios"));
    assert_eq!(cases.len(), 107);
    assert_eq!(cases.len(), results.len());
    let mut decoded = 0;
    for (case, result) in cases.iter().zip(results) {
        let name = text(field(case, "name"));
        assert_eq!(text(field(result, "name")), name);
        let model = load_model(text(field(case, "model")));
        let ctx = context(&model);
        let mut adapters: Vec<AdapterOwner> = array(field(case, "adapters"))
            .iter()
            .map(|v| load_adapter(&model, text(v)))
            .collect();
        let mut spare = Vec::new();
        if let Some(required) = case.get("require_order") {
            let required: Vec<usize> = array(field(required, profile_name()))
                .iter()
                .map(size)
                .collect();
            for attempt in 0..512 {
                let ids: Vec<usize> = adapters.iter().map(|a| a.0 as usize).collect();
                if pointer_map_order(&ids, profile().unwrap()).unwrap() == required {
                    break;
                }
                assert!(
                    attempt < 511,
                    "could not realize independent native pointer-map order"
                );
                let index = adapters.len() - 1;
                let replacement =
                    load_adapter(&model, text(&array(field(case, "adapters"))[index]));
                spare.push(std::mem::replace(&mut adapters[index], replacement));
            }
        }
        let ops = array(field(case, "ops"));
        let events = array(field(result, "events"));
        assert_eq!(ops.len(), events.len());
        for (step, (op, event)) in ops.iter().zip(events).enumerate() {
            let op = array(op);
            let location = format!("{name} step {step}");
            let rc = match text(&op[0]) {
                "set" => {
                    let mut selected: Vec<_> =
                        array(&op[1]).iter().map(|v| adapters[size(v)].0).collect();
                    let mut scales: Vec<f32> = array(&op[2])
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    unsafe {
                        llama_set_adapters_lora(
                            ctx.0,
                            selected.as_mut_ptr(),
                            selected.len(),
                            scales.as_mut_ptr(),
                        )
                    }
                }
                "clear" => unsafe {
                    llama_set_adapters_lora(ctx.0, ptr::null_mut(), 0, ptr::null_mut())
                },
                "cvec" => {
                    let values: Vec<f32> = array(&op[1])
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                        .collect();
                    unsafe {
                        llama_set_adapter_cvec(
                            ctx.0,
                            values.as_ptr(),
                            values.len(),
                            integer(&op[2]),
                            integer(&op[3]),
                            integer(&op[4]),
                        )
                    }
                }
                "cvec_null" => unsafe {
                    llama_set_adapter_cvec(ctx.0, ptr::null(), 999, -1, -7, -9)
                },
                "decode" | "embed" => {
                    let mut tokens: Vec<i32> = if text(&op[0]) == "decode" {
                        array(&op[1]).iter().map(integer).collect()
                    } else {
                        Vec::new()
                    };
                    let mut embeddings: Vec<f32> = if text(&op[0]) == "embed" {
                        array(&op[1])
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let batch = if text(&op[0]) == "decode" {
                        unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) }
                    } else {
                        LlamaBatch {
                            n_tokens: (embeddings.len() / 4) as i32,
                            embd: embeddings.as_mut_ptr(),
                            ..LlamaBatch::default()
                        }
                    };
                    let rc = unsafe { llama_decode(ctx.0, batch) };
                    unsafe {
                        llama_synchronize(ctx.0);
                    }
                    if rc == 0 {
                        decoded += 1;
                        let pointer = unsafe { llama_get_logits_ith(ctx.0, -1) };
                        assert!(!pointer.is_null(), "{location}");
                        let actual = unsafe { slice::from_raw_parts(pointer, 512) };
                        let expected = region(data, field(event, "logits"), 4);
                        assert_eq!(expected.len(), 2048);
                        for (index, (value, bytes)) in
                            actual.iter().zip(expected.as_chunks::<4>().0).enumerate()
                        {
                            assert_eq!(
                                value.to_bits(),
                                u32::from_le_bytes(*bytes),
                                "{location}: word {index}"
                            );
                        }
                    }
                    rc
                }
                unknown => panic!("unknown fixture operation {unknown}"),
            };
            assert_eq!(rc, integer(field(event, "rc")), "{location}: {}", error());
        }
        drop(ctx);
        drop(adapters);
        drop(spare);
    }
    assert_eq!(decoded, 333);
}

fn decoded_bits(ctx: &ContextOwner, tokens: &mut [i32]) -> Vec<u32> {
    let batch = unsafe { llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
    assert_eq!(unsafe { llama_decode(ctx.0, batch) }, 0, "{}", error());
    let pointer = unsafe { llama_get_logits_ith(ctx.0, -1) };
    assert!(!pointer.is_null());
    unsafe { slice::from_raw_parts(pointer, 512) }
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn failed_adapter_updates_preserve_active_weights_kv_and_outputs() {
    let model = load_model("model-four.gguf");
    let other = load_model("model-four.gguf");
    let good = load_adapter(&model, "multi-0-1.5.gguf");
    let wrong = load_adapter(&other, "multi-0-1.5.gguf");
    let left = context(&model);
    let right = context(&model);
    for ctx in [&left, &right] {
        let mut pointer = good.0;
        let mut scale = 0.5;
        assert_eq!(
            unsafe { llama_set_adapters_lora(ctx.0, &mut pointer, 1, &mut scale) },
            0
        );
    }
    assert_eq!(
        decoded_bits(&left, &mut [1, 2, 3]),
        decoded_bits(&right, &mut [1, 2, 3])
    );
    let before = unsafe { slice::from_raw_parts(llama_get_logits_ith(left.0, -1), 512) }
        .iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();
    let mut pointer = wrong.0;
    let mut scale = 1.0;
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, &mut pointer, 1, &mut scale) },
        -1
    );
    pointer = good.0;
    scale = f32::NAN;
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, &mut pointer, 1, &mut scale) },
        -1
    );
    scale = 1.0;
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, ptr::null_mut(), 1, &mut scale) },
        -1
    );
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, &mut pointer, 1, ptr::null_mut()) },
        -1
    );
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, ptr::null_mut(), 257, ptr::null_mut()) },
        -1
    );
    assert_eq!(
        unsafe { llama_set_adapter_cvec(left.0, [0.0; 4].as_ptr(), 4, 3, 1, 3) },
        -1
    );
    assert_eq!(
        unsafe { llama_set_adapter_cvec(left.0, [f32::INFINITY; 4].as_ptr(), 4, 4, 1, 3) },
        -1
    );
    let after = unsafe { slice::from_raw_parts(llama_get_logits_ith(left.0, -1), 512) }
        .iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(before, after);
    assert_eq!(
        decoded_bits(&left, &mut [4, 5]),
        decoded_bits(&right, &mut [4, 5])
    );
    // A zero scale never dereferences its unused adapter pointer.
    let mut unused = ptr::null_mut();
    let mut zero = 0.0;
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, &mut unused, 1, &mut zero) },
        0
    );
    assert!(add(usize::MAX, 1).is_err());
    assert!(mul(usize::MAX, 2).is_err());
    assert!(require(MAX_ADAPTER_BYTES + 1).is_err());
}

#[test]
fn loaded_adapter_and_context_own_weights_and_model_lifetimes() {
    let model = load_model("model-four.gguf");
    let ctx = context(&model);
    let bytes = std::fs::read(fixture("target-00.gguf")).unwrap();
    let path = std::env::temp_dir().join(format!(
        "rust-llama-owned-adapter-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let pointer = unsafe { llama_adapter_lora_init(model.0, filename(&path).as_ptr()) };
    assert!(!pointer.is_null());
    let weak = unsafe { reference(pointer) }.unwrap().model.clone();
    let payload = Arc::downgrade(&unsafe { reference(pointer) }.unwrap().data);
    std::fs::write(&path, b"replaced after owned load").unwrap();
    std::fs::remove_file(&path).unwrap();
    let mut selected = pointer;
    let mut scale = 0.25;
    assert_eq!(
        unsafe { llama_set_adapters_lora(ctx.0, &mut selected, 1, &mut scale) },
        0
    );
    // Automatic model cleanup must be deferred while a context owns its model.
    let count = unsafe { llama_adapter_meta_count(pointer) };
    assert!(count > 0);
    drop(model);
    assert!(weak.upgrade().is_some());
    assert_eq!(unsafe { llama_adapter_meta_count(pointer) }, count);
    decoded_bits(&ctx, &mut [1, 2, 3]);
    drop(ctx);
    assert!(weak.upgrade().is_none());
    assert!(payload.upgrade().is_none());
    // Explicitly freeing a selected adapter cannot invalidate Rust's owned copy.
    let model = load_model("model-four.gguf");
    let ctx = context(&model);
    let adapter = load_adapter(&model, "target-00.gguf");
    let mut selected = adapter.0;
    assert_eq!(
        unsafe { llama_set_adapters_lora(ctx.0, &mut selected, 1, &mut scale) },
        0
    );
    drop(adapter);
    decoded_bits(&ctx, &mut [1, 2, 3]);
    assert!(
        registry(&unsafe { reference(model.0) }.unwrap().data)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn metadata_and_tied_embedding_checked_boundaries_are_explicit() {
    let model = load_model("model-four.gguf");
    let adapter = load_adapter(&model, "metadata.gguf");
    assert_eq!(
        unsafe {
            llama_adapter_meta_val_str(adapter.0, c"general.type".as_ptr(), ptr::null_mut(), 0)
        },
        7
    );
    assert_eq!(
        unsafe {
            llama_adapter_meta_val_str(adapter.0, c"general.type".as_ptr(), ptr::null_mut(), 1)
        },
        -1
    );
    assert_eq!(
        unsafe { llama_adapter_meta_val_str(adapter.0, ptr::null(), ptr::null_mut(), 0) },
        -1
    );
    assert_eq!(unsafe { llama_adapter_meta_count(ptr::null()) }, -1);
    assert!(unsafe { llama_adapter_lora_init(model.0, ptr::null()) }.is_null());
    let tied = load_model("model-tied.gguf");
    let ctx = context(&tied);
    let embedding = load_adapter(&tied, "embedding-0.gguf");
    let mut selected = embedding.0;
    let mut scale = 1.0;
    assert_eq!(
        unsafe { llama_set_adapters_lora(ctx.0, &mut selected, 1, &mut scale) },
        -1
    );
    assert!(error().contains("tied output"));
}

#[test]
fn reused_handle_slot_refreshes_owned_execution_weights() {
    let model = load_model("model-four.gguf");
    let old = load_adapter(&model, "target-00.gguf");
    let replacement = load_adapter(&model, "target-01.gguf");
    let left = context(&model);
    let right = context(&model);
    let mut selected = old.0;
    let mut scale = 0.75;
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, &mut selected, 1, &mut scale) },
        0
    );
    let retired = Arc::downgrade(&unsafe { reference(old.0) }.unwrap().data);
    // Force a newly loaded data identity into the same valid allocation slot.
    // This models address reuse without dereferencing any freed allocation.
    unsafe { reference_mut(old.0) }.unwrap().data =
        unsafe { reference(replacement.0) }.unwrap().data.clone();
    assert_eq!(
        unsafe { llama_set_adapters_lora(left.0, &mut selected, 1, &mut scale) },
        0
    );
    selected = replacement.0;
    assert_eq!(
        unsafe { llama_set_adapters_lora(right.0, &mut selected, 1, &mut scale) },
        0
    );
    assert!(retired.upgrade().is_none());
    assert_eq!(
        decoded_bits(&left, &mut [1, 2, 3]),
        decoded_bits(&right, &mut [1, 2, 3])
    );
}
