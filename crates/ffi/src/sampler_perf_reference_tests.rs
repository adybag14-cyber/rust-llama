//! Replays pre-implementation MSVC/GNU public-header observations.
use crate::logging::{llama_log_get, llama_log_set};
use crate::sampler::*;
use crate::*;
use llama_behavior::json::{self, Value};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::time::Duration;

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}
macro_rules! json {
    ({$($key:literal:$value:expr),* $(,)?}) => {
        Value::object([$(($key,Value::from($value))),*])
    };
    ([$($value:expr),* $(,)?]) => {
        Value::Array(vec![$(Value::from($value)),*])
    };
    ($value:expr) => { Value::from($value) };
}

#[derive(Clone)]
struct Custom {
    apply_ms: u64,
    accept_ms: u64,
    applies: i32,
    accepts: i32,
    resets: i32,
}
unsafe fn custom<'a>(sampler: *const LlamaSampler) -> &'a Custom {
    unsafe { &*(*sampler).ctx.cast::<Custom>() }
}
unsafe fn custom_mut<'a>(sampler: *mut LlamaSampler) -> &'a mut Custom {
    unsafe { &mut *(*sampler).ctx.cast::<Custom>() }
}
unsafe extern "C" fn custom_name(_: *const LlamaSampler) -> *const c_char {
    c"perf-oracle".as_ptr()
}
unsafe extern "C" fn custom_accept(sampler: *mut LlamaSampler, _: i32) {
    let value = unsafe { custom_mut(sampler) };
    value.accepts += 1;
    std::thread::sleep(Duration::from_millis(value.accept_ms));
}
unsafe extern "C" fn custom_apply(sampler: *mut LlamaSampler, _: *mut LlamaTokenDataArray) {
    let value = unsafe { custom_mut(sampler) };
    value.applies += 1;
    std::thread::sleep(Duration::from_millis(value.apply_ms));
}
unsafe extern "C" fn custom_reset(sampler: *mut LlamaSampler) {
    unsafe { custom_mut(sampler) }.resets += 1;
}
unsafe extern "C" fn custom_clone(sampler: *const LlamaSampler) -> *mut LlamaSampler {
    let value = unsafe { custom(sampler) }.clone();
    unsafe { llama_sampler_init((*sampler).iface, Box::into_raw(Box::new(value)).cast()) }
}
unsafe extern "C" fn custom_free(sampler: *mut LlamaSampler) {
    unsafe { drop(Box::from_raw((*sampler).ctx.cast::<Custom>())) }
}
unsafe extern "C" fn custom_copy(source: *const LlamaSampler, target: *mut LlamaSampler) {
    let value = unsafe { custom(source) }.clone();
    *unsafe { custom_mut(target) } = value;
}
const CUSTOM_VALUE: LlamaSamplerI = LlamaSamplerI {
    name: Some(custom_name),
    accept: Some(custom_accept),
    apply: Some(custom_apply),
    reset: Some(custom_reset),
    clone: Some(custom_clone),
    free: Some(custom_free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: Some(custom_copy),
};
static CUSTOM_INTERFACE: LlamaSamplerI = CUSTOM_VALUE;
static BACKEND_INTERFACE: LlamaSamplerI = LlamaSamplerI {
    apply: None,
    ..CUSTOM_VALUE
};

struct LogState {
    calls: Vec<Value>,
    tag: i32,
    callback_roundtrip: bool,
}
impl LogState {
    fn new(tag: i32) -> Self {
        Self {
            calls: Vec::new(),
            tag,
            callback_roundtrip: true,
        }
    }
}
fn same_callback(a: LogCallback, b: LogCallback) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => std::ptr::fn_addr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}
unsafe extern "C" fn logger(level: i32, text: *const c_char, data: *mut c_void) {
    let state = unsafe { &mut *data.cast::<LogState>() };
    let mut current = None;
    let mut current_data = ptr::null_mut();
    unsafe { llama_log_get(&mut current, &mut current_data) };
    state.callback_roundtrip &= same_callback(current, Some(logger)) && current_data == data;
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes_with_nul();
    state.calls.push(json!({
        "level":level, "tag":state.tag,
        "bytes":String::from_utf8(bytes.to_vec()).unwrap()
    }));
}
struct RestoreLogger(LogCallback, *mut c_void);
impl Drop for RestoreLogger {
    fn drop(&mut self) {
        unsafe { llama_log_set(self.0, self.1) };
    }
}

#[test]
fn original_sampler_performance_and_logger_observations() {
    let _logger_guard = crate::logging::TEST_LOGGER_LOCK.lock().unwrap();
    let inputs: Value = json::parse(include_str!(
        "../../../fixtures/ffi/sampler-perf/inputs.json"
    ))
    .unwrap();
    let expected: Value = json::parse(include_str!(
        "../../../fixtures/ffi/sampler-perf/reference.json"
    ))
    .unwrap();
    unsafe {
        llama_rust_clear_error();
        let mut initial = None;
        let mut initial_data = ptr::null_mut();
        llama_log_get(&mut initial, &mut initial_data);
        let _restore = RestoreLogger(initial, initial_data);
        let mut result = BTreeMap::new();
        let layout = [
            size_of::<LlamaPerfSamplerData>(),
            align_of::<LlamaPerfSamplerData>(),
            offset_of!(LlamaPerfSamplerData, t_sample_ms),
            offset_of!(LlamaPerfSamplerData, n_sample),
        ];
        result.insert(
            "abi".to_string(),
            Value::Array(layout.map(|v| Value::from(v as u64)).into()),
        );
        result.insert(
            "logger_initial".to_string(),
            json!({"callback_nonnull":initial.is_some(),"userdata_null":initial_data.is_null()}),
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/ffi/native-context-model.gguf");
        let path = CString::new(path.to_str().unwrap()).unwrap();
        let mp = LlamaModelParams {
            n_gpu_layers: 0,
            load_mode: 0,
            ..LlamaModelParams::default()
        };
        let model = model::llama_model_load_from_file(path.as_ptr(), mp);
        assert!(!model.is_null());
        let cp = LlamaContextParams {
            n_ctx: 32,
            n_batch: 32,
            n_ubatch: 32,
            n_threads: 1,
            n_threads_batch: 1,
            type_k: 0,
            type_v: 0,
            flash_attn_type: 0,
            no_perf: 1,
            offload_kqv: 0,
            op_offload: 0,
            ..LlamaContextParams::default()
        };
        let context = context::llama_init_from_model(model, cp);
        assert!(!context.is_null());
        let mut prompt = [1, 2, 3];
        let batch = batch::llama_batch_get_one(prompt.as_mut_ptr(), 3);
        assert_eq!(context::llama_decode(context, batch), 0);
        let mut cases = Vec::new();
        for case in field(&inputs, "cases").as_array().unwrap() {
            let mut handles: BTreeMap<String, *mut LlamaSampler> = BTreeMap::new();
            let mut previous = BTreeMap::new();
            let mut trace = Vec::new();
            let mut logs = LogState::new(7);
            llama_log_set(Some(logger), (&raw mut logs).cast());
            for op in field(case, "ops").as_array().unwrap() {
                let name = field(op, "op").as_str().unwrap();
                let id = field(op, "id").as_str();
                let sampler = id.map_or(ptr::null_mut(), |id| handles[id]);
                match name {
                    "chain" => {
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_chain_init(LlamaSamplerChainParams {
                                no_perf: u8::from(field(op, "no_perf").as_bool().unwrap()),
                            }),
                        );
                    }
                    "greedy" => {
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_init_greedy(),
                        );
                    }
                    "temp" => {
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_init_temp(field(op, "value").as_f64().unwrap() as f32),
                        );
                    }
                    "top_k" => {
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_init_top_k(field(op, "value").as_i64().unwrap() as i32),
                        );
                    }
                    "custom" => {
                        let interface = if field(op, "backend_only").as_bool().unwrap_or(false) {
                            &raw const BACKEND_INTERFACE
                        } else {
                            &raw const CUSTOM_INTERFACE
                        };
                        let data = Box::new(Custom {
                            apply_ms: field(op, "apply_ms").as_u64().unwrap_or(0),
                            accept_ms: field(op, "accept_ms").as_u64().unwrap_or(0),
                            applies: 0,
                            accepts: 0,
                            resets: 0,
                        });
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_init(interface.cast_mut(), Box::into_raw(data).cast()),
                        );
                    }
                    "add" => llama_sampler_chain_add(
                        sampler,
                        handles[field(op, "child").as_str().unwrap()],
                    ),
                    "get" => {
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_chain_get(
                                sampler,
                                field(op, "index").as_i64().unwrap() as i32,
                            ),
                        );
                    }
                    "remove" => {
                        let child = llama_sampler_chain_remove(
                            sampler,
                            field(op, "index").as_i64().unwrap() as i32,
                        );
                        trace.push(json!({"op":name,"same_child":child==handles[field(op,"child").as_str().unwrap()]}));
                    }
                    "free" => llama_sampler_free(sampler),
                    "accept" => llama_sampler_accept(
                        sampler,
                        field(op, "token").as_i64().unwrap_or(3) as i32,
                    ),
                    "apply" => {
                        let mut data = [
                            LlamaTokenData {
                                id: 3,
                                logit: 0.5,
                                p: 0.0,
                            },
                            LlamaTokenData {
                                id: 1,
                                logit: 3.0,
                                p: 0.0,
                            },
                            LlamaTokenData {
                                id: 7,
                                logit: 1.0,
                                p: 0.0,
                            },
                            LlamaTokenData {
                                id: 5,
                                logit: -2.0,
                                p: 0.0,
                            },
                        ];
                        let mut array = LlamaTokenDataArray {
                            data: data.as_mut_ptr(),
                            size: data.len(),
                            selected: -1,
                            sorted: 0,
                        };
                        llama_sampler_apply(sampler, &mut array);
                        let values: Vec<_> = slice::from_raw_parts(array.data, array.size)
                            .iter()
                            .map(|v| json!([v.id, v.logit.to_bits(), v.p.to_bits()]))
                            .collect();
                        trace.push(json!({"op":name,"selected":array.selected,"sorted":array.sorted!=0,"values":Value::Array(values)}));
                    }
                    "sample" => trace
                        .push(json!({"op":name,"token":llama_sampler_sample(sampler,context,-1)})),
                    "reset" => llama_sampler_reset(sampler),
                    "perf_reset" => llama_perf_sampler_reset(sampler),
                    "clone" => {
                        handles.insert(
                            field(op, "out").as_str().unwrap().into(),
                            llama_sampler_clone(sampler),
                        );
                    }
                    "copy" => {
                        llama_sampler_copy(sampler, handles[field(op, "dst").as_str().unwrap()])
                    }
                    "inspect_custom" => {
                        let value = custom(sampler);
                        trace.push(json!({"op":name,"applies":value.applies,"accepts":value.accepts,"resets":value.resets}));
                    }
                    "perf" => {
                        let value = llama_perf_sampler(sampler);
                        let label = field(op, "label").as_str().unwrap();
                        let clock_ok = match field(op, "clock").as_str().unwrap() {
                            "zero" => value.t_sample_ms == 0.0,
                            "positive" => value.t_sample_ms > 0.0,
                            "any" => true,
                            _ => panic!("unknown clock"),
                        };
                        let relation_ok = field(op, "previous").as_str().is_none_or(|name| {
                            let old = previous[name];
                            match field(op, "relation").as_str().unwrap() {
                                "equal" => value.t_sample_ms == old,
                                "greater" => value.t_sample_ms > old,
                                _ => value.t_sample_ms >= old,
                            }
                        });
                        let valid_microseconds = value.t_sample_ms.is_finite()
                            && value.t_sample_ms >= 0.0
                            && (value.t_sample_ms * 1000.0 - (value.t_sample_ms * 1000.0).round())
                                .abs()
                                < 0.0001;
                        trace.push(json!({"op":name,"label":label,"n_sample":value.n_sample,
                            "clock_ok":clock_ok,"relation_ok":relation_ok,"valid_microseconds":valid_microseconds}));
                        previous.insert(label.to_string(), value.t_sample_ms);
                    }
                    "print" => {
                        assert_eq!(llama_perf_sampler(sampler).t_sample_ms, 0.0);
                        llama_perf_sampler_print(sampler);
                    }
                    _ => panic!("unknown operation {name}"),
                }
                assert_eq!(
                    CStr::from_ptr(llama_rust_last_error()).to_bytes(),
                    b"",
                    "{:?} / {op:?}",
                    field(case, "name").clone()
                );
            }
            cases.push(json!({"name":field(case,"name").clone(),"trace":Value::Array(trace),"logs":Value::Array(logs.calls),
                "callback_roundtrip":logs.callback_roundtrip}));
            llama_log_set(initial, initial_data);
        }
        result.insert("cases".to_string(), Value::Array(cases));
        let mut a = LogState::new(11);
        let mut b = LogState::new(22);
        let mut got = None;
        let mut data = ptr::null_mut();
        llama_log_set(Some(logger), (&raw mut a).cast());
        llama_log_get(&mut got, &mut data);
        result.insert(
            "logger_set".to_string(),
            json!(same_callback(got, Some(logger)) && data == (&raw mut a).cast()),
        );
        llama_log_set(Some(logger), (&raw mut b).cast());
        llama_log_get(&mut got, &mut data);
        result.insert(
            "logger_replace".to_string(),
            json!(same_callback(got, Some(logger)) && data == (&raw mut b).cast()),
        );
        let empty = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_perf_sampler_print(empty);
        result.insert(
            "logger_replacement_calls".to_string(),
            Value::Array(std::mem::take(&mut b.calls)),
        );
        result.insert(
            "logger_old_calls".to_string(),
            Value::Array(std::mem::take(&mut a.calls)),
        );
        llama_log_set(None, (&raw mut a).cast());
        llama_log_get(&mut got, &mut data);
        result.insert(
            "logger_default".to_string(),
            json!({"same_initial_callback":same_callback(got,initial),
            "retained_userdata":data==(&raw mut a).cast()}),
        );
        // The external original-header client also checks both platforms' exact
        // default stderr bytes, including MSVC text-mode newline translation.
        llama_sampler_free(empty);
        context::llama_free(context);
        model::llama_model_free(model);
        llama_log_set(initial, initial_data);
        llama_log_get(&mut got, &mut data);
        result.insert(
            "logger_restore".to_string(),
            json!(same_callback(got, initial) && data == initial_data),
        );
        assert_eq!(Value::Object(result), expected);
    }
}

#[test]
fn performance_misuse_and_invalid_logger_outputs_are_checked() {
    unsafe {
        let greedy = llama_sampler_init_greedy();
        for sampler in [ptr::null_mut(), greedy] {
            for operation in 0..3 {
                llama_rust_clear_error();
                match operation {
                    0 => {
                        let data = llama_perf_sampler(sampler);
                        assert_eq!(data.t_sample_ms, 0.0);
                        assert_eq!(data.n_sample, 0);
                    }
                    1 => llama_perf_sampler_reset(sampler),
                    _ => llama_perf_sampler_print(sampler),
                }
                assert!(
                    !CStr::from_ptr(llama_rust_last_error())
                        .to_bytes()
                        .is_empty()
                );
            }
        }
        llama_sampler_free(greedy);
        llama_rust_clear_error();
        assert!(llama_sampler_chain_init(LlamaSamplerChainParams { no_perf: 2 }).is_null());
        assert!(
            !CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .is_empty()
        );
        let mut callback = None;
        llama_rust_clear_error();
        llama_log_get(&mut callback, ptr::null_mut());
        assert!(callback.is_none());
        assert!(
            !CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .is_empty()
        );
        let mut user_data = ptr::dangling_mut::<u8>().cast::<c_void>();
        llama_rust_clear_error();
        llama_log_get(ptr::null_mut(), &mut user_data);
        assert_eq!(user_data, ptr::dangling_mut::<u8>().cast::<c_void>());
        assert!(
            !CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .is_empty()
        );
    }
}
