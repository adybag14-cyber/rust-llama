use super::*;
use crate::context::encoder::{llama_encode, llama_get_embeddings_seq};
use crate::context::*;
use crate::model::{llama_model_free, llama_model_load_from_file};
use crate::state::*;
use llama_behavior::json::{self, Value};
use std::mem::{align_of, offset_of, size_of};

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn int(value: &Value, key: &str) -> i32 {
    field(value, key).as_f64().unwrap() as i32
}
fn number(value: &Value, key: &str) -> f64 {
    field(value, key).as_f64().unwrap()
}
fn boolean_field(value: &Value, key: &str) -> bool {
    field(value, key).as_bool().unwrap()
}

struct Trace<'a> {
    context: *mut ContextHandle,
    expected: &'a [Value],
    cursor: usize,
    no_perf: bool,
}
impl Trace<'_> {
    fn record(&mut self, name: &str, status: i32) {
        let expected = &self.expected[self.cursor];
        assert_eq!(field(expected, "name").as_str().unwrap(), name);
        assert_eq!(int(expected, "status"), status, "{name}");
        let p = unsafe { llama_perf_context(self.context) };
        assert_eq!(
            (p.n_p_eval, p.n_eval),
            (int(expected, "n_p_eval"), int(expected, "n_eval")),
            "{name}"
        );
        assert!(p.t_start_ms >= 0.0 && p.t_load_ms >= 0.0);
        assert!(p.t_p_eval_ms >= 0.0 && p.t_eval_ms >= 0.0);
        assert_eq!(
            (p.n_reused, p.n_reallocs_same, p.n_reallocs_diff),
            (0, 0, 0),
            "direct execution has no ggml graph allocations/reuses"
        );
        if self.no_perf {
            assert_eq!((p.t_p_eval_ms, p.t_eval_ms), (0.0, 0.0), "{name}");
        }
        self.cursor += 1;
    }
    fn eval(&mut self, encoder: bool, position: &mut i32, n: i32, name: &str) {
        let mut tokens = vec![3; n as usize];
        let mut positions: Vec<i32> = (0..n)
            .map(|i| if encoder { i } else { *position + i })
            .collect();
        let batch = LlamaBatch {
            n_tokens: n,
            token: tokens.as_mut_ptr(),
            embd: ptr::null_mut(),
            pos: positions.as_mut_ptr(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        };
        let status = unsafe {
            if encoder {
                llama_encode(self.context, batch)
            } else {
                llama_decode(self.context, batch)
            }
        };
        assert_eq!(
            status,
            0,
            "{}",
            unsafe { CStr::from_ptr(llama_rust_last_error()) }.to_string_lossy()
        );
        *position += n;
        self.record(name, status);
    }
}

#[test]
fn original_context_performance_boundaries() {
    let _logger_guard = crate::logging::TEST_LOGGER_LOCK.lock().unwrap();
    let reference = json::parse(include_str!(
        "../../../../fixtures/ffi/context-perf/reference.json"
    ))
    .unwrap();
    let abi = field(&reference, "abi");
    assert_eq!(size_of::<LlamaPerfContextData>(), int(abi, "size") as usize);
    assert_eq!(
        align_of::<LlamaPerfContextData>(),
        int(abi, "align") as usize
    );
    let offsets = [
        offset_of!(LlamaPerfContextData, t_start_ms),
        offset_of!(LlamaPerfContextData, t_load_ms),
        offset_of!(LlamaPerfContextData, t_p_eval_ms),
        offset_of!(LlamaPerfContextData, t_eval_ms),
        offset_of!(LlamaPerfContextData, n_p_eval),
        offset_of!(LlamaPerfContextData, n_eval),
        offset_of!(LlamaPerfContextData, n_reused),
        offset_of!(LlamaPerfContextData, n_reallocs_same),
        offset_of!(LlamaPerfContextData, n_reallocs_diff),
    ];
    for (actual, expected) in offsets
        .iter()
        .zip(field(abi, "offsets").as_array().unwrap())
    {
        assert_eq!(*actual, expected.as_f64().unwrap() as usize);
    }
    unsafe {
        llama_rust_clear_error();
        assert_eq!(
            llama_perf_context(ptr::null()),
            LlamaPerfContextData::default()
        );
        assert_eq!(CStr::from_ptr(llama_rust_last_error()).to_bytes(), b"");
        for scenario in field(&reference, "scenarios").as_array().unwrap() {
            let encoder = boolean_field(scenario, "encoder");
            let no_perf = boolean_field(scenario, "no_perf");
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(if encoder {
                "../../fixtures/ffi/encoder/plain.gguf"
            } else {
                "../../fixtures/ffi/native-context-model.gguf"
            });
            let path = CString::new(path.to_str().unwrap()).unwrap();
            let model = llama_model_load_from_file(
                path.as_ptr(),
                LlamaModelParams {
                    n_gpu_layers: 0,
                    load_mode: 0,
                    ..LlamaModelParams::default()
                },
            );
            assert!(
                !model.is_null(),
                "{}",
                CStr::from_ptr(llama_rust_last_error()).to_string_lossy()
            );
            let context = llama_init_from_model(
                model,
                LlamaContextParams {
                    n_ctx: 512,
                    n_batch: 64,
                    n_ubatch: 64,
                    n_threads: 1,
                    n_threads_batch: 1,
                    type_k: 0,
                    type_v: 0,
                    flash_attn_type: 0,
                    no_perf: no_perf as u8,
                    embeddings: encoder as u8,
                    pooling_type: int(scenario, "pooling"),
                    ..LlamaContextParams::default()
                },
            );
            assert!(
                !context.is_null(),
                "{}",
                CStr::from_ptr(llama_rust_last_error()).to_string_lossy()
            );
            let mut trace = Trace {
                context,
                expected: field(scenario, "events").as_array().unwrap(),
                cursor: 0,
                no_perf,
            };
            let mut position = 0;
            trace.record("initial", 0);
            trace.eval(encoder, &mut position, 3, "prompt_queued");
            llama_get_logits(context);
            trace.record("prompt_logits", 0);
            let first_load = llama_perf_context(context).t_load_ms;
            trace.eval(encoder, &mut position, 1, "single_a_queued");
            trace.eval(encoder, &mut position, 1, "single_b_queued");
            llama_synchronize(context);
            trace.record("two_singles_sync", 0);
            llama_perf_context_reset(context);
            trace.record("reset", 0);
            assert_eq!(llama_perf_context(context).t_load_ms, first_load);
            llama_set_warmup(context, 1);
            trace.record("warmup_on", 0);
            trace.eval(encoder, &mut position, 1, "warmup_queued");
            if encoder {
                llama_get_embeddings(context);
            } else {
                llama_get_logits_ith(context, -1);
            }
            trace.record("warmup_output", 0);
            llama_set_warmup(context, 0);
            trace.record("warmup_off", 0);
            trace.eval(encoder, &mut position, 1, "single_c_queued");
            llama_get_embeddings(context);
            trace.record("embeddings_sync", 0);
            trace.eval(encoder, &mut position, 1, "pending_before_reset");
            llama_perf_context_reset(context);
            trace.record("pending_reset", 0);
            llama_synchronize(context);
            trace.record("pending_sync", 0);
            trace.eval(encoder, &mut position, 2, "final_prompt_queued");
            if encoder {
                llama_get_embeddings_seq(context, 0);
                trace.record("sequence_embeddings_sync", 0);
            } else {
                let n = llama_state_get_size(context);
                trace.record("state_size_no_sync", 0);
                let mut state = vec![0u8; n];
                let wrote = llama_state_get_data(context, state.as_mut_ptr(), state.len());
                trace.record("state_data_sync", if wrote == n { 0 } else { -9 });
                let restored = llama_state_set_data(context, state.as_ptr(), state.len());
                trace.record("state_restore", if restored == n { 0 } else { -9 });
                trace.eval(encoder, &mut position, 1, "state_sequence_queued");
                let n = llama_state_seq_get_size(context, 0);
                trace.record("state_sequence_size", 0);
                state.resize(n, 0);
                let wrote = llama_state_seq_get_data(context, state.as_mut_ptr(), state.len(), 0);
                trace.record("state_sequence_data", if wrote == n { 0 } else { -9 });
            }
            let empty = LlamaBatch {
                n_tokens: 0,
                token: ptr::null_mut(),
                embd: ptr::null_mut(),
                pos: ptr::null_mut(),
                n_seq_id: ptr::null_mut(),
                seq_id: ptr::null_mut(),
                logits: ptr::null_mut(),
            };
            let status = if encoder {
                llama_encode(context, empty)
            } else {
                llama_decode(context, empty)
            };
            trace.record("empty_rejected", status);
            assert_eq!(trace.cursor, trace.expected.len());
            let performance = llama_perf_context(context);
            if !no_perf {
                assert!(performance.t_p_eval_ms > 0.0 && performance.t_eval_ms > 0.0);
            }
            assert_eq!(
                performance.t_load_ms, first_load,
                "later evaluations/reset retain first load time"
            );
            if !encoder {
                let expected = field(scenario, "logits").as_array().unwrap();
                let values = llama_get_logits_ith(context, -1);
                assert!(!values.is_null());
                for (actual, expected) in slice::from_raw_parts(values, expected.len())
                    .iter()
                    .zip(expected)
                {
                    assert_eq!(actual.to_bits(), expected.as_f64().unwrap() as u32);
                }
            }
            llama_free(context);
            llama_model_free(model);
        }
    }
}

#[test]
#[cfg(any(target_os = "linux", all(target_os = "windows", target_env = "msvc")))]
fn original_context_performance_printing() {
    let text = if cfg!(target_os = "windows") {
        include_str!("../../../../fixtures/ffi/context-perf/msvc-printing.json")
    } else {
        include_str!("../../../../fixtures/ffi/context-perf/gnu-printing.json")
    };
    let cases = json::parse(text).unwrap();
    for case in cases.as_array().unwrap() {
        let p = field(case, "perf");
        let data = LlamaPerfContextData {
            t_start_ms: number(p, "t_start_ms"),
            t_load_ms: number(p, "t_load_ms"),
            t_p_eval_ms: number(p, "t_p_eval_ms"),
            t_eval_ms: number(p, "t_eval_ms"),
            n_p_eval: int(p, "n_p_eval"),
            n_eval: int(p, "n_eval"),
            n_reused: int(p, "n_reused"),
            n_reallocs_same: int(p, "n_reallocs_same"),
            n_reallocs_diff: int(p, "n_reallocs_diff"),
        };
        let actual = messages(data, data.t_start_ms).unwrap();
        let expected = field(case, "messages").as_array().unwrap();
        for i in [0, 1, 2, 4, 5] {
            assert_eq!(
                actual[i].to_bytes(),
                field(&expected[i], "message").as_str().unwrap().as_bytes(),
                "line {i}"
            );
        }
        // End time is an execution measurement; field widths/count grammar are fixed.
        assert!(
            actual[3]
                .to_bytes()
                .starts_with(b"llama_perf_context_print:       total time = ")
        );
    }
}

#[test]
fn performance_overflow_and_invalid_handles_are_checked() {
    let mut perf = Performance::new(monotonic_time_us(), 0, true);
    perf.prompt_count = i32::MAX;
    assert!(perf.queue(2).is_err());
    assert_eq!(perf.queued, 0);
    perf.prompt_count = 0;
    perf.eval_count = i32::MAX;
    assert!(perf.queue(1).is_err());
    assert_eq!(perf.queued, 0);
    unsafe {
        llama_rust_clear_error();
        llama_perf_context_reset(ptr::null_mut());
        assert!(
            !CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .is_empty()
        );
        llama_rust_clear_error();
        llama_set_warmup(ptr::null_mut(), 2);
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_string_lossy()
                .contains("C bool")
        );
        let first = llama_print_system_info();
        let second = llama_print_system_info();
        assert_eq!(first, second);
        assert_eq!(
            CStr::from_ptr(first).to_bytes(),
            if cfg!(all(
                target_arch = "x86_64",
                any(
                    all(target_os = "windows", target_env = "msvc"),
                    all(target_os = "linux", target_env = "gnu")
                )
            )) {
                b"CPU : RUST = 1 | SCALAR = 1 | FLASH_ATTN = 1 | THREADS = 1 | "
            } else {
                b"CPU : RUST = 1 | SCALAR = 1 | FLASH_ATTN = 0 | THREADS = 1 | "
            }
        );
    }
}
