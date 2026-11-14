use super::*;
use crate::{batch::*, model::*, sampler::*, *};
use llama_behavior::json::{self, Value};
use std::ffi::{CStr, CString};

fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn number(v: &Value, key: &str) -> i64 {
    field(v, key).as_i64().unwrap()
}
fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn flag(v: &Value, key: &str) -> bool {
    field(v, key).as_bool().unwrap()
}

unsafe fn model() -> *mut ModelHandle {
    let path = CString::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/ffi/native-context-model.gguf"
    ))
    .unwrap();
    let params = LlamaModelParams {
        n_gpu_layers: 0,
        load_mode: 0,
        ..LlamaModelParams::default()
    };
    let model = unsafe { llama_model_load_from_file(path.as_ptr(), params) };
    assert!(!model.is_null());
    model
}
fn parameters() -> LlamaContextParams {
    LlamaContextParams {
        n_ctx: 512,
        n_batch: 32,
        n_ubatch: 32,
        n_seq_max: 2,
        n_threads: 1,
        n_threads_batch: 1,
        type_k: 0,
        type_v: 0,
        flash_attn_type: 0,
        ..LlamaContextParams::default()
    }
}
unsafe fn chain(kind: i64) -> *mut LlamaSampler {
    if kind == 0 {
        return ptr::null_mut();
    }
    let sampler = unsafe { llama_sampler_chain_init(llama_sampler_chain_default_params()) };
    if kind == 2 {
        return sampler;
    }
    let mut stages = Vec::new();
    if [4, 8, 9, 11].contains(&kind) {
        stages.push(unsafe { llama_sampler_init_top_k(5) });
    }
    if kind == 7 {
        stages.push(unsafe { llama_sampler_init_top_k(0) });
    }
    if [5, 10].contains(&kind) {
        stages.push(unsafe { llama_sampler_init_temp(0.7) });
    }
    if [6, 8].contains(&kind) {
        stages.push(unsafe { llama_sampler_init_temp(0.0) });
    }
    if kind == 12 {
        stages.push(unsafe { llama_sampler_init_temp(-1.0) });
    }
    if [9, 10, 13].contains(&kind) {
        stages.push(unsafe { llama_sampler_init_typical(0.8, 1) });
    }
    if kind == 11 {
        stages.push(unsafe { llama_sampler_init_temp(0.7) });
    }
    stages.push(if kind == 10 {
        unsafe { llama_sampler_init_dist(42) }
    } else {
        unsafe { llama_sampler_init_greedy() }
    });
    for stage in stages {
        assert!(!stage.is_null());
        unsafe {
            llama_sampler_chain_add(sampler, stage);
        }
    }
    sampler
}

unsafe fn check_rows(c: *mut ContextHandle, event: &Value) {
    for expected in array(field(event, "rows")) {
        let i = number(expected, "i") as i32;
        let l = unsafe { llama_get_sampled_logits_ith(c, i) };
        let p = unsafe { llama_get_sampled_probs_ith(c, i) };
        let t = unsafe { llama_get_sampled_candidates_ith(c, i) };
        let lc = unsafe { llama_get_sampled_logits_count_ith(c, i) };
        let pc = unsafe { llama_get_sampled_probs_count_ith(c, i) };
        let tc = unsafe { llama_get_sampled_candidates_count_ith(c, i) };
        let label = field(event, "name").as_str().unwrap();
        assert_eq!(
            unsafe { llama_get_sampled_token_ith(c, i) },
            number(expected, "token") as i32,
            "{label} token {i}"
        );
        assert_eq!(
            (lc, pc, tc),
            (
                number(expected, "lc") as u32,
                number(expected, "pc") as u32,
                number(expected, "tc") as u32
            ),
            "{label} counts {i}"
        );
        assert_eq!(
            (!l.is_null(), !p.is_null(), !t.is_null()),
            (
                flag(expected, "lp"),
                flag(expected, "pp"),
                flag(expected, "tp")
            ),
            "{label} pointers {i}"
        );
        for (key, pointer) in [("logits", l), ("probs", p)] {
            let words = array(field(expected, key));
            if !words.is_empty() {
                let actual = unsafe { std::slice::from_raw_parts(pointer, words.len()) };
                for (index, (a, e)) in actual.iter().zip(words).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.as_i64().unwrap() as u32,
                        "{label} {key}[{index}] row {i}"
                    );
                }
            }
        }
        let ids = array(field(expected, "ids"));
        if !ids.is_empty() {
            let actual = unsafe { std::slice::from_raw_parts(t, ids.len()) };
            for (a, e) in actual.iter().zip(ids) {
                assert_eq!(*a, e.as_i64().unwrap() as i32, "{label} candidate row {i}");
            }
        }
        if let Some(words) = expected.get("query") {
            let q = unsafe { llama_get_logits_ith(c, i) };
            assert_eq!(
                !l.is_null() && q == l,
                flag(expected, "query_alias"),
                "{label} alias {i}"
            );
            let words = array(words);
            assert!(!q.is_null());
            for (a, e) in unsafe { std::slice::from_raw_parts(q, words.len()) }
                .iter()
                .zip(words)
            {
                assert_eq!(
                    a.to_bits(),
                    e.as_i64().unwrap() as u32,
                    "{label} queried row {i}"
                );
            }
        }
    }
}

#[test]
fn original_header_backend_sampling_context_and_lifecycle() {
    #[cfg(target_env = "msvc")]
    let expected = include_str!("../../../../../fixtures/ffi/backend-sampling/original-msvc.json");
    #[cfg(not(target_env = "msvc"))]
    let expected = include_str!("../../../../../fixtures/ffi/backend-sampling/original-gnu.json");
    let expected = json::parse(expected).unwrap();
    let scenarios = array(field(&expected, "scenarios"));
    assert_eq!(scenarios.len(), 26);
    unsafe {
        let m = model();
        for scenario in scenarios {
            let kind = number(scenario, "kind");
            let multi = number(scenario, "multi") != 0;
            let c = llama_init_from_model(m, parameters());
            assert!(!c.is_null());
            let s = chain(kind);
            let events = array(field(scenario, "events"));
            check_rows(c, &events[0]);
            assert_eq!(
                llama_set_sampler(c, 0, s),
                flag(scenario, "attached"),
                "kind {kind} attach"
            );
            let names = array(field(scenario, "names"));
            for (i, name) in names.iter().enumerate() {
                assert_eq!(
                    CStr::from_ptr(llama_sampler_name(llama_sampler_chain_get(s, i as i32)))
                        .to_str()
                        .unwrap(),
                    name.as_str().unwrap()
                );
            }
            check_rows(c, &events[1]);
            let mut tokens = vec![1, 3, 5, 7];
            let mut positions = if multi {
                vec![0, 0, 1, 1]
            } else {
                vec![0, 1, 2, 3]
            };
            let mut counts = vec![1; 4];
            let mut sequences = if multi { vec![1, 0, 1, 0] } else { vec![0; 4] };
            let mut pointers = sequences
                .iter_mut()
                .map(|s| s as *mut i32)
                .collect::<Vec<_>>();
            let mut flags = if multi {
                vec![0, 0, 1, 1]
            } else {
                vec![0, 0, 0, 1]
            };
            let b = LlamaBatch {
                n_tokens: 4,
                token: tokens.as_mut_ptr(),
                embd: ptr::null_mut(),
                pos: positions.as_mut_ptr(),
                n_seq_id: counts.as_mut_ptr(),
                seq_id: pointers.as_mut_ptr(),
                logits: flags.as_mut_ptr(),
            };
            assert_eq!(
                llama_decode(c, b),
                number(scenario, "decode") as i32,
                "kind {kind} multi {multi}"
            );
            check_rows(c, &events[2]);
            check_rows(c, &events[3]);
            if let Some(sample) = scenario.get("sample") {
                assert_eq!(
                    llama_sampler_sample(s, c, -1),
                    sample.as_i64().unwrap() as i32,
                    "kind {kind} sample"
                );
                assert_eq!(
                    llama_perf_sampler(s).n_sample,
                    number(scenario, "accepted") as i32
                );
            }
            assert_eq!(
                llama_set_sampler(c, 0, ptr::null_mut()),
                flag(scenario, "detached")
            );
            check_rows(c, &events[4]);
            let mut token = 9;
            let mut position = if multi { 2 } else { 4 };
            let b = LlamaBatch {
                n_tokens: 1,
                token: &mut token,
                embd: ptr::null_mut(),
                pos: &mut position,
                n_seq_id: ptr::null_mut(),
                seq_id: ptr::null_mut(),
                logits: ptr::null_mut(),
            };
            assert_eq!(
                llama_decode(c, b),
                number(scenario, "decode_after_detach") as i32
            );
            check_rows(c, &events[5]);
            llama_free(c);
            llama_sampler_free(s);
        }
        let s = chain(4);
        let mut config = LlamaSamplerSeqConfig {
            seq_id: 0,
            sampler: s,
        };
        let mut cp = parameters();
        cp.n_seq_max = 1;
        cp.samplers = (&mut config as *mut LlamaSamplerSeqConfig).cast();
        cp.n_samplers = 1;
        let c = llama_init_from_model(m, cp);
        assert!(!c.is_null());
        let mut tokens = [1, 3];
        let mut positions = [0, 1];
        let mut flags = [1, 1];
        let b = LlamaBatch {
            n_tokens: 2,
            token: tokens.as_mut_ptr(),
            embd: ptr::null_mut(),
            pos: positions.as_mut_ptr(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: flags.as_mut_ptr(),
        };
        assert_eq!(
            llama_decode(c, b),
            number(&expected, "multiple_outputs_status") as i32
        );
        flags = [0, 1];
        let b = LlamaBatch {
            logits: flags.as_mut_ptr(),
            ..b
        };
        assert_eq!(
            llama_decode(c, b),
            number(&expected, "recovery_status") as i32
        );
        assert_eq!(
            llama_sampler_sample(s, c, -1),
            number(&expected, "initial_config_sample") as i32
        );
        llama_free(c);
        llama_sampler_free(s);
        llama_model_free(m);
    }
}

#[test]
fn attached_lifetime_reinitialization_topology_and_checked_failures() {
    unsafe {
        let m = model();
        let c = llama_init_from_model(m, parameters());
        assert!(!c.is_null());
        let s = chain(4);
        assert!(llama_set_sampler(c, 0, s));
        llama_rust_clear_error();
        llama_sampler_free(s);
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_string_lossy()
                .contains("attached")
        );
        assert_eq!(llama_sampler_chain_n(s), 2);
        assert!(!llama_set_sampler(c, 1, s));
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_string_lossy()
                .contains("only once")
        );
        let extra = llama_sampler_init_temp(1.0);
        llama_sampler_chain_add(s, extra);
        assert_eq!(llama_sampler_chain_n(s), 2);
        assert!(llama_sampler_chain_remove(s, 0).is_null());
        llama_sampler_free(extra);
        let clone = llama_sampler_clone(s);
        assert!(!clone.is_null());
        assert_eq!(
            CStr::from_ptr(llama_sampler_name(llama_sampler_chain_get(clone, 0))).to_bytes(),
            b"top-k"
        );
        assert!(llama_set_sampler(c, 1, clone));
        let unsupported = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(unsupported, llama_sampler_init_dist(42));
        assert!(!llama_set_sampler(c, 0, unsupported));
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_string_lossy()
                .contains("not yet implemented")
        );
        // Failure preserves the prior attachment.
        let mut token = 3;
        let b = llama_batch_get_one(&mut token, 1);
        assert_eq!(llama_decode(c, b), 0);
        assert!(llama_get_sampled_token_ith(c, -1) >= 0);
        let l = llama_get_sampled_logits_ith(c, -1);
        assert!(!l.is_null());
        *l = 123.0;
        assert_eq!(*llama_get_logits_ith(c, -1), 123.0);
        let ids = llama_get_sampled_candidates_ith(c, -1);
        assert!(!ids.is_null());
        *ids = 11;
        assert_eq!(*llama_get_sampled_candidates_ith(c, -1), 11);
        llama_free(c);
        llama_sampler_free(s);
        llama_sampler_free(clone);
        llama_sampler_free(unsupported);
        llama_model_free(m);
    }
}

#[test]
fn original_cpu_suffix_typical_order_and_distribution() {
    #[cfg(target_env = "msvc")]
    let expected =
        include_str!("../../../../../fixtures/ffi/backend-sampling/cpu-suffix-original-msvc.json");
    #[cfg(not(target_env = "msvc"))]
    let expected =
        include_str!("../../../../../fixtures/ffi/backend-sampling/cpu-suffix-original-gnu.json");
    let expected = json::parse(expected).unwrap();
    for scenario in array(field(&expected, "scenarios")) {
        let input = field(scenario, "input");
        let ids = array(field(input, "ids"));
        let logits = array(field(input, "logits"));
        let probabilities = array(field(input, "probs"));
        let mut values: Vec<LlamaTokenData> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| LlamaTokenData {
                id: id.as_i64().unwrap() as i32,
                logit: f32::from_bits(logits[i].as_i64().unwrap() as u32),
                p: f32::from_bits(probabilities[i].as_i64().unwrap() as u32),
            })
            .collect();
        let mut candidates = LlamaTokenDataArray {
            data: values.as_mut_ptr(),
            size: values.len(),
            selected: number(input, "selected"),
            sorted: u8::from(flag(input, "sorted")),
        };
        unsafe {
            for (stage, sampler) in [
                ("typical", llama_sampler_init_typical(0.8, 1)),
                ("dist", llama_sampler_init_dist(42)),
            ] {
                assert!(!sampler.is_null());
                llama_sampler_apply(sampler, &mut candidates);
                let output = field(scenario, stage);
                assert_eq!(candidates.size, number(output, "size") as usize, "{stage}");
                assert_eq!(candidates.selected, number(output, "selected"), "{stage}");
                assert_eq!(candidates.sorted != 0, flag(output, "sorted"), "{stage}");
                let actual = std::slice::from_raw_parts(candidates.data, candidates.size);
                for (i, value) in actual.iter().enumerate() {
                    assert_eq!(
                        value.id,
                        array(field(output, "ids"))[i].as_i64().unwrap() as i32,
                        "{stage} id {i}"
                    );
                    assert_eq!(
                        value.logit.to_bits(),
                        array(field(output, "logits"))[i].as_i64().unwrap() as u32,
                        "{stage} logit {i}"
                    );
                    assert_eq!(
                        value.p.to_bits(),
                        array(field(output, "probs"))[i].as_i64().unwrap() as u32,
                        "{stage} probability {i}"
                    );
                }
                llama_sampler_free(sampler);
            }
        }
    }
}
