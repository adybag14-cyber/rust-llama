//! Independently captured frozen-DLL regressions; this test module contains no
//! native linkage and uses the exact source model's small vocabulary-only GGUF.
use crate::{sampler::*, *};
use llama_behavior::json::{self, Value};
use std::ffi::{CStr, CString};
fn field<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing {k}"))
}
fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn int(v: &Value) -> i64 {
    v.as_i64().unwrap()
}
fn f32_value(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}
fn number(v: &Value, k: &str) -> i64 {
    int(field(v, k))
}
fn boolean_value(v: &Value, k: &str) -> bool {
    field(v, k).as_bool().unwrap()
}
fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    field(v, k).as_str().unwrap()
}
fn tokens(v: &Value) -> Vec<i32> {
    array(v).iter().map(|v| int(v) as i32).collect()
}
fn cstrings(v: Option<&Value>) -> Vec<CString> {
    v.map(array)
        .unwrap_or_default()
        .iter()
        .map(|v| CString::new(v.as_str().unwrap()).unwrap())
        .collect()
}
fn clean() {
    unsafe {
        llama_rust_clear_error();
    }
}
fn okay() {
    unsafe {
        assert_eq!(CStr::from_ptr(llama_rust_last_error()).to_bytes(), b"");
    }
}
struct ModelOwner(*mut ModelHandle);
impl ModelOwner {
    fn load() -> Self {
        let path = CString::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/behavior/stories260k-vocab.gguf"
        ))
        .unwrap();
        unsafe {
            let mut params = llama_model_default_params();
            params.vocab_only = 1;
            clean();
            let ptr = model::llama_model_load_from_file(path.as_ptr(), params);
            okay();
            assert!(!ptr.is_null());
            Self(ptr)
        }
    }
    fn vocab(&self) -> *const VocabHandle {
        unsafe { model::llama_model_get_vocab(self.0) }
    }
}
impl Drop for ModelOwner {
    fn drop(&mut self) {
        unsafe {
            model::llama_model_free(self.0);
        }
    }
}
fn make(vocab: *const VocabHandle, kind: &str, c: &Value) -> *mut LlamaSampler {
    clean();
    unsafe {
        match kind {
            "grammar" => {
                if c.get("null").and_then(Value::as_bool) == Some(true) {
                    return llama_sampler_init_grammar(vocab, ptr::null(), ptr::null());
                }
                let source = CString::new(text(c, "grammar")).unwrap();
                let root =
                    CString::new(c.get("root").and_then(Value::as_str).unwrap_or("root")).unwrap();
                let mode = c.get("kind").and_then(Value::as_str).unwrap_or("grammar");
                let strings = cstrings(c.get("triggers"));
                let pointers: Vec<_> = strings.iter().map(|s| s.as_ptr()).collect();
                let ids = c.get("trigger_tokens").map(tokens).unwrap_or_default();
                match mode {
                    "grammar" => llama_sampler_init_grammar(vocab, source.as_ptr(), root.as_ptr()),
                    "lazy-words" => llama_sampler_init_grammar_lazy(
                        vocab,
                        source.as_ptr(),
                        root.as_ptr(),
                        pointers.as_ptr(),
                        pointers.len(),
                        ids.as_ptr(),
                        ids.len(),
                    ),
                    _ => llama_sampler_init_grammar_lazy_patterns(
                        vocab,
                        source.as_ptr(),
                        root.as_ptr(),
                        pointers.as_ptr(),
                        pointers.len(),
                        ids.as_ptr(),
                        ids.len(),
                    ),
                }
            }
            "dry" => {
                let strings = cstrings(c.get("breakers"));
                let pointers: Vec<_> = strings.iter().map(|s| s.as_ptr()).collect();
                llama_sampler_init_dry(
                    vocab,
                    f32_value(field(c, "multiplier")),
                    f32_value(field(c, "base")),
                    number(c, "allowed_length") as i32,
                    number(c, "last_n") as i32,
                    pointers.as_ptr(),
                    pointers.len(),
                )
            }
            "adaptive" => llama_sampler_init_adaptive_p(
                f32_value(field(c, "target")),
                f32_value(field(c, "decay")),
                number(c, "seed") as u32,
            ),
            "infill" => llama_sampler_init_infill(vocab),
            _ => panic!("unknown case"),
        }
    }
}
fn observe(s: *mut LlamaSampler, c: &Value, mode: &str) {
    let n = number(c, "n") as usize;
    let seed = number(c, "seed") as usize;
    let sorted = boolean_value(c, "input_sorted");
    let mut rows: Vec<LlamaTokenData> = if let Some(input) = c.get("input") {
        array(input)
            .iter()
            .map(|v| {
                let v = array(v);
                LlamaTokenData {
                    id: int(&v[0]) as i32,
                    logit: f32::from_bits(int(&v[1]) as u32),
                    p: f32::from_bits(int(&v[2]) as u32),
                }
            })
            .collect()
    } else {
        let mut rows: Vec<_> = (0..n)
            .map(|i| LlamaTokenData {
                id: i as i32,
                logit: ((i * 37 + seed * 71) % 101) as f32 * 0.0625 - 3.0,
                p: ((i + 7) % 19) as f32 * 0.03125,
            })
            .collect();
        if sorted {
            rows.sort_by(|a, b| b.logit.total_cmp(&a.logit));
        } else if n > 0 {
            rows.rotate_left(seed % n);
        }
        rows
    };
    let input = rows.clone();
    let start = rows.as_mut_ptr();
    let mut candidate = LlamaTokenDataArray {
        data: start,
        size: rows.len(),
        selected: number(c, "input_selected"),
        sorted: u8::from(sorted),
    };
    clean();
    unsafe {
        llama_sampler_apply(s, &mut candidate);
    }
    okay();
    assert_eq!(candidate.size, number(c, "size") as usize);
    assert_eq!(candidate.selected, number(c, "selected"));
    assert_eq!(candidate.sorted != 0, boolean_value(c, "sorted"));
    assert_eq!(
        unsafe { candidate.data.offset_from(start) },
        number(c, "offset") as isize
    );
    let output = unsafe { std::slice::from_raw_parts(candidate.data, candidate.size) };
    if mode == "grammar" {
        let mut allowed = Vec::new();
        for (a, b) in output.iter().zip(&input) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.p.to_bits(), b.p.to_bits());
            if a.logit.is_finite() {
                assert_eq!(a.logit.to_bits(), b.logit.to_bits());
                allowed.push(a.id);
            } else {
                assert_eq!(a.logit, f32::NEG_INFINITY);
            }
        }
        assert_eq!(allowed, tokens(field(c, "allowed")));
    } else if mode == "dry" {
        let mut changed = Vec::new();
        for (a, b) in output.iter().zip(&input) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.p.to_bits(), b.p.to_bits());
            if a.logit.to_bits() != b.logit.to_bits() {
                changed.push((a.id, a.logit.to_bits()));
            }
        }
        let expected: Vec<_> = array(field(c, "changed"))
            .iter()
            .map(|v| {
                let a = array(v);
                (int(&a[0]) as i32, int(&a[1]) as u32)
            })
            .collect();
        assert_eq!(changed, expected);
    } else {
        let expected = array(field(c, "output"));
        assert_eq!(output.len(), expected.len());
        for (a, b) in output.iter().zip(expected) {
            let b = array(b);
            assert_eq!(
                (a.id, a.logit.to_bits(), a.p.to_bits()),
                (int(&b[0]) as i32, int(&b[1]) as u32, int(&b[2]) as u32)
            );
        }
    }
}
fn accept(s: *mut LlamaSampler, token: i32) {
    clean();
    unsafe {
        llama_sampler_accept(s, token);
    }
    okay();
}
fn clone_sampler(s: *mut LlamaSampler) -> *mut LlamaSampler {
    clean();
    let p = unsafe { llama_sampler_clone(s) };
    okay();
    assert!(!p.is_null());
    p
}
fn reset(s: *mut LlamaSampler) {
    clean();
    unsafe {
        llama_sampler_reset(s);
    }
    okay();
}
fn copy(s: *mut LlamaSampler, d: *mut LlamaSampler) {
    clean();
    unsafe {
        llama_sampler_copy(s, d);
    }
    okay();
}
fn free(s: *mut LlamaSampler) {
    unsafe {
        llama_sampler_free(s);
    }
}
fn corpus() -> Value {
    json::parse(include_str!(
        "../../../fixtures/behavior/native-sampler-cpu-reference.json"
    ))
    .unwrap()
}
#[test]
fn frozen_dll_grammar_lazy_and_dry_corpus() {
    let model = ModelOwner::load();
    let v = model.vocab();
    let document = corpus();
    for c in array(field(&document, "grammar")) {
        let request = field(c, "request");
        let s = make(v, "grammar", request);
        assert_eq!(!s.is_null(), boolean_value(c, "constructed"));
        if s.is_null() {
            continue;
        }
        okay();
        let steps = array(field(c, "steps"));
        observe(s, &steps[0], "grammar");
        let accepted = tokens(field(c, "tokens"));
        for (i, &token) in accepted.iter().take(steps.len() - 1).enumerate() {
            accept(s, token);
            observe(s, &steps[i + 1], "grammar");
            if c.get("clone_at").and_then(Value::as_u64) == Some((i + 1) as u64) {
                let other = clone_sampler(s);
                observe(other, field(c, "clone"), "grammar");
                free(other);
                let other = make(v, "grammar", request);
                okay();
                copy(s, other);
                observe(other, field(c, "copy"), "grammar");
                free(other);
            }
        }
        if let Some(eos) = c.get("accepted_eos") {
            accept(s, int(eos) as i32);
            observe(s, field(c, "after_eos"), "grammar");
        }
        reset(s);
        observe(s, field(c, "reset"), "grammar");
        free(s);
    }
    for c in array(field(&document, "dry")) {
        let request = field(c, "request");
        let s = make(v, "dry", request);
        okay();
        assert!(!s.is_null());
        observe(s, field(c, "initial"), "dry");
        let history = tokens(field(c, "tokens"));
        let mut checkpoint = 0;
        let steps = array(field(c, "steps"));
        for (i, token) in history.into_iter().enumerate() {
            accept(s, token);
            if checkpoint < steps.len() && number(&steps[checkpoint], "accepted") as usize == i + 1
            {
                observe(s, field(&steps[checkpoint], "result"), "dry");
                checkpoint += 1;
            }
        }
        assert_eq!(checkpoint, steps.len());
        let other = clone_sampler(s);
        observe(other, field(c, "clone"), "dry");
        free(other);
        let other = make(v, "dry", request);
        okay();
        copy(s, other);
        observe(other, field(c, "copy"), "dry");
        free(other);
        reset(s);
        observe(s, field(c, "reset"), "dry");
        free(s);
    }
}
#[test]
#[cfg(target_env = "msvc")]
fn frozen_dll_adaptive_rng_and_infill_candidate_corpus() {
    let model = ModelOwner::load();
    let v = model.vocab();
    let document = corpus();
    for c in array(field(&document, "adaptive")) {
        let request = field(c, "request");
        let s = make(v, "adaptive", request);
        okay();
        assert!(!s.is_null());
        for (step, observation) in array(field(c, "steps")).iter().enumerate() {
            if step == 29 {
                reset(s);
            }
            observe(s, observation, "adaptive");
            accept(s, number(observation, "accepted") as i32);
            if step == 17 {
                let other = clone_sampler(s);
                for observation in array(field(c, "clone")) {
                    observe(other, observation, "adaptive");
                    accept(other, number(observation, "accepted") as i32);
                }
                free(other);
                let other = make(v, "adaptive", request);
                okay();
                copy(s, other);
                observe(other, field(c, "copy"), "adaptive");
                free(other);
            }
        }
        free(s);
    }
    for c in array(field(&document, "infill")) {
        let s = make(v, "infill", field(c, "request"));
        okay();
        assert!(!s.is_null());
        observe(s, field(c, "apply"), "infill");
        let other = clone_sampler(s);
        observe(other, field(c, "clone"), "infill");
        free(other);
        let other = make(v, "infill", field(c, "request"));
        okay();
        copy(s, other);
        observe(other, field(c, "copy"), "infill");
        free(other);
        reset(s);
        observe(s, field(c, "reset"), "infill");
        free(s);
    }
}

#[test]
fn pending_adaptive_clone_and_copy_fail_atomically() {
    unsafe {
        let source = llama_sampler_init_adaptive_p(0.3, 0.8, 42);
        let destination = llama_sampler_init_adaptive_p(0.3, 0.8, 42);
        let control = llama_sampler_clone(destination);
        let mut rows = vec![
            LlamaTokenData {
                id: 0,
                logit: 1.0,
                p: 0.0,
            },
            LlamaTokenData {
                id: 1,
                logit: 0.0,
                p: 0.0,
            },
        ];
        let mut candidates = LlamaTokenDataArray {
            data: rows.as_mut_ptr(),
            size: 2,
            selected: -1,
            sorted: 0,
        };
        clean();
        llama_sampler_apply(source, &mut candidates);
        okay();
        clean();
        assert!(llama_sampler_clone(source).is_null());
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_str()
                .unwrap()
                .contains("pending")
        );
        clean();
        llama_sampler_copy(source, destination);
        assert!(
            CStr::from_ptr(llama_rust_last_error())
                .to_str()
                .unwrap()
                .contains("pending")
        );
        let mut left = rows.clone();
        let mut right = rows.clone();
        let mut a = LlamaTokenDataArray {
            data: left.as_mut_ptr(),
            size: 2,
            selected: -1,
            sorted: 0,
        };
        let mut b = LlamaTokenDataArray {
            data: right.as_mut_ptr(),
            size: 2,
            selected: -1,
            sorted: 0,
        };
        clean();
        llama_sampler_apply(destination, &mut a);
        okay();
        clean();
        llama_sampler_apply(control, &mut b);
        okay();
        assert_eq!(a.selected, b.selected);
        for (a, b) in left.iter().zip(&right) {
            assert_eq!(
                (a.id, a.logit.to_bits(), a.p.to_bits()),
                (b.id, b.logit.to_bits(), b.p.to_bits())
            );
        }
        accept(
            source,
            (*candidates.data.add(candidates.selected as usize)).id,
        );
        let clone = clone_sampler(source);
        free(clone);
        free(source);
        free(destination);
        free(control);
    }
}
#[test]
fn empty_grammar_ignores_unused_pointers_and_owned_vocabulary_survives_model_free() {
    unsafe {
        clean();
        let empty = llama_sampler_init_grammar_lazy_patterns(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            usize::MAX,
            ptr::null(),
            usize::MAX,
        );
        okay();
        assert!(!empty.is_null());
        llama_sampler_apply(empty, ptr::null_mut());
        okay();
        llama_sampler_accept(empty, i32::MIN);
        okay();
        free(empty);
        let model = ModelOwner::load();
        let vocab = model.vocab();
        let grammar =
            llama_sampler_init_grammar(vocab, c"root ::= \"a\"\n".as_ptr(), c"root".as_ptr());
        let infill = llama_sampler_init_infill(vocab);
        let dry = llama_sampler_init_dry(vocab, 0.8, 1.75, 2, 16, ptr::null(), 0);
        okay();
        assert!(!grammar.is_null() && !infill.is_null() && !dry.is_null());
        let n = crate::vocab::llama_vocab_n_tokens(vocab) as usize;
        drop(model);
        let mut rows: Vec<_> = (0..n)
            .map(|id| LlamaTokenData {
                id: id as i32,
                logit: -(id as f32),
                p: 0.0,
            })
            .collect();
        let mut a = LlamaTokenDataArray {
            data: rows.as_mut_ptr(),
            size: n,
            selected: 17,
            sorted: 1,
        };
        clean();
        llama_sampler_apply(grammar, &mut a);
        okay();
        let before: Vec<_> = rows.iter().map(|r| r.logit.to_bits()).collect();
        clean();
        llama_sampler_accept(grammar, -1);
        assert!(
            !CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .is_empty()
        );
        clean();
        llama_sampler_apply(grammar, &mut a);
        okay();
        assert_eq!(
            before,
            rows.iter().map(|r| r.logit.to_bits()).collect::<Vec<_>>()
        );
        for row in &mut rows {
            row.logit = -(row.id as f32);
        }
        clean();
        llama_sampler_apply(infill, &mut a);
        okay();
        for token in [10, 20, 10, 20] {
            accept(dry, token);
        }
        a.data = rows.as_mut_ptr();
        a.size = n;
        clean();
        llama_sampler_apply(dry, &mut a);
        okay();
        free(grammar);
        free(infill);
        free(dry);
    }
}
