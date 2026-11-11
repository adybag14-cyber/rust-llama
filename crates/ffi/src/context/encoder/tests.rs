//! Persistent replay of independent original-header BERT C captures.
use super::*;
use crate::context::*;
use crate::memory::llama_get_memory;
use crate::model::*;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::hash_reader;
use std::{
    ffi::{CStr, CString},
    fs::File,
    path::{Path, PathBuf},
};

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("missing fixture field {name}"))
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture array")
}
fn integer(value: &Value) -> i32 {
    i32::try_from(value.as_i64().expect("fixture integer")).unwrap()
}
fn int_or(value: &Value, name: &str, default: i32) -> i32 {
    value.get(name).map_or(default, integer)
}
fn bool_or(value: &Value, name: &str, default: bool) -> bool {
    value
        .get(name)
        .map_or(default, |value| value.as_bool().expect("fixture boolean"))
}
fn corpus() -> Value {
    let text = if cfg!(target_os = "windows") {
        include_str!("../../../../../fixtures/ffi/encoder/windows-msvc.json")
    } else {
        include_str!("../../../../../fixtures/ffi/encoder/linux-gnu.json")
    };
    json::parse(text).expect("independent encoder corpus")
}
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/ffi/encoder")
        .join(name)
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
fn diagnostic() -> String {
    unsafe { CStr::from_ptr(llama_rust_last_error()) }
        .to_string_lossy()
        .into_owned()
}
unsafe fn load_raw(path: &Path) -> *mut ModelHandle {
    let path = CString::new(path.to_string_lossy().as_bytes()).unwrap();
    let params = LlamaModelParams {
        n_gpu_layers: 0,
        load_mode: 0,
        use_extra_bufts: 0,
        check_tensors: 1,
        ..unsafe { llama_model_default_params() }
    };
    unsafe { llama_model_load_from_file(path.as_ptr(), params) }
}
unsafe fn load(path: &Path) -> ModelOwner {
    let model = unsafe { load_raw(path) };
    assert!(!model.is_null(), "{}", diagnostic());
    ModelOwner(model)
}
fn parameters(p: &Value) -> LlamaContextParams {
    LlamaContextParams {
        n_ctx: int_or(p, "context", 32) as u32,
        n_batch: int_or(p, "batch", 32) as u32,
        n_ubatch: int_or(p, "ubatch", 32) as u32,
        n_seq_max: int_or(p, "sequences", 4) as u32,
        n_threads: int_or(p, "threads", 1),
        n_threads_batch: int_or(p, "threads_batch", 1),
        no_perf: 1,
        type_k: 0,
        type_v: 0,
        offload_kqv: 0,
        op_offload: 0,
        embeddings: u8::from(bool_or(p, "embeddings", true)),
        pooling_type: int_or(p, "pooling", -1),
        attention_type: int_or(p, "attention", -1),
        flash_attn_type: int_or(p, "flash", 0),
        kv_unified: u8::from(bool_or(p, "unified", true)),
        ..LlamaContextParams::default()
    }
}
#[derive(Default)]
struct Stats {
    values: usize,
    max_abs: f64,
}
unsafe fn vector(pointer: *const f32, expected: &Value, pretrained: bool, stats: &mut Stats) {
    if expected == &Value::Null {
        assert!(pointer.is_null());
        return;
    }
    assert!(!pointer.is_null(), "expected live output: {}", diagnostic());
    let words = array(expected);
    let values = unsafe { std::slice::from_raw_parts(pointer, words.len()) };
    for (&actual, expected) in values.iter().zip(words) {
        let expected = f32::from_bits(u32::try_from(expected.as_i64().unwrap()).unwrap());
        let error = (f64::from(actual) - f64::from(expected)).abs();
        let budget = if pretrained {
            2e-4 + 2e-5 * f64::from(expected).abs()
        } else {
            2e-6 + 2e-6 * f64::from(expected).abs()
        };
        assert!(
            actual.is_finite() && error <= budget,
            "value {actual} != {expected}: error {error}, budget {budget}"
        );
        stats.values += 1;
        stats.max_abs = stats.max_abs.max(error);
    }
}
unsafe fn snapshot(
    context: *mut ContextHandle,
    expected: &Value,
    pretrained: bool,
    stats: &mut Stats,
) {
    unsafe {
        assert_eq!(
            llama_get_causal_attn(context),
            bool_or(expected, "causal", false)
        );
        assert_eq!(
            !llama_get_memory(context).is_null(),
            bool_or(expected, "memory", false)
        );
        let raw = llama_get_embeddings(context);
        assert_eq!(!raw.is_null(), bool_or(expected, "raw_embeddings", false));
        assert_eq!(
            !llama_get_logits(context).is_null(),
            bool_or(expected, "raw_logits", false)
        );
        for row in array(field(expected, "rows")) {
            let index = integer(field(row, "index"));
            let embedding = llama_get_embeddings_ith(context, index);
            let logits = llama_get_logits_ith(context, index);
            vector(embedding, field(row, "embedding"), pretrained, stats);
            assert_eq!(!logits.is_null(), bool_or(row, "logits", false));
            if let Some(prefix) = row.get("logits_prefix") {
                vector(logits, prefix, pretrained, stats);
            }
            match field(row, "embedding_offset") {
                Value::Null => assert!(embedding.is_null() || raw.is_null()),
                offset => assert_eq!(embedding.offset_from(raw), integer(offset) as isize),
            }
        }
        for row in array(field(expected, "sequences")) {
            let id = integer(field(row, "id"));
            let pointer = llama_get_embeddings_seq(context, id);
            vector(pointer, field(row, "embedding"), pretrained, stats);
            assert_eq!(
                pointer == llama_get_embeddings_seq(context, id),
                bool_or(row, "stable", false)
            );
        }
    }
}
#[derive(Default)]
struct Abort {
    after: usize,
    calls: usize,
}
unsafe extern "C" fn abort_callback(data: *mut std::ffi::c_void) -> bool {
    let data = unsafe { &mut *data.cast::<Abort>() };
    data.calls += 1;
    data.after > 0 && data.calls >= data.after
}
struct Batch {
    tokens: Option<Vec<i32>>,
    embeddings: Option<Vec<f32>>,
    embedding_rows: usize,
    positions: Option<Vec<i32>>,
    sequences: Option<Vec<Vec<i32>>>,
    flags: Option<Vec<i8>>,
    counts: Vec<i32>,
    pointers: Vec<*mut i32>,
}
impl Batch {
    fn new(input: &Value) -> Self {
        let integers = |v: &Value| array(v).iter().map(integer).collect::<Vec<_>>();
        let sequences: Option<Vec<Vec<i32>>> = input
            .get("sequences")
            .map(|v| array(v).iter().map(integers).collect());
        Self {
            tokens: input.get("tokens").map(integers),
            embeddings: input.get("embedding").map(|v| {
                array(v)
                    .iter()
                    .flat_map(|row| array(row).iter().map(|v| v.as_f64().unwrap() as f32))
                    .collect()
            }),
            embedding_rows: input.get("embedding").map_or(0, |v| array(v).len()),
            positions: input.get("positions").map(integers),
            sequences,
            flags: input.get("flags").map(|v| {
                array(v)
                    .iter()
                    .map(|v| i8::try_from(integer(v)).unwrap())
                    .collect()
            }),
            counts: Vec::new(),
            pointers: Vec::new(),
        }
    }
    fn raw(&mut self) -> LlamaBatch {
        let n = if self.embeddings.is_some() {
            self.embedding_rows
        } else {
            self.tokens.as_ref().map_or(0, Vec::len)
        };
        if let Some(sequences) = &mut self.sequences {
            self.counts = sequences.iter().map(|ids| ids.len() as i32).collect();
            self.pointers = sequences.iter_mut().map(|ids| ids.as_mut_ptr()).collect();
        }
        LlamaBatch {
            n_tokens: n as i32,
            token: self
                .tokens
                .as_mut()
                .map_or(ptr::null_mut(), |v| v.as_mut_ptr()),
            embd: self
                .embeddings
                .as_mut()
                .map_or(ptr::null_mut(), |v| v.as_mut_ptr()),
            pos: self
                .positions
                .as_mut()
                .map_or(ptr::null_mut(), |v| v.as_mut_ptr()),
            n_seq_id: if self.sequences.is_some() {
                self.counts.as_mut_ptr()
            } else {
                ptr::null_mut()
            },
            seq_id: if self.sequences.is_some() {
                self.pointers.as_mut_ptr()
            } else {
                ptr::null_mut()
            },
            logits: self
                .flags
                .as_mut()
                .map_or(ptr::null_mut(), |v| v.as_mut_ptr()),
        }
    }
}
unsafe fn verify_case(case: &Value, path: &Path, pretrained: bool, stats: &mut Stats) {
    unsafe {
        let input = field(case, "input");
        let expected = field(case, "expected");
        if !bool_or(expected, "model_loaded", false) {
            let model = ModelOwner(load_raw(path));
            assert!(model.0.is_null(), "source rejects this model");
            return;
        }
        let model = load(path);
        let metadata = field(expected, "model");
        assert_eq!(
            llama_model_n_embd(model.0),
            integer(field(metadata, "width"))
        );
        assert_eq!(
            llama_model_n_cls_out(model.0),
            integer(field(metadata, "classes")) as u32
        );
        assert_eq!(
            llama_model_has_encoder(model.0),
            bool_or(metadata, "encoder", false)
        );
        assert_eq!(
            llama_model_has_decoder(model.0),
            bool_or(metadata, "decoder", false)
        );
        for (index, label) in array(field(metadata, "labels")).iter().enumerate() {
            let actual = llama_model_cls_label(model.0, index as u32);
            if label == &Value::Null {
                assert!(actual.is_null());
            } else {
                assert!(!actual.is_null());
                assert_eq!(
                    CStr::from_ptr(actual).to_bytes(),
                    label.as_str().unwrap().as_bytes()
                );
            }
        }
        assert!(llama_model_cls_label(model.0, u32::MAX).is_null());
        let context = ContextOwner(llama_init_from_model(
            model.0,
            parameters(field(input, "params")),
        ));
        assert!(!context.0.is_null(), "{}", diagnostic());
        let expected_context = field(expected, "context");
        for (name, actual) in [
            ("n_ctx", llama_n_ctx(context.0)),
            ("n_ctx_seq", llama_n_ctx_seq(context.0)),
            ("n_batch", llama_n_batch(context.0)),
            ("n_ubatch", llama_n_ubatch(context.0)),
            ("n_seq_max", llama_n_seq_max(context.0)),
            ("pooling", llama_pooling_type(context.0) as u32),
            ("threads", llama_n_threads(context.0) as u32),
            ("threads_batch", llama_n_threads_batch(context.0) as u32),
        ] {
            assert_eq!(
                actual,
                integer(field(expected_context, name)) as u32,
                "{name}"
            );
        }
        snapshot(context.0, field(expected, "initial"), pretrained, stats);
        let operations = array(field(input, "ops"));
        let events = array(field(expected, "events"));
        assert_eq!(operations.len(), events.len());
        let mut abort = Abort::default();
        for (operation, event) in operations.iter().zip(events) {
            let name = field(operation, "op").as_str().unwrap();
            assert_eq!(field(event, "op").as_str(), Some(name));
            match name {
                "embeddings" => {
                    llama_set_embeddings(context.0, u8::from(bool_or(operation, "value", false)))
                }
                "causal" => {
                    llama_set_causal_attn(context.0, u8::from(bool_or(operation, "value", false)))
                }
                "threads" => llama_set_n_threads(
                    context.0,
                    integer(field(operation, "generation")),
                    integer(field(operation, "batch")),
                ),
                "abort" => {
                    abort.after = integer(field(operation, "after")) as usize;
                    abort.calls = 0;
                    llama_set_abort_callback(
                        context.0,
                        if abort.after == 0 {
                            None
                        } else {
                            Some(abort_callback)
                        },
                        ptr::from_mut(&mut abort).cast(),
                    );
                }
                "encode" | "decode" => {
                    let mut batch = Batch::new(operation);
                    let result = if name == "encode" {
                        llama_encode(context.0, batch.raw())
                    } else {
                        llama_decode(context.0, batch.raw())
                    };
                    assert_eq!(result, integer(field(event, "rc")), "{}", diagnostic());
                    llama_synchronize(context.0);
                    assert_eq!(abort.calls > 0, bool_or(event, "abort_invoked", false));
                }
                _ => panic!("unknown operation {name}"),
            }
            snapshot(context.0, field(event, "outputs"), pretrained, stats);
        }
    }
}
#[test]
fn independent_native_bert_encoder_corpus() {
    let corpus = corpus();
    let mut stats = Stats::default();
    let mut count = 0;
    for case in array(field(&corpus, "cases")) {
        let input = field(case, "input");
        let name = field(input, "model").as_str().unwrap();
        if name == "@bge" {
            continue;
        }
        eprintln!(
            "native encoder case: {}",
            field(input, "name").as_str().unwrap()
        );
        unsafe {
            verify_case(case, &fixture(name), false, &mut stats);
        }
        count += 1;
    }
    assert_eq!(count, 188);
    eprintln!(
        "native encoder: {count} cases, {} values, max_abs={:.9e}",
        stats.values, stats.max_abs
    );
}
#[test]
#[ignore = "requires the pinned external BGE model; set LLAMA_TEST_BERT_MODEL"]
fn independent_pretrained_native_bert_encoder_corpus() {
    let path = PathBuf::from(
        std::env::var_os("LLAMA_TEST_BERT_MODEL").expect("set LLAMA_TEST_BERT_MODEL"),
    );
    let (bytes, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(bytes, 67_308_128);
    assert_eq!(
        hash,
        "f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999"
    );
    let corpus = corpus();
    let mut stats = Stats::default();
    let mut count = 0;
    for case in array(field(&corpus, "cases")) {
        if field(field(case, "input"), "model").as_str() != Some("@bge") {
            continue;
        }
        unsafe {
            verify_case(case, &path, true, &mut stats);
        }
        count += 1;
    }
    assert_eq!(count, 4);
    eprintln!(
        "pretrained native encoder: {count} cases, {} values, max_abs={:.9e}",
        stats.values, stats.max_abs
    );
}
#[test]
fn encoder_ownership_and_source_invalid_inputs_are_bounded() {
    unsafe {
        let model = load(&fixture("rank-full.gguf"));
        let params = parameters(&json::parse(r#"{"pooling":4}"#).unwrap());
        let context = ContextOwner(llama_init_from_model(model.0, params));
        assert!(!context.0.is_null(), "{}", diagnostic());
        let borrowed = llama_get_model(context.0);
        assert_ne!(borrowed, model.0);
        let label = llama_model_cls_label(borrowed, 1);
        assert_eq!(CStr::from_ptr(label).to_bytes(), b"middle");
        drop(model);
        assert_eq!(llama_model_n_cls_out(borrowed), 3);
        let mut valid = Batch::new(&json::parse(r#"{"tokens":[101,104,105,102]}"#).unwrap());
        assert_eq!(llama_encode(context.0, valid.raw()), 0, "{}", diagnostic());
        let pooled = llama_get_embeddings_seq(context.0, 0);
        assert!(!pooled.is_null());
        let previous = std::slice::from_raw_parts(pooled, 3).to_vec();
        for input in [
            r#"{"tokens":[101,104],"positions":[-1,0]}"#,
            r#"{"tokens":[101,104],"positions":[0,8]}"#,
            r#"{"tokens":[101],"sequences":[[]]}"#,
            r#"{"tokens":[101],"sequences":[[4]]}"#,
            r#"{"tokens":[108]}"#,
        ] {
            let mut batch = Batch::new(&json::parse(input).unwrap());
            assert_eq!(llama_encode(context.0, batch.raw()), -1, "{input}");
            assert_eq!(llama_get_embeddings_seq(context.0, 0), pooled);
            assert_eq!(std::slice::from_raw_parts(pooled, 3), previous);
        }
        assert_eq!(CStr::from_ptr(label).to_bytes(), b"middle");
        assert!(llama_get_memory(context.0).is_null());
        assert_eq!(crate::state::llama_state_get_size(context.0), 0);
        let bad = ContextOwner(llama_init_from_model(
            borrowed.cast_mut(),
            LlamaContextParams {
                n_ubatch: 2,
                ..params
            },
        ));
        assert!(!bad.0.is_null());
        assert_eq!(llama_encode(bad.0, valid.raw()), -1);
        assert!(llama_get_embeddings_seq(bad.0, 0).is_null());
    }
}

#[test]
fn native_i32_set_order_matches_both_independent_compiler_profiles() {
    for (profile, text) in [
        (
            Profile::Msvc,
            include_str!("../../../../../fixtures/ffi/encoder/sequence-order-windows-msvc.json"),
        ),
        (
            Profile::Gnu,
            include_str!("../../../../../fixtures/ffi/encoder/sequence-order-linux-gnu.json"),
        ),
    ] {
        let reference = json::parse(text).unwrap();
        let cases = array(field(&reference, "cases"));
        assert_eq!(cases.len(), 569);
        for case in cases {
            let input: Vec<_> = array(field(case, "input")).iter().map(integer).collect();
            let expected: Vec<_> = array(field(case, "copied")).iter().map(integer).collect();
            assert_eq!(
                sequence_set_order(&input, profile).unwrap(),
                expected,
                "{profile:?}: {}",
                field(case, "name").as_str().unwrap()
            );
        }
    }
    assert!(sequence_set_order(&[-1], Profile::Msvc).is_err());
    assert!(sequence_set_order(&[256], Profile::Gnu).is_err());
    assert!(sequence_set_order(&[0; 257], Profile::Gnu).is_err());
}

#[test]
fn rust_embedding_facade_executes_optional_classifier_heads() {
    let corpus = corpus();
    let mut stats = Stats::default();
    let mut count = 0;
    for case in array(field(&corpus, "cases")) {
        let input = field(case, "input");
        let expected = field(case, "expected");
        if field(input, "name").as_str().unwrap()
            != format!("{}:pool4", field(input, "model").as_str().unwrap())
            || !bool_or(expected, "model_loaded", false)
        {
            continue;
        }
        let model =
            llama_inference::Model::load(fixture(field(input, "model").as_str().unwrap())).unwrap();
        let ids: Vec<_> = array(field(&array(field(input, "ops"))[0], "tokens"))
            .iter()
            .map(integer)
            .collect();
        let output = model
            .embed_tokens(&ids, Pooling::Rank, &|| false)
            .unwrap()
            .pooled
            .unwrap();
        let event = &array(field(expected, "events"))[0];
        let sequence = array(field(field(event, "outputs"), "sequences"))
            .iter()
            .find(|sequence| integer(field(sequence, "id")) == 0)
            .unwrap();
        let expected = field(sequence, "embedding");
        assert_eq!(output.len(), array(expected).len());
        unsafe {
            vector(output.as_ptr(), expected, false, &mut stats);
        }
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn encoder_shape_preflight_precedes_foreign_input_copy() {
    unsafe {
        let model = load(&fixture("rank-full.gguf"));
        let params = parameters(
            &json::parse(r#"{"pooling":4,"flash":1,"context":65536,"batch":65536,"ubatch":65536}"#)
                .unwrap(),
        );
        let context = ContextOwner(llama_init_from_model(model.0, params));
        assert!(!context.0.is_null(), "{}", diagnostic());
        let mut valid = Batch::new(&json::parse(r#"{"tokens":[101,104,105,102]}"#).unwrap());
        assert_eq!(llama_encode(context.0, valid.raw()), 0, "{}", diagnostic());
        let pooled = llama_get_embeddings_seq(context.0, 0);
        assert!(!pooled.is_null());
        let previous = std::slice::from_raw_parts(pooled, 3).to_vec();
        // No foreign pointer is dereferenced: the shape exceeds the flash-mask
        // bound and must be rejected before even the missing-input check.
        let mut batch = LlamaBatch {
            n_tokens: 65_536,
            token: ptr::null_mut(),
            embd: ptr::null_mut(),
            pos: ptr::null_mut(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        };
        assert_eq!(llama_encode(context.0, batch), -1);
        assert!(diagnostic().contains("512 MiB payload or 2^36 scalar-work bound"));
        assert_eq!(llama_get_embeddings_seq(context.0, 0), pooled);
        assert_eq!(std::slice::from_raw_parts(pooled, 3), previous);
        batch.n_tokens = 1;
        assert_eq!(llama_encode(context.0, batch), -1);
        assert!(diagnostic().contains("neither tokens nor host embeddings"));
        assert_eq!(llama_get_embeddings_seq(context.0, 0), pooled);
    }
}

#[test]
fn independent_native_bert_vector_storage_admission() {
    let text = if cfg!(target_os = "windows") {
        include_str!("../../../../../fixtures/ffi/encoder/vector-admission-windows-msvc.json")
    } else {
        include_str!("../../../../../fixtures/ffi/encoder/vector-admission-linux-gnu.json")
    };
    let reference = json::parse(text).unwrap();
    let mut stats = Stats::default();
    let mut rejected = 0;
    let mut executed = 0;
    for case in array(field(&reference, "cases")) {
        let input = field(case, "input");
        let name = field(input, "model").as_str().unwrap();
        unsafe {
            if bool_or(case, "source_rejected", false) {
                assert!(bool_or(input, "active_non_f32", false));
                let model = ModelOwner(load_raw(&fixture(name)));
                assert!(model.0.is_null(), "{name}: expected a bounded loader error");
                assert!(diagnostic().contains("requires F32 for the source binary graph"));
                rejected += 1;
            } else {
                verify_case(case, &fixture(name), false, &mut stats);
                executed += 1;
            }
        }
    }
    assert_eq!((rejected, executed), (5, 4));
    eprintln!(
        "native BERT vector admission: {rejected} bounded rejections, {executed} original executions, {} values, max_abs={:.9e}",
        stats.values, stats.max_abs
    );
}
