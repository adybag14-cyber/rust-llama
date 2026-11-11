use llama_behavior::json::{self, Value};
use llama_model_quant::*;
use llama_quant::DType;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
fn field<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or_else(|| panic!("missing field {k}"))
}
fn number(v: &Value) -> i64 {
    v.as_i64().unwrap()
}
fn text(v: &Value) -> &str {
    v.as_str().unwrap()
}
fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}
fn profile() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows-msvc"
    } else {
        "linux-gnu"
    }
}
fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(profile())
        .join(name);
    json::parse(&fs::read_to_string(path).unwrap()).unwrap()
}
fn params(v: &Value) -> Result<QuantizeParams> {
    let mut p = QuantizeParams::default();
    if let Some(n) = v.get("ftype") {
        p.ftype = FileType::from_id(number(n) as u32)?
    }
    if let Some(n) = v.get("nthread") {
        p.nthread = number(n) as i32
    }
    if let Some(n) = v.get("max_buf_size") {
        p.max_buf_size = number(n) as usize
    }
    if let Some(n) = v.get("output_tensor_type") {
        p.output_tensor_type = Some(DType::from_id(number(n) as u32)?)
    }
    if let Some(n) = v.get("token_embedding_type") {
        p.token_embedding_type = Some(DType::from_id(number(n) as u32)?)
    }
    macro_rules! flag{($($f:ident),*)=>{$(if let Some(x)=v.get(stringify!($f)){p.$f=x.as_bool().unwrap();})*}}
    flag!(
        allow_requantize,
        quantize_output_tensor,
        only_copy,
        pure,
        keep_split,
        dry_run
    );
    if let Some(n) = v.get("prune_layers") {
        p.prune_layers = array(n).iter().map(|x| number(x) as i32).collect()
    }
    if let Some(n) = v.get("tt_overrides") {
        p.tt_overrides = array(n)
            .iter()
            .map(|x| TensorTypeOverride {
                pattern: text(field(x, "pattern")).as_bytes().to_vec(),
                dtype: DType::from_id(number(field(x, "type")) as u32).unwrap(),
            })
            .collect()
    }
    if let Some(n) = v.get("imatrix") {
        let mut im = BTreeMap::new();
        for x in array(n) {
            im.entry(text(field(x, "name")).to_owned())
                .or_insert_with(|| {
                    array(field(x, "data"))
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect()
                });
        }
        p.imatrix = Some(im);
    }
    if let Some(n) = v.get("kv_overrides") {
        p.kv_overrides = array(n)
            .iter()
            .map(|x| MetadataOverride {
                key: text(field(x, "key")).as_bytes().to_vec(),
                value: match text(field(x, "type")) {
                    "int" => OverrideValue::Integer(number(field(x, "value"))),
                    "float" => OverrideValue::Float(field(x, "value").as_f64().unwrap()),
                    "bool" => OverrideValue::Bool(field(x, "value").as_bool().unwrap()),
                    _ => OverrideValue::String(text(field(x, "value")).as_bytes().to_vec()),
                },
            })
            .collect();
    }
    Ok(p)
}
fn model(v: &Value) -> ModelDescriptor {
    let mut m = ModelDescriptor {
        architecture: text(field(v, "architecture")).into(),
        ..ModelDescriptor::default()
    };
    macro_rules! item{($($f:ident),*)=>{$(if let Some(x)=v.get(stringify!($f)){m.$f=number(x) as u32;})*}}
    item!(
        n_embd,
        n_ff,
        n_layer,
        n_head,
        n_head_kv,
        n_expert,
        n_embd_head_k,
        n_embd_head_v
    );
    m
}
#[test]
fn independent_original_type_assignments() {
    let data = fixture("types.json");
    let mut comparisons = 0;
    let mut failures = Vec::new();
    for case in array(&data) {
        let request = field(case, "request");
        if text(field(request, "mode")) == "defaults" {
            continue;
        }
        let id = text(field(case, "id"));
        let p = params(field(request, "params")).unwrap();
        let m = model(field(request, "model"));
        let specs: Vec<_> = array(field(request, "tensors"))
            .iter()
            .map(|t| TensorSpec {
                name: text(field(t, "name")).into(),
                dimensions: array(field(t, "dimensions"))
                    .iter()
                    .map(|n| number(n) as u64)
                    .collect(),
                dtype: DType::from_id(t.get("dtype").map_or(0, |n| number(n) as u32)).unwrap(),
            })
            .collect();
        // The exported original helper receives only eligible tensors. The real
        // file driver counts all tensors; the separate complete-file suite checks it.
        let selected: Vec<_> = specs
            .iter()
            .filter(|t| allows_quantization(&p, &m.architecture, t))
            .cloned()
            .collect();
        let result = assign_types(&m, &p, &selected, false);
        if text(field(case, "status")) == "error" {
            if result.is_ok() {
                failures.push(format!("{id}: original rejected but Rust accepted"))
            }
            continue;
        }
        let Ok(types) = result else {
            failures.push(format!("{id}: {result:?}"));
            continue;
        };
        let expected = array(field(field(case, "result"), "tensors"));
        let mut index = 0;
        for (spec, want) in specs.iter().zip(expected) {
            let allowed = allows_quantization(&p, &m.architecture, spec);
            let got = if allowed {
                let t = types[index];
                index += 1;
                t
            } else {
                spec.dtype
            };
            comparisons += 1;
            if allowed != field(want, "allowed").as_bool().unwrap()
                || got.id() != number(field(want, "type")) as u32
            {
                failures.push(format!(
                    "{id}/{}: allowed={allowed}, type={got:?}; native={want:?}",
                    spec.name
                ))
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("compared {comparisons} independent tensor assignments");
}
static TEMP: AtomicUsize = AtomicUsize::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "llama-model-quant-{}-{}",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn independent_original_complete_files() {
    let cases = fixture("files.json");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inputs");
    let mut comparisons = 0;
    let mut failures = Vec::new();
    for c in array(&cases) {
        let id = text(field(c, "id"));
        let dir = Directory::new();
        let input = dir.0.join("input.gguf");
        let output = dir.0.join("output.gguf");
        let input = if let Some(splits) = c.get("split_inputs") {
            for s in array(splits) {
                fs::copy(
                    root.join(text(field(s, "file"))),
                    dir.0.join(text(field(s, "name"))),
                )
                .unwrap();
            }
            dir.0.join(text(field(&array(splits)[0], "name")))
        } else {
            fs::copy(root.join(text(field(c, "input"))), &input).unwrap();
            input
        };
        let p = params(field(c, "params"));
        let expected_success = c.get("return_code").is_some_and(|v| v.as_i64() == Some(0))
            && text(field(c, "reference_status")) != "source-abort";
        let result = p.and_then(|p| quantize_file(&input, &output, &p));
        if !expected_success {
            if result.is_ok() {
                failures.push(format!("{id}: expected checked source rejection"))
            }
            continue;
        }
        let report = match result {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{id}: {e}"));
                continue;
            }
        };
        let expected = array(field(c, "output"));
        if expected.is_empty() {
            if report.written || output.exists() {
                failures.push(format!("{id}: dry-run wrote a file"))
            }
            continue;
        }
        if report.outputs.len() != expected.len() {
            failures.push(format!(
                "{id}: shard count {} != {}",
                report.outputs.len(),
                expected.len()
            ));
            continue;
        }
        for (path, want) in report.outputs.iter().zip(expected) {
            let bytes = fs::read(path).unwrap();
            let sha = llama_conformance::digest::sha256(&bytes);
            comparisons += 1;
            if bytes.len() as i64 != number(field(want, "bytes"))
                || sha != text(field(want, "sha256"))
            {
                failures.push(format!(
                    "{id}: output {} bytes {sha}; native {} bytes {}",
                    bytes.len(),
                    number(field(want, "bytes")),
                    text(field(want, "sha256"))
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("compared {comparisons} complete native GGUF outputs");
}
#[test]
fn failed_conversion_preserves_destination_and_dry_run_does_not_create_it() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inputs");
    let cases = fixture("files.json");
    let case = array(&cases)
        .iter()
        .find(|c| text(field(c, "id")) == "quant-15")
        .unwrap();
    let dir = Directory::new();
    let input = root.join(text(field(case, "input")));
    let output = dir.0.join("preserved.gguf");
    fs::write(&output, b"existing destination").unwrap();
    let p = QuantizeParams {
        ftype: FileType::from_id(19).unwrap(),
        ..QuantizeParams::default()
    };
    assert!(quantize_file(&input, &output, &p).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"existing destination");
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
    let p = QuantizeParams { dry_run: true, ..p };
    let result = quantize_file(&input, dir.0.join("absent.gguf"), &p).unwrap();
    assert!(!result.written);
    assert!(result.requires_imatrix);
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
}

#[test]
fn bounded_worker_slabs_preserve_the_independent_complete_file() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inputs");
    let cases = fixture("files.json");
    let case = array(&cases)
        .iter()
        .find(|c| text(field(c, "id")) == "quant-15")
        .unwrap();
    let dir = Directory::new();
    let input = root.join(text(field(case, "input")));
    let output = dir.0.join("bounded.gguf");
    let mut p = params(field(case, "params")).unwrap();
    p.nthread = 4;
    p.max_buf_size = usize::MAX;
    p.dry_run = true;
    let plan = quantize_file(&input, &output, &p).unwrap();
    // The minimum complete row consists of source bytes, an F32 decoding row,
    // and both the final packed row and one worker's returned packed row.
    // Copied tensors need only their source row plus the F32 validation row.
    let budget = plan
        .tensors
        .iter()
        .map(|t| {
            let width = t.dimensions[0] as usize;
            let source = t.source_type.row_bytes(width).unwrap();
            let temporary = width * 4;
            if t.source_type == t.target_type {
                source + temporary
            } else {
                source + temporary + 2 * t.target_type.row_bytes(width).unwrap()
            }
        })
        .max()
        .unwrap();
    let limits = llama_model_quant::QuantizeLimits {
        max_work_bytes: budget,
        ..Default::default()
    };
    p.dry_run = false;
    let result =
        llama_model_quant::quantize_file_with_limits(&input, &output, &p, &limits).unwrap();
    assert!(result.written);
    let want = &array(field(case, "output"))[0];
    assert_eq!(
        llama_conformance::digest::sha256(&fs::read(&output).unwrap()),
        text(field(want, "sha256"))
    );
    fs::write(&output, b"retained destination").unwrap();
    let limits = llama_model_quant::QuantizeLimits {
        max_work_bytes: budget - 1,
        ..limits
    };
    assert!(llama_model_quant::quantize_file_with_limits(&input, &output, &p, &limits).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"retained destination");
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
}
