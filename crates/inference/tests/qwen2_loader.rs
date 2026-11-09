//! Physical GGUF metadata and tensor boundaries for the Qwen2 graph loader.
#[path = "support/native_profile.rs"]
mod native_profile;
use llama_behavior::json;
use llama_conformance::digest::sha256;
use llama_gguf::{Gguf, Limits, TensorData, TensorInfo, Value, write};
use llama_inference::{Error, LoadOptions, Model};
use native_profile::source_profile;
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Cursor,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const TINY: &[u8] = include_bytes!("../../../fixtures/inference/tiny-qwen2-separate.gguf");
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct TemporaryModel(PathBuf);
impl TemporaryModel {
    fn changed(
        edit: impl FnOnce(&mut BTreeMap<String, Value>, &mut Vec<TensorInfo>, &mut Vec<Vec<u8>>),
    ) -> Self {
        let mut gguf = Gguf::read(Cursor::new(TINY), Limits::default()).unwrap();
        let mut infos = gguf.tensors.clone();
        let mut payloads: Vec<_> = infos
            .iter()
            .map(|info| gguf.read_tensor(&info.name).unwrap())
            .collect();
        edit(&mut gguf.metadata, &mut infos, &mut payloads);
        assert_eq!(infos.len(), payloads.len());
        let tensors: Vec<_> = infos
            .iter()
            .zip(&payloads)
            .map(|(info, data)| TensorData {
                name: &info.name,
                dimensions: &info.dimensions,
                dtype: info.dtype,
                data,
            })
            .collect();
        let path = std::env::temp_dir().join(format!(
            "rust-llama-qwen2-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        write(file, &gguf.metadata, &tensors).unwrap();
        Self(path)
    }
}
impl Drop for TemporaryModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn rejected_metadata(key: &str, value: Option<Value>) -> Error {
    let temporary = TemporaryModel::changed(|metadata, _, _| {
        if let Some(value) = value {
            metadata.insert(key.into(), value);
        } else {
            assert!(metadata.remove(key).is_some());
        }
    });
    Model::load(&temporary.0).unwrap_err()
}

#[test]
fn qwen2_source_typed_dimensions_and_limits_reject_malformed_files() {
    for key in [
        "qwen2.block_count",
        "qwen2.context_length",
        "qwen2.embedding_length",
        "qwen2.feed_forward_length",
        "qwen2.attention.head_count",
    ] {
        assert!(
            matches!(rejected_metadata(key, None), Error::Invalid(_)),
            "missing {key}"
        );
        assert!(
            matches!(
                rejected_metadata(key, Some(Value::U32(0))),
                Error::Invalid(_)
            ),
            "zero {key}"
        );
        for value in [Value::U8(4), Value::U16(4), Value::U64(4), Value::I32(4)] {
            assert!(
                matches!(rejected_metadata(key, Some(value)), Error::Invalid(_)),
                "typed {key}"
            );
        }
    }
    for (key, value) in [
        ("qwen2.block_count", Value::U32(513)),
        ("qwen2.embedding_length", Value::U32(7)),
        ("qwen2.attention.head_count", Value::U32(3)),
        ("qwen2.attention.head_count_kv", Value::U32(0)),
        ("qwen2.attention.head_count_kv", Value::U32(3)),
        ("qwen2.attention.layer_norm_rms_epsilon", Value::F32(0.0)),
        (
            "qwen2.attention.layer_norm_rms_epsilon",
            Value::F32(f32::NAN),
        ),
        ("qwen2.attention.layer_norm_rms_epsilon", Value::F64(1e-5)),
        ("qwen2.rope.freq_base", Value::F32(0.0)),
        ("qwen2.attention.causal", Value::U32(1)),
        ("qwen2.rope.scaling.type", Value::U32(0)),
        ("qwen2.rope.scaling.factor", Value::F64(1.0)),
        ("split.count", Value::U32(1)),
        ("general.type", Value::I32(0)),
    ] {
        assert!(
            matches!(rejected_metadata(key, Some(value)), Error::Invalid(_)),
            "{key}"
        );
    }
    assert!(matches!(
        rejected_metadata("qwen2.attention.layer_norm_rms_epsilon", None),
        Error::Invalid(_)
    ));
}

#[test]
fn qwen2_unimplemented_model_modes_are_explicit() {
    for (key, value) in [
        ("qwen2.attention.causal", Value::Bool(false)),
        ("qwen2.rope.scaling.type", Value::String("yarn".into())),
        ("qwen2.rope.scaling.factor", Value::F32(2.0)),
        ("qwen2.rope.scale_linear", Value::F32(2.0)),
        ("qwen2.rope.scaling.attn_factor", Value::F32(0.5)),
        ("qwen2.rope.dimension_count", Value::U32(2)),
        ("qwen2.attention.key_length", Value::U32(2)),
        ("qwen2.attention.value_length", Value::U32(2)),
        ("qwen2.attention.sliding_window", Value::U32(1)),
        ("qwen2.expert_count", Value::U32(1)),
        ("qwen2.nextn_predict_layers", Value::U32(1)),
        ("prism.hadamard.version", Value::U32(1)),
        ("general.type", Value::String("adapter".into())),
        ("split.count", Value::U16(2)),
    ] {
        assert!(
            matches!(rejected_metadata(key, Some(value)), Error::Unsupported(_)),
            "{key}"
        );
    }
}

#[test]
fn source_rope_defaults_factor_precedence_and_single_file_counts_are_preserved() {
    let metadata = source_profile(
        include_str!("../../../fixtures/inference/qwen2-metadata-native-provenance.json"),
        include_str!("../../../fixtures/inference/qwen2-metadata-gnu-native-provenance.json"),
    );
    assert_eq!(
        sha256(metadata.as_bytes()),
        source_profile(
            "038c122103f0c8a8f6ea86b71908312e76d2731f0e1b13b5932bbbdae296f4f5",
            "b1c491fd6374dd955485baab921a71487fd8dca244d6436034b0ec2121e72bcf"
        )
    );
    let native = json::parse(metadata).unwrap();
    let records = native.get("cases").unwrap().as_array().unwrap();
    assert_eq!(records.len(), 13);

    let baseline = TemporaryModel::changed(|_, _, _| {});
    let expected = Model::load(&baseline.0)
        .unwrap()
        .logits_for_tokens(&[1, 2, 3])
        .unwrap();
    let variants: Vec<(&str, Vec<(&str, Value)>)> = vec![
        ("split-zero", vec![("split.count", Value::U16(0))]),
        ("split-one", vec![("split.count", Value::U16(1))]),
        (
            "causal-true",
            vec![("qwen2.attention.causal", Value::Bool(true))],
        ),
        (
            "linear-default",
            vec![("qwen2.rope.scaling.type", Value::String("linear".into()))],
        ),
        (
            "new-zero",
            vec![("qwen2.rope.scaling.factor", Value::F32(0.0))],
        ),
        (
            "old-zero",
            vec![("qwen2.rope.scale_linear", Value::F32(0.0))],
        ),
        (
            "new-one",
            vec![("qwen2.rope.scaling.factor", Value::F32(1.0))],
        ),
        (
            "old-one",
            vec![("qwen2.rope.scale_linear", Value::F32(1.0))],
        ),
        (
            "new-one-before-old-eight",
            vec![
                ("qwen2.rope.scaling.factor", Value::F32(1.0)),
                ("qwen2.rope.scale_linear", Value::F32(8.0)),
            ],
        ),
        (
            "new-zero-before-old-eight",
            vec![
                ("qwen2.rope.scaling.factor", Value::F32(0.0)),
                ("qwen2.rope.scale_linear", Value::F32(8.0)),
            ],
        ),
        (
            "none-ignores-new-two",
            vec![
                ("qwen2.rope.scaling.type", Value::String("none".into())),
                ("qwen2.rope.scaling.factor", Value::F32(2.0)),
            ],
        ),
        (
            "none-ignores-old-two",
            vec![
                ("qwen2.rope.scaling.type", Value::String("none".into())),
                ("qwen2.rope.scale_linear", Value::F32(2.0)),
            ],
        ),
        (
            "attention-gain-one",
            vec![("qwen2.rope.scaling.attn_factor", Value::F32(1.0))],
        ),
    ];
    for (label, variant) in variants {
        let temporary = TemporaryModel::changed(|metadata, _, _| {
            for (key, value) in variant {
                metadata.insert(key.into(), value);
            }
        });
        let filename = format!("tiny-qwen2-metadata-{label}.gguf");
        let captured = records
            .iter()
            .find(|case| case.get("model").unwrap().as_str() == Some(filename.as_str()))
            .unwrap();
        assert_eq!(
            sha256(&std::fs::read(&temporary.0).unwrap()),
            captured.get("sha256").unwrap().as_str().unwrap()
        );
        assert_eq!(
            captured
                .get("native_cases_equal_to_baseline")
                .unwrap()
                .as_bool(),
            Some(true)
        );
        let model = Model::load(&temporary.0).unwrap();
        assert_eq!(model.logits_for_tokens(&[1, 2, 3]).unwrap(), expected);
    }
}

#[test]
fn missing_mis_shaped_or_unimplemented_qwen2_tensors_fail_explicitly() {
    let original = Gguf::read(Cursor::new(TINY), Limits::default()).unwrap();
    for tensor in original
        .tensors
        .iter()
        .filter(|tensor| tensor.name.ends_with(".weight"))
    {
        let temporary = TemporaryModel::changed(|_, infos, payloads| {
            let index = infos
                .iter()
                .position(|info| info.name == tensor.name)
                .unwrap();
            infos.remove(index);
            payloads.remove(index);
        });
        assert!(
            Model::load(&temporary.0).is_err(),
            "missing {}",
            tensor.name
        );
    }
    let wrong_shape = TemporaryModel::changed(|_, infos, _| {
        let tensor = infos
            .iter_mut()
            .find(|info| info.name == "token_embd.weight")
            .unwrap();
        tensor.dimensions.reverse();
    });
    assert!(matches!(
        Model::load(&wrong_shape.0),
        Err(Error::Invalid(_))
    ));
    let unconsumed = TemporaryModel::changed(|_, infos, payloads| {
        let mut extra = infos[0].clone();
        extra.name = "unused.weight".into();
        infos.push(extra);
        payloads.push(payloads[0].clone());
    });
    assert!(matches!(
        Model::load(&unconsumed.0),
        Err(Error::Unsupported(_))
    ));
    let temporary = TemporaryModel::changed(|_, _, _| {});
    assert!(
        Model::load_with_options(
            &temporary.0,
            &LoadOptions {
                context_length: 32,
                max_decoded_weight_bytes: 1
            }
        )
        .is_err()
    );
}
