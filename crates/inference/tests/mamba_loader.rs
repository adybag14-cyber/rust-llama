//! Loader failure and transactional state checks for the bundled Mamba models.
use llama_gguf::{Gguf, Limits, TensorData, TensorInfo, Value, write};
use llama_inference::{Error, GenerationOptions, LoadOptions, Model};
use llama_model_runtime::{BatchToken, Context, ContextOptions, Input, OutputSelection};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Cursor,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const TINY: &[u8] = include_bytes!("../../../fixtures/inference/tiny-mamba-default.gguf");
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
            "rust-llama-mamba-{}-{}.gguf",
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
fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/inference/tiny-mamba-default.gguf")
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
fn missing_invalid_and_overflowing_recurrent_metadata_fails_before_weights() {
    let dimensions = [
        "mamba.block_count",
        "mamba.context_length",
        "mamba.embedding_length",
        "mamba.ssm.inner_size",
        "mamba.ssm.state_size",
        "mamba.ssm.conv_kernel",
        "mamba.ssm.time_step_rank",
    ];
    for key in dimensions {
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
        assert!(
            matches!(
                rejected_metadata(key, Some(Value::String("4".into()))),
                Error::Invalid(_)
            ),
            "type {key}"
        );
    }
    for (key, value) in [
        ("mamba.embedding_length", Value::U64(u64::MAX)),
        ("mamba.ssm.state_size", Value::U64(u64::MAX)),
        ("mamba.ssm.conv_kernel", Value::U64(u64::MAX)),
        ("mamba.ssm.time_step_rank", Value::U64(u64::MAX)),
        ("mamba.ssm.inner_size", Value::U32(7)),
        ("mamba.block_count", Value::U32(513)),
        ("mamba.ssm.dt_b_c_rms", Value::U32(1)),
        ("general.type", Value::I32(0)),
        ("split.count", Value::U32(1)),
    ] {
        assert!(
            matches!(rejected_metadata(key, Some(value)), Error::Invalid(_)),
            "{key}"
        );
    }
    for value in [Value::U8(4), Value::U16(4), Value::U64(4), Value::I32(4)] {
        assert!(matches!(
            rejected_metadata("mamba.embedding_length", Some(value)),
            Error::Invalid(_)
        ));
    }
    assert!(matches!(
        rejected_metadata(
            "mamba.attention.layer_norm_rms_epsilon",
            Some(Value::F64(1e-5))
        ),
        Error::Invalid(_)
    ));
    for (key, value) in [
        ("mamba.nextn_predict_layers", Value::U32(1)),
        ("prism.hadamard.version", Value::U32(1)),
    ] {
        assert!(matches!(
            rejected_metadata(key, Some(value)),
            Error::Unsupported(_)
        ));
    }
    for count in [0, 1] {
        let temporary = TemporaryModel::changed(|metadata, _, _| {
            metadata.insert("split.count".into(), Value::U16(count));
        });
        assert_eq!(Model::load(&temporary.0).unwrap().vocabulary_size(), 512);
    }
    let epsilon = "mamba.attention.layer_norm_rms_epsilon";
    assert!(matches!(
        rejected_metadata(epsilon, None),
        Error::Invalid(_)
    ));
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            rejected_metadata(epsilon, Some(Value::F32(value))),
            Error::Invalid(_)
        ));
    }
    assert!(matches!(
        rejected_metadata("general.type", Some(Value::String("adapter".into()))),
        Error::Unsupported(_)
    ));
    // Sharded loading is implemented: this declaration still fails because
    // the primary file lacks the required split.no, before any weight reads.
    assert!(matches!(
        rejected_metadata("split.count", Some(Value::U16(2))),
        Error::Invalid(message) if message == "invalid GGUF: missing split.no"
    ));
}

#[test]
fn required_shapes_unused_tensors_and_weight_budget_are_enforced() {
    let fixture = Gguf::read(Cursor::new(TINY), Limits::default()).unwrap();
    assert_eq!(fixture.tensors.len(), 22);
    for info in &fixture.tensors {
        let temporary = TemporaryModel::changed(|_, infos, payloads| {
            let index = infos
                .iter()
                .position(|candidate| candidate.name == info.name)
                .unwrap();
            infos.remove(index);
            payloads.remove(index);
        });
        assert!(
            matches!(Model::load(&temporary.0), Err(Error::Invalid(_))),
            "missing {}",
            info.name
        );
    }
    let temporary = TemporaryModel::changed(|_, infos, _| {
        let input = infos
            .iter_mut()
            .find(|info| info.name == "blk.0.ssm_in.weight")
            .unwrap();
        input.dimensions = vec![8, 8]; // Same payload size, incompatible logical shape.
    });
    assert!(matches!(Model::load(&temporary.0), Err(Error::Invalid(_))));
    let temporary = TemporaryModel::changed(|_, infos, payloads| {
        let index = infos
            .iter()
            .position(|info| info.name == "blk.0.attn_norm.weight")
            .unwrap();
        let mut extra = infos[index].clone();
        extra.name = "blk.0.ssm_b_norm.weight".into();
        infos.push(extra);
        payloads.push(payloads[index].clone());
    });
    let error = Model::load(&temporary.0).unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)));
    assert!(error.to_string().contains("unimplemented tensors"));
    let model = Model::load(model_path()).unwrap();
    let bytes = model.decoded_weight_bytes();
    assert!(bytes > 1);
    let limited = LoadOptions {
        max_decoded_weight_bytes: bytes - 1,
        ..Default::default()
    };
    assert!(matches!(
        Model::load_with_options(model_path(), &limited),
        Err(Error::Invalid(_))
    ));
    let exact = LoadOptions {
        max_decoded_weight_bytes: bytes,
        context_length: 9,
    };
    assert_eq!(
        Model::load_with_options(model_path(), &exact)
            .unwrap()
            .context_length(),
        9
    );
    // Verify the same structural test data is still readable after failure.
    assert_eq!(
        File::open(model_path()).unwrap().metadata().unwrap().len(),
        TINY.len() as u64
    );
}

#[test]
fn cancellation_and_restore_keep_recurrent_state_transactional() {
    let model = Model::load(model_path()).unwrap();
    let runtime = model.runtime_model().unwrap();
    let mut context = Context::new(runtime, ContextOptions::default()).unwrap();
    let tokens = model.tokenize("hello", true).unwrap();
    let batch: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| BatchToken {
            sequence: 0,
            position,
            input: Input::Token(token as u32),
            output: OutputSelection::Logits,
        })
        .collect();
    let initial = context.save_state().unwrap();
    // A callback threshold crosses validation, token and layer boundaries.
    for threshold in 0..20 {
        let mut checks = 0;
        let outcome = context.decode_batch_with_cancel(&batch, || {
            checks += 1;
            checks > threshold
        });
        if matches!(outcome, Err(llama_model_runtime::Error::Cancelled)) {
            assert_eq!(
                context.save_state().unwrap(),
                initial,
                "threshold {threshold}"
            );
        } else {
            assert!(outcome.is_ok());
            context.restore_state(&initial).unwrap();
        }
    }
    let uninterrupted = context.decode_batch(&batch).unwrap();
    let completed = context.save_state().unwrap();
    context.restore_state(&initial).unwrap();
    let replayed = context.decode_batch(&batch).unwrap();
    assert_eq!(uninterrupted, replayed);
    assert_eq!(context.save_state().unwrap(), completed);

    let options = GenerationOptions {
        max_tokens: 4,
        ..Default::default()
    };
    let mut emitted = false;
    assert!(matches!(
        model.generate("hello", &options, &|| true, &mut |_| {
            emitted = true;
            true
        }),
        Err(Error::Cancelled)
    ));
    assert!(!emitted);
    let mut first = String::new();
    model
        .generate("hello", &options, &|| false, &mut |piece| {
            first.push_str(piece);
            true
        })
        .unwrap();
    let mut second = String::new();
    model
        .generate("hello", &options, &|| false, &mut |piece| {
            second.push_str(piece);
            true
        })
        .unwrap();
    assert_eq!(first, second);
}
