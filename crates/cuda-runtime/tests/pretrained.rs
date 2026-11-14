//! Real GPU decoding against frozen original C-API pretrained logits.
//! These corpora predate the CUDA implementation; no expected values come from
//! the Rust GPU graph or its instruction interpreter.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::hash_reader;
use llama_cuda_runtime::{CudaLlama, Options};
use llama_inference::{LoadOptions, Model};
use llama_model_runtime::Model as RuntimeModel;
use std::{cell::Cell, fs::File};
fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}
fn compare(
    actual: &[f32],
    expected: impl ExactSizeIterator<Item = f64>,
    absolute: f64,
    relative: f64,
    max: &mut f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - e).abs();
        *max = max.max(error);
        assert!(
            a.is_finite() && error <= absolute + relative * e.abs(),
            "logit {i}: {a} vs {e}, delta {error}, bound {}",
            absolute + relative * e.abs()
        );
    }
}
fn generation(model: &Model, case: &Value, gpu: &mut CudaLlama) {
    let hex = case.get("generated_hex").unwrap().as_str().unwrap();
    let expected: Vec<u8> = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect();
    let options = llama_inference::GenerationOptions {
        max_tokens: case.get("greedy_tokens").unwrap().as_array().unwrap().len(),
        ..llama_inference::GenerationOptions::default()
    };
    let prompt = case.get("prompt").unwrap().as_str().unwrap();
    gpu.reset();
    let mut text = String::new();
    let stats = model
        .generate_with_session(
            prompt,
            &options,
            &|| false,
            &mut |part| {
                text.push_str(part);
                true
            },
            gpu,
        )
        .unwrap();
    assert_eq!(text.as_bytes(), expected);
    assert_eq!(stats.completion_tokens, options.max_tokens);
    assert_eq!(
        stats.prompt_tokens,
        case.get("tokens").unwrap().as_array().unwrap().len()
    );
    assert_eq!(
        stats.finish_reason,
        if case.get("finished_eog").unwrap().as_bool().unwrap() {
            llama_inference::FinishReason::Eos
        } else {
            llama_inference::FinishReason::Length
        }
    );
    assert!(
        model
            .generate_with_session(prompt, &options, &|| false, &mut |_| true, gpu)
            .is_err()
    );
    if !text.is_empty() {
        let stop: String = text.chars().take(3).collect();
        gpu.reset();
        text.clear();
        let stopped = model
            .generate_with_session(
                prompt,
                &llama_inference::GenerationOptions {
                    stop: vec![stop],
                    ..options
                },
                &|| false,
                &mut |part| {
                    text.push_str(part);
                    true
                },
                gpu,
            )
            .unwrap();
        assert_eq!(
            stopped.finish_reason,
            llama_inference::FinishReason::StopSequence
        );
        assert!(
            text.is_empty(),
            "a matched stop prefix escaped the shared stream filter"
        );
    }
}

fn runtime(model: &Model) -> CudaLlama {
    let m = model.runtime_model().unwrap();
    let RuntimeModel::Llama(m) = m.as_ref() else {
        panic!("fixture is not Llama");
    };
    CudaLlama::new(m, &Options::default()).unwrap()
}
#[test]
#[ignore = "requires CUDA and hash-pinned LLAMA_TEST_MODEL stories260K.gguf"]
fn cuda_pretrained_f32_logits_tokens_and_transaction_boundaries() {
    let path = std::env::var_os("LLAMA_TEST_MODEL").expect("LLAMA_TEST_MODEL");
    assert_eq!(
        hash_reader(File::open(&path).unwrap()).unwrap(),
        (
            1_185_376,
            "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d".into()
        )
    );
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            context_length: 128,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let source = if cfg!(windows) {
        include_str!("../../../fixtures/inference/stories260k-reference.json")
    } else {
        include_str!("../../../fixtures/inference/stories260k-gnu-reference.json")
    };
    let reference = json::parse(source).unwrap();
    let mut gpu = runtime(&model);
    let (mut compared, mut greedy, mut maximum) = (0usize, 0usize, 0f64);
    for case in reference.get("cases").unwrap().as_array().unwrap() {
        gpu.reset();
        let prompt = ids(case.get("tokens").unwrap());
        let wanted = ids(case.get("greedy_tokens").unwrap());
        let mut sequence = prompt.clone();
        sequence.extend_from_slice(&wanted[..wanted.len() - 1]);
        let mut rows = Vec::new();
        for &id in &sequence {
            rows.push(gpu.decode(id).unwrap().logits);
        }
        for (row, values) in case
            .get("prompt_logits")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let e = values.as_array().unwrap();
            compare(
                &rows[row],
                e.iter().map(|v| v.as_f64().unwrap()),
                2e-4,
                2e-5,
                &mut maximum,
            );
            compared += e.len();
        }
        for (step, values) in case
            .get("greedy_logits")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let row = prompt.len() - 1 + step;
            let e = values.as_array().unwrap();
            compare(
                &rows[row],
                e.iter().map(|v| v.as_f64().unwrap()),
                2e-4,
                2e-5,
                &mut maximum,
            );
            assert_eq!(argmax(&rows[row]), wanted[step]);
            greedy += 1;
            compared += e.len();
        }
        generation(&model, case, &mut gpu);
    }
    assert_eq!(greedy, 96);
    // Repeat, truncate, and cancellation all retain the same committed prefix.
    gpu.reset();
    gpu.decode(1).unwrap();
    let before = gpu.position();
    assert!(gpu.decode(u32::MAX).is_err());
    assert_eq!(gpu.position(), before);
    let checks = Cell::new(0);
    assert!(
        gpu.decode_with_cancel(2, &|| {
            let n = checks.get();
            checks.set(n + 1);
            n >= 3
        })
        .is_err()
    );
    assert_eq!(gpu.position(), before);
    let next = gpu.decode(2).unwrap();
    gpu.truncate(before).unwrap();
    assert_eq!(next.logits, gpu.decode(2).unwrap().logits);
    let committed = gpu.position();
    assert!(matches!(
        gpu.decode_with_cancel(3, &|| true),
        Err(llama_cuda_runtime::Error::Cancelled)
    ));
    assert_eq!(gpu.position(), committed);
    let actual = gpu.decode(3).unwrap();
    let mut clean = runtime(&model);
    clean.decode(1).unwrap();
    clean.decode(2).unwrap();
    let expected = clean.decode(3).unwrap();
    assert_eq!(actual.logits, expected.logits);
    assert_eq!(actual.embedding, expected.embedding);
    assert!(gpu.truncate(gpu.position() + 1).is_err());
    eprintln!(
        "CUDA F32 pretrained: {compared} logits, {greedy} exact greedy IDs, max_abs={maximum:e}, device={}, stats={:?}",
        gpu.device_name(),
        gpu.statistics()
    );
}
#[test]
#[ignore = "requires CUDA and hash-pinned LLAMA_TEST_Q4_MODEL stories15M-q4_0.gguf"]
fn cuda_pretrained_q4_logits_and_greedy_tokens() {
    let path = std::env::var_os("LLAMA_TEST_Q4_MODEL").expect("LLAMA_TEST_Q4_MODEL");
    assert_eq!(
        hash_reader(File::open(&path).unwrap()).unwrap(),
        (
            19_077_344,
            "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739".into()
        )
    );
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            context_length: 64,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let reference = json::parse(include_str!(
        "../../../fixtures/inference/stories15m-q4-metadata.json"
    ))
    .unwrap();
    let expected = include_bytes!("../../../fixtures/inference/stories15m-q4-logits.f32le.bin");
    let mut gpu = runtime(&model);
    let (mut maximum, mut compared, mut greedy) = (0f64, 0usize, 0usize);
    for case in reference.get("cases").unwrap().as_array().unwrap() {
        let prompt = ids(case.get("tokens").unwrap());
        let wanted = ids(case.get("greedy_tokens").unwrap());
        let mut sequence = prompt.clone();
        sequence.extend_from_slice(&wanted[..wanted.len() - 1]);
        for (row, token) in sequence.into_iter().enumerate() {
            let actual = gpu.decode(token).unwrap();
            let bytes = &expected[row * 32_000 * 4..(row + 1) * 32_000 * 4];
            // Device Q8 activation conversion and integer block sums preserve
            // the frozen scalar CPU arithmetic. Reuse its original unchanged
            // numerical bound rather than accepting the earlier dense-dot drift.
            compare(
                &actual.logits,
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f64::from(f32::from_le_bytes(*v))),
                2e-4,
                2e-5,
                &mut maximum,
            );
            compared += actual.logits.len();
            if row >= prompt.len() - 1 {
                assert_eq!(argmax(&actual.logits), wanted[row - (prompt.len() - 1)]);
                greedy += 1;
            }
        }
        generation(&model, case, &mut gpu);
    }
    assert_eq!(compared, 640_000);
    assert_eq!(greedy, 16);
    eprintln!(
        "CUDA Q4 pretrained: {compared} logits, {greedy} exact greedy IDs, max_abs={maximum:e}, device={}, stats={:?}",
        gpu.device_name(),
        gpu.statistics()
    );
}

#[test]
#[ignore = "requires CUDA and hash-pinned LLAMA_TEST_F16_MODEL native F16 stories260K"]
fn cuda_pretrained_f16_logits_tokens_and_generation() {
    let path = std::env::var_os("LLAMA_TEST_F16_MODEL").expect("LLAMA_TEST_F16_MODEL");
    assert_eq!(
        hash_reader(File::open(&path).unwrap()).unwrap(),
        (
            601_248,
            "4cd29bbe9cb3b735750837816a43c3dd4b473dbd658e8612816f7c4d3926ac21".into()
        )
    );
    let model = Model::load_with_options(
        path,
        &LoadOptions {
            context_length: 128,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let m = model.runtime_model().unwrap();
    let RuntimeModel::Llama(m) = m.as_ref() else {
        panic!("fixture is not Llama");
    };
    assert_eq!(
        m.weights().layers[0].query.storage_dtype(),
        llama_quant::DType::F16
    );
    let reference = json::parse(include_str!(
        "../../../fixtures/inference/cuda/stories260k-f16-msvc-metadata.json"
    ))
    .unwrap();
    let expected =
        include_bytes!("../../../fixtures/inference/cuda/stories260k-f16-msvc-logits.f32le.bin");
    let mut gpu = runtime(&model);
    let (mut maximum, mut compared, mut greedy) = (0f64, 0usize, 0usize);
    for case in reference.get("cases").unwrap().as_array().unwrap() {
        gpu.reset();
        let prompt = ids(case.get("tokens").unwrap());
        let wanted = ids(case.get("greedy_tokens").unwrap());
        let mut sequence = prompt.clone();
        sequence.extend_from_slice(&wanted[..wanted.len() - 1]);
        let first = case.get("first_row").unwrap().as_u64().unwrap() as usize;
        assert_eq!(
            sequence.len(),
            case.get("row_count").unwrap().as_u64().unwrap() as usize
        );
        for (row, token) in sequence.into_iter().enumerate() {
            let actual = gpu.decode(token).unwrap();
            let bytes = &expected[(first + row) * 512 * 4..(first + row + 1) * 512 * 4];
            compare(
                &actual.logits,
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f64::from(f32::from_le_bytes(*v))),
                2e-4,
                2e-5,
                &mut maximum,
            );
            compared += actual.logits.len();
            if row >= prompt.len() - 1 {
                assert_eq!(argmax(&actual.logits), wanted[row - (prompt.len() - 1)]);
                greedy += 1;
            }
        }
        generation(&model, case, &mut gpu);
    }
    assert_eq!(compared, 59_904);
    assert_eq!(greedy, 96);
    eprintln!(
        "CUDA F16 pretrained: {compared} logits, {greedy} exact greedy IDs, max_abs={maximum:e}, device={}, stats={:?}",
        gpu.device_name(),
        gpu.statistics()
    );
}
