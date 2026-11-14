//! Device admission and failure transactions. The normal graph oracle was
//! captured independently from the original GGML graph before CUDA existed.
use llama_cuda_runtime::{CudaLlama, Error, Options};
use llama_model_runtime::{
    BatchToken, Context, ContextOptions, LlamaConfig, LlamaLayer, LlamaModel, LlamaWeights, Matrix,
    Model, fixtures,
};
use llama_quant::DType;
use std::sync::Arc;

fn oracle(name: &str) -> Vec<f32> {
    include_str!("../../model-runtime/tests/data/ggml_reference.tsv")
        .lines()
        .find_map(|line| {
            let mut fields = line.split('\t');
            (fields.next() == Some(name)).then(|| fields.map(|s| s.parse().unwrap()).collect())
        })
        .unwrap_or_else(|| panic!("missing independent GGML row {name}"))
}
fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-5 + 2e-5 * e.abs(),
            "independent GGML value {index}: {a} vs {e}"
        );
    }
}
#[test]
#[ignore = "requires an actual NVIDIA CUDA device; absence is a failure"]
fn cuda_biased_gqa_graph_matches_original_ggml_and_exact_budget() {
    let model = fixtures::tiny_llama().unwrap();
    assert_ne!(model.config().heads, model.config().kv_heads);
    assert!(model.weights().layers.iter().all(|l| {
        [
            &l.query_bias,
            &l.key_bias,
            &l.value_bias,
            &l.attention_output_bias,
            &l.gate_bias,
            &l.up_bias,
            &l.down_bias,
        ]
        .iter()
        .all(|b| b.is_some())
    }));
    let initial = CudaLlama::new(&model, &Options::default()).unwrap();
    let required = initial.statistics().allocated_bytes;
    drop(initial);
    let error = CudaLlama::new(
        &model,
        &Options {
            max_device_bytes: required - 1,
            ..Options::default()
        },
    )
    .err()
    .expect("one byte below the complete plan must be rejected");
    assert!(matches!(error, Error::Invalid(ref s) if s.contains("budget exceeded")));
    let mut gpu = CudaLlama::new(
        &model,
        &Options {
            max_device_bytes: required,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(gpu.statistics().allocated_bytes, required);
    let mut maximum = 0.0f32;
    for (position, token) in [1, 2, 3, 4].into_iter().enumerate() {
        let actual = gpu.decode(token).unwrap();
        for (name, values) in [("embedding", actual.embedding), ("logits", actual.logits)] {
            let expected = oracle(&format!("llama.{name}.{position}"));
            compare(&values, &expected);
            for (&a, &e) in values.iter().zip(&expected) {
                maximum = maximum.max((a - e).abs());
            }
        }
    }
    assert_eq!(gpu.position(), 4);
    eprintln!(
        "CUDA biased GQA: 44 independent GGML values, max_abs={maximum:e}; exact owned-byte budget={required}, one-byte-below rejected"
    );
}

// Both models contain only finite, structurally valid weights. Token zero is
// harmless, while token one reaches a numerical failure. The second model's
// zero quantized output projection used to erase NaN attention activations.
fn adversarial_model(quantized_attention_output: bool) -> LlamaModel {
    let width = 32;
    let kv = 16;
    let config = LlamaConfig {
        vocabulary: 2,
        width,
        feed_forward: width,
        heads: 2,
        kv_heads: 1,
        max_context: 4,
        rms_epsilon: 1e-5,
        rope_base: 10_000.,
        attention_scale: None,
    };
    let mut embedding = vec![0.; 2 * width];
    embedding[width..].fill(if quantized_attention_output {
        1.
    } else {
        f32::MAX
    });
    let attention_output = if quantized_attention_output {
        Matrix::from_quantized(
            width,
            width,
            DType::Q4_0,
            llama_quant::quantize(DType::Q4_0, &vec![0.; width * width]).unwrap(),
        )
        .unwrap()
    } else {
        Matrix::zeros(width, width).unwrap()
    };
    let layer = LlamaLayer {
        attention_norm: vec![1.; width],
        query: if quantized_attention_output {
            Matrix::new(width, width, vec![f32::MAX; width * width]).unwrap()
        } else {
            Matrix::zeros(width, width).unwrap()
        },
        key: Matrix::zeros(kv, width).unwrap(),
        value: Matrix::zeros(kv, width).unwrap(),
        attention_output,
        query_bias: None,
        key_bias: None,
        value_bias: None,
        attention_output_bias: None,
        feed_forward_norm: vec![1.; width],
        gate: Matrix::zeros(width, width).unwrap(),
        up: Matrix::zeros(width, width).unwrap(),
        down: Matrix::zeros(width, width).unwrap(),
        gate_bias: None,
        up_bias: None,
        down_bias: None,
    };
    LlamaModel::new(
        config,
        LlamaWeights {
            token_embedding: Matrix::new(2, width, embedding).unwrap(),
            layers: vec![layer],
            output_norm: vec![1.; width],
            output: Some(Matrix::zeros(2, width).unwrap()),
        },
    )
    .unwrap()
}
#[test]
#[ignore = "requires an actual NVIDIA CUDA device; absence is a failure"]
fn cuda_numeric_failures_cannot_commit_or_poison_a_reusable_prefix() {
    for quantized in [true, false] {
        let model = adversarial_model(quantized);
        let mut cpu = Context::new(
            Arc::new(Model::Llama(model.clone())),
            ContextOptions::default(),
        )
        .unwrap();
        cpu.decode_batch(&[BatchToken::token(0, 0, 0)]).unwrap();
        let cpu_before = cpu.save_state().unwrap();
        assert!(cpu.decode_batch(&[BatchToken::token(0, 1, 1)]).is_err());
        assert_eq!(cpu.save_state().unwrap(), cpu_before);
        let mut gpu = CudaLlama::new(&model, &Options::default()).unwrap();
        gpu.decode(0).unwrap();
        let before = gpu.position();
        let committed = gpu.statistics().committed_tokens;
        let result = gpu.decode(1);
        eprintln!(
            "numeric failure quantized={quantized}: CPU rejected, GPU rejected={}, position_before={before}, position_after={}",
            result.is_err(),
            gpu.position()
        );
        assert!(
            matches!(result, Err(Error::Invalid(ref s)) if s.contains("nonfinite")),
            "finite invalid graph must reject, quantized={quantized}"
        );
        assert_eq!(gpu.position(), before);
        assert_eq!(gpu.statistics().committed_tokens, committed);
        let actual = gpu.decode(0).unwrap();
        let mut clean = CudaLlama::new(&model, &Options::default()).unwrap();
        clean.decode(0).unwrap();
        let expected = clean.decode(0).unwrap();
        assert_eq!(actual.logits, expected.logits);
        assert_eq!(actual.embedding, expected.embedding);
        assert_eq!(gpu.position(), 2);
    }
}
