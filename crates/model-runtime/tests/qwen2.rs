use llama_model_runtime::*;
use std::sync::Arc;

/// A one-layer model whose recurrent transformations are all zero. The hidden
/// state therefore stays at the selected +/-1 embedding; its output projection
/// and bias are individually finite but the positive branch overflows on add.
fn output_bias_model() -> Arc<Model> {
    let config = Qwen2Config {
        vocabulary: 2,
        width: 2,
        feed_forward: 2,
        heads: 1,
        kv_heads: 1,
        max_context: 8,
        rms_epsilon: 1e-5,
        rope_base: 10000.0,
    };
    let zero = || Matrix::zeros(2, 2).unwrap();
    let layer = Qwen2Layer {
        attention_norm: vec![1.0; 2],
        qkv: Qwen2Qkv::Separate {
            query: zero(),
            key: zero(),
            value: zero(),
            query_bias: None,
            key_bias: None,
            value_bias: None,
        },
        attention_output: zero(),
        feed_forward_norm: vec![1.0; 2],
        gate: zero(),
        up: zero(),
        down: zero(),
    };
    Arc::new(Model::Qwen2(
        Qwen2Model::new(
            config,
            Qwen2Weights {
                token_embedding: Matrix::new(2, 2, vec![-1.0, -1.0, 1.0, 1.0]).unwrap(),
                layers: vec![layer],
                output_norm: vec![1.0; 2],
                output: Some(Matrix::new(2, 2, vec![f32::MAX / 2.0, 0.0, 0.0, 0.0]).unwrap()),
                output_bias: Some(vec![f32::MAX, 0.0]),
            },
        )
        .unwrap(),
    ))
}

#[test]
fn finite_qwen2_projection_and_bias_overflow_cannot_commit_state() {
    let mut context = Context::new(output_bias_model(), ContextOptions::default()).unwrap();
    let initial = context.decode_batch(&[BatchToken::token(0, 0, 0)]).unwrap();
    assert!(
        initial[0]
            .logits
            .as_ref()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
    let before = context.save_state().unwrap();
    let result = context.decode_batch(&[BatchToken::token(0, 1, 1)]);
    assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
    assert_eq!(context.save_state().unwrap(), before);
    assert_eq!(context.next_position(0), Some(1));
}

#[test]
fn qwen2_late_output_overflow_rolls_back_other_sequences() {
    let mut context = Context::new(output_bias_model(), ContextOptions::default()).unwrap();
    let before = context.save_state().unwrap();
    let result = context.decode_batch(&[
        BatchToken::token(0, 0, 0),
        BatchToken::token(1, 0, 0),
        BatchToken::token(0, 1, 1),
    ]);
    assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
    assert_eq!(context.save_state().unwrap(), before);
    assert_eq!(context.sequence_count(), 0);
}
