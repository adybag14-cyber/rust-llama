//! Strict BLOOM state and configuration boundaries. Independent numerical
//! expectations are in llama-inference/tests/bloom.rs.
use llama_model_runtime::*;
use std::sync::Arc;

#[test]
fn required_input_norm_head_shapes_and_overflow_are_checked() {
    let model = fixtures::tiny_bloom().unwrap();
    for field in 0..10 {
        let mut c = model.config().clone();
        match field {
            0 => c.vocabulary = 0,
            1 => c.width = 0,
            2 => c.heads = 0,
            3 => c.kv_heads = 3,
            4 => c.norm_epsilon = f32::NAN,
            5 => c.width = usize::MAX,
            6 => c.max_context = 0,
            7 => c.feed_forward = 0,
            8 => c.kv_heads = 0,
            9 => c.max_context = usize::MAX,
            _ => unreachable!(),
        }
        assert!(
            BloomModel::new(c, model.weights().clone()).is_err(),
            "field {field}"
        );
    }
    for field in 0..5 {
        let mut weights = model.weights().clone();
        match field {
            0 => {
                weights.embedding_norm.pop();
            }
            1 => {
                weights.embedding_norm_bias.pop();
            }
            2 => {
                weights.layers[0].qkv_bias.pop();
            }
            3 => {
                weights.layers[0].down_bias.pop();
            }
            4 => {
                weights.output_norm_bias.pop();
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            BloomModel::new(model.config().clone(), weights),
            Err(Error::Shape(_))
        ));
    }
    let mut weights = model.weights().clone();
    weights.embedding_norm_bias[0] = f32::NAN;
    assert!(matches!(
        BloomModel::new(model.config().clone(), weights),
        Err(Error::NonFinite(_))
    ));
}
#[test]
fn input_norm_affine_bias_and_output_changes_invalidate_saved_state() {
    let model = fixtures::tiny_bloom().unwrap();
    let mut original = Context::new(
        Arc::new(Model::Bloom(model.clone())),
        ContextOptions::default(),
    )
    .unwrap();
    original
        .decode_batch(&[BatchToken::token(0, 0, 1)])
        .unwrap();
    let state = original.save_state().unwrap();
    let checkpoint = original.checkpoint().unwrap();
    for field in 0..5 {
        let mut weights = model.weights().clone();
        match field {
            0 => weights.embedding_norm[0] += 0.5,
            1 => weights.embedding_norm_bias[0] += 0.5,
            2 => weights.output_norm_bias[0] += 0.5,
            3 => weights.layers[0].qkv_bias[0] += 0.5,
            4 => weights.output = Some(Matrix::zeros(7, 4).unwrap()),
            _ => unreachable!(),
        }
        let changed = BloomModel::new(model.config().clone(), weights).unwrap();
        assert_ne!(changed.fingerprint(), model.fingerprint());
        let mut context =
            Context::new(Arc::new(Model::Bloom(changed)), ContextOptions::default()).unwrap();
        assert!(matches!(
            context.restore_state(&state),
            Err(Error::InvalidState(_))
        ));
        assert!(matches!(
            context.restore_checkpoint(&checkpoint),
            Err(Error::InvalidState(_))
        ));
        assert_eq!(context.reserved_state_bytes(), 0);
    }
}
#[test]
fn finite_input_norm_weights_that_overflow_rollback_the_entire_batch() {
    let original = fixtures::tiny_bloom().unwrap();
    let mut weights = original.weights().clone();
    weights.embedding_norm = vec![f32::MAX; 4];
    weights.embedding_norm_bias = vec![f32::MAX; 4];
    let model = Arc::new(Model::Bloom(
        BloomModel::new(original.config().clone(), weights).unwrap(),
    ));
    let mut context = Context::new(
        model,
        ContextOptions {
            microbatch_tokens: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let before = context.save_state().unwrap();
    let result = context.decode_batch(&[BatchToken::token(0, 0, 1), BatchToken::token(7, 0, 2)]);
    assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
    assert_eq!(context.save_state().unwrap(), before);
    assert_eq!(context.reserved_state_bytes(), 0);
    assert_eq!(context.sequence_count(), 0);
}
