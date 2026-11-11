//! Observable GPT-2 context contracts; original numerical captures are in
//! llama-inference/tests/gpt2.rs.
use llama_model_runtime::*;
use std::sync::Arc;

#[test]
fn learned_position_capacity_and_required_shapes_are_checked() {
    let model = fixtures::tiny_gpt2().unwrap();
    for field in 0..7 {
        let mut c = model.config().clone();
        match field {
            0 => c.positions = 0,
            1 => c.positions = c.max_context - 1,
            2 => c.heads = 0,
            3 => c.kv_heads = 3,
            4 => c.norm_epsilon = f32::NAN,
            5 => c.width = usize::MAX,
            6 => c.max_context = 0,
            _ => unreachable!(),
        }
        assert!(Gpt2Model::new(c, model.weights().clone()).is_err());
    }
    let mut weights = model.weights().clone();
    weights.position_embedding = Matrix::zeros(31, 4).unwrap();
    assert!(matches!(
        Gpt2Model::new(model.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = model.weights().clone();
    weights.layers[0].qkv_bias.pop();
    assert!(matches!(
        Gpt2Model::new(model.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = model.weights().clone();
    weights.output_norm_bias[0] = f32::NAN;
    assert!(matches!(
        Gpt2Model::new(model.config().clone(), weights),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn learned_position_and_affine_bias_changes_invalidate_saved_state() {
    let model = fixtures::tiny_gpt2().unwrap();
    let mut original = Context::new(
        Arc::new(Model::Gpt2(model.clone())),
        ContextOptions::default(),
    )
    .unwrap();
    original
        .decode_batch(&[BatchToken::token(0, 0, 1)])
        .unwrap();
    let state = original.save_state().unwrap();
    let checkpoint = original.checkpoint().unwrap();
    for field in 0..3 {
        let mut weights = model.weights().clone();
        match field {
            0 => {
                let mut values = weights.position_embedding.values().to_vec();
                values[4] += 0.5;
                weights.position_embedding = Matrix::new(32, 4, values).unwrap();
            }
            1 => weights.output_norm_bias[0] += 0.5,
            2 => weights.layers[0].qkv_bias[0] += 0.5,
            _ => unreachable!(),
        }
        let changed = Gpt2Model::new(model.config().clone(), weights).unwrap();
        assert_ne!(changed.fingerprint(), model.fingerprint());
        let mut context =
            Context::new(Arc::new(Model::Gpt2(changed)), ContextOptions::default()).unwrap();
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
fn finite_affine_weights_that_overflow_rollback_the_entire_batch() {
    let original = fixtures::tiny_gpt2().unwrap();
    let mut weights = original.weights().clone();
    weights.output_bias = Some(vec![f32::MAX; 7]);
    weights.output_norm_bias = vec![f32::MAX; 4];
    let model = Arc::new(Model::Gpt2(
        Gpt2Model::new(original.config().clone(), weights).unwrap(),
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
