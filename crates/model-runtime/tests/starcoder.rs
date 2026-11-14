//! Observable StarCoder context contracts; original numerical captures are in
//! llama-inference/tests/starcoder.rs.
use llama_model_runtime::*;
use std::sync::Arc;

#[test]
fn learned_position_capacity_and_required_shapes_are_checked() {
    let model = fixtures::tiny_starcoder().unwrap();
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
        assert!(StarcoderModel::new(c, model.weights().clone()).is_err());
    }
    let mut weights = model.weights().clone();
    weights.position_embedding = Matrix::zeros(31, 4).unwrap();
    assert!(matches!(
        StarcoderModel::new(model.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = model.weights().clone();
    weights.layers[0].qkv_bias.pop();
    assert!(matches!(
        StarcoderModel::new(model.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = model.weights().clone();
    weights.output_norm_bias[0] = f32::NAN;
    assert!(matches!(
        StarcoderModel::new(model.config().clone(), weights),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn learned_position_and_affine_bias_changes_invalidate_saved_state() {
    let model = fixtures::tiny_starcoder().unwrap();
    let mut original = Context::new(
        Arc::new(Model::Starcoder(model.clone())),
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
        let changed = StarcoderModel::new(model.config().clone(), weights).unwrap();
        assert_ne!(changed.fingerprint(), model.fingerprint());
        let mut context = Context::new(
            Arc::new(Model::Starcoder(changed)),
            ContextOptions::default(),
        )
        .unwrap();
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
    let original = fixtures::tiny_starcoder().unwrap();
    let mut weights = original.weights().clone();
    weights.output_norm = vec![f32::MAX; 4];
    weights.output_norm_bias = vec![f32::MAX; 4];
    let model = Arc::new(Model::Starcoder(
        StarcoderModel::new(original.config().clone(), weights).unwrap(),
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

#[test]
fn rotary_family_and_qkv_identity_are_checked() {
    let model = fixtures::tiny_starcoder2().unwrap();
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut config = model.config().clone();
        config.rope_base = value;
        assert!(Starcoder2Model::new(config, model.weights().clone()).is_err());
    }
    let mut original = Context::new(
        Arc::new(Model::Starcoder2(model.clone())),
        ContextOptions::default(),
    )
    .unwrap();
    original
        .decode_batch(&[BatchToken::token(0, 0, 1)])
        .unwrap();
    let state = original.save_state().unwrap();
    let checkpoint = original.checkpoint().unwrap();
    for field in 0..6 {
        let mut config = model.config().clone();
        let mut weights = model.weights().clone();
        match field {
            0 => config.rope_base = 500.0,
            1 => weights.layers[0].feed_forward_norm_bias[0] += 0.5,
            2 => weights.layers[0].attention_output_bias[0] += 0.5,
            3 => weights.layers[0].up_bias[0] += 0.5,
            4 => {
                let Starcoder2Qkv::Fused { bias, .. } = &mut weights.layers[0].qkv else {
                    panic!("fixture")
                };
                bias.as_mut().unwrap()[0] += 0.5;
            }
            5 => weights.output = None,
            _ => unreachable!(),
        }
        let changed = Starcoder2Model::new(config, weights).unwrap();
        assert_ne!(changed.fingerprint(), model.fingerprint());
        let mut context = Context::new(
            Arc::new(Model::Starcoder2(changed)),
            ContextOptions::default(),
        )
        .unwrap();
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
    let mut weights = model.weights().clone();
    let Starcoder2Qkv::Fused { query_bias, .. } = &mut weights.layers[0].qkv else {
        panic!("fixture")
    };
    *query_bias = Some(vec![0.0; 4]);
    assert!(matches!(
        Starcoder2Model::new(model.config().clone(), weights),
        Err(Error::InvalidConfig(_))
    ));
}

#[test]
fn rotary_nonfinite_execution_rolls_back_all_sequences() {
    let original = fixtures::tiny_starcoder2().unwrap();
    let mut weights = original.weights().clone();
    weights.layers[0].up_bias = vec![f32::MAX; 6];
    weights.layers[0].attention_output_bias = vec![f32::MAX; 4];
    let model = Arc::new(Model::Starcoder2(
        Starcoder2Model::new(original.config().clone(), weights).unwrap(),
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
