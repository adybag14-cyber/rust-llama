//! GPT-NeoX state contracts; independent native numerics live in inference/tests/gptneox.rs.
use llama_model_runtime::*;
use std::sync::Arc;

#[test]
fn rotary_and_affine_shape_boundaries_are_checked() {
    let model = fixtures::tiny_gptneox().unwrap();
    for field in 0..8 {
        let mut c = model.config().clone();
        match field {
            0 => c.rope_dimensions = 0,
            1 => c.rope_dimensions = 1,
            2 => c.rope_dimensions = 4,
            3 => c.heads = 0,
            4 => c.kv_heads = 3,
            5 => c.rope_base = f32::INFINITY,
            6 => c.width = usize::MAX,
            7 => c.max_context = 0,
            _ => unreachable!(),
        }
        assert!(GptNeoXModel::new(c, model.weights().clone()).is_err());
    }
    let mut weights = model.weights().clone();
    weights.output = Matrix::zeros(6, 4).unwrap();
    assert!(matches!(
        GptNeoXModel::new(model.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = model.weights().clone();
    weights.layers[0].qkv_bias.pop();
    assert!(matches!(
        GptNeoXModel::new(model.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = model.weights().clone();
    weights.output_norm_bias[0] = f32::NAN;
    assert!(matches!(
        GptNeoXModel::new(model.config().clone(), weights),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn residual_rope_and_affine_changes_invalidate_saved_state() {
    let model = fixtures::tiny_gptneox().unwrap();
    let mut original = Context::new(
        Arc::new(Model::GptNeoX(model.clone())),
        ContextOptions::default(),
    )
    .unwrap();
    original
        .decode_batch(&[BatchToken::token(0, 0, 1)])
        .unwrap();
    let state = original.save_state().unwrap();
    let checkpoint = original.checkpoint().unwrap();
    for field in 0..4 {
        let mut config = model.config().clone();
        let mut weights = model.weights().clone();
        match field {
            0 => config.parallel_residual = !config.parallel_residual,
            1 => config.rope_base = 1000.0,
            2 => weights.output_norm_bias[0] += 0.5,
            3 => weights.layers[0].qkv_bias[0] += 0.5,
            _ => unreachable!(),
        }
        let changed = GptNeoXModel::new(config, weights).unwrap();
        assert_ne!(changed.fingerprint(), model.fingerprint());
        let mut context =
            Context::new(Arc::new(Model::GptNeoX(changed)), ContextOptions::default()).unwrap();
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
fn finite_projection_overflow_rolls_back_all_sequences_in_both_residual_modes() {
    let original = fixtures::tiny_gptneox().unwrap();
    for parallel_residual in [false, true] {
        let mut config = original.config().clone();
        config.parallel_residual = parallel_residual;
        let mut weights = original.weights().clone();
        weights.output = Matrix::new(7, 4, vec![f32::MAX; 28]).unwrap();
        weights.output_norm_bias = vec![f32::MAX; 4];
        let model = Arc::new(Model::GptNeoX(GptNeoXModel::new(config, weights).unwrap()));
        let mut context = Context::new(
            model,
            ContextOptions {
                microbatch_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let before = context.save_state().unwrap();
        let result =
            context.decode_batch(&[BatchToken::token(0, 0, 1), BatchToken::token(7, 0, 2)]);
        assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
        assert_eq!(context.save_state().unwrap(), before);
        assert_eq!(context.reserved_state_bytes(), 0);
        assert_eq!(context.sequence_count(), 0);
    }
}
