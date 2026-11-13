//! Falcon strict-state boundaries; original numerical outputs are exercised by
//! llama-inference/tests/falcon.rs.
use llama_model_runtime::*;
use std::sync::Arc;

#[test]
fn shape_positivity_rotary_and_optional_norm_vectors_are_checked() {
    let model = fixtures::tiny_falcon().unwrap();
    for field in 0..12 {
        let mut config = model.config().clone();
        match field {
            0 => config.vocabulary = 0,
            1 => config.width = 0,
            2 => config.heads = 0,
            3 => config.kv_heads = 3,
            4 => config.norm_epsilon = f32::NAN,
            5 => config.width = usize::MAX,
            6 => config.max_context = 0,
            7 => config.feed_forward = 0,
            8 => config.kv_heads = 0,
            9 => config.max_context = usize::MAX,
            10 => config.rope_base = f32::INFINITY,
            11 => config.rope_base = 0.0,
            _ => unreachable!(),
        }
        assert!(
            FalconModel::new(config, model.weights().clone()).is_err(),
            "field {field}"
        );
    }
    for field in 0..5 {
        let mut weights = model.weights().clone();
        match field {
            0 => {
                weights.layers[0].attention_norm.pop();
            }
            1 => {
                weights.layers[0].attention_norm_bias.pop();
            }
            2 => {
                weights.layers[0].attention_norm_2.as_mut().unwrap().pop();
            }
            3 => {
                weights.layers[0]
                    .attention_norm_2_bias
                    .as_mut()
                    .unwrap()
                    .pop();
            }
            4 => {
                weights.output_norm_bias.pop();
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            FalconModel::new(model.config().clone(), weights),
            Err(Error::Shape(_))
        ));
    }
    let mut weights = model.weights().clone();
    weights.layers[0].attention_norm_2.as_mut().unwrap()[0] = f32::NAN;
    assert!(matches!(
        FalconModel::new(model.config().clone(), weights),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn model_identity_includes_both_norms_optional_bias_and_output_tying() {
    let model = fixtures::tiny_falcon().unwrap();
    let mut original = Context::new(
        Arc::new(Model::Falcon(model.clone())),
        ContextOptions::default(),
    )
    .unwrap();
    original
        .decode_batch(&[BatchToken::token(0, 0, 1)])
        .unwrap();
    let state = original.save_state().unwrap();
    let checkpoint = original.checkpoint().unwrap();
    for field in 0..7 {
        let mut weights = model.weights().clone();
        match field {
            0 => weights.layers[0].attention_norm[0] += 0.5,
            1 => weights.layers[0].attention_norm_bias[0] += 0.5,
            2 => weights.output_norm_bias[0] += 0.5,
            3 => weights.layers[0].attention_norm_2.as_mut().unwrap()[0] += 0.5,
            4 => weights.layers[0].attention_norm_2 = None,
            5 => weights.layers[0].attention_norm_2_bias = None,
            6 => weights.output = None,
            _ => unreachable!(),
        }
        let changed = FalconModel::new(model.config().clone(), weights).unwrap();
        assert_ne!(changed.fingerprint(), model.fingerprint());
        let mut context =
            Context::new(Arc::new(Model::Falcon(changed)), ContextOptions::default()).unwrap();
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
fn failed_second_norm_execution_rolls_back_all_sequences() {
    let original = fixtures::tiny_falcon().unwrap();
    let mut weights = original.weights().clone();
    weights.layers[0].attention_norm_2 = Some(vec![f32::MAX; 4]);
    weights.layers[0].attention_norm_2_bias = Some(vec![f32::MAX; 4]);
    let model = Arc::new(Model::Falcon(
        FalconModel::new(original.config().clone(), weights).unwrap(),
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
fn inactive_second_bias_is_retained_without_affecting_execution() {
    let original = fixtures::tiny_falcon().unwrap();
    let mut with_bias = original.weights().clone();
    for layer in &mut with_bias.layers {
        layer.attention_norm_2 = None;
    }
    let mut without_bias = with_bias.clone();
    for layer in &mut without_bias.layers {
        layer.attention_norm_2_bias = None;
    }
    let run = |weights| {
        let model = FalconModel::new(original.config().clone(), weights).unwrap();
        let mut context =
            Context::new(Arc::new(Model::Falcon(model)), ContextOptions::default()).unwrap();
        context
            .decode_batch(&[BatchToken::token(0, 0, 1), BatchToken::token(0, 1, 2)])
            .unwrap()
    };
    assert_eq!(run(with_bias), run(without_bias));
}
