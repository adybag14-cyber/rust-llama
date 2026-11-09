//! Observable Qwen3 state and failure contracts; numeric parity is covered by
//! the independent original GGUF corpus in llama-inference/tests/qwen3.rs.
use llama_model_runtime::*;
use std::sync::Arc;

fn config() -> Qwen3Config {
    Qwen3Config {
        vocabulary: 3,
        width: 3,
        feed_forward: 5,
        heads: 2,
        head_dim: 4,
        kv_heads: 1,
        max_context: 64,
        rms_epsilon: 1e-5,
        rope_base: 10000.0,
    }
}
fn matrix(rows: usize, columns: usize, seed: usize) -> Matrix {
    Matrix::new(
        rows,
        columns,
        (0..rows * columns)
            .map(|i| (((i + seed) * 13 % 17) as i32 - 8) as f32 / 32.0)
            .collect(),
    )
    .unwrap()
}
fn ordinary() -> Qwen3Model {
    let c = config();
    let layers = (0..2)
        .map(|i| Qwen3Layer {
            attention_norm: vec![1.0, 0.75, 1.25],
            qkv: Qwen3Qkv::Separate {
                query: matrix(8, 3, i + 1),
                key: matrix(4, 3, i + 2),
                value: matrix(4, 3, i + 3),
                query_bias: None,
                key_bias: None,
                value_bias: None,
            },
            attention_output: matrix(3, 8, i + 4),
            query_norm: vec![1.0, 0.5, 1.25, 0.75],
            key_norm: vec![0.75, 1.5, 0.5, 1.0],
            feed_forward_norm: vec![1.0; 3],
            gate: matrix(5, 3, i + 5),
            up: matrix(5, 3, i + 6),
            down: matrix(3, 5, i + 7),
        })
        .collect();
    Qwen3Model::new(
        c,
        Qwen3Weights {
            token_embedding: matrix(3, 3, 9),
            layers,
            output_norm: vec![1.0; 3],
            output: None,
        },
    )
    .unwrap()
}
fn context(model: Arc<Model>, limit: usize) -> Context {
    Context::new(
        model,
        ContextOptions {
            max_state_bytes: limit,
            microbatch_tokens: 1,
            ..ContextOptions::default()
        },
    )
    .unwrap()
}
fn budget_error<T: std::fmt::Debug>(result: Result<T>) {
    assert!(
        matches!(result, Err(Error::InvalidState(ref message)) if message.contains("state budget")),
        "{result:?}"
    );
}

#[test]
fn explicit_head_width_and_checked_shape_products() {
    let c = config();
    c.validate().unwrap();
    assert_eq!(
        (
            c.query_width().unwrap(),
            c.kv_width().unwrap(),
            c.qkv_width().unwrap()
        ),
        (8, 4, 16)
    );
    assert!(!c.width.is_multiple_of(c.heads));
    for change in 0..10 {
        let mut c = c.clone();
        match change {
            0 => c.heads = 0,
            1 => c.kv_heads = 0,
            2 => c.kv_heads = 3,
            3 => c.head_dim = 0,
            4 => c.head_dim = 3,
            5 => c.width = 0,
            6 => c.max_context = 0,
            7 => c.rms_epsilon = f32::NAN,
            8 => c.rope_base = f32::INFINITY,
            9 => {
                c.heads = u32::MAX as usize;
                c.kv_heads = c.heads;
                c.head_dim = (u32::MAX - 1) as usize;
            }
            _ => unreachable!(),
        }
        assert!(c.validate().is_err(), "invalid configuration {change}");
    }
    let good = ordinary();
    let mut weights = good.weights().clone();
    weights.layers[0].query_norm.pop();
    assert!(matches!(
        Qwen3Model::new(good.config().clone(), weights),
        Err(Error::Shape(_))
    ));
    let mut weights = good.weights().clone();
    weights.layers[0].key_norm[0] = f32::NAN;
    assert!(matches!(
        Qwen3Model::new(good.config().clone(), weights),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn cancellation_at_every_boundary_rolls_back_all_sequences_and_reservations() {
    let model = Arc::new(Model::Qwen3(ordinary()));
    let mut initial = context(model.clone(), 1 << 20);
    initial.decode_batch(&[BatchToken::token(0, 0, 0)]).unwrap();
    let before = initial.save_state().unwrap();
    let resident = initial.reserved_state_bytes();
    let mut failures = 0;
    let mut completed = false;
    for stop_at in 1..64 {
        let mut tested = context(model.clone(), 1 << 20);
        tested.restore_state(&before).unwrap();
        let mut calls = 0;
        let result = tested.decode_batch_with_cancel(
            &[BatchToken::token(0, 1, 1), BatchToken::token(7, 0, 2)],
            || {
                calls += 1;
                calls == stop_at
            },
        );
        match result {
            Err(Error::Cancelled) => {
                failures += 1;
                assert_eq!(tested.save_state().unwrap(), before, "cancel {stop_at}");
                assert_eq!(tested.reserved_state_bytes(), resident, "cancel {stop_at}");
            }
            Ok(outputs) => {
                assert_eq!(outputs.len(), 2);
                assert_eq!(tested.next_position(0), Some(2));
                assert_eq!(tested.next_position(7), Some(1));
                completed = true;
                break;
            }
            other => panic!("unexpected cancellation result: {other:?}"),
        }
    }
    assert!(completed && failures >= 8);
}

#[test]
fn actual_kv_head_width_controls_growth_and_copy_budgets() {
    let model = Arc::new(Model::Qwen3(ordinary()));
    let mut reference = context(model.clone(), 1 << 20);
    reference
        .decode_batch(&[BatchToken::token(0, 0, 0)])
        .unwrap();
    let resident = reference.state_bytes();
    reference
        .decode_batch(&[BatchToken::token(0, 1, 1)])
        .unwrap();
    // Two layers, K and V, four values/head and one KV head, each an F32.
    assert_eq!(reference.state_bytes() - resident, 2 * 2 * 4 * 4);
    let mut limited = context(model, resident * 2 + 32);
    limited.decode_batch(&[BatchToken::token(0, 0, 0)]).unwrap();
    let before = limited.save_state().unwrap();
    let reserved = limited.reserved_state_bytes();
    let growth: Vec<_> = (1..32)
        .map(|position| BatchToken::token(0, position, 1))
        .collect();
    budget_error(limited.decode_batch(&growth));
    budget_error(
        limited.decode_batch(
            &(1..9)
                .map(|id| BatchToken::token(id, 0, 1))
                .collect::<Vec<_>>(),
        ),
    );
    budget_error(limited.fork_sequence(0, 9));
    assert_eq!(limited.reserved_state_bytes(), reserved);
    assert_eq!(limited.save_state().unwrap(), before);
    let checkpoint = limited.checkpoint().unwrap();
    assert_eq!(
        limited.reserved_state_bytes(),
        reserved + checkpoint.state_bytes()
    );
    budget_error(checkpoint.try_clone());
    budget_error(limited.restore_checkpoint(&checkpoint));
    drop(checkpoint);
    assert_eq!(limited.save_state().unwrap(), before);
    assert!(limited.peak_reserved_state_bytes() <= limited.state_budget_bytes());
}

#[test]
fn fork_truncate_serialization_and_model_identity_preserve_state() {
    let native = ordinary();
    let model = Arc::new(Model::Qwen3(native.clone()));
    let mut tested = context(model.clone(), 1 << 20);
    let first = tested
        .decode_batch(&[BatchToken::token(0, 0, 0), BatchToken::token(0, 1, 1)])
        .unwrap();
    let state = tested.save_state().unwrap();
    let checkpoint = tested.checkpoint().unwrap();
    tested.fork_sequence(0, 9).unwrap();
    let forked = tested
        .decode_batch(&[BatchToken::token(0, 2, 2), BatchToken::token(9, 2, 2)])
        .unwrap();
    assert_eq!(forked[0].logits, forked[1].logits);
    tested.truncate_sequence(0, 1).unwrap();
    let replay = tested.decode_batch(&[BatchToken::token(0, 1, 1)]).unwrap();
    assert_eq!(replay[0].logits, first[1].logits);
    tested.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(tested.save_state().unwrap(), state);
    let mut other = context(model, 1 << 20);
    other.restore_state(&state).unwrap();
    assert_eq!(
        other.decode_batch(&[BatchToken::token(0, 2, 2)]).unwrap()[0].logits,
        forked[0].logits
    );
    for query in [false, true] {
        let mut weights = native.weights().clone();
        if query {
            weights.layers[0].query_norm[0] += 0.25;
        } else {
            weights.layers[0].key_norm[0] += 0.25;
        }
        let changed = Arc::new(Model::Qwen3(
            Qwen3Model::new(native.config().clone(), weights).unwrap(),
        ));
        let mut different = context(changed, 1 << 20);
        assert!(matches!(
            different.restore_state(&state),
            Err(Error::InvalidState(_))
        ));
        assert!(matches!(
            different.restore_checkpoint(&checkpoint),
            Err(Error::InvalidState(_))
        ));
        assert_eq!(different.reserved_state_bytes(), 0);
    }
}

fn finite_weight_overflow() -> Arc<Model> {
    let c = Qwen3Config {
        vocabulary: 2,
        width: 2,
        feed_forward: 2,
        heads: 2,
        head_dim: 4,
        kv_heads: 1,
        max_context: 8,
        rms_epsilon: 1e-5,
        rope_base: 10000.0,
    };
    Arc::new(Model::Qwen3(
        Qwen3Model::new(
            c,
            Qwen3Weights {
                token_embedding: Matrix::new(2, 2, vec![0.0, 0.0, 1.0, 0.0]).unwrap(),
                layers: vec![Qwen3Layer {
                    attention_norm: vec![1.0; 2],
                    qkv: Qwen3Qkv::Separate {
                        query: Matrix::zeros(8, 2).unwrap(),
                        key: Matrix::zeros(4, 2).unwrap(),
                        value: Matrix::zeros(4, 2).unwrap(),
                        query_bias: None,
                        key_bias: None,
                        value_bias: None,
                    },
                    attention_output: Matrix::zeros(2, 8).unwrap(),
                    query_norm: vec![1.0; 4],
                    key_norm: vec![1.0; 4],
                    feed_forward_norm: vec![1.0; 2],
                    gate: Matrix::zeros(2, 2).unwrap(),
                    up: Matrix::zeros(2, 2).unwrap(),
                    down: Matrix::zeros(2, 2).unwrap(),
                }],
                output_norm: vec![f32::MAX; 2],
                output: None,
            },
        )
        .unwrap(),
    ))
}

#[test]
fn finite_weights_with_nonfinite_execution_never_commit_partial_state() {
    let model = finite_weight_overflow();
    for preexisting in [false, true] {
        let mut tested = context(model.clone(), 1 << 20);
        let position = usize::from(preexisting);
        if preexisting {
            tested.decode_batch(&[BatchToken::token(0, 0, 0)]).unwrap();
        }
        let before = tested.save_state().unwrap();
        let reserved = tested.reserved_state_bytes();
        let result = tested.decode_batch(&[
            BatchToken::token(0, position, 0),
            BatchToken::token(7, 0, 0),
            BatchToken::token(0, position + 1, 1),
        ]);
        assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
        assert_eq!(tested.save_state().unwrap(), before);
        assert_eq!(tested.reserved_state_bytes(), reserved);
        assert_eq!(tested.sequence_count(), usize::from(preexisting));
    }
}
