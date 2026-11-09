//! Resource-boundary regressions. Numeric oracle tests live separately; these
//! assert observable atomicity and reservation lifetime under real state copies.
use super::*;
use crate::{Qwen2Config, Qwen2Layer, Qwen2Model, Qwen2Qkv, Qwen2Weights, fixtures};
use crate::{Qwen3Config, Qwen3Layer, Qwen3Model, Qwen3Qkv, Qwen3Weights};
use owned::AllocationFailure;

fn models() -> Vec<Arc<Model>> {
    let llama = fixtures::tiny_llama().unwrap();
    let c = llama.config();
    let w = llama.weights();
    let qwen = Qwen2Model::new(
        Qwen2Config {
            vocabulary: c.vocabulary,
            width: c.width,
            feed_forward: c.feed_forward,
            heads: c.heads,
            kv_heads: c.kv_heads,
            max_context: c.max_context,
            rms_epsilon: c.rms_epsilon,
            rope_base: c.rope_base,
        },
        Qwen2Weights {
            token_embedding: w.token_embedding.clone(),
            output_norm: w.output_norm.clone(),
            output: w.output.clone(),
            output_bias: None,
            layers: w
                .layers
                .iter()
                .map(|layer| Qwen2Layer {
                    attention_norm: layer.attention_norm.clone(),
                    qkv: Qwen2Qkv::Separate {
                        query: layer.query.clone(),
                        key: layer.key.clone(),
                        value: layer.value.clone(),
                        query_bias: layer.query_bias.clone(),
                        key_bias: layer.key_bias.clone(),
                        value_bias: layer.value_bias.clone(),
                    },
                    attention_output: layer.attention_output.clone(),
                    feed_forward_norm: layer.feed_forward_norm.clone(),
                    gate: layer.gate.clone(),
                    up: layer.up.clone(),
                    down: layer.down.clone(),
                })
                .collect(),
        },
    )
    .unwrap();
    let qwen3 = Qwen3Model::new(
        Qwen3Config {
            vocabulary: c.vocabulary,
            width: c.width,
            feed_forward: c.feed_forward,
            heads: c.heads,
            head_dim: c.head_dim().unwrap(),
            kv_heads: c.kv_heads,
            max_context: c.max_context,
            rms_epsilon: c.rms_epsilon,
            rope_base: c.rope_base,
        },
        Qwen3Weights {
            token_embedding: w.token_embedding.clone(),
            output_norm: w.output_norm.clone(),
            output: w.output.clone(),
            layers: w
                .layers
                .iter()
                .map(|layer| Qwen3Layer {
                    attention_norm: layer.attention_norm.clone(),
                    qkv: Qwen3Qkv::Separate {
                        query: layer.query.clone(),
                        key: layer.key.clone(),
                        value: layer.value.clone(),
                        query_bias: layer.query_bias.clone(),
                        key_bias: layer.key_bias.clone(),
                        value_bias: layer.value_bias.clone(),
                    },
                    attention_output: layer.attention_output.clone(),
                    query_norm: vec![1.0; c.head_dim().unwrap()],
                    key_norm: vec![1.0; c.head_dim().unwrap()],
                    feed_forward_norm: layer.feed_forward_norm.clone(),
                    gate: layer.gate.clone(),
                    up: layer.up.clone(),
                    down: layer.down.clone(),
                })
                .collect(),
        },
    )
    .unwrap();
    vec![
        Arc::new(Model::Llama(llama)),
        Arc::new(Model::Mamba(fixtures::tiny_mamba().unwrap())),
        Arc::new(Model::Qwen2(qwen)),
        Arc::new(Model::Qwen3(qwen3)),
    ]
}
fn context(model: Arc<Model>, limit: usize) -> Context {
    Context::new(
        model,
        ContextOptions {
            max_state_bytes: limit,
            ..Default::default()
        },
    )
    .unwrap()
}
fn start(context: &mut Context) {
    context.decode_batch(&[BatchToken::token(0, 0, 1)]).unwrap();
}
fn assert_budget_error<T: std::fmt::Debug>(result: Result<T>) {
    match result {
        Err(Error::InvalidState(message)) => assert!(message.contains("state budget"), "{message}"),
        other => panic!("expected checked state budget failure, got {other:?}"),
    }
}

#[test]
fn explicit_default_and_zero_state_budget() {
    assert_eq!(
        ContextOptions::default().max_state_bytes,
        1024 * 1024 * 1024
    );
    for model in models() {
        assert!(matches!(
            Context::new(
                model.clone(),
                ContextOptions {
                    max_state_bytes: 0,
                    ..Default::default()
                }
            ),
            Err(Error::InvalidConfig(_))
        ));
        let context = context(model, 1024);
        assert_eq!(context.state_bytes(), 0);
        assert_eq!(context.reserved_state_bytes(), 0);
        assert_eq!(context.state_budget_bytes(), 1024);
    }
}

#[test]
fn initial_state_capacity_has_an_exact_checked_threshold() {
    for model in models() {
        let mut reference = context(model.clone(), 1 << 20);
        start(&mut reference);
        let required = reference.peak_reserved_state_bytes();
        assert!(required > reference.state_bytes()); // the allocation plan is included
        let mut exact = context(model.clone(), required);
        start(&mut exact);
        assert_eq!(exact.state_bytes(), reference.state_bytes());
        assert_eq!(exact.peak_reserved_state_bytes(), required);
        let mut too_small = context(model, required - 1);
        assert_budget_error(too_small.decode_batch(&[BatchToken::token(0, 0, 1)]));
        assert_eq!(too_small.sequence_count(), 0);
        assert_eq!(too_small.reserved_state_bytes(), 0);
        assert!(too_small.peak_reserved_state_bytes() < required);
    }
}

#[test]
fn pending_kv_growth_and_sequence_multiplicity_cannot_bypass_budget() {
    for model in models() {
        let mut reference = context(model.clone(), 1 << 20);
        start(&mut reference);
        let resident = reference.state_bytes();
        let mut limited = context(model.clone(), resident * 2 + 256);
        start(&mut limited);
        let original = limited.save_state().unwrap();
        let original_reserved = limited.reserved_state_bytes();
        let many_sequences: Vec<_> = (1..9).map(|id| BatchToken::token(id, 0, 2)).collect();
        assert_budget_error(limited.decode_batch(&many_sequences));
        assert_eq!(limited.sequence_count(), 1);
        assert_eq!(limited.next_position(0), Some(1));
        assert_eq!(limited.reserved_state_bytes(), original_reserved);
        assert_eq!(limited.save_state().unwrap(), original);
        let growth: Vec<_> = (1..32)
            .map(|position| BatchToken::token(0, position, 2))
            .collect();
        if matches!(model.as_ref(), Model::Mamba(_)) {
            // Recurrent state is fixed-size: token count does not multiply it.
            limited.decode_batch(&growth).unwrap();
            assert_eq!(limited.state_bytes(), resident);
            assert_eq!(limited.next_position(0), Some(32));
        } else {
            assert_budget_error(limited.decode_batch(&growth));
            assert_eq!(limited.save_state().unwrap(), original);
        }
        assert!(limited.peak_reserved_state_bytes() <= limited.state_budget_bytes());
    }
}

#[test]
fn live_checkpoints_and_fallible_copies_share_the_originating_budget() {
    for model in models() {
        let mut reference = context(model.clone(), 1 << 20);
        start(&mut reference);
        let resident = reference.state_bytes();
        let mut context = context(model, resident * 4 + 256);
        start(&mut context);
        let checkpoint = context.checkpoint().unwrap();
        assert_eq!(
            context.reserved_state_bytes(),
            resident + checkpoint.state_bytes()
        );
        let mut copies = Vec::new();
        loop {
            let before = context.reserved_state_bytes();
            match checkpoint.try_clone() {
                Ok(copy) => {
                    assert!(
                        copies.len() < 16,
                        "checkpoint copies escaped the shared ledger"
                    );
                    assert_eq!(context.reserved_state_bytes(), before + copy.state_bytes());
                    copies.push(copy);
                }
                Err(error) => {
                    assert_budget_error::<()>(Err(error));
                    assert_eq!(context.reserved_state_bytes(), before);
                    break;
                }
            }
        }
        assert!(!copies.is_empty());
        context.reset();
        assert_eq!(context.state_bytes(), 0);
        let extra = checkpoint.try_clone().unwrap();
        drop(extra);
        drop(copies);
        assert_eq!(context.reserved_state_bytes(), checkpoint.state_bytes());
        let ledger = Arc::clone(&context.budget);
        drop(context);
        let after_owner_drop = checkpoint.try_clone().unwrap();
        assert_eq!(
            ledger.used(),
            checkpoint.state_bytes() + after_owner_drop.state_bytes()
        );
        drop(after_owner_drop);
        drop(checkpoint);
        assert_eq!(ledger.used(), 0);
    }
}

#[test]
fn fork_checkpoint_restore_and_serialized_restore_fail_atomically() {
    for model in models() {
        let mut reference = context(model.clone(), 1 << 20);
        start(&mut reference);
        let resident = reference.state_bytes();
        reference.fork_sequence(0, 8).unwrap();
        let two_sequences = reference.save_state().unwrap();
        let mut limited = context(model.clone(), resident * 2 + 32);
        start(&mut limited);
        let before = limited.save_state().unwrap();
        let bytes = limited.reserved_state_bytes();
        assert_budget_error(limited.fork_sequence(0, 8));
        assert_eq!(limited.sequence_count(), 1);
        assert_eq!(limited.reserved_state_bytes(), bytes);
        assert_budget_error(limited.restore_state(&two_sequences));
        assert_eq!(limited.save_state().unwrap(), before);
        let checkpoint = limited.checkpoint().unwrap();
        let with_checkpoint = limited.reserved_state_bytes();
        assert_budget_error(limited.restore_checkpoint(&checkpoint));
        assert_eq!(limited.reserved_state_bytes(), with_checkpoint);
        drop(checkpoint);
        assert_eq!(limited.save_state().unwrap(), before);
        let mut other = context(model, 1 << 20);
        let checkpoint = limited.checkpoint().unwrap();
        let origin = limited.reserved_state_bytes();
        other.restore_checkpoint(&checkpoint).unwrap();
        assert_eq!(other.save_state().unwrap(), before);
        assert_eq!(other.reserved_state_bytes(), resident);
        assert_eq!(limited.reserved_state_bytes(), origin);
    }
}

#[test]
fn serialized_output_reserve_is_checked_and_then_returns_caller_owned_bytes() {
    for model in models() {
        let mut reference = context(model.clone(), 1 << 20);
        start(&mut reference);
        let expected = reference.save_state().unwrap();
        let required = reference.state_bytes() + expected.len();
        let mut exact = context(model.clone(), required);
        start(&mut exact);
        let before = exact.reserved_state_bytes();
        let bytes = exact.save_state().unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(exact.reserved_state_bytes(), before);
        assert_eq!(exact.peak_reserved_state_bytes(), required);
        let mut too_small = context(model, required - 1);
        start(&mut too_small);
        let before = too_small.reserved_state_bytes();
        assert_budget_error(too_small.save_state());
        assert_eq!(too_small.reserved_state_bytes(), before);
    }
}

#[test]
fn every_fallible_state_reservation_rolls_back_when_allocation_fails() {
    for model in models() {
        let mut reference = context(model.clone(), 1 << 20);
        start(&mut reference);
        reference.fork_sequence(0, 8).unwrap();
        let replacement = reference.save_state().unwrap();
        for operation in [
            "decode",
            "fork",
            "checkpoint",
            "checkpoint-copy",
            "restore",
            "serialize",
        ] {
            let mut failures = 0;
            let mut success = false;
            for fault_after in 0..64 {
                let mut context = context(model.clone(), 1 << 20);
                start(&mut context);
                let checkpoint = context.checkpoint().unwrap();
                let before = context.save_state().unwrap();
                let reserved = context.reserved_state_bytes();
                let fault = AllocationFailure::after(fault_after);
                let result = match operation {
                    "decode" => context
                        .decode_batch(&[BatchToken::token(0, 1, 2), BatchToken::token(1, 0, 3)])
                        .map(drop),
                    "fork" => context.fork_sequence(0, 8),
                    "checkpoint" => context.checkpoint().map(drop),
                    "checkpoint-copy" => checkpoint.try_clone().map(drop),
                    "restore" => context.restore_state(&replacement),
                    "serialize" => context.save_state().map(drop),
                    _ => unreachable!(),
                };
                drop(fault);
                match result {
                    Err(Error::InvalidState(message))
                        if message.contains("injected allocation") =>
                    {
                        failures += 1;
                        assert_eq!(
                            context.reserved_state_bytes(),
                            reserved,
                            "{operation}/{fault_after}"
                        );
                        assert_eq!(
                            context.save_state().unwrap(),
                            before,
                            "{operation}/{fault_after}"
                        );
                    }
                    Ok(()) => {
                        success = true;
                        break;
                    }
                    other => panic!("{operation}/{fault_after}: {other:?}"),
                }
            }
            assert!(
                success,
                "allocation enumeration never succeeded for {operation}"
            );
            assert!(
                failures > 0,
                "no fallible allocation exercised for {operation}"
            );
        }
    }
}

#[test]
fn reset_removal_and_truncation_preserve_truthful_capacity_accounting() {
    let model = models().remove(0);
    let mut context = context(model, 1 << 20);
    let input: Vec<_> = (0..8)
        .map(|position| BatchToken::token(0, position, 1))
        .collect();
    context.decode_batch(&input).unwrap();
    let capacity = context.state_bytes();
    context.truncate_sequence(0, 1).unwrap();
    assert_eq!(
        context.state_bytes(),
        capacity,
        "retained allocation must remain charged"
    );
    context.decode_batch(&[BatchToken::token(0, 1, 2)]).unwrap();
    assert!(
        context.state_bytes() < capacity,
        "the new transaction compacts retained capacity"
    );
    context.fork_sequence(0, 8).unwrap();
    let both = context.state_bytes();
    assert!(context.remove_sequence(8));
    assert!(context.state_bytes() < both);
    assert!(context.remove_sequence(0));
    assert_eq!(context.state_bytes(), 0);
    assert_eq!(context.reserved_state_bytes(), 0);
}

#[test]
fn checked_overflow_and_concurrent_reservations_never_exceed_the_ledger() {
    assert!(owned::allocate::<f32>(usize::MAX, "oversized test").is_err());
    let budget = StateBudget::new(usize::MAX);
    let all = budget.reserve(usize::MAX).unwrap();
    assert!(budget.reserve(1).is_err());
    drop(all);
    assert_eq!(budget.used(), 0);
    let budget = StateBudget::new(128);
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let budget = Arc::clone(&budget);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let reservation = budget.reserve(32);
                barrier.wait(); // every attempted reservation stays live until all attempts finish
                reservation.is_ok()
            })
        })
        .collect();
    let successes = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .filter(|success| *success)
        .count();
    assert_eq!(successes, 4);
    assert_eq!(budget.peak(), 128);
    assert_eq!(budget.used(), 0);
}
