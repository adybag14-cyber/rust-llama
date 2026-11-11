use llama_model_runtime::*;
use std::{collections::BTreeSet, sync::Arc};

fn llama() -> Arc<Model> {
    Arc::new(Model::Llama(fixtures::tiny_llama().unwrap()))
}
fn mamba() -> Arc<Model> {
    Arc::new(Model::Mamba(fixtures::tiny_mamba().unwrap()))
}
fn context(model: Arc<Model>) -> Context {
    Context::new(model, ContextOptions::default()).unwrap()
}
fn batch(sequence: i32, tokens: &[u32]) -> Vec<BatchToken> {
    tokens
        .iter()
        .enumerate()
        .map(|(p, t)| BatchToken::token(sequence, p, *t))
        .collect()
}

#[test]
fn exact_registry_ids_names_and_explicit_unsupported_models() {
    assert_eq!(registry::ARCHITECTURES.len(), 173);
    let mut names = BTreeSet::new();
    for (index, arch) in registry::ARCHITECTURES.iter().enumerate() {
        assert_eq!(arch.id as usize, index);
        assert!(names.insert(arch.name));
        assert_eq!(registry::by_name(arch.name).unwrap().id, arch.id);
        assert_eq!(registry::by_id(arch.id).unwrap().name, arch.name);
        assert_eq!(
            registry::require_execution_slice(arch.name).is_ok(),
            arch.executable_slice.is_some()
        );
    }
    assert_eq!(registry::ARCHITECTURES[1].name, "llama");
    assert_eq!(registry::ARCHITECTURES[13].name, "bert");
    assert_eq!(registry::ARCHITECTURES[60].name, "mamba");
    assert_eq!(registry::ARCHITECTURES[172].name, "(unknown)");
    assert!(registry::by_id(173).is_none());
    assert!(registry::require_execution_slice("invented").is_err());
    assert!(registry::by_name("gptj").unwrap().source_factory.is_none());
    assert_eq!(registry::MODEL_SOURCE_FILES.len(), 176);
    assert_eq!(
        registry::CONVERSION_ALIASES
            .iter()
            .filter(|a| a.text_architecture_id.is_some())
            .count(),
        312
    );
    assert_eq!(
        registry::CONVERSION_ALIASES
            .iter()
            .filter(|a| a.text_architecture_id.is_none())
            .count(),
        92
    );
}

#[test]
fn matrix_shape_nonfinite_and_overflow_are_rejected() {
    assert!(Matrix::new(2, 3, vec![0.0; 5]).is_err());
    assert!(Matrix::new(0, 3, vec![]).is_err());
    assert!(Matrix::new(usize::MAX, 2, vec![]).is_err());
    assert!(Matrix::new(1, 1, vec![f32::NAN]).is_err());
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(m.multiply(&[2.0, 3.0]).unwrap(), vec![8.0, 18.0]);
    assert!(m.multiply(&[1.0]).is_err());
    assert!(m.multiply(&[f32::INFINITY, 1.0]).is_err());
    assert!(m.row(2).is_err());
}

#[test]
fn malformed_model_shapes_and_norms_fail_before_execution() {
    let m = fixtures::tiny_llama().unwrap();
    let mut config = m.config().clone();
    config.heads = 3;
    assert!(LlamaModel::new(config, m.weights().clone()).is_err());
    let mut config = m.config().clone();
    config.rms_epsilon = 0.0;
    assert!(LlamaModel::new(config, m.weights().clone()).is_err());
    let mut weights = m.weights().clone();
    weights.layers[0].key = Matrix::zeros(4, 4).unwrap();
    assert!(LlamaModel::new(m.config().clone(), weights).is_err());
    let m = fixtures::tiny_mamba().unwrap();
    let mut config = m.config().clone();
    config.inner = 4;
    assert!(MambaModel::new(config, m.weights().clone()).is_err());
    let m = fixtures::tiny_bert().unwrap();
    let mut weights = m.weights().clone();
    weights.embedding_norm.pop();
    assert!(BertModel::new(m.config().clone(), weights).is_err());
}

#[test]
fn public_config_dimension_helpers_reject_zero_invalid_and_overflowing_shapes() {
    for (width, heads, kv_heads, head_error, kv_error) in [
        (4, 2, 1, false, false),
        (0, 2, 1, true, true),
        (4, 0, 1, true, true),
        (4, 3, 1, true, true),
        (4, 2, 0, false, true),
        (4, 2, 3, false, true),
        (usize::MAX, 1, 2, usize::MAX > u32::MAX as usize, true),
        (4, 8, 1, true, true),
    ] {
        let mut llama = fixtures::tiny_llama().unwrap().config().clone();
        llama.width = width;
        llama.heads = heads;
        llama.kv_heads = kv_heads;
        assert_eq!(llama.head_dim().is_err(), head_error);
        assert_eq!(llama.kv_width().is_err(), kv_error);
        let mut bert = fixtures::tiny_bert().unwrap().config().clone();
        bert.width = width;
        bert.heads = heads;
        bert.kv_heads = kv_heads;
        assert_eq!(bert.head_dim().is_err(), head_error);
        assert_eq!(bert.kv_width().is_err(), kv_error);
    }
}

#[test]
fn mamba_dimensions_are_checked_before_state_arithmetic() {
    let original = fixtures::tiny_mamba().unwrap();
    for field in 0..4 {
        let mut config = original.config().clone();
        match field {
            0 => config.convolution_width = 0,
            1 => config.inner = usize::MAX,
            2 => config.width = usize::MAX,
            _ => config.state_size = usize::MAX,
        }
        assert!(MambaModel::new(config, original.weights().clone()).is_err());
    }
}

#[test]
fn batching_and_microbatch_boundaries_preserve_decode() {
    for model in [llama(), mamba()] {
        let prompt = batch(0, &[1, 2, 3, 4]);
        let mut one = context(model.clone());
        let expected = one.decode_batch(&prompt).unwrap();
        for microbatch_tokens in [1, 2, 3, 4] {
            let options = ContextOptions {
                microbatch_tokens,
                ..Default::default()
            };
            let mut split = Context::new(model.clone(), options).unwrap();
            assert_eq!(split.decode_batch(&prompt).unwrap(), expected);
            assert_eq!(split.save_state().unwrap(), one.save_state().unwrap());
        }
        let mut sequential = context(model.clone());
        for (token, expected) in prompt.iter().zip(&expected) {
            let actual = sequential
                .decode_batch(std::slice::from_ref(token))
                .unwrap();
            assert_eq!(actual[0].logits, expected.logits);
            assert_eq!(actual[0].embedding, expected.embedding);
        }
        assert_eq!(sequential.save_state().unwrap(), one.save_state().unwrap());
    }
}

#[test]
fn interleaved_sequences_remain_independent() {
    for model in [llama(), mamba()] {
        let mut mixed = context(model.clone());
        let input = [
            BatchToken::token(5, 0, 1),
            BatchToken::token(9, 0, 4),
            BatchToken::token(5, 1, 2),
            BatchToken::token(9, 1, 3),
        ];
        let actual = mixed.decode_batch(&input).unwrap();
        let mut a = context(model.clone());
        let a = a.decode_batch(&batch(5, &[1, 2])).unwrap();
        let mut b = context(model);
        let b = b.decode_batch(&batch(9, &[4, 3])).unwrap();
        assert_eq!(actual[0].logits, a[0].logits);
        assert_eq!(actual[2].logits, a[1].logits);
        assert_eq!(actual[1].logits, b[0].logits);
        assert_eq!(actual[3].logits, b[1].logits);
        assert_eq!(mixed.next_position(5), Some(2));
        assert_eq!(mixed.next_position(9), Some(2));
    }
}

#[test]
fn late_failure_rolls_back_every_sequence_and_prior_microbatch() {
    for model in [llama(), mamba()] {
        let mut c = Context::new(
            model,
            ContextOptions {
                microbatch_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
        c.decode_batch(&batch(0, &[1])).unwrap();
        let before = c.save_state().unwrap();
        let input = [
            BatchToken::token(0, 1, 2),
            BatchToken::token(1, 0, 3),
            BatchToken::token(0, 2, 999),
        ];
        assert!(matches!(
            c.decode_batch(&input),
            Err(Error::InvalidToken { .. })
        ));
        assert_eq!(c.save_state().unwrap(), before);
        assert_eq!(c.next_position(1), None);
    }
}

#[test]
fn cancellation_inside_layer_and_before_commit_is_atomic() {
    for model in [llama(), mamba()] {
        let mut c = context(model);
        c.decode_batch(&batch(0, &[1])).unwrap();
        let before = c.save_state().unwrap();
        let input = [BatchToken::token(0, 1, 2), BatchToken::token(0, 2, 3)];
        // 9 callbacks: batch, microbatch, token/layer/layer twice, final commit.
        for stop in [1, 4, 8, 9] {
            let mut checks = 0;
            let result = c.decode_batch_with_cancel(&input, || {
                checks += 1;
                checks == stop
            });
            assert_eq!(
                result,
                Err(Error::Cancelled),
                "stop {stop}, checks {checks}"
            );
            assert_eq!(c.save_state().unwrap(), before);
        }
    }
}

#[test]
fn invalid_positions_sequence_ids_and_capacity_do_not_commit() {
    let mut c = Context::new(
        llama(),
        ContextOptions {
            maximum_sequences: 1,
            maximum_batch_tokens: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let before = c.save_state().unwrap();
    assert!(matches!(
        c.decode_batch(&[BatchToken::token(-1, 0, 0)]),
        Err(Error::InvalidSequence(-1))
    ));
    assert!(matches!(
        c.decode_batch(&[BatchToken::token(0, 1, 0)]),
        Err(Error::InvalidPosition { .. })
    ));
    assert!(matches!(
        c.decode_batch(&[BatchToken::token(0, 0, 0), BatchToken::token(1, 0, 0)]),
        Err(Error::SequenceLimit { .. })
    ));
    assert!(c.decode_batch(&batch(0, &[1, 2, 3])).is_err());
    assert_eq!(c.save_state().unwrap(), before);
    assert!(
        Context::new(
            llama(),
            ContextOptions {
                microbatch_tokens: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn context_limit_failure_preserves_prior_state() {
    let original = fixtures::tiny_llama().unwrap();
    let mut config = original.config().clone();
    config.max_context = 2;
    let model = LlamaModel::new(config, original.weights().clone()).unwrap();
    let mut c = context(Arc::new(Model::Llama(model)));
    c.decode_batch(&batch(0, &[1])).unwrap();
    let before = c.save_state().unwrap();
    assert!(matches!(
        c.decode_batch(&[BatchToken::token(0, 1, 2), BatchToken::token(0, 2, 3)]),
        Err(Error::ContextFull { maximum: 2 })
    ));
    assert_eq!(c.save_state().unwrap(), before);
}

#[test]
fn requested_outputs_are_selected_without_dropping_state_updates() {
    let mut c = context(llama());
    let mut input = batch(0, &[1, 2, 3, 4]);
    input[0].output = OutputSelection::None;
    input[1].output = OutputSelection::Embedding;
    input[2].output = OutputSelection::Logits;
    let output = c.decode_batch(&input).unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(output[0].batch_index, 1);
    assert!(output[0].logits.is_none());
    assert!(output[0].embedding.is_some());
    assert!(output[1].logits.is_some());
    assert!(output[1].embedding.is_none());
    assert_eq!(c.next_position(0), Some(4));
}

#[test]
fn supplied_embeddings_match_tokens_and_validate_shape() {
    let model = fixtures::tiny_llama().unwrap();
    let embedding = model.weights().token_embedding.row(2).unwrap().to_vec();
    let model = Arc::new(Model::Llama(model));
    let mut a = context(model.clone());
    let mut b = context(model);
    let expected = a.decode_batch(&[BatchToken::token(0, 0, 2)]).unwrap();
    let input = BatchToken {
        sequence: 0,
        position: 0,
        input: Input::Embedding(embedding),
        output: OutputSelection::Both,
    };
    assert_eq!(b.decode_batch(&[input]).unwrap(), expected);
    let before = b.save_state().unwrap();
    let malformed = BatchToken {
        sequence: 0,
        position: 1,
        input: Input::Embedding(vec![0.0; 3]),
        output: OutputSelection::Both,
    };
    assert!(b.decode_batch(&[malformed]).is_err());
    assert_eq!(b.save_state().unwrap(), before);
}

#[test]
fn copied_prefix_diverges_without_mutating_original_sequence() {
    for model in [llama(), mamba()] {
        let mut c = context(model.clone());
        c.decode_batch(&batch(0, &[1, 2])).unwrap();
        c.fork_sequence(0, 7).unwrap();
        let result = c.decode_batch(&[BatchToken::token(7, 2, 3)]).unwrap();
        assert_eq!(c.next_position(0), Some(2));
        let mut reference = context(model);
        let reference = reference.decode_batch(&batch(7, &[1, 2, 3])).unwrap();
        assert_eq!(result[0].logits, reference[2].logits);
        assert!(matches!(
            c.fork_sequence(0, 7),
            Err(Error::SequenceExists(7))
        ));
        assert!(c.remove_sequence(7));
        assert!(!c.remove_sequence(7));
    }
}

#[test]
fn kv_suffix_truncation_matches_recomputed_prefix() {
    let model = llama();
    let mut c = context(model.clone());
    c.decode_batch(&batch(0, &[1, 2, 3, 4])).unwrap();
    c.truncate_sequence(0, 2).unwrap();
    let actual = c.decode_batch(&[BatchToken::token(0, 2, 5)]).unwrap();
    let mut reference = context(model);
    let expected = reference.decode_batch(&batch(0, &[1, 2, 5])).unwrap();
    assert_eq!(actual[0].logits, expected[2].logits);
    assert_eq!(c.save_state().unwrap(), reference.save_state().unwrap());
}

#[test]
fn recurrent_rejection_uses_checkpoint_and_rejects_fake_inverse() {
    let mut c = context(mamba());
    c.decode_batch(&batch(0, &[1, 2])).unwrap();
    let checkpoint = c.checkpoint().unwrap();
    let state = c.save_state().unwrap();
    c.decode_batch(&[BatchToken::token(0, 2, 3)]).unwrap();
    assert!(matches!(
        c.truncate_sequence(0, 2),
        Err(Error::Unsupported(_))
    ));
    c.restore_checkpoint(&checkpoint).unwrap();
    assert_eq!(c.save_state().unwrap(), state);
}

#[test]
fn state_roundtrip_resumes_both_attention_and_recurrence() {
    for model in [llama(), mamba()] {
        let mut a = context(model.clone());
        a.decode_batch(&batch(0, &[1, 2, 3])).unwrap();
        a.fork_sequence(0, 8).unwrap();
        let bytes = a.save_state().unwrap();
        let mut b = context(model);
        b.restore_state(&bytes).unwrap();
        assert_eq!(b.save_state().unwrap(), bytes);
        let continuation = [BatchToken::token(0, 3, 4), BatchToken::token(8, 3, 5)];
        assert_eq!(
            a.decode_batch(&continuation).unwrap(),
            b.decode_batch(&continuation).unwrap()
        );
        assert_eq!(a.save_state().unwrap(), b.save_state().unwrap());
    }
}

#[test]
fn truncated_or_corrupted_state_never_mutates_context() {
    for model in [llama(), mamba()] {
        let mut c = context(model);
        c.decode_batch(&batch(0, &[1, 2])).unwrap();
        let bytes = c.save_state().unwrap();
        for length in 0..bytes.len() {
            assert!(c.restore_state(&bytes[..length]).is_err());
        }
        for offset in 0..bytes.len() {
            let mut corrupt = bytes.clone();
            corrupt[offset] ^= 0x80;
            assert!(c.restore_state(&corrupt).is_err());
        }
        assert_eq!(c.save_state().unwrap(), bytes);
    }
}

fn crc(bytes: &[u8]) -> u32 {
    let mut x = 0xffff_ffffu32;
    for b in bytes {
        x ^= *b as u32;
        for _ in 0..8 {
            x = (x >> 1) ^ (0xedb8_8320u32 & 0u32.wrapping_sub(x & 1));
        }
    }
    !x
}
fn recheck(bytes: &mut [u8]) {
    let end = bytes.len() - 4;
    let crc = crc(&bytes[..end]);
    bytes[end..].copy_from_slice(&crc.to_le_bytes());
}

#[test]
fn checked_state_decoder_rejects_forged_shapes_counts_and_nonfinite() {
    let mut c = context(llama());
    c.decode_batch(&batch(0, &[1])).unwrap();
    let bytes = c.save_state().unwrap();
    for (offset, value) in [
        (32, u32::MAX),
        (36, u32::MAX),
        (49, u32::MAX),
        (53, u32::MAX),
        (57, f32::NAN.to_bits()),
    ] {
        let mut corrupt = bytes.clone();
        corrupt[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        recheck(&mut corrupt);
        assert!(
            c.restore_state(&corrupt).is_err(),
            "accepted corrupt field at {offset}"
        );
    }
    let mut trailing = bytes[..bytes.len() - 4].to_vec();
    trailing.push(0);
    let checksum = crc(&trailing);
    trailing.extend(checksum.to_le_bytes());
    assert!(c.restore_state(&trailing).is_err());
    assert_eq!(c.save_state().unwrap(), bytes);
}

#[test]
fn state_rejects_different_model_weights_or_adapters() {
    let mut c = context(llama());
    c.decode_batch(&batch(0, &[1])).unwrap();
    let bytes = c.save_state().unwrap();
    let checkpoint = c.checkpoint().unwrap();
    let original = fixtures::tiny_llama().unwrap();
    let mut w = original.weights().clone();
    w.output_norm[0] += 0.25;
    let mut other = context(Arc::new(Model::Llama(
        LlamaModel::new(original.config().clone(), w).unwrap(),
    )));
    assert!(other.restore_state(&bytes).is_err());
    assert!(other.restore_checkpoint(&checkpoint).is_err());
    let mut adapters = AdapterSet::default();
    adapters.control_vectors.insert(1, vec![0.1; 4]);
    c.replace_adapters_and_reset(adapters).unwrap();
    assert!(c.restore_state(&bytes).is_err());
    assert!(c.restore_checkpoint(&checkpoint).is_err());
}

#[test]
fn lora_rank_scaling_and_composition_have_independent_expected_delta() {
    let original = fixtures::tiny_llama().unwrap();
    let model = Arc::new(Model::Llama(original));
    let mut baseline = context(model.clone());
    let output = baseline.decode_batch(&batch(0, &[1])).unwrap();
    let embedding = output[0].embedding.as_ref().unwrap();
    let logits = output[0].logits.as_ref().unwrap();
    let adapter = Lora {
        target: "output.weight".into(),
        a: Matrix::new(1, 4, vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        b: Matrix::new(7, 1, vec![1.0; 7]).unwrap(),
        alpha: 2.0,
        scale: 0.25,
    };
    let mut adapted = context(model);
    adapted
        .replace_adapters_and_reset(AdapterSet {
            lora: vec![adapter.clone(), adapter],
            ..Default::default()
        })
        .unwrap();
    let out = adapted.decode_batch(&batch(0, &[1])).unwrap();
    assert_eq!(out[0].embedding.as_ref().unwrap(), embedding);
    // Two adapters, each 0.25 * alpha(2) / rank(1): total delta = x[0].
    for (a, b) in out[0].logits.as_ref().unwrap().iter().zip(logits) {
        assert!((a - (b + embedding[0])).abs() < 2e-7);
    }
}

#[test]
fn bad_adapters_leave_configuration_and_state_unchanged() {
    let mut c = context(llama());
    c.decode_batch(&batch(0, &[1])).unwrap();
    let bytes = c.save_state().unwrap();
    let mut adapters = AdapterSet::default();
    adapters.control_vectors.insert(0, vec![1.0; 4]);
    assert!(c.replace_adapters_and_reset(adapters).is_err());
    assert_eq!(c.configuration_generation(), 0);
    assert_eq!(c.save_state().unwrap(), bytes);
    let adapter = Lora {
        target: "missing.weight".into(),
        a: Matrix::zeros(1, 4).unwrap(),
        b: Matrix::zeros(7, 1).unwrap(),
        alpha: 0.0,
        scale: 1.0,
    };
    assert!(
        c.replace_adapters_and_reset(AdapterSet {
            lora: vec![adapter],
            ..Default::default()
        })
        .is_err()
    );
    assert_eq!(c.save_state().unwrap(), bytes);
}

#[test]
fn noncanonical_adapter_layer_names_are_rejected_instead_of_silently_ignored() {
    for (model, field) in [(llama(), "attn_v.weight"), (mamba(), "ssm_in.weight")] {
        let canonical = format!("blk.0.{field}");
        let weight = model.weight(&canonical).unwrap();
        let mut adapter = Lora {
            target: canonical.clone(),
            a: Matrix::zeros(1, weight.cols()).unwrap(),
            b: Matrix::zeros(weight.rows(), 1).unwrap(),
            alpha: 0.0,
            scale: 1.0,
        };
        assert!(
            AdapterSet {
                lora: vec![adapter.clone()],
                ..Default::default()
            }
            .validate(&model)
            .is_ok()
        );
        let mut context = context(model.clone());
        context.decode_batch(&batch(0, &[1])).unwrap();
        let original = context.save_state().unwrap();
        for spelling in ["00", "+0", "000", "0 "] {
            adapter.target = format!("blk.{spelling}.{field}");
            assert!(model.weight(&adapter.target).is_none());
            assert!(
                context
                    .replace_adapters_and_reset(AdapterSet {
                        lora: vec![adapter.clone()],
                        ..Default::default()
                    })
                    .is_err()
            );
            assert_eq!(context.save_state().unwrap(), original);
            assert_eq!(context.configuration_generation(), 0);
        }
    }
}

#[test]
fn valid_control_vector_resets_cache_and_changes_model_output() {
    let mut c = context(llama());
    let before = c.decode_batch(&batch(0, &[1])).unwrap();
    let mut adapters = AdapterSet::default();
    adapters.control_vectors.insert(1, vec![0.5, 0.0, 0.0, 0.0]);
    c.replace_adapters_and_reset(adapters).unwrap();
    assert_eq!(c.sequence_count(), 0);
    assert_eq!(c.configuration_generation(), 1);
    assert_ne!(before, c.decode_batch(&batch(0, &[1])).unwrap());
}

#[test]
fn numeric_overflow_is_an_error_and_cannot_poison_cache() {
    let original = fixtures::tiny_llama().unwrap();
    let mut weights = original.weights().clone();
    weights.layers[1].query = Matrix::new(4, 4, vec![f32::MAX; 16]).unwrap();
    let model = LlamaModel::new(original.config().clone(), weights).unwrap();
    let mut c = context(Arc::new(Model::Llama(model)));
    let before = c.save_state().unwrap();
    let input = BatchToken {
        sequence: 0,
        position: 0,
        input: Input::Embedding(vec![1.0; 4]),
        output: OutputSelection::Both,
    };
    assert!(matches!(c.decode_batch(&[input]), Err(Error::NonFinite(_))));
    assert_eq!(c.save_state().unwrap(), before);
}

#[test]
fn bert_is_bidirectional_and_pooling_preserves_token_results() {
    let m = fixtures::tiny_bert().unwrap();
    let a = m.encode(&[1, 2], Pooling::Mean).unwrap();
    let b = m.encode(&[1, 4], Pooling::Mean).unwrap();
    assert_ne!(a.tokens[0], b.tokens[0]);
    let first = m.encode(&[1, 2], Pooling::First).unwrap();
    let last = m.encode(&[1, 2], Pooling::Last).unwrap();
    assert_eq!(first.pooled.as_ref().unwrap(), &a.tokens[0]);
    assert_eq!(last.pooled.as_ref().unwrap(), &a.tokens[1]);
    for (i, v) in a.pooled.as_ref().unwrap().iter().enumerate() {
        assert!((v - 0.5 * (a.tokens[0][i] + a.tokens[1][i])).abs() < 1e-7);
    }
    assert!(m.encode(&[], Pooling::None).is_err());
    assert!(m.encode_positions(&[1], &[32], Pooling::None).is_err());
    assert_eq!(
        m.encode_positions_with_cancel(&[1], &[0], Pooling::None, || true),
        Err(Error::Cancelled)
    );
}

#[test]
fn bert_mean_pooling_preserves_large_finite_values_without_sum_overflow() {
    let original = fixtures::tiny_bert().unwrap();
    let mut weights = original.weights().clone();
    let last = weights.layers.last_mut().unwrap();
    last.output_norm = vec![0.0; 4];
    last.output_norm_bias = vec![f32::MAX; 4];
    let model = BertModel::new(original.config().clone(), weights).unwrap();
    let output = model.encode(&[1, 2], Pooling::Mean).unwrap();
    assert!(output.tokens.iter().flatten().all(|v| *v == f32::MAX));
    assert_eq!(output.pooled.unwrap(), vec![f32::MAX; 4]);
}

#[test]
fn mamba_optional_dt_b_c_norm_and_one_tap_convolution_execute() {
    let original = fixtures::tiny_mamba().unwrap();
    let mut config = original.config().clone();
    config.normalize_dt_b_c = true;
    config.convolution_width = 1;
    let mut weights = original.weights().clone();
    for layer in &mut weights.layers {
        layer.convolution = Matrix::new(8, 1, vec![1.0; 8]).unwrap();
    }
    let model = Arc::new(Model::Mamba(MambaModel::new(config, weights).unwrap()));
    let mut c = context(model.clone());
    c.decode_batch(&batch(0, &[1, 2, 3])).unwrap();
    let state = c.save_state().unwrap();
    let mut restored = context(model);
    restored.restore_state(&state).unwrap();
    assert_eq!(restored.save_state().unwrap(), state);
}

#[test]
fn smoke_demo_is_real_execution_with_reproducible_state() {
    let report = smoke_demo().unwrap();
    assert!(report.state_roundtrips_match);
    assert_eq!(report.llama_logits.len(), 7);
    assert_eq!(report.mamba_logits.len(), 7);
    assert_eq!(report.bert_mean_embedding.len(), 4);
    assert_eq!(report.executable_f32_slices, 8);
}
