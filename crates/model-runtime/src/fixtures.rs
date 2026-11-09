//! Deterministic synthetic weights used for structural/numerical validation.
//! These are not trained models, a tokenizer, or a meaningful text generator.
use crate::*;
use std::sync::Arc;

pub fn values(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|i| (((i * 17 + seed * 13) % 31) as i32 - 15) as f32 / 64.0)
        .collect()
}
pub fn norms(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 1.0 + (((i * 3 + seed) % 7) as i32 - 3) as f32 / 64.0)
        .collect()
}
fn matrix(rows: usize, cols: usize, seed: usize) -> Result<Matrix> {
    Matrix::new(rows, cols, values(rows * cols, seed))
}
fn bias(length: usize, seed: usize) -> Option<Vec<f32>> {
    Some(values(length, seed).into_iter().map(|x| x / 8.0).collect())
}

pub fn tiny_llama() -> Result<LlamaModel> {
    let config = LlamaConfig {
        vocabulary: 7,
        width: 4,
        feed_forward: 6,
        heads: 2,
        kv_heads: 1,
        max_context: 32,
        rms_epsilon: 1e-5,
        rope_base: 10000.0,
        attention_scale: None,
    };
    let mut layers = Vec::new();
    for index in 0..2 {
        let seed = index * 20;
        layers.push(LlamaLayer {
            attention_norm: norms(4, seed + 2),
            query: matrix(4, 4, seed + 3)?,
            key: matrix(2, 4, seed + 4)?,
            value: matrix(2, 4, seed + 5)?,
            attention_output: matrix(4, 4, seed + 6)?,
            query_bias: bias(4, seed + 7),
            key_bias: bias(2, seed + 8),
            value_bias: bias(2, seed + 9),
            attention_output_bias: bias(4, seed + 10),
            feed_forward_norm: norms(4, seed + 11),
            gate: matrix(6, 4, seed + 12)?,
            up: matrix(6, 4, seed + 13)?,
            down: matrix(4, 6, seed + 14)?,
            gate_bias: bias(6, seed + 15),
            up_bias: bias(6, seed + 16),
            down_bias: bias(4, seed + 17),
        });
    }
    LlamaModel::new(
        config,
        LlamaWeights {
            token_embedding: matrix(7, 4, 1)?,
            layers,
            output_norm: norms(4, 50),
            output: Some(matrix(7, 4, 51)?),
        },
    )
}

pub fn tiny_mamba() -> Result<MambaModel> {
    let config = MambaConfig {
        vocabulary: 7,
        width: 4,
        inner: 8,
        state_size: 3,
        convolution_width: 3,
        time_step_rank: 2,
        max_context: 32,
        rms_epsilon: 1e-5,
        normalize_dt_b_c: false,
    };
    let mut layers = Vec::new();
    for index in 0..2 {
        let seed = index * 20;
        let a: Vec<f32> = (0..24)
            .map(|i| -0.1 - ((i + seed) % 9) as f32 / 16.0)
            .collect();
        layers.push(MambaLayer {
            norm: norms(4, seed + 2),
            input: matrix(16, 4, seed + 3)?,
            convolution: matrix(8, 3, seed + 4)?,
            convolution_bias: bias(8, seed + 5).expect("fixture bias"),
            x_projection: matrix(8, 8, seed + 6)?,
            dt_projection: matrix(8, 2, seed + 7)?,
            dt_bias: values(8, seed + 8),
            a: Matrix::new(8, 3, a)?,
            d: norms(8, seed + 9),
            output: matrix(4, 8, seed + 10)?,
            dt_norm: None,
            b_norm: None,
            c_norm: None,
        });
    }
    MambaModel::new(
        config,
        MambaWeights {
            token_embedding: matrix(7, 4, 1)?,
            layers,
            output_norm: norms(4, 50),
            output: Some(matrix(7, 4, 51)?),
        },
    )
}

pub fn tiny_bert() -> Result<BertModel> {
    let config = BertConfig {
        vocabulary: 7,
        width: 4,
        feed_forward: 6,
        heads: 2,
        kv_heads: 2,
        max_positions: 32,
        layer_norm_epsilon: 1e-5,
    };
    let mut layers = Vec::new();
    for index in 0..2 {
        let seed = index * 20;
        layers.push(BertLayer {
            query: matrix(4, 4, seed + 3)?,
            key: matrix(4, 4, seed + 4)?,
            value: matrix(4, 4, seed + 5)?,
            query_bias: bias(4, seed + 7),
            key_bias: bias(4, seed + 8),
            value_bias: bias(4, seed + 9),
            attention_output: matrix(4, 4, seed + 6)?,
            attention_output_bias: bias(4, seed + 10),
            attention_norm: norms(4, seed + 11),
            attention_norm_bias: bias(4, seed + 12).expect("fixture bias"),
            up: matrix(6, 4, seed + 13)?,
            up_bias: bias(6, seed + 14),
            down: matrix(4, 6, seed + 15)?,
            down_bias: bias(4, seed + 16),
            output_norm: norms(4, seed + 17),
            output_norm_bias: bias(4, seed + 18).expect("fixture bias"),
        });
    }
    BertModel::new(
        config,
        BertWeights {
            token_embedding: matrix(7, 4, 1)?,
            position_embedding: matrix(32, 4, 60)?,
            token_type_embedding: Some(matrix(2, 4, 61)?),
            embedding_norm: norms(4, 62),
            embedding_norm_bias: bias(4, 63).expect("fixture bias"),
            layers,
        },
    )
}

#[derive(Debug, Clone)]
pub struct SmokeReport {
    pub architecture_slots: usize,
    pub executable_f32_slices: usize,
    pub llama_logits: Vec<f32>,
    pub mamba_logits: Vec<f32>,
    pub bert_mean_embedding: Vec<f32>,
    pub llama_state_bytes: usize,
    pub mamba_state_bytes: usize,
    pub state_roundtrips_match: bool,
}
pub fn smoke_demo() -> Result<SmokeReport> {
    let prompt: Vec<_> = [1, 2, 3]
        .iter()
        .enumerate()
        .map(|(p, t)| BatchToken::token(0, p, *t))
        .collect();
    let options = ContextOptions {
        microbatch_tokens: 2,
        ..Default::default()
    };
    let llama = Arc::new(Model::Llama(tiny_llama()?));
    let mut llama_context = Context::new(llama.clone(), options.clone())?;
    let llama_outputs = llama_context.decode_batch(&prompt)?;
    let llama_state = llama_context.save_state()?;
    let expected = llama_context.decode_batch(&[BatchToken::token(0, 3, 4)])?;
    let mut restored = Context::new(llama, options.clone())?;
    restored.restore_state(&llama_state)?;
    let llama_match = expected == restored.decode_batch(&[BatchToken::token(0, 3, 4)])?;
    let mamba = Arc::new(Model::Mamba(tiny_mamba()?));
    let mut mamba_context = Context::new(mamba.clone(), options.clone())?;
    let mamba_outputs = mamba_context.decode_batch(&prompt)?;
    let mamba_state = mamba_context.save_state()?;
    let expected = mamba_context.decode_batch(&[BatchToken::token(0, 3, 4)])?;
    let mut restored = Context::new(mamba, options)?;
    restored.restore_state(&mamba_state)?;
    let mamba_match = expected == restored.decode_batch(&[BatchToken::token(0, 3, 4)])?;
    Ok(SmokeReport {
        architecture_slots: registry::ARCHITECTURES.len(),
        executable_f32_slices: registry::ARCHITECTURES
            .iter()
            .filter(|a| a.executable_slice.is_some())
            .count(),
        llama_logits: llama_outputs
            .last()
            .and_then(|o| o.logits.clone())
            .expect("output requested"),
        mamba_logits: mamba_outputs
            .last()
            .and_then(|o| o.logits.clone())
            .expect("output requested"),
        bert_mean_embedding: tiny_bert()?
            .encode(&[1, 2, 3], Pooling::Mean)?
            .pooled
            .expect("pooling requested"),
        llama_state_bytes: llama_state.len(),
        mamba_state_bytes: mamba_state.len(),
        state_roundtrips_match: llama_match && mamba_match,
    })
}
