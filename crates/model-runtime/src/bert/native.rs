//! Complete no-cache BERT batches from the frozen native encoder graph.
//! Attention uses the first native-order sequence ID; pooling uses all memberships.
use super::{BertModel, Pooling};
use crate::{Error, Input, NativeBatchToken, NativeExecutionOptions, Result, math};
use llama_quant::flash_attention::{Inputs, Limits, Params, Plan, Selection, View4};
use std::collections::{BTreeMap, BTreeSet};

const MAX_WORK_BYTES: usize = 512 * 1024 * 1024;
const MAX_SCALAR_WORK: u128 = 1 << 36;

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub maximum_sequences: usize,
    pub execution: NativeExecutionOptions,
}
impl Options {
    pub fn validate(self) -> Result<()> {
        let execution = self.execution;
        if self.maximum_sequences == 0
            || self.maximum_sequences > 256
            || execution.threads == 0
            || execution.threads > 256
            || execution.threads_batch == 0
            || execution.threads_batch > 256
        {
            return Err(Error::InvalidConfig(
                "BERT sequence/worker limits must be in 1..=256".into(),
            ));
        }
        if !execution.flash_attention && (execution.threads != 1 || execution.threads_batch != 1) {
            return Err(Error::Unsupported(
                "nonflash multiple-worker BERT execution is not implemented".into(),
            ));
        }
        if execution.flash_attention
            && !cfg!(any(
                all(
                    target_arch = "x86_64",
                    target_os = "windows",
                    target_env = "msvc"
                ),
                all(
                    target_arch = "x86_64",
                    target_os = "linux",
                    target_env = "gnu"
                ),
            ))
        {
            return Err(Error::Unsupported(
                "native BERT flash requires a measured x64 MSVC or GNU CPU profile".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Output {
    pub tokens: Vec<Vec<f32>>,
    /// Ascending native sequence IDs and the actual graph output width.
    /// The C facade applies the source classifier extraction stride separately.
    pub sequences: BTreeMap<i32, Vec<f32>>,
}

fn invalid(message: &str) -> Error {
    Error::InvalidConfig(message.into())
}
fn reserved<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("BERT allocation failed"))?;
    Ok(values)
}
fn zeros<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
    let mut values = reserved(count)?;
    values.resize(count, T::default());
    Ok(values)
}
fn copy(values: &[f32]) -> Result<Vec<f32>> {
    let mut output = reserved(values.len())?;
    output.extend_from_slice(values);
    Ok(output)
}

impl BertModel {
    /// Bound all shape-dependent payload and arithmetic before callers copy input rows.
    /// This method does not inspect or allocate token, embedding or membership data.
    pub fn validate_native_shape(
        &self,
        rows: usize,
        pooling: Pooling,
        options: Options,
    ) -> Result<()> {
        options.validate()?;
        let c = &self.config;
        if rows == 0 || rows > 65_536 {
            return Err(invalid("BERT batch must contain 1..=65536 rows"));
        }
        let n = rows as u128;
        let width = c.width as u128;
        let kv = c.kv_width()? as u128;
        let ff = c.feed_forward as u128;
        let padded = n.div_ceil(256) * 256;
        // Concurrent model-execution payload: old/new hidden rows, Q/K/V,
        // gathered K/V, attention/FFN scratch, pooling, descriptors and masks.
        // Model weights, caller-owned input and allocator bookkeeping are excluded.
        let sequence_count = options.maximum_sequences as u128;
        let classifier_width = if pooling == Pooling::Rank && self.classifier.output.is_some() {
            self.classifier.output_count as u128
        } else {
            width
        };
        let memberships = (n * sequence_count).min(1_048_576);
        let payload = n * (width * 8 + kv * 6 + ff * 2 + 256) * 4
            + memberships * 64
            + sequence_count * classifier_width * 4
            + options.maximum_sequences as u128 * (width * 3 + 256) * 8
            + if options.execution.flash_attention {
                padded * n * 2 + padded * kv * 4
            } else {
                padded * 4
            };
        let matrix_work = n * (width * width * 2 + width * kv * 2 + width * ff * 2);
        let attention_work = n * n * width * 2;
        let classifier_work = if pooling == Pooling::Rank {
            sequence_count * (width * width + classifier_width * width)
        } else {
            0
        };
        if payload > MAX_WORK_BYTES as u128
            || (matrix_work + attention_work) * self.weights.layers.len() as u128 + classifier_work
                > MAX_SCALAR_WORK
        {
            return Err(Error::Unsupported(
                "BERT batch exceeds the 512 MiB payload or 2^36 scalar-work bound".into(),
            ));
        }
        if pooling == Pooling::Rank
            && self.classifier.output.is_none()
            && self.classifier.output_count > c.width
        {
            return Err(Error::Unsupported(
                "BERT classifier extraction exceeds the graph width without an output head".into(),
            ));
        }
        Ok(())
    }

    fn native_plan(
        &self,
        batch: &[NativeBatchToken],
        pooling: Pooling,
        options: Options,
    ) -> Result<BTreeMap<i32, Vec<usize>>> {
        self.validate_native_shape(batch.len(), pooling, options)?;
        let c = &self.config;
        let mut rows = BTreeMap::<i32, Vec<usize>>::new();
        let mut positions = BTreeMap::<i32, BTreeSet<i32>>::new();
        let mut previous = BTreeMap::<i32, i32>::new();
        let mut memberships = 0usize;
        for (index, item) in batch.iter().enumerate() {
            if item.position < 0 || item.position as usize >= c.max_positions {
                return Err(invalid(
                    "BERT position is outside the learned position table",
                ));
            }
            if item.sequences.is_empty()
                || item.sequences.len() > options.maximum_sequences
                || item
                    .sequences
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != item.sequences.len()
            {
                return Err(invalid(
                    "BERT sequence memberships must be nonempty and unique in native iteration order",
                ));
            }
            memberships = memberships
                .checked_add(item.sequences.len())
                .ok_or_else(|| invalid("BERT membership overflow"))?;
            if memberships > 1_048_576 {
                return Err(Error::Unsupported(
                    "BERT batch exceeds one million sequence memberships".into(),
                ));
            }
            match &item.input {
                Input::Token(token) if *token as usize >= c.vocabulary => {
                    return Err(Error::InvalidToken {
                        token: *token,
                        vocabulary: c.vocabulary,
                    });
                }
                Input::Embedding(values) => math::vector(values, c.width, "BERT direct embedding")?,
                Input::Token(_) => {}
            }
            for &sequence in &item.sequences {
                if sequence < 0 || sequence as usize >= options.maximum_sequences {
                    return Err(invalid(
                        "BERT sequence ID exceeds configured sequence count",
                    ));
                }
                if previous
                    .insert(sequence, item.position)
                    .is_some_and(|value| item.position < value)
                {
                    return Err(invalid("BERT positions decrease within a sequence"));
                }
                rows.entry(sequence).or_default().push(index);
                positions.entry(sequence).or_default().insert(item.position);
            }
        }
        for set in positions.values() {
            let first = *set
                .first()
                .ok_or_else(|| invalid("empty BERT position set"))?;
            let last = *set
                .last()
                .ok_or_else(|| invalid("empty BERT position set"))?;
            if (last - first) as usize + 1 != set.len() {
                return Err(invalid("BERT positions contain a gap within a sequence"));
            }
        }
        Ok(rows)
    }

    /// Validate input and bounded execution requirements before changing C outputs.
    pub fn validate_native_batch(
        &self,
        batch: &[NativeBatchToken],
        pooling: Pooling,
        options: Options,
    ) -> Result<()> {
        self.native_plan(batch, pooling, options).map(|_| ())
    }

    /// Execute one complete, noncausal batch without decoder KV memory.
    pub fn encode_native_with_cancel(
        &self,
        batch: &[NativeBatchToken],
        pooling: Pooling,
        options: Options,
        mut cancel: impl FnMut() -> bool,
    ) -> Result<Output> {
        let sequence_rows = self.native_plan(batch, pooling, options)?;
        let c = &self.config;
        let n = batch.len();
        let kv_width = c.kv_width()?;
        let head_dim = c.head_dim()?;
        let row_len = n.div_ceil(256) * 256;
        let mut hidden = reserved(n)?;
        for item in batch {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let mut embedding = match &item.input {
                Input::Token(token) => copy(self.weights.token_embedding.row(*token as usize)?)?,
                Input::Embedding(values) => copy(values)?,
            };
            if let Some(types) = &self.weights.token_type_embedding {
                math::add(&mut embedding, types.row(0)?);
            }
            math::add(
                &mut embedding,
                self.weights
                    .position_embedding
                    .row(item.position as usize)?,
            );
            hidden.push(math::layer_norm(
                &embedding,
                &self.weights.embedding_norm,
                &self.weights.embedding_norm_bias,
                c.layer_norm_epsilon,
            )?);
        }
        let mut attention_groups = BTreeMap::<i32, Vec<usize>>::new();
        for (index, item) in batch.iter().enumerate() {
            attention_groups
                .entry(item.sequences[0])
                .or_default()
                .push(index);
        }
        let scale = (head_dim as f32).sqrt().recip();
        for layer in &self.weights.layers {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let mut queries = reserved(n * c.width)?;
            let mut keys = reserved(n * kv_width)?;
            let mut values = reserved(n * kv_width)?;
            for input in &hidden {
                let mut query = layer.query.multiply(input)?;
                math::bias(&mut query, &layer.query_bias);
                queries.extend(query);
                let mut key = layer.key.multiply(input)?;
                math::bias(&mut key, &layer.key_bias);
                keys.extend(key);
                let mut value = layer.value.multiply(input)?;
                math::bias(&mut value, &layer.value_bias);
                values.extend(value);
            }
            let attention = if options.execution.flash_attention {
                let mut keys_half = zeros(row_len * kv_width)?;
                let mut values_half = zeros(row_len * kv_width)?;
                for (to, from) in keys_half.iter_mut().zip(&keys) {
                    *to = llama_quant::f32_to_f16(*from);
                }
                for (to, from) in values_half.iter_mut().zip(&values) {
                    *to = llama_quant::f32_to_f16(*from);
                }
                let mut mask = reserved(row_len * n)?;
                mask.resize(row_len * n, 0xfc00u16);
                for (query, item) in batch.iter().enumerate() {
                    for &column in &attention_groups[&item.sequences[0]] {
                        mask[query * row_len + column] = 0;
                    }
                }
                let tensor_error = |error: llama_quant::flash_attention::Error| {
                    Error::InvalidState(error.to_string())
                };
                let q = View4::from_f32(
                    &queries,
                    [head_dim, n, c.heads, 1],
                    [4, c.width * 4, head_dim * 4, n * c.width * 4],
                    0,
                )
                .map_err(tensor_error)?;
                let k = View4::from_f16(
                    &keys_half,
                    [head_dim, row_len, c.kv_heads, 1],
                    [2, kv_width * 2, head_dim * 2, row_len * kv_width * 2],
                    0,
                )
                .map_err(tensor_error)?;
                let v = View4::from_f16(
                    &values_half,
                    [head_dim, row_len, c.kv_heads, 1],
                    [2, kv_width * 2, head_dim * 2, row_len * kv_width * 2],
                    0,
                )
                .map_err(tensor_error)?;
                let mask = View4::from_f16(
                    &mask,
                    [row_len, n, 1, 1],
                    [2, row_len * 2, row_len * n * 2, row_len * n * 2],
                    0,
                )
                .map_err(tensor_error)?;
                let plan = Plan::new(
                    Inputs {
                        q,
                        k,
                        v,
                        mask: Some(mask),
                        sinks: None,
                    },
                    Params {
                        scale,
                        workers: if n == 1 {
                            options.execution.threads
                        } else {
                            options.execution.threads_batch
                        },
                        ..Params::default()
                    },
                )
                .map_err(tensor_error)?;
                // The flash engine accounts for its own output, worker scratch and
                // requested thread stacks; its allocation bound is independent.
                plan.execute(
                    Selection::All,
                    Limits {
                        max_bytes: MAX_WORK_BYTES,
                        ..Limits::default()
                    },
                )
                .map_err(tensor_error)?
                .values
            } else {
                let mut attention = zeros(n * c.width)?;
                for columns in attention_groups.values() {
                    let mut group_keys = reserved(columns.len() * kv_width)?;
                    let mut group_values = reserved(columns.len() * kv_width)?;
                    for &column in columns {
                        group_keys
                            .extend_from_slice(&keys[column * kv_width..(column + 1) * kv_width]);
                        group_values
                            .extend_from_slice(&values[column * kv_width..(column + 1) * kv_width]);
                    }
                    for &query in columns {
                        if cancel() {
                            return Err(Error::Cancelled);
                        }
                        let output = math::attention_with_layout(
                            &queries[query * c.width..(query + 1) * c.width],
                            &group_keys,
                            &group_values,
                            c.heads,
                            c.kv_heads,
                            head_dim,
                            scale,
                            math::AttentionLayout {
                                row_len,
                                columns: Some(columns),
                            },
                        )?;
                        attention[query * c.width..(query + 1) * c.width].copy_from_slice(&output);
                    }
                }
                attention
            };
            let mut next = reserved(n)?;
            for (index, input) in hidden.iter().enumerate() {
                if cancel() {
                    return Err(Error::Cancelled);
                }
                let mut out = layer
                    .attention_output
                    .multiply(&attention[index * c.width..(index + 1) * c.width])?;
                math::bias(&mut out, &layer.attention_output_bias);
                math::add(&mut out, input);
                let norm = math::layer_norm(
                    &out,
                    &layer.attention_norm,
                    &layer.attention_norm_bias,
                    c.layer_norm_epsilon,
                )?;
                let mut up = layer.up.multiply(&norm)?;
                math::bias(&mut up, &layer.up_bias);
                for value in &mut up {
                    *value = math::gelu(*value);
                }
                let mut ffn = layer.down.multiply(&up)?;
                math::bias(&mut ffn, &layer.down_bias);
                math::add(&mut ffn, &norm);
                let output = math::layer_norm(
                    &ffn,
                    &layer.output_norm,
                    &layer.output_norm_bias,
                    c.layer_norm_epsilon,
                )?;
                math::finite(&output, "BERT native layer output")?;
                next.push(output);
            }
            hidden = next;
        }
        if cancel() {
            return Err(Error::Cancelled);
        }
        let mut sequences = BTreeMap::new();
        if pooling != Pooling::None {
            let selected_pooling = if pooling == Pooling::Rank {
                if self.classifier.mean_pooling {
                    Pooling::Mean
                } else {
                    Pooling::First
                }
            } else {
                pooling
            };
            for (sequence, rows) in sequence_rows {
                let pooled = match selected_pooling {
                    Pooling::Mean => {
                        let reciprocal = 1.0f32 / rows.len() as f32;
                        let mut sums = zeros::<f64>(c.width)?;
                        for &row in &rows {
                            for (sum, value) in sums.iter_mut().zip(&hidden[row]) {
                                *sum += f64::from(*value * reciprocal);
                            }
                        }
                        sums.into_iter().map(|value| value as f32).collect()
                    }
                    Pooling::Last => {
                        let row = rows
                            .iter()
                            .copied()
                            .max_by_key(|&row| batch[row].position)
                            .ok_or_else(|| invalid("empty BERT pooling sequence"))?;
                        copy(&hidden[row])?
                    }
                    _ => {
                        let row = rows
                            .iter()
                            .copied()
                            .min_by_key(|&row| batch[row].position)
                            .ok_or_else(|| invalid("empty BERT pooling sequence"))?;
                        copy(&hidden[row])?
                    }
                };
                let pooled = if pooling == Pooling::Rank {
                    self.classify(pooled)?
                } else {
                    pooled
                };
                math::finite(&pooled, "BERT native pooled output")?;
                sequences.insert(sequence, pooled);
            }
        }
        Ok(Output {
            tokens: hidden,
            sequences,
        })
    }
}
