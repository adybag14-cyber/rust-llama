//! Layer-wise native microbatches. All K/V writes for a layer precede attention.
//!
//! The full physical cache row is retained, including holes and masked cells.
//! The caller-owned input/model and the existing caller stack are outside the
//! owned working-set ledger. Every heap payload created here is preflighted.
use super::native_batch::{Microbatch, Planner};
use super::native_kv::KvBuffer;
use super::native_row_map::{self, RowMap};
use super::{
    Cell, MAX_NATIVE_WORK_BYTES, NativeBatchToken, NativeContext, NativeExecutionStats,
    copy_values, invalid,
};
use crate::{Error, Input, Model, OutputSelection, Result, TokenOutput, math};
use std::mem::size_of;

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("native decode byte addition overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("native decode byte multiplication overflow"))
}
fn zeros<T: Default + Clone>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("native microbatch allocation failed"))?;
    values.resize(count, T::default());
    Ok(values)
}

/// This bounds simultaneous owned buffers independently of how many early
/// microbatches have executed. Full physical cache capacity bounds later rows.
fn owned_work(context: &NativeContext, batch: &[NativeBatchToken]) -> Result<usize> {
    let Model::Llama(model) = context.model.as_ref() else {
        return Err(invalid("native layer executor requires Llama"));
    };
    let options = &context.options;
    let chunk = Planner::maximum_chunk_tokens(
        batch.len(),
        options.microbatch_tokens,
        options.unified,
        context.execution.embeddings,
    )?;
    let streams = if options.unified || context.execution.embeddings {
        1
    } else {
        chunk.min(context.streams.len())
    };
    let width = model.config().width;
    let mut bytes = Planner::heap_bytes(
        batch.len(),
        options.microbatch_tokens,
        options.unified,
        context.execution.embeddings,
    )?;
    // One stored slot index per query and a temporary find_slots result.
    bytes = add(bytes, mul(chunk, 2 * size_of::<usize>())?)?;
    // The hidden matrix and the complete current-layer query matrix coexist.
    bytes = add(bytes, mul(mul(chunk, width)?, 8)?)?;
    if context.execution.flash_attention {
        if options.unified {
            // Every logical Q slice owns its full I32 physical-index map.
            // Bound all later simple microbatches before cloning or allocation.
            bytes = add(
                bytes,
                native_row_map::maximum_heap_bytes(
                    batch,
                    options.microbatch_tokens,
                    context.capacity,
                )?,
            )?;
        }
        // Source dense Llama casts both complete physical K and V views to F16,
        // including when resident storage is F32. Preserve every masked lane.
        bytes = add(
            bytes,
            mul(mul(mul(context.capacity, context.width)?, streams)?, 4)?,
        )?;
        bytes = add(bytes, mul(mul(context.capacity, chunk)?, 2)?)?;
        // The shared flash executor accounts its output, staged results, worker
        // scratch, descriptors and requested worker stacks separately at call time.
    } else {
        bytes = add(bytes, mul(mul(chunk, width)?, 4)?)?; // all attention rows
        bytes = add(bytes, mul(context.capacity, 4)?)?; // one physical score row
        bytes = add(bytes, mul(model.config().head_dim()?, 8)?)?; // F64 PV sum
    }
    // LoRA rank is independent of model width and may exceed its projections.
    bytes = add(bytes, context.adapters.scratch_bytes()?)?;
    // Per-token projection/normalization/FFN and final vocabulary temporaries.
    // These do not scale with the number of live hidden/query rows above.
    bytes = add(bytes, mul(width, 128)?)?;
    bytes = add(bytes, mul(model.config().feed_forward, 32)?)?;
    bytes = add(bytes, mul(model.config().vocabulary, 16)?)?;
    let mut output_floats = 0usize;
    for item in batch {
        let count = match item.output {
            OutputSelection::None => 0,
            OutputSelection::Logits => context.model.vocabulary(),
            OutputSelection::Embedding => context.model.width(),
            OutputSelection::Both => add(context.model.vocabulary(), context.model.width())?,
        };
        output_floats = add(output_floats, count)?;
    }
    bytes = add(bytes, mul(output_floats, 4)?)?;
    bytes = add(bytes, mul(batch.len(), size_of::<TokenOutput>())?)?;
    // Original + transactional cache are both live throughout this operation.
    add(mul(context.storage_bytes()?, 2)?, bytes)
}

pub(super) fn execute(
    context: &mut NativeContext,
    batch: &[NativeBatchToken],
    mut cancelled: impl FnMut() -> bool,
    work_limit: usize,
    accept_outputs: impl FnOnce(&[TokenOutput]) -> Result<()>,
) -> Result<Vec<TokenOutput>> {
    let limit = work_limit.min(MAX_NATIVE_WORK_BYTES);
    let owned = owned_work(context, batch)?;
    if owned > limit {
        return Err(Error::Unsupported(
            "native microbatch exceeds the two-GiB owned working-set budget".into(),
        ));
    }
    let mut planner = Planner::new(
        batch,
        context.options.microbatch_tokens,
        context.options.unified,
        context.execution.embeddings,
    )?;
    let mut staged = context.try_clone()?;
    staged.apply_pending()?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(batch.len())
        .map_err(|_| invalid("native output descriptor allocation failed"))?;
    let mut statistics = NativeExecutionStats::default();
    while let Some(chunk) = planner.next()? {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        statistics.microbatches += 1;
        staged.execute_chunk(
            batch,
            &chunk,
            &mut outputs,
            &mut cancelled,
            limit - owned,
            &mut statistics,
        )?;
    }
    // Native output IDs are remapped to the original caller order after its
    // sequence-oriented execution order; negative C indices use that same order.
    outputs.sort_unstable_by_key(|row| row.batch_index);
    accept_outputs(&outputs)?;
    if cancelled() {
        return Err(Error::Cancelled);
    }
    staged.last_execution = statistics;
    *context = staged;
    Ok(outputs)
}

impl NativeContext {
    #[allow(clippy::too_many_arguments)] // Explicit immutable batch and owned-work ledger.
    fn execute_chunk(
        &mut self,
        batch: &[NativeBatchToken],
        chunk: &Microbatch,
        outputs: &mut Vec<TokenOutput>,
        cancelled: &mut impl FnMut() -> bool,
        flash_budget: usize,
        statistics: &mut NativeExecutionStats,
    ) -> Result<()> {
        let mut slots = zeros::<usize>(chunk.indices.len())?;
        for selected in 0..chunk.streams {
            let stream = chunk.first_stream + selected;
            let start = selected * chunk.queries_per_stream;
            let allocated = self.find_slots(stream, chunk.queries_per_stream)?;
            for (within, &slot) in allocated.iter().enumerate() {
                let token = &batch[chunk.indices[start + within]];
                slots[start + within] = slot;
                self.streams[stream].cells[slot] = Cell {
                    position: token.position,
                    shift: 0,
                    sequences: token.sequences.iter().copied().collect(),
                };
            }
            self.streams[stream].head = allocated.last().copied().unwrap_or(0) + 1;
        }
        let row_len = (chunk.first_stream..chunk.first_stream + chunk.streams).try_fold(
            0usize,
            |maximum, stream| {
                let used = self.streams[stream]
                    .cells
                    .iter()
                    .rposition(Cell::occupied)
                    .map_or(0, |index| index + 1);
                let padded = used
                    .checked_add(255)
                    .map(|count| count / 256 * 256)
                    .ok_or_else(|| invalid("native physical row padding overflow"))?
                    .max(256)
                    .min(self.capacity);
                Ok::<_, Error>(maximum.max(padded))
            },
        )?;
        let mapped = if self.execution.flash_attention && self.options.unified {
            let start = *chunk
                .indices
                .first()
                .ok_or_else(|| invalid("empty unified microbatch"))?;
            let end = start
                .checked_add(chunk.indices.len())
                .ok_or_else(|| invalid("unified microbatch range overflow"))?;
            let tokens = batch
                .get(start..end)
                .ok_or_else(|| invalid("unified microbatch range"))?;
            if chunk
                .indices
                .iter()
                .enumerate()
                .any(|(index, &original)| original != start + index)
            {
                return Err(invalid(
                    "unified row maps require source simple microbatch order",
                ));
            }
            Some(RowMap::new(&self.streams[0].cells, tokens)?)
        } else {
            None
        };
        let Model::Llama(model) = self.model.as_ref() else {
            return Err(invalid("native layer executor requires Llama"));
        };
        let width = model.config().width;
        let mut hidden = zeros::<f32>(mul(chunk.indices.len(), width)?)?;
        for (within, &original) in chunk.indices.iter().enumerate() {
            let input = match &batch[original].input {
                Input::Token(token) => self.adapters.embedding(*token, model.embedding(*token)?)?,
                Input::Embedding(values) => copy_values(values)?,
            };
            hidden[within * width..(within + 1) * width].copy_from_slice(&input);
        }
        for layer in 0..model.layer_count() {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let mut queries = zeros::<f32>(hidden.len())?;
            // Publish the entire current layer, including future/duplicate
            // positions. Causal masking controls visibility, not write order.
            for (within, (&original, &slot)) in chunk.indices.iter().zip(&slots).enumerate() {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let projected = model.project_layer(
                    layer,
                    &hidden[within * width..(within + 1) * width],
                    batch[original].position as usize,
                    &self.adapters,
                )?;
                queries[within * width..(within + 1) * width].copy_from_slice(&projected.query);
                let stream = chunk.first_stream + within / chunk.queries_per_stream;
                let stored = &mut self.streams[stream].layers[layer];
                stored.keys.write_f32(slot * self.width, &projected.key);
                stored.values.write_f32(slot * self.width, &projected.value);
            }
            let attention = if self.execution.flash_attention {
                self.flash_layer(
                    batch,
                    chunk,
                    layer,
                    row_len,
                    mapped.as_ref(),
                    &queries,
                    flash_budget,
                    statistics,
                )?
            } else {
                self.nonflash_layer(batch, chunk, layer, row_len, &queries)?
            };
            drop(queries);
            for within in 0..chunk.indices.len() {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let result = model.finish_layer(
                    layer,
                    copy_values(&hidden[within * width..(within + 1) * width])?,
                    &attention[within * width..(within + 1) * width],
                    &self.adapters,
                )?;
                hidden[within * width..(within + 1) * width].copy_from_slice(&result);
            }
        }
        for (within, &original) in chunk.indices.iter().enumerate() {
            let item = &batch[original];
            if item.output == OutputSelection::None {
                continue;
            }
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let (embedding, logits) = model.finish_hidden(
                &hidden[within * width..(within + 1) * width],
                &self.adapters,
            )?;
            outputs.push(TokenOutput {
                batch_index: original,
                sequence: item.sequences[0],
                position: item.position as usize,
                logits: matches!(item.output, OutputSelection::Logits | OutputSelection::Both)
                    .then_some(logits),
                embedding: matches!(
                    item.output,
                    OutputSelection::Embedding | OutputSelection::Both
                )
                .then_some(embedding),
            });
        }
        Ok(())
    }

    fn nonflash_layer(
        &self,
        batch: &[NativeBatchToken],
        chunk: &Microbatch,
        layer: usize,
        row_len: usize,
        queries: &[f32],
    ) -> Result<Vec<f32>> {
        let Model::Llama(model) = self.model.as_ref() else {
            return Err(invalid("native attention model mismatch"));
        };
        let config = model.config();
        let head_dim = config.head_dim()?;
        let scale = config
            .attention_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());
        let mut result = zeros::<f32>(queries.len())?;
        let mut scores = zeros::<f32>(row_len)?;
        let mut sum = zeros::<f64>(head_dim)?;
        for (within, &original) in chunk.indices.iter().enumerate() {
            let item = &batch[original];
            let stream = &self.streams[chunk.first_stream + within / chunk.queries_per_stream];
            let cache = &stream.layers[layer];
            for head in 0..config.heads {
                let kv_head = head / (config.heads / config.kv_heads);
                let offset = within * config.width + head * head_dim;
                let query = &queries[offset..offset + head_dim];
                for (column, score) in scores.iter_mut().enumerate() {
                    let cache_offset = column * self.width + kv_head * head_dim;
                    let mut dot = 0.0f64;
                    for (channel, &value) in query.iter().enumerate() {
                        dot += f64::from(cache.keys.get(cache_offset + channel) * value);
                    }
                    let cell = &stream.cells[column];
                    let mask = if cell.occupied()
                        && cell.position <= item.position
                        && cell.sequences.contains(&item.sequences[0])
                    {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    };
                    // Source applies mask after the complete physical KQ
                    // multiplication, preserving masked exceptional operands.
                    *score = (dot as f32) * scale + mask;
                }
                math::softmax(&mut scores)?;
                sum.fill(0.0);
                for (column, &probability) in scores.iter().enumerate() {
                    let cache_offset = column * self.width + kv_head * head_dim;
                    for (channel, value) in sum.iter_mut().enumerate() {
                        *value += f64::from(cache.values.get(cache_offset + channel) * probability);
                    }
                }
                for (target, &value) in result[offset..offset + head_dim].iter_mut().zip(&sum) {
                    *target = value as f32;
                }
            }
        }
        // The compatibility context retains its documented checked finite
        // arithmetic boundary, rather than publishing exceptional native logits.
        math::finite(&result, "native nonflash attention output")?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)] // Original geometry, physical storage and caller budget.
    fn flash_layer(
        &self,
        batch: &[NativeBatchToken],
        chunk: &Microbatch,
        layer: usize,
        row_len: usize,
        mapped: Option<&RowMap>,
        queries: &[f32],
        budget: usize,
        statistics: &mut NativeExecutionStats,
    ) -> Result<Vec<f32>> {
        use llama_quant::flash_attention::{
            Inputs, KvRows, Limits, Params, Plan, Selection, View4,
        };
        let Model::Llama(model) = self.model.as_ref() else {
            return Err(invalid("native flash model mismatch"));
        };
        let config = model.config();
        let head_dim = config.head_dim()?;
        let elements = mul(mul(row_len, self.width)?, chunk.streams)?;
        let mut keys = zeros::<u16>(elements)?;
        let mut values = zeros::<u16>(elements)?;
        for selected in 0..chunk.streams {
            let stored = &self.streams[chunk.first_stream + selected].layers[layer];
            let start = selected * row_len * self.width;
            let end = start + row_len * self.width;
            for (source, target) in [
                (&stored.keys, &mut keys[start..end]),
                (&stored.values, &mut values[start..end]),
            ] {
                match source {
                    KvBuffer::F16(raw) => target.copy_from_slice(&raw[..target.len()]),
                    KvBuffer::F32(raw) => {
                        for (to, &from) in target.iter_mut().zip(raw) {
                            *to = llama_quant::f32_to_f16(from);
                        }
                    }
                }
            }
        }
        let columns = mapped.map_or(row_len, |rows| rows.columns);
        let query_batches = mapped.map_or(chunk.streams, |rows| rows.batches);
        let queries_per_batch = mapped.map_or(chunk.queries_per_stream, |rows| rows.queries);
        let mut mask = zeros::<u16>(mul(columns, chunk.indices.len())?)?;
        for (within, &original) in chunk.indices.iter().enumerate() {
            let token = &batch[original];
            let stream = &self.streams[chunk.first_stream + within / chunk.queries_per_stream];
            for (column, target) in mask[within * columns..(within + 1) * columns]
                .iter_mut()
                .enumerate()
            {
                let physical = if let Some(rows) = mapped {
                    let index = rows.indices[(within / queries_per_batch) * columns + column];
                    if index < 0 {
                        *target = 0xfc00;
                        continue;
                    }
                    index as usize
                } else {
                    column
                };
                let cell = &stream.cells[physical];
                *target = if cell.occupied()
                    && cell.position <= token.position
                    && cell.sequences.contains(&token.sequences[0])
                {
                    0
                } else {
                    0xfc00
                };
            }
        }
        let q_stride = mul(config.width, 4)?;
        let kv_stride = mul(self.width, 2)?;
        let q = View4::from_f32(
            queries,
            [head_dim, queries_per_batch, config.heads, query_batches],
            [
                4,
                q_stride,
                mul(head_dim, 4)?,
                mul(q_stride, queries_per_batch)?,
            ],
            0,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let cache_shape = [head_dim, row_len, config.kv_heads, chunk.streams];
        let cache_strides = [2, kv_stride, mul(head_dim, 2)?, mul(kv_stride, row_len)?];
        let k = View4::from_f16(&keys, cache_shape, cache_strides, 0)
            .map_err(|e| invalid(e.to_string()))?;
        let v = View4::from_f16(&values, cache_shape, cache_strides, 0)
            .map_err(|e| invalid(e.to_string()))?;
        let mask_stride = mul(columns, 2)?;
        let masks = View4::from_f16(
            &mask,
            [columns, queries_per_batch, 1, query_batches],
            [
                2,
                mask_stride,
                mul(mask_stride, queries_per_batch)?,
                mul(mask_stride, queries_per_batch)?,
            ],
            0,
        )
        .map_err(|e| invalid(e.to_string()))?;
        // Source chooses batch workers by actual microbatch token count, not
        // caller batch size or the number of queries in each separate stream.
        let workers = if chunk.indices.len() > 1 {
            self.execution.threads_batch
        } else {
            self.execution.threads
        };
        let inputs = Inputs {
            q,
            k,
            v,
            mask: Some(masks),
            sinks: None,
        };
        let params = Params {
            scale: config
                .attention_scale
                .unwrap_or_else(|| (head_dim as f32).sqrt().recip()),
            workers,
            ..Params::default()
        };
        let plan = if let Some(rows) = mapped {
            let rows = KvRows::from_i32(&rows.indices, [rows.columns, rows.batches], 0)
                .map_err(|error| invalid(error.to_string()))?;
            Plan::new_with_rows(inputs, rows, params)
        } else {
            Plan::new(inputs, params)
        }
        .map_err(|error| invalid(error.to_string()))?;
        let output = plan
            .execute(
                Selection::All,
                Limits {
                    max_bytes: budget,
                    ..Limits::default()
                },
            )
            .map_err(|e| Error::Unsupported(format!("native flash execution failed: {e}")))?;
        statistics.flash_graphs += 1;
        statistics.configured_worker_partitions += output.stats.completed_partitions;
        statistics.threads_executing_partitions += output.stats.worker_threads;
        math::finite(&output.values, "native flash attention output")?;
        Ok(output.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeContextOptions, NativeExecutionOptions, NativeKvType};
    use std::sync::Arc;

    fn flash_context(threads: usize, threads_batch: usize) -> NativeContext {
        NativeContext::new_with_execution(
            Arc::new(Model::Llama(crate::fixtures::tiny_llama().unwrap())),
            NativeContextOptions {
                context_tokens: 1024,
                maximum_sequences: 1,
                microbatch_tokens: 128,
                maximum_batch_tokens: 512,
                unified: false,
            },
            NativeKvType::F16,
            NativeKvType::F16,
            NativeExecutionOptions {
                flash_attention: true,
                threads,
                threads_batch,
                embeddings: false,
            },
        )
        .unwrap()
    }
    fn batch(count: usize, start: i32) -> Vec<NativeBatchToken> {
        (0..count)
            .map(|index| NativeBatchToken {
                sequences: vec![0],
                position: start + index as i32,
                input: Input::Token(1),
                output: OutputSelection::None,
            })
            .collect()
    }
    #[test]
    fn configured_workers_execute_each_actual_microbatch() {
        let mut context = flash_context(3, 7);
        let layers = context.model.layer_count();
        context.decode_batch(&batch(257, 0)).unwrap();
        let stats = context.execution_stats();
        assert_eq!(stats.microbatches, 3);
        assert_eq!(stats.flash_graphs, 3 * layers);
        assert_eq!(stats.threads_executing_partitions, (7 + 7 + 3) * layers);
        context.decode_batch(&batch(1, 257)).unwrap();
        assert_eq!(
            context.execution_stats().threads_executing_partitions,
            3 * layers
        );
        context.set_threads(2, 4).unwrap();
        context.decode_batch(&batch(2, 258)).unwrap();
        assert_eq!(
            context.execution_stats().threads_executing_partitions,
            4 * layers
        );
        let config = context.execution_options();
        assert!(context.set_threads(0, 4).is_err());
        assert!(context.set_threads(2, 257).is_err());
        assert_eq!(context.execution_options(), config);
    }
    #[test]
    fn rejected_output_consumer_preserves_native_state_and_statistics() {
        let mut context = flash_context(1, 1);
        context.decode_batch(&batch(3, 0)).unwrap();
        let before = context.save_native_state(None, 0).unwrap();
        let previous = context.execution_stats();
        let mut request = batch(1, 3);
        request[0].output = OutputSelection::Logits;
        let mut calls = 0;
        let result = context.decode_batch_with_output_check(
            &request,
            || false,
            |outputs| {
                calls += 1;
                assert_eq!(outputs.len(), 1);
                assert!(outputs[0].logits.as_ref().is_some_and(|v| !v.is_empty()));
                Err(Error::InvalidState("deliberate output rejection".into()))
            },
        );
        assert!(matches!(result, Err(Error::InvalidState(_))));
        assert_eq!(calls, 1);
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        assert_eq!(context.execution_stats(), previous);
        let cancel = std::cell::Cell::new(false);
        let result = context.decode_batch_with_output_check(
            &request,
            || cancel.get(),
            |_| {
                cancel.set(true);
                Ok(())
            },
        );
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        assert_eq!(context.execution_stats(), previous);
        let actual = context.decode_batch(&request).unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(context.position_max(0).unwrap(), 3);
    }
    #[test]
    fn worker_stack_budget_and_cancelled_decode_do_not_commit() {
        let mut context = flash_context(64, 64);
        let before = context.save_native_state(None, 0).unwrap();
        let prior_stats = context.execution_stats();
        // Sixty-four requested worker stacks alone require 32 MiB; a 16-MiB
        // operation ceiling must reject before any worker is spawned.
        let result = context.decode_batch_with_work_limit(&batch(1, 0), || false, 16 * 1024 * 1024);
        assert!(matches!(result, Err(Error::Unsupported(_))));
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        assert_eq!(context.execution_stats(), prior_stats);
        assert_eq!(
            context.decode_batch_with_cancel(&batch(1, 0), || true),
            Err(Error::Cancelled)
        );
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
    }
    #[test]
    fn flash_state_uses_untransposed_raw_half_bytes_and_rejects_cross_layout() {
        let mut context = flash_context(1, 1);
        assert!(!context.values_are_transposed());
        for index in 0..4 {
            context.streams[0].cells[index] = Cell {
                position: index as i32,
                shift: 0,
                sequences: [0].into_iter().collect(),
            };
        }
        for layer in &mut context.streams[0].layers {
            for index in 0..4 * context.width {
                layer.keys.set_bits(index, 0x7c01 + index as u32);
                layer
                    .values
                    .set_bits(index, if index % 2 == 0 { 0x8000 } else { 0xfe35 });
            }
        }
        let bytes = context.save_native_state(None, 0).unwrap();
        let mut restored = flash_context(1, 1);
        assert_eq!(
            restored.restore_native_state(&bytes, None, 0).unwrap(),
            bytes.len()
        );
        assert_eq!(restored.save_native_state(None, 0).unwrap(), bytes);
        let mut other = NativeContext::new_with_cache_types(
            context.model.clone(),
            context.options.clone(),
            NativeKvType::F16,
            NativeKvType::F16,
        )
        .unwrap();
        let before = other.save_native_state(None, 0).unwrap();
        assert!(other.restore_native_state(&bytes, None, 0).is_err());
        assert_eq!(other.save_native_state(None, 0).unwrap(), before);
        // Finite arithmetic admission remains distinct from raw state I/O.
        let state_before = restored.save_native_state(None, 0).unwrap();
        assert!(matches!(
            restored.decode_batch(&batch(1, 4)),
            Err(Error::NonFinite(_))
        ));
        assert_eq!(restored.save_native_state(None, 0).unwrap(), state_before);
    }
    #[test]
    fn complete_logical_row_maps_are_charged_before_state_clone() {
        let mut context = NativeContext::new_with_execution(
            Arc::new(Model::Llama(crate::fixtures::tiny_llama().unwrap())),
            NativeContextOptions {
                context_tokens: 16_384,
                maximum_sequences: 2,
                microbatch_tokens: 2,
                maximum_batch_tokens: 2,
                unified: true,
            },
            NativeKvType::F16,
            NativeKvType::F16,
            NativeExecutionOptions {
                flash_attention: true,
                ..NativeExecutionOptions::default()
            },
        )
        .unwrap();
        let capacity = context.capacity;
        for (index, cell) in context.streams[0]
            .cells
            .iter_mut()
            .take(capacity - 2)
            .enumerate()
        {
            *cell = Cell {
                position: index as i32,
                shift: 0,
                sequences: [0, 1].into_iter().collect(),
            };
        }
        let request = [0, 1]
            .into_iter()
            .map(|sequence| NativeBatchToken {
                sequences: vec![sequence],
                position: (capacity - 2) as i32,
                input: Input::Token(1),
                output: OutputSelection::None,
            })
            .collect::<Vec<_>>();
        let map_bytes = native_row_map::maximum_heap_bytes(&request, 2, capacity).unwrap();
        assert_eq!(map_bytes, capacity * 2 * 4);
        let before = context.save_native_state(None, 0).unwrap();
        // This limit is independent of owned_work. The fixture has two layers
        // of two-element half K/V rows. Two resident cell/KV payload copies,
        // both physical half casts, the two-query mask and two I32 maps alone
        // require 132 bytes per cell on the measured x64 profiles, before any
        // descriptors or projection scratch. At 16,384 cells that exceeds 2 MiB.
        // Omitting the map charge incorrectly admits 2 MiB in the retained
        // negative control; do not derive this threshold from the estimator.
        let resident_payload = size_of::<Cell>() + 2 * 2 * 2 * size_of::<u16>();
        let simultaneous_payload = capacity * (resident_payload * 2 + 8 + 4 + 8);
        let too_small = 2 * 1024 * 1024;
        assert!(simultaneous_payload > too_small);
        assert!(matches!(
            context.decode_batch_with_work_limit(&request, || false, too_small),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        context
            .decode_batch_with_work_limit(&request, || false, 4 * 1024 * 1024)
            .unwrap();
        assert_eq!(context.position_max(0).unwrap(), (capacity - 2) as i32);
        assert_eq!(context.position_max(1).unwrap(), (capacity - 2) as i32);
    }
}
