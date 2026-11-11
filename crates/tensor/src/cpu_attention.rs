//! Frozen CPU gated-delta recurrence and DeepSeek V4 hyper-connection kernels.
//! Explicit byte strides, scalar F32 operations and source reduction order.
use crate::graph::check_cancel;
use crate::{DType, Op, Result, Tensor, TensorDesc, TensorError, zeros};
use std::sync::atomic::AtomicBool;

pub(super) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    Some(match op {
        Op::GatedDeltaNet { snapshots } => gated_delta(src, *snapshots, cancel),
        Op::Dsv4HcComb {
            epsilon,
            iterations,
        } => combine(src, *epsilon, *iterations, cancel),
        Op::Dsv4HcPre => pre(src[0], src[1], None, cancel),
        Op::Dsv4HcPreGated { scale } => pre(src[0], src[1], Some(*scale), cancel),
        Op::Dsv4HcPost { with_comb } => post(src, *with_comb, cancel),
        _ => return None,
    })
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(TensorError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(TensorError::Overflow)
}
fn positive_i32(n: usize) -> Result<()> {
    if n == 0 || n > i32::MAX as usize {
        return Err(TensorError::Parameters(
            "CPU attention geometry must fit positive I32",
        ));
    }
    Ok(())
}
fn contiguous(t: &Tensor) -> Result<()> {
    if !t.desc().is_contiguous() {
        return Err(TensorError::Layout(
            "gated delta state, beta and gate must be contiguous",
        ));
    }
    Ok(())
}
fn finite(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(TensorError::Unsupported(
            "gated-delta and hyper-connection nonfinite arithmetic is unverified",
        ));
    }
    Ok(value)
}

/// Reads the declared view's raw F32 storage, including strides and holes.
/// Input values are checked only when the original kernel actually reads them.
struct Floats<'a> {
    tensor: &'a Tensor,
    bytes: &'a [u8],
}
impl<'a> Floats<'a> {
    fn new(tensor: &'a Tensor) -> Result<Self> {
        if tensor.dtype() != DType::F32 {
            return Err(TensorError::Type {
                expected: DType::F32,
                actual: tensor.dtype(),
            });
        }
        for n in tensor.shape() {
            positive_i32(n)?;
        }
        if tensor
            .desc()
            .nb
            .iter()
            .any(|stride| !stride.is_multiple_of(4))
        {
            return Err(TensorError::Layout(
                "CPU attention byte strides must be F32-aligned",
            ));
        }
        let end = add(tensor.offset, tensor.desc().byte_span()?)?;
        let bytes = tensor
            .data
            .get(tensor.offset..end)
            .ok_or(TensorError::Bounds)?;
        Ok(Self { tensor, bytes })
    }
    fn physical(&self, offset: usize) -> Result<f32> {
        let bytes = self
            .bytes
            .get(offset..add(offset, 4)?)
            .ok_or(TensorError::Bounds)?;
        finite(f32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }
    fn at(&self, indices: [usize; 4]) -> Result<f32> {
        let mut offset = 0;
        for (axis, index) in indices.into_iter().enumerate() {
            if index >= self.tensor.shape()[axis] {
                return Err(TensorError::Bounds);
            }
            offset = add(offset, mul(index, self.tensor.desc().nb[axis])?)?;
        }
        self.physical(offset)
    }
}
fn output(shape: [usize; 4]) -> Result<Vec<f32>> {
    let desc = TensorDesc::contiguous(DType::F32, shape)?;
    if desc.byte_span()? > 256 * 1024 * 1024 {
        return Err(TensorError::Unsupported(
            "scalar attention output exceeds the 256 MiB allocation limit",
        ));
    }
    zeros(desc.elements()?)
}
fn finish(shape: [usize; 4], output: &[f32]) -> Result<Tensor> {
    for &value in output {
        finite(value)?;
    }
    Tensor::from_f32(shape, output)
}

fn gated_delta(src: &[&Tensor], snapshots: usize, cancel: &AtomicBool) -> Result<Tensor> {
    let [q, k, v, g, beta, state] = [src[0], src[1], src[2], src[3], src[4], src[5]];
    let qv = Floats::new(q)?;
    let kv = Floats::new(k)?;
    let vv = Floats::new(v)?;
    let gv = Floats::new(g)?;
    let bv = Floats::new(beta)?;
    let initial = Floats::new(state)?;
    for t in [q, k, v] {
        if t.shape()[0] != 1 && t.desc().nb[0] != 4 {
            return Err(TensorError::Layout(
                "gated delta Q/K/V rows must be contiguous",
            ));
        }
    }
    for t in [g, beta, state] {
        contiguous(t)?;
    }
    let [width, heads, tokens, sequences] = v.shape();
    positive_i32(snapshots)?;
    if snapshots > tokens {
        return Err(TensorError::Parameters(
            "gated delta snapshots must be 1..=tokens; older native slots are unwritten",
        ));
    }
    for t in [q, k] {
        let n = t.shape();
        if n[0] < width || n[2] < tokens || !sequences.is_multiple_of(n[3]) {
            return Err(TensorError::Shape(
                "gated delta Q/K need complete rows/tokens and divisor sequence groups",
            ));
        }
    }
    if g.shape()[0] != 1 && g.shape()[0] != width {
        return Err(TensorError::Shape(
            "gated delta gate must be scalar or per-channel",
        ));
    }
    if beta.shape()[0] != 1 {
        return Err(TensorError::Shape(
            "gated delta beta must be scalar per head",
        ));
    }
    for t in [g, beta] {
        let n = t.shape();
        if n[1] < heads || n[2] < tokens || n[3] < sequences {
            return Err(TensorError::Shape(
                "gated delta gate/beta must cover every head, token and sequence",
            ));
        }
    }
    if state.shape() != [width, width, heads, sequences] {
        return Err(TensorError::Shape(
            "gated delta state must contain one matrix per head and sequence",
        ));
    }
    let channels = mul(width, heads)?;
    let head_state = mul(width, width)?;
    // The frozen ggml_vec_* helpers receive signed-int widths/state lengths.
    positive_i32(head_state)?;
    let state_per_snapshot = mul(mul(head_state, heads)?, sequences)?;
    let score_count = mul(mul(channels, tokens)?, sequences)?;
    let shape = [
        channels,
        add(
            mul(tokens, sequences)?,
            mul(mul(snapshots, width)?, sequences)?,
        )?,
        1,
        1,
    ];
    let mut result = output(shape)?;
    let mut working = zeros(head_state)?;
    let mut delta = zeros(width)?;
    let scale = 1.0_f32 / (width as f32).sqrt();
    let vector_gate = g.shape()[0] == width;
    for sequence in 0..sequences {
        let q_sequence = sequence / (sequences / q.shape()[3]);
        let k_sequence = sequence / (sequences / k.shape()[3]);
        for head in 0..heads {
            check_cancel(cancel)?;
            let q_head = head % q.shape()[1];
            let k_head = head % k.shape()[1];
            let initial_base = add(
                mul(sequence, state.desc().nb[3])?,
                mul(mul(head, head_state)?, 4)?,
            )?;
            for (i, value) in working.iter_mut().enumerate() {
                *value = initial.physical(add(initial_base, mul(i, 4)?)?)?;
            }
            for token in 0..tokens {
                check_cancel(cancel)?;
                let beta = bv.at([0, head, token, sequence])?;
                if vector_gate {
                    for (i, d) in delta.iter_mut().enumerate() {
                        *d = gv.at([i, head, token, sequence])?.exp();
                    }
                    for row in working.chunks_exact_mut(width) {
                        for (value, decay) in row.iter_mut().zip(&delta) {
                            *value *= *decay;
                        }
                    }
                } else {
                    let decay = gv.at([0, head, token, sequence])?.exp();
                    for value in &mut working {
                        *value *= decay;
                    }
                }
                for j in 0..width {
                    check_cancel(cancel)?;
                    // Scalar native dot: each product rounds to F32, then sums in F64.
                    let mut sum = 0.0_f64;
                    for i in 0..width {
                        sum += (working[j * width + i] * kv.at([i, k_head, token, k_sequence])?)
                            as f64;
                    }
                    delta[j] = (vv.at([j, head, token, sequence])? - sum as f32) * beta;
                }
                for j in 0..width {
                    for i in 0..width {
                        working[j * width + i] += kv.at([i, k_head, token, k_sequence])? * delta[j];
                    }
                }
                let score_start = ((sequence * tokens + token) * heads + head) * width;
                for j in 0..width {
                    let mut sum = 0.0_f64;
                    for i in 0..width {
                        sum += (working[j * width + i] * qv.at([i, q_head, token, q_sequence])?)
                            as f64;
                    }
                    result[score_start + j] = sum as f32 * scale;
                }
                let slot = tokens - 1 - token;
                if slot < snapshots {
                    let start = score_count
                        + slot * state_per_snapshot
                        + (sequence * heads + head) * head_state;
                    result[start..start + head_state].copy_from_slice(&working);
                }
            }
        }
    }
    finish(shape, &result)
}

fn norm_columns(comb: &mut [f32; 16], epsilon: f32) {
    for destination in 0..4 {
        let mut sum = epsilon;
        for source in 0..4 {
            sum += comb[destination + 4 * source];
        }
        let inverse = 1.0_f32 / sum;
        for source in 0..4 {
            comb[destination + 4 * source] *= inverse;
        }
    }
}
fn norm_rows(comb: &mut [f32; 16], epsilon: f32) {
    for source in 0..4 {
        let mut sum = epsilon;
        for destination in 0..4 {
            sum += comb[destination + 4 * source];
        }
        let inverse = 1.0_f32 / sum;
        for destination in 0..4 {
            comb[destination + 4 * source] *= inverse;
        }
    }
}
fn combine(
    src: &[&Tensor],
    epsilon: f32,
    iterations: usize,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    finite(epsilon)?;
    positive_i32(iterations)?;
    let mixes = Floats::new(src[0])?;
    let scale = Floats::new(src[1])?;
    let base = Floats::new(src[2])?;
    let tokens = src[0].shape()[1];
    if src[0].shape() != [24, tokens, 1, 1]
        || src[1].shape()[0] < 3
        || src[1].shape()[1..] != [1, 1, 1]
        || src[2].shape() != [24, 1, 1, 1]
    {
        return Err(TensorError::Shape(
            "DSV4_HC_COMB requires 24 mixes, at least three scales and 24 base values",
        ));
    }
    let shape = [4, 4, tokens, 1];
    let mut result = output(shape)?;
    let scale_comb = scale.at([2, 0, 0, 0])?;
    for token in 0..tokens {
        check_cancel(cancel)?;
        let mut comb = [0.0_f32; 16];
        for source in 0..4 {
            let mut maximum = f32::NEG_INFINITY;
            for destination in 0..4 {
                let index = destination + 4 * source;
                let value = mixes.at([8 + index, token, 0, 0])? * scale_comb
                    + base.at([8 + index, 0, 0, 0])?;
                comb[index] = value;
                maximum = if maximum > value { maximum } else { value };
            }
            let mut sum = 0.0_f32;
            for destination in 0..4 {
                let index = destination + 4 * source;
                let value = (comb[index] - maximum).exp();
                comb[index] = value;
                sum += value;
            }
            let inverse = 1.0_f32 / sum;
            for destination in 0..4 {
                let index = destination + 4 * source;
                comb[index] = comb[index] * inverse + epsilon;
            }
        }
        norm_columns(&mut comb, epsilon);
        for _ in 1..iterations {
            check_cancel(cancel)?;
            norm_rows(&mut comb, epsilon);
            norm_columns(&mut comb, epsilon);
        }
        result[token * 16..(token + 1) * 16].copy_from_slice(&comb);
    }
    finish(shape, &result)
}

fn pre(
    x: &Tensor,
    weights: &Tensor,
    gated_scale: Option<f32>,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let input = Floats::new(x)?;
    let weight = Floats::new(weights)?;
    let [embedding, streams, tokens, outer] = x.shape();
    if outer != 1
        || weights.shape()
            != if gated_scale.is_some() {
                [embedding, streams, tokens, 1]
            } else {
                [streams, tokens, 1, 1]
            }
    {
        return Err(TensorError::Shape(
            "DSV4_HC_PRE requires matching weights or per-embedding gates",
        ));
    }
    let scale = finite(gated_scale.unwrap_or(1.0))?;
    let shape = [embedding, tokens, 1, 1];
    let mut result = output(shape)?;
    for token in 0..tokens {
        for channel in 0..embedding {
            check_cancel(cancel)?;
            let mut sum = 0.0_f32;
            for stream in 0..streams {
                let w = if gated_scale.is_some() {
                    let gate = weight.at([channel, stream, token, 0])?;
                    1.0_f32 / (1.0_f32 + (-gate).exp())
                } else {
                    weight.at([stream, token, 0, 0])?
                };
                sum += input.at([channel, stream, token, 0])? * w;
            }
            result[channel + embedding * token] = scale * sum;
        }
    }
    finish(shape, &result)
}

fn post(src: &[&Tensor], with_comb: bool, cancel: &AtomicBool) -> Result<Tensor> {
    let input = Floats::new(src[0])?;
    let residual = Floats::new(src[1])?;
    let weights = Floats::new(src[2])?;
    let comb = if with_comb {
        Some(Floats::new(src[3])?)
    } else {
        None
    };
    let [embedding, tokens, outer, batch] = src[0].shape();
    let streams = src[1].shape()[1];
    if outer != 1
        || batch != 1
        || src[1].shape() != [embedding, streams, tokens, 1]
        || src[2].shape() != [streams, tokens, 1, 1]
        || (with_comb && src[3].shape() != [streams, streams, tokens, 1])
    {
        return Err(TensorError::Shape(
            "DSV4_HC_POST inputs must have matching embedding, stream and token axes",
        ));
    }
    let shape = [embedding, streams, tokens, 1];
    let mut result = output(shape)?;
    for token in 0..tokens {
        for destination in 0..streams {
            for channel in 0..embedding {
                check_cancel(cancel)?;
                let mut sum =
                    input.at([channel, token, 0, 0])? * weights.at([destination, token, 0, 0])?;
                if let Some(comb) = &comb {
                    for source in 0..streams {
                        sum += residual.at([channel, source, token, 0])?
                            * comb.at([destination, source, token, 0])?;
                    }
                } else {
                    sum += residual.at([channel, destination, token, 0])?;
                }
                result[channel + embedding * (destination + streams * token)] = sum;
            }
        }
    }
    finish(shape, &result)
}
