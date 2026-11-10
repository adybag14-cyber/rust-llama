//! Frozen scalar recurrent CPU operators. Source state is immutable; the result
//! concatenates model output and owned final/snapshot state exactly as GGML.
//! These kernels use the scalar recurrent branch on both verified reference
//! builds. Other ISA arithmetic and general parallel DAG scheduling stay open.
use crate::graph::check_cancel;
use crate::{DType, Op, Result, Tensor, TensorError, zeros};
use std::sync::atomic::AtomicBool;

pub(super) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    Some(match op {
        Op::SsmConv => ssm_conv(src[0], src[1], cancel),
        Op::SsmScan { snapshots } => ssm_scan(src, *snapshots, cancel),
        Op::RwkvWkv6 => recurrence(src, Recurrence::Wkv6, cancel),
        Op::GatedLinearAttn { scale } => recurrence(src, Recurrence::Gla(*scale), cancel),
        Op::RwkvWkv7 => recurrence(src, Recurrence::Wkv7, cancel),
        _ => return None,
    })
}

// A single NaN has an unambiguous quieted payload. Choosing between different
// NaNs depends on original compiler register allocation and can change across
// vector chunks and scalar tails. Preserve single-payload IEEE arithmetic, but
// return a checked boundary instead of inventing cross-compiler precedence.
const COMPETING_NANS: &str = "recurrent arithmetic with competing NaN payloads is unsupported";
fn nan_result(a: f32, b: f32) -> Result<Option<f32>> {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => Ok(None),
        (true, false) => Ok(Some(f32::from_bits(a.to_bits() | 0x0040_0000))),
        (false, true) => Ok(Some(f32::from_bits(b.to_bits() | 0x0040_0000))),
        (true, true) => {
            let left = a.to_bits() | 0x0040_0000;
            let right = b.to_bits() | 0x0040_0000;
            if left != right {
                Err(TensorError::Unsupported(COMPETING_NANS))
            } else {
                Ok(Some(f32::from_bits(left)))
            }
        }
    }
}
fn fadd(a: f32, b: f32) -> Result<f32> {
    Ok(nan_result(a, b)?.unwrap_or(a + b))
}
fn fmul(a: f32, b: f32) -> Result<f32> {
    Ok(nan_result(a, b)?.unwrap_or(a * b))
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(TensorError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(TensorError::Overflow)
}
fn size32(n: usize) -> Result<()> {
    i32::try_from(n)
        .map(|_| ())
        .map_err(|_| TensorError::Unsupported("recurrent CPU dimensions must fit signed I32"))
}
fn shape_error(message: &'static str) -> TensorError {
    TensorError::Shape(message)
}
fn dtype(tensor: &Tensor, expected: DType) -> Result<()> {
    if tensor.dtype() != expected {
        return Err(TensorError::Type {
            expected,
            actual: tensor.dtype(),
        });
    }
    for dimension in tensor.shape() {
        size32(dimension)?;
    }
    Ok(())
}
fn contiguous(tensor: &Tensor) -> Result<()> {
    if !tensor.desc().is_contiguous() {
        return Err(TensorError::Layout("recurrent input must be contiguous"));
    }
    Ok(())
}

/// Physical source addressing matters: e.g. the one-worker SSM convolution
/// ignores the weight row stride. Never widen accesses to the owning buffer
/// beyond this tensor's declared view.
struct Floats<'a> {
    bytes: &'a [u8],
}
impl<'a> Floats<'a> {
    fn new(tensor: &'a Tensor) -> Result<Self> {
        dtype(tensor, DType::F32)?;
        let end = add(tensor.offset, tensor.desc().byte_span()?)?;
        Ok(Self {
            bytes: tensor
                .data
                .get(tensor.offset..end)
                .ok_or(TensorError::Bounds)?,
        })
    }
    fn read(&self, index: usize) -> Result<f32> {
        let start = mul(index, 4)?;
        let bytes = self
            .bytes
            .get(start..add(start, 4)?)
            .ok_or(TensorError::Bounds)?;
        Ok(f32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }
    fn require(&self, count: usize) -> Result<()> {
        if mul(count, 4)? > self.bytes.len() {
            Err(TensorError::Bounds)
        } else {
            Ok(())
        }
    }
}
fn strides(tensor: &Tensor) -> Result<[usize; 4]> {
    if tensor.desc().nb.iter().any(|v| !v.is_multiple_of(4)) {
        return Err(TensorError::Layout(
            "recurrent source strides must be scalar-aligned",
        ));
    }
    Ok(tensor.desc().nb.map(|v| v / 4))
}
fn packed_plane(tensor: &Tensor) -> Result<[usize; 4]> {
    let nb = strides(tensor)?;
    if nb[0] != 1 || nb[1] != tensor.shape()[0] {
        return Err(TensorError::Layout(
            "SSM input requires a contiguous first-two-axis plane",
        ));
    }
    Ok(nb)
}
fn plane_offset(token: usize, sequence: usize, nb: [usize; 4]) -> Result<usize> {
    add(mul(token, nb[2])?, mul(sequence, nb[3])?)
}

fn ssm_conv(x: &Tensor, weights: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let input = Floats::new(x)?;
    let weight = Floats::new(weights)?;
    let [window, channels, sequences, outer] = x.shape();
    let [kernel, weight_channels, weight_outer, weight_batch] = weights.shape();
    if outer != 1 || weight_outer != 1 || weight_batch != 1 || weight_channels != channels {
        return Err(shape_error(
            "SSM_CONV requires a 3D input and matching matrix weights",
        ));
    }
    if kernel == 0 {
        return Err(shape_error("SSM_CONV requires a nonzero kernel"));
    }
    let tokens = add(window, 1)?
        .checked_sub(kernel)
        .ok_or_else(|| shape_error("SSM_CONV window is shorter than kernel - 1"))?;
    let shape = [channels, tokens, sequences, 1];
    let count = mul(mul(channels, tokens)?, sequences)?;
    let mut output = zeros(count)?;
    // The original dispatcher does not call the kernel for an empty output.
    if count == 0 {
        return Tensor::from_f32(shape, &output);
    }
    let nb = packed_plane(x)?;
    if weights.desc().nb[0] != 4 {
        return Err(TensorError::Layout(
            "SSM_CONV weight scalar stride must be four bytes",
        ));
    }
    size32(mul(channels, window)?)?;
    weight.require(mul(kernel, channels)?)?;
    for sequence in 0..sequences {
        let sequence_offset = mul(sequence, nb[2])?;
        for token in 0..tokens {
            for channel in 0..channels {
                check_cancel(cancel)?;
                let row = add(add(sequence_offset, mul(channel, window)?)?, token)?;
                let mut sum = 0.0_f32;
                for tap in 0..kernel {
                    sum = fadd(
                        sum,
                        fmul(
                            input.read(add(row, tap)?)?,
                            weight.read(channel * kernel + tap)?,
                        )?,
                    )?;
                }
                output[channel + channels * (token + tokens * sequence)] = sum;
            }
        }
    }
    Tensor::from_f32(shape, &output)
}

fn ssm_scan(src: &[&Tensor], snapshots: usize, cancel: &AtomicBool) -> Result<Tensor> {
    let [s, x, dt, a, b, c, ids] = [src[0], src[1], src[2], src[3], src[4], src[5], src[6]];
    let state = Floats::new(s)?;
    let input = Floats::new(x)?;
    let delta = Floats::new(dt)?;
    let decay = Floats::new(a)?;
    let weights_b = Floats::new(b)?;
    let weights_c = Floats::new(c)?;
    dtype(ids, DType::I32)?;
    for tensor in [s, dt, a] {
        contiguous(tensor)?;
    }
    let snb = strides(s)?;
    let xnb = packed_plane(x)?;
    let dnb = strides(dt)?;
    let anb = strides(a)?;
    let bnb = packed_plane(b)?;
    let cnb = packed_plane(c)?;
    if snb[0] != 1 || dnb[0] != 1 || anb[0] != 1 || ids.desc().nb[0] != 4 {
        return Err(TensorError::Layout(
            "SSM_SCAN scalar strides must be four bytes",
        ));
    }
    let [state_width, state_dim, state_heads, stored_sequences] = s.shape();
    let [dim, heads, tokens, sequences] = x.shape();
    let groups = b.shape()[1];
    if s.shape().contains(&0) || x.shape().contains(&0) || groups == 0 {
        return Err(shape_error(
            "SSM_SCAN requires nonempty state, heads, tokens and sequences",
        ));
    }
    if state_dim != dim
        || state_heads != heads
        || dt.shape() != [heads, tokens, sequences, 1]
        || b.shape() != [state_width, groups, tokens, sequences]
        || c.shape() != b.shape()
        || ids.shape() != [sequences, 1, 1, 1]
        || a.shape()[1..] != [heads, 1, 1]
        || !heads.is_multiple_of(groups)
    {
        return Err(shape_error(
            "SSM_SCAN state, input, group or sequence shapes do not match",
        ));
    }
    let scalar_decay = a.shape()[0] == 1;
    if !scalar_decay && a.shape()[0] != state_width {
        return Err(shape_error(
            "SSM_SCAN decay must have one or d_state values per head",
        ));
    }
    size32(snapshots)?;
    if snapshots == 0 || snapshots > tokens || (!scalar_decay && snapshots != 1) {
        return Err(TensorError::Parameters(
            "SSM_SCAN snapshots must be 1..=tokens; per-state decay requires one",
        ));
    }
    let channels = mul(dim, heads)?;
    let state_per_sequence = mul(state_width, channels)?;
    // The native inner loops use signed int flattened state/row indices.
    size32(state_per_sequence)?;
    let output_count = x.desc().elements()?;
    let state_count = mul(state_per_sequence, sequences)?;
    let total = add(output_count, mul(snapshots, state_count)?)?;
    // The source derives destination/snapshot strides from the input state.
    // Unit-axis strides can otherwise leave unwritten output holes or overlap.
    if (sequences > 1 || snapshots > 1) && snb[3] != state_per_sequence {
        return Err(TensorError::Layout(
            "SSM_SCAN state sequence stride must cover one complete output state",
        ));
    }
    if snapshots > 1 && snb[2] != mul(state_width, dim)? {
        return Err(TensorError::Layout(
            "SSM_SCAN snapshot head stride must cover one complete head",
        ));
    }
    let sequence_ids = ids.to_i32()?;
    for &id in &sequence_ids {
        if id < 0 || id as usize >= stored_sequences {
            return Err(TensorError::Bounds);
        }
        let first = mul(id as usize, snb[3])?;
        state.require(add(first, state_per_sequence)?)?;
    }
    let mut output = zeros(total)?;
    for (sequence, &id) in sequence_ids.iter().enumerate() {
        let initial_offset = mul(id as usize, snb[3])?;
        let current = output_count + sequence * state_per_sequence;
        for token in 0..tokens {
            check_cancel(cancel)?;
            let xoff = plane_offset(token, sequence, xnb)?;
            let doff = add(mul(token, dnb[1])?, mul(sequence, dnb[2])?)?;
            let boff = plane_offset(token, sequence, bnb)?;
            let coff = plane_offset(token, sequence, cnb)?;
            for head in 0..heads {
                let d = delta.read(add(doff, head)?)?;
                // The frozen helper is logf(1 + expf(x)), not log1pf(expf(x)).
                let dt_softplus = if d > 20.0 {
                    d
                } else {
                    fadd(1.0, d.exp())?.ln()
                };
                let d_a = if scalar_decay {
                    fmul(dt_softplus, decay.read(head)?)?.exp()
                } else {
                    0.0
                };
                let group = head / (heads / groups);
                for channel in 0..dim {
                    check_cancel(cancel)?;
                    let row = channel + head * dim;
                    let x_dt = fmul(input.read(add(xoff, row)?)?, dt_softplus)?;
                    let mut sum = 0.0_f32;
                    for column in 0..state_width {
                        let index = column + row * state_width;
                        let group_index = column + group * state_width;
                        let previous = if token == 0 {
                            state.read(add(initial_offset, index)?)?
                        } else {
                            output[current + index]
                        };
                        let exp_a = if scalar_decay {
                            d_a
                        } else {
                            fmul(dt_softplus, decay.read(column + head * state_width)?)?.exp()
                        };
                        let value = fadd(
                            fmul(previous, exp_a)?,
                            fmul(weights_b.read(add(boff, group_index)?)?, x_dt)?,
                        )?;
                        sum = fadd(sum, fmul(value, weights_c.read(add(coff, group_index)?)?)?)?;
                        output[current + index] = value;
                    }
                    output[row + channels * (token + tokens * sequence)] = sum;
                }
            }
            let slot = tokens - 1 - token;
            if slot > 0 && slot < snapshots {
                let destination = output_count + slot * state_count + sequence * state_per_sequence;
                output.copy_within(current..current + state_per_sequence, destination);
            }
        }
    }
    Tensor::from_f32([total, 1, 1, 1], &output)
}

#[derive(Clone, Copy)]
enum Recurrence {
    Wkv6,
    Gla(f32),
    Wkv7,
}
fn recurrence(src: &[&Tensor], kind: Recurrence, cancel: &AtomicBool) -> Result<Tensor> {
    let inputs = src
        .iter()
        .map(|tensor| Floats::new(tensor))
        .collect::<Result<Vec<_>>>()?;
    for tensor in src {
        contiguous(tensor)?;
    }
    // Source uses first-three-axis token slabs and flat state bytes. Additional
    // fourth-axis slabs are retained storage but deliberately not evaluated.
    let state_index = src.len() - 1;
    let [width, heads, tokens, _] = src[0].shape();
    let sequences = src[state_index].shape()[1];
    if width == 0
        || heads == 0
        || tokens == 0
        || sequences == 0
        || !tokens.is_multiple_of(sequences)
    {
        return Err(shape_error(
            "recurrence requires positive heads and equal nonempty token runs per sequence",
        ));
    }
    for (index, tensor) in src[..state_index].iter().enumerate() {
        if matches!(kind, Recurrence::Wkv6) && index == 3 {
            continue;
        }
        if tensor.shape()[..3] != [width, heads, tokens] || tensor.shape()[3] == 0 {
            return Err(shape_error("recurrent token tensor shapes do not match"));
        }
    }
    let channels = mul(width, heads)?;
    let head_state = mul(width, width)?;
    let state_per_sequence = mul(head_state, heads)?;
    let state_count = mul(state_per_sequence, sequences)?;
    if src[state_index].desc().elements()? != state_count {
        return Err(shape_error(
            "recurrent state element count does not match head matrices",
        ));
    }
    let output_rows = add(tokens, mul(width, sequences)?)?;
    let shape = [channels, output_rows, 1, 1];
    let count = mul(channels, output_rows)?;
    let output_count = mul(channels, tokens)?;
    for (index, input) in inputs[..state_index].iter().enumerate() {
        input.require(if matches!(kind, Recurrence::Wkv6) && index == 3 {
            channels
        } else {
            output_count
        })?;
    }
    inputs[state_index].require(state_count)?;
    let mut output = zeros(count)?;
    let tokens_per_sequence = tokens / sequences;
    for token in 0..tokens {
        check_cancel(cancel)?;
        let sequence = token / tokens_per_sequence;
        let state_start = sequence * state_per_sequence;
        let continuing = token % tokens_per_sequence != 0;
        for head in 0..heads {
            let token_head = token * channels + head * width;
            let state_head = state_start + head * head_state;
            for row in 0..width {
                check_cancel(cancel)?;
                let token_row = token_head + row;
                let state_row = state_head + row * width;
                match kind {
                    Recurrence::Wkv6 => {
                        let k = inputs[0].read(token_row)?;
                        let r = inputs[2].read(token_row)?;
                        let time_first = inputs[3].read(head * width + row)?;
                        let time_decay = inputs[4].read(token_row)?;
                        for column in 0..width {
                            let token_column = token_head + column;
                            let state_column = state_row + column;
                            let v = inputs[1].read(token_column)?;
                            let kv = fmul(v, k)?;
                            let previous = if continuing {
                                output[output_count + state_column]
                            } else {
                                inputs[state_index].read(state_column)?
                            };
                            let contribution = fadd(fmul(kv, time_first)?, previous)?;
                            output[token_column] =
                                fadd(output[token_column], fmul(contribution, r)?)?;
                            output[output_count + state_column] =
                                fadd(fmul(previous, time_decay)?, kv)?;
                        }
                    }
                    Recurrence::Gla(scale) => {
                        let k = inputs[0].read(token_row)?;
                        let q = fmul(inputs[2].read(token_row)?, scale)?;
                        let gate = inputs[3].read(token_row)?;
                        for column in 0..width {
                            let token_column = token_head + column;
                            let state_column = state_row + column;
                            let kv = fmul(inputs[1].read(token_column)?, k)?;
                            let previous = if continuing {
                                output[output_count + state_column]
                            } else {
                                inputs[state_index].read(state_column)?
                            };
                            let value = fadd(fmul(previous, gate)?, kv)?;
                            output[token_column] = fadd(output[token_column], fmul(value, q)?)?;
                            output[output_count + state_column] = value;
                        }
                    }
                    Recurrence::Wkv7 => {
                        let v = inputs[3].read(token_row)?;
                        let mut state_a = 0.0_f32;
                        for column in 0..width {
                            let previous = if continuing {
                                output[output_count + state_row + column]
                            } else {
                                inputs[state_index].read(state_row + column)?
                            };
                            state_a = fadd(
                                state_a,
                                fmul(inputs[4].read(token_head + column)?, previous)?,
                            )?;
                        }
                        let mut result = 0.0_f32;
                        for column in 0..width {
                            let token_column = token_head + column;
                            let state_column = state_row + column;
                            let r = inputs[0].read(token_column)?;
                            let decay = inputs[1].read(token_column)?;
                            let k = inputs[2].read(token_column)?;
                            let b = inputs[5].read(token_column)?;
                            let previous = if continuing {
                                output[output_count + state_column]
                            } else {
                                inputs[state_index].read(state_column)?
                            };
                            let value = fadd(
                                fadd(fmul(previous, decay)?, fmul(v, k)?)?,
                                fmul(state_a, b)?,
                            )?;
                            output[output_count + state_column] = value;
                            result = fadd(result, fmul(value, r)?)?;
                        }
                        output[token_row] = result;
                    }
                }
            }
        }
    }
    Tensor::from_f32(shape, &output)
}
