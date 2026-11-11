//! Frozen CPU optimizer arithmetic with owned, transactional state snapshots.
//!
//! GGML mutates the parameter and Adam moments in place. This immutable tensor
//! engine returns all three buffers together; AdamWState commits them only after
//! the complete step succeeds. The graph exposes moments as auxiliary outputs.
use crate::graph::check_cancel;
use crate::{DType, Result, Tensor, TensorError, reserved};
use std::sync::atomic::AtomicBool;

/// Bounds apply to the complete newly owned result, including physical padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptimizerLimits {
    pub max_output_bytes: usize,
    pub max_elements: usize,
}
impl Default for OptimizerLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: 512 * 1024 * 1024,
            max_elements: 128 * 1024 * 1024,
        }
    }
}

/// Complete parameter/moment ownership. Failed or cancelled steps preserve it.
#[derive(Clone, Debug)]
pub struct AdamWState {
    pub weights: Tensor,
    pub first_moment: Tensor,
    pub second_moment: Tensor,
}
impl AdamWState {
    pub fn new(weights: Tensor, first_moment: Tensor, second_moment: Tensor) -> Result<Self> {
        for t in [&weights, &first_moment, &second_moment] {
            require_f32(t)?;
            if t.shape() != weights.shape() {
                return Err(TensorError::Shape(
                    "AdamW parameter and moments must have the same shape",
                ));
            }
        }
        Ok(Self {
            weights,
            first_moment,
            second_moment,
        })
    }

    /// Parameters are the source's seven physical F32 words: learning rate,
    /// beta1, beta2, epsilon, decay, first correction and second correction.
    /// The corrections are supplied by the caller, as in ggml_opt_step_adamw.
    pub fn step(
        &mut self,
        gradient: &Tensor,
        parameters: &Tensor,
        limits: OptimizerLimits,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let next = adamw_step(
            &self.weights,
            gradient,
            &self.first_moment,
            &self.second_moment,
            parameters,
            limits,
            cancel,
        )?;
        *self = next;
        Ok(())
    }
}

fn require_f32(t: &Tensor) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(TensorError::Type {
            expected: DType::F32,
            actual: t.dtype(),
        });
    }
    Ok(())
}
fn bytes(t: &Tensor) -> &[u8] {
    let end = t.offset + t.desc.byte_span().expect("validated tensor span");
    &t.data[t.offset..end]
}
fn copy_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let raw = bytes(t);
    let mut out = reserved(raw.len())?;
    out.extend_from_slice(raw);
    Ok(out)
}
fn read(raw: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        raw[offset..offset + 4]
            .try_into()
            .expect("validated physical optimizer address"),
    )
}
fn write(raw: &mut [u8], offset: usize, value: f32) {
    raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn overlaps(a: &Tensor, b: &Tensor) -> Result<bool> {
    if !a.shares_storage(b) {
        return Ok(false);
    }
    let ae = a
        .offset
        .checked_add(a.desc.byte_span()?)
        .ok_or(TensorError::Overflow)?;
    let be = b
        .offset
        .checked_add(b.desc.byte_span()?)
        .ok_or(TensorError::Overflow)?;
    Ok(a.offset < be && b.offset < ae)
}

fn validate(
    inputs: &[&Tensor],
    parameters: &Tensor,
    parameter_words: usize,
    output_indices: &[usize],
    limits: OptimizerLimits,
) -> Result<()> {
    let weights = inputs[0];
    for t in inputs {
        require_f32(t)?;
        if t.shape() != weights.shape() {
            return Err(TensorError::Shape(
                "optimizer parameter, gradient and moments must have the same shape",
            ));
        }
    }
    require_f32(parameters)?;
    if parameters.desc.elements()? != parameter_words {
        return Err(TensorError::Shape(
            "optimizer parameter tensor has the wrong element count",
        ));
    }
    if bytes(parameters).len() < parameter_words * 4 {
        return Err(TensorError::Bounds);
    }
    let elements = weights.desc.elements()?;
    if elements > limits.max_elements {
        return Err(TensorError::Parameters("optimizer element limit exceeded"));
    }
    let total = output_indices.iter().try_fold(0usize, |sum, &i| {
        sum.checked_add(inputs[i].desc.byte_span()?)
            .ok_or(TensorError::Overflow)
    })?;
    if total > limits.max_output_bytes {
        return Err(TensorError::Parameters(
            "optimizer output byte limit exceeded",
        ));
    }
    // Source writes w/m/v and can reread g/params. Alias-dependent mutations do
    // not have immutable snapshot semantics, so reject overlap explicitly.
    for (i, a) in inputs.iter().enumerate() {
        for b in inputs.iter().skip(i + 1) {
            if overlaps(a, b)? {
                return Err(TensorError::Unsupported(
                    "overlapping optimizer input buffers",
                ));
            }
        }
        if overlaps(a, parameters)? {
            return Err(TensorError::Unsupported(
                "optimizer parameters overlap an input buffer",
            ));
        }
    }
    let [n0, n1, n2, n3] = weights.shape();
    let nb = weights.desc.nb;
    if nb[0] != 4 || nb.iter().any(|n| !n.is_multiple_of(4)) {
        return Err(TensorError::Layout(
            "optimizer requires aligned contiguous F32 rows",
        ));
    }
    let rows = n1
        .checked_mul(n2)
        .and_then(|n| n.checked_mul(n3))
        .ok_or(TensorError::Overflow)?;
    if rows > i32::MAX as usize || n0 > i32::MAX as usize {
        return Err(TensorError::Unsupported(
            "optimizer dimensions exceed source CPU integer bounds",
        ));
    }
    if elements == 0 {
        return Ok(());
    }
    // Monotonically separated padded rows have a defined task-independent
    // source update. Overlapping/permuted rows need a separate mutable policy.
    let mut extent = n0.checked_mul(4).ok_or(TensorError::Overflow)?;
    for (axis, &stride) in nb.iter().enumerate().skip(1) {
        let count = weights.shape()[axis];
        if count > 1 && stride < extent {
            return Err(TensorError::Unsupported(
                "overlapping or permuted optimizer rows",
            ));
        }
        extent = extent
            .checked_add(
                (count - 1)
                    .checked_mul(stride)
                    .ok_or(TensorError::Overflow)?,
            )
            .ok_or(TensorError::Overflow)?;
    }
    // The native kernel uses the parameter's physical offsets for every input,
    // even when the other tensor descriptor has different strides.
    for t in inputs {
        if bytes(t).len() < extent {
            return Err(TensorError::Bounds);
        }
    }
    Ok(())
}

/// One source SGD step. Parameters contain physical learning-rate/decay words.
pub fn sgd_step(
    weights: &Tensor,
    gradient: &Tensor,
    parameters: &Tensor,
    limits: OptimizerLimits,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    check_cancel(cancel)?;
    validate(&[weights, gradient], parameters, 2, &[0], limits)?;
    let mut out = copy_bytes(weights)?;
    if weights.desc.elements()? == 0 {
        check_cancel(cancel)?;
        return Tensor::from_bytes(weights.desc, out);
    }
    let g = bytes(gradient);
    let p = bytes(parameters);
    let alpha = read(p, 0);
    let keep = 1.0f32 - alpha * read(p, 4);
    let [n0, n1, n2, n3] = weights.shape();
    let nb = weights.desc.nb;
    for i3 in 0..n3 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                check_cancel(cancel)?;
                let row = i3 * nb[3] + i2 * nb[2] + i1 * nb[1];
                for i0 in 0..n0 {
                    if i0.is_multiple_of(1024) {
                        check_cancel(cancel)?;
                    }
                    let offset = row + i0 * 4;
                    let value = read(&out, offset) * keep - alpha * read(g, offset);
                    write(&mut out, offset, value);
                }
            }
        }
    }
    check_cancel(cancel)?;
    Tensor::from_bytes(weights.desc, out)
}

/// One source AdamW step; all three updated buffers are published together.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    weights: &Tensor,
    gradient: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    parameters: &Tensor,
    limits: OptimizerLimits,
    cancel: &AtomicBool,
) -> Result<AdamWState> {
    check_cancel(cancel)?;
    validate(
        &[weights, gradient, first_moment, second_moment],
        parameters,
        7,
        &[0, 2, 3],
        limits,
    )?;
    let mut w = copy_bytes(weights)?;
    let mut m = copy_bytes(first_moment)?;
    let mut v = copy_bytes(second_moment)?;
    if weights.desc.elements()? == 0 {
        check_cancel(cancel)?;
        return AdamWState::new(
            Tensor::from_bytes(weights.desc, w)?,
            Tensor::from_bytes(first_moment.desc, m)?,
            Tensor::from_bytes(second_moment.desc, v)?,
        );
    }
    let g = bytes(gradient);
    let p = bytes(parameters);
    let alpha = read(p, 0);
    let beta1 = read(p, 4);
    let beta2 = read(p, 8);
    let eps = read(p, 12);
    let wd = read(p, 16);
    let beta1h = read(p, 20);
    let beta2h = read(p, 24);
    let keep = 1.0f32 - alpha * wd;
    let [n0, n1, n2, n3] = weights.shape();
    let nb = weights.desc.nb;
    for i3 in 0..n3 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                check_cancel(cancel)?;
                let row = i3 * nb[3] + i2 * nb[2] + i1 * nb[1];
                for i0 in 0..n0 {
                    if i0.is_multiple_of(1024) {
                        check_cancel(cancel)?;
                    }
                    let offset = row + i0 * 4;
                    let grad = read(g, offset);
                    let mv = read(&m, offset) * beta1 + grad * (1.0f32 - beta1);
                    let vv = read(&v, offset) * beta2 + grad * grad * (1.0f32 - beta2);
                    write(&mut m, offset, mv);
                    write(&mut v, offset, vv);
                    let mh = mv * beta1h;
                    let vh = (vv * beta2h).sqrt() + eps;
                    let value = read(&w, offset) * keep - alpha * mh / vh;
                    write(&mut w, offset, value);
                }
            }
        }
    }
    check_cancel(cancel)?;
    AdamWState::new(
        Tensor::from_bytes(weights.desc, w)?,
        Tensor::from_bytes(first_moment.desc, m)?,
        Tensor::from_bytes(second_moment.desc, v)?,
    )
}
