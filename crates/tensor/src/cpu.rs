use crate::graph::check_cancel;
use crate::{DType, Op, Result, RopeMode, Tensor, TensorDesc, TensorError, UnaryOp, zeros};
use std::sync::atomic::AtomicBool;

fn f32_values(t: &Tensor) -> Result<Vec<f32>> {
    if t.dtype() != DType::F32 {
        return Err(TensorError::Type {
            expected: DType::F32,
            actual: t.dtype(),
        });
    }
    t.to_f32()
}
fn index(mut flat: usize, shape: [usize; 4]) -> [usize; 4] {
    let mut out = [0; 4];
    for axis in 0..4 {
        out[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    out
}
fn flat(i: [usize; 4], n: [usize; 4]) -> usize {
    i[0] + n[0] * (i[1] + n[1] * (i[2] + n[2] * i[3]))
}
fn count(n: [usize; 4]) -> Result<usize> {
    TensorDesc::contiguous(DType::F32, n)?.elements()
}
fn broadcastable(small: [usize; 4], big: [usize; 4]) -> bool {
    small
        .into_iter()
        .zip(big)
        .all(|(a, b)| a > 0 && b.is_multiple_of(a))
}
fn eps(v: f32) -> Result<()> {
    if !v.is_finite() || v < 0.0 {
        Err(TensorError::Parameters(
            "epsilon must be finite and nonnegative",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Result<Tensor> {
    use Op::*;
    if let Some(result) = crate::cpu_flash::evaluate(op, src, cancel) {
        return result;
    }
    if let Some(result) = crate::cpu_conv::evaluate(op, src, cancel) {
        return result;
    }
    if let Some(result) = crate::cpu_index::evaluate(op, src, cancel) {
        return result;
    }
    if let Some(result) = crate::cpu_extra::evaluate(op, src, cancel) {
        return result;
    }
    let a = src[0];
    let n = a.shape();
    match op {
        Dup | Contiguous => return a.contiguous(),
        Reshape { shape } => return a.reshape(*shape),
        Permute { axes } => return a.permute(*axes),
        Transpose => return a.permute([1, 0, 2, 3]),
        GetRows => return get_rows(a, src[1], cancel),
        _ => {}
    }
    let x = f32_values(a)?;
    match op {
        Add | Sub | Mul | Div | AddScalar => {
            let b = src[1];
            let bn = b.shape();
            let y = f32_values(b)?;
            if matches!(op, AddScalar) && y.len() != 1 {
                return Err(TensorError::Shape("ADD1 requires a scalar second tensor"));
            }
            if !broadcastable(bn, n) {
                return Err(TensorError::Shape(
                    "second tensor must repeat into the first tensor shape",
                ));
            }
            let mut out = zeros(x.len())?;
            for i in 0..out.len() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let coords = index(i, n);
                let j = flat(std::array::from_fn(|d| coords[d] % bn[d]), bn);
                out[i] = match op {
                    Add | AddScalar => x[i] + y[j],
                    Sub => x[i] - y[j],
                    Mul => x[i] * y[j],
                    Div => x[i] / y[j],
                    _ => unreachable!(),
                };
            }
            Tensor::from_f32(n, &out)
        }
        Scale { scale, bias } => {
            if !scale.is_finite() || !bias.is_finite() {
                return Err(TensorError::Parameters("scale and bias must be finite"));
            }
            map(n, &x, cancel, |v| Ok(v * scale + bias))
        }
        Square | Sqrt | Log | Sin | Cos | Unary(_) => unreachable!("typed unary handled first"),
        Clamp { min, max } => {
            if !min.is_finite() || !max.is_finite() || min > max {
                return Err(TensorError::Parameters(
                    "clamp range must be finite and ordered",
                ));
            }
            map(n, &x, cancel, |v| Ok(v.max(*min).min(*max)))
        }
        Sum => unreachable!("typed sum handled first"),
        SumRows | Mean => {
            if n[0] == 0 {
                return Err(TensorError::Shape("row reduction requires a nonempty row"));
            }
            let mut out = zeros(x.len() / n[0])?;
            for (row, chunk) in x.chunks_exact(n[0]).enumerate() {
                check_cancel(cancel)?;
                let sum = chunk.iter().fold(0.0_f64, |sum, &v| sum + v as f64) as f32;
                out[row] = if matches!(op, Mean) {
                    sum / n[0] as f32
                } else {
                    sum
                };
            }
            Tensor::from_f32([1, n[1], n[2], n[3]], &out)
        }
        Repeat { shape } => {
            if !broadcastable(n, *shape) {
                return Err(TensorError::Shape(
                    "input does not tile the repeat output shape",
                ));
            }
            let mut out = zeros(count(*shape)?)?;
            for (i, v) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let coords = index(i, *shape);
                *v = x[flat(std::array::from_fn(|d| coords[d] % n[d]), n)];
            }
            Tensor::from_f32(*shape, &out)
        }
        Concat { axis } => {
            if *axis >= 4 {
                return Err(TensorError::Parameters("concat axis is outside 0..4"));
            }
            let b = src[1];
            let bn = b.shape();
            let y = f32_values(b)?;
            for d in 0..4 {
                if d != *axis && n[d] != bn[d] {
                    return Err(TensorError::Shape("concat dimensions do not match"));
                }
            }
            let mut shape = n;
            shape[*axis] = shape[*axis]
                .checked_add(bn[*axis])
                .ok_or(TensorError::Overflow)?;
            let mut out = zeros(count(shape)?)?;
            for (i, v) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let mut at = index(i, shape);
                *v = if at[*axis] < n[*axis] {
                    x[flat(at, n)]
                } else {
                    at[*axis] -= n[*axis];
                    y[flat(at, bn)]
                };
            }
            Tensor::from_f32(shape, &out)
        }
        MatMul => unreachable!("mixed-precision matmul handled first"),
        Softmax { scale } => {
            if !scale.is_finite() {
                return Err(TensorError::Parameters("softmax scale must be finite"));
            }
            if n[0] == 0 {
                return Err(TensorError::Shape("softmax requires a nonempty row"));
            }
            let mut out = zeros(x.len())?;
            for (input, output) in x.chunks_exact(n[0]).zip(out.chunks_exact_mut(n[0])) {
                check_cancel(cancel)?;
                let max = input
                    .iter()
                    .map(|v| v * scale)
                    .fold(f32::NEG_INFINITY, f32::max);
                // The C implementation asserts on an entirely masked or non-finite row.
                // Return an error rather than fabricating a probability distribution.
                if !max.is_finite() || input.iter().any(|v| v.is_nan()) {
                    return Err(TensorError::Parameters(
                        "softmax row has no finite maximum or contains NaN",
                    ));
                }
                for (o, &v) in output.iter_mut().zip(input) {
                    *o = v * scale;
                }
                if output.iter().any(|value| value.is_nan()) {
                    return Err(TensorError::Parameters("softmax scaling produced a NaN"));
                }
                let sum = llama_quant::cpu_math::softmax_exp_sum_f32_in_place(output, max);
                let inv = (1.0 / sum) as f32;
                for o in output {
                    *o *= inv;
                }
            }
            Tensor::from_f32(n, &out)
        }
        RmsNorm { epsilon } | Norm { epsilon } => {
            eps(*epsilon)?;
            if n[0] == 0 {
                return Err(TensorError::Shape("normalization requires a nonempty row"));
            }
            let mut out = zeros(x.len())?;
            for (input, output) in x.chunks_exact(n[0]).zip(out.chunks_exact_mut(n[0])) {
                check_cancel(cancel)?;
                // The native NORM kernel deliberately uses F32 sums on its
                // strided first-axis path, and F64 on its contiguous-row path.
                let strided_norm = matches!(op, Norm { .. }) && a.desc.nb[0] != 4;
                let mean = if strided_norm {
                    input.iter().fold(0.0_f32, |sum, &v| sum + v) / n[0] as f32
                } else if matches!(op, Norm { .. }) {
                    (input.iter().fold(0.0_f64, |sum, &v| sum + v as f64) as f32) / n[0] as f32
                } else {
                    0.0
                };
                let variance = if strided_norm {
                    let mut squares = 0.0_f32;
                    for (o, &v) in output.iter_mut().zip(input) {
                        *o = v - mean;
                        squares += *o * *o;
                    }
                    squares / n[0] as f32
                } else if matches!(op, Norm { .. }) {
                    output.copy_from_slice(input);
                    llama_quant::cpu_math::center_f32_and_variance(output, mean) as f32
                } else {
                    // RMS_NORM retains the source scalar F32-product/F64-sum
                    // reduction; it does not use the NORM SSE2 variance path.
                    let mut squares = 0.0_f64;
                    for (o, &v) in output.iter_mut().zip(input) {
                        *o = v - mean;
                        squares += (*o * *o) as f64;
                    }
                    (squares / n[0] as f64) as f32
                };
                let inv = 1.0 / (variance + epsilon).sqrt();
                for o in output {
                    *o *= inv;
                }
            }
            Tensor::from_f32(n, &out)
        }
        Glu { .. } => unreachable!("typed GLU handled first"),
        DiagMask { past, zero } => {
            let mut out = x;
            for (i, v) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let at = index(i, n);
                let visible = past.checked_add(at[1]).ok_or(TensorError::Overflow)?;
                if at[0] > visible {
                    *v = if *zero { 0.0 } else { f32::NEG_INFINITY };
                }
            }
            Tensor::from_f32(n, &out)
        }
        Rope {
            dimensions,
            positions,
            mode,
            freq_base,
            freq_scale,
        } => {
            if dimensions % 2 != 0
                || *dimensions == 0
                || *dimensions > n[0]
                || positions.len() != n[2]
            {
                return Err(TensorError::Shape(
                    "RoPE requires positive even dimensions <= ne0 and one position per ne2",
                ));
            }
            if !freq_base.is_finite() || *freq_base <= 0.0 || !freq_scale.is_finite() {
                return Err(TensorError::Parameters("invalid RoPE frequencies"));
            }
            let mut out = x.clone();
            let theta_scale = freq_base.powf(-2.0 / (*dimensions as f32));
            for i3 in 0..n[3] {
                for (i2, &position) in positions.iter().enumerate() {
                    for i1 in 0..n[1] {
                        check_cancel(cancel)?;
                        let base = flat([0, i1, i2, i3], n);
                        let mut theta = position as f32;
                        for pair in 0..dimensions / 2 {
                            let angle = theta * freq_scale;
                            let co = angle.cos();
                            let si = angle.sin();
                            let (lo, hi) = match mode {
                                RopeMode::Normal => (2 * pair, 2 * pair + 1),
                                RopeMode::NeoX => (pair, pair + dimensions / 2),
                            };
                            out[base + lo] = x[base + lo] * co - x[base + hi] * si;
                            out[base + hi] = x[base + lo] * si + x[base + hi] * co;
                            theta *= theta_scale;
                        }
                    }
                }
            }
            Tensor::from_f32(n, &out)
        }
        Dup | Contiguous | Reshape { .. } | Permute { .. } | Transpose | GetRows => {
            unreachable!("view and gather handled first")
        }
        _ => unreachable!("extended operators handled first"),
    }
}

fn map(
    n: [usize; 4],
    x: &[f32],
    cancel: &AtomicBool,
    f: impl Fn(f32) -> Result<f32>,
) -> Result<Tensor> {
    let mut out = zeros(x.len())?;
    for (i, (&v, o)) in x.iter().zip(out.iter_mut()).enumerate() {
        if i % 4096 == 0 {
            check_cancel(cancel)?;
        }
        *o = f(v)?;
    }
    Tensor::from_f32(n, &out)
}

fn hard_sigmoid(x: f32) -> f32 {
    // C fmaxf returns its numeric argument when the other argument is NaN.
    let v = (x + 3.0) / 6.0;
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

pub(crate) fn unary(op: UnaryOp, x: f32) -> Result<f32> {
    use UnaryOp::*;
    Ok(match op {
        ABS => x.abs(),
        SGN => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        NEG => -x,
        STEP => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        TANH => x.tanh(),
        ELU => {
            if x > 0.0 {
                x
            } else {
                x.exp_m1()
            }
        }
        RELU => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        SIGMOID => 1.0 / (1.0 + (-x).exp()),
        GELU => {
            // The frozen CPU enables GGML_GELU_FP16: round its lookup input/output.
            if x <= -10.0 {
                0.0
            } else if x >= 10.0 {
                x
            } else {
                let h = llama_quant::f16_to_f32(llama_quant::f32_to_f16(x));
                let v = 0.5 * h * (1.0 + (0.797_884_6 * h * (1.0 + 0.044715 * h * h)).tanh());
                llama_quant::f16_to_f32(llama_quant::f32_to_f16(v))
            }
        }
        GELU_QUICK => x * (1.0 / (1.0 + (-1.702 * x).exp())),
        SILU => x / (1.0 + (-x).exp()),
        HARDSWISH => x * hard_sigmoid(x),
        HARDSIGMOID => hard_sigmoid(x),
        EXP => x.exp(),
        EXPM1 => x.exp() - 1.0,
        SOFTPLUS => {
            if x > 20.0 {
                x
            } else {
                x.exp().ln_1p()
            }
        }
        FLOOR => x.floor(),
        CEIL => x.ceil(),
        ROUND => x.round(),
        TRUNC => x.trunc(),
        GELU_ERF => {
            0.5 * x * (1.0 + crate::cpu_extra::erf_f32(x * std::f32::consts::FRAC_1_SQRT_2))
        }
        XIELU => {
            return Err(TensorError::Unsupported(
                "this unary operation has not been ported",
            ));
        }
    })
}

fn get_rows(a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    if matches!(
        a.dtype(),
        DType::I8 | DType::I16 | DType::I64 | DType::F64 | DType::Q8_K
    ) {
        return Err(TensorError::Unsupported(
            "source dtype is unsupported by frozen GGML GET_ROWS",
        ));
    }
    let an = a.shape();
    let bn = b.shape();
    let indices = b.to_i32()?;
    if an[2] != bn[1] || an[3] != bn[2] || bn[3] != 1 {
        return Err(TensorError::Shape(
            "GET_ROWS batch dimensions are incompatible",
        ));
    }
    for &v in &indices {
        if v < 0 || v as usize >= an[1] {
            return Err(TensorError::Bounds);
        }
    }
    let shape = [an[0], bn[0], bn[1], bn[2]];
    let out_count = count(shape)?;
    if a.dtype() == DType::I32 {
        let x = a.to_i32()?;
        let mut out = crate::reserved(out_count)?;
        for i3 in 0..shape[3] {
            for i2 in 0..shape[2] {
                for i1 in 0..shape[1] {
                    check_cancel(cancel)?;
                    let row = indices[flat([i1, i2, i3, 0], bn)] as usize;
                    let base = flat([0, row, i2, i3], an);
                    out.extend_from_slice(&x[base..base + an[0]]);
                }
            }
        }
        Tensor::from_i32(shape, &out)
    } else {
        let x = a.to_f32()?;
        let mut out = crate::reserved(out_count)?;
        for i3 in 0..shape[3] {
            for i2 in 0..shape[2] {
                for i1 in 0..shape[1] {
                    check_cancel(cancel)?;
                    let row = indices[flat([i1, i2, i3, 0], bn)] as usize;
                    let base = flat([0, row, i2, i3], an);
                    out.extend_from_slice(&x[base..base + an[0]]);
                }
            }
        }
        Tensor::from_f32(shape, &out)
    }
}
