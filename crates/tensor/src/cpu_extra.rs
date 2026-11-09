//! Additional scalar CPU operations. Arithmetic order follows the frozen GGML
//! CPU kernels; immutable tensors always produce separate output storage.
use crate::graph::check_cancel;
use crate::{
    DType, GluOp, Op, Result, Tensor, TensorDesc, TensorError, Triangle, UnaryOp, reserved, zeros,
};
use std::sync::atomic::AtomicBool;

pub(crate) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    use Op::*;
    if !matches!(
        op,
        View { .. }
            | Sum
            | Unary(_)
            | Square
            | Sqrt
            | Log
            | Sin
            | Cos
            | Glu { .. }
            | CumSum
            | ArgMax
            | CountEqual
            | RepeatBack { .. }
            | GroupNorm { .. }
            | L2Norm { .. }
            | OutProd
            | Diag
            | Pad { .. }
            | PadReflect1d { .. }
            | Roll { .. }
            | Arange { .. }
            | TimestepEmbedding { .. }
            | LeakyRelu { .. }
            | Tri { .. }
            | Fill { .. }
            | SiluBack
            | RmsNormBack { .. }
            | SoftmaxBack { .. }
            | GetRowsBack { .. }
            | Xielu { .. }
            | GluPacked { .. }
            | SwiGluOai { .. }
            | SwiGluClamp { .. }
            | SwiGluOaiPacked { .. }
            | SwiGluClampPacked { .. }
            | MatMul
    ) {
        return None;
    }
    Some(run(op, src, cancel))
}
fn count(shape: [usize; 4]) -> Result<usize> {
    TensorDesc::contiguous(DType::F32, shape)?.elements()
}
fn flat(i: [usize; 4], n: [usize; 4]) -> usize {
    i[0] + n[0] * (i[1] + n[1] * (i[2] + n[2] * i[3]))
}
fn coordinates(mut index: usize, n: [usize; 4]) -> [usize; 4] {
    std::array::from_fn(|axis| {
        let i = index % n[axis];
        index /= n[axis];
        i
    })
}
fn f32_values(a: &Tensor) -> Result<Vec<f32>> {
    if a.dtype() != DType::F32 {
        return Err(TensorError::Type {
            expected: DType::F32,
            actual: a.dtype(),
        });
    }
    a.to_f32()
}
fn float_values(a: &Tensor) -> Result<Vec<f32>> {
    if !matches!(a.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(TensorError::Type {
            expected: DType::F32,
            actual: a.dtype(),
        });
    }
    a.to_f32()
}
fn typed_map(
    a: &Tensor,
    cancel: &AtomicBool,
    mut operation: impl FnMut(f32) -> Result<f32>,
) -> Result<Tensor> {
    let mut output = float_values(a)?;
    for (i, value) in output.iter_mut().enumerate() {
        if i % 4096 == 0 {
            check_cancel(cancel)?;
        }
        *value = operation(*value)?;
    }
    typed_float(a.dtype(), a.shape(), &output)
}
/// BF16 GELU computes directly in F32, while the F16 table rounds the activation
/// before a subsequent GLU multiplication. F32 GELU uses the source's F16 table.
fn activation(kind: UnaryOp, x: f32, dtype: DType) -> Result<f32> {
    let value = if kind == UnaryOp::GELU && dtype != DType::F32 {
        0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
    } else {
        crate::cpu::unary(kind, x)?
    };
    Ok(
        if dtype == DType::F16 && matches!(kind, UnaryOp::GELU | UnaryOp::GELU_QUICK) {
            llama_quant::f16_to_f32(llama_quant::f32_to_f16(value))
        } else {
            value
        },
    )
}

fn same_shape(a: &Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() {
        Err(TensorError::Shape(
            "operator inputs must have identical shapes",
        ))
    } else {
        Ok(())
    }
}
fn finite(parameters: &[f32]) -> Result<()> {
    if parameters.iter().any(|v| !v.is_finite()) {
        Err(TensorError::Parameters(
            "operator parameters must be finite",
        ))
    } else {
        Ok(())
    }
}
fn epsilon(e: f32) -> Result<()> {
    finite(&[e])?;
    if e < 0.0 {
        Err(TensorError::Parameters("epsilon must be nonnegative"))
    } else {
        Ok(())
    }
}
fn nonempty_row(n: [usize; 4]) -> Result<()> {
    if n[0] == 0 {
        Err(TensorError::Shape("operator requires a nonempty row"))
    } else {
        Ok(())
    }
}
fn typed_float(dtype: DType, n: [usize; 4], values: &[f32]) -> Result<Tensor> {
    if dtype == DType::F32 {
        return Tensor::from_f32(n, values);
    }
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err(TensorError::Unsupported(
            "operator output dtype is unsupported",
        ));
    }
    let desc = TensorDesc::contiguous(dtype, n)?;
    let mut bytes = reserved(desc.byte_span()?)?;
    for &x in values {
        let bits = if dtype == DType::F16 {
            llama_quant::f32_to_f16(x)
        } else {
            llama_quant::f32_to_bf16(x)
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    Tensor::from_bytes(desc, bytes)
}

fn run(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Result<Tensor> {
    use Op::*;
    if let Arange { start, stop, step } = *op {
        finite(&[start, stop, step])?;
        if stop <= start || step <= 0.0 {
            return Err(TensorError::Parameters(
                "ARANGE requires stop > start and step > 0",
            ));
        }
        let steps = ((stop - start) / step).ceil();
        if !steps.is_finite() || steps as f64 > usize::MAX as f64 {
            return Err(TensorError::Overflow);
        }
        let n = [steps as usize, 1, 1, 1];
        let mut out = zeros(count(n)?)?;
        for (i, value) in out.iter_mut().enumerate() {
            if i % 4096 == 0 {
                check_cancel(cancel)?;
            }
            *value = start + step * i as f32;
        }
        return Tensor::from_f32(n, &out);
    }
    let a = src[0];
    let n = a.shape();
    match op {
        View {
            shape,
            strides,
            offset,
        } => a.view(
            TensorDesc::with_strides(a.dtype(), *shape, *strides)?,
            *offset,
        ),
        Sum => sum(a, cancel),
        Unary(kind) => {
            if *kind == UnaryOp::XIELU {
                return Err(TensorError::Unsupported(
                    "XIELU requires explicit constructor parameters",
                ));
            }
            if a.dtype() == DType::F32 && matches!(kind, UnaryOp::SILU | UnaryOp::GELU_QUICK) {
                let mut values = float_values(a)?;
                if n[0] != 0 {
                    for row in values.chunks_exact_mut(n[0]) {
                        check_cancel(cancel)?;
                        if *kind == UnaryOp::SILU {
                            llama_quant::cpu_math::silu_f32_in_place(row);
                        } else {
                            llama_quant::cpu_math::gelu_quick_f32_in_place(row);
                        }
                    }
                }
                return Tensor::from_f32(n, &values);
            }
            typed_map(a, cancel, |x| activation(*kind, x, a.dtype()))
        }
        Square => typed_map(a, cancel, |x| Ok(x * x)),
        Sqrt => typed_map(a, cancel, |x| Ok(x.sqrt())),
        Log => typed_map(a, cancel, |x| Ok(x.ln())),
        Sin => typed_map(a, cancel, |x| Ok(x.sin())),
        Cos => typed_map(a, cancel, |x| Ok(x.cos())),
        Glu { kind, swap } => glu(a, Some(src[1]), *kind, *swap, None, None, cancel),
        MatMul => matmul(a, src[1], cancel),
        CountEqual => count_equal(a, src[1], cancel),
        OutProd => out_prod(a, src[1], cancel),
        GetRowsBack { rows } => get_rows_back(a, src[1], *rows, cancel),
        GluPacked { kind, swap } => glu(a, None, *kind, *swap, None, None, cancel),
        SwiGluOai { alpha, limit } => glu(
            a,
            Some(src[1]),
            GluOp::SWIGLU_OAI,
            false,
            Some(*alpha),
            Some(*limit),
            cancel,
        ),
        SwiGluClamp { limit } => glu(
            a,
            Some(src[1]),
            GluOp::SWIGLU_CLAMP,
            false,
            None,
            Some(*limit),
            cancel,
        ),
        SwiGluOaiPacked { alpha, limit, swap } => glu(
            a,
            None,
            GluOp::SWIGLU_OAI,
            *swap,
            Some(*alpha),
            Some(*limit),
            cancel,
        ),
        SwiGluClampPacked { limit, swap } => glu(
            a,
            None,
            GluOp::SWIGLU_CLAMP,
            *swap,
            None,
            Some(*limit),
            cancel,
        ),
        Fill { value } => {
            if !matches!(a.dtype(), DType::F32 | DType::F16) {
                return Err(TensorError::Unsupported("FILL supports F32 or F16"));
            }
            let mut out = zeros(count(n)?)?;
            for (i, v) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                *v = *value;
            }
            typed_float(a.dtype(), n, &out)
        }
        LeakyRelu { negative_slope } => {
            finite(&[*negative_slope])?;
            if !matches!(a.dtype(), DType::F32 | DType::F16) {
                return Err(TensorError::Unsupported("LEAKY_RELU supports F32 or F16"));
            }
            let mut out = a.to_f32()?;
            for (i, v) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                *v = (if *v > 0.0 { *v } else { 0.0 })
                    + negative_slope * (if *v < 0.0 { *v } else { 0.0 });
            }
            typed_float(a.dtype(), n, &out)
        }
        Xielu {
            alpha_n,
            alpha_p,
            beta,
            epsilon: e,
        } => {
            finite(&[*alpha_n, *alpha_p, *beta, *e])?;
            let softplus = |v: f32| if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
            let negative = beta + softplus(*alpha_n);
            let positive = softplus(*alpha_p);
            let mut out = float_values(a)?;
            for (i, v) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                *v = if *v > 0.0 {
                    positive * *v * *v + beta * *v
                } else {
                    (v.min(*e).exp_m1() - *v) * negative + beta * *v
                };
            }
            typed_float(a.dtype(), n, &out)
        }
        SiluBack => {
            same_shape(a, src[1])?;
            if a.dtype() != src[1].dtype() || !matches!(a.dtype(), DType::F32 | DType::F16) {
                return Err(TensorError::Unsupported(
                    "SILU_BACK requires matching F32 or F16 tensors",
                ));
            }
            let gradient = a.to_f32()?;
            let input = src[1].to_f32()?;
            let mut out = zeros(gradient.len())?;
            for (i, o) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let x = input[i];
                let s = 1.0 / (1.0 + (-x).exp());
                *o = gradient[i] * s * (1.0 + x * (1.0 - s));
            }
            typed_float(a.dtype(), n, &out)
        }
        _ => run_f32(op, a, src, cancel),
    }
}

fn run_f32(op: &Op, a: &Tensor, src: &[&Tensor], cancel: &AtomicBool) -> Result<Tensor> {
    use Op::*;
    let n = a.shape();
    let x = f32_values(a)?;
    match op {
        CumSum => {
            nonempty_row(n)?;
            let mut out = zeros(x.len())?;
            for (input, output) in x.chunks_exact(n[0]).zip(out.chunks_exact_mut(n[0])) {
                check_cancel(cancel)?;
                output[0] = input[0];
                for i in 1..input.len() {
                    output[i] = output[i - 1] + input[i];
                }
            }
            Tensor::from_f32(n, &out)
        }
        ArgMax => {
            nonempty_row(n)?;
            if n[2] != 1 || n[3] != 1 || n[0] > i32::MAX as usize {
                return Err(TensorError::Shape(
                    "ARGMAX requires a matrix with an I32 row width",
                ));
            }
            let mut out = reserved(n[1])?;
            for row in x.chunks_exact(n[0]) {
                check_cancel(cancel)?;
                let mut maximum = f32::NEG_INFINITY;
                let mut at = 0i32;
                for (i, &v) in row.iter().enumerate() {
                    // GGML's MAX macro takes its second argument on equality
                    // or unordered comparison; the final tied index wins.
                    maximum = if maximum > v { maximum } else { v };
                    if maximum == v {
                        at = i as i32;
                    }
                }
                out.push(at);
            }
            Tensor::from_i32([n[1], 1, 1, 1], &out)
        }
        RepeatBack { shape } => {
            if !shape
                .iter()
                .zip(n)
                .all(|(&small, big)| small > 0 && big.is_multiple_of(small))
            {
                return Err(TensorError::Shape("REPEAT_BACK output must tile its input"));
            }
            let mut out = zeros(count(*shape)?)?;
            let repeats: [usize; 4] = std::array::from_fn(|d| n[d] / shape[d]);
            // For each output element the source visits repeat indices with
            // axis 0 fastest; every addition rounds to F32.
            for i3 in 0..repeats[3] {
                for k3 in 0..shape[3] {
                    for i2 in 0..repeats[2] {
                        for k2 in 0..shape[2] {
                            for i1 in 0..repeats[1] {
                                for k1 in 0..shape[1] {
                                    check_cancel(cancel)?;
                                    let dst = flat([0, k1, k2, k3], *shape);
                                    for i0 in 0..repeats[0] {
                                        let source = flat(
                                            [
                                                i0 * shape[0],
                                                i1 * shape[1] + k1,
                                                i2 * shape[2] + k2,
                                                i3 * shape[3] + k3,
                                            ],
                                            n,
                                        );
                                        for k0 in 0..shape[0] {
                                            out[dst + k0] += x[source + k0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Tensor::from_f32(*shape, &out)
        }
        GroupNorm { groups, epsilon: e } => group_norm(n, &x, *groups, *e, cancel),
        L2Norm { epsilon: e } => {
            epsilon(*e)?;
            nonempty_row(n)?;
            let mut out = zeros(x.len())?;
            for (input, output) in x.chunks_exact(n[0]).zip(out.chunks_exact_mut(n[0])) {
                check_cancel(cancel)?;
                let sum = input.iter().fold(0.0_f64, |sum, v| sum + (v * v) as f64);
                // sqrtf receives the rounded F32 sum, then fmaxf applies eps.
                let scale = 1.0 / (sum as f32).sqrt().max(*e);
                for (o, &v) in output.iter_mut().zip(input) {
                    *o = v * scale;
                }
            }
            Tensor::from_f32(n, &out)
        }
        Diag => {
            if n[1] != 1 {
                return Err(TensorError::Shape("DIAG requires ne1 == 1"));
            }
            let shape = [n[0], n[0], n[2], n[3]];
            let mut out = zeros(count(shape)?)?;
            for i3 in 0..n[3] {
                for i2 in 0..n[2] {
                    check_cancel(cancel)?;
                    for i in 0..n[0] {
                        out[flat([i, i, i2, i3], shape)] = x[flat([i, 0, i2, i3], n)];
                    }
                }
            }
            Tensor::from_f32(shape, &out)
        }
        Pad {
            left,
            right,
            circular,
        } => {
            let mut shape = [0usize; 4];
            for d in 0..4 {
                let value = n[d] as i128 + left[d] as i128 + right[d] as i128;
                shape[d] = usize::try_from(value)
                    .map_err(|_| TensorError::Shape("padding removes an entire axis"))?;
                if shape[d] == 0 || n[d] == 0 {
                    return Err(TensorError::Shape(
                        "padding requires positive input and output axes",
                    ));
                }
                if *circular && left[d] as i128 > n[d] as i128 {
                    return Err(TensorError::Bounds);
                }
                if !circular
                    && (-(left[d] as i128) > n[d] as i128 || -(right[d] as i128) > n[d] as i128)
                {
                    return Err(TensorError::Bounds);
                }
            }
            let mut out = zeros(count(shape)?)?;
            for (i, value) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let at = coordinates(i, shape);
                let source: [i128; 4] = std::array::from_fn(|d| at[d] as i128 - left[d] as i128);
                if *circular {
                    let wrapped =
                        std::array::from_fn(|d| source[d].rem_euclid(n[d] as i128) as usize);
                    *value = x[flat(wrapped, n)];
                } else if (0..4).all(|d| source[d] >= 0 && source[d] < n[d] as i128) {
                    *value = x[flat(std::array::from_fn(|d| source[d] as usize), n)];
                }
            }
            Tensor::from_f32(shape, &out)
        }
        PadReflect1d { left, right } => {
            if *left >= n[0] || *right >= n[0] {
                return Err(TensorError::Parameters(
                    "reflection pads must be smaller than the input row",
                ));
            }
            let mut shape = n;
            shape[0] = n[0]
                .checked_add(*left)
                .and_then(|v| v.checked_add(*right))
                .ok_or(TensorError::Overflow)?;
            let mut out = zeros(count(shape)?)?;
            for (input, output) in x.chunks_exact(n[0]).zip(out.chunks_exact_mut(shape[0])) {
                check_cancel(cancel)?;
                output[*left..left + n[0]].copy_from_slice(input);
                for i in 1..=*left {
                    output[left - i] = input[i];
                }
                for i in 1..=*right {
                    output[left + n[0] - 1 + i] = input[n[0] - 1 - i];
                }
            }
            Tensor::from_f32(shape, &out)
        }
        Roll { shifts } => {
            if (0..4).any(|d| n[d] == 0 || (shifts[d] as i128).abs() >= n[d] as i128) {
                return Err(TensorError::Parameters(
                    "ROLL shifts must lie strictly inside each signed axis length",
                ));
            }
            let mut out = zeros(x.len())?;
            for (i, value) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let at = coordinates(i, n);
                let from = std::array::from_fn(|d| {
                    (at[d] as i128 - shifts[d] as i128).rem_euclid(n[d] as i128) as usize
                });
                *value = x[flat(from, n)];
            }
            Tensor::from_f32(n, &out)
        }
        TimestepEmbedding {
            dimensions,
            max_period,
        } => {
            if *dimensions == 0
                || *dimensions > i32::MAX as usize
                || *max_period == 0
                || *max_period > i32::MAX as usize
                || n[1..] != [1, 1, 1]
            {
                return Err(TensorError::Parameters(
                    "timestep embedding requires a vector and positive I32 dimensions/period",
                ));
            }
            let shape = [*dimensions, n[0], 1, 1];
            let mut out = zeros(count(shape)?)?;
            let half = dimensions / 2;
            for (i, &timestep) in x.iter().enumerate() {
                check_cancel(cancel)?;
                for j in 0..half {
                    let frequency = (-(*max_period as f32).ln() * j as f32 / half as f32).exp();
                    let angle = timestep * frequency;
                    out[i * dimensions + j] = angle.cos();
                    out[i * dimensions + j + half] = angle.sin();
                }
            }
            Tensor::from_f32(shape, &out)
        }
        Tri { triangle } => {
            if n[0] != n[1] {
                return Err(TensorError::Shape("TRI requires square matrices"));
            }
            let mut out = x;
            for (i, value) in out.iter_mut().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                let at = coordinates(i, n);
                let keep = match triangle {
                    Triangle::Lower => at[0] < at[1],
                    Triangle::LowerDiagonal => at[0] <= at[1],
                    Triangle::Upper => at[0] > at[1],
                    Triangle::UpperDiagonal => at[0] >= at[1],
                };
                if !keep {
                    *value = 0.0;
                }
            }
            Tensor::from_f32(n, &out)
        }
        RmsNormBack { epsilon: e } | SoftmaxBack { scale: e } => {
            same_shape(a, src[1])?;
            nonempty_row(n)?;
            if matches!(op, RmsNormBack { .. }) {
                epsilon(*e)?;
            } else {
                finite(&[*e])?;
            }
            let y = f32_values(src[1])?;
            let mut out = zeros(x.len())?;
            for ((gradient, input), output) in x
                .chunks_exact(n[0])
                .zip(y.chunks_exact(n[0]))
                .zip(out.chunks_exact_mut(n[0]))
            {
                check_cancel(cancel)?;
                let dot = gradient
                    .iter()
                    .zip(input)
                    .fold(0.0_f64, |sum, (&g, &v)| sum + (g * v) as f64);
                if matches!(op, RmsNormBack { .. }) {
                    let squares = input.iter().fold(0.0_f64, |sum, v| sum + (v * v) as f64) as f32;
                    let rrms = 1.0 / (squares / n[0] as f32 + e).sqrt();
                    let scale_x = (-dot) as f32 / (squares + e * n[0] as f32);
                    for ((o, &g), &v) in output.iter_mut().zip(gradient).zip(input) {
                        *o = (g + v * scale_x) * rrms;
                    }
                } else {
                    let dot = dot as f32;
                    for ((o, &g), &v) in output.iter_mut().zip(gradient).zip(input) {
                        *o = e * (g - dot) * v;
                    }
                }
            }
            Tensor::from_f32(n, &out)
        }
        _ => unreachable!("extended operator dispatch"),
    }
}

fn group_norm(
    n: [usize; 4],
    x: &[f32],
    groups: usize,
    eps: f32,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    epsilon(eps)?;
    if groups == 0 || groups > i32::MAX as usize || n[0] == 0 || n[1] == 0 || n[2] == 0 {
        return Err(TensorError::Parameters(
            "GROUP_NORM requires positive channels, rows and I32 groups",
        ));
    }
    if x.is_empty() {
        return Tensor::from_f32(n, &[]);
    }
    let per_group = n[2].div_ceil(groups);
    let mut out = zeros(x.len())?;
    for start in (0..n[2]).step_by(per_group) {
        let end = start.saturating_add(per_group).min(n[2]);
        let elements = n[0]
            .checked_mul(n[1])
            .and_then(|v| v.checked_mul(end - start))
            .ok_or(TensorError::Overflow)?;
        for i3 in 0..n[3] {
            check_cancel(cancel)?;
            let mut sum = 0.0_f64;
            for i2 in start..end {
                for i1 in 0..n[1] {
                    let base = flat([0, i1, i2, i3], n);
                    let row_sum = x[base..base + n[0]]
                        .iter()
                        .fold(0.0_f64, |sum, &v| sum + v as f64);
                    sum += row_sum;
                }
            }
            let mean = (sum / elements as f64) as f32;
            let mut squares = 0.0_f64;
            for i2 in start..end {
                for i1 in 0..n[1] {
                    let base = flat([0, i1, i2, i3], n);
                    let mut row_sum = 0.0_f64;
                    for i0 in 0..n[0] {
                        let v = x[base + i0] - mean;
                        out[base + i0] = v;
                        row_sum += (v * v) as f64;
                    }
                    squares += row_sum;
                }
            }
            let variance = (squares / elements as f64) as f32;
            let scale = 1.0 / (variance + eps).sqrt();
            for i2 in start..end {
                for i1 in 0..n[1] {
                    let base = flat([0, i1, i2, i3], n);
                    for v in &mut out[base..base + n[0]] {
                        *v *= scale;
                    }
                }
            }
        }
    }
    Tensor::from_f32(n, &out)
}

fn count_equal(a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    same_shape(a, b)?;
    for tensor in [a, b] {
        if tensor.dtype() != DType::I32 {
            return Err(TensorError::Type {
                expected: DType::I32,
                actual: tensor.dtype(),
            });
        }
    }
    let n = a.shape();
    let mut equal = 0i64;
    if !n.contains(&0) {
        let nr = n[1]
            .checked_mul(n[2])
            .and_then(|v| v.checked_mul(n[3]))
            .ok_or(TensorError::Overflow)?;
        for row in 0..nr {
            check_cancel(cancel)?;
            // Preserve the frozen kernel's indexing, including its unusual
            // higher-axis formula. Reject any resulting out-of-buffer access.
            let i3 = (row / (n[2] * n[1])) as i128;
            let i2 = (row as i128 - i3 * n[3] as i128) / n[1] as i128;
            let i1 = row as i128 - i3 * n[3] as i128 - i2 * n[2] as i128;
            for i0 in 0..n[0] {
                let read = |t: &Tensor| -> Result<i32> {
                    let offset = i0 as i128 * t.desc.nb[0] as i128
                        + i1 * t.desc.nb[1] as i128
                        + i2 * t.desc.nb[2] as i128
                        + i3 * t.desc.nb[3] as i128;
                    let offset = usize::try_from(offset).map_err(|_| TensorError::Bounds)?;
                    let end = offset.checked_add(4).ok_or(TensorError::Overflow)?;
                    if end > t.desc.byte_span()? {
                        return Err(TensorError::Bounds);
                    }
                    let bytes = &t.data[t.offset + offset..t.offset + end];
                    Ok(i32::from_le_bytes(
                        bytes.try_into().expect("four I32 bytes"),
                    ))
                };
                if read(a)? == read(b)? {
                    equal = equal.checked_add(1).ok_or(TensorError::Overflow)?;
                }
            }
        }
    }
    Tensor::from_bytes(
        TensorDesc::contiguous(DType::I64, [1; 4])?,
        equal.to_le_bytes().to_vec(),
    )
}

pub(super) fn matmul(a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let an = a.shape();
    let bn = b.shape();
    let activation = llama_quant::matvec_activation_dtype(a.dtype())?;
    if an[0] != bn[0]
        || an[0] == 0
        || an[2] == 0
        || an[3] == 0
        || !bn[2].is_multiple_of(an[2])
        || !bn[3].is_multiple_of(an[3])
    {
        return Err(TensorError::Shape(
            "GGML matmul reduction or grouped batch dimensions are incompatible",
        ));
    }
    if a.desc.nb[0] != a.dtype().type_size()
        || b.desc.nb[0] != b.dtype().type_size()
        || a.desc.nb[0] > a.desc.nb[1]
    {
        return Err(TensorError::Layout(
            "GGML matmul requires contiguous blocks within rows and an untransposed left operand",
        ));
    }
    let rhs_direct = b.dtype() == activation;
    let widen = activation == DType::F32 && matches!(b.dtype(), DType::F16 | DType::BF16);
    if !rhs_direct && b.dtype() != DType::F32 && !widen {
        return Err(TensorError::Unsupported(
            "matmul RHS must be F32, its native activation dtype, or a source-supported F16/BF16 widening",
        ));
    }
    let shape = [an[1], bn[1], bn[2], bn[3]];
    let mut out = zeros(count(shape)?)?;
    let left = a.contiguous()?;
    let packed_right = if rhs_direct && activation.block_size() > 1 {
        Some(b.contiguous()?)
    } else {
        None
    };
    let decoded_right = if packed_right.is_none() {
        Some(b.to_f32()?)
    } else {
        None
    };
    let decoded_left = if a.dtype().block_size() == 1 {
        Some(a.to_f32()?)
    } else {
        None
    };
    let left_row_bytes = a.dtype().row_bytes(an[0])?;
    let right_row_bytes = activation.row_bytes(an[0])?;
    let repeat2 = bn[2] / an[2];
    let repeat3 = bn[3] / an[3];
    for i3 in 0..shape[3] {
        for i2 in 0..shape[2] {
            for rb in 0..bn[1] {
                check_cancel(cancel)?;
                let right_base = flat([0, rb, i2, i3], bn);
                let conversion;
                let right_bytes: &[u8] = if let Some(right) = &packed_right {
                    let start = right_base / an[0] * right_row_bytes;
                    &right.data[start..start + right_row_bytes]
                } else if a.dtype().block_size() > 1 {
                    let y = decoded_right.as_ref().expect("decoded RHS");
                    conversion =
                        llama_quant::quantize(activation, &y[right_base..right_base + an[0]])?;
                    &conversion
                } else {
                    &[]
                };
                let converted_scalars;
                let scalars: &[f32] = if let Some(y) = &decoded_right {
                    let row = &y[right_base..right_base + an[0]];
                    if activation == DType::BF16 && b.dtype() == DType::F32 {
                        let mut values = reserved(an[0])?;
                        values.extend(
                            row.iter()
                                .map(|&v| llama_quant::bf16_to_f32(llama_quant::f32_to_bf16(v))),
                        );
                        converted_scalars = values;
                        &converted_scalars
                    } else {
                        row
                    }
                } else {
                    &[]
                };
                for ra in 0..an[1] {
                    let left_base = flat([0, ra, i2 / repeat2, i3 / repeat3], an);
                    let value = if let Some(x) = &decoded_left {
                        if a.dtype() == DType::F8_E4M3 {
                            let mut sum = 0.0_f32;
                            for k in 0..an[0] {
                                sum += x[left_base + k] * scalars[k];
                            }
                            sum
                        } else {
                            let mut sum = 0.0_f64;
                            for k in 0..an[0] {
                                sum += (x[left_base + k] * scalars[k]) as f64;
                            }
                            sum as f32
                        }
                    } else {
                        let start = left_base / an[0] * left_row_bytes;
                        llama_quant::dot_packed(
                            a.dtype(),
                            &left.data[start..start + left_row_bytes],
                            right_bytes,
                            an[0],
                        )?
                    };
                    out[flat([ra, rb, i2, i3], shape)] = value;
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}

/// The source groups contiguous rows before the outer sum. Its F16/BF16
/// reductions accumulate in F32, while F32 reductions accumulate in F64.
fn sum(a: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(TensorError::Unsupported("SUM supports F32, F16 or BF16"));
    }
    let width = a.shape()[0];
    if dtype != DType::F32 && a.desc.nb[0] != 2 {
        return Err(TensorError::Unsupported(
            "F16/BF16 SUM requires contiguous rows",
        ));
    }
    let x = a.to_f32()?;
    let value = if dtype == DType::F32 {
        let mut total = 0.0_f64;
        if width > 0 && a.desc.nb[0] == 4 {
            for row in x.chunks_exact(width) {
                check_cancel(cancel)?;
                let mut row_sum = 0.0_f64;
                for &v in row {
                    row_sum += v as f64;
                }
                total += row_sum;
            }
        } else {
            for (i, &v) in x.iter().enumerate() {
                if i % 4096 == 0 {
                    check_cancel(cancel)?;
                }
                total += v as f64;
            }
        }
        total as f32
    } else {
        let mut total = 0.0_f32;
        if width > 0 {
            for row in x.chunks_exact(width) {
                check_cancel(cancel)?;
                let mut row_sum = 0.0_f32;
                for &v in row {
                    row_sum += v;
                }
                total += row_sum;
            }
        }
        total
    };
    typed_float(dtype, [1; 4], &[value])
}

fn out_prod(a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let an = a.shape();
    let bn = b.shape();
    if a.dtype() == DType::BF16 || llama_quant::matvec_activation_dtype(a.dtype()).is_err() {
        return Err(TensorError::Unsupported(
            "OUT_PROD left dtype has no frozen CPU implementation",
        ));
    }
    if an[1] != bn[1]
        || an[2] == 0
        || an[3] == 0
        || !bn[2].is_multiple_of(an[2])
        || !bn[3].is_multiple_of(an[3])
    {
        return Err(TensorError::Shape(
            "OUT_PROD reduction or grouped batch dimensions are incompatible",
        ));
    }
    if a.dtype() != DType::F32 && (an[2] != bn[2] || an[3] != bn[3]) {
        return Err(TensorError::Unsupported(
            "frozen non-F32 OUT_PROD does not broadcast batches",
        ));
    }
    let x = a.to_f32()?;
    let y = f32_values(b)?;
    let shape = [an[0], bn[0], bn[2], bn[3]];
    let mut out = zeros(count(shape)?)?;
    let repeat2 = bn[2] / an[2];
    let repeat3 = bn[3] / an[3];
    for i3 in 0..shape[3] {
        for i2 in 0..shape[2] {
            for row in 0..shape[1] {
                check_cancel(cancel)?;
                let target = flat([0, row, i2, i3], shape);
                for k in 0..an[1] {
                    let left = flat([0, k, i2 / repeat2, i3 / repeat3], an);
                    let value = y[flat([row, k, i2, i3], bn)];
                    for i in 0..an[0] {
                        out[target + i] += x[left + i] * value;
                    }
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}

fn get_rows_back(a: &Tensor, b: &Tensor, rows: usize, cancel: &AtomicBool) -> Result<Tensor> {
    let n = a.shape();
    let bn = b.shape();
    if !matches!(a.dtype(), DType::F32 | DType::F16)
        || n[2..] != [1, 1]
        || bn[1..] != [1, 1, 1]
        || bn[0] != n[1]
    {
        return Err(TensorError::Shape(
            "GET_ROWS_BACK requires an F32/F16 matrix and one I32 vector index per row",
        ));
    }
    let ids = b.to_i32()?;
    for &row in &ids {
        if row < 0 || row as usize >= rows {
            return Err(TensorError::Bounds);
        }
    }
    let x = a.to_f32()?;
    let shape = [n[0], rows, 1, 1];
    let mut out = zeros(count(shape)?)?;
    for (i, &row) in ids.iter().enumerate() {
        check_cancel(cancel)?;
        for j in 0..n[0] {
            out[row as usize * n[0] + j] += x[i * n[0] + j];
        }
    }
    Tensor::from_f32(shape, &out)
}

#[allow(clippy::too_many_arguments)] // Mirrors distinct source GLU parameters.
fn glu(
    a: &Tensor,
    b: Option<&Tensor>,
    kind: GluOp,
    swap: bool,
    alpha: Option<f32>,
    limit: Option<f32>,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let an = a.shape();
    let x = float_values(a)?;
    let y = if let Some(b) = b {
        same_shape(a, b)?;
        if a.dtype() != b.dtype() {
            return Err(TensorError::Type {
                expected: a.dtype(),
                actual: b.dtype(),
            });
        }
        Some(float_values(b)?)
    } else {
        None
    };
    let mut shape = an;
    if b.is_none() {
        // The frozen constructor deliberately floors odd packed row widths.
        shape[0] /= 2;
    }
    if a.dtype() == DType::BF16 && matches!(kind, GluOp::SWIGLU_OAI | GluOp::SWIGLU_CLAMP) {
        return Err(TensorError::Unsupported(
            "frozen clamped GLU kernels do not support BF16",
        ));
    }
    if matches!(kind, GluOp::SWIGLU_OAI | GluOp::SWIGLU_CLAMP) {
        let limit = limit.ok_or(TensorError::Parameters(
            "clamped GLU requires an explicit limit",
        ))?;
        finite(&[limit])?;
        if limit < 0.0 {
            return Err(TensorError::Parameters(
                "GLU clamp limit must be nonnegative",
            ));
        }
    }
    if kind == GluOp::SWIGLU_OAI {
        finite(&[alpha.ok_or(TensorError::Parameters(
            "OAI GLU requires an explicit alpha",
        ))?])?;
    }
    let mut out = zeros(count(shape)?)?;
    if a.dtype() == DType::F32 && matches!(kind, GluOp::SWIGLU | GluOp::GEGLU_QUICK) {
        if shape[0] != 0 {
            for (row_index, output) in out.chunks_exact_mut(shape[0]).enumerate() {
                check_cancel(cancel)?;
                let offset = row_index * an[0];
                let first = &x[offset..offset + shape[0]];
                let second = if let Some(y) = &y {
                    &y[offset..offset + shape[0]]
                } else {
                    &x[offset + shape[0]..offset + 2 * shape[0]]
                };
                let (gate, value) = if swap {
                    (second, first)
                } else {
                    (first, second)
                };
                output.copy_from_slice(gate);
                if kind == GluOp::SWIGLU {
                    llama_quant::cpu_math::silu_f32_in_place(output);
                } else {
                    llama_quant::cpu_math::gelu_quick_f32_in_place(output);
                }
                for (output, value) in output.iter_mut().zip(value) {
                    *output = glu_product(a.dtype(), *output, *value);
                }
            }
        }
        return Tensor::from_f32(shape, &out);
    }
    for (i, o) in out.iter_mut().enumerate() {
        if i % 4096 == 0 {
            check_cancel(cancel)?;
        }
        let (gate, value) = if let Some(y) = &y {
            (x[i], y[i])
        } else {
            let row = i / shape[0];
            let col = i % shape[0];
            (x[row * an[0] + col], x[row * an[0] + col + shape[0]])
        };
        let (gate, value) = if swap { (value, gate) } else { (gate, value) };
        *o = match kind {
            GluOp::REGLU => {
                if gate > 0.0 {
                    gate * value
                } else {
                    0.0
                }
            }
            GluOp::GEGLU => activation(UnaryOp::GELU, gate, a.dtype())? * value,
            GluOp::SWIGLU => glu_product(a.dtype(), gate / (1.0 + (-gate).exp()), value),
            GluOp::GEGLU_ERF => crate::cpu::unary(UnaryOp::GELU_ERF, gate)? * value,
            GluOp::GEGLU_QUICK => glu_product(
                a.dtype(),
                activation(UnaryOp::GELU_QUICK, gate, a.dtype())?,
                value,
            ),
            GluOp::SWIGLU_OAI | GluOp::SWIGLU_CLAMP => {
                let limit = limit.expect("validated limit");
                // std::min/std::clamp preserve NaN first arguments.
                let gate = if limit < gate { limit } else { gate };
                let value = if value < -limit {
                    -limit
                } else if limit < value {
                    limit
                } else {
                    value
                };
                let slope = alpha.unwrap_or(1.0);
                let silu = gate / (1.0 + (slope * -gate).exp());
                silu * if kind == GluOp::SWIGLU_OAI {
                    value + 1.0
                } else {
                    value
                }
            }
        };
    }
    typed_float(a.dtype(), shape, &out)
}

// Frozen scalar/SSE2 GLU kernels select the activation NaN first for F32.
// GNU's inline BF16 loop has the value operand first in its final multiply.
// Express that captured operand policy explicitly: Rust does not specify which
// NaN payload an ordinary multiplication selects. Finite arithmetic is unchanged.
fn glu_product(dtype: DType, activation: f32, value: f32) -> f32 {
    let value_first = dtype == DType::BF16
        && cfg!(all(
            target_arch = "x86_64",
            target_os = "linux",
            target_env = "gnu"
        ));
    let (first, second) = if value_first {
        (value, activation)
    } else {
        (activation, value)
    };
    if first.is_nan() {
        f32::from_bits(first.to_bits() | 0x0040_0000)
    } else if second.is_nan() {
        f32::from_bits(second.to_bits() | 0x0040_0000)
    } else {
        activation * value
    }
}

/// The original graph calls the standard OS C-runtime erff. Use that same
/// independently verified boundary when available; retain the mathematical
/// implementation on targets whose system ABI has not been exercised.
pub(crate) fn erf_f32(x: f32) -> f32 {
    #[cfg(all(
        target_arch = "x86_64",
        any(
            all(target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", target_env = "gnu")
        )
    ))]
    if let Some(value) = llama_platform_loader::math::erf_f32(x) {
        return value;
    }
    portable_erf_f32(x)
}

/// Positive-term erf series, evaluated in F64 then rounded once to F32.
/// For |x| >= 4 the true result rounds to +/-1 in F32. Its independent 80-digit
/// accuracy corpus remains separate from platform-specific native parity.
pub(crate) fn portable_erf_f32(x: f32) -> f32 {
    if x.is_nan() || x == 0.0 {
        return x;
    }
    let magnitude = x.abs() as f64;
    if magnitude >= 4.0 {
        return 1.0_f32.copysign(x);
    }
    let square = magnitude * magnitude;
    let mut term = magnitude;
    let mut sum = term;
    for n in 1..160 {
        term *= 2.0 * square / (2 * n + 1) as f64;
        sum += term;
        // Subsequent ratios decrease, so a geometric series bounds the
        // omitted positive tail. Stop only below one eighth of F64 epsilon.
        let ratio = 2.0 * square / (2 * n + 3) as f64;
        if ratio < 1.0 && term * ratio / (1.0 - ratio) <= sum * f64::EPSILON * 0.125 {
            break;
        }
    }
    ((2.0 / std::f64::consts::PI.sqrt() * (-square).exp() * sum) as f32).copysign(x)
}

#[cfg(test)]
mod accuracy_tests {
    #[test]
    fn erf_rounding_matches_independent_80_digit_references() {
        let bytes = include_bytes!("../tests/fixtures/erf-high-precision-reference.bin");
        assert_eq!(&bytes[..8], b"ERFMP001");
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 12 + count * 16);
        let mut maximum_error = 0.0_f64;
        for record in bytes[12..].as_chunks::<16>().0 {
            let x = f32::from_le_bytes(record[..4].try_into().unwrap());
            let expected = f32::from_le_bytes(record[4..8].try_into().unwrap());
            let reference = f64::from_le_bytes(record[8..16].try_into().unwrap());
            let actual = super::portable_erf_f32(x);
            assert_eq!(actual.to_bits(), expected.to_bits(), "erf({x:?})");
            assert_eq!(
                super::portable_erf_f32(-x).to_bits(),
                (-expected).to_bits(),
                "erf({:?})",
                -x
            );
            maximum_error = maximum_error.max((actual as f64 - reference).abs());
        }
        println!(
            "{} independent 80-digit ERF checks including both signs; maximum binary32 absolute rounding error {}",
            count * 2,
            maximum_error
        );
    }
}
