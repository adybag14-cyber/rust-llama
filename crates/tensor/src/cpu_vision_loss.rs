//! Frozen CPU window/relative-position, triangular solve and loss operators.
//! Source flat byte addressing and one-worker loss reduction are intentional.
use crate::graph::check_cancel;
use crate::{DType, Op, Result, Tensor, TensorDesc, TensorError, reserved, zeros};
use std::sync::atomic::AtomicBool;

pub(super) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    Some(match op {
        Op::WindowPartition { window } => partition(src[0], *window, cancel),
        Op::WindowUnpartition {
            width,
            height,
            window,
        } => unpartition(src[0], *width, *height, *window, cancel),
        Op::GetRelativePosition { query, key } => relative(src[0], *query, *key, cancel),
        Op::AddRelativePosition => add_relative(src[0], src[1], src[2], cancel),
        Op::SolveTriangular => solve(src[0], src[1], cancel),
        Op::CrossEntropyLoss => loss(src[0], src[1], cancel),
        Op::CrossEntropyLossBack => loss_back(src[0], src[1], src[2], cancel),
        _ => return None,
    })
}

// Original MSVC and GNU allocate operands differently for commutative
// arithmetic. Quiet single NaNs explicitly; two NaNs use the measured operand
// policy instead of depending on Rust/LLVM register choices.
fn nan_pair(a: f32, b: f32, prefer_right: bool) -> Result<Option<f32>> {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => Ok(None),
        (true, false) => Ok(Some(f32::from_bits(a.to_bits() | 0x0040_0000))),
        (false, true) => Ok(Some(f32::from_bits(b.to_bits() | 0x0040_0000))),
        (true, true) => {
            let left = a.to_bits() | 0x0040_0000;
            let right = b.to_bits() | 0x0040_0000;
            if left != right
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
                    )
                ))
            {
                return Err(TensorError::Unsupported(
                    "competing NaN policy is unverified on this target",
                ));
            }
            Ok(Some(f32::from_bits(if prefer_right {
                right
            } else {
                left
            })))
        }
    }
}
const MSVC_RIGHT_NAN: bool = cfg!(all(
    target_arch = "x86_64",
    target_os = "windows",
    target_env = "msvc"
));
fn fadd(a: f32, b: f32, prefer_right: bool) -> Result<f32> {
    Ok(nan_pair(a, b, prefer_right)?.unwrap_or(a + b))
}
fn fmul(a: f32, b: f32) -> Result<f32> {
    Ok(nan_pair(a, b, MSVC_RIGHT_NAN)?.unwrap_or(a * b))
}
fn fsub(a: f32, b: f32) -> Result<f32> {
    Ok(nan_pair(a, b, false)?.unwrap_or(a - b))
}
fn fdiv(a: f32, b: f32) -> Result<f32> {
    Ok(nan_pair(a, b, false)?.unwrap_or(a / b))
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(TensorError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(TensorError::Overflow)
}
fn positive_i32(value: usize) -> Result<()> {
    if value == 0 || value > i32::MAX as usize {
        return Err(TensorError::Parameters(
            "CPU geometry must be positive signed I32",
        ));
    }
    Ok(())
}
fn dimensions(tensor: &Tensor) -> Result<()> {
    for n in tensor.shape() {
        positive_i32(n)?;
    }
    Ok(())
}
fn require_type(tensor: &Tensor, dtype: DType) -> Result<()> {
    if tensor.dtype() != dtype {
        return Err(TensorError::Type {
            expected: dtype,
            actual: tensor.dtype(),
        });
    }
    dimensions(tensor)
}
fn require_contiguous(tensor: &Tensor) -> Result<()> {
    if !tensor.desc().is_contiguous() {
        return Err(TensorError::Layout("operator requires contiguous input"));
    }
    Ok(())
}
fn raw(tensor: &Tensor) -> Result<&[u8]> {
    let end = add(tensor.offset, tensor.desc().byte_span()?)?;
    tensor
        .data
        .get(tensor.offset..end)
        .ok_or(TensorError::Bounds)
}
fn bytes_at(bytes: &[u8], index: usize, size: usize) -> Result<&[u8]> {
    let start = mul(index, size)?;
    bytes
        .get(start..add(start, size)?)
        .ok_or(TensorError::Bounds)
}
fn float_at(bytes: &[u8], index: usize) -> Result<f32> {
    Ok(f32::from_le_bytes(
        bytes_at(bytes, index, 4)?.try_into().expect("four bytes"),
    ))
}
fn output_bytes(dtype: DType, shape: [usize; 4]) -> Result<(TensorDesc, Vec<u8>)> {
    let desc = TensorDesc::contiguous(dtype, shape)?;
    let len = desc.byte_span()?;
    if len > 256 * 1024 * 1024 {
        return Err(TensorError::Unsupported(
            "window/relative output exceeds the 256 MiB allocation limit",
        ));
    }
    let mut bytes = reserved(len)?;
    bytes.resize(len, 0);
    Ok((desc, bytes))
}

fn partition(a: &Tensor, window: usize, cancel: &AtomicBool) -> Result<Tensor> {
    require_type(a, DType::F32)?;
    positive_i32(window)?;
    let [channels, width, height, batches] = a.shape();
    if batches != 1 {
        return Err(TensorError::Shape("WIN_PART requires one image"));
    }
    let nx = width.div_ceil(window);
    let ny = height.div_ceil(window);
    let patches = mul(nx, ny)?;
    positive_i32(patches)?;
    let input = raw(a)?;
    if mul(mul(mul(channels, width)?, height)?, 4)? > input.len() {
        return Err(TensorError::Bounds);
    }
    let (desc, mut out) = output_bytes(DType::F32, [channels, window, window, patches])?;
    let mut at = 0;
    for py in 0..ny {
        for px in 0..nx {
            for y in 0..window {
                check_cancel(cancel)?;
                let sy = add(mul(py, window)?, y)?;
                for x in 0..window {
                    let sx = add(mul(px, window)?, x)?;
                    let count = mul(channels, 4)?;
                    if sx < width && sy < height {
                        let first = mul(add(mul(sy, width)?, sx)?, channels)?;
                        let start = mul(first, 4)?;
                        let source = input
                            .get(start..add(start, count)?)
                            .ok_or(TensorError::Bounds)?;
                        out[at..at + count].copy_from_slice(source);
                    }
                    at += count;
                }
            }
        }
    }
    Tensor::from_bytes(desc, out)
}

// Maximum source coordinate over complete blocks and the last partial block.
// The last output pixel need not address the greatest source offset when the
// source's stored window is smaller than the requested window size.
fn maximum_window_axis(
    length: usize,
    window: usize,
    block_stride: usize,
    inner_stride: usize,
) -> Result<usize> {
    let block = (length - 1) / window;
    let last = add(
        mul(block, block_stride)?,
        mul((length - 1) % window, inner_stride)?,
    )?;
    if block == 0 {
        return Ok(last);
    }
    let previous = add(
        mul(block - 1, block_stride)?,
        mul(window - 1, inner_stride)?,
    )?;
    Ok(last.max(previous))
}

fn unpartition(
    a: &Tensor,
    width: usize,
    height: usize,
    window: usize,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    require_type(a, DType::F32)?;
    for n in [width, height, window] {
        positive_i32(n)?;
    }
    let [channels, ax, ay, _] = a.shape();
    let nx = width.div_ceil(window);
    let input = raw(a)?;
    let window_area = mul(ax, ay)?;
    let last_x = maximum_window_axis(width, window, window_area, 1)?;
    let last_y = maximum_window_axis(height, window, mul(nx, window_area)?, ax)?;
    let required = mul(mul(add(add(last_x, last_y)?, 1)?, channels)?, 4)?;
    if required > input.len() {
        return Err(TensorError::Bounds);
    }
    let (desc, mut out) = output_bytes(DType::F32, [channels, width, height, 1])?;
    let row_bytes = mul(channels, 4)?;
    let mut at = 0;
    for y in 0..height {
        check_cancel(cancel)?;
        for x in 0..width {
            let patch = add(mul(y / window, nx)?, x / window)?;
            let source_row = add(
                mul(patch, mul(ax, ay)?)?,
                add(mul(y % window, ax)?, x % window)?,
            )?;
            let start = mul(mul(source_row, channels)?, 4)?;
            let source = input
                .get(start..add(start, row_bytes)?)
                .ok_or(TensorError::Bounds)?;
            out[at..at + row_bytes].copy_from_slice(source);
            at += row_bytes;
        }
    }
    Tensor::from_bytes(desc, out)
}

fn relative(a: &Tensor, query: usize, key: usize, cancel: &AtomicBool) -> Result<Tensor> {
    dimensions(a)?;
    if !matches!(a.dtype(), DType::F16 | DType::BF16) {
        return Err(TensorError::Unsupported(
            "GET_REL_POS CPU accepts F16 or BF16 storage",
        ));
    }
    positive_i32(query)?;
    positive_i32(key)?;
    if query != key || a.shape()[1] != mul(key, 2)? - 1 {
        return Err(TensorError::Shape(
            "relative positions require equal query/key and 2*width-1 rows",
        ));
    }
    let channels = a.shape()[0];
    // The original constructor always produces F16, even for BF16 input.
    // Its kernel copies u16 payloads without any numerical conversion.
    let input = raw(a)?;
    if mul(mul(channels, mul(key, 2)? - 1)?, 2)? > input.len() {
        return Err(TensorError::Bounds);
    }
    let (desc, mut out) = output_bytes(DType::F16, [channels, key, query, 1])?;
    let row_bytes = mul(channels, 2)?;
    let mut at = 0;
    for q in 0..query {
        check_cancel(cancel)?;
        for k in 0..key {
            let row = add(key - k - 1, q)?;
            let start = mul(row, row_bytes)?;
            let source = input
                .get(start..add(start, row_bytes)?)
                .ok_or(TensorError::Bounds)?;
            out[at..at + row_bytes].copy_from_slice(source);
            at += row_bytes;
        }
    }
    Tensor::from_bytes(desc, out)
}

fn add_relative(a: &Tensor, pw: &Tensor, ph: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    for t in [a, pw, ph] {
        require_type(t, DType::F32)?;
        require_contiguous(t)?;
    }
    let [key, qw, qh, patches] = pw.shape();
    if pw.shape() != ph.shape()
        || patches != a.shape()[2]
        || mul(key, key)? != a.shape()[0]
        || mul(qw, qh)? != a.shape()[1]
    {
        return Err(TensorError::Shape(
            "ADD_REL_POS tensor geometry does not match",
        ));
    }
    // The frozen GNU compiler specializes the one-key, multiple-query loop.
    // With independent output storage it loads height before the first add;
    // its scalar fallback and the general-key loop load the accumulator first.
    let height_right_nan = MSVC_RIGHT_NAN
        || (cfg!(all(
            target_arch = "x86_64",
            target_os = "linux",
            target_env = "gnu"
        )) && key == 1
            && qw > 1);
    let mut out = a.to_f32()?;
    let widths = pw.to_f32()?;
    let heights = ph.to_f32()?;
    for p in 0..patches {
        for y in 0..qh {
            check_cancel(cancel)?;
            for x in 0..qw {
                let base = ((p * qh + y) * qw + x) * key;
                for k in 0..key {
                    let pos = base + k;
                    let h = pos * key;
                    let w = h - (key - 1) * k;
                    for j in 0..key {
                        out[h + j] = fadd(out[h + j], heights[pos], height_right_nan)?;
                        out[w + j * key] = fadd(out[w + j * key], widths[pos], MSVC_RIGHT_NAN)?;
                    }
                }
            }
        }
    }
    Tensor::from_f32(a.shape(), &out)
}

fn solve(a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    for t in [a, b] {
        require_type(t, DType::F32)?;
        require_contiguous(t)?;
    }
    let [n, rows, b2, b3] = a.shape();
    let [columns, br, bb2, bb3] = b.shape();
    if n != rows || br != n || b2 != bb2 || b3 != bb3 {
        return Err(TensorError::Shape(
            "triangular solve requires square A, matching rows and batches",
        ));
    }
    let av = a.to_f32()?;
    let bv = b.to_f32()?;
    let mut out = zeros(bv.len())?;
    // Only the source's left, lower, non-unit diagonal form exists.
    for batch in 0..b2 * b3 {
        let ab = batch * n * n;
        let bb = batch * n * columns;
        for row in 0..n {
            if av[ab + row * n + row] == 0.0 {
                return Err(TensorError::Parameters(
                    "triangular solve has a zero diagonal",
                ));
            }
        }
        for column in 0..columns {
            check_cancel(cancel)?;
            for row in 0..n {
                let mut sum = 0.0_f32;
                for t in 0..row {
                    sum = fadd(
                        sum,
                        fmul(av[ab + row * n + t], out[bb + t * columns + column])?,
                        false,
                    )?;
                }
                out[bb + row * columns + column] = fdiv(
                    fsub(bv[bb + row * columns + column], sum)?,
                    av[ab + row * n + row],
                )?;
            }
        }
    }
    Tensor::from_f32(b.shape(), &out)
}

fn loss_inputs(a: &Tensor, b: &Tensor) -> Result<(usize, usize)> {
    for t in [a, b] {
        require_type(t, DType::F32)?;
    }
    if a.shape() != b.shape() {
        return Err(TensorError::Shape(
            "cross entropy inputs must have equal shape",
        ));
    }
    if a.desc().nb[0] != 4
        || b.desc().nb[0] != 4
        || !a.desc().nb[1].is_multiple_of(4)
        || !b.desc().nb[1].is_multiple_of(4)
    {
        return Err(TensorError::Layout(
            "cross entropy requires aligned contiguous inner rows",
        ));
    }
    let n = a.shape();
    let rows = mul(mul(n[1], n[2])?, n[3])?;
    if mul(n[0], rows)? > 64 * 1024 * 1024 {
        return Err(TensorError::Unsupported(
            "scalar cross-entropy work exceeds 64 million logical cells",
        ));
    }
    for tensor in [a, b] {
        let end = add(mul(rows - 1, tensor.desc().nb[1])?, mul(n[0], 4)?)?;
        if end > tensor.desc().byte_span()? {
            return Err(TensorError::Bounds);
        }
    }
    Ok((n[0], rows))
}

fn loss(a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let (classes, rows) = loss_inputs(a, b)?;
    let aa = raw(a)?;
    let bb = raw(b)?;
    let mut row = zeros(classes)?;
    let mut labels = zeros(classes)?;
    let mut sum_thread = 0.0_f32;
    for r in 0..rows {
        check_cancel(cancel)?;
        let start_a = mul(r, a.desc().nb[1] / 4)?;
        let start_b = mul(r, b.desc().nb[1] / 4)?;
        let mut maximum = f32::NEG_INFINITY;
        for c in 0..classes {
            row[c] = float_at(aa, add(start_a, c)?)?;
            labels[c] = float_at(bb, add(start_b, c)?)?;
            if !row[c].is_finite() || !labels[c].is_finite() {
                return Err(TensorError::Parameters(
                    "cross entropy inputs must be finite",
                ));
            }
            maximum = maximum.max(row[c]);
        }
        let mut exp_sum = 0.0_f64;
        for value in &mut row {
            *value -= maximum;
            // Log-softmax is the source scalar expf loop on both profiles.
            exp_sum += value.exp() as f64;
        }
        let log_sum = (exp_sum as f32).ln();
        let mut sum_row = 0.0_f64;
        for (value, label) in row.iter().zip(&labels) {
            sum_row += ((*value - log_sum) * *label) as f64;
        }
        sum_thread += sum_row as f32;
    }
    Tensor::from_f32([1; 4], &[sum_thread * (-1.0 / rows as f32)])
}

fn loss_back(grad: &Tensor, a: &Tensor, b: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let (classes, rows) = loss_inputs(a, b)?;
    require_type(grad, DType::F32)?;
    for t in [grad, a, b] {
        require_contiguous(t)?;
    }
    if grad.desc().elements()? != 1 {
        return Err(TensorError::Shape(
            "cross entropy output gradient must be scalar",
        ));
    }
    let gradient = float_at(raw(grad)?, 0)?;
    if !gradient.is_finite() {
        return Err(TensorError::Parameters(
            "cross entropy gradient must be finite",
        ));
    }
    let mut out = a.to_f32()?;
    let labels = b.to_f32()?;
    let scale = gradient / rows as f32;
    for (row, target) in out
        .chunks_exact_mut(classes)
        .zip(labels.chunks_exact(classes))
    {
        check_cancel(cancel)?;
        if row.iter().chain(target).any(|v| !v.is_finite()) {
            return Err(TensorError::Parameters(
                "cross entropy inputs must be finite",
            ));
        }
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = llama_quant::cpu_math::softmax_exp_sum_f32_in_place(row, maximum);
        let inverse = (1.0 / sum) as f32;
        for (value, label) in row.iter_mut().zip(target) {
            *value = (*value * inverse - *label) * scale;
        }
    }
    Tensor::from_f32(a.shape(), &out)
}
