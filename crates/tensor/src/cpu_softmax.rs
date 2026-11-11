//! Frozen F32 masked softmax, ALiBi and attention sinks.
//! GGML MIT, copyright (c) 2023-2026 The ggml authors.
use crate::graph::check_cancel;
use crate::{DType, Result, Tensor, TensorDesc, TensorError, reserved};
use std::sync::atomic::AtomicBool;

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(TensorError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(TensorError::Overflow)
}
fn raw(tensor: &Tensor) -> Result<&[u8]> {
    tensor
        .data
        .get(tensor.offset..add(tensor.offset, tensor.desc().byte_span()?)?)
        .ok_or(TensorError::Bounds)
}
fn scalar(bytes: &[u8], dtype: DType, at: usize) -> Result<f32> {
    let offset = mul(at, dtype.type_size())?;
    let bytes = bytes
        .get(offset..add(offset, dtype.type_size())?)
        .ok_or(TensorError::Bounds)?;
    Ok(if dtype == DType::F16 {
        llama_quant::f16_to_f32(u16::from_le_bytes(bytes.try_into().expect("half word")))
    } else {
        f32::from_le_bytes(bytes.try_into().expect("float word"))
    })
}
fn boundary() -> TensorError {
    TensorError::Unsupported("softmax requires a defined finite normalized output")
}

/// Inputs are logits, optional mask, optional sinks, in that order.
pub(crate) fn evaluate(
    scale: f32,
    max_bias: f32,
    with_mask: bool,
    with_sinks: bool,
    src: &[&Tensor],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let input = src[0];
    if input.dtype() != DType::F32 {
        return Err(TensorError::Type {
            expected: DType::F32,
            actual: input.dtype(),
        });
    }
    if !input.desc().is_contiguous() {
        return Err(TensorError::Layout(
            "original softmax requires contiguous logits",
        ));
    }
    let mask = if with_mask { Some(src[1]) } else { None };
    let sinks = if with_sinks {
        Some(src[1 + usize::from(with_mask)])
    } else {
        None
    };
    let [width, queries, heads, batches] = input.shape();
    for tensor in src {
        if tensor.shape().iter().any(|&n| n > i32::MAX as usize) {
            return Err(TensorError::Parameters(
                "softmax dimensions must fit signed I32",
            ));
        }
    }
    if let Some(mask) = mask {
        if !matches!(mask.dtype(), DType::F32 | DType::F16) {
            return Err(TensorError::Unsupported(
                "original softmax masks must be F32 or F16",
            ));
        }
        if !mask.desc().is_contiguous() {
            return Err(TensorError::Layout(
                "original softmax requires a contiguous mask",
            ));
        }
        let [mw, mq, mh, mb] = mask.shape();
        if mw != width
            || mq < queries
            || mh == 0
            || mb == 0
            || !heads.is_multiple_of(mh)
            || !batches.is_multiple_of(mb)
        {
            return Err(TensorError::Shape(
                "softmax mask shape cannot broadcast to logits",
            ));
        }
    } else if max_bias > 0.0 {
        return Err(TensorError::Parameters(
            "positive softmax ALiBi bias requires a mask",
        ));
    }
    if let Some(sinks) = sinks {
        if sinks.dtype() != DType::F32 {
            return Err(TensorError::Type {
                expected: DType::F32,
                actual: sinks.dtype(),
            });
        }
        if sinks.shape()[0] != heads {
            return Err(TensorError::Shape(
                "softmax sink first axis must equal head count",
            ));
        }
    }
    let desc = TensorDesc::contiguous(DType::F32, input.shape())?;
    let count = desc.elements()?;
    // Original CPU dispatcher returns before evaluating empty outputs.
    if count == 0 {
        return Tensor::from_bytes(desc, Vec::new());
    }
    let bytes = desc.byte_span()?;
    if bytes > 256 * 1024 * 1024 || count > 64 * 1024 * 1024 {
        return Err(TensorError::Unsupported(
            "softmax exceeds local 256 MiB output/work limit",
        ));
    }
    if !scale.is_finite() {
        return Err(TensorError::Unsupported("softmax scale must be finite"));
    }
    let mut output = reserved(count)?;
    output.resize(count, 0.0_f32);
    let input_bytes = raw(input)?;
    let mask_bytes = mask.map(raw).transpose()?;
    let sink_bytes = sinks.map(raw).transpose()?;
    let head_pow2 = 1usize << heads.ilog2();
    let m0 = 2.0_f32.powf(-max_bias / head_pow2 as f32);
    let m1 = 2.0_f32.powf(-(max_bias / 2.0) / head_pow2 as f32);
    for batch in 0..batches {
        for head in 0..heads {
            // Bias <= 0 and NaN disable ALiBi in the original comparison.
            let slope = if max_bias > 0.0 {
                if head < head_pow2 {
                    m0.powf((head + 1) as f32)
                } else {
                    m1.powf((2 * (head - head_pow2) + 1) as f32)
                }
            } else {
                1.0
            };
            // The original sinks getter reads a flat prefix, ignoring nb[0]
            // and any surplus sink axes. Preserve that checked physical read.
            let sink = sink_bytes
                .map(|bytes| scalar(bytes, DType::F32, head))
                .transpose()?;
            if sink.is_some_and(|value| value.is_nan() || value == f32::INFINITY) {
                return Err(boundary());
            }
            for query in 0..queries {
                check_cancel(cancel)?;
                let row = add(query, mul(queries, add(head, mul(heads, batch)?)?)?)?;
                let start = mul(row, width)?;
                let out = &mut output[start..start + width];
                let mask_start = if let Some(mask) = mask {
                    let [mw, mq, mh, mb] = mask.shape();
                    mul(
                        mw,
                        add(query, mul(mq, add(head % mh, mul(mh, batch % mb)?)?)?)?,
                    )?
                } else {
                    0
                };
                let mut maximum = f32::NEG_INFINITY;
                for (column, target) in out.iter_mut().enumerate() {
                    let mut value = scalar(input_bytes, DType::F32, start + column)? * scale;
                    if let (Some(mask), Some(bytes)) = (mask, mask_bytes) {
                        value += slope * scalar(bytes, mask.dtype(), mask_start + column)?;
                    }
                    if value.is_nan() || value == f32::INFINITY {
                        return Err(boundary());
                    }
                    *target = value;
                    maximum = maximum.max(value);
                }
                if let Some(sink) = sink {
                    maximum = if maximum > sink { maximum } else { sink };
                }
                if !maximum.is_finite() {
                    return Err(boundary());
                }
                let mut sum = llama_quant::cpu_math::softmax_exp_sum_f32_in_place(out, maximum);
                if let Some(sink) = sink {
                    sum += (sink - maximum).exp() as f64;
                }
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(boundary());
                }
                let inverse = (1.0 / sum) as f32;
                for value in out {
                    *value *= inverse;
                    if !value.is_finite() {
                        return Err(boundary());
                    }
                }
            }
        }
    }
    Tensor::from_f32(input.shape(), &output)
}
