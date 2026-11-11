//! Original F32/F16 backward rotary kernels and checked extended parameters.
//! GGML MIT, copyright (c) 2023-2026 The ggml authors.
//! YaRN equations: MIT, copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
use crate::graph::check_cancel;
use crate::{DType, Result, Tensor, TensorDesc, TensorError, reserved, zeros};
use std::sync::atomic::AtomicBool;

/// The five frozen CPU rotary modes, including their original numeric IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RopeBackMode {
    Normal = 0,
    NeoX = 2,
    Multi = 8,
    Vision = 24,
    Interleaved = 40,
}

/// Parameters for the original backward rotation, independent of forward APIs.
#[derive(Clone, Debug)]
pub struct RopeBackOptions {
    pub dimensions: usize,
    pub mode: RopeBackMode,
    pub context_original: usize,
    pub frequency_base: f32,
    pub frequency_scale: f32,
    pub extension_factor: f32,
    pub attention_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub sections: [usize; 4],
    pub offset: usize,
    pub truncate: bool,
}
impl RopeBackOptions {
    /// Original unscaled defaults. Multi-axis modes also need explicit sections.
    pub fn new(dimensions: usize, mode: RopeBackMode) -> Self {
        Self {
            dimensions,
            mode,
            context_original: 0,
            frequency_base: 10_000.0,
            frequency_scale: 1.0,
            extension_factor: 0.0,
            attention_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            sections: [0; 4],
            offset: 0,
            truncate: true,
        }
    }
}
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
fn bytes_at(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    bytes
        .get(offset..add(offset, size)?)
        .ok_or(TensorError::Bounds)
}
fn scalar(bytes: &[u8], dtype: DType, index: usize) -> Result<f32> {
    let word = bytes_at(bytes, mul(index, dtype.type_size())?, dtype.type_size())?;
    Ok(if dtype == DType::F16 {
        llama_quant::f16_to_f32(u16::from_le_bytes(word.try_into().expect("half word")))
    } else {
        f32::from_le_bytes(word.try_into().expect("float word"))
    })
}
fn write_scalar(bytes: &mut [u8], dtype: DType, index: usize, value: f32) -> Result<()> {
    let size = dtype.type_size();
    let offset = mul(index, size)?;
    let target = bytes
        .get_mut(offset..add(offset, size)?)
        .ok_or(TensorError::Bounds)?;
    if dtype == DType::F16 {
        target.copy_from_slice(&llama_quant::f32_to_f16(value).to_le_bytes());
    } else {
        target.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}
fn multi(mode: RopeBackMode) -> bool {
    !matches!(mode, RopeBackMode::Normal | RopeBackMode::NeoX)
}
fn numerical_boundary() -> TensorError {
    TensorError::Unsupported(
        "RoPE backward requires finite rotated values, coefficients and F32 products",
    )
}
fn validate_frequencies(p: &RopeBackOptions) -> Result<()> {
    if !p.frequency_base.is_finite()
        || p.frequency_base <= 0.0
        || !p.frequency_scale.is_finite()
        || !p.extension_factor.is_finite()
        || !p.attention_factor.is_finite()
    {
        return Err(TensorError::Unsupported(
            "RoPE backward frequency parameters are outside the finite domain",
        ));
    }
    // The source computes these fields even when ext=0, but only uses their
    // results in the active YaRN branch. Preserve ignored nonfinite fields.
    if p.extension_factor != 0.0
        && (p.context_original == 0
            || p.frequency_scale <= 0.0
            || !p.beta_fast.is_finite()
            || p.beta_fast <= 0.0
            || !p.beta_slow.is_finite()
            || p.beta_slow <= 0.0)
    {
        return Err(TensorError::Unsupported(
            "active backward YaRN requires positive context, scale and finite betas",
        ));
    }
    Ok(())
}
fn correction(p: &RopeBackOptions, rotations: f32) -> f32 {
    p.dimensions as f32
        * (p.context_original as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * p.frequency_base.ln())
}
fn cache(
    p: &RopeBackOptions,
    width: usize,
    positions: [f32; 4],
    factors: Option<&[u8]>,
    values: &mut [f32],
) -> Result<()> {
    let theta_scale = p.frequency_base.powf(-2.0 / p.dimensions as f32);
    let mut theta = positions;
    let sections = p.sections;
    let section_total: usize = sections.iter().sum();
    let sec_w = sections[0] + sections[1];
    let sec_e = sec_w + sections[2];
    let mut low = 0.0;
    let mut high = 0.0;
    if p.extension_factor != 0.0 {
        let start = correction(p, p.beta_fast);
        let end = correction(p, p.beta_slow);
        let start = if p.truncate { start.floor() } else { start };
        let end = if p.truncate { end.ceil() } else { end };
        // Original MAX/MIN operand order, including intermediate infinities.
        low = if 0.0 > start { 0.0 } else { start };
        let upper = (p.dimensions - 1) as f32;
        high = if upper < end { upper } else { end };
    }
    let used_pairs = if p.mode == RopeBackMode::Vision {
        width / 2
    } else {
        p.dimensions / 2
    };
    for pair in 0..width / 2 {
        let mut angle_base = theta[0];
        if multi(p.mode) {
            let sector = pair % section_total;
            if p.mode == RopeBackMode::Vision {
                if sector == 0 {
                    theta[0] = positions[0];
                } else if sector == sections[0] {
                    theta[1] = positions[1];
                } else if sector == sec_w {
                    theta[2] = positions[2];
                } else if sector == sec_e {
                    theta[3] = positions[3];
                }
            }
            let axis = if p.mode == RopeBackMode::Interleaved {
                if sector % 3 == 1 && sector < 3 * sections[1] {
                    1
                } else if sector % 3 == 2 && sector < 3 * sections[2] {
                    2
                } else if sector.is_multiple_of(3) && sector < 3 * sections[0] {
                    0
                } else {
                    3
                }
            } else if sector >= sections[0] && sector < sec_w {
                1
            } else if sector >= sec_w && sector < sec_e {
                2
            } else if sector >= sec_e {
                3
            } else {
                0
            };
            angle_base = theta[axis];
        }
        let ff = match factors {
            Some(bytes) => scalar(bytes, DType::F32, pair)?,
            None => 1.0,
        };
        if pair < used_pairs && (!ff.is_finite() || ff == 0.0) {
            return Err(TensorError::Unsupported(
                "used backward RoPE frequency factors must be finite and nonzero",
            ));
        }
        let extrapolated = angle_base / ff;
        let interpolated = p.frequency_scale * extrapolated;
        let mut angle = interpolated;
        let mut magnitude = p.attention_factor;
        if p.extension_factor != 0.0 {
            let difference = high - low;
            let divisor = if 0.001 > difference {
                0.001
            } else {
                difference
            };
            let ramp = (pair as f32 - low) / divisor;
            let ramp = if 0.0 > ramp { 0.0 } else { ramp };
            let ramp = if 1.0 < ramp { 1.0 } else { ramp };
            let mix = (1.0 - ramp) * p.extension_factor;
            angle = interpolated * (1.0 - mix) + extrapolated * mix;
            magnitude *= 1.0 + 0.1 * (1.0 / p.frequency_scale).ln();
        }
        let cosine = angle.cos() * magnitude;
        let sine = -(angle.sin() * magnitude);
        if pair < used_pairs && (!cosine.is_finite() || !sine.is_finite()) {
            return Err(numerical_boundary());
        }
        values[2 * pair] = cosine;
        values[2 * pair + 1] = sine;
        for value in &mut theta {
            *value *= theta_scale;
        }
    }
    Ok(())
}

pub(super) fn evaluate(
    p: &RopeBackOptions,
    src: &[&Tensor],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let a = src[0];
    let positions = src[1];
    let factors = src.get(2).copied();
    if !matches!(a.dtype(), DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "original RoPE backward supports only F32 and F16",
        ));
    }
    if positions.dtype() != DType::I32 {
        return Err(TensorError::Type {
            expected: DType::I32,
            actual: positions.dtype(),
        });
    }
    for tensor in src {
        if tensor.shape().into_iter().any(|n| n > i32::MAX as usize) {
            return Err(TensorError::Parameters(
                "RoPE backward dimensions must fit signed I32",
            ));
        }
    }
    if [p.dimensions, p.offset, p.context_original]
        .into_iter()
        .chain(p.sections)
        .any(|v| v > i32::MAX as usize)
    {
        return Err(TensorError::Parameters(
            "RoPE backward integer parameters must fit signed I32",
        ));
    }
    let [width, heads, tokens, batches] = a.shape();
    if p.dimensions == 0
        || !p.dimensions.is_multiple_of(2)
        || !width.is_multiple_of(2)
        || !p.offset.is_multiple_of(2)
        || add(p.offset, p.dimensions)? > width
    {
        return Err(TensorError::Parameters(
            "RoPE backward needs positive even dimensions, even width/offset and a contained rotation",
        ));
    }
    if p.mode == RopeBackMode::Vision && (p.dimensions != width / 2 || p.offset != 0) {
        return Err(TensorError::Parameters(
            "vision RoPE backward requires half-width dimensions and zero offset",
        ));
    }
    let sections = p.sections.iter().try_fold(0usize, |n, &v| add(n, v))?;
    if multi(p.mode) && (p.sections[..3].iter().all(|&v| v == 0) || sections > width) {
        return Err(TensorError::Parameters(
            "multi-axis backward RoPE requires nonzero T/H/W sections with a bounded total",
        ));
    }
    let position_count = mul(tokens, if multi(p.mode) { 4 } else { 1 })?;
    if positions.shape() != [position_count, 1, 1, 1] {
        return Err(TensorError::Shape(
            "RoPE backward position vector does not match tokens and mode",
        ));
    }
    if let Some(factors) = factors {
        if factors.dtype() != DType::F32 {
            return Err(TensorError::Type {
                expected: DType::F32,
                actual: factors.dtype(),
            });
        }
        if factors.shape()[0] < p.dimensions / 2 {
            return Err(TensorError::Shape(
                "RoPE backward frequency tensor has too few constructor entries",
            ));
        }
    }
    let desc = TensorDesc::contiguous(a.dtype(), a.shape())?;
    let count = desc.elements()?;
    if count == 0 {
        return Tensor::from_bytes(desc, Vec::new());
    }
    if mul(mul(heads, tokens)?, batches)? > i32::MAX as usize {
        return Err(TensorError::Parameters(
            "RoPE backward row count must fit signed I32",
        ));
    }
    let size = a.dtype().type_size();
    if a.desc().nb[0] != size || a.desc().nb.iter().any(|n| !n.is_multiple_of(size)) {
        return Err(TensorError::Layout(
            "RoPE backward requires contiguous inner rows and aligned scalar strides",
        ));
    }
    let output_bytes = desc.byte_span()?;
    let cache_bytes = mul(width, 4)?;
    let work = add(count, mul(mul(width, tokens)?, batches)?)?;
    if output_bytes > 256 * 1024 * 1024
        || cache_bytes > 256 * 1024 * 1024
        || work > 256 * 1024 * 1024
    {
        return Err(TensorError::Unsupported(
            "RoPE backward exceeds the local 256 MiB allocation or 256 Mi scalar-work limit",
        ));
    }
    validate_frequencies(p)?;
    let input = raw(a)?;
    let pos = bytes_at(raw(positions)?, 0, mul(position_count, 4)?)?;
    let factors = factors
        .map(|t| bytes_at(raw(t)?, 0, mul(width / 2, 4)?))
        .transpose()?;
    let mut trig = zeros(width)?;
    let mut output = reserved(output_bytes)?;
    output.resize(output_bytes, 0);
    let row_bytes = mul(width, size)?;
    let pairs = if p.mode == RopeBackMode::Vision {
        width / 2
    } else {
        p.dimensions / 2
    };
    for batch in 0..batches {
        for token in 0..tokens {
            check_cancel(cancel)?;
            let mut coordinates = [0.0_f32; 4];
            for (axis, coordinate) in
                coordinates
                    .iter_mut()
                    .enumerate()
                    .take(if multi(p.mode) { 4 } else { 1 })
            {
                let offset = mul(add(token, mul(tokens, axis)?)?, 4)?;
                *coordinate = i32::from_le_bytes(
                    bytes_at(pos, offset, 4)?.try_into().expect("position word"),
                ) as f32;
            }
            cache(p, width, coordinates, factors, &mut trig)?;
            for head in 0..heads {
                check_cancel(cancel)?;
                let offset = add(
                    add(mul(batch, a.desc().nb[3])?, mul(token, a.desc().nb[2])?)?,
                    mul(head, a.desc().nb[1])?,
                )?;
                let input_row = bytes_at(input, offset, row_bytes)?;
                let row = add(head, mul(heads, add(token, mul(tokens, batch)?)?)?)?;
                let offset = mul(row, row_bytes)?;
                let output_row = output
                    .get_mut(offset..add(offset, row_bytes)?)
                    .ok_or(TensorError::Bounds)?;
                // Untouched channels preserve raw half and float payloads.
                output_row.copy_from_slice(input_row);
                for pair in 0..pairs {
                    let (low, high) = if p.mode == RopeBackMode::Normal {
                        (p.offset + 2 * pair, p.offset + 2 * pair + 1)
                    } else {
                        (p.offset + pair, p.offset + pair + p.dimensions / 2)
                    };
                    let high = if p.mode == RopeBackMode::Vision {
                        pair + p.dimensions
                    } else {
                        high
                    };
                    let x = scalar(input_row, a.dtype(), low)?;
                    let y = scalar(input_row, a.dtype(), high)?;
                    if !x.is_finite() || !y.is_finite() {
                        return Err(numerical_boundary());
                    }
                    let co = trig[2 * pair];
                    let si = trig[2 * pair + 1];
                    let products = [x * co, y * si, x * si, y * co];
                    if products.iter().any(|v| !v.is_finite()) {
                        return Err(numerical_boundary());
                    }
                    write_scalar(output_row, a.dtype(), low, products[0] - products[1])?;
                    write_scalar(output_row, a.dtype(), high, products[2] + products[3])?;
                }
            }
        }
    }
    Tensor::from_bytes(desc, output)
}
