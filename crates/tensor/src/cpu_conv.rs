//! Frozen one-thread convolution, extraction, pooling and resize kernels.
//! Scalar products/reductions, explicit source byte addressing and final dtype
//! rounding are part of the contract. Every result owns independent storage.
use crate::graph::check_cancel;
use crate::{
    Conv2dParams, Conv3dParams, DType, Op, PoolMode, ResizeMode, Result, Tensor, TensorDesc,
    TensorError, reserved, zeros,
};
use std::sync::atomic::AtomicBool;

pub(super) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    use Op::*;
    Some(match op {
        Im2col {
            params,
            is_2d,
            dtype,
        } => im2col(src[0], src[1], *params, *is_2d, *dtype, cancel),
        Im2colBack {
            params,
            is_2d,
            shape,
        } => im2col_back(src[0], src[1], *params, *is_2d, *shape, cancel),
        Im2col3d {
            params,
            input_channels,
            dtype,
        } => im2col_3d(src[0], src[1], *params, *input_channels, *dtype, cancel),
        Col2im1d {
            stride,
            output_channels,
            padding,
        } => col2im(src[0], *stride, *output_channels, *padding, cancel),
        ConvTranspose1d { stride } => transpose1d(src[0], src[1], *stride, cancel),
        Conv2d { params } => conv2d(src[0], src[1], *params, cancel),
        Conv3d {
            params,
            input_channels,
        } => conv3d(src[0], src[1], *params, *input_channels, cancel),
        Conv2dDepthwise { params } => depthwise(src[0], src[1], *params, cancel),
        ConvTranspose2d { stride } => transpose2d(src[0], src[1], *stride, cancel),
        Pool1d {
            mode,
            kernel,
            stride,
            padding,
        } => pool1d(src[0], *mode, *kernel, *stride, *padding, cancel),
        Pool2d {
            mode,
            kernel,
            stride,
            padding,
        } => pool2d(src[0], *mode, *kernel, *stride, *padding, cancel),
        Pool2dBack {
            mode,
            kernel,
            stride,
            padding,
        } => pool2d_back(src[0], src[1], *mode, *kernel, *stride, *padding, cancel),
        Resize {
            shape,
            mode,
            align_corners,
            antialias,
        } => resize(src[0], *shape, *mode, *align_corners, *antialias, cancel),
        _ => return None,
    })
}
fn parameter(message: &'static str) -> TensorError {
    TensorError::Parameters(message)
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(TensorError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(TensorError::Overflow)
}
fn size32(value: usize) -> Result<i64> {
    i32::try_from(value)
        .map(i64::from)
        .map_err(|_| TensorError::Unsupported("convolution CPU dimensions must fit signed I32"))
}
fn nonempty(shape: [usize; 4]) -> Result<[usize; 4]> {
    if shape.contains(&0) {
        return Err(TensorError::Shape(
            "image operations require nonempty dimensions",
        ));
    }
    for dimension in shape {
        size32(dimension)?;
    }
    Ok(shape)
}
fn elements(shape: [usize; 4]) -> Result<usize> {
    TensorDesc::contiguous(DType::F32, shape)?.elements()
}
fn flat(at: [usize; 4], shape: [usize; 4]) -> usize {
    at[0] + shape[0] * (at[1] + shape[1] * (at[2] + shape[2] * at[3]))
}
fn positive(value: usize) -> Result<i64> {
    let value = size32(value)?;
    if value == 0 {
        return Err(parameter(
            "kernel sizes, channels and strides must be positive",
        ));
    }
    Ok(value)
}
fn validate_convolution(stride: &[i32], padding: &[i32], dilation: &[i32]) -> Result<()> {
    for ((&stride, &padding), &dilation) in stride.iter().zip(padding).zip(dilation) {
        if stride <= 0 || dilation <= 0 {
            return Err(parameter(
                "convolution stride and dilation must be positive",
            ));
        }
        padding.checked_mul(2).ok_or(TensorError::Overflow)?;
    }
    Ok(())
}
fn output_axis(
    input: usize,
    kernel: usize,
    stride: i32,
    padding: i32,
    dilation: i32,
) -> Result<usize> {
    let numerator =
        input as i128 + 2 * i128::from(padding) - i128::from(dilation) * (kernel as i128 - 1) - 1;
    // C signed division truncates toward zero, including a negative numerator.
    let output = numerator / i128::from(stride) + 1;
    if output <= 0 || output > i128::from(i32::MAX) {
        return Err(TensorError::Shape(
            "convolution output dimension is invalid",
        ));
    }
    Ok(output as usize)
}
fn transposed_axis(input: usize, kernel: usize, stride: usize, padding: usize) -> Result<usize> {
    let size = (input as i128 - 1) * stride as i128 + kernel as i128 - 2 * padding as i128;
    if size <= 0 || size > i128::from(i32::MAX) {
        return Err(TensorError::Shape(
            "transposed convolution output dimension is invalid",
        ));
    }
    Ok(size as usize)
}
fn position(out: usize, kernel: usize, stride: i32, padding: i32, dilation: i32) -> i64 {
    out as i64 * i64::from(stride) + kernel as i64 * i64::from(dilation) - i64::from(padding)
}
fn in_axis(value: i64, size: usize) -> bool {
    value >= 0 && (value as u64) < size as u64
}
struct Work<'a> {
    cancelled: &'a AtomicBool,
    count: usize,
}
impl<'a> Work<'a> {
    fn new(cancelled: &'a AtomicBool) -> Result<Self> {
        check_cancel(cancelled)?;
        Ok(Self {
            cancelled,
            count: 0,
        })
    }
    fn tick(&mut self) -> Result<()> {
        self.count = self.count.wrapping_add(1);
        if self.count.is_multiple_of(4096) {
            check_cancel(self.cancelled)?;
        }
        Ok(())
    }
}

/// A checked window of the declared view span. Frozen kernels sometimes use
/// flat physical offsets rather than the descriptor's logical stride formula.
struct Reader<'a> {
    tensor: &'a Tensor,
    bytes: &'a [u8],
}
impl<'a> Reader<'a> {
    fn new(tensor: &'a Tensor) -> Result<Self> {
        if !matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(TensorError::Unsupported(
                "image kernel scalar dtype is unsupported",
            ));
        }
        let end = add(tensor.offset, tensor.desc.byte_span()?)?;
        let bytes = tensor
            .data
            .get(tensor.offset..end)
            .ok_or(TensorError::Bounds)?;
        Ok(Self { tensor, bytes })
    }
    fn word(&self, offset: usize) -> Result<&[u8]> {
        let size = self.tensor.dtype().type_size();
        if !offset.is_multiple_of(size) {
            return Err(TensorError::Layout("native scalar address must be aligned"));
        }
        self.bytes
            .get(offset..add(offset, size)?)
            .ok_or(TensorError::Bounds)
    }
    fn read(&self, offset: usize) -> Result<f32> {
        let bytes = self.word(offset)?;
        Ok(match self.tensor.dtype() {
            DType::F32 => f32::from_le_bytes(bytes.try_into().expect("checked F32 word")),
            DType::F16 => llama_quant::f16_to_f32(u16::from_le_bytes(
                bytes.try_into().expect("checked F16 word"),
            )),
            DType::BF16 => llama_quant::bf16_to_f32(u16::from_le_bytes(
                bytes.try_into().expect("checked BF16 word"),
            )),
            _ => unreachable!("reader dtype checked"),
        })
    }
    fn offset(&self, at: [usize; 4]) -> Result<usize> {
        let mut offset = 0;
        for ((index, size), stride) in at
            .into_iter()
            .zip(self.tensor.shape())
            .zip(self.tensor.desc.nb)
        {
            if index >= size {
                return Err(TensorError::Bounds);
            }
            offset = add(offset, mul(index, stride)?)?;
        }
        Ok(offset)
    }
    fn at(&self, at: [usize; 4]) -> Result<f32> {
        self.read(self.offset(at)?)
    }
}
fn raw_output(dtype: DType, shape: [usize; 4]) -> Result<(TensorDesc, Vec<u8>)> {
    let desc = TensorDesc::contiguous(dtype, shape)?;
    let mut bytes = reserved(desc.byte_span()?)?;
    bytes.resize(desc.byte_span()?, 0);
    Ok((desc, bytes))
}
fn write_value(bytes: &mut [u8], dtype: DType, index: usize, value: f32) {
    match dtype {
        DType::F32 => bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes()),
        DType::F16 => bytes[index * 2..index * 2 + 2]
            .copy_from_slice(&llama_quant::f32_to_f16(value).to_le_bytes()),
        DType::BF16 => bytes[index * 2..index * 2 + 2]
            .copy_from_slice(&llama_quant::f32_to_bf16(value).to_le_bytes()),
        _ => unreachable!("image output dtype checked"),
    }
}
fn typed_output(desc: TensorDesc, values: &[f32]) -> Result<Tensor> {
    if desc.elements()? != values.len()
        || desc.byte_span()? != mul(values.len(), desc.dtype.type_size())?
    {
        return Err(TensorError::Layout(
            "image output descriptor is not densely packed",
        ));
    }
    let mut bytes = reserved(desc.byte_span()?)?;
    bytes.resize(desc.byte_span()?, 0);
    for (index, &value) in values.iter().enumerate() {
        write_value(&mut bytes, desc.dtype, index, value);
    }
    Tensor::from_bytes(desc, bytes)
}
fn require_f32(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::F32 {
        return Err(TensorError::Type {
            expected: DType::F32,
            actual: tensor.dtype(),
        });
    }
    Ok(())
}
fn extraction_shape(
    kernel: [usize; 4],
    image: [usize; 4],
    p: Conv2dParams,
    is_2d: bool,
) -> Result<[usize; 4]> {
    nonempty(kernel)?;
    nonempty(image)?;
    let dimensions = if is_2d { 2 } else { 1 };
    validate_convolution(
        &p.stride[..dimensions],
        &p.padding[..dimensions],
        &p.dilation[..dimensions],
    )?;
    if if is_2d {
        kernel[2] != image[2]
    } else {
        kernel[1] != image[1] || image[3] != 1
    } {
        return Err(TensorError::Shape(
            "im2col kernel/input channels do not match",
        ));
    }
    let width = output_axis(
        image[0],
        kernel[0],
        p.stride[0],
        p.padding[0],
        p.dilation[0],
    )?;
    Ok(if is_2d {
        [
            mul(mul(kernel[0], kernel[1])?, image[2])?,
            width,
            output_axis(
                image[1],
                kernel[1],
                p.stride[1],
                p.padding[1],
                p.dilation[1],
            )?,
            image[3],
        ]
    } else {
        [mul(kernel[0], image[1])?, width, image[2], 1]
    })
}
fn im2col(
    kernel: &Tensor,
    image: &Tensor,
    p: Conv2dParams,
    is_2d: bool,
    dtype: DType,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let shape = extraction_shape(kernel.shape(), image.shape(), p, is_2d)?;
    if !matches!(
        (image.dtype(), dtype),
        (DType::F32, DType::F32 | DType::F16) | (DType::F16, DType::F16)
    ) {
        return Err(TensorError::Unsupported(
            "im2col supports F32 to F32/F16 or raw F16 to F16",
        ));
    }
    let n = image.shape();
    let k = kernel.shape();
    let scalar_size = image.dtype().type_size();
    if image.desc.nb[0] != scalar_size {
        return Err(TensorError::Layout(
            "im2col requires a contiguous first axis",
        ));
    }
    let (batch_stride, channel_stride) = if is_2d {
        (image.desc.nb[3], image.desc.nb[2])
    } else {
        (image.desc.nb[2], image.desc.nb[1])
    };
    size32(batch_stride)?;
    size32(channel_stride)?;
    let (batches, channels, height, kh, oh) = if is_2d {
        (n[3], n[2], n[1], k[1], shape[2])
    } else {
        (n[2], n[1], 1, 1, 1)
    };
    let source = Reader::new(image)?;
    let (desc, mut out) = raw_output(dtype, shape)?;
    let mut work = Work::new(cancel)?;
    for batch in 0..batches {
        for y in 0..oh {
            for x in 0..shape[1] {
                for channel in 0..channels {
                    let base = add(mul(batch, batch_stride)?, mul(channel, channel_stride)?)?;
                    for ky in 0..kh {
                        for kx in 0..k[0] {
                            work.tick()?;
                            let ix = position(x, kx, p.stride[0], p.padding[0], p.dilation[0]);
                            let iy = position(y, ky, p.stride[1], p.padding[1], p.dilation[1]);
                            if !in_axis(ix, n[0]) || !in_axis(iy, height) {
                                continue;
                            }
                            // Frozen 1D/2D CPU ignores the row stride inside a channel.
                            let offset = add(
                                base,
                                mul(add(mul(iy as usize, n[0])?, ix as usize)?, scalar_size)?,
                            )?;
                            let index = ((batch * oh + y) * shape[1] + x) * shape[0]
                                + (channel * kh + ky) * k[0]
                                + kx;
                            if dtype == image.dtype() {
                                let size = dtype.type_size();
                                out[index * size..(index + 1) * size]
                                    .copy_from_slice(source.word(offset)?);
                            } else {
                                write_value(&mut out, dtype, index, source.read(offset)?);
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_bytes(desc, out)
}
fn im2col_back(
    gradient: &Tensor,
    kernel: &Tensor,
    p: Conv2dParams,
    is_2d: bool,
    shape: [usize; 4],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let expected = extraction_shape(kernel.shape(), shape, p, is_2d)?;
    require_f32(gradient)?;
    if !matches!(kernel.dtype(), DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "im2col backward kernel metadata must be F32/F16",
        ));
    }
    if gradient.shape() != expected {
        return Err(TensorError::Shape(
            "im2col gradient shape does not match the forward geometry",
        ));
    }
    let k = kernel.shape();
    let (batches, channels, height, kh, oh) = if is_2d {
        (shape[3], shape[2], shape[1], k[1], expected[2])
    } else {
        (shape[2], shape[1], 1, 1, 1)
    };
    let input = Reader::new(gradient)?;
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    for batch in 0..batches {
        for channel in 0..channels {
            for iy in 0..height {
                for ix in 0..shape[0] {
                    let mut sum = 0.0_f32;
                    for ky in 0..kh {
                        for kx in 0..k[0] {
                            work.tick()?;
                            let tx = ix as i64 + i64::from(p.padding[0])
                                - kx as i64 * i64::from(p.dilation[0]);
                            if tx % i64::from(p.stride[0]) != 0 {
                                continue;
                            }
                            let x = tx / i64::from(p.stride[0]);
                            let y = if is_2d {
                                let ty = iy as i64 + i64::from(p.padding[1])
                                    - ky as i64 * i64::from(p.dilation[1]);
                                if ty % i64::from(p.stride[1]) != 0 {
                                    continue;
                                }
                                ty / i64::from(p.stride[1])
                            } else {
                                0
                            };
                            if !in_axis(x, expected[1]) || !in_axis(y, oh) {
                                continue;
                            }
                            let index = ((batch * oh + y as usize) * expected[1] + x as usize)
                                * expected[0]
                                + (channel * kh + ky) * k[0]
                                + kx;
                            // The source backward kernel reads the gradient as a flat array.
                            sum += input.read(mul(index, 4)?)?;
                        }
                    }
                    let index = ((batch * channels + channel) * height + iy) * shape[0] + ix;
                    out[index] = sum;
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
fn extraction3d_shape(
    kernel: [usize; 4],
    image: [usize; 4],
    p: Conv3dParams,
    channels: usize,
) -> Result<[usize; 4]> {
    nonempty(kernel)?;
    nonempty(image)?;
    positive(channels)?;
    validate_convolution(&p.stride, &p.padding, &p.dilation)?;
    if !kernel[3].is_multiple_of(channels) || !image[3].is_multiple_of(channels) {
        return Err(TensorError::Shape(
            "3D packed channel dimensions are incompatible",
        ));
    }
    let width = output_axis(
        image[0],
        kernel[0],
        p.stride[0],
        p.padding[0],
        p.dilation[0],
    )?;
    let height = output_axis(
        image[1],
        kernel[1],
        p.stride[1],
        p.padding[1],
        p.dilation[1],
    )?;
    let depth = output_axis(
        image[2],
        kernel[2],
        p.stride[2],
        p.padding[2],
        p.dilation[2],
    )?;
    Ok([
        mul(mul(mul(kernel[0], kernel[1])?, kernel[2])?, channels)?,
        width,
        height,
        mul(depth, image[3] / channels)?,
    ])
}
fn im2col_3d(
    kernel: &Tensor,
    image: &Tensor,
    p: Conv3dParams,
    channels: usize,
    dtype: DType,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let shape = extraction3d_shape(kernel.shape(), image.shape(), p, channels)?;
    require_f32(image)?;
    if !matches!(dtype, DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "3D im2col supports F32/F16 output",
        ));
    }
    if image.desc.nb[0] != 4 {
        return Err(TensorError::Layout(
            "3D im2col requires a contiguous first axis",
        ));
    }
    let n = image.shape();
    let k = kernel.shape();
    let batches = n[3] / channels;
    let depth = shape[3] / batches;
    let source = Reader::new(image)?;
    let (desc, mut out) = raw_output(dtype, shape)?;
    let mut work = Work::new(cancel)?;
    for batch in 0..batches {
        for z in 0..depth {
            for y in 0..shape[2] {
                for x in 0..shape[1] {
                    for channel in 0..channels {
                        for kz in 0..k[2] {
                            for ky in 0..k[1] {
                                for kx in 0..k[0] {
                                    work.tick()?;
                                    let ix =
                                        position(x, kx, p.stride[0], p.padding[0], p.dilation[0]);
                                    let iy =
                                        position(y, ky, p.stride[1], p.padding[1], p.dilation[1]);
                                    let iz =
                                        position(z, kz, p.stride[2], p.padding[2], p.dilation[2]);
                                    if !in_axis(ix, n[0])
                                        || !in_axis(iy, n[1])
                                        || !in_axis(iz, n[2])
                                    {
                                        continue;
                                    }
                                    let offset = source.offset([
                                        ix as usize,
                                        iy as usize,
                                        iz as usize,
                                        batch * channels + channel,
                                    ])?;
                                    let index = (((batch * depth + z) * shape[2] + y) * shape[1]
                                        + x)
                                        * shape[0]
                                        + ((channel * k[2] + kz) * k[1] + ky) * k[0]
                                        + kx;
                                    if dtype == DType::F32 {
                                        out[index * 4..index * 4 + 4]
                                            .copy_from_slice(source.word(offset)?);
                                    } else {
                                        write_value(&mut out, dtype, index, source.read(offset)?);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_bytes(desc, out)
}
fn col2im(
    source: &Tensor,
    stride: usize,
    channels: usize,
    padding: usize,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let n = nonempty(source.shape())?;
    positive(stride)?;
    positive(channels)?;
    size32(padding)?;
    if n[2] != 1 || n[3] != 1 || !n[0].is_multiple_of(channels) || !source.desc.is_contiguous() {
        return Err(TensorError::Shape(
            "col2im requires a contiguous matrix with complete output-channel blocks",
        ));
    }
    let input = Reader::new(source)?;
    let k = n[0] / channels;
    let width = transposed_axis(n[1], k, stride, padding)?;
    let shape = [width, channels, 1, 1];
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    for channel in 0..channels {
        for t in 0..width {
            let absolute = t as i64 + padding as i64;
            let first = ((absolute - k as i64 + 1 + stride as i64 - 1) / stride as i64).max(0);
            let last = (absolute / stride as i64).min(n[1] as i64 - 1);
            let mut sum = 0.0_f32;
            for ti in first..=last {
                work.tick()?;
                let ki = absolute - ti * stride as i64;
                if !in_axis(ki, k) {
                    continue;
                }
                let index = (channel * k + ki as usize) + ti as usize * n[0];
                sum += input.read(mul(index, source.dtype().type_size())?)?;
            }
            out[t + channel * width] = sum;
        }
    }
    typed_output(TensorDesc::contiguous(source.dtype(), shape)?, &out)
}

fn convolution_types(kernel: &Tensor, image: &Tensor) -> Result<()> {
    if !matches!(kernel.dtype(), DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "native direct/transpose convolution weights must be F32/F16",
        ));
    }
    require_f32(image)
}
fn transpose1d(
    kernel: &Tensor,
    image: &Tensor,
    stride: usize,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    convolution_types(kernel, image)?;
    let k = nonempty(kernel.shape())?;
    let n = nonempty(image.shape())?;
    positive(stride)?;
    if n[2] != 1 || n[3] != 1 || k[3] != 1 || k[2] != n[1] {
        return Err(TensorError::Shape(
            "transpose-1D kernel/input channels or matrix shape do not match",
        ));
    }
    if kernel.desc.nb[0] != kernel.dtype().type_size() || image.desc.nb[0] != 4 {
        return Err(TensorError::Layout(
            "transpose convolution requires contiguous first axes",
        ));
    }
    let shape = [transposed_axis(n[0], k[0], stride, 0)?, k[1], 1, 1];
    let mut out = zeros(elements(shape)?)?;
    let weights = Reader::new(kernel)?;
    let input = Reader::new(image)?;
    let mut work = Work::new(cancel)?;
    for channel in 0..k[1] {
        for x in 0..n[0] {
            for kx in 0..k[0] {
                let mut sum = 0.0_f64;
                for ci in 0..k[2] {
                    work.tick()?;
                    sum += f64::from(weights.at([kx, channel, ci, 0])? * input.at([x, ci, 0, 0])?);
                }
                out[x * stride + kx + channel * shape[0]] += sum as f32;
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
fn transpose2d(
    kernel: &Tensor,
    image: &Tensor,
    stride: usize,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    convolution_types(kernel, image)?;
    let k = nonempty(kernel.shape())?;
    let n = nonempty(image.shape())?;
    positive(stride)?;
    if k[3] != n[2] {
        return Err(TensorError::Shape(
            "transpose-2D kernel/input channels do not match",
        ));
    }
    let weight_size = kernel.dtype().type_size();
    if kernel.desc.nb[0] != weight_size || image.desc.nb[0] != 4 {
        return Err(TensorError::Layout(
            "transpose convolution requires contiguous first axes",
        ));
    }
    let shape = [
        transposed_axis(n[0], k[0], stride, 0)?,
        transposed_axis(n[1], k[1], stride, 0)?,
        k[2],
        n[3],
    ];
    let mut out = zeros(elements(shape)?)?;
    let weights = Reader::new(kernel)?;
    let input = Reader::new(image)?;
    let mut work = Work::new(cancel)?;
    for channel in 0..k[2] {
        for batch in 0..n[3] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    for ky in 0..k[1] {
                        for kx in 0..k[0] {
                            let mut sum = 0.0_f64;
                            for ci in 0..n[2] {
                                work.tick()?;
                                // Source kernel preparation flattens each KW*KH plane.
                                let offset = add(
                                    add(
                                        mul(ci, kernel.desc.nb[3])?,
                                        mul(channel, kernel.desc.nb[2])?,
                                    )?,
                                    mul(ky * k[0] + kx, weight_size)?,
                                )?;
                                sum +=
                                    f64::from(weights.read(offset)? * input.at([x, y, ci, batch])?);
                            }
                            out[flat([x * stride + kx, y * stride + ky, channel, batch], shape)] +=
                                sum as f32;
                        }
                    }
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
fn conv2d(kernel: &Tensor, image: &Tensor, p: Conv2dParams, cancel: &AtomicBool) -> Result<Tensor> {
    conv_direct(
        kernel,
        image,
        Conv3dParams {
            stride: [p.stride[0], p.stride[1], 1],
            padding: [p.padding[0], p.padding[1], 0],
            dilation: [p.dilation[0], p.dilation[1], 1],
        },
        image.shape()[2],
        false,
        cancel,
    )
}
fn conv3d(
    kernel: &Tensor,
    image: &Tensor,
    p: Conv3dParams,
    channels: usize,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    conv_direct(kernel, image, p, channels, true, cancel)
}
fn conv_direct(
    kernel: &Tensor,
    image: &Tensor,
    p: Conv3dParams,
    channels: usize,
    is_3d: bool,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    convolution_types(kernel, image)?;
    let k = nonempty(kernel.shape())?;
    let n = nonempty(image.shape())?;
    positive(channels)?;
    validate_convolution(&p.stride, &p.padding, &p.dilation)?;
    if !kernel.desc.is_contiguous() {
        return Err(TensorError::Layout(
            "direct convolution requires contiguous kernel storage",
        ));
    }
    let (depth, kd, batches, output_channels) = if is_3d {
        if !k[3].is_multiple_of(channels) || !n[3].is_multiple_of(channels) {
            return Err(TensorError::Shape(
                "direct 3D convolution packed channel dimensions are incompatible",
            ));
        }
        (n[2], k[2], n[3] / channels, k[3] / channels)
    } else {
        if k[2] != channels || n[2] != channels {
            return Err(TensorError::Shape(
                "direct 2D convolution channels do not match",
            ));
        }
        (1, 1, n[3], k[3])
    };
    let out_w = output_axis(n[0], k[0], p.stride[0], p.padding[0], p.dilation[0])?;
    let out_h = output_axis(n[1], k[1], p.stride[1], p.padding[1], p.dilation[1])?;
    let out_d = output_axis(depth, kd, p.stride[2], p.padding[2], p.dilation[2])?;
    let shape = if is_3d {
        [out_w, out_h, out_d, mul(output_channels, batches)?]
    } else {
        [out_w, out_h, output_channels, batches]
    };
    let mut out = zeros(elements(shape)?)?;
    let weights = kernel.to_f32()?;
    let input = Reader::new(image)?;
    let mut work = Work::new(cancel)?;
    for batch in 0..batches {
        for z in 0..out_d {
            for y in 0..out_h {
                for x in 0..out_w {
                    for co in 0..output_channels {
                        let mut sum = 0.0_f64;
                        for ci in 0..channels {
                            for kz in 0..kd {
                                for ky in 0..k[1] {
                                    for kx in 0..k[0] {
                                        work.tick()?;
                                        let ix = position(
                                            x,
                                            kx,
                                            p.stride[0],
                                            p.padding[0],
                                            p.dilation[0],
                                        );
                                        let iy = position(
                                            y,
                                            ky,
                                            p.stride[1],
                                            p.padding[1],
                                            p.dilation[1],
                                        );
                                        let iz = position(
                                            z,
                                            kz,
                                            p.stride[2],
                                            p.padding[2],
                                            p.dilation[2],
                                        );
                                        let value = if in_axis(ix, n[0])
                                            && in_axis(iy, n[1])
                                            && in_axis(iz, depth)
                                        {
                                            let at = if is_3d {
                                                [
                                                    ix as usize,
                                                    iy as usize,
                                                    iz as usize,
                                                    batch * channels + ci,
                                                ]
                                            } else {
                                                [ix as usize, iy as usize, ci, batch]
                                            };
                                            input.at(at)?
                                        } else {
                                            0.0
                                        };
                                        let wi = (((co * channels + ci) * kd + kz) * k[1] + ky)
                                            * k[0]
                                            + kx;
                                        // The frozen scalar matmul rounds the product to F32
                                        // before F64 summation. F16 weights do not round RHS.
                                        sum += f64::from(weights[wi] * value);
                                    }
                                }
                            }
                        }
                        let at = if is_3d {
                            [x, y, z, batch * output_channels + co]
                        } else {
                            [x, y, co, batch]
                        };
                        out[flat(at, shape)] = sum as f32;
                    }
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
fn contiguous_channels(tensor: &Tensor) -> bool {
    tensor.desc.nb[0] > tensor.desc.nb[2]
        && tensor.desc.nb[1] > tensor.desc.nb[0]
        && tensor.desc.nb[2] == tensor.dtype().type_size()
}
fn depthwise(
    kernel: &Tensor,
    image: &Tensor,
    p: Conv2dParams,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    convolution_types(kernel, image)?;
    let k = nonempty(kernel.shape())?;
    let n = nonempty(image.shape())?;
    validate_convolution(&p.stride, &p.padding, &p.dilation)?;
    if k[2] != 1 || k[3] != n[2] {
        return Err(TensorError::Shape(
            "depthwise kernel must have one input channel and one filter per image channel",
        ));
    }
    let channel_layout = contiguous_channels(image);
    let channel_kernel = !image.desc.is_contiguous() && channel_layout;
    if !image.desc.is_contiguous() && !channel_layout {
        return Err(TensorError::Unsupported(
            "depthwise input requires native WHCN or channels-contiguous storage",
        ));
    }
    if channel_kernel
        && (kernel.desc.nb[0] < kernel.desc.nb[2] || kernel.desc.nb[1] < kernel.desc.nb[0])
    {
        return Err(TensorError::Layout(
            "channels-contiguous depthwise kernel strides are incompatible",
        ));
    }
    let shape = [
        output_axis(n[0], k[0], p.stride[0], p.padding[0], p.dilation[0])?,
        output_axis(n[1], k[1], p.stride[1], p.padding[1], p.dilation[1])?,
        n[2],
        n[3],
    ];
    let mut desc = TensorDesc::contiguous(DType::F32, shape)?;
    if channel_layout {
        desc.nb[0] = mul(shape[2], 4)?;
        desc.nb[1] = mul(shape[0], desc.nb[0])?;
        desc.nb[2] = 4;
        desc.validate()?;
    }
    let input = Reader::new(image)?;
    let weights = Reader::new(kernel)?;
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    for batch in 0..n[3] {
        for channel in 0..n[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let mut sum = 0.0_f32;
                    for ky in 0..k[1] {
                        for kx in 0..k[0] {
                            work.tick()?;
                            let ix = position(x, kx, p.stride[0], p.padding[0], p.dilation[0]);
                            let iy = position(y, ky, p.stride[1], p.padding[1], p.dilation[1]);
                            if !in_axis(ix, n[0]) || !in_axis(iy, n[1]) {
                                continue;
                            }
                            let (ii, wi) = if channel_kernel {
                                (
                                    ((batch * n[1] + iy as usize) * n[0] + ix as usize) * n[2]
                                        + channel,
                                    (ky * k[0] + kx) * n[2] + channel,
                                )
                            } else {
                                (
                                    ((batch * n[2] + channel) * n[1] + iy as usize) * n[0]
                                        + ix as usize,
                                    (channel * k[1] + ky) * k[0] + kx,
                                )
                            };
                            sum += weights.read(mul(wi, kernel.dtype().type_size())?)?
                                * input.read(mul(ii, 4)?)?;
                        }
                    }
                    let index = if channel_kernel {
                        (((batch * shape[1] + y) * shape[0] + x) * shape[2]) + channel
                    } else {
                        flat([x, y, channel, batch], shape)
                    };
                    out[index] = sum;
                }
            }
        }
    }
    typed_output(desc, &out)
}

fn pool_size(input: usize, kernel: usize, stride: usize, padding: f32) -> Result<usize> {
    positive(kernel)?;
    positive(stride)?;
    if !padding.is_finite() || padding <= i32::MIN as f32 || padding >= 2_147_483_648.0_f32 {
        return Err(parameter(
            "pool padding must convert to a negatable signed I32",
        ));
    }
    // The constructor's float padding promotes the complete size expression to F32.
    let value = (input as f32 + 2.0 * padding - kernel as f32) / stride as f32 + 1.0;
    if !value.is_finite() || value < 1.0 || value >= 2_147_483_648.0_f32 {
        return Err(TensorError::Shape("pool output dimension is invalid"));
    }
    let output = value as usize;
    let integer_pad = padding as i32;
    let first = -i128::from(integer_pad);
    let last = (output as i128 - 1) * stride as i128 - i128::from(integer_pad) + kernel as i128 - 1;
    if first < i128::from(i32::MIN)
        || first > i128::from(i32::MAX)
        || last < i128::from(i32::MIN)
        || last > i128::from(i32::MAX)
    {
        return Err(TensorError::Overflow);
    }
    Ok(output)
}
fn pool_start(mode: PoolMode) -> f32 {
    match mode {
        PoolMode::Max => -f32::MAX,
        PoolMode::Average => 0.0,
    }
}
fn pool_add(mode: PoolMode, sum: &mut f32, value: f32) {
    match mode {
        PoolMode::Average => *sum = pool_ordered_add(*sum, value),
        // std::max(value, previous), including its NaN and signed-zero ordering.
        PoolMode::Max => {
            if value.partial_cmp(sum) != Some(std::cmp::Ordering::Less) {
                *sum = value;
            }
        }
    }
}
// Original pooling loops keep the first NaN operand while accumulating.
// Rust/LLVM may commute an addition and otherwise choose a later payload.
fn pool_ordered_add(left: f32, right: f32) -> f32 {
    if left.is_nan() {
        f32::from_bits(left.to_bits() | 0x0040_0000)
    } else if right.is_nan() {
        f32::from_bits(right.to_bits() | 0x0040_0000)
    } else {
        left + right
    }
}
// The two frozen compilers choose different scalar ADDSS destination
// operands in average-backward. GNU keeps the accumulated NaN; MSVC chooses
// the new gradient. Both frozen max-backward paths choose the new gradient.
// Preserve this measured operand policy explicitly so Rust optimization cannot
// commute the addition and change its NaN payload. Finite arithmetic is equal.
fn pool_backward_add(accumulator: f32, gradient: f32, mode: PoolMode) -> f32 {
    if mode == PoolMode::Average
        && cfg!(all(
            target_arch = "x86_64",
            target_os = "linux",
            target_env = "gnu"
        ))
    {
        pool_ordered_add(accumulator, gradient)
    } else {
        pool_ordered_add(gradient, accumulator)
    }
}
fn pooling_type(input: &Tensor) -> Result<()> {
    if !matches!(input.dtype(), DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "native pooling input must be F32/F16",
        ));
    }
    Ok(())
}
fn pool1d(
    source: &Tensor,
    mode: PoolMode,
    kernel: usize,
    stride: usize,
    padding: i32,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    pooling_type(source)?;
    let n = nonempty(source.shape())?;
    let shape = [
        pool_size(n[0], kernel, stride, padding as f32)?,
        n[1],
        n[2],
        n[3],
    ];
    let input = Reader::new(source)?;
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    let rows = mul(mul(n[1], n[2])?, n[3])?;
    for row in 0..rows {
        // The original flattens all outer axes using nb[1].
        let base = mul(row, source.desc.nb[1])?;
        for x in 0..shape[0] {
            let mut sum = pool_start(mode);
            let mut count = 0_usize;
            let start = x as i64 * stride as i64 - i64::from(padding);
            for k in 0..kernel {
                work.tick()?;
                let ix = start + k as i64;
                if !in_axis(ix, n[0]) {
                    continue;
                }
                pool_add(
                    mode,
                    &mut sum,
                    input.read(add(base, mul(ix as usize, source.dtype().type_size())?)?)?,
                );
                count += 1;
            }
            if mode == PoolMode::Average {
                sum = if count > 0 { sum / count as f32 } else { 0.0 };
            }
            out[row * shape[0] + x] = sum;
        }
    }
    Tensor::from_f32(shape, &out)
}
fn pool2d_geometry(
    n: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [f32; 2],
) -> Result<([usize; 4], usize, [i32; 2])> {
    let shape = [
        pool_size(n[0], kernel[0], stride[0], padding[0])?,
        pool_size(n[1], kernel[1], stride[1], padding[1])?,
        n[2],
        n[3],
    ];
    let area = mul(kernel[0], kernel[1])?;
    size32(area)?;
    Ok((shape, area, padding.map(|p| p as i32)))
}
fn pool2d(
    source: &Tensor,
    mode: PoolMode,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [f32; 2],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    pooling_type(source)?;
    let n = nonempty(source.shape())?;
    let (shape, area, pad) = pool2d_geometry(n, kernel, stride, padding)?;
    let input = Reader::new(source)?;
    let planes = mul(n[2], n[3])?;
    // Source advances cdata by nb[2] until byte-span end. Some exotic descriptors
    // would write more output planes than allocated in C; reject those domains.
    if source.desc.nb[2] == 0 || input.bytes.len().div_ceil(source.desc.nb[2]) != planes {
        return Err(TensorError::Layout(
            "pool2D native plane stepping exceeds its declared output",
        ));
    }
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    for plane in 0..planes {
        let base = mul(plane, source.desc.nb[2])?;
        for y in 0..shape[1] {
            for x in 0..shape[0] {
                let mut sum = pool_start(mode);
                let sx = x as i64 * stride[0] as i64 - i64::from(pad[0]);
                let sy = y as i64 * stride[1] as i64 - i64::from(pad[1]);
                for ky in 0..kernel[1] {
                    for kx in 0..kernel[0] {
                        work.tick()?;
                        let ix = sx + kx as i64;
                        let iy = sy + ky as i64;
                        if !in_axis(ix, n[0]) || !in_axis(iy, n[1]) {
                            continue;
                        }
                        let offset = add(
                            add(base, mul(iy as usize, source.desc.nb[1])?)?,
                            mul(ix as usize, source.dtype().type_size())?,
                        )?;
                        pool_add(mode, &mut sum, input.read(offset)?);
                    }
                }
                if mode == PoolMode::Average {
                    sum /= area as f32;
                }
                out[(plane * shape[1] + y) * shape[0] + x] = sum;
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
fn pool2d_back(
    gradient: &Tensor,
    forward: &Tensor,
    mode: PoolMode,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [f32; 2],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    require_f32(gradient)?;
    // Constructor output is always F32 and the CPU chooses forward loads from
    // that output dtype. A half forward buffer would be reinterpreted as F32.
    require_f32(forward)?;
    let shape = nonempty(forward.shape())?;
    let (expected, area, pad) = pool2d_geometry(shape, kernel, stride, padding)?;
    if gradient.shape() != expected {
        return Err(TensorError::Shape(
            "pool backward gradient shape differs from forward geometry",
        ));
    }
    let grad = Reader::new(gradient)?;
    let input = Reader::new(forward)?;
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    for plane in 0..mul(shape[2], shape[3])? {
        for y in 0..expected[1] {
            for x in 0..expected[0] {
                let g = grad.read(mul((plane * expected[1] + y) * expected[0] + x, 4)?)?;
                let sx = x as i64 * stride[0] as i64 - i64::from(pad[0]);
                let sy = y as i64 * stride[1] as i64 - i64::from(pad[1]);
                let mut maximum = -f32::MAX;
                let mut chosen = None;
                for ky in 0..kernel[1] {
                    for kx in 0..kernel[0] {
                        work.tick()?;
                        let ix = sx + kx as i64;
                        let iy = sy + ky as i64;
                        if !in_axis(ix, shape[0]) || !in_axis(iy, shape[1]) {
                            continue;
                        }
                        let at = (plane * shape[1] + iy as usize) * shape[0] + ix as usize;
                        if mode == PoolMode::Average {
                            out[at] = pool_backward_add(out[at], g / area as f32, mode);
                        } else {
                            // Source intentionally addresses the forward data
                            // using contiguous result strides, ignoring its own.
                            let value = input.read(mul(at, 4)?)?;
                            if value <= maximum {
                                continue;
                            }
                            maximum = value;
                            chosen = Some(at);
                        }
                    }
                }
                if let Some(at) = chosen {
                    out[at] = pool_backward_add(out[at], g, mode);
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
fn cubic(p: [f32; 4], x: f32) -> f32 {
    let a = -0.75_f32;
    let weight1 = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let weight2 = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    let w0 = weight2(x + 1.0);
    let w1 = weight1(x);
    let w2 = weight1(1.0 - x);
    let w3 = weight2(2.0 - x);
    p[0] * w0 + p[1] * w1 + p[2] * w2 + p[3] * w3
}
fn resize(
    source: &Tensor,
    shape: [usize; 4],
    mode: ResizeMode,
    align: bool,
    antialias: bool,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    require_f32(source)?;
    let n = nonempty(source.shape())?;
    nonempty(shape)?;
    if antialias && mode != ResizeMode::Bilinear {
        return Err(parameter(
            "native antialiasing is available only for bilinear resize",
        ));
    }
    let mut scale = std::array::from_fn::<_, 4, _>(|i| shape[i] as f32 / n[i] as f32);
    let pixel_offset = if align {
        for i in 0..2 {
            if shape[i] > 1 && n[i] > 1 {
                scale[i] = (shape[i] - 1) as f32 / (n[i] - 1) as f32;
            }
        }
        0.0_f32
    } else {
        0.5_f32
    };
    let input = Reader::new(source)?;
    let mut out = zeros(elements(shape)?)?;
    let mut work = Work::new(cancel)?;
    for b in 0..shape[3] {
        let ib = (b as f32 / scale[3]) as usize;
        for channel in 0..shape[2] {
            let ic = (channel as f32 / scale[2]) as usize;
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    work.tick()?;
                    let value = match mode {
                        ResizeMode::Nearest => {
                            let ix = (x as f32 / scale[0]) as usize;
                            let iy = (y as f32 / scale[1]) as usize;
                            input.at([ix, iy, ic, ib])?
                        }
                        ResizeMode::Bilinear if antialias => {
                            let support_x = (1.0 / scale[0]).max(1.0);
                            let support_y = (1.0 / scale[1]).max(1.0);
                            let inverse_x = 1.0 / support_x;
                            let inverse_y = 1.0 / support_y;
                            let xf = (x as f32 + pixel_offset) / scale[0];
                            let yf = (y as f32 + pixel_offset) / scale[1];
                            let x_min = ((xf - support_x + pixel_offset) as i64).max(0);
                            let x_max = ((xf + support_x + pixel_offset) as i64).min(n[0] as i64);
                            let y_min = ((yf - support_y + pixel_offset) as i64).max(0);
                            let y_max = ((yf + support_y + pixel_offset) as i64).min(n[1] as i64);
                            let mut sum = 0.0_f32;
                            let mut total = 0.0_f32;
                            for iy in y_min..y_max {
                                let wy = (1.0
                                    - ((iy as f32 - yf + pixel_offset) * inverse_y).abs())
                                .max(0.0);
                                for ix in x_min..x_max {
                                    work.tick()?;
                                    let wx = (1.0
                                        - ((ix as f32 - xf + pixel_offset) * inverse_x).abs())
                                    .max(0.0);
                                    let weight = wx * wy;
                                    if weight <= 0.0 {
                                        continue;
                                    }
                                    sum += input.at([ix as usize, iy as usize, ic, ib])? * weight;
                                    total += weight;
                                }
                            }
                            if total > 0.0 {
                                sum /= total;
                            }
                            sum
                        }
                        ResizeMode::Bilinear => {
                            let xf = (x as f32 + pixel_offset) / scale[0] - pixel_offset;
                            let yf = (y as f32 + pixel_offset) / scale[1] - pixel_offset;
                            let x0 = xf.floor() as i64;
                            let y0 = yf.floor() as i64;
                            let x1 = (x0 + 1).clamp(0, n[0] as i64 - 1) as usize;
                            let y1 = (y0 + 1).clamp(0, n[1] as i64 - 1) as usize;
                            let x0 = x0.clamp(0, n[0] as i64 - 1) as usize;
                            let y0 = y0.clamp(0, n[1] as i64 - 1) as usize;
                            let dx = (xf - x0 as f32).clamp(0.0, 1.0);
                            let dy = (yf - y0 as f32).clamp(0.0, 1.0);
                            let a = input.at([x0, y0, ic, ib])?;
                            let b = input.at([x1, y0, ic, ib])?;
                            let c = input.at([x0, y1, ic, ib])?;
                            let d = input.at([x1, y1, ic, ib])?;
                            a * (1.0 - dx) * (1.0 - dy)
                                + b * dx * (1.0 - dy)
                                + c * (1.0 - dx) * dy
                                + d * dx * dy
                        }
                        ResizeMode::Bicubic => {
                            let xf = (x as f32 + pixel_offset) / scale[0] - pixel_offset;
                            let yf = (y as f32 + pixel_offset) / scale[1] - pixel_offset;
                            let x0 = xf.floor() as i64;
                            let y0 = yf.floor() as i64;
                            let dx = xf - x0 as f32;
                            let dy = yf - y0 as f32;
                            let mut rows = [0.0; 4];
                            for (ri, row) in rows.iter_mut().enumerate() {
                                let iy = (y0 + ri as i64 - 1).clamp(0, n[1] as i64 - 1) as usize;
                                let mut pixels = [0.0; 4];
                                for (ci, pixel) in pixels.iter_mut().enumerate() {
                                    let ix =
                                        (x0 + ci as i64 - 1).clamp(0, n[0] as i64 - 1) as usize;
                                    *pixel = input.at([ix, iy, ic, ib])?;
                                }
                                *row = cubic(pixels, dx);
                            }
                            cubic(rows, dy)
                        }
                    };
                    out[flat([x, y, channel, b], shape)] = value;
                }
            }
        }
    }
    Tensor::from_f32(shape, &out)
}
