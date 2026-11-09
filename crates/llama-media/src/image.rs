//! RGB image IO and the frozen snapshot's separable fixed-point resampler.
//!
//! Algorithm source: `tools/mtmd/mtmd-image.cpp::img_tool` and
//! `tools/mtmd/mtmd-impl.h::mtmd_image_f32` (upstream MIT notice in LICENSE).
use crate::{DecodeLimits, MediaError, Result, bytes, check_limit, product, u16le, u32le, zeros};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    width: usize,
    height: usize,
}
impl ImageSize {
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(MediaError::Invalid("zero image dimension"));
        }
        check_limit("image width", width, 65_536)?;
        check_limit("image height", height, 65_536)?;
        product(width, height)?;
        Ok(Self { width, height })
    }
    pub fn width(self) -> usize {
        self.width
    }
    pub fn height(self) -> usize {
        self.height
    }
    pub fn pixels(self) -> usize {
        self.width * self.height
    }
    fn elements(self) -> Result<usize> {
        product(self.pixels(), 3)
    }
}

/// Tightly packed RGB8, row-major with top-left origin. Alpha is discarded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    size: ImageSize,
    data: Vec<u8>,
}
impl RgbImage {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        let size = ImageSize::new(width, height)?;
        if data.len() != size.elements()? {
            return Err(MediaError::Invalid("RGB byte count does not match shape"));
        }
        Ok(Self { size, data })
    }
    pub fn solid(size: ImageSize, color: [u8; 3]) -> Result<Self> {
        let mut data = zeros(size.elements()?, 0u8)?;
        for pixel in data.as_chunks_mut::<3>().0 {
            pixel.copy_from_slice(&color);
        }
        Ok(Self { size, data })
    }
    pub fn size(&self) -> ImageSize {
        self.size
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
    pub fn pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.size.width || y >= self.size.height {
            return None;
        }
        let i = (y * self.size.width + x) * 3;
        Some([self.data[i], self.data[i + 1], self.data[i + 2]])
    }
    pub fn crop(&self, x: usize, y: usize, size: ImageSize) -> Result<Self> {
        if x.checked_add(size.width).ok_or(MediaError::Overflow)? > self.size.width
            || y.checked_add(size.height).ok_or(MediaError::Overflow)? > self.size.height
        {
            return Err(MediaError::Invalid("crop outside image"));
        }
        let mut data = zeros(size.elements()?, 0u8)?;
        for row in 0..size.height {
            let src = ((row + y) * self.size.width + x) * 3;
            let dst = row * size.width * 3;
            data[dst..dst + size.width * 3].copy_from_slice(&self.data[src..src + size.width * 3]);
        }
        Ok(Self { size, data })
    }
    /// Crop the central rectangle; odd excess pixels stay on the right/bottom.
    pub fn center_crop(&self, size: ImageSize) -> Result<Self> {
        let x = self
            .size
            .width
            .checked_sub(size.width)
            .ok_or(MediaError::Invalid("crop too wide"))?
            / 2;
        let y = self
            .size
            .height
            .checked_sub(size.height)
            .ok_or(MediaError::Invalid("crop too high"))?
            / 2;
        self.crop(x, y, size)
    }
    /// Composite with clipping. Offset arithmetic is checked even for extreme inputs.
    pub fn composite(&mut self, image: &Self, x: i64, y: i64) -> Result<()> {
        x.checked_add(image.size.width as i64)
            .ok_or(MediaError::Overflow)?;
        y.checked_add(image.size.height as i64)
            .ok_or(MediaError::Overflow)?;
        for sy in 0..image.size.height {
            let dy = y + sy as i64;
            if dy < 0 || dy >= self.size.height as i64 {
                continue;
            }
            for sx in 0..image.size.width {
                let dx = x + sx as i64;
                if dx < 0 || dx >= self.size.width as i64 {
                    continue;
                }
                let src = (sy * image.size.width + sx) * 3;
                let dst = (dy as usize * self.size.width + dx as usize) * 3;
                self.data[dst..dst + 3].copy_from_slice(&image.data[src..src + 3]);
            }
        }
        Ok(())
    }
    /// Split an already resized image into an exact row-major grid.
    pub fn tiles(&self, size: ImageSize) -> Result<Vec<Self>> {
        if !self.size.width.is_multiple_of(size.width)
            || !self.size.height.is_multiple_of(size.height)
        {
            return Err(MediaError::Invalid("tile shape does not divide image"));
        }
        let count = self.size.pixels() / size.pixels();
        check_limit("tile count", count, 65_536)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| MediaError::Allocation)?;
        for y in (0..self.size.height).step_by(size.height) {
            for x in (0..self.size.width).step_by(size.width) {
                out.push(self.crop(x, y, size)?);
            }
        }
        Ok(out)
    }
    /// Convert to interleaved f32 with exactly `(u8 / 255 - mean) / std`.
    pub fn normalize(&self, mean: [f32; 3], std: [f32; 3]) -> Result<ImageTensor> {
        if mean.iter().chain(&std).any(|x| !x.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        if std.iter().any(|x| *x <= 0.0) {
            return Err(MediaError::Invalid("nonpositive image standard deviation"));
        }
        let mut data = zeros(self.data.len(), 0f32)?;
        for (i, out) in data.iter_mut().enumerate() {
            *out = (self.data[i] as f32 / 255.0 - mean[i % 3]) / std[i % 3];
            if !out.is_finite() {
                return Err(MediaError::NonFinite);
            }
        }
        Ok(ImageTensor {
            size: self.size,
            data,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageTensor {
    size: ImageSize,
    data: Vec<f32>,
}
impl ImageTensor {
    pub fn size(&self) -> ImageSize {
        self.size
    }
    pub fn data_hwc(&self) -> &[f32] {
        &self.data
    }
    pub fn to_chw(&self) -> Result<Vec<f32>> {
        let mut out = zeros(self.data.len(), 0f32)?;
        for channel in 0..3 {
            for i in 0..self.size.pixels() {
                out[channel * self.size.pixels() + i] = self.data[3 * i + channel];
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    Bilinear,
    Bicubic,
    Lanczos,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPrecision {
    Pillow22,
    Torch16,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    None,
    Ceil,
    Nearest,
}

fn filter_weight(filter: ResizeFilter, x: f64) -> f64 {
    if filter == ResizeFilter::Lanczos {
        if !(-3.0..3.0).contains(&x) {
            return 0.0;
        }
        let sinc = |x: f64| {
            if x == 0.0 {
                1.0
            } else {
                let p = x * std::f64::consts::PI;
                p.sin() / p
            }
        };
        return sinc(x) * sinc(x / 3.0);
    }
    let x = x.abs();
    if filter == ResizeFilter::Bilinear {
        return if x < 1.0 { 1.0 - x } else { 0.0 };
    }
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

struct Weights {
    start: Vec<usize>,
    counts: Vec<usize>,
    weights: Vec<i32>,
    stride: usize,
}
fn weights(
    input: usize,
    output: usize,
    filter: ResizeFilter,
    precision: WeightPrecision,
) -> Result<Weights> {
    let radius = match filter {
        ResizeFilter::Bilinear => 1.0,
        ResizeFilter::Bicubic => 2.0,
        ResizeFilter::Lanczos => 3.0,
    };
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = radius * filterscale;
    let stride = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(MediaError::Overflow)?;
    let mut raw = zeros(product(output, stride)?, 0f64)?;
    let mut start = zeros(output, 0usize)?;
    let mut counts = zeros(output, 0usize)?;
    for index in 0..output {
        let center = (index as f64 + 0.5) * scale;
        // C++ truncation to int (towards zero) is significant at the left edge.
        let lo = ((center - support + 0.5) as i64).max(0) as usize;
        let hi = ((center + support + 0.5) as usize).min(input);
        let count = hi - lo;
        let mut sum = 0.0;
        for k in 0..count {
            let w = filter_weight(filter, (k as f64 + lo as f64 - center + 0.5) / filterscale);
            raw[index * stride + k] = w;
            sum += w;
        }
        if sum != 0.0 {
            for w in &mut raw[index * stride..index * stride + count] {
                *w /= sum;
            }
        }
        start[index] = lo;
        counts[index] = count;
    }
    let mut bits = 22u32;
    if precision == WeightPrecision::Torch16 {
        let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for p in 0..22 {
            bits = p;
            if (0.5 + max * (1u32 << (p + 1)) as f64) as i64 >= 32768 {
                break;
            }
            bits = p + 1;
        }
    }
    let mut result = zeros(raw.len(), 0i32)?;
    let factor = (1u32 << bits) as f64;
    for (out, w) in result.iter_mut().zip(raw) {
        let rounded = w * factor + if w < 0.0 { -0.5 } else { 0.5 };
        *out = (rounded as i32) * (1 << (22 - bits));
    }
    Ok(Weights {
        start,
        counts,
        weights: result,
        stride,
    })
}

/// Pillow-compatible two-pass resize with antialiasing and clipped RGB8 passes.
pub fn resize(
    image: &RgbImage,
    size: ImageSize,
    filter: ResizeFilter,
    precision: WeightPrecision,
) -> Result<RgbImage> {
    if precision == WeightPrecision::Torch16 && filter == ResizeFilter::Lanczos {
        return Err(MediaError::Unsupported("torchvision Lanczos weights"));
    }
    if image.size == size {
        return Ok(image.clone());
    }
    let mut intermediate;
    let horizontal = if size.width != image.size.width {
        let w = weights(image.size.width, size.width, filter, precision)?;
        intermediate = zeros(product(product(size.width, image.size.height)?, 3)?, 0u8)?;
        for row in 0..image.size.height {
            for col in 0..size.width {
                for channel in 0..3 {
                    let mut sum = 1i64 << 21;
                    for k in 0..w.counts[col] {
                        let pixel = image.data
                            [(row * image.size.width + w.start[col] + k) * 3 + channel]
                            as i64;
                        sum += pixel * w.weights[col * w.stride + k] as i64;
                    }
                    intermediate[(row * size.width + col) * 3 + channel] =
                        (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        &intermediate
    } else {
        &image.data
    };
    let mut data = zeros(size.elements()?, 0u8)?;
    if size.height != image.size.height {
        let w = weights(image.size.height, size.height, filter, precision)?;
        let row_len = size.width * 3;
        for row in 0..size.height {
            for channel_pixel in 0..row_len {
                let mut sum = 1i64 << 21;
                for k in 0..w.counts[row] {
                    sum += horizontal[(w.start[row] + k) * row_len + channel_pixel] as i64
                        * w.weights[row * w.stride + k] as i64;
                }
                data[row * row_len + channel_pixel] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    } else {
        data.copy_from_slice(horizontal);
    }
    Ok(RgbImage { size, data })
}

/// Aspect-preserving letterbox with the snapshot's ceil/nearest rounding.
pub fn resize_padded(
    image: &RgbImage,
    size: ImageSize,
    filter: ResizeFilter,
    precision: WeightPrecision,
    padding: Padding,
    color: [u8; 3],
) -> Result<RgbImage> {
    if padding == Padding::None {
        return resize(image, size, filter, precision);
    }
    let scale = (size.width as f32 / image.size.width as f32)
        .min(size.height as f32 / image.size.height as f32);
    let round = |v: f32| {
        if padding == Padding::Nearest {
            v.round()
        } else {
            v.ceil()
        }
    };
    let width = (round(image.size.width as f32 * scale) as usize).min(size.width);
    let height = (round(image.size.height as f32 * scale) as usize).min(size.height);
    let scaled = resize(image, ImageSize::new(width, height)?, filter, precision)?;
    let mut out = RgbImage::solid(size, color)?;
    let x = if padding == Padding::Nearest {
        ((size.width - width) as f32 / 2.0).round() as i64
    } else {
        ((size.width - width) / 2) as i64
    };
    let y = if padding == Padding::Nearest {
        ((size.height - height) as f32 / 2.0).round() as i64
    } else {
        ((size.height - height) / 2) as i64
    };
    out.composite(&scaled, x, y)?;
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmartResize {
    pub alignment: usize,
    pub min_pixels: Option<usize>,
    pub max_pixels: Option<usize>,
    pub longest_edge: Option<usize>,
    pub ties_even: bool,
}
impl Default for SmartResize {
    fn default() -> Self {
        Self {
            alignment: 1,
            min_pixels: None,
            max_pixels: None,
            longest_edge: None,
            ties_even: false,
        }
    }
}
impl SmartResize {
    /// Preserve reference order: alignment/longest edge, then max or min pixels.
    /// A minimum pixel constraint can expand beyond `longest_edge`, as upstream.
    pub fn size_for(self, input: ImageSize) -> Result<ImageSize> {
        if self.alignment == 0
            || self.alignment > 65_536
            || self.min_pixels == Some(0)
            || self.max_pixels == Some(0)
            || self.longest_edge == Some(0)
        {
            return Err(MediaError::Invalid("invalid smart resize bounds"));
        }
        if let (Some(lo), Some(hi)) = (self.min_pixels, self.max_pixels)
            && lo > hi
        {
            return Err(MediaError::Invalid("minimum pixels exceed maximum"));
        }
        let factor = self.alignment as f32;
        let rounded = |x: f32| {
            if self.ties_even {
                (x / factor).round_ties_even() * factor
            } else {
                (x / factor).round() * factor
            }
        };
        let up = |x: f32| (x / factor).ceil() * factor;
        let down = |x: f32| (x / factor).floor() * factor;
        let width = input.width as f32;
        let height = input.height as f32;
        let (mut w, mut h) = if let Some(edge) = self.longest_edge {
            let scale = (edge as f32 / width).min(edge as f32 / height);
            (up(width * scale), up(height * scale))
        } else {
            (factor.max(rounded(width)), factor.max(rounded(height)))
        };
        if self
            .max_pixels
            .is_some_and(|max| w as f64 * h as f64 > max as f64)
        {
            let beta = (height * width / self.max_pixels.unwrap() as f32).sqrt();
            h = factor.max(down(height / beta));
            w = factor.max(down(width / beta));
        } else if self
            .min_pixels
            .is_some_and(|min| w as f64 * (h as f64) < min as f64)
        {
            let beta = (self.min_pixels.unwrap() as f32 / (height * width)).sqrt();
            h = up(height * beta);
            w = up(width * beta);
        }
        if !w.is_finite() || !h.is_finite() {
            return Err(MediaError::NonFinite);
        }
        ImageSize::new(w as usize, h as usize)
    }
}

/// Detect and decode only implemented variants. Unknown codecs never fall back
/// to a native executable. See the codec ledger for the full source obligation.
pub fn decode(input: &[u8], limits: DecodeLimits) -> Result<RgbImage> {
    check_limit("encoded bytes", input.len(), limits.max_input_bytes)?;
    if input.starts_with(b"\x89PNG\r\n\x1a\n") {
        crate::png::decode(input, limits)
    } else if input.starts_with(b"P5") || input.starts_with(b"P6") {
        decode_pnm(input, limits)
    } else if input.starts_with(b"BM") {
        decode_bmp(input, limits)
    } else {
        Err(MediaError::Unsupported(
            "image codec (implemented: noninterlaced PNG, binary PGM/PPM, uncompressed RGB BMP)",
        ))
    }
}

fn decode_pnm(input: &[u8], limits: DecodeLimits) -> Result<RgbImage> {
    let channels = if input.starts_with(b"P6") { 3 } else { 1 };
    let mut cursor = 2usize;
    fn number(input: &[u8], cursor: &mut usize) -> Result<usize> {
        loop {
            while input.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
                *cursor += 1;
            }
            if input.get(*cursor) == Some(&b'#') {
                while input
                    .get(*cursor)
                    .is_some_and(|c| *c != b'\n' && *c != b'\r')
                {
                    *cursor += 1;
                }
            } else {
                break;
            }
        }
        let start = *cursor;
        let mut out = 0usize;
        while let Some(&c) = input.get(*cursor).filter(|c| c.is_ascii_digit()) {
            out = out
                .checked_mul(10)
                .and_then(|n| n.checked_add((c - b'0') as usize))
                .ok_or(MediaError::Overflow)?;
            *cursor += 1;
        }
        if *cursor == start {
            return Err(MediaError::Invalid("missing PNM integer"));
        }
        if !input
            .get(*cursor)
            .is_some_and(|c| c.is_ascii_whitespace() || *c == b'#')
        {
            return Err(MediaError::Invalid("PNM integer delimiter"));
        }
        Ok(out)
    }
    let width = number(input, &mut cursor)?;
    let height = number(input, &mut cursor)?;
    let max = number(input, &mut cursor)?;
    if max != 255 {
        return Err(MediaError::Unsupported("PNM maxval other than 255"));
    }
    // Exactly one separator: binary pixels may themselves be ASCII whitespace.
    if !input.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        return Err(MediaError::Invalid("PNM raster separator"));
    }
    cursor += 1;
    let size = ImageSize::new(width, height)?;
    check_limit("image pixels", size.pixels(), limits.max_image_pixels)?;
    let raster = bytes(input, cursor, product(size.pixels(), channels)?)?;
    let mut data = zeros(size.elements()?, 0u8)?;
    if channels == 3 {
        data.copy_from_slice(raster);
    } else {
        for (pixel, &grey) in data.as_chunks_mut::<3>().0.iter_mut().zip(raster) {
            pixel.fill(grey);
        }
    }
    Ok(RgbImage { size, data })
}

fn decode_bmp(input: &[u8], limits: DecodeLimits) -> Result<RgbImage> {
    let data_offset = u32le(input, 10)? as usize;
    let header_size = u32le(input, 14)? as usize;
    if ![40, 56, 108, 124].contains(&header_size) {
        return Err(MediaError::Unsupported("BMP DIB header version"));
    }
    bytes(input, 14, header_size)?;
    let width = u32le(input, 18)? as i32;
    let raw_height = u32le(input, 22)? as i32;
    if width <= 0 || raw_height == 0 || raw_height == i32::MIN {
        return Err(MediaError::Invalid("BMP signed dimensions"));
    }
    let size = ImageSize::new(width as usize, raw_height.unsigned_abs() as usize)?;
    check_limit("image pixels", size.pixels(), limits.max_image_pixels)?;
    if u16le(input, 26)? != 1 {
        return Err(MediaError::Invalid("BMP planes must equal one"));
    }
    let depth = u16le(input, 28)?;
    if depth != 24 && depth != 32 {
        return Err(MediaError::Unsupported(
            "BMP depth (only 24 and 32 bits implemented)",
        ));
    }
    if u32le(input, 30)? != 0 {
        return Err(MediaError::Unsupported("BMP compression/bitfields"));
    }
    if data_offset < 14 + header_size {
        return Err(MediaError::Invalid("BMP raster overlaps header"));
    }
    let pixel_bytes = depth as usize / 8;
    let stride = product(size.width, pixel_bytes)?
        .checked_add(3)
        .ok_or(MediaError::Overflow)?
        & !3;
    bytes(input, data_offset, product(stride, size.height)?)?;
    let mut data = zeros(size.elements()?, 0u8)?;
    for y in 0..size.height {
        let source_y = if raw_height < 0 {
            y
        } else {
            size.height - y - 1
        };
        for x in 0..size.width {
            let source = data_offset + source_y * stride + x * pixel_bytes;
            let target = (y * size.width + x) * 3;
            data[target] = input[source + 2];
            data[target + 1] = input[source + 1];
            data[target + 2] = input[source];
        }
    }
    Ok(RgbImage { size, data })
}
