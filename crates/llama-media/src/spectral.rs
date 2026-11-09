//! Scalar FFT, window/filterbank preprocessing and incremental waveform synthesis.
//!
//! Reference algorithms: `tools/mtmd/mtmd-audio.cpp` (cache, FFT, log-mel frame,
//! streaming ISTFT). No GGML graph or native FFT library supplies computation.
use crate::{MediaError, Result, check_limit, product, zeros};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}
impl Complex32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    pub fn norm_squared(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Cached trig table and mixed radix-2/odd-DFT transform, matching the reference
/// decomposition. Odd transform factors above 1024 are explicitly bounded.
#[derive(Debug, Clone)]
pub struct FftPlan {
    len: usize,
    sin: Vec<f32>,
    cos: Vec<f32>,
}
impl FftPlan {
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(MediaError::Invalid("zero FFT length"));
        }
        check_limit("FFT length", len, 65_536)?;
        let odd_factor = len >> len.trailing_zeros();
        check_limit("odd FFT factor", odd_factor, 1024)?;
        let mut sin = zeros(len, 0f32)?;
        let mut cos = zeros(len, 0f32)?;
        for i in 0..len {
            let angle = (2.0 * std::f64::consts::PI * i as f64 / len as f64) as f32;
            sin[i] = angle.sin();
            cos[i] = angle.cos();
        }
        Ok(Self { len, sin, cos })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn forward_real(&self, input: &[f32]) -> Result<Vec<Complex32>> {
        if input.len() != self.len {
            return Err(MediaError::Invalid("FFT input length"));
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        let mut complex = zeros(self.len, Complex32::default())?;
        for (out, &value) in complex.iter_mut().zip(input) {
            out.re = value;
        }
        self.transform(&complex, false)
    }
    pub fn forward(&self, input: &[Complex32]) -> Result<Vec<Complex32>> {
        self.transform(input, false)
    }
    pub fn inverse(&self, input: &[Complex32]) -> Result<Vec<Complex32>> {
        self.transform(input, true)
    }
    fn transform(&self, input: &[Complex32], inverse: bool) -> Result<Vec<Complex32>> {
        if input.len() != self.len {
            return Err(MediaError::Invalid("FFT input length"));
        }
        if input.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        let mut out = zeros(self.len, Complex32::default())?;
        let mut scratch = zeros(self.len, Complex32::default())?;
        self.recurse(input, 0, 1, inverse, &mut out, &mut scratch);
        if out.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        Ok(out)
    }
    fn recurse(
        &self,
        input: &[Complex32],
        offset: usize,
        stride: usize,
        inverse: bool,
        output: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        let n = output.len();
        if n == 1 {
            output[0] = input[offset];
            return;
        }
        let sign = if inverse { 1.0 } else { -1.0 };
        if n % 2 == 1 {
            let table_step = self.len / n;
            let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
            for (k, out) in output.iter_mut().enumerate() {
                let mut re = 0f32;
                let mut im = 0f32;
                for j in 0..n {
                    let index = (k * j * table_step) % self.len;
                    let value = input[offset + j * stride];
                    re += value.re * self.cos[index] - sign * value.im * self.sin[index];
                    im += sign * value.re * self.sin[index] + value.im * self.cos[index];
                }
                *out = Complex32::new(re * scale, im * scale);
            }
            return;
        }
        let half = n / 2;
        let (even, odd) = scratch.split_at_mut(half);
        let (temp_even, temp_odd) = output.split_at_mut(half);
        self.recurse(input, offset, stride * 2, inverse, even, temp_even);
        self.recurse(input, offset + stride, stride * 2, inverse, odd, temp_odd);
        let table_step = self.len / n;
        let scale = if inverse { 0.5 } else { 1.0 };
        for k in 0..half {
            let re = self.cos[k * table_step];
            let im = sign * self.sin[k * table_step];
            let a = even[k];
            let b = odd[k];
            output[k] = Complex32::new(
                scale * (a.re + re * b.re - im * b.im),
                scale * (a.im + re * b.im + im * b.re),
            );
            output[k + half] = Complex32::new(
                scale * (a.re - re * b.re + im * b.im),
                scale * (a.im - re * b.im - im * b.re),
            );
        }
    }
}

/// The f32 trigonometric argument is intentional: source calls `cosf`.
pub fn hann_window(len: usize, periodic: bool) -> Result<Vec<f32>> {
    if len == 0 || (!periodic && len == 1) {
        return Err(MediaError::Invalid("Hann window length"));
    }
    check_limit("Hann length", len, 65_536)?;
    let denominator = if periodic { len } else { len - 1 };
    let mut window = zeros(len, 0f32)?;
    for (i, out) in window.iter_mut().enumerate() {
        let angle = (2.0 * std::f64::consts::PI * i as f64 / denominator as f64) as f32;
        *out = (0.5 * (1.0 - angle.cos() as f64)) as f32;
    }
    Ok(window)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    Slaney,
    Htk,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelConfig {
    pub fft_len: usize,
    pub bands: usize,
    pub sample_rate: u32,
    pub min_hz: f32,
    /// `None` uses Nyquist.
    pub max_hz: Option<f32>,
    pub scale: MelScale,
    pub area_normalize: bool,
    pub multiplier: f32,
}
impl MelConfig {
    pub fn slaney(fft_len: usize, bands: usize, sample_rate: u32) -> Self {
        Self {
            fft_len,
            bands,
            sample_rate,
            min_hz: 0.0,
            max_hz: None,
            scale: MelScale::Slaney,
            area_normalize: true,
            multiplier: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelFilterbank {
    config: MelConfig,
    weights: Vec<f32>,
}
impl MelFilterbank {
    pub fn new(config: MelConfig) -> Result<Self> {
        if config.fft_len < 2
            || config.bands == 0
            || config.sample_rate == 0
            || config.sample_rate > 768_000
        {
            return Err(MediaError::Invalid("mel dimensions or rate"));
        }
        check_limit("mel FFT length", config.fft_len, 65_536)?;
        check_limit("mel band count", config.bands, 4096)?;
        let max_hz = config.max_hz.unwrap_or(config.sample_rate as f32 * 0.5);
        if !config.min_hz.is_finite() || !max_hz.is_finite() || !config.multiplier.is_finite() {
            return Err(MediaError::NonFinite);
        }
        if config.min_hz < 0.0
            || max_hz <= config.min_hz
            || max_hz > config.sample_rate as f32 / 2.0
            || config.multiplier <= 0.0
        {
            return Err(MediaError::Invalid("mel frequency range or scale"));
        }
        let hz_to_mel = |hz: f64| match config.scale {
            MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            MelScale::Slaney => {
                if hz < 1000.0 {
                    hz * (3.0 / 200.0)
                } else {
                    15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
                }
            }
        };
        let mel_to_hz = |mel: f64| match config.scale {
            MelScale::Htk => 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0),
            MelScale::Slaney => {
                if mel < 15.0 {
                    mel / (3.0 / 200.0)
                } else {
                    1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
                }
            }
        };
        let low = hz_to_mel(config.min_hz as f64);
        let high = hz_to_mel(max_hz as f64);
        let mut hz = zeros(config.bands + 2, 0f64)?;
        for (i, hz) in hz.iter_mut().enumerate() {
            *hz = mel_to_hz(low + (high - low) * (i as f64 / (config.bands + 1) as f64));
        }
        let bins = config.fft_len / 2 + 1;
        let mut weights = zeros(product(config.bands, bins)?, 0f32)?;
        let hz_step = config.sample_rate as f64 / config.fft_len as f64;
        for band in 0..config.bands {
            let (left, center, right) = (hz[band], hz[band + 1], hz[band + 2]);
            let dl = (center - left).max(1e-30);
            let dr = (right - center).max(1e-30);
            let norm = if config.area_normalize {
                2.0 / (right - left).max(1e-30)
            } else {
                1.0
            };
            for bin in 0..bins {
                let f = bin as f64 * hz_step;
                let value = if f >= left && f <= center {
                    (f - left) / dl
                } else if f > center && f <= right {
                    (right - f) / dr
                } else {
                    0.0
                };
                weights[band * bins + bin] = (value * norm * config.multiplier as f64) as f32;
            }
        }
        Ok(Self { config, weights })
    }
    pub fn config(&self) -> MelConfig {
        self.config
    }
    pub fn bins(&self) -> usize {
        self.config.fft_len / 2 + 1
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn energies(&self, spectrum: &[Complex32], magnitude: bool) -> Result<Vec<f32>> {
        if spectrum.len() < self.bins() {
            return Err(MediaError::Invalid("spectrum shorter than mel bins"));
        }
        if spectrum[..self.bins()]
            .iter()
            .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return Err(MediaError::NonFinite);
        }
        let mut power = zeros(self.bins(), 0f32)?;
        for (out, sample) in power.iter_mut().zip(spectrum) {
            *out = if magnitude {
                sample.norm_squared().sqrt()
            } else {
                sample.norm_squared()
            };
        }
        let mut energies = zeros(self.config.bands, 0f32)?;
        for (band, out) in energies.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (bin, &p) in power.iter().enumerate() {
                sum += (p * self.weights[band * self.bins() + bin]) as f64;
            }
            *out = sum as f32;
            if !out.is_finite() {
                return Err(MediaError::NonFinite);
            }
        }
        Ok(energies)
    }
    /// Exactly the source's standalone `mtmd_audio_log_mel_frame` convention:
    /// periodic Hann, unnormalized FFT power, double accumulation, log10 floor.
    pub fn log_mel_frame(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() != self.config.fft_len {
            return Err(MediaError::Invalid("mel frame length"));
        }
        let plan = FftPlan::new(self.config.fft_len)?;
        let window = hann_window(self.config.fft_len, true)?;
        let mut windowed = zeros(samples.len(), 0f32)?;
        for (out, (&sample, &w)) in windowed.iter_mut().zip(samples.iter().zip(&window)) {
            *out = sample * w;
        }
        let spectrum = plan.forward_real(&windowed)?;
        let mut power = zeros(self.bins(), 0f32)?;
        for (out, x) in power.iter_mut().zip(spectrum) {
            *out = x.norm_squared();
            if !out.is_finite() {
                return Err(MediaError::NonFinite);
            }
        }
        let mut out = zeros(self.config.bands, 0f32)?;
        for (band, value) in out.iter_mut().enumerate() {
            let mut sum = 0f64;
            for (bin, &p) in power.iter().enumerate() {
                sum += (p * self.weights[band * self.bins() + bin]) as f64;
            }
            *value = sum.max(1e-10).log10() as f32;
            if !value.is_finite() {
                return Err(MediaError::NonFinite);
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StftPadding {
    None,
    CenterZero,
    CenterReflect,
}

/// Full frame-major complex STFT. This is a primitive; model-specific long
/// padding, preemphasis, normalization and chunk policies are separate gates.
pub fn stft(
    samples: &[f32],
    fft_len: usize,
    hop: usize,
    padding: StftPadding,
) -> Result<Vec<Vec<Complex32>>> {
    let plan = FftPlan::new(fft_len)?;
    if hop == 0 || hop > fft_len {
        return Err(MediaError::Invalid("STFT hop"));
    }
    if samples.iter().any(|x| !x.is_finite()) {
        return Err(MediaError::NonFinite);
    }
    let pad = if padding == StftPadding::None {
        0
    } else {
        fft_len / 2
    };
    if padding == StftPadding::CenterReflect && samples.len() <= pad {
        return Err(MediaError::Invalid(
            "audio too short for reflection padding",
        ));
    }
    let padded_len = samples
        .len()
        .checked_add(product(pad, 2)?)
        .ok_or(MediaError::Overflow)?;
    if padded_len < fft_len {
        return Err(MediaError::Invalid("audio shorter than STFT window"));
    }
    let count = (padded_len - fft_len) / hop + 1;
    check_limit(
        "STFT output bytes",
        product(product(count, fft_len)?, std::mem::size_of::<Complex32>())?,
        crate::MAX_ALLOCATION_BYTES,
    )?;
    let mut padded = zeros(padded_len, 0f32)?;
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    if padding == StftPadding::CenterReflect {
        for i in 0..pad {
            padded[i] = samples[pad - i];
            padded[pad + samples.len() + i] = samples[samples.len() - 2 - i];
        }
    }
    let window = hann_window(fft_len, true)?;
    let mut frame = zeros(fft_len, 0f32)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| MediaError::Allocation)?;
    for start in (0..=padded.len() - fft_len).step_by(hop) {
        for i in 0..fft_len {
            frame[i] = padded[start + i] * window[i];
        }
        result.push(plan.forward_real(&frame)?);
    }
    Ok(result)
}

/// Frame-major log-mel projection over the STFT primitive. Model-specific
/// matrices may use a different layout; `to_mel_major` converts explicitly.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram {
    frames: usize,
    bands: usize,
    data: Vec<f32>,
}
impl Spectrogram {
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn bands(&self) -> usize {
        self.bands
    }
    pub fn data_frame_major(&self) -> &[f32] {
        &self.data
    }
    pub fn to_mel_major(&self) -> Result<Vec<f32>> {
        let mut out = zeros(self.data.len(), 0f32)?;
        for t in 0..self.frames {
            for m in 0..self.bands {
                out[m * self.frames + t] = self.data[t * self.bands + m];
            }
        }
        Ok(out)
    }
}
pub fn log_mel(
    samples: &[f32],
    bank: &MelFilterbank,
    hop: usize,
    padding: StftPadding,
    floor: f32,
) -> Result<Spectrogram> {
    if !floor.is_finite() || floor <= 0.0 {
        return Err(MediaError::Invalid("log-mel floor"));
    }
    let spectra = stft(samples, bank.config.fft_len, hop, padding)?;
    let mut data = zeros(product(spectra.len(), bank.config.bands)?, 0f32)?;
    for (frame, spectrum) in spectra.iter().enumerate() {
        let energies = bank.energies(spectrum, false)?;
        for (band, energy) in energies.into_iter().enumerate() {
            data[frame * bank.config.bands + band] = energy.max(floor).log10();
        }
    }
    Ok(Spectrogram {
        frames: spectra.len(),
        bands: bank.config.bands,
        data,
    })
}

/// Streaming inverse STFT with periodic Hann and the snapshot's initial
/// `(fft-hop)/2` trimming. A successful flush may be called again (returns empty).
/// This explicit terminal state corrects the reference's repeated-flush zeros.
#[derive(Debug, Clone)]
pub struct StreamingIstft {
    plan: FftPlan,
    hop: usize,
    window: Vec<f32>,
    overlap: Vec<f32>,
    window_sum: Vec<f32>,
    trim: usize,
    frames_seen: usize,
    finished: bool,
}
impl StreamingIstft {
    pub fn new(fft_len: usize, hop: usize) -> Result<Self> {
        if fft_len < 2 || !fft_len.is_multiple_of(2) || hop == 0 || hop > fft_len {
            return Err(MediaError::Invalid(
                "ISTFT needs even FFT and 1 <= hop <= FFT",
            ));
        }
        Ok(Self {
            plan: FftPlan::new(fft_len)?,
            hop,
            window: hann_window(fft_len, true)?,
            overlap: zeros(fft_len, 0f32)?,
            window_sum: zeros(fft_len, 0f32)?,
            trim: (fft_len - hop) / 2,
            frames_seen: 0,
            finished: false,
        })
    }
    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
        self.window_sum.fill(0.0);
        self.trim = (self.plan.len - self.hop) / 2;
        self.frames_seen = 0;
        self.finished = false;
    }
    pub fn process_frame(&mut self, positive_spectrum: &[Complex32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(MediaError::State("ISTFT finished; reset before reuse"));
        }
        let bins = self.plan.len / 2 + 1;
        if positive_spectrum.len() != bins {
            return Err(MediaError::Invalid("ISTFT spectrum bin count"));
        }
        let next_frames = self
            .frames_seen
            .checked_add(1)
            .ok_or(MediaError::Overflow)?;
        let mut full = zeros(self.plan.len, Complex32::default())?;
        full[..bins].copy_from_slice(positive_spectrum);
        for j in 1..bins - 1 {
            full[self.plan.len - j] = Complex32::new(full[j].re, -full[j].im);
        }
        // Compute and validate before mutating the overlap state.
        let frame = self.plan.inverse(&full)?;
        let mut out = zeros(self.hop, 0f32)?;
        for (index, value) in frame.iter().enumerate() {
            let new = self.overlap[index] + value.re * self.window[index];
            if !new.is_finite() {
                return Err(MediaError::NonFinite);
            }
            if index < self.hop {
                let weight = self.window_sum[index] + self.window[index] * self.window[index];
                out[index] = if weight > 1e-8 { new / weight } else { new };
                if !out[index].is_finite() {
                    return Err(MediaError::NonFinite);
                }
            }
        }
        for (index, value) in frame.iter().enumerate() {
            self.window_sum[index] += self.window[index] * self.window[index];
            self.overlap[index] += value.re * self.window[index];
        }
        self.shift(self.hop);
        let trim = self.trim.min(out.len());
        self.trim -= trim;
        out.drain(..trim);
        self.frames_seen = next_frames;
        Ok(out)
    }
    fn extract(&mut self, count: usize) -> Result<Vec<f32>> {
        let mut out = zeros(count, 0f32)?;
        for (i, value) in out.iter_mut().enumerate() {
            *value = if self.window_sum[i] > 1e-8 {
                self.overlap[i] / self.window_sum[i]
            } else {
                self.overlap[i]
            };
            if !value.is_finite() {
                return Err(MediaError::NonFinite);
            }
        }
        self.shift(count);
        Ok(out)
    }
    fn shift(&mut self, count: usize) {
        self.overlap.copy_within(count.., 0);
        self.window_sum.copy_within(count.., 0);
        let length = self.plan.len;
        self.overlap[length - count..].fill(0.0);
        self.window_sum[length - count..].fill(0.0);
    }
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let out = if self.frames_seen == 0 {
            Vec::new()
        } else {
            self.extract(self.plan.len - self.hop)?
        };
        self.finished = true;
        Ok(out)
    }
}
