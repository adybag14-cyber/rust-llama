//! Bounded RIFF/WAVE IO and explicit PCM utilities.
//!
//! The reference helper uses miniaudio. Implemented WAV sample conversion is
//! validated separately from resampling: the linear resampler here is useful
//! Rust functionality, **not** a claim of miniaudio's low-pass filter parity.
use crate::{DecodeLimits, MediaError, Result, bytes, check_limit, product, u16le, u32le, zeros};

#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}
impl AudioBuffer {
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<Self> {
        if sample_rate == 0 || sample_rate > 768_000 {
            return Err(MediaError::Invalid("sample rate outside 1..=768000"));
        }
        if channels == 0 || channels > 32 {
            return Err(MediaError::Invalid("channel count outside 1..=32"));
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(MediaError::Invalid("incomplete audio frame"));
        }
        if samples.iter().any(|n| !n.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        Ok(Self {
            sample_rate,
            channels,
            samples,
        })
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.channels
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
    /// Stereo is the reference's equal-weight mix; multichannel speaker maps
    /// require a separately verified matrix and are rejected in this checkpoint.
    pub fn mono(&self) -> Result<Self> {
        if self.channels == 1 {
            return Ok(self.clone());
        }
        if self.channels != 2 {
            return Err(MediaError::Unsupported(
                "multichannel speaker-matrix downmix",
            ));
        }
        let mut out = zeros(self.frames(), 0f32)?;
        for (dest, frame) in out.iter_mut().zip(self.samples.as_chunks::<2>().0) {
            // Scale before adding to avoid overflow for large finite float WAV.
            *dest = frame[0] * 0.5 + frame[1] * 0.5;
        }
        Self::new(self.sample_rate, 1, out)
    }
}

#[derive(Debug, Clone, Copy)]
struct WaveFormat {
    kind: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    block_align: usize,
}
fn format(chunk: &[u8]) -> Result<WaveFormat> {
    bytes(chunk, 0, 16)?;
    let mut kind = u16le(chunk, 0)?;
    let channels = u16le(chunk, 2)?;
    let sample_rate = u32le(chunk, 4)?;
    let byte_rate = u32le(chunk, 8)?;
    let block_align = u16le(chunk, 12)? as usize;
    let bits = u16le(chunk, 14)?;
    if kind == 0xfffe {
        bytes(chunk, 0, 40)?;
        if u16le(chunk, 16)? < 22 {
            return Err(MediaError::Invalid("short extensible WAV format"));
        }
        let valid_bits = u16le(chunk, 18)?;
        if valid_bits != 0 && valid_bits != bits {
            return Err(MediaError::Unsupported("WAV padded valid bits"));
        }
        // KSDATAFORMAT_SUBTYPE_{PCM,IEEE_FLOAT}: little-endian GUID data1.
        const GUID_TAIL: [u8; 14] = [0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71];
        if bytes(chunk, 26, 14)? != GUID_TAIL {
            return Err(MediaError::Unsupported("WAV extensible subformat GUID"));
        }
        kind = u16le(chunk, 24)?;
    }
    if kind != 1 && kind != 3 {
        return Err(MediaError::Unsupported("compressed WAV format"));
    }
    if channels == 0 || channels > 32 || sample_rate == 0 || sample_rate > 768_000 {
        return Err(MediaError::Invalid("WAV rate or channels"));
    }
    if (kind == 1 && ![8, 16, 24, 32].contains(&bits)) || (kind == 3 && ![32, 64].contains(&bits)) {
        return Err(MediaError::Unsupported("WAV sample depth"));
    }
    if block_align != product(channels as usize, bits as usize / 8)? {
        return Err(MediaError::Invalid("WAV block alignment"));
    }
    if byte_rate as u64 != sample_rate as u64 * block_align as u64 {
        return Err(MediaError::Invalid("WAV byte rate"));
    }
    Ok(WaveFormat {
        kind,
        channels,
        sample_rate,
        bits,
        block_align,
    })
}

/// Decode RIFF little-endian PCM8/16/24/32 and IEEE f32/f64 WAV, including the
/// corresponding extensible subformats. Data stays interleaved at source rate.
pub fn decode_wav(input: &[u8], limits: DecodeLimits) -> Result<AudioBuffer> {
    check_limit("encoded bytes", input.len(), limits.max_input_bytes)?;
    if bytes(input, 0, 4)? != b"RIFF" || bytes(input, 8, 4)? != b"WAVE" {
        return Err(MediaError::Unsupported(
            "audio container (implemented: RIFF/WAVE)",
        ));
    }
    let end = (u32le(input, 4)? as usize)
        .checked_add(8)
        .ok_or(MediaError::Overflow)?;
    if end < 12 {
        return Err(MediaError::Invalid("short RIFF form"));
    }
    let input = bytes(input, 0, end)?;
    let mut fmt = None;
    let mut data = None;
    let mut cursor = 12;
    while cursor < end {
        let id = bytes(input, cursor, 4)?;
        let length = u32le(input, cursor + 4)? as usize;
        let chunk = bytes(input, cursor + 8, length)?;
        if id == b"fmt " {
            if fmt.is_some() {
                return Err(MediaError::Invalid("duplicate WAV format chunk"));
            }
            fmt = Some(format(chunk)?);
        } else if id == b"data" {
            if data.is_some() {
                return Err(MediaError::Unsupported("multiple WAV data chunks"));
            }
            data = Some(chunk);
        }
        cursor = cursor
            .checked_add(8)
            .and_then(|n| n.checked_add(length))
            .and_then(|n| n.checked_add(length & 1))
            .ok_or(MediaError::Overflow)?;
        if cursor > end {
            return Err(MediaError::Truncated {
                offset: end,
                required: 1,
            });
        }
    }
    let fmt = fmt.ok_or(MediaError::Invalid("missing WAV format chunk"))?;
    let data = data.ok_or(MediaError::Invalid("missing WAV data chunk"))?;
    if !data.len().is_multiple_of(fmt.block_align) {
        return Err(MediaError::Invalid("partial WAV frame"));
    }
    let sample_bytes = fmt.bits as usize / 8;
    let count = data.len() / sample_bytes;
    check_limit("audio samples", count, limits.max_audio_samples)?;
    let mut samples = zeros(count, 0f32)?;
    for (out, sample) in samples.iter_mut().zip(data.chunks_exact(sample_bytes)) {
        *out = match (fmt.kind, fmt.bits) {
            (1, 8) => (sample[0] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0,
            (1, 24) => {
                let integer =
                    ((sample[0] as i32) | ((sample[1] as i32) << 8) | ((sample[2] as i32) << 16))
                        << 8
                        >> 8;
                integer as f32 / 8_388_608.0
            }
            (1, 32) => {
                i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]) as f32
                    / 2_147_483_648.0
            }
            (3, 32) => f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
            (3, 64) => f64::from_le_bytes([
                sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6],
                sample[7],
            ]) as f32,
            _ => return Err(MediaError::Unsupported("WAV sample representation")),
        };
        if !out.is_finite() {
            return Err(MediaError::NonFinite);
        }
    }
    AudioBuffer::new(fmt.sample_rate, fmt.channels, samples)
}

/// Write finite IEEE float32 WAV, including the non-PCM `fact` chunk.
pub fn encode_wav_f32(audio: &AudioBuffer) -> Result<Vec<u8>> {
    let payload = product(audio.samples.len(), 4)?;
    let file_size = payload.checked_add(56).ok_or(MediaError::Overflow)?;
    let riff_size = u32::try_from(file_size - 8).map_err(|_| MediaError::Overflow)?;
    let data_size = u32::try_from(payload).map_err(|_| MediaError::Overflow)?;
    let frames = u32::try_from(audio.frames()).map_err(|_| MediaError::Overflow)?;
    let mut result = zeros(file_size, 0u8)?;
    result[0..4].copy_from_slice(b"RIFF");
    result[4..8].copy_from_slice(&riff_size.to_le_bytes());
    result[8..12].copy_from_slice(b"WAVE");
    result[12..16].copy_from_slice(b"fmt ");
    result[16..20].copy_from_slice(&16u32.to_le_bytes());
    result[20..22].copy_from_slice(&3u16.to_le_bytes());
    result[22..24].copy_from_slice(&audio.channels.to_le_bytes());
    result[24..28].copy_from_slice(&audio.sample_rate.to_le_bytes());
    let align = audio.channels * 4;
    let rate = audio
        .sample_rate
        .checked_mul(align as u32)
        .ok_or(MediaError::Overflow)?;
    result[28..32].copy_from_slice(&rate.to_le_bytes());
    result[32..34].copy_from_slice(&align.to_le_bytes());
    result[34..36].copy_from_slice(&32u16.to_le_bytes());
    result[36..40].copy_from_slice(b"fact");
    result[40..44].copy_from_slice(&4u32.to_le_bytes());
    result[44..48].copy_from_slice(&frames.to_le_bytes());
    result[48..52].copy_from_slice(b"data");
    result[52..56].copy_from_slice(&data_size.to_le_bytes());
    for (dest, sample) in result[56..]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(&audio.samples)
    {
        dest.copy_from_slice(&sample.to_le_bytes());
    }
    Ok(result)
}

/// A deterministic scalar linear resampler with rational phase. No low-pass
/// antialias filter is applied, so this must not be selected as a transparent
/// replacement for the reference miniaudio resampler.
pub fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>> {
    let mut stream = LinearResampler::new(input_rate, output_rate)?;
    let mut out = stream.push(samples)?;
    let tail = stream.finish()?;
    check_limit(
        "resampled samples",
        out.len()
            .checked_add(tail.len())
            .ok_or(MediaError::Overflow)?,
        crate::MAX_ALLOCATION_BYTES / 4,
    )?;
    out.try_reserve_exact(tail.len())
        .map_err(|_| MediaError::Allocation)?;
    out.extend(tail);
    Ok(out)
}

/// Bounded stateful resampling; output is identical for any input chunking.
/// `finish` clamps to the final sample and is idempotent. Reset is explicit.
#[derive(Debug, Clone)]
pub struct LinearResampler {
    input_rate: u32,
    output_rate: u32,
    samples: Vec<f32>,
    base: u64,
    total: u64,
    next: u64,
    finished: bool,
}
impl LinearResampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Result<Self> {
        if input_rate == 0 || output_rate == 0 || input_rate > 768_000 || output_rate > 768_000 {
            return Err(MediaError::Invalid("resampler rate"));
        }
        Ok(Self {
            input_rate,
            output_rate,
            samples: Vec::new(),
            base: 0,
            total: 0,
            next: 0,
            finished: false,
        })
    }
    pub fn reset(&mut self) {
        self.samples.clear();
        self.base = 0;
        self.total = 0;
        self.next = 0;
        self.finished = false;
    }
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(MediaError::State("resampler finished; reset before reuse"));
        }
        if input.iter().any(|s| !s.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        let next_total = self
            .total
            .checked_add(input.len() as u64)
            .ok_or(MediaError::Overflow)?;
        // Allocate all fallible output storage before accepting the new input.
        let (end, out) = self.allocate_output(next_total, false)?;
        check_limit(
            "resampler retained samples",
            self.samples
                .len()
                .checked_add(input.len())
                .ok_or(MediaError::Overflow)?,
            crate::MAX_ALLOCATION_BYTES / 4,
        )?;
        self.samples
            .try_reserve_exact(input.len())
            .map_err(|_| MediaError::Allocation)?;
        self.samples.extend_from_slice(input);
        self.total = next_total;
        Ok(self.emit_preallocated(end, out))
    }
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let (end, out) = self.allocate_output(self.total, true)?;
        let out = self.emit_preallocated(end, out);
        self.finished = true;
        self.samples.clear();
        Ok(out)
    }
    fn allocate_output(&self, total: u64, final_input: bool) -> Result<(u64, Vec<f32>)> {
        let end = if total == 0 {
            0
        } else if final_input {
            (total as u128 * self.output_rate as u128).div_ceil(self.input_rate as u128)
        } else {
            // Non-final input can emit up to and including the last exact sample.
            (total as u128 - 1) * self.output_rate as u128 / self.input_rate as u128 + 1
        };
        let end = u64::try_from(end)
            .map_err(|_| MediaError::Overflow)?
            .max(self.next);
        let count = usize::try_from(end - self.next).map_err(|_| MediaError::Overflow)?;
        Ok((end, zeros(count, 0f32)?))
    }
    fn emit_preallocated(&mut self, end: u64, mut out: Vec<f32>) -> Vec<f32> {
        for (ordinal, value) in out.iter_mut().enumerate() {
            let numerator = (self.next + ordinal as u64) as u128 * self.input_rate as u128;
            let index = numerator / self.output_rate as u128;
            let remainder = numerator % self.output_rate as u128;
            // allocate_output bounds every phase to the retained input range.
            let relative = (index as u64 - self.base) as usize;
            let first = self.samples[relative];
            let second = self.samples.get(relative + 1).copied().unwrap_or(first);
            let fraction = remainder as f64 / self.output_rate as f64;
            // f64 intermediate avoids overflow for opposite extreme f32 samples.
            *value = (first as f64 * (1.0 - fraction) + second as f64 * fraction) as f32;
        }
        self.next = end;
        if self.total == 0 {
            return out;
        }
        let next_source = (self.next as u128 * self.input_rate as u128 / self.output_rate as u128)
            .min(self.total as u128 - 1) as u64;
        let discard = next_source.saturating_sub(self.base) as usize;
        if discard > 0 {
            self.samples.drain(..discard);
            self.base = next_source;
        }
        out
    }
}

/// Source-present raw-waveform path used by the PocketTTS/Muse families: align
/// to complete frames with zeros. This does not execute an acoustic model.
pub fn pad_waveform(
    samples: &[f32],
    frame_samples: usize,
    max_samples: Option<usize>,
) -> Result<Vec<f32>> {
    if frame_samples == 0 {
        return Err(MediaError::Invalid("zero waveform frame size"));
    }
    if samples.iter().any(|x| !x.is_finite()) {
        return Err(MediaError::NonFinite);
    }
    let count = samples.len().min(max_samples.unwrap_or(samples.len()));
    let frames = count.div_ceil(frame_samples);
    let mut out = zeros(product(frames, frame_samples)?, 0f32)?;
    out[..count].copy_from_slice(&samples[..count]);
    Ok(out)
}
