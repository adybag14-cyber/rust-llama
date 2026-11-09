//! Owned, typed integration seams until T01/T02/T05 contracts are supplied.
//!
//! These are deliberately crate-local Rust contracts, not a frozen C ABI and
//! not serializations of native Rust layout.
use crate::{MediaError, Result, audio::AudioBuffer, image::RgbImage, product};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimestampMicros(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub enum InputChunk {
    Text {
        sequence: SequenceId,
        text: String,
    },
    Image {
        sequence: SequenceId,
        image: Arc<RgbImage>,
    },
    Audio {
        sequence: SequenceId,
        start: TimestampMicros,
        audio: Arc<AudioBuffer>,
    },
    VideoFrame {
        sequence: SequenceId,
        timestamp: TimestampMicros,
        image: Arc<RgbImage>,
    },
}
impl InputChunk {
    pub fn sequence(&self) -> SequenceId {
        match self {
            Self::Text { sequence, .. }
            | Self::Image { sequence, .. }
            | Self::Audio { sequence, .. }
            | Self::VideoFrame { sequence, .. } => *sequence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub token: u64,
    pub temporal: u32,
    pub y: u32,
    pub x: u32,
}

/// A validated container for **externally computed** embeddings. Its existence
/// does not imply any model builder/projector has been implemented.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingBatch {
    sequence: SequenceId,
    width: usize,
    positions: Vec<Position>,
    embeddings: Vec<f32>,
}
impl EmbeddingBatch {
    pub fn new(
        sequence: SequenceId,
        width: usize,
        positions: Vec<Position>,
        embeddings: Vec<f32>,
    ) -> Result<Self> {
        if width == 0 || product(width, positions.len())? != embeddings.len() {
            return Err(MediaError::Invalid("embedding shape mismatch"));
        }
        if embeddings.iter().any(|n| !n.is_finite()) {
            return Err(MediaError::NonFinite);
        }
        Ok(Self {
            sequence,
            width,
            positions,
            embeddings,
        })
    }
    pub fn sequence(&self) -> SequenceId {
        self.sequence
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }
}

/// Per-operation cancellation: no process-global flag or unrelated sequence reset.
#[derive(Debug, Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(MediaError::State("cancelled"))
        } else {
            Ok(())
        }
    }
}

/// Timestamp selection over frames already decoded by a caller. Container
/// probing and video codecs remain pending; this never invokes FFmpeg.
pub fn select_video_frames(
    timestamps: &[TimestampMicros],
    interval_micros: u64,
    max_frames: usize,
) -> Result<Vec<usize>> {
    if interval_micros == 0 {
        return Err(MediaError::Invalid("zero frame interval"));
    }
    if timestamps.windows(2).any(|w| w[1] < w[0]) {
        return Err(MediaError::Invalid("nonmonotonic frame timestamps"));
    }
    let mut out = Vec::new();
    let mut next = None;
    for (index, timestamp) in timestamps.iter().enumerate() {
        if out.len() >= max_frames {
            break;
        }
        if next.is_none_or(|t| timestamp.0 >= t) {
            out.push(index);
            next = Some(
                timestamp
                    .0
                    .checked_add(interval_micros)
                    .ok_or(MediaError::Overflow)?,
            );
        }
    }
    Ok(out)
}

/// Frozen fixed-count sampling from `mtmd_helper_video_seq::configure`:
/// `round_ties_even(linspace(0, source_frames - 1, target_frames))`.
/// Repeated indices are intentional when more frames than the source are asked
/// for. Decoding, soundtrack alignment and model timestamp formatting are open.
pub fn uniform_frame_indices(source_frames: usize, target_frames: usize) -> Result<Vec<usize>> {
    if source_frames == 0 || target_frames == 0 {
        return Err(MediaError::Invalid("zero video frame count"));
    }
    crate::check_limit("source video frames", source_frames, i32::MAX as usize)?;
    crate::check_limit("selected video frames", target_frames, 65_536)?;
    let step = if target_frames > 1 {
        (source_frames - 1) as f64 / (target_frames - 1) as f64
    } else {
        0.0
    };
    let mut result = crate::zeros(target_frames, 0usize)?;
    for (i, value) in result.iter_mut().enumerate() {
        *value = (i as f64 * step).round_ties_even() as usize;
    }
    Ok(result)
}
