//! Frozen dense F32 KV memory semantics, separate from the strict Context API.
//!
//! Physical cache slots and pending copy/shift work are deliberately retained:
//! native state bytes describe stored data and position metadata, and do not
//! serialize pending updates. This module does not claim recurrent cache support.
use super::{BatchToken, Input, OutputSelection, TokenOutput};
use crate::{AdapterSet, Error, Model, Result, llama::KvLayer, math};
use std::sync::Arc;

#[path = "native_state.rs"]
mod native_state;
pub use native_state::NativeState;

/// Dense F32 compatibility cache configuration.
#[derive(Debug, Clone)]
pub struct NativeContextOptions {
    pub context_tokens: usize,
    pub maximum_sequences: usize,
    pub microbatch_tokens: usize,
    pub maximum_batch_tokens: usize,
    pub unified: bool,
}

/// A native batch item can share one physical cache cell between sequences.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeBatchToken {
    pub sequences: Vec<i32>,
    pub position: i32,
    pub input: Input,
    pub output: OutputSelection,
}
impl From<BatchToken> for NativeBatchToken {
    fn from(value: BatchToken) -> Self {
        Self {
            sequences: vec![value.sequence],
            position: i32::try_from(value.position).unwrap_or(-1),
            input: value.input,
            output: value.output,
        }
    }
}

const MAX_NATIVE_WORK_BYTES: usize = 2 * 1024 * 1024 * 1024;
fn copy_values<T: Copy>(source: &[T]) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source.len())
        .map_err(|_| invalid("native allocation failed"))?;
    result.extend_from_slice(source);
    Ok(result)
}
fn copy_layers(source: &[KvLayer]) -> Result<Vec<KvLayer>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source.len())
        .map_err(|_| invalid("native layer allocation failed"))?;
    for layer in source {
        result.push(KvLayer {
            keys: copy_values(&layer.keys)?,
            values: copy_values(&layer.values)?,
        });
    }
    Ok(result)
}
const SEQUENCE_IDS: [i32; 256] = {
    let mut ids = [0; 256];
    let mut i = 0;
    while i < 256 {
        ids[i] = i as i32;
        i += 1;
    }
    ids
};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SequenceSet([u64; 4]);
struct SequenceIter {
    bits: [u64; 4],
    word: usize,
}
impl Iterator for SequenceIter {
    type Item = &'static i32;
    fn next(&mut self) -> Option<Self::Item> {
        while self.word < 4 {
            let word = &mut self.bits[self.word];
            if *word != 0 {
                let bit = word.trailing_zeros() as usize;
                *word &= *word - 1;
                return Some(&SEQUENCE_IDS[self.word * 64 + bit]);
            }
            self.word += 1;
        }
        None
    }
}
impl SequenceSet {
    fn new() -> Self {
        Self::default()
    }
    fn contains(&self, id: &i32) -> bool {
        (0..256).contains(id) && self.0[*id as usize / 64] & (1u64 << (*id as usize % 64)) != 0
    }
    fn insert(&mut self, id: i32) -> bool {
        if !(0..256).contains(&id) {
            return false;
        }
        let had = self.contains(&id);
        self.0[id as usize / 64] |= 1u64 << (id as usize % 64);
        !had
    }
    fn remove(&mut self, id: &i32) -> bool {
        if !(0..256).contains(id) {
            return false;
        }
        let had = self.contains(id);
        self.0[*id as usize / 64] &= !(1u64 << (*id as usize % 64));
        had
    }
    fn clear(&mut self) {
        self.0 = [0; 4];
    }
    fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }
    fn len(&self) -> usize {
        self.0.iter().map(|word| word.count_ones() as usize).sum()
    }
    fn iter(&self) -> SequenceIter {
        SequenceIter {
            bits: self.0,
            word: 0,
        }
    }
}
impl FromIterator<i32> for SequenceSet {
    fn from_iter<T: IntoIterator<Item = i32>>(iter: T) -> Self {
        let mut result = Self::new();
        for id in iter {
            result.insert(id);
        }
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Cell {
    position: i32,
    shift: i32,
    sequences: SequenceSet,
}
impl Default for Cell {
    fn default() -> Self {
        Self {
            position: -1,
            shift: 0,
            sequences: SequenceSet::new(),
        }
    }
}
impl Cell {
    fn occupied(&self) -> bool {
        self.position >= 0
    }
    fn in_range(&self, p0: i32, p1: i32) -> bool {
        self.position >= p0 && self.position < p1
    }
}
#[derive(Debug, Clone)]
struct Stream {
    cells: Vec<Cell>,
    layers: Vec<KvLayer>,
    head: usize,
}

/// Scalar implementation of the frozen Llama F32 memory contract.
///
/// The strict Context type remains the safe sequential API. This compatibility
/// executor owns physical cells, can retain holes and duplicate positions after
/// position transforms, and serializes the source's unversioned memory format.
#[derive(Debug, Clone)]
pub struct NativeContext {
    model: Arc<Model>,
    options: NativeContextOptions,
    adapters: AdapterSet,
    streams: Vec<Stream>,
    pending_copies: Vec<(usize, usize)>,
    capacity: usize,
    width: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidState(message.into())
}
fn range(p0: i32, p1: i32) -> (i32, i32) {
    (p0.max(0), if p1 < 0 { i32::MAX } else { p1 })
}
fn zeroes(count: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("cache allocation failed"))?;
    values.resize(count, 0.0);
    Ok(values)
}
impl NativeContext {
    pub fn new(model: Arc<Model>, options: NativeContextOptions) -> Result<Self> {
        if options.maximum_sequences == 0
            || options.maximum_sequences > 256
            || options.context_tokens == 0
            || options.context_tokens > i32::MAX as usize
            || options.microbatch_tokens == 0
            || options.maximum_batch_tokens == 0
            || options.maximum_batch_tokens > 65_536
            || options.microbatch_tokens > options.maximum_batch_tokens
        {
            return Err(Error::InvalidConfig("invalid native context limits".into()));
        }
        let Model::Llama(llama) = model.as_ref() else {
            return Err(Error::Unsupported(
                "native recurrent memory and state layout".into(),
            ));
        };
        let rounded = options
            .context_tokens
            .checked_add(255)
            .map(|n| n / 256 * 256)
            .ok_or_else(|| invalid("context size overflow"))?;
        let capacity = if options.unified {
            rounded
        } else {
            (rounded / options.maximum_sequences)
                .checked_add(255)
                .map(|n| n / 256 * 256)
                .ok_or_else(|| invalid("sequence capacity overflow"))?
        };
        if capacity == 0 {
            return Err(invalid("zero sequence capacity"));
        }
        let streams = if options.unified {
            1
        } else {
            options.maximum_sequences
        };
        let width = llama.config().kv_width()?;
        let elements = capacity
            .checked_mul(width)
            .ok_or_else(|| invalid("cache shape overflow"))?;
        let bytes = elements
            .checked_mul(llama.layer_count())
            .and_then(|n| n.checked_mul(streams))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("cache bytes overflow"))?;
        // A transactional copy is needed for fallible decode/state input. Bound
        // aggregate cache allocation before constructing any individual vector.
        if bytes > 1024 * 1024 * 1024 || capacity.checked_mul(streams).is_none_or(|n| n > 1_000_000)
        {
            return Err(Error::Unsupported(
                "native validation executor cache exceeds one GiB or one million cells".into(),
            ));
        }
        let metadata_bytes = capacity
            .checked_mul(streams)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Cell>()))
            .ok_or_else(|| invalid("native cell accounting overflow"))?;
        if bytes
            .checked_add(metadata_bytes)
            .and_then(|n| n.checked_mul(2))
            .is_none_or(|n| n > MAX_NATIVE_WORK_BYTES)
        {
            return Err(Error::Unsupported(
                "native transactional cache exceeds the two-GiB owned working-set budget".into(),
            ));
        }
        let mut all = Vec::new();
        all.try_reserve_exact(streams)
            .map_err(|_| invalid("stream allocation failed"))?;
        for _ in 0..streams {
            let mut layers = Vec::new();
            layers
                .try_reserve_exact(llama.layer_count())
                .map_err(|_| invalid("layer allocation failed"))?;
            for _ in 0..llama.layer_count() {
                layers.push(KvLayer {
                    keys: zeroes(elements)?,
                    values: zeroes(elements)?,
                });
            }
            let mut cells = Vec::new();
            cells
                .try_reserve_exact(capacity)
                .map_err(|_| invalid("cell allocation failed"))?;
            cells.resize_with(capacity, Cell::default);
            all.push(Stream {
                cells,
                layers,
                head: 0,
            });
        }
        Ok(Self {
            model,
            options,
            adapters: AdapterSet::default(),
            streams: all,
            pending_copies: Vec::new(),
            capacity,
            width,
        })
    }
    fn storage_bytes(&self) -> Result<usize> {
        let mut bytes = self
            .pending_copies
            .capacity()
            .checked_mul(std::mem::size_of::<(usize, usize)>())
            .ok_or_else(|| invalid("native memory accounting overflow"))?;
        for stream in &self.streams {
            bytes = bytes
                .checked_add(stream.cells.capacity() * std::mem::size_of::<Cell>())
                .and_then(|n| {
                    n.checked_add(stream.layers.capacity() * std::mem::size_of::<KvLayer>())
                })
                .ok_or_else(|| invalid("native metadata accounting overflow"))?;
            for layer in &stream.layers {
                bytes = bytes
                    .checked_add(
                        layer
                            .keys
                            .capacity()
                            .checked_mul(4)
                            .ok_or_else(|| invalid("key allocation overflow"))?,
                    )
                    .and_then(|n| n.checked_add(layer.values.capacity().checked_mul(4)?))
                    .ok_or_else(|| invalid("native tensor accounting overflow"))?;
            }
        }
        Ok(bytes)
    }
    fn check_work(&self, extra: usize) -> Result<()> {
        self.check_work_limit(extra, MAX_NATIVE_WORK_BYTES)
    }
    fn check_work_limit(&self, extra: usize, limit: usize) -> Result<()> {
        if self
            .storage_bytes()?
            .checked_add(extra)
            .is_none_or(|bytes| bytes > limit.min(MAX_NATIVE_WORK_BYTES))
        {
            return Err(Error::Unsupported(
                "native operation exceeds the two-GiB owned working-set budget".into(),
            ));
        }
        Ok(())
    }
    fn try_clone(&self) -> Result<Self> {
        self.check_work(self.storage_bytes()?)?;
        let mut streams = Vec::new();
        streams
            .try_reserve_exact(self.streams.len())
            .map_err(|_| invalid("native stream allocation failed"))?;
        for stream in &self.streams {
            streams.push(Stream {
                cells: copy_values(&stream.cells)?,
                layers: copy_layers(&stream.layers)?,
                head: stream.head,
            });
        }
        Ok(Self {
            model: self.model.clone(),
            options: self.options.clone(),
            adapters: self.adapters.clone(),
            streams,
            pending_copies: copy_values(&self.pending_copies)?,
            capacity: self.capacity,
            width: self.width,
        })
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub fn context_tokens(&self) -> usize {
        self.capacity * self.streams.len()
    }
    pub fn context_tokens_per_sequence(&self) -> usize {
        self.capacity
    }
    pub fn is_unified(&self) -> bool {
        self.options.unified
    }
    pub fn sequence_id_limit(&self) -> usize {
        if self.options.unified {
            256
        } else {
            self.options.maximum_sequences
        }
    }
    fn stream_id(&self, sequence: i32) -> Result<usize> {
        if sequence < 0 || sequence as usize >= self.sequence_id_limit() {
            return Err(Error::InvalidSequence(sequence));
        }
        Ok(if self.options.unified {
            0
        } else {
            sequence as usize
        })
    }
    pub fn position_min(&self, sequence: i32) -> Result<i32> {
        Ok(self.streams[self.stream_id(sequence)?]
            .cells
            .iter()
            .filter(|cell| cell.sequences.contains(&sequence))
            .map(|cell| cell.position)
            .min()
            .unwrap_or(-1))
    }
    pub fn position_max(&self, sequence: i32) -> Result<i32> {
        Ok(self.streams[self.stream_id(sequence)?]
            .cells
            .iter()
            .filter(|cell| cell.sequences.contains(&sequence))
            .map(|cell| cell.position)
            .max()
            .unwrap_or(-1))
    }
    pub fn next_position(&self, sequence: i32) -> Result<i32> {
        self.position_max(sequence)?
            .checked_add(1)
            .ok_or_else(|| invalid("next position overflow"))
    }
    pub fn can_shift(&self) -> bool {
        true
    }
    /// Reset metadata; clearing data is distinct because pending copies and
    /// later full-buffer copies can observe stale storage.
    pub fn clear(&mut self, data: bool) {
        for stream in &mut self.streams {
            stream.cells.fill_with(Cell::default);
            stream.head = 0;
            if data {
                for layer in &mut stream.layers {
                    layer.keys.fill(0.0);
                    layer.values.fill(0.0);
                }
            }
        }
        // The original clear does not erase the queued stream-copy list.
    }
    pub fn remove_range(&mut self, sequence: i32, p0: i32, p1: i32) -> Result<bool> {
        let (p0, p1) = range(p0, p1);
        let selected = if sequence == -1 {
            None
        } else {
            Some(self.stream_id(sequence)?)
        };
        if sequence < -1 {
            return Err(Error::InvalidSequence(sequence));
        }
        for (index, stream) in self.streams.iter_mut().enumerate() {
            if selected.is_some_and(|s| s != index) {
                continue;
            }
            let mut first = None;
            for (i, cell) in stream.cells.iter_mut().enumerate() {
                if !cell.in_range(p0, p1) {
                    continue;
                }
                if sequence == -1 || (cell.sequences.remove(&sequence) && cell.sequences.is_empty())
                {
                    *cell = Cell::default();
                    first.get_or_insert(i);
                }
            }
            if let Some(first) = first {
                stream.head = stream.head.min(first);
            }
        }
        Ok(true)
    }
    pub fn copy_range(&mut self, source: i32, destination: i32, p0: i32, p1: i32) -> Result<()> {
        let src = self.stream_id(source)?;
        let dst = self.stream_id(destination)?;
        if src == dst {
            if source == destination {
                return Ok(());
            }
            let (p0, p1) = range(p0, p1);
            for cell in &mut self.streams[src].cells {
                if cell.in_range(p0, p1) && cell.sequences.contains(&source) {
                    cell.sequences.insert(destination);
                }
            }
            return Ok(());
        }
        let partial = |p: i32| p > 0 && i64::from(p) + 1 < self.capacity as i64;
        if partial(p0) || partial(p1) {
            return Err(Error::Unsupported(
                "cross-stream copy requires the frozen full-buffer range".into(),
            ));
        }
        if self.pending_copies.len() >= 1_000_000 {
            return Err(invalid("pending copy limit"));
        }
        self.check_work(self.capacity * std::mem::size_of::<Cell>())?;
        let mut copied = copy_values(&self.streams[src].cells)?;
        self.pending_copies
            .try_reserve_exact(1)
            .map_err(|_| invalid("pending copy allocation failed"))?;
        for cell in &mut copied {
            if cell.sequences.contains(&source) {
                cell.sequences.clear();
                cell.sequences.insert(destination);
            } else {
                *cell = Cell::default();
            }
        }
        self.streams[dst].cells = copied;
        self.streams[dst].head = self.streams[src].head;
        self.pending_copies.push((src, dst));
        Ok(())
    }
    pub fn keep_sequence(&mut self, sequence: i32) -> Result<()> {
        let index = self.stream_id(sequence)?;
        let stream = &mut self.streams[index];
        let mut first = None;
        for (i, cell) in stream.cells.iter_mut().enumerate() {
            if cell.sequences.contains(&sequence) {
                cell.sequences.clear();
                cell.sequences.insert(sequence);
            } else if !cell.sequences.is_empty() {
                *cell = Cell::default();
                first.get_or_insert(i);
            }
        }
        if let Some(first) = first {
            stream.head = stream.head.min(first);
        }
        Ok(())
    }
    pub fn shift_range(&mut self, sequence: i32, p0: i32, p1: i32, delta: i32) -> Result<()> {
        let s = self.stream_id(sequence)?;
        if delta == 0 {
            return Ok(());
        }
        let (p0, p1) = range(p0, p1);
        if p0 == p1 {
            return Ok(());
        }
        // Check every arithmetic operation before any metadata mutation.
        for cell in &self.streams[s].cells {
            if cell.in_range(p0, p1) && cell.sequences.contains(&sequence) {
                cell.position
                    .checked_add(delta)
                    .ok_or_else(|| invalid("position shift overflow"))?;
                cell.shift
                    .checked_add(delta)
                    .ok_or_else(|| invalid("pending shift overflow"))?;
            }
        }
        let stream = &mut self.streams[s];
        let mut first = None;
        for (i, cell) in stream.cells.iter_mut().enumerate() {
            if cell.in_range(p0, p1) && cell.sequences.contains(&sequence) {
                cell.position += delta;
                cell.shift += delta;
                if cell.position < 0 {
                    *cell = Cell::default();
                    first.get_or_insert(i);
                }
            }
        }
        stream.head = first.unwrap_or(0);
        Ok(())
    }
    pub fn divide_range(&mut self, sequence: i32, p0: i32, p1: i32, divisor: i32) -> Result<()> {
        let s = self.stream_id(sequence)?;
        if divisor <= 0 {
            return Err(invalid("position divisor must be positive"));
        }
        if divisor == 1 {
            return Ok(());
        }
        let (p0, p1) = range(p0, p1);
        for cell in &self.streams[s].cells {
            if cell.in_range(p0, p1) && cell.sequences.contains(&sequence) {
                cell.shift
                    .checked_add(cell.position - cell.position / divisor)
                    .ok_or_else(|| invalid("pending division shift overflow"))?;
            }
        }
        for cell in &mut self.streams[s].cells {
            if cell.in_range(p0, p1) && cell.sequences.contains(&sequence) {
                let old = cell.position;
                cell.position /= divisor;
                // Frozen source intentionally accumulates old - divided, unlike
                // seq_add's signed delta. Preserve the actual source algorithm.
                cell.shift += old - cell.position;
            }
        }
        Ok(())
    }

    fn apply_pending(&mut self) -> Result<()> {
        for &(source, destination) in &self.pending_copies {
            if source == destination {
                continue;
            }
            // Distinct streams already own equally shaped buffers. Copy in
            // place, preserving queue order without a third complete KV copy.
            let (source, destination) = if source < destination {
                let (left, right) = self.streams.split_at_mut(destination);
                (&left[source], &mut right[0])
            } else {
                let (left, right) = self.streams.split_at_mut(source);
                (&right[0], &mut left[destination])
            };
            for (from, to) in source.layers.iter().zip(&mut destination.layers) {
                to.keys.copy_from_slice(&from.keys);
                to.values.copy_from_slice(&from.values);
            }
        }
        self.pending_copies.clear();
        let Model::Llama(model) = self.model.as_ref() else {
            return Err(invalid("native cache model mismatch"));
        };
        let head_dim = model.config().head_dim()?;
        let theta_scale = model.config().rope_base.powf(-2.0 / head_dim as f32);
        for stream in &mut self.streams {
            for (index, cell) in stream.cells.iter_mut().enumerate() {
                if cell.shift == 0 {
                    continue;
                }
                for layer in &mut stream.layers {
                    let row = &mut layer.keys[index * self.width..(index + 1) * self.width];
                    for head in row.chunks_exact_mut(head_dim) {
                        let mut theta = cell.shift as f32;
                        for pair in head.as_chunks_mut::<2>().0 {
                            let (sin, cos) = theta.sin_cos();
                            let (x, y) = (pair[0], pair[1]);
                            pair[0] = x * cos - y * sin;
                            pair[1] = x * sin + y * cos;
                            theta *= theta_scale;
                        }
                    }
                    math::finite(row, "shifted native keys")?;
                }
                cell.shift = 0;
            }
        }
        Ok(())
    }
    fn find_slots(&self, stream: usize, count: usize) -> Result<Vec<usize>> {
        let stream = &self.streams[stream];
        if count > self.capacity {
            return Err(Error::ContextFull {
                maximum: self.capacity,
            });
        }
        let used = stream.cells.iter().filter(|cell| cell.occupied()).count();
        let mut head = stream.head;
        if head > used.saturating_add(count.saturating_mul(2)) {
            head = 0
        }
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(count)
            .map_err(|_| invalid("native slot allocation failed"))?;
        for offset in 0..self.capacity {
            let index = (head + offset) % self.capacity;
            if !stream.cells[index].occupied() {
                selected.push(index);
                if selected.len() == count {
                    return Ok(selected);
                }
            }
        }
        Err(Error::ContextFull {
            maximum: self.capacity,
        })
    }
    pub fn decode_batch(&mut self, batch: &[NativeBatchToken]) -> Result<Vec<TokenOutput>> {
        self.decode_batch_with_cancel(batch, || false)
    }
    /// Validates and stages the batch before committing it. This scalar
    /// compatibility executor retains atomic cancellation while native graph
    /// cancellation can preserve completed microbatches.
    pub fn decode_batch_with_cancel(
        &mut self,
        batch: &[NativeBatchToken],
        cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<TokenOutput>> {
        self.decode_batch_with_work_limit(batch, cancelled, MAX_NATIVE_WORK_BYTES)
    }
    // Keep one production preflight path; module tests lower its ceiling to
    // exercise near-limit rejection without allocating multi-GiB fixtures.
    fn decode_batch_with_work_limit(
        &mut self,
        batch: &[NativeBatchToken],
        mut cancelled: impl FnMut() -> bool,
        work_limit: usize,
    ) -> Result<Vec<TokenOutput>> {
        if batch.is_empty() || batch.len() > self.options.maximum_batch_tokens {
            return Err(Error::InvalidConfig(
                "empty or oversized native batch".into(),
            ));
        }
        let mut last = std::collections::BTreeMap::new();
        for item in batch {
            if item.position < 0
                || item.sequences.is_empty()
                || item.sequences.len() > self.options.maximum_sequences
            {
                return Err(invalid("invalid native batch position or sequence count"));
            }
            if !self.options.unified && item.sequences.len() != 1 {
                return Err(Error::Unsupported(
                    "shared token requires unified native KV".into(),
                ));
            }
            let mut seen = SequenceSet::new();
            for &sequence in &item.sequences {
                self.stream_id(sequence)?;
                if !seen.insert(sequence) {
                    return Err(invalid("duplicate sequence ID in token"));
                }
                if self.position_min(sequence)? != self.position_min(item.sequences[0])?
                    || self.position_max(sequence)? != self.position_max(item.sequences[0])?
                {
                    return Err(invalid(
                        "coupled native sequences have diverged position ranges",
                    ));
                }
                if let Some(old) = last.insert(sequence, item.position) {
                    if old == item.position {
                        return Err(Error::Unsupported("duplicate positions in one native microbatch require batched layer execution".into()));
                    }
                    if old.checked_add(1) != Some(item.position) {
                        return Err(invalid(
                            "native batch sequence positions must be consecutive and nondecreasing",
                        ));
                    }
                } else {
                    let maximum = self.position_max(sequence)?;
                    if maximum >= 0 && maximum.checked_add(1) != Some(item.position) {
                        return Err(invalid(
                            "native sequence must resume after its maximum stored position",
                        ));
                    }
                }
            }
            match &item.input {
                Input::Token(token) if *token as usize >= self.model.vocabulary() => {
                    return Err(Error::InvalidToken {
                        token: *token,
                        vocabulary: self.model.vocabulary(),
                    });
                }
                Input::Embedding(value) => {
                    math::vector(value, self.model.width(), "native input embedding")?
                }
                _ => {}
            }
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let output_floats = batch.iter().try_fold(0usize, |sum, item| {
            let count = match item.output {
                OutputSelection::None => 0,
                OutputSelection::Logits => self.model.vocabulary(),
                OutputSelection::Embedding => self.model.width(),
                OutputSelection::Both => self
                    .model
                    .vocabulary()
                    .checked_add(self.model.width())
                    .ok_or_else(|| invalid("native output shape overflow"))?,
            };
            sum.checked_add(count)
                .ok_or_else(|| invalid("native output count overflow"))
        })?;
        let Model::Llama(model) = self.model.as_ref() else {
            return Err(invalid("native model mismatch"));
        };
        let active = self
            .streams
            .iter()
            .map(|stream| stream.cells.iter().filter(|cell| cell.occupied()).count())
            .max()
            .unwrap_or(0)
            // Earlier chunks remain in the staged cache while later chunks
            // gather their visible rows and retained outputs stay alive.
            .checked_add(batch.len())
            .ok_or_else(|| invalid("native active cache bound overflow"))?
            .min(self.capacity);
        let transient = active
            .checked_mul(self.width)
            .and_then(|n| n.checked_mul(model.layer_count()))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(self.capacity * 16))
            .and_then(|n| n.checked_add(output_floats.checked_mul(4)?))
            .and_then(|n| n.checked_add(batch.len() * std::mem::size_of::<TokenOutput>()))
            .and_then(|n| n.checked_add(model.config().width.checked_mul(128)?))
            .and_then(|n| n.checked_add(model.config().feed_forward.checked_mul(32)?))
            .and_then(|n| n.checked_add(model.config().vocabulary.checked_mul(16)?))
            .ok_or_else(|| invalid("native decode working-set overflow"))?;
        self.check_work_limit(
            self.storage_bytes()?
                .checked_add(transient)
                .ok_or_else(|| invalid("native decode budget overflow"))?,
            work_limit,
        )?;
        let mut staged = self.try_clone()?;
        staged.apply_pending()?;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(batch.len())
            .map_err(|_| invalid("native output allocation failed"))?;
        for (chunk_index, chunk) in batch.chunks(self.options.microbatch_tokens).enumerate() {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let mut by_stream = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for (index, item) in chunk.iter().enumerate() {
                let indices = by_stream
                    .entry(staged.stream_id(item.sequences[0])?)
                    .or_default();
                indices
                    .try_reserve_exact(1)
                    .map_err(|_| invalid("native stream index allocation failed"))?;
                indices.push(index);
            }
            let mut slots = Vec::new();
            slots
                .try_reserve_exact(chunk.len())
                .map_err(|_| invalid("native batch slot allocation failed"))?;
            slots.resize(chunk.len(), (0, 0));
            for (&stream, items) in &by_stream {
                let allocated = staged.find_slots(stream, items.len())?;
                for (&item, &slot) in items.iter().zip(&allocated) {
                    slots[item] = (stream, slot);
                    staged.streams[stream].cells[slot] = Cell {
                        position: chunk[item].position,
                        shift: 0,
                        sequences: chunk[item].sequences.iter().copied().collect(),
                    };
                }
                staged.streams[stream].head = allocated.last().copied().unwrap_or(0) + 1;
            }
            let Model::Llama(model) = staged.model.as_ref() else {
                return Err(invalid("native cache model mismatch"));
            };
            for (within, item) in chunk.iter().enumerate() {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let (stream, slot) = slots[within];
                let mut visible = Vec::new();
                visible
                    .try_reserve_exact(
                        staged.streams[stream]
                            .cells
                            .iter()
                            .filter(|cell| cell.occupied())
                            .count(),
                    )
                    .map_err(|_| invalid("native visible cell allocation failed"))?;
                visible.extend(
                    staged.streams[stream]
                        .cells
                        .iter()
                        .enumerate()
                        .filter(|(index, cell)| {
                            *index != slot
                                && cell.position >= 0
                                && cell.position <= item.position
                                && cell.sequences.contains(&item.sequences[0])
                        })
                        .map(|(index, _)| index),
                );
                let insert = visible.partition_point(|index| *index < slot);
                let mut cache = Vec::new();
                cache
                    .try_reserve_exact(staged.streams[stream].layers.len())
                    .map_err(|_| invalid("native visible layer allocation failed"))?;
                for layer in &staged.streams[stream].layers {
                    let mut keys = Vec::new();
                    let mut values = Vec::new();
                    let count = (visible.len() + 1)
                        .checked_mul(staged.width)
                        .ok_or_else(|| invalid("native visible cache size overflow"))?;
                    keys.try_reserve_exact(count)
                        .map_err(|_| invalid("native visible key allocation failed"))?;
                    values
                        .try_reserve_exact(count)
                        .map_err(|_| invalid("native visible value allocation failed"))?;
                    for &index in &visible {
                        keys.extend_from_slice(
                            &layer.keys[index * staged.width..(index + 1) * staged.width],
                        );
                        values.extend_from_slice(
                            &layer.values[index * staged.width..(index + 1) * staged.width],
                        );
                    }
                    cache.push(KvLayer { keys, values });
                }
                let input = match &item.input {
                    Input::Token(token) => staged.model.embedding(*token)?,
                    Input::Embedding(values) => copy_values(values)?,
                };
                let (embedding, logits) = model.step_inserting(
                    input,
                    item.position as usize,
                    &mut cache,
                    insert,
                    &staged.adapters,
                    &mut cancelled,
                )?;
                for (source, destination) in cache.iter().zip(&mut staged.streams[stream].layers) {
                    destination.keys[slot * staged.width..(slot + 1) * staged.width]
                        .copy_from_slice(
                            &source.keys[insert * staged.width..(insert + 1) * staged.width],
                        );
                    destination.values[slot * staged.width..(slot + 1) * staged.width]
                        .copy_from_slice(
                            &source.values[insert * staged.width..(insert + 1) * staged.width],
                        );
                }
                if item.output != OutputSelection::None {
                    outputs.push(TokenOutput {
                        batch_index: chunk_index * self.options.microbatch_tokens + within,
                        sequence: item.sequences[0],
                        position: item.position as usize,
                        logits: matches!(
                            item.output,
                            OutputSelection::Logits | OutputSelection::Both
                        )
                        .then_some(logits),
                        embedding: matches!(
                            item.output,
                            OutputSelection::Embedding | OutputSelection::Both
                        )
                        .then_some(embedding),
                    });
                }
            }
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        *self = staged;
        Ok(outputs)
    }
    pub fn native_state_size(&self, sequence: Option<i32>, flags: u32) -> Result<usize> {
        native_state::encoded_size(self, sequence, flags)
    }
    pub fn prepare_native_state(&self, sequence: Option<i32>, flags: u32) -> Result<NativeState> {
        native_state::encode(self, sequence, flags)
    }
    pub fn restore_native_state_with_tensors(
        &mut self,
        bytes: &[u8],
        sequence: Option<i32>,
        flags: u32,
        tensors: &[u8],
    ) -> Result<usize> {
        native_state::decode(self, bytes, sequence, flags, Some(tensors))
    }
    /// Source-format bytes for dense F32 memory; separate from Context::save_state.
    /// Pending copy/shift tasks are intentionally not serialized by that format.
    pub fn save_native_state(&self, sequence: Option<i32>, flags: u32) -> Result<Vec<u8>> {
        Ok(native_state::encode(self, sequence, flags)?.bytes)
    }
    /// Bounded parsing, all-or-nothing commit; returns the consumed byte count.
    pub fn restore_native_state(
        &mut self,
        bytes: &[u8],
        sequence: Option<i32>,
        flags: u32,
    ) -> Result<usize> {
        native_state::decode(self, bytes, sequence, flags, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context(unified: bool) -> NativeContext {
        NativeContext::new(
            Arc::new(Model::Llama(crate::fixtures::tiny_llama().unwrap())),
            NativeContextOptions {
                context_tokens: 512,
                maximum_sequences: 4,
                microbatch_tokens: 32,
                maximum_batch_tokens: 128,
                unified,
            },
        )
        .unwrap()
    }
    #[test]
    fn metadata_membership_is_fixed_and_queued_copies_reuse_buffers_in_order() {
        assert_eq!(std::mem::size_of::<SequenceSet>(), 32);
        let mut set = SequenceSet::default();
        for id in (0..256).rev() {
            assert!(set.insert(id));
        }
        assert_eq!(set.len(), 256);
        assert_eq!(
            set.iter().copied().collect::<Vec<_>>(),
            (0..256).collect::<Vec<_>>()
        );
        assert!(!set.insert(256));
        assert!(!set.insert(-1));
        let mut context = context(false);
        for (index, stream) in context.streams.iter_mut().enumerate() {
            for layer in &mut stream.layers {
                layer.keys.fill(index as f32);
                layer.values.fill((index + 10) as f32);
            }
        }
        let pointers: Vec<_> = context
            .streams
            .iter()
            .flat_map(|stream| {
                stream
                    .layers
                    .iter()
                    .map(|layer| (layer.keys.as_ptr(), layer.values.as_ptr()))
            })
            .collect();
        context.copy_range(2, 0, -1, -1).unwrap();
        context.copy_range(0, 1, -1, -1).unwrap();
        context.copy_range(1, 3, -1, -1).unwrap();
        context.apply_pending().unwrap();
        for stream in &context.streams {
            for layer in &stream.layers {
                assert!(layer.keys.iter().all(|f| *f == 2.0));
                assert!(layer.values.iter().all(|f| *f == 12.0));
            }
        }
        let after: Vec<_> = context
            .streams
            .iter()
            .flat_map(|stream| {
                stream
                    .layers
                    .iter()
                    .map(|layer| (layer.keys.as_ptr(), layer.values.as_ptr()))
            })
            .collect();
        assert_eq!(pointers, after);
        assert!(context.pending_copies.is_empty());
    }
    #[test]
    fn later_microbatches_are_counted_before_any_state_is_committed() {
        let mut context = context(true);
        context.options.microbatch_tokens = 4;
        context.options.maximum_batch_tokens = 512;
        let batch = (0..512)
            .map(|position| NativeBatchToken {
                sequences: vec![0],
                position,
                input: Input::Token(1),
                output: OutputSelection::None,
            })
            .collect::<Vec<_>>();
        let before = context.save_native_state(None, 0).unwrap();
        // Allow both persistent copies, the reserved result descriptors and
        // 12 KiB of remaining work. The first four-row gather fits, but the
        // last of 128 microbatches needs all 512 rows and must be rejected.
        let retained =
            2 * context.storage_bytes().unwrap() + batch.len() * std::mem::size_of::<TokenOutput>();
        let outcome = context.decode_batch_with_work_limit(&batch, || false, retained + 12 * 1024);
        assert!(matches!(outcome, Err(Error::Unsupported(_))));
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        // A sufficient ceiling executes the identical valid batch fully.
        assert!(
            context
                .decode_batch_with_work_limit(&batch, || false, retained + 32 * 1024,)
                .unwrap()
                .is_empty()
        );
        assert_eq!(context.position_max(0).unwrap(), 511);
    }
    #[test]
    fn native_resource_limits_fail_before_allocating_or_mutating() {
        let mut context = context(true);
        assert!(context.check_work(MAX_NATIVE_WORK_BYTES).is_err());
        assert!(context.check_work(usize::MAX).is_err());
        let before = context.save_native_state(None, 0).unwrap();
        let malformed = [u32::MAX.to_le_bytes(), [0; 4]].concat();
        assert!(context.restore_native_state(&malformed, None, 0).is_err());
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        let model = context.model.clone();
        assert!(
            NativeContext::new(
                model,
                NativeContextOptions {
                    context_tokens: usize::MAX,
                    maximum_sequences: 256,
                    microbatch_tokens: 1,
                    maximum_batch_tokens: 1,
                    unified: false
                }
            )
            .is_err()
        );
        // Size estimation and encoding agree even at full sequence membership.
        context.streams[0].cells[0] = Cell {
            position: 0,
            shift: 0,
            sequences: (0..256).collect(),
        };
        for sequence in [None, Some(-1), Some(0), Some(255)] {
            for flags in [0, 1, 2, 3, 4] {
                assert_eq!(
                    context.native_state_size(sequence, flags).unwrap(),
                    context
                        .prepare_native_state(sequence, flags)
                        .unwrap()
                        .bytes
                        .len()
                );
            }
        }
    }
}
