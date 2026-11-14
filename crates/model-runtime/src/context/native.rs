//! Frozen dense F32/F16 KV memory semantics, separate from the strict Context API.
//!
//! Physical cache slots and pending copy/shift work are deliberately retained:
//! native state bytes describe stored data and position metadata, and do not
//! serialize pending updates. This module does not claim recurrent cache support.
use super::{BatchToken, Input, OutputSelection, TokenOutput};
use crate::{AdapterSet, Error, Model, Result, math};
use std::sync::Arc;

#[path = "native_state.rs"]
mod native_state;
pub use native_state::NativeState;
#[path = "native_kv.rs"]
mod native_kv;
pub use native_kv::NativeKvType;
use native_kv::{KvBuffer, StoredKvLayer};
#[path = "native_batch.rs"]
mod native_batch;
#[path = "native_decode.rs"]
mod native_decode;
#[path = "native_row_map.rs"]
mod native_row_map;

/// Dense F32/F16 compatibility cache configuration.
#[derive(Debug, Clone)]
pub struct NativeContextOptions {
    pub context_tokens: usize,
    pub maximum_sequences: usize,
    pub microbatch_tokens: usize,
    pub maximum_batch_tokens: usize,
    pub unified: bool,
}

/// Explicit CPU execution configuration, independent of cache storage types.
/// The old constructors retain one-worker nonflash behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeExecutionOptions {
    /// Resolved native CPU flash decision; the C facade resolves AUTO by the
    /// frozen CPU device-support rule before constructing this safe executor.
    pub flash_attention: bool,
    pub threads: usize,
    pub threads_batch: usize,
    /// Source embeddings mode changes separate-stream microbatch splitting.
    pub embeddings: bool,
}
impl Default for NativeExecutionOptions {
    fn default() -> Self {
        Self {
            flash_attention: false,
            threads: 1,
            threads_batch: 1,
            embeddings: false,
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeExecutionStats {
    pub microbatches: usize,
    pub flash_graphs: usize,
    pub configured_worker_partitions: usize,
    pub threads_executing_partitions: usize,
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
fn copy_layers(source: &[StoredKvLayer]) -> Result<Vec<StoredKvLayer>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source.len())
        .map_err(|_| invalid("native layer allocation failed"))?;
    for layer in source {
        result.push(StoredKvLayer {
            keys: layer.keys.try_clone()?,
            values: layer.values.try_clone()?,
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
    layers: Vec<StoredKvLayer>,
    head: usize,
}

/// CPU implementation of the frozen Llama F32/F16 memory contract.
///
/// The strict Context type remains the safe sequential API. This compatibility
/// executor owns physical cells, can retain holes and duplicate positions after
/// position transforms, and serializes the source's unversioned memory format.
#[derive(Debug, Clone)]
pub struct NativeContext {
    model: Arc<Model>,
    options: NativeContextOptions,
    adapters: Arc<AdapterSet>,
    streams: Vec<Stream>,
    pending_copies: Vec<(usize, usize)>,
    capacity: usize,
    width: usize,
    key_type: NativeKvType,
    value_type: NativeKvType,
    execution: NativeExecutionOptions,
    last_execution: NativeExecutionStats,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidState(message.into())
}
fn range(p0: i32, p1: i32) -> (i32, i32) {
    (p0.max(0), if p1 < 0 { i32::MAX } else { p1 })
}
impl NativeContext {
    /// Preserve the original safe F32/nonflash compatibility constructor.
    pub fn new(model: Arc<Model>, options: NativeContextOptions) -> Result<Self> {
        Self::new_with_cache_types(model, options, NativeKvType::F32, NativeKvType::F32)
    }
    /// Select the source K/V element types independently. Attention remains
    /// explicit nonflash: query and probability operands retain F32 precision.
    pub fn new_with_cache_types(
        model: Arc<Model>,
        options: NativeContextOptions,
        key_type: NativeKvType,
        value_type: NativeKvType,
    ) -> Result<Self> {
        Self::new_with_execution(
            model,
            options,
            key_type,
            value_type,
            NativeExecutionOptions::default(),
        )
    }
    /// Select explicit measured CPU execution while preserving independently
    /// typed resident K/V storage and the strict Context API.
    pub fn new_with_execution(
        model: Arc<Model>,
        options: NativeContextOptions,
        key_type: NativeKvType,
        value_type: NativeKvType,
        execution: NativeExecutionOptions,
    ) -> Result<Self> {
        Self::validate_execution(execution)?;
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
                "native compatibility memory currently supports dense Llama models only".into(),
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
            .and_then(|n| n.checked_mul(key_type.size() + value_type.size()))
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
        let descriptor_bytes = streams
            .checked_mul(std::mem::size_of::<Stream>())
            .and_then(|n| {
                n.checked_add(
                    streams
                        .checked_mul(llama.layer_count())?
                        .checked_mul(std::mem::size_of::<StoredKvLayer>())?,
                )
            })
            .ok_or_else(|| invalid("native descriptor accounting overflow"))?;
        if bytes
            .checked_add(metadata_bytes)
            .and_then(|n| n.checked_add(descriptor_bytes))
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
                layers.push(StoredKvLayer {
                    keys: KvBuffer::new(key_type, elements)?,
                    values: KvBuffer::new(value_type, elements)?,
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
            adapters: Arc::new(AdapterSet::default()),
            streams: all,
            pending_copies: Vec::new(),
            capacity,
            width,
            key_type,
            value_type,
            execution,
            last_execution: NativeExecutionStats::default(),
        })
    }
    fn storage_bytes(&self) -> Result<usize> {
        let mut bytes = self
            .pending_copies
            .capacity()
            .checked_mul(std::mem::size_of::<(usize, usize)>())
            .and_then(|n| {
                n.checked_add(
                    self.streams
                        .capacity()
                        .checked_mul(std::mem::size_of::<Stream>())?,
                )
            })
            .ok_or_else(|| invalid("native memory accounting overflow"))?;
        for stream in &self.streams {
            bytes = bytes
                .checked_add(stream.cells.capacity() * std::mem::size_of::<Cell>())
                .and_then(|n| {
                    n.checked_add(stream.layers.capacity() * std::mem::size_of::<StoredKvLayer>())
                })
                .ok_or_else(|| invalid("native metadata accounting overflow"))?;
            for layer in &stream.layers {
                bytes = bytes
                    .checked_add(layer.keys.capacity_bytes()?)
                    .and_then(|n| n.checked_add(layer.values.capacity_bytes().ok()?))
                    .ok_or_else(|| invalid("native tensor accounting overflow"))?;
            }
        }
        bytes = bytes
            .checked_add(self.adapters.owned_bytes()?)
            .ok_or_else(|| invalid("native adapter byte accounting overflow"))?;
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
            key_type: self.key_type,
            value_type: self.value_type,
            execution: self.execution,
            last_execution: self.last_execution,
        })
    }
    pub fn cache_types(&self) -> (NativeKvType, NativeKvType) {
        (self.key_type, self.value_type)
    }
    pub fn values_are_transposed(&self) -> bool {
        !self.execution.flash_attention
    }
    fn validate_execution(execution: NativeExecutionOptions) -> Result<()> {
        if execution.threads == 0
            || execution.threads > 256
            || execution.threads_batch == 0
            || execution.threads_batch > 256
        {
            return Err(Error::InvalidConfig(
                "native worker counts must be in 1..=256".into(),
            ));
        }
        if !execution.flash_attention && (execution.threads != 1 || execution.threads_batch != 1) {
            return Err(Error::Unsupported(
                "nonflash multiple-worker execution is not implemented".into(),
            ));
        }
        if execution.flash_attention
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
                ),
            ))
        {
            return Err(Error::Unsupported(
                "native flash context requires a measured x64 MSVC or GNU CPU profile".into(),
            ));
        }
        Ok(())
    }
    pub fn execution_options(&self) -> NativeExecutionOptions {
        self.execution
    }
    /// Counts returned by actual shared-kernel worker executions in the last
    /// successful decode. Failed/cancelled transactions retain prior statistics.
    pub fn execution_stats(&self) -> NativeExecutionStats {
        self.last_execution
    }
    pub fn set_threads(&mut self, generation: usize, batch: usize) -> Result<()> {
        let execution = NativeExecutionOptions {
            threads: generation,
            threads_batch: batch,
            ..self.execution
        };
        Self::validate_execution(execution)?;
        self.execution = execution;
        Ok(())
    }
    pub fn set_embeddings(&mut self, enabled: bool) {
        self.execution.embeddings = enabled;
    }
    /// Replace adapters atomically without clearing already evaluated KV state.
    /// Existing contexts and transactional copies share immutable adapter weights.
    /// Check replacement payload before a C facade allocates its execution copy.
    pub fn preflight_adapter_storage(&self, bytes: usize) -> Result<()> {
        self.check_work(bytes)
    }
    pub fn set_adapters(&mut self, adapters: AdapterSet) -> Result<()> {
        adapters.validate(&self.model)?;
        let bytes = adapters.owned_bytes()?;
        self.check_work(bytes)?;
        self.adapters = Arc::new(adapters);
        Ok(())
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
                    layer.keys.clear();
                    layer.values.clear();
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
                to.keys.copy_from(&from.keys)?;
                to.values.copy_from(&from.values)?;
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
                    let start = index * self.width;
                    for head in (start..start + self.width).step_by(head_dim) {
                        let mut theta = cell.shift as f32;
                        for pair in (head..head + head_dim).step_by(2) {
                            let (sin, cos) = theta.sin_cos();
                            let (x, y) = (layer.keys.get(pair), layer.keys.get(pair + 1));
                            layer.keys.set(pair, x * cos - y * sin);
                            layer.keys.set(pair + 1, x * sin + y * cos);
                            theta *= theta_scale;
                        }
                    }
                    if (start..start + self.width).any(|index| !layer.keys.get(index).is_finite()) {
                        return Err(Error::NonFinite("shifted native keys".into()));
                    }
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
    /// Validates and stages the batch before committing it. This
    /// compatibility executor retains atomic cancellation while native graph
    /// cancellation can preserve completed microbatches.
    pub fn decode_batch_with_cancel(
        &mut self,
        batch: &[NativeBatchToken],
        cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<TokenOutput>> {
        self.decode_batch_with_work_limit(batch, cancelled, MAX_NATIVE_WORK_BYTES)
    }
    /// Run a fallible output consumer before committing KV, pending copies,
    /// execution statistics or any other native context state. The consumer
    /// must stage its own state; it is invoked once after all model outputs.
    pub fn decode_batch_with_output_check(
        &mut self,
        batch: &[NativeBatchToken],
        cancelled: impl FnMut() -> bool,
        accept_outputs: impl FnOnce(&[TokenOutput]) -> Result<()>,
    ) -> Result<Vec<TokenOutput>> {
        self.decode_batch_checked_with_work_limit(
            batch,
            cancelled,
            MAX_NATIVE_WORK_BYTES,
            accept_outputs,
        )
    }
    // Keep one production preflight path; module tests lower its ceiling to
    // exercise near-limit rejection without allocating multi-GiB fixtures.
    fn decode_batch_with_work_limit(
        &mut self,
        batch: &[NativeBatchToken],
        cancelled: impl FnMut() -> bool,
        work_limit: usize,
    ) -> Result<Vec<TokenOutput>> {
        self.decode_batch_checked_with_work_limit(batch, cancelled, work_limit, |_| Ok(()))
    }
    fn decode_batch_checked_with_work_limit(
        &mut self,
        batch: &[NativeBatchToken],
        mut cancelled: impl FnMut() -> bool,
        work_limit: usize,
        accept_outputs: impl FnOnce(&[TokenOutput]) -> Result<()>,
    ) -> Result<Vec<TokenOutput>> {
        if batch.is_empty() || batch.len() > self.options.maximum_batch_tokens {
            return Err(Error::InvalidConfig(
                "empty or oversized native batch".into(),
            ));
        }
        // Sequence IDs are already bounded to 0..256; avoid an unaccounted
        // heap-backed tree for this validation-only lookup.
        let mut last = [None; 256];
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
                // Batch allocation uses the configured ID range even when the
                // unified memory API can address all 256 native sequence IDs.
                if sequence < 0 || sequence as usize >= self.options.maximum_sequences {
                    return Err(Error::InvalidSequence(sequence));
                }
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
                if let Some(old) = last[sequence as usize].replace(item.position) {
                    if old != item.position && old.checked_add(1) != Some(item.position) {
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
        native_decode::execute(self, batch, cancelled, work_limit, accept_outputs)
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
    /// Source-format bytes for typed dense memory; separate from Context::save_state.
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
    fn address(buffer: &KvBuffer) -> usize {
        match buffer {
            KvBuffer::F32(v) => v.as_ptr().addr(),
            KvBuffer::F16(v) => v.as_ptr().addr(),
        }
    }
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
                for at in 0..layer.keys.len() {
                    layer.keys.set(at, index as f32);
                    layer.values.set(at, (index + 10) as f32);
                }
            }
        }
        let pointers: Vec<_> = context
            .streams
            .iter()
            .flat_map(|stream| {
                stream
                    .layers
                    .iter()
                    .map(|layer| (address(&layer.keys), address(&layer.values)))
            })
            .collect();
        context.copy_range(2, 0, -1, -1).unwrap();
        context.copy_range(0, 1, -1, -1).unwrap();
        context.copy_range(1, 3, -1, -1).unwrap();
        context.apply_pending().unwrap();
        for stream in &context.streams {
            for layer in &stream.layers {
                assert!((0..layer.keys.len()).all(|index| layer.keys.get(index) == 2.0));
                assert!((0..layer.values.len()).all(|index| layer.values.get(index) == 12.0));
            }
        }
        let after: Vec<_> = context
            .streams
            .iter()
            .flat_map(|stream| {
                stream
                    .layers
                    .iter()
                    .map(|layer| (address(&layer.keys), address(&layer.values)))
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
        // The complete physical score row is retained even in the earliest
        // small microbatch. A ceiling below that independently known payload
        // must reject before any of the 128 microbatches can commit.
        let retained =
            2 * context.storage_bytes().unwrap() + batch.len() * std::mem::size_of::<TokenOutput>();
        let outcome = context.decode_batch_with_work_limit(
            &batch,
            || false,
            retained + context.capacity * std::mem::size_of::<f32>() - 1,
        );
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
    fn full_microbatch_metadata_is_counted_separately_from_attention_rows() {
        let mut context = context(true);
        context.options.microbatch_tokens = 512;
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
        let retained =
            2 * context.storage_bytes().unwrap() + batch.len() * std::mem::size_of::<TokenOutput>();
        // Whole-microbatch hidden/query matrices, physical scores and slot
        // bookkeeping coexist; they cannot share a single row's allowance.
        assert!(matches!(
            context.decode_batch_with_work_limit(&batch, || false, retained + 32 * 1024),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        assert!(
            context
                .decode_batch_with_work_limit(&batch, || false, retained + 64 * 1024)
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
    #[test]
    fn native_half_state_preserves_raw_payloads_and_rejects_wrong_storage_type_atomically() {
        let model = Arc::new(Model::Llama(crate::fixtures::tiny_llama().unwrap()));
        let Model::Llama(llama) = model.as_ref() else {
            unreachable!()
        };
        let width = llama.config().kv_width().unwrap();
        let options = NativeContextOptions {
            context_tokens: 65_536usize.div_ceil(width),
            maximum_sequences: 1,
            microbatch_tokens: 1,
            maximum_batch_tokens: 1,
            unified: true,
        };
        let mut original = NativeContext::new_with_cache_types(
            model.clone(),
            options.clone(),
            NativeKvType::F16,
            NativeKvType::F16,
        )
        .unwrap();
        for (position, cell) in original.streams[0].cells.iter_mut().enumerate() {
            *cell = Cell {
                position: position as i32,
                shift: 0,
                sequences: [0].into_iter().collect(),
            };
        }
        for layer in &mut original.streams[0].layers {
            for index in 0..layer.keys.len() {
                layer.keys.set_bits(index, (index & 0xffff) as u32);
                layer.values.set_bits(index, ((!index) & 0xffff) as u32);
            }
        }
        let bytes = original.save_native_state(None, 0).unwrap();
        let mut restored = NativeContext::new_with_cache_types(
            model.clone(),
            options.clone(),
            NativeKvType::F16,
            NativeKvType::F16,
        )
        .unwrap();
        assert_eq!(
            restored.restore_native_state(&bytes, None, 0).unwrap(),
            bytes.len()
        );
        assert_eq!(restored.save_native_state(None, 0).unwrap(), bytes);
        let device = original.prepare_native_state(Some(0), 2).unwrap();
        restored.clear(true);
        restored
            .restore_native_state_with_tensors(&device.bytes, Some(0), 2, &device.tensors)
            .unwrap();
        assert_eq!(restored.save_native_state(None, 0).unwrap(), bytes);
        // A type mismatch cannot reinterpret two-byte payloads as F32 or
        // commit the already parsed metadata in the transactional executor.
        let mut wrong = NativeContext::new(model, options).unwrap();
        let before = wrong.save_native_state(None, 0).unwrap();
        assert!(wrong.restore_native_state(&bytes, None, 0).is_err());
        assert_eq!(wrong.save_native_state(None, 0).unwrap(), before);
        for cut in [0, 4, 8, 16, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                restored
                    .restore_native_state(&bytes[..cut], None, 0)
                    .is_err()
            );
            assert_eq!(restored.save_native_state(None, 0).unwrap(), bytes);
        }
    }
    #[test]
    fn decode_ids_use_configured_limit_while_memory_keeps_256_ids() {
        let mut context = context(true);
        context
            .decode_batch(&[NativeBatchToken {
                sequences: vec![0],
                position: 0,
                input: Input::Token(1),
                output: OutputSelection::Logits,
            }])
            .unwrap();
        context.copy_range(0, 255, -1, -1).unwrap();
        assert_eq!(context.position_max(255).unwrap(), 0);
        let before = context.save_native_state(None, 0).unwrap();
        assert!(matches!(
            context.decode_batch(&[NativeBatchToken {
                sequences: vec![255],
                position: 1,
                input: Input::Token(1),
                output: OutputSelection::Logits,
            }]),
            Err(Error::InvalidSequence(255))
        ));
        assert_eq!(context.save_native_state(None, 0).unwrap(), before);
        assert_eq!(context.position_max(255).unwrap(), 0);
    }
}
