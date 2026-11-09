//! Safe CPU FLASH_ATTN_EXT for the frozen scalar-dot / scalar-GEMM profiles.
//!
//! Borrowed views retain the complete physical Q/K/V and F16 mask geometry.
//! Query tiling, masked tile skipping, half accumulation, split-KV partitions,
//! and the final reduction order are part of the numerical contract.
//!
//! The executor starts real scoped workers when more than one worker is
//! requested. The separately named serial-partition entry is a diagnostic,
//! not a claim that multiple workers ran. No project C or vendor math is used.
#![forbid(unsafe_code)]

use crate::cpu_math::{MathPolicy, native_math_policy, softmax_exp_sum_f32_with_policy};
use crate::{DType, f16_to_f32, f32_to_f16};
use std::fmt;
use std::mem::size_of;
use std::thread::{self, ScopedJoinHandle, ThreadId};
mod rows;
pub use rows::KvRows;

pub const QUERY_TILE: usize = 64;
pub const KV_TILE: usize = 64;
pub const MAX_WORKERS: usize = 256;
/// Requested stack reservation of each additional OS worker.
pub const WORKER_STACK_BYTES: usize = 512 * 1024;
/// Bounded allowance for std scoped-thread ownership and OS creation metadata.
/// OS bookkeeping and allocator overhead are not a resident-memory guarantee.
pub const WORKER_METADATA_ALLOWANCE: usize = 4096;

#[derive(Debug)]
pub enum Error {
    Invalid(&'static str),
    UnsupportedDType(DType),
    Overflow,
    Allocation,
    ResourceLimit { required: usize, limit: usize },
    WorkLimit { required: u64, limit: u64 },
    ThreadSpawn(std::io::Error),
    WorkerPanic,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid flash attention input: {message}"),
            Self::UnsupportedDType(dtype) => {
                write!(f, "unsupported flash attention dtype: {dtype:?}")
            }
            Self::Overflow => f.write_str("flash attention size arithmetic overflow"),
            Self::Allocation => f.write_str("flash attention allocation failed"),
            Self::ResourceLimit { required, limit } => {
                write!(
                    f,
                    "flash attention requires {required} budget bytes; limit is {limit}"
                )
            }
            Self::WorkLimit { required, limit } => {
                write!(
                    f,
                    "flash attention requires {required} scalar work items; limit is {limit}"
                )
            }
            Self::ThreadSpawn(error) => {
                write!(f, "flash attention worker creation failed: {error}")
            }
            Self::WorkerPanic => f.write_str("flash attention worker panicked"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or(Error::Overflow)
}
fn mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Error::Overflow)
}
fn zeros<T: Default + Clone>(length: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Allocation)?;
    values.resize(length, T::default());
    Ok(values)
}
fn reserved<T>(length: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::Allocation)?;
    Ok(values)
}

#[derive(Clone, Copy, Debug)]
enum Storage<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    Bytes(&'a [u8]),
}

/// Immutable GGML-order dimensions and byte strides. Axis zero is the scalar
/// channel axis. Typed and raw views never cast a byte pointer to a Rust scalar.
#[derive(Clone, Copy, Debug)]
pub struct View4<'a> {
    storage: Storage<'a>,
    dtype: DType,
    shape: [usize; 4],
    strides: [usize; 4],
    offset: usize,
}
impl<'a> View4<'a> {
    pub fn from_f32(
        values: &'a [f32],
        shape: [usize; 4],
        byte_strides: [usize; 4],
        byte_offset: usize,
    ) -> Result<Self> {
        Self::new(
            Storage::F32(values),
            DType::F32,
            mul(values.len(), 4)?,
            shape,
            byte_strides,
            byte_offset,
        )
    }
    pub fn from_f16(
        values: &'a [u16],
        shape: [usize; 4],
        byte_strides: [usize; 4],
        byte_offset: usize,
    ) -> Result<Self> {
        Self::new(
            Storage::F16(values),
            DType::F16,
            mul(values.len(), 2)?,
            shape,
            byte_strides,
            byte_offset,
        )
    }
    pub fn from_bytes(
        dtype: DType,
        values: &'a [u8],
        shape: [usize; 4],
        byte_strides: [usize; 4],
        byte_offset: usize,
    ) -> Result<Self> {
        Self::new(
            Storage::Bytes(values),
            dtype,
            values.len(),
            shape,
            byte_strides,
            byte_offset,
        )
    }
    fn new(
        storage: Storage<'a>,
        dtype: DType,
        bytes: usize,
        shape: [usize; 4],
        strides: [usize; 4],
        offset: usize,
    ) -> Result<Self> {
        let width = match dtype {
            DType::F32 => 4,
            DType::F16 => 2,
            _ => return Err(Error::UnsupportedDType(dtype)),
        };
        if shape.contains(&0) {
            return Err(Error::Invalid(
                "empty views are outside the source CPU domain",
            ));
        }
        if !offset.is_multiple_of(width)
            || strides.iter().any(|stride| !stride.is_multiple_of(width))
        {
            return Err(Error::Invalid("unaligned scalar byte offset or stride"));
        }
        let mut end = add(offset, width)?;
        for axis in 0..4 {
            end = add(end, mul(shape[axis] - 1, strides[axis])?)?;
        }
        if end > bytes {
            return Err(Error::Invalid("view exceeds its borrowed storage"));
        }
        Ok(Self {
            storage,
            dtype,
            shape,
            strides,
            offset,
        })
    }
    pub const fn dtype(&self) -> DType {
        self.dtype
    }
    pub const fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub const fn byte_strides(&self) -> [usize; 4] {
        self.strides
    }
    pub const fn byte_offset(&self) -> usize {
        self.offset
    }
    fn width(&self) -> usize {
        if self.dtype == DType::F32 { 4 } else { 2 }
    }
    fn contiguous(&self) -> Result<bool> {
        let mut expected = self.width();
        for axis in 0..4 {
            if self.shape[axis] != 1 && self.strides[axis] != expected {
                return Ok(false);
            }
            expected = mul(expected, self.shape[axis])?;
        }
        Ok(true)
    }
    fn value(&self, coordinates: [usize; 4]) -> f32 {
        let at = self.offset
            + coordinates[0] * self.strides[0]
            + coordinates[1] * self.strides[1]
            + coordinates[2] * self.strides[2]
            + coordinates[3] * self.strides[3];
        match self.storage {
            Storage::F32(values) => values[at / 4],
            Storage::F16(values) => f16_to_f32(values[at / 2]),
            Storage::Bytes(values) if self.dtype == DType::F32 => {
                f32::from_le_bytes(values[at..at + 4].try_into().unwrap())
            }
            Storage::Bytes(values) => {
                f16_to_f32(u16::from_le_bytes(values[at..at + 2].try_into().unwrap()))
            }
        }
    }
}

/// Q is [DK, queries, Q heads, batches], K is [DK, physical KV, K heads, batches],
/// V is [DV, physical KV, V heads, batches]. Mask is the original full contiguous
/// F16 [physical columns, padded queries, mask heads, mask batches] tensor.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub q: View4<'a>,
    pub k: View4<'a>,
    pub v: View4<'a>,
    pub mask: Option<View4<'a>>,
    pub sinks: Option<View4<'a>>,
}
/// NaN operand selection of the two independently captured C compiler profiles.
/// This is separate from the explicit SIMD exponential policy. Neither policy
/// changes finite scalar arithmetic or uses foreign-platform C runtime calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilerPolicy {
    Msvc,
    Gnu,
}

pub const fn native_compiler_policy() -> CompilerPolicy {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        CompilerPolicy::Gnu
    } else {
        CompilerPolicy::Msvc
    }
}

// Rust leaves NaN payload choice unspecified. Make the original instruction
// operand choice explicit while retaining ordinary arithmetic for finite inputs.
#[inline]
fn preferred_add(preferred: f32, other: f32) -> f32 {
    if preferred.is_nan() {
        f32::from_bits(preferred.to_bits() | 0x0040_0000)
    } else if other.is_nan() {
        f32::from_bits(other.to_bits() | 0x0040_0000)
    } else {
        preferred + other
    }
}
#[inline]
fn preferred_add_f64(preferred: f64, other: f64) -> f64 {
    if preferred.is_nan() {
        f64::from_bits(preferred.to_bits() | 0x0008_0000_0000_0000)
    } else if other.is_nan() {
        f64::from_bits(other.to_bits() | 0x0008_0000_0000_0000)
    } else {
        preferred + other
    }
}
#[inline]
fn preferred_mul(preferred: f32, other: f32) -> f32 {
    if preferred.is_nan() {
        f32::from_bits(preferred.to_bits() | 0x0040_0000)
    } else if other.is_nan() {
        f32::from_bits(other.to_bits() | 0x0040_0000)
    } else {
        preferred * other
    }
}
fn source_scale(values: &mut [f32], scale: f32, policy: CompilerPolicy) {
    let width = match policy {
        CompilerPolicy::Msvc => 4,
        CompilerPolicy::Gnu => 2,
    };
    scale_with_prefix(values, scale, width);
}
fn source_tile_or_split_scale(values: &mut [f32], scale: f32, policy: CompilerPolicy) {
    let width = match policy {
        CompilerPolicy::Msvc => 16,
        CompilerPolicy::Gnu => 2,
    };
    scale_with_prefix(values, scale, width);
}
fn scale_with_prefix(values: &mut [f32], scale: f32, width: usize) {
    let prefix = values.len() / width * width;
    for (index, value) in values.iter_mut().enumerate() {
        *value = if index < prefix {
            preferred_mul(*value, scale)
        } else {
            preferred_mul(scale, *value)
        };
    }
}
fn source_sum(previous: f32, scale: f32, probability: f32, policy: CompilerPolicy) -> f32 {
    let scaled = preferred_mul(previous, scale);
    match policy {
        CompilerPolicy::Msvc => preferred_add(scaled, probability),
        CompilerPolicy::Gnu => preferred_add(probability, scaled),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub scale: f32,
    pub max_bias: f32,
    pub logit_softcap: f32,
    pub workers: usize,
    pub math_policy: MathPolicy,
    pub compiler_policy: CompilerPolicy,
    pub force_reference: bool,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            scale: 1.0,
            max_bias: 0.0,
            logit_softcap: 0.0,
            workers: 1,
            math_policy: native_math_policy(),
            compiler_policy: native_compiler_policy(),
            force_reference: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    OneChunk,
    Tiled,
    SplitKv,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    All,
    /// Coordinate in the original complete Q tensor, not a one-row replacement
    /// geometry. The full physical mask determines the source tile-skip result.
    Row {
        query: usize,
        head: usize,
        batch: usize,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    ScopedWorkers,
    SerialPartitions,
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_bytes: usize,
    pub max_scalar_work: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024 * 1024,
            max_scalar_work: 1 << 36,
        }
    }
}
/// These categories are non-overlapping and summed once for the execution peak.
/// Borrowed input storage and the caller's pre-existing thread stack are excluded.
/// Allocator/OS implementation bookkeeping is not a resident-memory guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRequirements {
    pub output_bytes: usize,
    pub staged_result_bytes: usize,
    pub worker_scratch_bytes: usize,
    pub descriptor_bytes: usize,
    pub requested_thread_stack_bytes: usize,
    pub thread_metadata_allowance_bytes: usize,
    pub peak_bytes: usize,
    pub scalar_work_items: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionStats {
    pub configured_workers: usize,
    pub completed_partitions: usize,
    pub worker_threads: usize,
    pub mode: ExecutionMode,
}
#[derive(Debug)]
pub struct Output {
    pub values: Vec<f32>,
    pub shape: [usize; 4],
    pub branch: Branch,
    pub stats: ExecutionStats,
    pub requirements: MemoryRequirements,
}

#[derive(Clone, Copy, Debug)]
pub struct Plan<'a> {
    inputs: Inputs<'a>,
    kv_rows: Option<KvRows<'a>>,
    params: Params,
    branch: Branch,
    dk: usize,
    dv: usize,
    nq: usize,
    heads: usize,
    batches: usize,
    nkv: usize,
    rows: usize,
    query_chunks: usize,
    rows_per_chunk: usize,
    scale: f32,
    head_power: usize,
    slope0: f32,
    slope1: f32,
}
impl<'a> Plan<'a> {
    pub fn new(inputs: Inputs<'a>, params: Params) -> Result<Self> {
        Self::new_impl(inputs, None, params)
    }
    /// Source FLASH_ATTN_EXT_ROWS: a complete row map over shared physical K/V.
    /// Shape/storage checks happen here. Index values are validated after work
    /// preflight and before execution allocates output or worker buffers.
    pub fn new_with_rows(inputs: Inputs<'a>, rows: KvRows<'a>, params: Params) -> Result<Self> {
        Self::new_impl(inputs, Some(rows), params)
    }
    fn new_impl(inputs: Inputs<'a>, kv_rows: Option<KvRows<'a>>, params: Params) -> Result<Self> {
        if inputs.q.dtype != DType::F32 {
            return Err(Error::UnsupportedDType(inputs.q.dtype));
        }
        if params.workers == 0 || params.workers > MAX_WORKERS {
            return Err(Error::Invalid("worker count must be in 1..=256"));
        }
        if !params.scale.is_finite()
            || !params.max_bias.is_finite()
            || !params.logit_softcap.is_finite()
        {
            return Err(Error::Invalid("scale, max_bias and softcap must be finite"));
        }
        for tensor in [inputs.q, inputs.k, inputs.v] {
            if tensor.strides[0] != tensor.width() {
                return Err(Error::Invalid("Q/K/V channel rows must be contiguous"));
            }
            if tensor
                .shape
                .iter()
                .any(|&extent| extent > i32::MAX as usize)
            {
                return Err(Error::Invalid(
                    "source CPU signed-I32 dimension domain exceeded",
                ));
            }
        }
        let [dk, nq, heads, batches] = inputs.q.shape;
        let [kd, physical_kv, kh, kb] = inputs.k.shape;
        let [dv, nv, vh, vb] = inputs.v.shape;
        let (nkv, batches_match) = if let Some(rows) = kv_rows {
            let Some(mask) = inputs.mask else {
                return Err(Error::Invalid("kv_rows requires a complete physical mask"));
            };
            if rows.shape()[1] != batches || mask.shape[0] != rows.shape()[0] {
                return Err(Error::Invalid(
                    "kv_rows dimensions must match mask columns and Q batches",
                ));
            }
            (rows.shape()[0], kb == 1 && vb == 1)
        } else {
            (physical_kv, kb == batches && vb == batches)
        };
        if dk != kd
            || (kv_rows.is_none() && nv < physical_kv)
            || !batches_match
            || !heads.is_multiple_of(kh)
            || !heads.is_multiple_of(vh)
        {
            return Err(Error::Invalid(
                "Q/K/V dimensions or grouped head broadcasting mismatch",
            ));
        }
        if let Some(mask) = inputs.mask {
            if mask.dtype != DType::F16
                || !mask.contiguous()?
                || mask.shape[0] < nkv
                || mask.shape[1] < nq
                || !heads.is_multiple_of(mask.shape[2])
                || !batches.is_multiple_of(mask.shape[3])
            {
                return Err(Error::Invalid(
                    "mask must be full contiguous F16 with source broadcasting",
                ));
            }
        } else if params.max_bias > 0.0 {
            return Err(Error::Invalid("positive max_bias requires a physical mask"));
        }
        if let Some(sinks) = inputs.sinks
            && (sinks.dtype != DType::F32 || sinks.shape[0] != heads || sinks.strides[0] != 4)
        {
            return Err(Error::Invalid(
                "sinks must expose one contiguous F32 per query head",
            ));
        }
        let rows = mul(mul(nq, heads)?, batches)?;
        if rows > i32::MAX as usize {
            return Err(Error::Invalid("source CPU query-row index domain exceeded"));
        }
        let matched = inputs.k.dtype == inputs.v.dtype;
        let branch = if !params.force_reference
            && kv_rows.is_none()
            && matched
            && nq == 1
            && batches == 1
            && nkv >= 512
        {
            Branch::SplitKv
        } else if !params.force_reference && matched && nq >= QUERY_TILE {
            Branch::Tiled
        } else {
            Branch::OneChunk
        };
        let initial_size = rows.div_ceil(mul(params.workers, 4)?);
        let mut query_chunks = rows.div_ceil(initial_size);
        if params.workers == 1 || query_chunks < params.workers {
            query_chunks = params.workers;
        }
        let rows_per_chunk = rows.div_ceil(query_chunks);
        let head_power = 1_usize << heads.ilog2();
        let slope0 = 2.0_f32.powf(-params.max_bias / head_power as f32);
        let slope1 = 2.0_f32.powf(-(params.max_bias / 2.0) / head_power as f32);
        let scale = if params.logit_softcap != 0.0 {
            params.scale / params.logit_softcap
        } else {
            params.scale
        };
        if !scale.is_finite() {
            return Err(Error::Invalid(
                "softcap-scaled attention scale is not finite",
            ));
        }
        Ok(Self {
            inputs,
            kv_rows,
            params,
            branch,
            dk,
            dv,
            nq,
            heads,
            batches,
            nkv,
            rows,
            query_chunks,
            rows_per_chunk,
            scale,
            head_power,
            slope0,
            slope1,
        })
    }
    pub const fn branch(&self) -> Branch {
        self.branch
    }
    pub fn output_shape(&self, selection: Selection) -> Result<[usize; 4]> {
        self.selected_row(selection)?;
        Ok(match selection {
            Selection::All => [self.dv, self.heads, self.nq, self.batches],
            _ => [self.dv, 1, 1, 1],
        })
    }
    fn selected_row(&self, selection: Selection) -> Result<Option<usize>> {
        match selection {
            Selection::All => Ok(None),
            Selection::Row { query, head, batch }
                if query < self.nq && head < self.heads && batch < self.batches =>
            {
                Ok(Some(query + self.nq * (head + self.heads * batch)))
            }
            _ => Err(Error::Invalid(
                "selected row is outside original Q geometry",
            )),
        }
    }
    fn worker_rows(&self, worker: usize, selected: Option<usize>) -> usize {
        if self.branch == Branch::SplitKv {
            return if selected.is_some() { 1 } else { self.heads };
        }
        if let Some(row) = selected {
            return usize::from((row / self.rows_per_chunk) % self.params.workers == worker);
        }
        (worker..self.query_chunks)
            .step_by(self.params.workers)
            .map(|chunk| {
                let start = chunk * self.rows_per_chunk;
                self.rows.saturating_sub(start).min(self.rows_per_chunk)
            })
            .sum()
    }
    pub fn requirements(&self, selection: Selection) -> Result<MemoryRequirements> {
        self.requirements_for(selection, ExecutionMode::ScopedWorkers)
    }
    fn requirements_for(
        &self,
        selection: Selection,
        mode: ExecutionMode,
    ) -> Result<MemoryRequirements> {
        let selected = self.selected_row(selection)?;
        let output_rows = if selected.is_some() { 1 } else { self.rows };
        let output_bytes = mul(mul(output_rows, self.dv)?, 4)?;
        let staged_result_bytes = if self.branch == Branch::SplitKv {
            mul(
                mul(mul(output_rows, self.params.workers)?, add(self.dv, 2)?)?,
                4,
            )?
        } else {
            output_bytes
        };
        let half = if self.branch != Branch::Tiled && self.inputs.v.dtype == DType::F16 {
            mul(self.dv, 2)?
        } else {
            0
        };
        let packed_v = if self.branch == Branch::Tiled {
            mul(mul(KV_TILE, self.dv)?, 4)?
        } else {
            0
        };
        let each_scratch = add(add(mul(self.dv, 4)?, half)?, packed_v)?;
        let worker_scratch_bytes = mul(each_scratch, self.params.workers)?;
        let descriptor_bytes = mul(
            self.params.workers,
            add(
                add(size_of::<WorkerBuffers>(), size_of::<WorkerResult>())?,
                size_of::<ScopedJoinHandle<'_, WorkerResult>>(),
            )?,
        )?;
        let threaded = mode == ExecutionMode::ScopedWorkers && self.params.workers > 1;
        let requested_thread_stack_bytes = if threaded {
            mul(self.params.workers, WORKER_STACK_BYTES)?
        } else {
            0
        };
        let thread_metadata_allowance_bytes = if threaded {
            mul(self.params.workers, WORKER_METADATA_ALLOWANCE)?
        } else {
            0
        };
        let mut peak_bytes = 0;
        for bytes in [
            output_bytes,
            staged_result_bytes,
            worker_scratch_bytes,
            descriptor_bytes,
            requested_thread_stack_bytes,
            thread_metadata_allowance_bytes,
        ] {
            peak_bytes = add(peak_bytes, bytes)?;
        }
        // Conservative logical work units, not instruction or CPU-cycle
        // estimates. Input/accumulator visits, conversions and libm calls have
        // fixed accounting factors. Thirty-two items per
        // channel/column cover dot, online rescale, half conversion, tile packing,
        // and V accumulation. The complete source query tile can be rescanned
        // for every selected row, so mask work includes QUERY_TILE even when
        // Selection::Row requests only one result.
        let rounded_kv = mul(self.nkv.div_ceil(KV_TILE), KV_TILE)?;
        let per_column = mul(add(add(self.dk, self.dv)?, QUERY_TILE + 32)?, 32)?;
        let row_work = mul(mul(output_rows, rounded_kv)?, per_column)?;
        // execute preallocates every worker's complete buffers, even when a
        // selected row leaves that worker idle. In particular packed_v contains
        // KV_TILE * DV F32 elements for each tiled worker.
        let scratch_elements_per_worker = add(self.dv, add(half / 2, packed_v / 4)?)?;
        let initialized_elements = add(
            mul(self.params.workers, scratch_elements_per_worker)?,
            add(output_bytes, staged_result_bytes)? / 4,
        )?;
        let initialization = mul(initialized_elements, 2)?;
        let copies = mul(add(output_bytes, staged_result_bytes)? / 4, 8)?;
        let bookkeeping = mul(mul(self.params.workers, self.params.workers)?, 4)?;
        // Even a selected row validates every supplied row-map entry. Charge
        // this input scan before touching map values, including idle batches.
        let row_map_validation = mul(self.kv_rows.map_or(0, |rows| rows.len()), 4)?;
        let work = add(
            add(add(row_work, initialization)?, add(copies, bookkeeping)?)?,
            row_map_validation,
        )?;
        let scalar_work_items = u64::try_from(work).map_err(|_| Error::Overflow)?;
        Ok(MemoryRequirements {
            output_bytes,
            staged_result_bytes,
            worker_scratch_bytes,
            descriptor_bytes,
            requested_thread_stack_bytes,
            thread_metadata_allowance_bytes,
            peak_bytes,
            scalar_work_items,
        })
    }
    pub fn execute(&self, selection: Selection, limits: Limits) -> Result<Output> {
        self.execute_impl(
            selection,
            limits,
            ExecutionMode::ScopedWorkers,
            Faults::default(),
        )
    }
    /// Evaluate the same source worker partitions in the calling thread for
    /// diagnostics. The returned stats explicitly report one actual thread.
    pub fn execute_serial_partitions(
        &self,
        selection: Selection,
        limits: Limits,
    ) -> Result<Output> {
        self.execute_impl(
            selection,
            limits,
            ExecutionMode::SerialPartitions,
            Faults::default(),
        )
    }
    fn execute_impl(
        &self,
        selection: Selection,
        limits: Limits,
        mode: ExecutionMode,
        faults: Faults,
    ) -> Result<Output> {
        let selected = self.selected_row(selection)?;
        let requirements = self.requirements_for(selection, mode)?;
        if requirements.peak_bytes > limits.max_bytes {
            return Err(Error::ResourceLimit {
                required: requirements.peak_bytes,
                limit: limits.max_bytes,
            });
        }
        if requirements.scalar_work_items > limits.max_scalar_work {
            return Err(Error::WorkLimit {
                required: requirements.scalar_work_items,
                limit: limits.max_scalar_work,
            });
        }
        self.validate_rows()?;
        let mut output = zeros::<f32>(requirements.output_bytes / 4)?;
        let mut prepared = reserved(self.params.workers)?;
        for worker in 0..self.params.workers {
            let rows = self.worker_rows(worker, selected);
            let stride = self.dv + usize::from(self.branch == Branch::SplitKv) * 2;
            prepared.push(WorkerBuffers {
                worker,
                result: zeros(mul(rows, stride)?)?,
                accumulator: zeros(self.dv)?,
                half: zeros(
                    if self.branch != Branch::Tiled && self.inputs.v.dtype == DType::F16 {
                        self.dv
                    } else {
                        0
                    },
                )?,
                packed_v: zeros(if self.branch == Branch::Tiled {
                    mul(KV_TILE, self.dv)?
                } else {
                    0
                })?,
            });
        }
        let mut results = reserved::<WorkerResult>(self.params.workers)?;
        if mode == ExecutionMode::SerialPartitions || self.params.workers == 1 {
            for buffers in prepared {
                results.push(self.run_worker(buffers, selected));
            }
        } else {
            thread::scope(|scope| -> Result<()> {
                let mut handles = reserved(self.params.workers)?;
                let mut error = None;
                for buffers in prepared {
                    if faults.fail_spawn_at == Some(buffers.worker) {
                        error = Some(Error::ThreadSpawn(std::io::Error::other(
                            "injected worker spawn failure",
                        )));
                        break;
                    }
                    let spawn = thread::Builder::new()
                        .stack_size(WORKER_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            assert_ne!(
                                faults.panic_at,
                                Some(buffers.worker),
                                "injected worker panic"
                            );
                            self.run_worker(buffers, selected)
                        });
                    match spawn {
                        Ok(handle) => handles.push(handle),
                        Err(failure) => {
                            error = Some(Error::ThreadSpawn(failure));
                            break;
                        }
                    }
                }
                for handle in handles {
                    match handle.join() {
                        Ok(result) => results.push(result),
                        Err(_) => {
                            if error.is_none() {
                                error = Some(Error::WorkerPanic);
                            }
                        }
                    }
                }
                if let Some(failure) = error {
                    Err(failure)
                } else {
                    Ok(())
                }
            })?;
        }
        let worker_threads = results
            .iter()
            .enumerate()
            .filter(|(index, worker)| {
                !results[..*index]
                    .iter()
                    .any(|earlier| earlier.thread == worker.thread)
            })
            .count();
        if self.branch == Branch::SplitKv {
            self.reduce_partials(&results, selected, &mut output);
        } else {
            for result in &results {
                let mut offset = 0;
                self.for_worker_rows(result.worker, selected, |row| {
                    let destination = if selected.is_some() {
                        0
                    } else {
                        self.output_offset(row)
                    };
                    output[destination..destination + self.dv]
                        .copy_from_slice(&result.values[offset..offset + self.dv]);
                    offset += self.dv;
                });
                debug_assert_eq!(offset, result.values.len());
            }
        }
        Ok(Output {
            values: output,
            shape: self.output_shape(selection)?,
            branch: self.branch,
            stats: ExecutionStats {
                configured_workers: self.params.workers,
                completed_partitions: results.len(),
                worker_threads,
                mode,
            },
            requirements,
        })
    }
    fn for_worker_rows(
        &self,
        worker: usize,
        selected: Option<usize>,
        mut operation: impl FnMut(usize),
    ) {
        if let Some(row) = selected {
            if (row / self.rows_per_chunk) % self.params.workers == worker {
                operation(row);
            }
            return;
        }
        for chunk in (worker..self.query_chunks).step_by(self.params.workers) {
            let first = chunk * self.rows_per_chunk;
            let end = (first + self.rows_per_chunk).min(self.rows);
            for row in first..end {
                operation(row);
            }
        }
    }
    fn run_worker(&self, mut buffers: WorkerBuffers, selected: Option<usize>) -> WorkerResult {
        if self.branch == Branch::SplitKv {
            let chunk = self.nkv.div_ceil(self.params.workers);
            let begin = buffers.worker * chunk;
            let end = (begin + chunk).min(self.nkv);
            let count = if selected.is_some() { 1 } else { self.heads };
            for item in 0..count {
                let row = selected.unwrap_or(item);
                let (maximum, sum) = if begin < self.nkv {
                    self.one_chunk(row, begin, end, &mut buffers)
                } else {
                    buffers.accumulator.fill(0.0);
                    (f32::NEG_INFINITY, 0.0)
                };
                let offset = item * (self.dv + 2);
                buffers.result[offset] = maximum;
                buffers.result[offset + 1] = sum;
                buffers.result[offset + 2..offset + 2 + self.dv]
                    .copy_from_slice(&buffers.accumulator);
            }
        } else {
            let mut offset = 0;
            self.for_worker_rows(buffers.worker, selected, |row| {
                if self.branch == Branch::Tiled {
                    self.tiled_row(row, &mut buffers);
                } else {
                    let (_, sum) = self.one_chunk(row, 0, self.nkv, &mut buffers);
                    let inverse = if sum == 0.0 { 0.0 } else { 1.0 / sum };
                    source_scale(
                        &mut buffers.accumulator,
                        inverse,
                        self.params.compiler_policy,
                    );
                }
                buffers.result[offset..offset + self.dv].copy_from_slice(&buffers.accumulator);
                offset += self.dv;
            });
        }
        WorkerResult {
            worker: buffers.worker,
            values: buffers.result,
            thread: thread::current().id(),
        }
    }
    fn coordinates(&self, row: usize) -> (usize, usize, usize) {
        let batch = row / (self.nq * self.heads);
        let head = (row / self.nq) % self.heads;
        (row % self.nq, head, batch)
    }
    fn output_offset(&self, row: usize) -> usize {
        let (query, head, batch) = self.coordinates(row);
        ((batch * self.nq + query) * self.heads + head) * self.dv
    }
    fn slope(&self, head: usize) -> f32 {
        if self.params.max_bias <= 0.0 {
            1.0
        } else if head < self.head_power {
            self.slope0.powf((head + 1) as f32)
        } else {
            self.slope1.powf((2 * (head - self.head_power) + 1) as f32)
        }
    }
    fn mask_value(
        &self,
        column: usize,
        query: usize,
        head: usize,
        batch: usize,
        slope: f32,
    ) -> f32 {
        self.inputs.mask.map_or(0.0, |mask| {
            slope * mask.value([column, query, head % mask.shape[2], batch % mask.shape[3]])
        })
    }
    fn validate_rows(&self) -> Result<()> {
        if let Some(rows) = self.kv_rows {
            let physical_limit = self.inputs.k.shape[1].min(self.inputs.v.shape[1]);
            for batch in 0..rows.shape()[1] {
                for column in 0..rows.shape()[0] {
                    let index = rows.value(column, batch);
                    if index >= 0 && index as usize >= physical_limit {
                        return Err(Error::Invalid(
                            "nonnegative kv_rows index exceeds physical K/V storage",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    fn kv_row(&self, column: usize, batch: usize) -> i64 {
        self.kv_rows
            .map_or(column as i64, |rows| i64::from(rows.value(column, batch)))
    }
    fn kv_batch(&self, batch: usize) -> usize {
        if self.kv_rows.is_some() { 0 } else { batch }
    }
    fn score(&self, row: usize, column: usize, tiled: bool) -> f32 {
        let (query, head, batch) = self.coordinates(row);
        let k_head = head / (self.heads / self.inputs.k.shape[2]);
        let q = self.inputs.q;
        let k = self.inputs.k;
        let kv_batch = self.kv_batch(batch);
        let query_first = if tiled {
            self.params.compiler_policy == CompilerPolicy::Gnu
        } else {
            k.dtype == DType::F32 && self.params.compiler_policy == CompilerPolicy::Msvc
        };
        let product = |channel| {
            let query_value = q.value([channel, query, head, batch]);
            let key_value = k.value([channel, column, k_head, kv_batch]);
            if query_first {
                preferred_mul(query_value, key_value)
            } else {
                preferred_mul(key_value, query_value)
            }
        };
        let dot = if tiled {
            let mut sum = 0.0_f32;
            let source_prefix = self.dk / 4 * 4;
            for channel in 0..self.dk {
                let value = product(channel);
                // The captured MSVC score GEMM has a four-channel prefix,
                // with product-first additions at its block boundaries.
                // Its scalar tail and the GNU profile retain the accumulator.
                sum = if self.params.compiler_policy == CompilerPolicy::Msvc
                    && channel < source_prefix
                    && channel.is_multiple_of(4)
                {
                    preferred_add(value, sum)
                } else {
                    preferred_add(sum, value)
                };
            }
            sum
        } else {
            let mut sum = 0.0_f64;
            for channel in 0..self.dk {
                sum = preferred_add_f64(sum, product(channel) as f64);
            }
            sum as f32
        };
        let scaled = dot * self.scale;
        if self.params.logit_softcap != 0.0 {
            self.params.logit_softcap * scaled.tanh()
        } else {
            scaled
        }
    }
    fn one_chunk(
        &self,
        row: usize,
        begin: usize,
        end: usize,
        buffers: &mut WorkerBuffers,
    ) -> (f32, f32) {
        let (query, head, batch) = self.coordinates(row);
        let slope = self.slope(head);
        let v_head = head / (self.heads / self.inputs.v.shape[2]);
        let half = self.inputs.v.dtype == DType::F16;
        let kv_batch = self.kv_batch(batch);
        // Native scratch captures distinguish the compiler's partial-writing
        // specialization from its normalized GNU one-chunk specialization.
        let sum_policy = if self.branch == Branch::SplitKv {
            CompilerPolicy::Msvc
        } else {
            self.params.compiler_policy
        };
        buffers.accumulator.fill(0.0);
        buffers.half.fill(0);
        let mut maximum = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        for column in begin..end {
            let mask = self.mask_value(column, query, head, batch, slope);
            if mask == f32::NEG_INFINITY {
                continue;
            }
            let physical_row = self.kv_row(column, batch);
            if physical_row < 0 {
                continue;
            }
            let physical_row = physical_row as usize;
            let score = match self.params.compiler_policy {
                CompilerPolicy::Msvc => preferred_add(self.score(row, physical_row, false), mask),
                CompilerPolicy::Gnu => preferred_add(mask, self.score(row, physical_row, false)),
            };
            let old_maximum = maximum;
            let mut rescale = 1.0;
            let mut probability = 1.0;
            if score > maximum {
                maximum = score;
                rescale = (old_maximum - maximum).exp();
                if half {
                    for value in &mut buffers.half {
                        *value = f32_to_f16(f16_to_f32(*value) * rescale);
                    }
                } else {
                    source_scale(
                        &mut buffers.accumulator,
                        rescale,
                        self.params.compiler_policy,
                    );
                }
            } else {
                probability = (score - maximum).exp();
            }
            for channel in 0..self.dv {
                let value = self
                    .inputs
                    .v
                    .value([channel, physical_row, v_head, kv_batch]);
                let product = match self.params.compiler_policy {
                    CompilerPolicy::Msvc => preferred_mul(probability, value),
                    CompilerPolicy::Gnu => preferred_mul(value, probability),
                };
                if half {
                    buffers.half[channel] =
                        f32_to_f16(preferred_add(product, f16_to_f32(buffers.half[channel])));
                } else {
                    buffers.accumulator[channel] =
                        preferred_add(product, buffers.accumulator[channel]);
                }
            }
            sum = source_sum(sum, rescale, probability, sum_policy);
        }
        if half {
            for (value, &bits) in buffers.accumulator.iter_mut().zip(&buffers.half) {
                *value = f16_to_f32(bits);
            }
        }
        if begin == 0
            && let Some(sinks) = self.inputs.sinks
        {
            let sink = sinks.value([head, 0, 0, 0]);
            let mut rescale = 1.0;
            let mut probability = 1.0;
            if sink > maximum {
                rescale = (maximum - sink).exp();
                maximum = sink;
                source_scale(
                    &mut buffers.accumulator,
                    rescale,
                    self.params.compiler_policy,
                );
            } else {
                probability = (sink - maximum).exp();
            }
            sum = source_sum(sum, rescale, probability, sum_policy);
        }
        (maximum, sum)
    }
    fn source_query_tile(&self, row: usize) -> (usize, usize) {
        let chunk = row / self.rows_per_chunk;
        let chunk_begin = chunk * self.rows_per_chunk;
        let chunk_end = (chunk_begin + self.rows_per_chunk).min(self.rows);
        let head_begin = row / self.nq * self.nq;
        let segment_begin = chunk_begin.max(head_begin);
        let tile_begin = segment_begin + (row - segment_begin) / QUERY_TILE * QUERY_TILE;
        let tile_end = (tile_begin + QUERY_TILE)
            .min(chunk_end)
            .min(head_begin + self.nq);
        (tile_begin, tile_end)
    }
    fn tiled_row(&self, row: usize, buffers: &mut WorkerBuffers) {
        let (query, head, batch) = self.coordinates(row);
        let slope = self.slope(head);
        let v_head = head / (self.heads / self.inputs.v.shape[2]);
        let (tile_begin, tile_end) = self.source_query_tile(row);
        let kv_batch = self.kv_batch(batch);
        buffers.accumulator.fill(0.0);
        buffers.packed_v.fill(0.0);
        let mut maximum = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        for begin in (0..self.nkv).step_by(KV_TILE) {
            let count = (self.nkv - begin).min(KV_TILE);
            if self.inputs.mask.is_some() {
                let can_skip = (tile_begin..tile_end).all(|tile_row| {
                    let (tile_query, _, _) = self.coordinates(tile_row);
                    (0..count).all(|offset| {
                        self.mask_value(begin + offset, tile_query, head, batch, slope)
                            == f32::NEG_INFINITY
                    })
                });
                if can_skip {
                    continue;
                }
            }
            let mut scores = [f32::NEG_INFINITY; KV_TILE];
            for (offset, score) in scores[..count].iter_mut().enumerate() {
                // The source tiled gather reads physical row zero for negative
                // padding entries; masking does not erase nonfinite V payloads.
                let physical_row = self.kv_row(begin + offset, batch).max(0) as usize;
                *score = self.score(row, physical_row, true);
            }
            if self.inputs.mask.is_some() {
                for (offset, score) in scores[..count].iter_mut().enumerate() {
                    *score = preferred_add(
                        self.mask_value(begin + offset, query, head, batch, slope),
                        *score,
                    );
                }
                // The source adds padded -Inf to already padded -Inf.
                for score in &mut scores[count..] {
                    *score += f32::NEG_INFINITY;
                }
            }
            let mut tile_maximum = f32::NEG_INFINITY;
            for &score in &scores {
                tile_maximum = if tile_maximum > score {
                    tile_maximum
                } else {
                    score
                };
            }
            let skip = tile_maximum == f32::NEG_INFINITY;
            if !skip {
                let next = maximum.max(tile_maximum);
                if next > maximum {
                    let rescale = (maximum - next).exp();
                    source_scale(
                        &mut buffers.accumulator,
                        rescale,
                        self.params.compiler_policy,
                    );
                    sum *= rescale;
                }
                maximum = next;
                let tile_sum =
                    softmax_exp_sum_f32_with_policy(&mut scores, next, self.params.math_policy);
                // Original optimized C operand order chooses the tile's NaN on
                // MSVC and the retained sum's NaN on GNU. Make that explicit in
                // both Rust build profiles without changing finite rounding.
                sum = match self.params.compiler_policy {
                    CompilerPolicy::Msvc => preferred_add_f64(tile_sum, sum as f64),
                    CompilerPolicy::Gnu => preferred_add_f64(sum as f64, tile_sum),
                } as f32;
            }
            // The original V scratch is zeroed once per query tile, not again
            // for its final partial KV tile. Preserve retained lanes.
            for offset in 0..count {
                let physical_row = self.kv_row(begin + offset, batch).max(0) as usize;
                for channel in 0..self.dv {
                    buffers.packed_v[offset * self.dv + channel] =
                        self.inputs
                            .v
                            .value([channel, physical_row, v_head, kv_batch]);
                }
            }
            if skip {
                scores.fill(0.0);
            }
            for channel in 0..self.dv {
                let mut value = buffers.accumulator[channel];
                for (offset, &probability) in scores.iter().enumerate() {
                    let product =
                        preferred_mul(probability, buffers.packed_v[offset * self.dv + channel]);
                    // The captured MSVC scalar GEMM places the product first
                    // at each 16-element unrolled block; the other iterations
                    // retain the accumulator's NaN. Original operand-position
                    // sweeps cover every offset for dense and row-mapped graphs.
                    value = if offset.is_multiple_of(16)
                        && self.params.compiler_policy == CompilerPolicy::Msvc
                    {
                        preferred_add(product, value)
                    } else {
                        preferred_add(value, product)
                    };
                }
                buffers.accumulator[channel] = value;
            }
        }
        if let Some(sinks) = self.inputs.sinks {
            let sink = sinks.value([head, 0, 0, 0]);
            let mut rescale = 1.0;
            let mut probability = 1.0;
            if sink > maximum {
                rescale = (maximum - sink).exp();
                source_scale(
                    &mut buffers.accumulator,
                    rescale,
                    self.params.compiler_policy,
                );
            } else {
                probability = (sink - maximum).exp();
            }
            sum = source_sum(sum, rescale, probability, self.params.compiler_policy);
        }
        let inverse = if sum == 0.0 { 0.0 } else { 1.0 / sum };
        source_tile_or_split_scale(
            &mut buffers.accumulator,
            inverse,
            self.params.compiler_policy,
        );
    }
    fn reduce_partials(
        &self,
        results: &[WorkerResult],
        selected: Option<usize>,
        output: &mut [f32],
    ) {
        let count = if selected.is_some() { 1 } else { self.heads };
        let chunk_size = self.nkv.div_ceil(self.params.workers);
        for head in 0..count {
            let accumulator = &mut output[head * self.dv..(head + 1) * self.dv];
            let mut maximum = f32::NEG_INFINITY;
            let mut sum = 0.0_f32;
            for worker in results {
                if worker.worker * chunk_size >= self.nkv {
                    continue;
                }
                let at = head * (self.dv + 2);
                let chunk_maximum = worker.values[at];
                let chunk_sum = worker.values[at + 1];
                if chunk_sum == 0.0 {
                    continue;
                }
                let next = maximum.max(chunk_maximum);
                let old_scale = (maximum - next).exp();
                let new_scale = (chunk_maximum - next).exp();
                // The MSVC reducer is auto-vectorized in sixteen-element
                // groups; GNU uses four-element groups plus a two-element
                // remainder. Their scalar tails put each scale operand first.
                let lane_group = match self.params.compiler_policy {
                    CompilerPolicy::Msvc => 16,
                    CompilerPolicy::Gnu => 2,
                };
                let prefix = self.dv / lane_group * lane_group;
                for (channel, value) in accumulator.iter_mut().enumerate() {
                    let chunk_value = worker.values[at + 2 + channel];
                    let (old, new) = if channel < prefix {
                        (
                            preferred_mul(*value, old_scale),
                            preferred_mul(chunk_value, new_scale),
                        )
                    } else {
                        (
                            preferred_mul(old_scale, *value),
                            preferred_mul(new_scale, chunk_value),
                        )
                    };
                    *value = preferred_add(new, old);
                }
                let old_sum = preferred_mul(old_scale, sum);
                let new_sum = preferred_mul(new_scale, chunk_sum);
                sum = match self.params.compiler_policy {
                    CompilerPolicy::Msvc => preferred_add(new_sum, old_sum),
                    CompilerPolicy::Gnu => preferred_add(old_sum, new_sum),
                };
                maximum = next;
            }
            if sum != 0.0 {
                let inverse = 1.0 / sum;
                source_tile_or_split_scale(accumulator, inverse, self.params.compiler_policy);
            }
        }
    }
}
#[derive(Default)]
struct WorkerBuffers {
    worker: usize,
    result: Vec<f32>,
    accumulator: Vec<f32>,
    half: Vec<u16>,
    packed_v: Vec<f32>,
}
struct WorkerResult {
    worker: usize,
    values: Vec<f32>,
    thread: ThreadId,
}
#[derive(Clone, Copy, Default)]
struct Faults {
    fail_spawn_at: Option<usize>,
    panic_at: Option<usize>,
}

#[cfg(test)]
mod rows_tests;
#[cfg(test)]
mod tests;
