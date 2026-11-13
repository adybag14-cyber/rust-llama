//! Persistent, single-sequence dense Llama execution using Rust-authored CUDA
//! kernels. CPU work is limited to loading, fixed RoPE tables and token handling;
//! embeddings, projections, attention, norms and feed-forward execute on device.
#![forbid(unsafe_code)]
use llama_device_family::{
    DeviceError, Family, ProbeStatus,
    driver::{CudaBuffer, CudaContext, CudaDriver, CudaKernel, CudaStream},
    probe,
    ptx::{self, DecoderKernel as D, KernelKind, KernelSpec, PtxTarget},
};
use llama_model_runtime::{LlamaConfig, LlamaModel, Matrix};
use llama_quant::DType;
use std::{fmt, rc::Rc};

#[derive(Debug)]
pub enum Error {
    Device(DeviceError),
    Invalid(String),
    Unsupported(String),
    Cancelled,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(e) => write!(f, "{e}"),
            Self::Invalid(s) => write!(f, "invalid CUDA decoder input: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported CUDA decoder configuration: {s}"),
            Self::Cancelled => f.write_str("CUDA decoder cancelled"),
        }
    }
}
impl std::error::Error for Error {}
impl From<DeviceError> for Error {
    fn from(e: DeviceError) -> Self {
        Self::Device(e)
    }
}
impl From<ptx::KernelError> for Error {
    fn from(e: ptx::KernelError) -> Self {
        Self::Invalid(e.to_string())
    }
}
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Options {
    pub device: i32,
    /// Sum of requested CUDA-buffer lengths, not a total VRAM ceiling. Driver
    /// allocation granularity, module/context overhead and the caller's CPU
    /// model storage are additional. Allocation failures return an error.
    pub max_device_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            max_device_bytes: usize::try_from(8u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX),
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Statistics {
    /// Requested lengths of live CUDA buffers, excluding driver overhead.
    pub allocated_bytes: usize,
    pub uploaded_bytes: usize,
    pub kernel_dispatches: u64,
    pub committed_tokens: usize,
}
#[derive(Debug, Clone)]
pub struct Output {
    pub logits: Vec<f32>,
    pub embedding: Vec<f32>,
}
struct DeviceMatrix {
    weights: Rc<CudaBuffer>,
    kernel: usize,
    activation: Option<usize>,
}
struct Layer {
    attention_norm: CudaBuffer,
    q: DeviceMatrix,
    k: DeviceMatrix,
    v: DeviceMatrix,
    o: DeviceMatrix,
    q_bias: Option<CudaBuffer>,
    k_bias: Option<CudaBuffer>,
    v_bias: Option<CudaBuffer>,
    o_bias: Option<CudaBuffer>,
    ffn_norm: CudaBuffer,
    gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
    gate_bias: Option<CudaBuffer>,
    up_bias: Option<CudaBuffer>,
    down_bias: Option<CudaBuffer>,
    keys: CudaBuffer,
    values: CudaBuffer,
}
struct Workspace {
    hidden: CudaBuffer,
    residual: CudaBuffer,
    norm: CudaBuffer,
    q: CudaBuffer,
    q_tmp: CudaBuffer,
    k: CudaBuffer,
    k_tmp: CudaBuffer,
    v: CudaBuffer,
    v_tmp: CudaBuffer,
    attention: CudaBuffer,
    projection: CudaBuffer,
    projection_tmp: CudaBuffer,
    ffn_norm: CudaBuffer,
    gate: CudaBuffer,
    gate_tmp: CudaBuffer,
    up: CudaBuffer,
    up_tmp: CudaBuffer,
    product: CudaBuffer,
    down: CudaBuffer,
    down_tmp: CudaBuffer,
    embedding: CudaBuffer,
    logits: CudaBuffer,
    scale: CudaBuffer,
    scores: CudaBuffer,
    softmax_stats: CudaBuffer,
    probabilities: CudaBuffer,
    uniform: CudaBuffer,
    quantized: CudaBuffer,
}
struct Graph {
    embedding: usize,
    rms_scale: usize,
    rms_apply: usize,
    rope_q: usize,
    rope_k: usize,
    cache: usize,
    scores: usize,
    softmax_stats: usize,
    softmax_apply: usize,
    attention: usize,
    silu: usize,
    add_width: usize,
    add_kv: usize,
    add_ff: usize,
}
struct Allocator<'a> {
    context: &'a CudaContext,
    maximum: usize,
    allocated: usize,
    uploaded: usize,
}
fn bytes(n: usize) -> Result<usize> {
    n.checked_mul(4)
        .ok_or_else(|| Error::Invalid("buffer byte size overflow".into()))
}
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Invalid("buffer element size overflow".into()))
}
fn dim(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Invalid("dimension exceeds CUDA addressing".into()))
}
impl Allocator<'_> {
    fn allocate(&mut self, n: usize) -> Result<CudaBuffer> {
        let total = self
            .allocated
            .checked_add(n)
            .filter(|n| *n <= self.maximum)
            .ok_or_else(|| Error::Invalid("owned device allocation budget exceeded".into()))?;
        let buffer = self.context.allocate(n)?;
        self.allocated = total;
        Ok(buffer)
    }
    fn f32(&mut self, n: usize) -> Result<CudaBuffer> {
        self.allocate(bytes(n)?)
    }
    fn upload(&mut self, data: &[f32]) -> Result<CudaBuffer> {
        let mut buffer = self.f32(data.len())?;
        // Bounded conversion staging; large matrices are never duplicated in
        // a second host-sized byte vector just to cross the driver boundary.
        for (part, chunk) in data.chunks(262_144).enumerate() {
            let value: Vec<u8> = chunk.iter().flat_map(|x| x.to_le_bytes()).collect();
            buffer.copy_from(part * 1_048_576, &value)?;
        }
        self.uploaded += buffer.len();
        Ok(buffer)
    }
    fn raw(&mut self, data: &[u8]) -> Result<CudaBuffer> {
        let mut buffer = self.allocate(data.len())?;
        buffer.copy_from(0, data)?;
        self.uploaded += data.len();
        Ok(buffer)
    }
    fn bias(&mut self, data: &Option<Vec<f32>>) -> Result<Option<CudaBuffer>> {
        data.as_deref().map(|x| self.upload(x)).transpose()
    }
}
struct Compiler<'a> {
    context: &'a CudaContext,
    kernels: Vec<(KernelKind, CudaKernel)>,
}
impl Compiler<'_> {
    fn kernel(&mut self, kind: KernelKind) -> Result<usize> {
        if let Some(i) = self.kernels.iter().position(|(k, _)| *k == kind) {
            return Ok(i);
        }
        let generated = ptx::generate(&KernelSpec { kind: kind.clone() }, PtxTarget::default())?;
        let compiled = self.context.compile_kernel(&generated)?;
        let i = self.kernels.len();
        self.kernels.push((kind, compiled));
        Ok(i)
    }
    fn decoder(&mut self, kind: D) -> Result<usize> {
        self.kernel(KernelKind::Decoder(kind))
    }
    fn add(&mut self, elements: u32) -> Result<usize> {
        self.kernel(KernelKind::AddF32 {
            elements,
            lhs_stride: 1,
            rhs_stride: 1,
            output_stride: 1,
        })
    }
    fn matrix(&mut self, a: &mut Allocator<'_>, m: &Matrix) -> Result<DeviceMatrix> {
        let (rows, columns) = (dim(m.rows())?, dim(m.cols())?);
        let kind = matrix_kind(m)?;
        let kernel = self.kernel(kind)?;
        let weights = match m.packed_bytes() {
            None => a.upload(m.values())?,
            Some((DType::F16 | DType::Q4_0 | DType::Q8_0, data)) => a.raw(data)?,
            Some((dtype, _)) => return Err(Error::Unsupported(format!("matrix dtype {dtype:?}"))),
        };
        debug_assert!(rows > 0 && columns > 0);
        let activation = if matches!(m.storage_dtype(), DType::Q4_0 | DType::Q8_0) {
            Some(self.decoder(D::ActivationQ8 { columns })?)
        } else {
            None
        };
        Ok(DeviceMatrix {
            weights: Rc::new(weights),
            kernel,
            activation,
        })
    }
}
fn matrix_kind(m: &Matrix) -> Result<KernelKind> {
    let (rows, columns) = (dim(m.rows())?, dim(m.cols())?);
    let kind = match m.storage_dtype() {
        DType::F32 | DType::F16 => KernelKind::Decoder(D::MatVec {
            rows,
            columns,
            half: m.storage_dtype() == DType::F16,
        }),
        DType::Q4_0 | DType::Q8_0 => KernelKind::Decoder(D::QuantizedDot {
            rows,
            columns,
            q4: m.storage_dtype() == DType::Q4_0,
        }),
        dtype => {
            return Err(Error::Unsupported(format!(
                "matrix dtype {dtype:?}; implemented F32, F16, Q4_0 and Q8_0"
            )));
        }
    };
    KernelSpec { kind: kind.clone() }.buffer_bytes()?;
    Ok(kind)
}
fn run(
    kernels: &[(KernelKind, CudaKernel)],
    id: usize,
    stream: &mut CudaStream,
    stats: &mut Statistics,
    inputs: &[&CudaBuffer],
    output: &mut CudaBuffer,
) -> Result<()> {
    kernels[id].1.run(stream, inputs, output)?;
    stats.kernel_dispatches = stats.kernel_dispatches.saturating_add(1);
    Ok(())
}
fn project(
    kernels: &[(KernelKind, CudaKernel)],
    m: &DeviceMatrix,
    stream: &mut CudaStream,
    stats: &mut Statistics,
    (input, output, activation): (&CudaBuffer, &mut CudaBuffer, &mut CudaBuffer),
) -> Result<()> {
    if let Some(id) = m.activation {
        run(kernels, id, stream, stats, &[input], activation)?;
        run(
            kernels,
            m.kernel,
            stream,
            stats,
            &[&m.weights, activation],
            output,
        )
    } else {
        run(
            kernels,
            m.kernel,
            stream,
            stats,
            &[&m.weights, input],
            output,
        )
    }
}
fn bias(
    kernels: &[(KernelKind, CudaKernel)],
    add: usize,
    stream: &mut CudaStream,
    stats: &mut Statistics,
    bias: &Option<CudaBuffer>,
    value: &mut CudaBuffer,
    temp: &mut CudaBuffer,
) -> Result<()> {
    if let Some(b) = bias {
        run(kernels, add, stream, stats, &[value, b], temp)?;
        std::mem::swap(value, temp);
    }
    Ok(())
}
fn read(buffer: &CudaBuffer) -> Result<Vec<f32>> {
    let mut data = vec![0; buffer.len()];
    buffer.copy_to(0, &mut data)?;
    let out: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    if out.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("nonfinite device output".into()));
    }
    Ok(out)
}

fn allocation_plan(model: &LlamaModel) -> Result<usize> {
    let c = model.config();
    let kv = c.kv_width().map_err(|e| Error::Invalid(e.to_string()))?;
    let mut total = 0usize;
    let mut add = |n: usize| -> Result<()> {
        total = total
            .checked_add(n)
            .ok_or_else(|| Error::Invalid("allocation plan overflow".into()))?;
        Ok(())
    };
    let matrix_bytes = |m: &Matrix| -> Result<usize> {
        matrix_kind(m)?;
        match m.packed_bytes() {
            Some((_, data)) => Ok(data.len()),
            None => bytes(m.values().len()),
        }
    };
    KernelSpec {
        kind: KernelKind::Decoder(D::Embedding {
            vocabulary: dim(c.vocabulary)?,
            width: dim(c.width)?,
        }),
    }
    .buffer_bytes()?;
    add(bytes(model.weights().token_embedding.values().len())?)?;
    if let Some(output) = &model.weights().output {
        add(matrix_bytes(output)?)?;
    } else if model.weights().token_embedding.storage_dtype() != DType::F32 {
        add(matrix_bytes(&model.weights().token_embedding)?)?;
    }
    add(bytes(c.width)?)?;
    let dimension = c.head_dim().map_err(|e| Error::Invalid(e.to_string()))?;
    KernelSpec {
        kind: KernelKind::Decoder(D::Rope {
            width: dim(c.width)?,
            head_dimension: dim(dimension)?,
            context: dim(c.max_context)?,
        }),
    }
    .buffer_bytes()?;
    // Validate the output-score and KV extents before any host phase allocation.
    KernelSpec {
        kind: KernelKind::Decoder(D::AttentionScores {
            heads: dim(c.heads)?,
            kv_heads: dim(c.kv_heads)?,
            dimension: dim(dimension)?,
            context: dim(c.max_context)?,
            scale: 1.0,
        }),
    }
    .buffer_bytes()?;
    add(bytes(product(c.max_context, dimension)?)?)?;
    for layer in &model.weights().layers {
        add(bytes(product(c.width, 2)?)?)?;
        for m in [
            &layer.query,
            &layer.key,
            &layer.value,
            &layer.attention_output,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            add(matrix_bytes(m)?)?;
        }
        for b in [
            &layer.query_bias,
            &layer.key_bias,
            &layer.value_bias,
            &layer.attention_output_bias,
            &layer.gate_bias,
            &layer.up_bias,
            &layer.down_bias,
        ]
        .into_iter()
        .flatten()
        {
            add(bytes(b.len())?)?;
        }
        add(bytes(product(product(kv, c.max_context)?, 2)?)?)?;
    }
    for n in [
        product(c.width, 12)?,
        product(kv, 4)?,
        product(c.feed_forward, 5)?,
        c.vocabulary,
        1,
        product(product(c.heads, c.max_context)?, 2)?,
        product(c.heads, 2)?,
    ] {
        add(bytes(n)?)?;
    }
    add(8)?;
    add(bytes(product(
        c.width.max(c.feed_forward).div_ceil(32),
        33,
    )?)?)?;
    Ok(total)
}

/// Single sequence with append/truncate/reset. Cancelled or failed steps do not
/// advance the committed position. Partial writes are confined to that unused
/// cache row and are overwritten by the next complete step.
pub struct CudaLlama {
    config: LlamaConfig,
    device_name: String,
    token_embedding: Rc<CudaBuffer>,
    output: DeviceMatrix,
    output_norm: CudaBuffer,
    phases: CudaBuffer,
    layers: Vec<Layer>,
    kernels: Vec<(KernelKind, CudaKernel)>,
    graph: Graph,
    workspace: Workspace,
    stream: CudaStream,
    statistics: Statistics,
    position: usize,
}
impl CudaLlama {
    pub fn new(model: &LlamaModel, options: &Options) -> Result<Self> {
        if options.device < 0 || options.max_device_bytes == 0 {
            return Err(Error::Invalid(
                "invalid device or zero device budget".into(),
            ));
        }
        let planned = allocation_plan(model)?;
        if planned > options.max_device_bytes {
            return Err(Error::Invalid(format!(
                "owned device allocation budget exceeded: requires {planned} bytes, limit {}",
                options.max_device_bytes
            )));
        }
        let c = model.config().clone();
        let (width, kv, ff, ctx, heads, kv_heads, head_dimension, vocabulary) = (
            dim(c.width)?,
            dim(c.kv_width().map_err(|e| Error::Invalid(e.to_string()))?)?,
            dim(c.feed_forward)?,
            dim(c.max_context)?,
            dim(c.heads)?,
            dim(c.kv_heads)?,
            dim(c.head_dim().map_err(|e| Error::Invalid(e.to_string()))?)?,
            dim(c.vocabulary)?,
        );
        // Validate every weight's representation before opening or allocating a
        // device. An unsupported packed type never silently becomes F32 math.
        matrix_kind(&model.weights().token_embedding)?;
        if let Some(output) = &model.weights().output {
            matrix_kind(output)?;
        }
        for layer in &model.weights().layers {
            for m in [
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.attention_output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                matrix_kind(m)?;
            }
        }
        let report = probe(Family::Cuda);
        if report.status != ProbeStatus::Available {
            return Err(Error::Device(DeviceError::DriverUnavailable(report.detail)));
        }
        let device = report
            .devices
            .iter()
            .find(|d| d.ordinal == options.device)
            .ok_or_else(|| Error::Invalid("CUDA device ordinal is unavailable".into()))?;
        if device.compute_capability.is_none_or(|cc| cc < (6, 0)) {
            return Err(Error::Unsupported(
                "requires NVIDIA compute capability 6.0 or later".into(),
            ));
        }
        let device_name = device.name.clone();
        let context = CudaDriver::load()?.context(options.device)?;
        let mut allocator = Allocator {
            context: &context,
            maximum: options.max_device_bytes,
            allocated: 0,
            uploaded: 0,
        };
        let mut compiler = Compiler {
            context: &context,
            kernels: Vec::new(),
        };
        let (token_embedding, output) = if let Some(output) = &model.weights().output {
            let embedding = Rc::new(allocator.upload(model.weights().token_embedding.values())?);
            (embedding, compiler.matrix(&mut allocator, output)?)
        } else {
            let matrix = compiler.matrix(&mut allocator, &model.weights().token_embedding)?;
            let embedding = if model.weights().token_embedding.storage_dtype() == DType::F32 {
                matrix.weights.clone()
            } else {
                Rc::new(allocator.upload(model.weights().token_embedding.values())?)
            };
            (embedding, matrix)
        };
        let output_norm = allocator.upload(&model.weights().output_norm)?;
        let mut phases = Vec::new();
        phases
            .try_reserve_exact(product(c.max_context, head_dimension as usize)?)
            .map_err(|_| Error::Invalid("RoPE phase allocation failed".into()))?;
        let theta_scale = c.rope_base.powf(-2.0 / head_dimension as f32);
        for position in 0..c.max_context {
            let mut theta = position as f32;
            for _ in 0..head_dimension / 2 {
                let (sin, cos) = theta.sin_cos();
                phases.extend_from_slice(&[cos, sin]);
                theta *= theta_scale;
            }
        }
        let phases = allocator.upload(&phases)?;
        let mut layers = Vec::new();
        for layer in &model.weights().layers {
            layers.push(Layer {
                attention_norm: allocator.upload(&layer.attention_norm)?,
                q: compiler.matrix(&mut allocator, &layer.query)?,
                k: compiler.matrix(&mut allocator, &layer.key)?,
                v: compiler.matrix(&mut allocator, &layer.value)?,
                o: compiler.matrix(&mut allocator, &layer.attention_output)?,
                q_bias: allocator.bias(&layer.query_bias)?,
                k_bias: allocator.bias(&layer.key_bias)?,
                v_bias: allocator.bias(&layer.value_bias)?,
                o_bias: allocator.bias(&layer.attention_output_bias)?,
                ffn_norm: allocator.upload(&layer.feed_forward_norm)?,
                gate: compiler.matrix(&mut allocator, &layer.gate)?,
                up: compiler.matrix(&mut allocator, &layer.up)?,
                down: compiler.matrix(&mut allocator, &layer.down)?,
                gate_bias: allocator.bias(&layer.gate_bias)?,
                up_bias: allocator.bias(&layer.up_bias)?,
                down_bias: allocator.bias(&layer.down_bias)?,
                keys: allocator.f32(product(kv as usize, c.max_context)?)?,
                values: allocator.f32(product(kv as usize, c.max_context)?)?,
            });
        }
        let graph = Graph {
            embedding: compiler.decoder(D::Embedding { vocabulary, width })?,
            rms_scale: compiler.decoder(D::RmsScale {
                width,
                epsilon: c.rms_epsilon,
            })?,
            rms_apply: compiler.decoder(D::RmsApply { width })?,
            rope_q: compiler.decoder(D::Rope {
                width,
                head_dimension,
                context: ctx,
            })?,
            rope_k: compiler.decoder(D::Rope {
                width: kv,
                head_dimension,
                context: ctx,
            })?,
            cache: compiler.decoder(D::CacheWrite {
                width: kv,
                context: ctx,
            })?,
            scores: compiler.decoder(D::AttentionScores {
                heads,
                kv_heads,
                dimension: head_dimension,
                context: ctx,
                scale: c
                    .attention_scale
                    .unwrap_or_else(|| (head_dimension as f32).sqrt().recip()),
            })?,
            softmax_stats: compiler.decoder(D::SoftmaxStats {
                heads,
                context: ctx,
            })?,
            softmax_apply: compiler.decoder(D::SoftmaxApply {
                heads,
                context: ctx,
            })?,
            attention: compiler.decoder(D::AttentionValues {
                heads,
                kv_heads,
                dimension: head_dimension,
                context: ctx,
            })?,
            silu: compiler.decoder(D::SiluGate { width: ff })?,
            add_width: compiler.add(width)?,
            add_kv: compiler.add(kv)?,
            add_ff: compiler.add(ff)?,
        };
        let workspace = Workspace {
            hidden: allocator.f32(c.width)?,
            residual: allocator.f32(c.width)?,
            norm: allocator.f32(c.width)?,
            q: allocator.f32(c.width)?,
            q_tmp: allocator.f32(c.width)?,
            k: allocator.f32(kv as usize)?,
            k_tmp: allocator.f32(kv as usize)?,
            v: allocator.f32(kv as usize)?,
            v_tmp: allocator.f32(kv as usize)?,
            attention: allocator.f32(c.width)?,
            projection: allocator.f32(c.width)?,
            projection_tmp: allocator.f32(c.width)?,
            ffn_norm: allocator.f32(c.width)?,
            gate: allocator.f32(c.feed_forward)?,
            gate_tmp: allocator.f32(c.feed_forward)?,
            up: allocator.f32(c.feed_forward)?,
            up_tmp: allocator.f32(c.feed_forward)?,
            product: allocator.f32(c.feed_forward)?,
            down: allocator.f32(c.width)?,
            down_tmp: allocator.f32(c.width)?,
            embedding: allocator.f32(c.width)?,
            logits: allocator.f32(c.vocabulary)?,
            scale: allocator.f32(1)?,
            scores: allocator.f32(product(c.heads, c.max_context)?)?,
            softmax_stats: allocator.f32(product(c.heads, 2)?)?,
            probabilities: allocator.f32(product(c.heads, c.max_context)?)?,
            uniform: allocator.allocate(8)?,
            quantized: allocator.f32(product(c.width.max(c.feed_forward).div_ceil(32), 33)?)?,
        };
        debug_assert_eq!(allocator.allocated, planned);
        let statistics = Statistics {
            allocated_bytes: allocator.allocated,
            uploaded_bytes: allocator.uploaded,
            ..Statistics::default()
        };
        Ok(Self {
            config: c,
            device_name,
            token_embedding,
            output,
            output_norm,
            phases,
            layers,
            kernels: compiler.kernels,
            graph,
            workspace,
            stream: context.stream()?,
            statistics,
            position: 0,
        })
    }
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn statistics(&self) -> Statistics {
        self.statistics
    }
    pub fn context_length(&self) -> usize {
        self.config.max_context
    }
    pub fn reset(&mut self) {
        self.position = 0;
    }
    pub fn truncate(&mut self, position: usize) -> Result<()> {
        if position > self.position {
            return Err(Error::Invalid(
                "cannot truncate beyond committed tokens".into(),
            ));
        }
        self.position = position;
        Ok(())
    }
    pub fn decode(&mut self, token: u32) -> Result<Output> {
        self.decode_with_cancel(token, &|| false)
    }
    pub fn decode_with_cancel(&mut self, token: u32, cancel: &dyn Fn() -> bool) -> Result<Output> {
        if cancel() {
            return Err(Error::Cancelled);
        }
        if token as usize >= self.config.vocabulary {
            return Err(Error::Invalid("token outside vocabulary".into()));
        }
        if self.position >= self.config.max_context {
            return Err(Error::Invalid("context is full".into()));
        }
        let mut params = [0u8; 8];
        params[..4].copy_from_slice(&token.to_le_bytes());
        params[4..].copy_from_slice(&dim(self.position)?.to_le_bytes());
        let (w, g, k, s, stats) = (
            &mut self.workspace,
            &self.graph,
            &self.kernels,
            &mut self.stream,
            &mut self.statistics,
        );
        w.uniform.copy_from(0, &params)?;
        run(
            k,
            g.embedding,
            s,
            stats,
            &[&self.token_embedding, &w.uniform],
            &mut w.hidden,
        )?;
        for layer in &mut self.layers {
            if cancel() {
                return Err(Error::Cancelled);
            }
            run(k, g.rms_scale, s, stats, &[&w.hidden], &mut w.scale)?;
            run(
                k,
                g.rms_apply,
                s,
                stats,
                &[&w.hidden, &layer.attention_norm, &w.scale],
                &mut w.norm,
            )?;
            project(k, &layer.q, s, stats, (&w.norm, &mut w.q, &mut w.quantized))?;
            project(k, &layer.k, s, stats, (&w.norm, &mut w.k, &mut w.quantized))?;
            project(k, &layer.v, s, stats, (&w.norm, &mut w.v, &mut w.quantized))?;
            bias(
                k,
                g.add_width,
                s,
                stats,
                &layer.q_bias,
                &mut w.q,
                &mut w.q_tmp,
            )?;
            bias(k, g.add_kv, s, stats, &layer.k_bias, &mut w.k, &mut w.k_tmp)?;
            bias(k, g.add_kv, s, stats, &layer.v_bias, &mut w.v, &mut w.v_tmp)?;
            run(
                k,
                g.rope_q,
                s,
                stats,
                &[&w.q, &self.phases, &w.uniform],
                &mut w.q_tmp,
            )?;
            run(
                k,
                g.rope_k,
                s,
                stats,
                &[&w.k, &self.phases, &w.uniform],
                &mut w.k_tmp,
            )?;
            run(
                k,
                g.cache,
                s,
                stats,
                &[&w.k_tmp, &w.uniform],
                &mut layer.keys,
            )?;
            run(k, g.cache, s, stats, &[&w.v, &w.uniform], &mut layer.values)?;
            run(
                k,
                g.scores,
                s,
                stats,
                &[&w.q_tmp, &layer.keys, &w.uniform],
                &mut w.scores,
            )?;
            run(
                k,
                g.softmax_stats,
                s,
                stats,
                &[&w.scores, &w.uniform],
                &mut w.softmax_stats,
            )?;
            run(
                k,
                g.softmax_apply,
                s,
                stats,
                &[&w.scores, &w.softmax_stats, &w.uniform],
                &mut w.probabilities,
            )?;
            run(
                k,
                g.attention,
                s,
                stats,
                &[&w.probabilities, &layer.values, &w.uniform],
                &mut w.attention,
            )?;
            project(
                k,
                &layer.o,
                s,
                stats,
                (&w.attention, &mut w.projection, &mut w.quantized),
            )?;
            bias(
                k,
                g.add_width,
                s,
                stats,
                &layer.o_bias,
                &mut w.projection,
                &mut w.projection_tmp,
            )?;
            run(
                k,
                g.add_width,
                s,
                stats,
                &[&w.hidden, &w.projection],
                &mut w.residual,
            )?;
            run(k, g.rms_scale, s, stats, &[&w.residual], &mut w.scale)?;
            run(
                k,
                g.rms_apply,
                s,
                stats,
                &[&w.residual, &layer.ffn_norm, &w.scale],
                &mut w.ffn_norm,
            )?;
            project(
                k,
                &layer.gate,
                s,
                stats,
                (&w.ffn_norm, &mut w.gate, &mut w.quantized),
            )?;
            project(
                k,
                &layer.up,
                s,
                stats,
                (&w.ffn_norm, &mut w.up, &mut w.quantized),
            )?;
            bias(
                k,
                g.add_ff,
                s,
                stats,
                &layer.gate_bias,
                &mut w.gate,
                &mut w.gate_tmp,
            )?;
            bias(
                k,
                g.add_ff,
                s,
                stats,
                &layer.up_bias,
                &mut w.up,
                &mut w.up_tmp,
            )?;
            run(k, g.silu, s, stats, &[&w.gate, &w.up], &mut w.product)?;
            project(
                k,
                &layer.down,
                s,
                stats,
                (&w.product, &mut w.down, &mut w.quantized),
            )?;
            bias(
                k,
                g.add_width,
                s,
                stats,
                &layer.down_bias,
                &mut w.down,
                &mut w.down_tmp,
            )?;
            run(
                k,
                g.add_width,
                s,
                stats,
                &[&w.residual, &w.down],
                &mut w.hidden,
            )?;
        }
        if cancel() {
            return Err(Error::Cancelled);
        }
        run(k, g.rms_scale, s, stats, &[&w.hidden], &mut w.scale)?;
        run(
            k,
            g.rms_apply,
            s,
            stats,
            &[&w.hidden, &self.output_norm, &w.scale],
            &mut w.embedding,
        )?;
        project(
            k,
            &self.output,
            s,
            stats,
            (&w.embedding, &mut w.logits, &mut w.quantized),
        )?;
        let output = Output {
            logits: read(&w.logits)?,
            embedding: read(&w.embedding)?,
        };
        if cancel() {
            return Err(Error::Cancelled);
        }
        self.position += 1;
        stats.committed_tokens = stats.committed_tokens.saturating_add(1);
        Ok(output)
    }
}

impl CudaLlama {
    /// Upload the exact dense-Llama weights owned by the inference facade.
    pub fn from_inference_model(model: &llama_inference::Model, options: &Options) -> Result<Self> {
        let runtime = model
            .runtime_model()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        let llama_model_runtime::Model::Llama(llama) = runtime.as_ref() else {
            return Err(Error::Unsupported(
                "CUDA generation currently requires dense Llama".into(),
            ));
        };
        Self::new(llama, options)
    }
}
fn inference_error(error: Error) -> llama_inference::Error {
    match error {
        Error::Cancelled => llama_inference::Error::Cancelled,
        Error::Unsupported(s) => llama_inference::Error::Unsupported(s),
        Error::Invalid(s) => llama_inference::Error::Invalid(s),
        Error::Device(e) => llama_inference::Error::Internal(e.to_string()),
    }
}
impl llama_inference::DecoderSession for CudaLlama {
    fn context_length(&self) -> usize {
        self.config.max_context
    }
    fn vocabulary_size(&self) -> usize {
        self.config.vocabulary
    }
    fn next_position(&self) -> usize {
        self.position
    }
    fn evaluate(
        &mut self,
        tokens: &[i32],
        cancel: &dyn Fn() -> bool,
    ) -> llama_inference::Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(llama_inference::Error::Invalid(
                "decoder evaluation requires tokens".into(),
            ));
        }
        if tokens.len() > self.config.max_context.saturating_sub(self.position)
            || tokens
                .iter()
                .any(|&id| id < 0 || id as usize >= self.config.vocabulary)
        {
            return Err(llama_inference::Error::Invalid(
                "tokens exceed decoder vocabulary or remaining context".into(),
            ));
        }
        let mut last = Vec::new();
        for &token in tokens {
            last = self
                .decode_with_cancel(token as u32, cancel)
                .map_err(inference_error)?
                .logits;
        }
        Ok(last)
    }
}
