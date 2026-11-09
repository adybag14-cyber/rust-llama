use crate::{DType, GluOp, OpCode, Result, Tensor, TensorError, UnaryOp};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId {
    index: usize,
    graph: u64,
}
impl NodeId {
    pub fn index(self) -> usize {
        self.index
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    Normal,
    NeoX,
}

/// Numeric values match the frozen ggml_tri_type enum.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Triangle {
    UpperDiagonal = 0,
    Upper = 1,
    LowerDiagonal = 2,
    Lower = 3,
}

/// Numeric values match ggml_sort_order.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortOrder {
    Ascending = 0,
    Descending = 1,
}

/// Only these explicitly modeled operations can be submitted to this executor.
/// Registry entries without a variant remain pending implementation.
#[derive(Clone, Debug)]
pub enum Op {
    Dup,
    /// Inputs: values, expert bias rows, I32 expert indices. -1 copies a row.
    AddId,
    /// Inputs: expert matrices, activation rows, I32 expert indices.
    MatMulId,
    /// Inputs: destination, replacement rows, I32/I64 row indices.
    /// Returns independent storage with the destination's byte layout.
    SetRows,
    Argsort {
        order: SortOrder,
    },
    /// Native TOP_K intentionally swaps its first two selected indices.
    TopK {
        count: usize,
    },
    /// Inputs: mask, top-k indices, then enabled score/blocks/extra tensors.
    TopKMask {
        with_scores: bool,
        with_blocks: bool,
        with_extra: bool,
    },
    /// Byte-strided update into a contiguous base. Axis 0 has scalar stride 4.
    /// Overlapping updates read the original base, matching out-of-place ACC.
    Acc {
        strides: [usize; 3],
        offset: usize,
    },
    Set {
        strides: [usize; 3],
        offset: usize,
    },
    /// CPY with a new dtype and the same shape. Only source CPU conversions.
    Cast {
        dtype: DType,
    },
    /// Checked immutable view, with a byte offset relative to the input.
    View {
        shape: [usize; 4],
        strides: [usize; 4],
        offset: usize,
    },
    CumSum,
    ArgMax,
    CountEqual,
    RepeatBack {
        shape: [usize; 4],
    },
    GroupNorm {
        groups: usize,
        epsilon: f32,
    },
    L2Norm {
        epsilon: f32,
    },
    OutProd,
    Diag,
    /// Signed padding permits cropping. Circular padding wraps every axis.
    Pad {
        left: [i32; 4],
        right: [i32; 4],
        circular: bool,
    },
    PadReflect1d {
        left: usize,
        right: usize,
    },
    Roll {
        shifts: [i32; 4],
    },
    /// A zero-input graph operation; stop must exceed start and step be positive.
    Arange {
        start: f32,
        stop: f32,
        step: f32,
    },
    TimestepEmbedding {
        dimensions: usize,
        max_period: usize,
    },
    LeakyRelu {
        negative_slope: f32,
    },
    Tri {
        triangle: Triangle,
    },
    Fill {
        value: f32,
    },
    /// Inputs are output gradient, then the original forward input.
    SiluBack,
    RmsNormBack {
        epsilon: f32,
    },
    /// Inputs are output gradient, then the softmax output. ALiBi backward is
    /// not defined by the frozen CPU and is not accepted here.
    SoftmaxBack {
        scale: f32,
    },
    GetRowsBack {
        rows: usize,
    },
    /// Constructor parameters, before the source's softplus transform.
    Xielu {
        alpha_n: f32,
        alpha_p: f32,
        beta: f32,
        epsilon: f32,
    },
    GluPacked {
        kind: GluOp,
        swap: bool,
    },
    SwiGluOai {
        alpha: f32,
        limit: f32,
    },
    SwiGluClamp {
        limit: f32,
    },
    SwiGluOaiPacked {
        alpha: f32,
        limit: f32,
        swap: bool,
    },
    SwiGluClampPacked {
        limit: f32,
        swap: bool,
    },
    Add,
    Sub,
    Mul,
    Div,
    AddScalar,
    Scale {
        scale: f32,
        bias: f32,
    },
    Square,
    Sqrt,
    Log,
    Sin,
    Cos,
    Sum,
    SumRows,
    Mean,
    Repeat {
        shape: [usize; 4],
    },
    Concat {
        axis: usize,
    },
    MatMul,
    Contiguous,
    Reshape {
        shape: [usize; 4],
    },
    Permute {
        axes: [usize; 4],
    },
    Transpose,
    GetRows,
    Softmax {
        scale: f32,
    },
    RmsNorm {
        epsilon: f32,
    },
    Norm {
        epsilon: f32,
    },
    Unary(UnaryOp),
    Glu {
        kind: GluOp,
        swap: bool,
    },
    Clamp {
        min: f32,
        max: f32,
    },
    DiagMask {
        past: usize,
        zero: bool,
    },
    /// Positions correspond to ne[2]. Basic normal/NeoX RoPE only:
    /// no YaRN, LongRoPE, attention scaling or frequency-factor tensor.
    Rope {
        dimensions: usize,
        positions: Vec<i32>,
        mode: RopeMode,
        freq_base: f32,
        freq_scale: f32,
    },
}
impl Op {
    pub fn code(&self) -> OpCode {
        use Op::*;
        match self {
            Dup => OpCode::DUP,
            AddId => OpCode::ADD_ID,
            MatMulId => OpCode::MUL_MAT_ID,
            SetRows => OpCode::SET_ROWS,
            Argsort { .. } => OpCode::ARGSORT,
            TopK { .. } => OpCode::TOP_K,
            TopKMask { .. } => OpCode::TOP_K_MASK,
            Acc { .. } => OpCode::ACC,
            Set { .. } => OpCode::SET,
            Cast { .. } => OpCode::CPY,
            View { .. } => OpCode::VIEW,
            CumSum => OpCode::CUMSUM,
            ArgMax => OpCode::ARGMAX,
            CountEqual => OpCode::COUNT_EQUAL,
            RepeatBack { .. } => OpCode::REPEAT_BACK,
            GroupNorm { .. } => OpCode::GROUP_NORM,
            L2Norm { .. } => OpCode::L2_NORM,
            OutProd => OpCode::OUT_PROD,
            Diag => OpCode::DIAG,
            Pad { .. } => OpCode::PAD,
            PadReflect1d { .. } => OpCode::PAD_REFLECT_1D,
            Roll { .. } => OpCode::ROLL,
            Arange { .. } => OpCode::ARANGE,
            TimestepEmbedding { .. } => OpCode::TIMESTEP_EMBEDDING,
            LeakyRelu { .. } => OpCode::LEAKY_RELU,
            Tri { .. } => OpCode::TRI,
            Fill { .. } => OpCode::FILL,
            SiluBack => OpCode::SILU_BACK,
            RmsNormBack { .. } => OpCode::RMS_NORM_BACK,
            SoftmaxBack { .. } => OpCode::SOFT_MAX_BACK,
            GetRowsBack { .. } => OpCode::GET_ROWS_BACK,
            Xielu { .. } => OpCode::UNARY,
            GluPacked { .. }
            | SwiGluOai { .. }
            | SwiGluClamp { .. }
            | SwiGluOaiPacked { .. }
            | SwiGluClampPacked { .. } => OpCode::GLU,
            Add => OpCode::ADD,
            Sub => OpCode::SUB,
            Mul => OpCode::MUL,
            Div => OpCode::DIV,
            AddScalar => OpCode::ADD1,
            Scale { .. } => OpCode::SCALE,
            Square => OpCode::SQR,
            Sqrt => OpCode::SQRT,
            Log => OpCode::LOG,
            Sin => OpCode::SIN,
            Cos => OpCode::COS,
            Sum => OpCode::SUM,
            SumRows => OpCode::SUM_ROWS,
            Mean => OpCode::MEAN,
            Repeat { .. } => OpCode::REPEAT,
            Concat { .. } => OpCode::CONCAT,
            MatMul => OpCode::MUL_MAT,
            Contiguous => OpCode::CONT,
            Reshape { .. } => OpCode::RESHAPE,
            Permute { .. } => OpCode::PERMUTE,
            Transpose => OpCode::TRANSPOSE,
            GetRows => OpCode::GET_ROWS,
            Softmax { .. } => OpCode::SOFT_MAX,
            RmsNorm { .. } => OpCode::RMS_NORM,
            Norm { .. } => OpCode::NORM,
            Unary(_) => OpCode::UNARY,
            Glu { .. } => OpCode::GLU,
            Clamp { .. } => OpCode::CLAMP,
            DiagMask { zero: true, .. } => OpCode::DIAG_MASK_ZERO,
            DiagMask { zero: false, .. } => OpCode::DIAG_MASK_INF,
            Rope { .. } => OpCode::ROPE,
        }
    }
    fn arity(&self) -> usize {
        match self {
            Self::Arange { .. } => 0,
            Self::AddId | Self::MatMulId | Self::SetRows => 3,
            Self::TopKMask {
                with_scores,
                with_blocks,
                with_extra,
            } => {
                2 + usize::from(*with_scores) + usize::from(*with_blocks) + usize::from(*with_extra)
            }
            Self::Acc { .. } | Self::Set { .. } => 2,
            Self::CountEqual
            | Self::OutProd
            | Self::SiluBack
            | Self::RmsNormBack { .. }
            | Self::SoftmaxBack { .. }
            | Self::GetRowsBack { .. }
            | Self::SwiGluOai { .. }
            | Self::SwiGluClamp { .. }
            | Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::AddScalar
            | Self::Concat { .. }
            | Self::MatMul
            | Self::GetRows
            | Self::Glu { .. } => 2,
            _ => 1,
        }
    }
}
#[derive(Clone, Debug)]
enum Node {
    Input(Tensor),
    Compute { op: Op, inputs: Vec<NodeId> },
}
#[derive(Debug)]
pub struct Graph {
    id: u64,
    nodes: Vec<Node>,
}
impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
impl Graph {
    pub fn new() -> Self {
        Self {
            id: NEXT_GRAPH.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::new(),
        }
    }
    pub fn input(&mut self, tensor: Tensor) -> NodeId {
        let id = NodeId {
            index: self.nodes.len(),
            graph: self.id,
        };
        self.nodes.push(Node::Input(tensor));
        id
    }
    /// Dependencies must already belong to this graph's node range.
    /// Appending only backward edges makes cycles unrepresentable.
    pub fn op(&mut self, op: Op, inputs: &[NodeId]) -> Result<NodeId> {
        if inputs.len() != op.arity() {
            return Err(TensorError::InvalidArity {
                expected: op.arity(),
                actual: inputs.len(),
            });
        }
        for id in inputs {
            if id.graph != self.id || id.index >= self.nodes.len() {
                return Err(TensorError::InvalidNode(id.index));
            }
        }
        let id = NodeId {
            index: self.nodes.len(),
            graph: self.id,
        };
        self.nodes
            .try_reserve(1)
            .map_err(|_| TensorError::AllocationFailed)?;
        self.nodes.push(Node::Compute {
            op,
            inputs: inputs.to_vec(),
        });
        Ok(id)
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn evaluate(&self, target: NodeId) -> Result<Tensor> {
        self.evaluate_with_cancel(target, &AtomicBool::new(false))
    }
    /// Only target dependencies execute. Cancellation never modifies input storage.
    pub fn evaluate_with_cancel(&self, target: NodeId, cancel: &AtomicBool) -> Result<Tensor> {
        if target.graph != self.id || target.index >= self.nodes.len() {
            return Err(TensorError::InvalidNode(target.index));
        }
        let mut needed = crate::reserved(target.index + 1)?;
        needed.resize(target.index + 1, false);
        let mut stack = vec![target];
        while let Some(id) = stack.pop() {
            check_cancel(cancel)?;
            if needed[id.index] {
                continue;
            }
            needed[id.index] = true;
            if let Node::Compute { inputs, .. } = &self.nodes[id.index] {
                stack.extend(inputs);
            }
        }
        let mut values = crate::reserved(target.index + 1)?;
        values.resize_with(target.index + 1, || None);
        for (index, node) in self.nodes.iter().enumerate().take(target.index + 1) {
            if !needed[index] {
                continue;
            }
            check_cancel(cancel)?;
            let value = match node {
                Node::Input(t) => t.clone(),
                Node::Compute { op, inputs } => {
                    let args = inputs
                        .iter()
                        .map(|id| values[id.index].as_ref().expect("earlier graph dependency"))
                        .collect::<Vec<_>>();
                    crate::cpu::evaluate(op, &args, cancel)?
                }
            };
            values[index] = Some(value);
        }
        check_cancel(cancel)?;
        Ok(values[target.index].take().expect("evaluated target"))
    }
}
pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(TensorError::Cancelled)
    } else {
        Ok(())
    }
}
