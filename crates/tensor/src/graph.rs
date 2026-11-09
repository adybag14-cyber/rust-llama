use crate::{GluOp, OpCode, Result, Tensor, TensorError, UnaryOp};
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

/// Only these explicitly modeled operations can be submitted to this executor.
/// Registry entries without a variant remain pending implementation.
#[derive(Clone, Debug)]
pub enum Op {
    Dup,
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
            Self::Add
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
