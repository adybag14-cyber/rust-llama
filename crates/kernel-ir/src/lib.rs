//! Versioned, Rust-authored scalar kernel IR and deterministic GPU source emitters.
//!
//! Supported lowering is deliberately explicit: storage buffers, F32/U32/I32,
//! bounded row loops and independent work items. Subgroups, barriers, atomics,
//! cooperative matrices, textures and training kernels remain open work.
#![forbid(unsafe_code)]

mod emit;
mod hash;
mod interpreter;
pub mod kernels;

pub use emit::{Artifact, Target, emit};
pub use hash::{CacheIdentity, sha256, sha256_hex};
pub use interpreter::{BufferData, ExecutionReport, execute, half_to_float};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const IR_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    F32,
    U32,
    I32,
    Bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    ReadWrite,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    pub name: String,
    pub scalar: Scalar,
    pub access: Access,
    /// Minimum number of 32-bit elements, including stride gaps and packed padding.
    pub min_elements: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Neg,
    Not,
    Exp,
    Sqrt,
    Rsqrt,
    Tanh,
    Abs,
}

/// Semantic expressions only. There is no raw shader/source escape hatch.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Float(f32),
    Uint(u32),
    Int(i32),
    Bool(bool),
    Var(String),
    GlobalId,
    Load {
        buffer: String,
        index: Box<Expr>,
    },
    Binary {
        op: Binary,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: Unary,
        value: Box<Expr>,
    },
    Cast {
        ty: Scalar,
        value: Box<Expr>,
    },
    Bitcast {
        ty: Scalar,
        value: Box<Expr>,
    },
    Select {
        condition: Box<Expr>,
        yes: Box<Expr>,
        no: Box<Expr>,
    },
    Max(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    /// Low 16 bits are an IEEE binary16 value. No FP16 device feature required.
    HalfToFloat(Box<Expr>),
}
impl Expr {
    pub fn var(name: &str) -> Self {
        Self::Var(name.into())
    }
    pub fn load(buffer: &str, index: Expr) -> Self {
        Self::Load {
            buffer: buffer.into(),
            index: Box::new(index),
        }
    }
    pub fn binary(self, op: Binary, rhs: Expr) -> Self {
        Self::Binary {
            op,
            left: Box::new(self),
            right: Box::new(rhs),
        }
    }
    pub fn unary(self, op: Unary) -> Self {
        Self::Unary {
            op,
            value: Box::new(self),
        }
    }
    pub fn cast(self, ty: Scalar) -> Self {
        Self::Cast {
            ty,
            value: Box::new(self),
        }
    }
    pub fn bitcast(self, ty: Scalar) -> Self {
        Self::Bitcast {
            ty,
            value: Box::new(self),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Declare {
        name: String,
        ty: Scalar,
        value: Expr,
    },
    Assign {
        name: String,
        value: Expr,
    },
    Store {
        buffer: String,
        index: Expr,
        value: Expr,
    },
    /// For index in [0, end), with the index read-only inside the loop body.
    For {
        index: String,
        end: Expr,
        body: Vec<Stmt>,
    },
    If {
        condition: Expr,
        yes: Vec<Stmt>,
        no: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    pub version: u32,
    pub name: String,
    /// A 1D dispatch. Tail invocations are guarded by every emitter.
    pub invocations: u32,
    pub workgroup_size: u32,
    pub buffers: Vec<Buffer>,
    pub body: Vec<Stmt>,
    /// Frozen source paths and operation semantics; metadata, not shader text.
    pub source_refs: Vec<String>,
}

fn identifier(value: &str) -> bool {
    // A deliberately small namespace excludes syntax injection and internal names.
    let mut chars = value.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_alphabetic());
    first_ok
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !value.starts_with("ir_")
        && !value.contains("__")
        && ![
            "if", "else", "for", "while", "return", "void", "float", "uint", "int", "bool", "var",
            "let", "fn", "kernel", "device", "constant", "struct", "buffer", "uniform", "shared",
            "true", "false", "main",
        ]
        .contains(&value)
}

impl Kernel {
    pub fn validate(&self) -> Result<()> {
        if self.version != IR_VERSION {
            return Err(Error(format!("unsupported IR version {}", self.version)));
        }
        if !identifier(&self.name) {
            return Err(Error("invalid kernel identifier".into()));
        }
        if self.workgroup_size == 0 || self.workgroup_size > 1024 {
            return Err(Error("workgroup size outside 1..=1024".into()));
        }
        if self.invocations == 0 {
            return Err(Error(
                "zero-sized dispatch must be handled on the host".into(),
            ));
        }
        self.invocations
            .checked_add(self.workgroup_size - 1)
            .ok_or_else(|| Error("dispatch rounding overflows u32".into()))?;
        if self.buffers.is_empty() || self.buffers.len() > 16 {
            return Err(Error("kernel must bind 1..=16 buffers".into()));
        }
        let mut buffers = BTreeMap::new();
        for buffer in &self.buffers {
            if !identifier(&buffer.name)
                || buffer.scalar == Scalar::Bool
                || buffer.min_elements == 0
            {
                return Err(Error(format!("invalid buffer {}", buffer.name)));
            }
            if buffers
                .insert(buffer.name.clone(), (buffer.scalar, buffer.access))
                .is_some()
            {
                return Err(Error("duplicate buffer name".into()));
            }
        }
        let mut variables = BTreeMap::new();
        let mut immutable = BTreeSet::new();
        validate_body(&self.body, &buffers, &mut variables, &mut immutable, 0)?;
        Ok(())
    }
    pub fn workgroups(&self) -> Result<u32> {
        self.validate()?;
        Ok(self.invocations.div_ceil(self.workgroup_size))
    }
}

type Buffers = BTreeMap<String, (Scalar, Access)>;
type Variables = BTreeMap<String, Scalar>;

fn infer(expr: &Expr, buffers: &Buffers, variables: &Variables, depth: usize) -> Result<Scalar> {
    if depth > 128 {
        return Err(Error("expression nesting exceeds 128".into()));
    }
    let sub = |e: &Expr| infer(e, buffers, variables, depth + 1);
    Ok(match expr {
        Expr::Float(x) => {
            if !x.is_finite() {
                return Err(Error("nonfinite literal: use an explicit bitcast".into()));
            }
            Scalar::F32
        }
        Expr::Uint(_) | Expr::GlobalId => Scalar::U32,
        Expr::Int(_) => Scalar::I32,
        Expr::Bool(_) => Scalar::Bool,
        Expr::Var(name) => *variables
            .get(name)
            .ok_or_else(|| Error(format!("undeclared variable {name}")))?,
        Expr::Load { buffer, index } => {
            if sub(index)? != Scalar::U32 {
                return Err(Error("buffer index must be U32".into()));
            }
            let (ty, access) = buffers
                .get(buffer)
                .ok_or_else(|| Error(format!("unknown buffer {buffer}")))?;
            if *access == Access::ReadWrite {
                return Err(Error("reads from writable buffers require synchronization semantics not implemented in IR v1".into()));
            }
            *ty
        }
        Expr::Binary { op, left, right } => {
            let l = sub(left)?;
            let r = sub(right)?;
            if l != r {
                return Err(Error("binary operand type mismatch".into()));
            }
            match op {
                Binary::Eq | Binary::Ne => Scalar::Bool,
                Binary::Lt | Binary::Le | Binary::Gt | Binary::Ge if l != Scalar::Bool => {
                    Scalar::Bool
                }
                Binary::And | Binary::Or if l == Scalar::Bool => Scalar::Bool,
                Binary::BitAnd | Binary::BitOr | Binary::Xor | Binary::Shl | Binary::Shr
                    if l == Scalar::U32 =>
                {
                    Scalar::U32
                }
                Binary::Rem if l == Scalar::U32 => Scalar::U32,
                Binary::Add | Binary::Sub | Binary::Mul | Binary::Div if l != Scalar::Bool => l,
                _ => return Err(Error(format!("invalid operator {op:?} for {l:?}"))),
            }
        }
        Expr::Unary { op, value } => {
            let t = sub(value)?;
            match op {
                Unary::Not if t == Scalar::Bool => Scalar::Bool,
                Unary::Neg | Unary::Abs if t == Scalar::F32 => Scalar::F32,
                Unary::Exp | Unary::Sqrt | Unary::Rsqrt | Unary::Tanh if t == Scalar::F32 => {
                    Scalar::F32
                }
                _ => return Err(Error("invalid unary operand".into())),
            }
        }
        Expr::Cast { ty, value } => {
            if *ty == Scalar::Bool || sub(value)? == Scalar::Bool {
                return Err(Error("boolean numeric casts unsupported".into()));
            }
            *ty
        }
        Expr::Bitcast { ty, value } => {
            if *ty == Scalar::Bool || sub(value)? == Scalar::Bool {
                return Err(Error("boolean bitcasts unsupported".into()));
            }
            *ty
        }
        Expr::Select { condition, yes, no } => {
            let condition_type = sub(condition)?;
            let yes_type = sub(yes)?;
            let no_type = sub(no)?;
            if condition_type != Scalar::Bool || yes_type != no_type {
                return Err(Error("select type mismatch".into()));
            }
            yes_type
        }
        Expr::Max(a, b) | Expr::Min(a, b) => {
            let ty = sub(a)?;
            if ty == Scalar::Bool || ty != sub(b)? {
                return Err(Error("min/max type mismatch".into()));
            }
            ty
        }
        Expr::HalfToFloat(value) => {
            if sub(value)? != Scalar::U32 {
                return Err(Error("half input must be U32".into()));
            }
            Scalar::F32
        }
    })
}

fn validate_body(
    body: &[Stmt],
    buffers: &Buffers,
    vars: &mut Variables,
    immutable: &mut BTreeSet<String>,
    depth: usize,
) -> Result<()> {
    if depth > 64 || body.len() > 4096 {
        return Err(Error("statement complexity limit".into()));
    }
    for statement in body {
        match statement {
            Stmt::Declare { name, ty, value } => {
                if !identifier(name) || vars.contains_key(name) || buffers.contains_key(name) {
                    return Err(Error(format!("invalid/duplicate variable {name}")));
                }
                if infer(value, buffers, vars, 0)? != *ty {
                    return Err(Error(format!("initializer type mismatch for {name}")));
                }
                vars.insert(name.clone(), *ty);
            }
            Stmt::Assign { name, value } => {
                if immutable.contains(name) {
                    return Err(Error("loop index cannot be assigned".into()));
                }
                if vars.get(name).copied() != Some(infer(value, buffers, vars, 0)?) {
                    return Err(Error(format!("assignment type mismatch for {name}")));
                }
            }
            Stmt::Store {
                buffer,
                index,
                value,
            } => {
                let (ty, access) = buffers
                    .get(buffer)
                    .ok_or_else(|| Error(format!("unknown buffer {buffer}")))?;
                if *access != Access::ReadWrite {
                    return Err(Error("store into read-only buffer".into()));
                }
                if infer(index, buffers, vars, 0)? != Scalar::U32
                    || infer(value, buffers, vars, 0)? != *ty
                {
                    return Err(Error("store type mismatch".into()));
                }
            }
            Stmt::For { index, end, body } => {
                if !identifier(index)
                    || vars.contains_key(index)
                    || buffers.contains_key(index)
                    || infer(end, buffers, vars, 0)? != Scalar::U32
                {
                    return Err(Error("invalid loop index or bound".into()));
                }
                let mut nested = vars.clone();
                nested.insert(index.clone(), Scalar::U32);
                let mut nested_immutable = immutable.clone();
                nested_immutable.insert(index.clone());
                validate_body(body, buffers, &mut nested, &mut nested_immutable, depth + 1)?;
            }
            Stmt::If { condition, yes, no } => {
                if infer(condition, buffers, vars, 0)? != Scalar::Bool {
                    return Err(Error("condition must be Bool".into()));
                }
                validate_body(
                    yes,
                    buffers,
                    &mut vars.clone(),
                    &mut immutable.clone(),
                    depth + 1,
                )?;
                validate_body(
                    no,
                    buffers,
                    &mut vars.clone(),
                    &mut immutable.clone(),
                    depth + 1,
                )?;
            }
        }
    }
    Ok(())
}
