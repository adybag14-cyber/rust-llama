//! Bounded scalar interpreter for IR semantics; never reported as GPU execution.
use crate::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum BufferData {
    F32(Vec<f32>),
    U32(Vec<u32>),
    I32(Vec<i32>),
}
impl BufferData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn scalar(&self) -> Scalar {
        match self {
            Self::F32(_) => Scalar::F32,
            Self::U32(_) => Scalar::U32,
            Self::I32(_) => Scalar::I32,
        }
    }
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Self::U32(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Self::I32(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }
    pub fn from_le_bytes(ty: Scalar, bytes: &[u8]) -> Result<Self> {
        if !bytes.len().is_multiple_of(4) {
            return Err(Error("buffer bytes are not a multiple of four".into()));
        }
        Ok(match ty {
            Scalar::F32 => Self::F32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            ),
            Scalar::U32 => Self::U32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect(),
            ),
            Scalar::I32 => Self::I32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| i32::from_le_bytes(*b))
                    .collect(),
            ),
            Scalar::Bool => return Err(Error("Bool storage is unsupported".into())),
        })
    }
}
#[derive(Debug, Clone, Copy)]
enum Value {
    F(f32),
    U(u32),
    I(i32),
    B(bool),
}
impl Value {
    fn uint(self) -> Result<u32> {
        if let Self::U(value) = self {
            Ok(value)
        } else {
            Err(Error("expected U32 value".into()))
        }
    }
    fn boolean(self) -> Result<bool> {
        if let Self::B(value) = self {
            Ok(value)
        } else {
            Err(Error("expected Bool value".into()))
        }
    }
    fn bits(self) -> Result<u32> {
        match self {
            Self::F(x) => Ok(x.to_bits()),
            Self::U(x) => Ok(x),
            Self::I(x) => Ok(x as u32),
            _ => Err(Error("Bool has no numeric bitcast".into())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReport {
    pub invocations: u32,
    pub statements: u64,
    pub writes: u64,
}

pub fn half_to_float(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits as u32) >> 10) & 31;
    let mantissa = (bits as u32) & 1023;
    match exponent {
        0 => {
            if sign == 0 {
                mantissa as f32 * 2f32.powi(-24)
            } else {
                -(mantissa as f32 * 2f32.powi(-24))
            }
        }
        31 => f32::from_bits(sign | 0x7f800000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

fn evaluate(
    e: &Expr,
    vars: &BTreeMap<String, Value>,
    buffers: &BTreeMap<String, BufferData>,
    gid: u32,
) -> Result<Value> {
    let eval = |e: &Expr| evaluate(e, vars, buffers, gid);
    Ok(match e {
        Expr::Float(x) => Value::F(*x),
        Expr::Uint(x) => Value::U(*x),
        Expr::Int(x) => Value::I(*x),
        Expr::Bool(x) => Value::B(*x),
        Expr::GlobalId => Value::U(gid),
        Expr::Var(name) => *vars
            .get(name)
            .ok_or_else(|| Error(format!("unbound variable {name}")))?,
        Expr::Load { buffer, index } => {
            let i = eval(index)?.uint()? as usize;
            let b = buffers
                .get(buffer)
                .ok_or_else(|| Error(format!("missing buffer {buffer}")))?;
            if i >= b.len() {
                return Err(Error(format!(
                    "out-of-bounds read {buffer}[{i}] of {}",
                    b.len()
                )));
            }
            match b {
                BufferData::F32(v) => Value::F(v[i]),
                BufferData::U32(v) => Value::U(v[i]),
                BufferData::I32(v) => Value::I(v[i]),
            }
        }
        Expr::Binary { op, left, right } => {
            let l = eval(left)?;
            // Preserve && / || short-circuit behavior of all shader targets.
            if *op == Binary::And && !l.boolean()? {
                return Ok(Value::B(false));
            }
            if *op == Binary::Or && l.boolean()? {
                return Ok(Value::B(true));
            }
            let r = eval(right)?;
            match (l, r) {
                (Value::F(a), Value::F(b)) => match op {
                    Binary::Add => Value::F(a + b),
                    Binary::Sub => Value::F(a - b),
                    Binary::Mul => Value::F(a * b),
                    Binary::Div => Value::F(a / b),
                    Binary::Eq => Value::B(a == b),
                    Binary::Ne => Value::B(a != b),
                    Binary::Lt => Value::B(a < b),
                    Binary::Le => Value::B(a <= b),
                    Binary::Gt => Value::B(a > b),
                    Binary::Ge => Value::B(a >= b),
                    _ => return Err(Error("invalid F32 binary operation".into())),
                },
                (Value::U(a), Value::U(b)) => match op {
                    Binary::Add => Value::U(a.wrapping_add(b)),
                    Binary::Sub => Value::U(a.wrapping_sub(b)),
                    Binary::Mul => Value::U(a.wrapping_mul(b)),
                    Binary::Div => Value::U(
                        a.checked_div(b)
                            .ok_or_else(|| Error("integer division by zero".into()))?,
                    ),
                    Binary::Rem => Value::U(
                        a.checked_rem(b)
                            .ok_or_else(|| Error("integer remainder by zero".into()))?,
                    ),
                    Binary::BitAnd => Value::U(a & b),
                    Binary::BitOr => Value::U(a | b),
                    Binary::Xor => Value::U(a ^ b),
                    Binary::Shl => {
                        if b >= 32 {
                            return Err(Error("shift outside 0..32".into()));
                        }
                        Value::U(a << b)
                    }
                    Binary::Shr => {
                        if b >= 32 {
                            return Err(Error("shift outside 0..32".into()));
                        }
                        Value::U(a >> b)
                    }
                    Binary::Eq => Value::B(a == b),
                    Binary::Ne => Value::B(a != b),
                    Binary::Lt => Value::B(a < b),
                    Binary::Le => Value::B(a <= b),
                    Binary::Gt => Value::B(a > b),
                    Binary::Ge => Value::B(a >= b),
                    _ => return Err(Error("invalid U32 binary operation".into())),
                },
                (Value::I(a), Value::I(b)) => match op {
                    Binary::Add => Value::I(
                        a.checked_add(b)
                            .ok_or_else(|| Error("I32 addition overflow".into()))?,
                    ),
                    Binary::Sub => Value::I(
                        a.checked_sub(b)
                            .ok_or_else(|| Error("I32 subtraction overflow".into()))?,
                    ),
                    Binary::Mul => Value::I(
                        a.checked_mul(b)
                            .ok_or_else(|| Error("I32 multiplication overflow".into()))?,
                    ),
                    Binary::Div => Value::I(
                        a.checked_div(b)
                            .ok_or_else(|| Error("I32 division error".into()))?,
                    ),
                    Binary::Eq => Value::B(a == b),
                    Binary::Ne => Value::B(a != b),
                    Binary::Lt => Value::B(a < b),
                    Binary::Le => Value::B(a <= b),
                    Binary::Gt => Value::B(a > b),
                    Binary::Ge => Value::B(a >= b),
                    _ => return Err(Error("invalid I32 binary operation".into())),
                },
                (Value::B(a), Value::B(b)) => match op {
                    Binary::Eq => Value::B(a == b),
                    Binary::Ne => Value::B(a != b),
                    Binary::And => Value::B(a && b),
                    Binary::Or => Value::B(a || b),
                    _ => return Err(Error("invalid Bool operation".into())),
                },
                _ => return Err(Error("binary types disagree".into())),
            }
        }
        Expr::Unary { op, value } => match (op, eval(value)?) {
            (Unary::Neg, Value::F(x)) => Value::F(-x),
            (Unary::Not, Value::B(x)) => Value::B(!x),
            (Unary::Exp, Value::F(x)) => Value::F(x.exp()),
            (Unary::Sqrt, Value::F(x)) => Value::F(x.sqrt()),
            (Unary::Rsqrt, Value::F(x)) => Value::F(1.0 / x.sqrt()),
            (Unary::Tanh, Value::F(x)) => Value::F(x.tanh()),
            (Unary::Abs, Value::F(x)) => Value::F(x.abs()),
            _ => return Err(Error("unary type mismatch".into())),
        },
        Expr::Cast { ty, value } => match (ty, eval(value)?) {
            (Scalar::F32, Value::F(x)) => Value::F(x),
            (Scalar::F32, Value::U(x)) => Value::F(x as f32),
            (Scalar::F32, Value::I(x)) => Value::F(x as f32),
            (Scalar::U32, Value::U(x)) => Value::U(x),
            (Scalar::U32, Value::I(x)) => Value::U(x as u32),
            (Scalar::U32, Value::F(x))
                if x.is_finite() && x >= 0.0 && (x as f64) < 4294967296.0 =>
            {
                Value::U(x as u32)
            }
            (Scalar::I32, Value::I(x)) => Value::I(x),
            (Scalar::I32, Value::U(x)) => Value::I(x as i32),
            (Scalar::I32, Value::F(x))
                if x.is_finite() && (x as f64) >= -2147483648.0 && (x as f64) < 2147483648.0 =>
            {
                Value::I(x as i32)
            }
            _ => return Err(Error("unsupported or out-of-range numeric cast".into())),
        },
        Expr::Bitcast { ty, value } => {
            let bits = eval(value)?.bits()?;
            match ty {
                Scalar::F32 => Value::F(f32::from_bits(bits)),
                Scalar::U32 => Value::U(bits),
                Scalar::I32 => Value::I(bits as i32),
                Scalar::Bool => return Err(Error("Bool bitcast".into())),
            }
        }
        Expr::Select { condition, yes, no } => {
            // Evaluate both branches: WGSL select is not a control-flow guard.
            let a = eval(yes)?;
            let b = eval(no)?;
            if eval(condition)?.boolean()? { a } else { b }
        }
        Expr::Max(a, b) => match (eval(a)?, eval(b)?) {
            (Value::F(a), Value::F(b)) => Value::F(a.max(b)),
            (Value::U(a), Value::U(b)) => Value::U(a.max(b)),
            (Value::I(a), Value::I(b)) => Value::I(a.max(b)),
            _ => return Err(Error("max type mismatch".into())),
        },
        Expr::Min(a, b) => match (eval(a)?, eval(b)?) {
            (Value::F(a), Value::F(b)) => Value::F(a.min(b)),
            (Value::U(a), Value::U(b)) => Value::U(a.min(b)),
            (Value::I(a), Value::I(b)) => Value::I(a.min(b)),
            _ => return Err(Error("min type mismatch".into())),
        },
        Expr::HalfToFloat(value) => Value::F(half_to_float((eval(value)?.uint()? & 0xffff) as u16)),
    })
}

struct State<'a> {
    buffers: &'a mut BTreeMap<String, BufferData>,
    report: ExecutionReport,
    writers: BTreeMap<(String, u32), u32>,
    budget: u64,
}
impl State<'_> {
    fn body(&mut self, body: &[Stmt], vars: &mut BTreeMap<String, Value>, gid: u32) -> Result<()> {
        for stmt in body {
            if self.budget == 0 {
                return Err(Error("IR interpreter operation budget exhausted".into()));
            }
            self.budget -= 1;
            self.report.statements += 1;
            match stmt {
                Stmt::Declare { name, value, .. } | Stmt::Assign { name, value } => {
                    let value = evaluate(value, vars, self.buffers, gid)?;
                    vars.insert(name.clone(), value);
                }
                Stmt::Store {
                    buffer,
                    index,
                    value,
                } => {
                    let index = evaluate(index, vars, self.buffers, gid)?.uint()?;
                    let value = evaluate(value, vars, self.buffers, gid)?;
                    if let Some(previous) = self.writers.insert((buffer.clone(), index), gid)
                        && previous != gid
                    {
                        return Err(Error(format!(
                            "cross-invocation write race in {buffer}[{index}]"
                        )));
                    }
                    let b = self
                        .buffers
                        .get_mut(buffer)
                        .ok_or_else(|| Error("missing output buffer".into()))?;
                    if index as usize >= b.len() {
                        return Err(Error(format!("out-of-bounds write {buffer}[{index}]")));
                    }
                    match (b, value) {
                        (BufferData::F32(v), Value::F(x)) => v[index as usize] = x,
                        (BufferData::U32(v), Value::U(x)) => v[index as usize] = x,
                        (BufferData::I32(v), Value::I(x)) => v[index as usize] = x,
                        _ => return Err(Error("store type mismatch".into())),
                    }
                    self.report.writes += 1;
                }
                Stmt::For { index, end, body } => {
                    let bound = evaluate(end, vars, self.buffers, gid)?.uint()?;
                    if bound as u64 > self.budget {
                        return Err(Error("loop exceeds remaining interpreter budget".into()));
                    }
                    for i in 0..bound {
                        self.budget = self.budget.checked_sub(1).ok_or_else(|| {
                            Error(
                                "IR interpreter operation budget exhausted at loop iteration"
                                    .into(),
                            )
                        })?;
                        let old_keys: Vec<_> = vars.keys().cloned().collect();
                        vars.insert(index.clone(), Value::U(i));
                        self.body(body, vars, gid)?;
                        vars.retain(|name, _| old_keys.contains(name));
                    }
                }
                Stmt::If { condition, yes, no } => {
                    let body = if evaluate(condition, vars, self.buffers, gid)?.boolean()? {
                        yes
                    } else {
                        no
                    };
                    let old_keys: Vec<_> = vars.keys().cloned().collect();
                    self.body(body, vars, gid)?;
                    vars.retain(|name, _| old_keys.contains(name));
                }
            }
        }
        Ok(())
    }
}

/// Validate then execute transactionally. On any bounds/type/race/budget error,
/// the caller's buffers are unchanged. This is a validation interpreter, not a
/// production tensor backend or performance benchmark.
pub fn execute(
    kernel: &Kernel,
    buffers: &mut BTreeMap<String, BufferData>,
    budget: u64,
) -> Result<ExecutionReport> {
    kernel.validate()?;
    if u64::from(kernel.invocations) > budget {
        return Err(Error(
            "dispatch exceeds interpreter operation budget".into(),
        ));
    }
    for descriptor in &kernel.buffers {
        let data = buffers
            .get(&descriptor.name)
            .ok_or_else(|| Error(format!("missing buffer {}", descriptor.name)))?;
        if data.scalar() != descriptor.scalar || data.len() < descriptor.min_elements as usize {
            return Err(Error(format!(
                "buffer {} has wrong type or insufficient storage",
                descriptor.name
            )));
        }
    }
    let mut staged: BTreeMap<_, _> = kernel
        .buffers
        .iter()
        .map(|descriptor| (descriptor.name.clone(), buffers[&descriptor.name].clone()))
        .collect();
    let mut state = State {
        buffers: &mut staged,
        report: ExecutionReport {
            invocations: kernel.invocations,
            statements: 0,
            writes: 0,
        },
        writers: BTreeMap::new(),
        budget,
    };
    for gid in 0..kernel.invocations {
        state.budget = state.budget.checked_sub(1).ok_or_else(|| {
            Error("IR interpreter operation budget exhausted at dispatch invocation".into())
        })?;
        state.body(&kernel.body, &mut BTreeMap::new(), gid)?;
    }
    let report = state.report;
    for descriptor in &kernel.buffers {
        if descriptor.access == Access::ReadWrite
            && let Some(value) = staged.remove(&descriptor.name)
        {
            buffers.insert(descriptor.name.clone(), value);
        }
    }
    Ok(report)
}
