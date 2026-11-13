//! Typed Rust semantics → PTX, plus an instruction interpreter for CPU checks.
//!
//! No archived CUDA source is embedded. Kernels are deliberately scalar per
//! output lane: they establish numerical/addressing behavior, not tensor-core
//! or throughput parity. Interpretation is not device execution or PTX assembly.
use std::fmt::{self, Write};

pub const IR_VERSION: u32 = 2;

mod decoder;
pub use decoder::DecoderKernel;

#[derive(Debug, Clone, PartialEq)]
pub enum KernelKind {
    /// Checked kernels for a persistent dense decoder graph.
    Decoder(DecoderKernel),
    AddF32 {
        elements: u32,
        lhs_stride: u32,
        rhs_stride: u32,
        output_stride: u32,
    },
    ScaleF32 {
        elements: u32,
        input_stride: u32,
        output_stride: u32,
        scale: f32,
    },
    SumF32 {
        elements: u32,
        input_stride: u32,
    },
    /// Dense F32 matrix-vector product, one output lane per matrix row.
    /// All strides are measured in F32 elements; input rows may overlap or
    /// broadcast, but output lanes may not alias.
    MatVecF32 {
        rows: u32,
        columns: u32,
        row_stride: u32,
        column_stride: u32,
        vector_stride: u32,
        output_stride: u32,
    },
    /// Frozen GGML Q4_0: 32 numbers in 18 bytes: little-endian FP16 scale,
    /// 16 packed bytes; low nibbles are values 0..15, high are 16..31.
    Q4MatVec {
        rows: u32,
        columns: u32,
        row_stride_bytes: u32,
        block_stride_bytes: u32,
        vector_stride: u32,
    },
}
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    pub kind: KernelKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtxTarget {
    pub sm: u32,
    pub isa_major: u32,
    pub isa_minor: u32,
}
impl Default for PtxTarget {
    fn default() -> Self {
        Self {
            sm: 60,
            isa_major: 7,
            isa_minor: 0,
        }
    }
}
impl PtxTarget {
    pub fn validate(self) -> Result<(), KernelError> {
        let minimum = match self.sm {
            60..=62 => (5, 0),
            70 => (6, 0),
            72 => (6, 1),
            75 => (6, 3),
            80 => (7, 0),
            86 => (7, 1),
            87 => (7, 4),
            89 | 90 => (7, 8),
            _ => {
                return Err(KernelError(
                    "SM target not covered by this emitter's verified version table".into(),
                ));
            }
        };
        let version = (self.isa_major, self.isa_minor);
        // The Rust emitter uses only this bounded ISA subset. Other targets
        // need explicit proof instead of accepting arbitrary numeric labels.
        let known_version = matches!(version, (6, 0..=5) | (7, 0..=8) | (8, 0));
        if version < minimum || !known_version {
            return Err(KernelError(format!(
                "PTX {}.{} does not satisfy this emitter's SM {} contract",
                self.isa_major, self.isa_minor, self.sm
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError(pub String);
impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for KernelError {}

fn extent(elements: u32, stride: u32, width: u32) -> Result<usize, KernelError> {
    if elements == 0 || elements > u32::MAX - 255 {
        return Err(KernelError(
            "element count must be nonzero and leave room for rounded launch lanes".into(),
        ));
    }
    let bytes = (elements as u64 - 1)
        .checked_mul(stride as u64)
        .and_then(|x| x.checked_add(width as u64))
        .ok_or_else(|| KernelError("extent overflow".into()))?;
    usize::try_from(bytes)
        .ok()
        .filter(|v| *v <= isize::MAX as usize)
        .ok_or_else(|| KernelError("extent exceeds host addressable storage".into()))
}
fn float_stride(elements: u32, stride: u32) -> Result<usize, KernelError> {
    let bytes = stride
        .checked_mul(4)
        .ok_or_else(|| KernelError("F32 byte stride overflow".into()))?;
    extent(elements, bytes, 4)
}
impl KernelSpec {
    pub fn buffer_bytes(&self) -> Result<Vec<usize>, KernelError> {
        let out_stride_check = |elements: u32, stride: u32| {
            if elements > 1 && stride == 0 {
                Err(KernelError("output lanes must not alias".into()))
            } else {
                float_stride(elements, stride)
            }
        };
        match self.kind {
            KernelKind::Decoder(ref kind) => kind.buffer_bytes(),
            KernelKind::AddF32 {
                elements,
                lhs_stride,
                rhs_stride,
                output_stride,
            } => Ok(vec![
                float_stride(elements, lhs_stride)?,
                float_stride(elements, rhs_stride)?,
                out_stride_check(elements, output_stride)?,
            ]),
            KernelKind::ScaleF32 {
                elements,
                input_stride,
                output_stride,
                ..
            } => Ok(vec![
                float_stride(elements, input_stride)?,
                out_stride_check(elements, output_stride)?,
            ]),
            KernelKind::SumF32 {
                elements,
                input_stride,
            } => Ok(vec![float_stride(elements, input_stride)?, 4]),
            KernelKind::MatVecF32 {
                rows,
                columns,
                row_stride,
                column_stride,
                vector_stride,
                output_stride,
            } => {
                let row_bytes = row_stride
                    .checked_mul(4)
                    .ok_or_else(|| KernelError("matrix row byte stride overflow".into()))?;
                let row_width = float_stride(columns, column_stride)?;
                let matrix_bytes = extent(rows, row_bytes, 1)?
                    .checked_add(row_width - 1)
                    .filter(|bytes| *bytes <= isize::MAX as usize)
                    .ok_or_else(|| {
                        KernelError("matrix extent exceeds host address space".into())
                    })?;
                Ok(vec![
                    matrix_bytes,
                    float_stride(columns, vector_stride)?,
                    out_stride_check(rows, output_stride)?,
                ])
            }
            KernelKind::Q4MatVec {
                rows,
                columns,
                row_stride_bytes,
                block_stride_bytes,
                vector_stride,
            } => {
                if columns == 0
                    || columns % 32 != 0
                    || block_stride_bytes < 18
                    || block_stride_bytes % 2 != 0
                    || row_stride_bytes % 2 != 0
                {
                    return Err(KernelError(
                        "Q4_0 requires 32-value blocks, 18+ even bytes per block, aligned rows"
                            .into(),
                    ));
                }
                let row_bytes = (columns as u64 / 32 - 1) * block_stride_bytes as u64 + 18;
                if (row_stride_bytes as u64) < row_bytes {
                    return Err(KernelError(
                        "Q4 rows overlap or row stride is too small".into(),
                    ));
                }
                let matrix = extent(
                    rows,
                    row_stride_bytes,
                    u32::try_from(row_bytes)
                        .map_err(|_| KernelError("Q4 row exceeds u32 byte addressing".into()))?,
                )?;
                Ok(vec![
                    matrix,
                    float_stride(columns, vector_stride)?,
                    float_stride(rows, 1)?,
                ])
            }
        }
    }
    pub fn outputs(&self) -> u32 {
        match self.kind {
            KernelKind::Decoder(ref kind) => kind.outputs(),
            KernelKind::AddF32 { elements, .. } | KernelKind::ScaleF32 { elements, .. } => elements,
            KernelKind::SumF32 { .. } => 1,
            KernelKind::Q4MatVec { rows, .. } | KernelKind::MatVecF32 { rows, .. } => rows,
        }
    }
    pub fn output_buffer(&self) -> usize {
        match self.kind {
            KernelKind::Decoder(ref kind) => kind.output_buffer(),
            KernelKind::AddF32 { .. }
            | KernelKind::Q4MatVec { .. }
            | KernelKind::MatVecF32 { .. } => 2,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Pred,
    B16,
    U32,
    I32,
    U64,
    F32,
    F64,
}
impl Type {
    fn ptx(self) -> &'static str {
        match self {
            Self::Pred => "pred",
            Self::B16 => "b16",
            Self::U32 => "u32",
            Self::I32 => "s32",
            Self::U64 => "u64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Reg {
    index: usize,
    ty: Type,
}
impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%v{}", self.index)
    }
}
#[derive(Debug, Clone, Copy)]
enum Special {
    ThreadX,
    BlockX,
    BlockWidth,
}
#[derive(Debug, Clone, Copy)]
enum Operand {
    Reg(Reg),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    Special(Special),
}
impl From<Reg> for Operand {
    fn from(value: Reg) -> Self {
        Self::Reg(value)
    }
}
impl From<u32> for Operand {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}
impl Operand {
    fn ty(self) -> Type {
        match self {
            Self::Reg(r) => r.ty,
            Self::U32(_) | Self::Special(_) => Type::U32,
            Self::I32(_) => Type::I32,
            Self::F32(_) => Type::F32,
            Self::F64(_) => Type::F64,
        }
    }
}
impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(r) => write!(f, "{r}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "0f{:08x}", v.to_bits()),
            Self::F64(v) => write!(f, "0d{:016x}", v.to_bits()),
            Self::Special(v) => f.write_str(match v {
                Special::ThreadX => "%tid.x",
                Special::BlockX => "%ctaid.x",
                Special::BlockWidth => "%ntid.x",
            }),
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum Binary {
    Add,
    Sub,
    Mul,
    WideMul,
    And,
    Shr,
    Div,
    Rem,
    Max,
}
#[derive(Debug, Clone, Copy)]
enum Unary {
    Sqrt,
    Exp2,
}
#[derive(Debug, Clone, Copy)]
enum Conversion {
    U32ToI32,
    I32ToF32,
    HalfToF32,
    F32ToF64,
    F64ToF32,
    F32ToHalf,
    F32ToI32Trunc,
}
#[derive(Debug, Clone, Copy)]
enum LoadType {
    U8,
    U32,
    F16,
    F32,
}
#[derive(Debug, Clone)]
enum Instruction {
    Parameter {
        dst: Reg,
        index: usize,
    },
    Move {
        dst: Reg,
        value: Operand,
    },
    Binary {
        dst: Reg,
        op: Binary,
        a: Operand,
        b: Operand,
    },
    Unary {
        dst: Reg,
        op: Unary,
        src: Reg,
    },
    Convert {
        dst: Reg,
        conversion: Conversion,
        src: Reg,
    },
    Load {
        dst: Reg,
        address: Reg,
        kind: LoadType,
    },
    Store {
        value: Reg,
        address: Reg,
    },
    /// PTX testp.finite: accepts zero/subnormal/normal, rejects NaN and either infinity.
    Finite {
        pred: Reg,
        src: Reg,
    },
    GreaterEqual {
        pred: Reg,
        a: Operand,
        b: Operand,
    },
    Label(usize),
    Branch {
        label: usize,
        pred: Option<Reg>,
    },
    Return,
}
#[derive(Debug, Clone, Default)]
struct Program {
    registers: Vec<Reg>,
    instructions: Vec<Instruction>,
    parameters: usize,
    labels: usize,
}
impl Program {
    fn reg(&mut self, ty: Type) -> Reg {
        let r = Reg {
            index: self.registers.len(),
            ty,
        };
        self.registers.push(r);
        r
    }
    fn label(&mut self) -> usize {
        let l = self.labels;
        self.labels += 1;
        l
    }
    fn push(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }
    fn param(&mut self, index: usize) -> Reg {
        let dst = self.reg(Type::U64);
        self.push(Instruction::Parameter { dst, index });
        dst
    }
    fn value(&mut self, value: Operand) -> Reg {
        let dst = self.reg(value.ty());
        self.push(Instruction::Move { dst, value });
        dst
    }
    fn binary(&mut self, op: Binary, a: impl Into<Operand>, b: impl Into<Operand>) -> Reg {
        let (a, b) = (a.into(), b.into());
        let ty = if matches!(op, Binary::WideMul) {
            Type::U64
        } else {
            a.ty()
        };
        let dst = self.reg(ty);
        self.push(Instruction::Binary { dst, op, a, b });
        dst
    }
    fn assign_binary(
        &mut self,
        dst: Reg,
        op: Binary,
        a: impl Into<Operand>,
        b: impl Into<Operand>,
    ) {
        self.push(Instruction::Binary {
            dst,
            op,
            a: a.into(),
            b: b.into(),
        });
    }
    fn address(&mut self, base: Reg, index: Reg, stride_bytes: u32) -> Reg {
        let off = self.binary(Binary::WideMul, index, stride_bytes);
        self.binary(Binary::Add, base, off)
    }
    fn load(&mut self, address: Reg, kind: LoadType) -> Reg {
        let ty = match kind {
            LoadType::U8 | LoadType::U32 => Type::U32,
            LoadType::F16 => Type::B16,
            LoadType::F32 => Type::F32,
        };
        let dst = self.reg(ty);
        self.push(Instruction::Load { dst, address, kind });
        dst
    }
    fn convert(&mut self, conversion: Conversion, src: Reg) -> Reg {
        let dst = self.reg(match conversion {
            Conversion::U32ToI32 | Conversion::F32ToI32Trunc => Type::I32,
            Conversion::F32ToHalf => Type::B16,
            Conversion::F32ToF64 => Type::F64,
            _ => Type::F32,
        });
        self.push(Instruction::Convert {
            dst,
            conversion,
            src,
        });
        dst
    }
    fn unary(&mut self, op: Unary, src: Reg) -> Reg {
        let dst = self.reg(Type::F32);
        self.push(Instruction::Unary { dst, op, src });
        dst
    }
    // NVIDIA PTX ISA 9.7.3.1: testp.finite.f32 returns a predicate. Introduced
    // in PTX 2.0, it is available for every target admitted by this emitter.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-testp
    fn finite(&mut self, src: Reg) -> Reg {
        let pred = self.reg(Type::Pred);
        self.push(Instruction::Finite { pred, src });
        pred
    }
    fn guard_nonfinite(&mut self, src: Reg, label: usize) {
        let valid = self.label();
        let pred = self.finite(src);
        self.push(Instruction::Branch {
            label: valid,
            pred: Some(pred),
        });
        self.push(Instruction::Branch { label, pred: None });
        self.push(Instruction::Label(valid));
    }
    fn finite_or_nan(&mut self, value: Reg) -> Reg {
        let valid = self.label();
        let pred = self.finite(value);
        self.push(Instruction::Branch {
            label: valid,
            pred: Some(pred),
        });
        self.push(Instruction::Move {
            dst: value,
            value: Operand::F32(f32::NAN),
        });
        self.push(Instruction::Label(valid));
        value
    }
    fn guard_ge(&mut self, a: impl Into<Operand>, b: impl Into<Operand>, label: usize) {
        let pred = self.reg(Type::Pred);
        self.push(Instruction::GreaterEqual {
            pred,
            a: a.into(),
            b: b.into(),
        });
        self.push(Instruction::Branch {
            label,
            pred: Some(pred),
        });
    }
    fn validate(&self) -> Result<(), KernelError> {
        let mut labels = vec![false; self.labels];
        let bad = || KernelError("typed PTX instruction violated its register contract".into());
        let same = |a: Operand, b: Operand, ty: Type| a.ty() == ty && b.ty() == ty;
        for instruction in &self.instructions {
            let valid = match *instruction {
                Instruction::Parameter { dst, index } => {
                    dst.ty == Type::U64 && index < self.parameters
                }
                Instruction::Move { dst, value } => dst.ty == value.ty(),
                Instruction::Binary { dst, op, a, b } => match op {
                    Binary::WideMul => dst.ty == Type::U64 && same(a, b, Type::U32),
                    Binary::And | Binary::Shr => dst.ty == Type::U32 && same(a, b, Type::U32),
                    Binary::Add | Binary::Sub => {
                        matches!(
                            dst.ty,
                            Type::U32 | Type::I32 | Type::U64 | Type::F32 | Type::F64
                        ) && same(a, b, dst.ty)
                    }
                    Binary::Mul => {
                        matches!(dst.ty, Type::U32 | Type::F32 | Type::F64) && same(a, b, dst.ty)
                    }
                    Binary::Div => matches!(dst.ty, Type::U32 | Type::F32) && same(a, b, dst.ty),
                    Binary::Rem => dst.ty == Type::U32 && same(a, b, Type::U32),
                    Binary::Max => dst.ty == Type::F32 && same(a, b, Type::F32),
                },
                Instruction::Unary { dst, src, .. } => dst.ty == Type::F32 && src.ty == Type::F32,
                Instruction::Convert {
                    dst,
                    conversion,
                    src,
                } => match conversion {
                    Conversion::U32ToI32 => dst.ty == Type::I32 && src.ty == Type::U32,
                    Conversion::I32ToF32 => dst.ty == Type::F32 && src.ty == Type::I32,
                    Conversion::HalfToF32 => dst.ty == Type::F32 && src.ty == Type::B16,
                    Conversion::F32ToF64 => dst.ty == Type::F64 && src.ty == Type::F32,
                    Conversion::F64ToF32 => dst.ty == Type::F32 && src.ty == Type::F64,
                    Conversion::F32ToHalf => dst.ty == Type::B16 && src.ty == Type::F32,
                    Conversion::F32ToI32Trunc => dst.ty == Type::I32 && src.ty == Type::F32,
                },
                Instruction::Load { dst, address, kind } => {
                    address.ty == Type::U64
                        && dst.ty
                            == match kind {
                                LoadType::U8 | LoadType::U32 => Type::U32,
                                LoadType::F16 => Type::B16,
                                LoadType::F32 => Type::F32,
                            }
                }
                Instruction::Store { value, address } => {
                    value.ty == Type::F32 && address.ty == Type::U64
                }
                Instruction::Finite { pred, src } => pred.ty == Type::Pred && src.ty == Type::F32,
                Instruction::GreaterEqual { pred, a, b } => {
                    pred.ty == Type::Pred
                        && matches!(a.ty(), Type::U32 | Type::F32)
                        && a.ty() == b.ty()
                }
                Instruction::Label(label) => {
                    let slot = labels.get_mut(label).ok_or_else(bad)?;
                    if *slot {
                        false
                    } else {
                        *slot = true;
                        true
                    }
                }
                Instruction::Branch { label, pred } => {
                    label < self.labels && pred.is_none_or(|p| p.ty == Type::Pred)
                }
                Instruction::Return => true,
            };
            if !valid {
                return Err(bad());
            }
        }
        if labels.iter().any(|defined| !*defined) {
            return Err(KernelError("undefined branch label".into()));
        }
        Ok(())
    }
    fn emit(&self, target: PtxTarget) -> String {
        let mut text = format!(
            "// Generated from Rust T06 typed IR v{IR_VERSION}; do not edit.\n.version {}.{}\n.target sm_{}\n.address_size 64\n\n.visible .entry rewrite_kernel(\n",
            target.isa_major, target.isa_minor, target.sm
        );
        for index in 0..self.parameters {
            writeln!(
                text,
                "    .param .u64 p{index}{}",
                if index + 1 == self.parameters {
                    ""
                } else {
                    ","
                }
            )
            .unwrap();
        }
        text.push_str(") {\n");
        for r in &self.registers {
            writeln!(text, "    .reg .{} {r};", r.ty.ptx()).unwrap();
        }
        for instruction in &self.instructions {
            match *instruction {
                Instruction::Parameter { dst, index } => {
                    writeln!(text, "    ld.param.u64 {dst}, [p{index}];")
                }
                Instruction::Move { dst, value } => {
                    writeln!(text, "    mov.{} {dst}, {value};", dst.ty.ptx())
                }
                Instruction::Binary { dst, op, a, b } => {
                    let mnemonic = match op {
                        Binary::Add => "add",
                        Binary::Sub => "sub",
                        Binary::Mul => "mul",
                        Binary::WideMul => "mul.wide",
                        Binary::And => "and",
                        Binary::Shr => "shr",
                        Binary::Div => "div",
                        Binary::Rem => "rem",
                        Binary::Max => "max",
                    };
                    let modifier =
                        if matches!(dst.ty, Type::F32 | Type::F64) && !matches!(op, Binary::Max) {
                            ".rn"
                        } else if matches!(op, Binary::Mul) {
                            ".lo"
                        } else {
                            ""
                        };
                    let ty = if matches!(op, Binary::WideMul) {
                        "u32"
                    } else if matches!(op, Binary::And) {
                        "b32"
                    } else {
                        dst.ty.ptx()
                    };
                    writeln!(text, "    {mnemonic}{modifier}.{ty} {dst}, {a}, {b};")
                }
                Instruction::Unary { dst, op, src } => writeln!(
                    text,
                    "    {} {dst}, {src};",
                    match op {
                        Unary::Sqrt => "sqrt.rn.f32",
                        Unary::Exp2 => "ex2.approx.f32",
                    }
                ),
                Instruction::Convert {
                    dst,
                    conversion,
                    src,
                } => writeln!(
                    text,
                    "    {} {dst}, {src};",
                    match conversion {
                        Conversion::U32ToI32 => "cvt.s32.u32",
                        Conversion::I32ToF32 => "cvt.rn.f32.s32",
                        Conversion::HalfToF32 => "cvt.f32.f16",
                        Conversion::F32ToF64 => "cvt.f64.f32",
                        Conversion::F64ToF32 => "cvt.rn.f32.f64",
                        Conversion::F32ToHalf => "cvt.rn.f16.f32",
                        Conversion::F32ToI32Trunc => "cvt.rzi.s32.f32",
                    }
                ),
                Instruction::Load { dst, address, kind } => writeln!(
                    text,
                    "    ld.global.{} {dst}, [{address}];",
                    match kind {
                        LoadType::U8 => "u8",
                        LoadType::U32 => "u32",
                        LoadType::F16 => "b16",
                        LoadType::F32 => "f32",
                    }
                ),
                Instruction::Store { value, address } => {
                    writeln!(text, "    st.global.f32 [{address}], {value};")
                }
                Instruction::Finite { pred, src } => {
                    writeln!(text, "    testp.finite.f32 {pred}, {src};")
                }
                Instruction::GreaterEqual { pred, a, b } => {
                    writeln!(text, "    setp.ge.{} {pred}, {a}, {b};", a.ty().ptx())
                }
                Instruction::Label(label) => writeln!(text, "L{label}:"),
                Instruction::Branch {
                    label,
                    pred: Some(pred),
                } => writeln!(text, "    @{pred} bra L{label};"),
                Instruction::Branch { label, pred: None } => writeln!(text, "    bra L{label};"),
                Instruction::Return => writeln!(text, "    ret;"),
            }
            .unwrap();
        }
        text.push_str("}\n");
        text
    }
}

/// Source, IR and buffer contract cannot be independently changed by callers.
#[derive(Debug, Clone)]
pub struct GeneratedPtx {
    spec: KernelSpec,
    target: PtxTarget,
    program: Program,
    source: String,
    bytes: Vec<usize>,
}
impl GeneratedPtx {
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn spec(&self) -> &KernelSpec {
        &self.spec
    }
    pub fn target(&self) -> PtxTarget {
        self.target
    }
    pub fn entry_point(&self) -> &'static str {
        "rewrite_kernel"
    }
    pub fn buffer_bytes(&self) -> &[usize] {
        &self.bytes
    }
    pub fn output_buffer(&self) -> usize {
        self.spec.output_buffer()
    }
    pub fn launch_dimensions(&self) -> (u32, u32) {
        (self.spec.outputs().div_ceil(256), 256)
    }
    pub fn validate_buffers(&self, buffers: &[Vec<u8>]) -> Result<(), KernelError> {
        if buffers.len() != self.bytes.len() {
            return Err(KernelError("incorrect number of kernel buffers".into()));
        }
        for (index, (buffer, bytes)) in buffers.iter().zip(&self.bytes).enumerate() {
            if buffer.len() < *bytes {
                return Err(KernelError(format!(
                    "buffer {index} has {} bytes, needs {bytes}",
                    buffer.len()
                )));
            }
        }
        Ok(())
    }
    /// Interpret the very instruction tree emitted to PTX. This checks codegen
    /// semantics on CPU; it is intentionally named differently from GPU launch.
    /// A caller-supplied fuel limit bounds malformed/huge workloads.
    pub fn interpret(
        &self,
        buffers: &mut [Vec<u8>],
        max_instructions: u64,
    ) -> Result<u64, KernelError> {
        self.validate_buffers(buffers)?;
        self.program
            .interpret(buffers, self.launch_dimensions(), max_instructions)
    }
}

pub fn generate(spec: &KernelSpec, target: PtxTarget) -> Result<GeneratedPtx, KernelError> {
    target.validate()?;
    let bytes = spec.buffer_bytes()?;
    let mut p = Program {
        parameters: bytes.len(),
        ..Program::default()
    };
    let pointers: Vec<Reg> = (0..bytes.len()).map(|i| p.param(i)).collect();
    let block = p.value(Operand::Special(Special::BlockX));
    let width = p.value(Operand::Special(Special::BlockWidth));
    let tid = p.value(Operand::Special(Special::ThreadX));
    let base = p.binary(Binary::Mul, block, width);
    let index = p.binary(Binary::Add, base, tid);
    let done = p.label();
    p.guard_ge(index, spec.outputs(), done);
    match spec.kind {
        KernelKind::Decoder(ref kind) => decoder::emit(kind, &mut p, &pointers, index, done),
        KernelKind::AddF32 {
            lhs_stride,
            rhs_stride,
            output_stride,
            ..
        } => {
            let a = p.address(pointers[0], index, lhs_stride * 4);
            let a = p.load(a, LoadType::F32);
            let b = p.address(pointers[1], index, rhs_stride * 4);
            let b = p.load(b, LoadType::F32);
            let value = p.binary(Binary::Add, a, b);
            let address = p.address(pointers[2], index, output_stride * 4);
            p.push(Instruction::Store { value, address });
        }
        KernelKind::ScaleF32 {
            input_stride,
            output_stride,
            scale,
            ..
        } => {
            let address = p.address(pointers[0], index, input_stride * 4);
            let value = p.load(address, LoadType::F32);
            let value = p.binary(Binary::Mul, value, Operand::F32(scale));
            let address = p.address(pointers[1], index, output_stride * 4);
            p.push(Instruction::Store { value, address });
        }
        KernelKind::SumF32 {
            elements,
            input_stride,
        } => {
            let cursor = p.value(0.into());
            let sum = p.value(Operand::F32(0.0));
            let loop_start = p.label();
            let loop_end = p.label();
            p.push(Instruction::Label(loop_start));
            p.guard_ge(cursor, elements, loop_end);
            let address = p.address(pointers[0], cursor, input_stride * 4);
            let value = p.load(address, LoadType::F32);
            p.assign_binary(sum, Binary::Add, sum, value);
            p.assign_binary(cursor, Binary::Add, cursor, 1u32);
            p.push(Instruction::Branch {
                label: loop_start,
                pred: None,
            });
            p.push(Instruction::Label(loop_end));
            p.push(Instruction::Store {
                value: sum,
                address: pointers[1],
            });
        }
        KernelKind::MatVecF32 {
            columns,
            row_stride,
            column_stride,
            vector_stride,
            output_stride,
            ..
        } => {
            let row = p.address(pointers[0], index, row_stride * 4);
            let cursor = p.value(0.into());
            let sum = p.value(Operand::F32(0.0));
            let loop_start = p.label();
            let loop_end = p.label();
            p.push(Instruction::Label(loop_start));
            p.guard_ge(cursor, columns, loop_end);
            let matrix_address = p.address(row, cursor, column_stride * 4);
            let matrix_value = p.load(matrix_address, LoadType::F32);
            let vector_address = p.address(pointers[1], cursor, vector_stride * 4);
            let vector_value = p.load(vector_address, LoadType::F32);
            let product = p.binary(Binary::Mul, matrix_value, vector_value);
            p.assign_binary(sum, Binary::Add, sum, product);
            p.assign_binary(cursor, Binary::Add, cursor, 1u32);
            p.push(Instruction::Branch {
                label: loop_start,
                pred: None,
            });
            p.push(Instruction::Label(loop_end));
            let address = p.address(pointers[2], index, output_stride * 4);
            p.push(Instruction::Store {
                value: sum,
                address,
            });
        }
        KernelKind::Q4MatVec {
            columns,
            row_stride_bytes,
            block_stride_bytes,
            vector_stride,
            ..
        } => {
            let row = p.address(pointers[0], index, row_stride_bytes);
            let block_index = p.value(0.into());
            let sum = p.value(Operand::F32(0.0));
            let block_start = p.label();
            let block_end = p.label();
            p.push(Instruction::Label(block_start));
            p.guard_ge(block_index, columns / 32, block_end);
            let block = p.address(row, block_index, block_stride_bytes);
            let half = p.load(block, LoadType::F16);
            let scale = p.convert(Conversion::HalfToF32, half);
            let block_column = p.binary(Binary::Mul, block_index, 32u32);
            let nibble = p.value(0.into());
            let nibble_start = p.label();
            let nibble_end = p.label();
            p.push(Instruction::Label(nibble_start));
            p.guard_ge(nibble, 16u32, nibble_end);
            let byte_offset = p.binary(Binary::Add, nibble, 2u32);
            let q_address = p.address(block, byte_offset, 1);
            let packed = p.load(q_address, LoadType::U8);
            let low = p.binary(Binary::And, packed, 15u32);
            let high = p.binary(Binary::Shr, packed, 4u32);
            let low_col = p.binary(Binary::Add, block_column, nibble);
            let high_col = p.binary(Binary::Add, low_col, 16u32);
            for (quant, column) in [(low, low_col), (high, high_col)] {
                let signed = p.convert(Conversion::U32ToI32, quant);
                let centered = p.binary(Binary::Sub, signed, Operand::I32(8));
                let value = p.convert(Conversion::I32ToF32, centered);
                let value = p.binary(Binary::Mul, value, scale);
                let x_address = p.address(pointers[1], column, vector_stride * 4);
                let x = p.load(x_address, LoadType::F32);
                let product = p.binary(Binary::Mul, value, x);
                p.assign_binary(sum, Binary::Add, sum, product);
            }
            p.assign_binary(nibble, Binary::Add, nibble, 1u32);
            p.push(Instruction::Branch {
                label: nibble_start,
                pred: None,
            });
            p.push(Instruction::Label(nibble_end));
            p.assign_binary(block_index, Binary::Add, block_index, 1u32);
            p.push(Instruction::Branch {
                label: block_start,
                pred: None,
            });
            p.push(Instruction::Label(block_end));
            let address = p.address(pointers[2], index, 4);
            p.push(Instruction::Store {
                value: sum,
                address,
            });
        }
    }
    p.push(Instruction::Label(done));
    p.push(Instruction::Return);
    p.validate()?;
    let source = p.emit(target);
    Ok(GeneratedPtx {
        spec: spec.clone(),
        target,
        program: p,
        source,
        bytes,
    })
}

fn half_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 31) as i32;
    let fraction = (half & 1023) as u32;
    if exponent == 0 {
        if fraction == 0 {
            f32::from_bits(sign)
        } else {
            let value = (fraction as f32) * 2f32.powi(-24);
            if sign == 0 { value } else { -value }
        }
    } else if exponent == 31 {
        f32::from_bits(sign | 0x7f800000 | (fraction << 13))
    } else {
        f32::from_bits(sign | (((exponent + 112) as u32) << 23) | (fraction << 13))
    }
}

impl Program {
    fn interpret(
        &self,
        buffers: &mut [Vec<u8>],
        (grid, block): (u32, u32),
        fuel: u64,
    ) -> Result<u64, KernelError> {
        // Fake addresses identify interpreter-owned byte arrays only. They are
        // never passed to a driver or called a real device buffer.
        const SHIFT: u32 = 56;
        const MASK: u64 = (1u64 << SHIFT) - 1;
        if buffers.iter().any(|b| b.len() as u128 > MASK as u128) {
            return Err(KernelError(
                "interpreter buffer exceeds address domain".into(),
            ));
        }
        let mut labels = vec![0usize; self.labels];
        for (i, ins) in self.instructions.iter().enumerate() {
            if let Instruction::Label(l) = ins {
                labels[*l] = i;
            }
        }
        let mut used = 0u64;
        for cta in 0..grid {
            for tid in 0..block {
                let mut regs = vec![0u64; self.registers.len()];
                let mut pc = 0;
                let operand = |value: Operand, regs: &[u64]| -> u64 {
                    match value {
                        Operand::Reg(r) => regs[r.index],
                        Operand::U32(v) => v as u64,
                        Operand::I32(v) => v as u32 as u64,
                        Operand::F32(v) => v.to_bits() as u64,
                        Operand::F64(v) => v.to_bits(),
                        Operand::Special(s) => match s {
                            Special::ThreadX => tid as u64,
                            Special::BlockX => cta as u64,
                            Special::BlockWidth => block as u64,
                        },
                    }
                };
                while pc < self.instructions.len() {
                    if used >= fuel {
                        return Err(KernelError(
                            "instruction interpreter exhausted its fuel limit".into(),
                        ));
                    }
                    used += 1;
                    match self.instructions[pc] {
                        Instruction::Parameter { dst, index } => {
                            regs[dst.index] = ((index + 1) as u64) << SHIFT
                        }
                        Instruction::Move { dst, value } => regs[dst.index] = operand(value, &regs),
                        Instruction::Binary { dst, op, a, b } => {
                            let (a, b) = (operand(a, &regs), operand(b, &regs));
                            regs[dst.index] = if dst.ty == Type::F32 {
                                let (a, b) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                                match op {
                                    Binary::Add => a + b,
                                    Binary::Sub => a - b,
                                    Binary::Mul => a * b,
                                    Binary::Div => a / b,
                                    Binary::Max => a.max(b),
                                    _ => return Err(KernelError("invalid float opcode".into())),
                                }
                                .to_bits() as u64
                            } else if dst.ty == Type::F64 {
                                let (a, b) = (f64::from_bits(a), f64::from_bits(b));
                                match op {
                                    Binary::Add => a + b,
                                    Binary::Sub => a - b,
                                    Binary::Mul => a * b,
                                    _ => return Err(KernelError("invalid F64 opcode".into())),
                                }
                                .to_bits()
                            } else {
                                let result = match op {
                                    Binary::Add => a.wrapping_add(b),
                                    Binary::Sub => a.wrapping_sub(b),
                                    Binary::Mul | Binary::WideMul => a.wrapping_mul(b),
                                    Binary::And => a & b,
                                    Binary::Div => a.checked_div(b).ok_or_else(|| {
                                        KernelError("integer division by zero".into())
                                    })?,
                                    Binary::Rem => a.checked_rem(b).ok_or_else(|| {
                                        KernelError("integer remainder by zero".into())
                                    })?,
                                    Binary::Max => {
                                        return Err(KernelError("invalid integer opcode".into()));
                                    }
                                    Binary::Shr => {
                                        (a as u32).checked_shr(b as u32).unwrap_or(0) as u64
                                    }
                                };
                                if matches!(dst.ty, Type::U32 | Type::I32) {
                                    result as u32 as u64
                                } else {
                                    result
                                }
                            };
                        }
                        Instruction::Unary { dst, op, src } => {
                            let value = f32::from_bits(regs[src.index] as u32);
                            regs[dst.index] = match op {
                                Unary::Sqrt => value.sqrt(),
                                Unary::Exp2 => value.exp2(),
                            }
                            .to_bits() as u64;
                        }
                        Instruction::Convert {
                            dst,
                            conversion,
                            src,
                        } => {
                            let value = regs[src.index];
                            regs[dst.index] = match conversion {
                                Conversion::U32ToI32 => value as u32 as i32 as u32 as u64,
                                Conversion::I32ToF32 => {
                                    (value as u32 as i32 as f32).to_bits() as u64
                                }
                                Conversion::HalfToF32 => half_to_f32(value as u16).to_bits() as u64,
                                Conversion::F32ToF64 => {
                                    f64::from(f32::from_bits(value as u32)).to_bits()
                                }
                                Conversion::F64ToF32 => {
                                    (f64::from_bits(value) as f32).to_bits() as u64
                                }
                                Conversion::F32ToHalf => {
                                    llama_quant::f32_to_f16(f32::from_bits(value as u32)) as u64
                                }
                                Conversion::F32ToI32Trunc => {
                                    (f32::from_bits(value as u32) as i32) as u32 as u64
                                }
                            };
                        }
                        Instruction::Load { dst, address, kind } => {
                            let address = regs[address.index];
                            let (index, offset) =
                                decode_address(address, SHIFT, MASK, buffers.len())?;
                            let width = match kind {
                                LoadType::U8 => 1,
                                LoadType::F16 => 2,
                                LoadType::F32 | LoadType::U32 => 4,
                            };
                            let end = offset
                                .checked_add(width)
                                .ok_or_else(|| KernelError("interpreter read overflow".into()))?;
                            let bytes = buffers[index].get(offset..end).ok_or_else(|| {
                                KernelError("interpreter out-of-bounds read".into())
                            })?;
                            regs[dst.index] = match kind {
                                LoadType::U8 => bytes[0] as u64,
                                LoadType::F16 => {
                                    u16::from_le_bytes(bytes.try_into().unwrap()) as u64
                                }
                                LoadType::F32 | LoadType::U32 => {
                                    u32::from_le_bytes(bytes.try_into().unwrap()) as u64
                                }
                            };
                        }
                        Instruction::Store { value, address } => {
                            let (index, offset) =
                                decode_address(regs[address.index], SHIFT, MASK, buffers.len())?;
                            let end = offset
                                .checked_add(4)
                                .ok_or_else(|| KernelError("interpreter write overflow".into()))?;
                            let output = buffers[index].get_mut(offset..end).ok_or_else(|| {
                                KernelError("interpreter out-of-bounds write".into())
                            })?;
                            output.copy_from_slice(&(regs[value.index] as u32).to_le_bytes());
                        }
                        Instruction::Finite { pred, src } => {
                            regs[pred.index] =
                                u64::from(f32::from_bits(regs[src.index] as u32).is_finite());
                        }
                        Instruction::GreaterEqual { pred, a, b } => {
                            regs[pred.index] = if a.ty() == Type::F32 {
                                u64::from(
                                    f32::from_bits(operand(a, &regs) as u32)
                                        >= f32::from_bits(operand(b, &regs) as u32),
                                )
                            } else {
                                u64::from((operand(a, &regs) as u32) >= (operand(b, &regs) as u32))
                            }
                        }
                        Instruction::Label(_) => {}
                        Instruction::Branch { label, pred } => {
                            if pred.is_none_or(|p| regs[p.index] != 0) {
                                pc = labels[label];
                                continue;
                            }
                        }
                        Instruction::Return => break,
                    }
                    pc += 1;
                }
            }
        }
        Ok(used)
    }
}
fn decode_address(
    address: u64,
    shift: u32,
    mask: u64,
    count: usize,
) -> Result<(usize, usize), KernelError> {
    let number = (address >> shift) as usize;
    let offset = (address & mask) as usize;
    if number == 0 || number > count {
        return Err(KernelError(
            "interpreter encountered an invalid address".into(),
        ));
    }
    Ok((number - 1, offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
    fn compiled(kind: KernelKind) -> GeneratedPtx {
        generate(&KernelSpec { kind }, PtxTarget::default()).unwrap()
    }
    #[test]
    fn noncontiguous_add_preserves_padding_and_handles_tail() {
        let k = compiled(KernelKind::AddF32 {
            elements: 3,
            lhs_stride: 2,
            rhs_stride: 0,
            output_stride: 2,
        });
        let mut b = vec![
            f32_bytes(&[1., 99., 2., 99., 3.]),
            f32_bytes(&[10.]),
            f32_bytes(&[0., 123., 0., 123., 0.]),
        ];
        k.interpret(&mut b, 10_000).unwrap();
        assert_eq!(f32_values(&b[2]), vec![11., 123., 12., 123., 13.]);
        assert!(k.source().contains("ld.global.f32"));
        assert!(!k.source().contains("#include"));
    }
    #[test]
    fn scale_emits_exact_float_bits_and_crosses_block_boundary() {
        let k = compiled(KernelKind::ScaleF32 {
            elements: 257,
            input_stride: 1,
            output_stride: 1,
            scale: -0.25,
        });
        let input: Vec<f32> = (0..257).map(|i| i as f32).collect();
        let mut b = vec![f32_bytes(&input), vec![0; 257 * 4]];
        k.interpret(&mut b, 30_000).unwrap();
        assert_eq!(
            f32_values(&b[1]),
            input.iter().map(|x| x * (-0.25)).collect::<Vec<_>>()
        );
        assert!(k.source().contains("0fbe800000"));
        assert_eq!(k.launch_dimensions(), (2, 256));
    }
    #[test]
    fn sum_runs_generated_loop_with_strided_input() {
        let k = compiled(KernelKind::SumF32 {
            elements: 4,
            input_stride: 2,
        });
        let mut b = vec![f32_bytes(&[1., 100., 2., 100., -4., 100., 8.]), vec![0; 4]];
        k.interpret(&mut b, 10_000).unwrap();
        assert_eq!(f32_values(&b[1]), [7.]);
    }
    #[test]
    fn q4_zero_negative_and_subnormal_scales_decode() {
        assert_eq!(half_to_f32(0x3c00), 1.);
        assert_eq!(half_to_f32(0xb800), -0.5);
        assert_eq!(half_to_f32(1), 2f32.powi(-24));
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
        assert_eq!(half_to_f32(0x8000).to_bits(), (-0f32).to_bits());
    }
    #[test]
    fn noncontiguous_q4_matvec_matches_independent_dense_dot() {
        let k = compiled(KernelKind::Q4MatVec {
            rows: 2,
            columns: 64,
            row_stride_bytes: 48,
            block_stride_bytes: 22,
            vector_stride: 2,
        });
        let mut weights = vec![0xa5; k.buffer_bytes()[0]];
        let scales = [0x3c00u16, 0x3800, 0xc000, 0x3400];
        let mut dense = vec![vec![0f64; 64]; 2];
        for row in 0..2 {
            for block in 0..2 {
                let offset = row * 48 + block * 22;
                weights[offset..offset + 2].copy_from_slice(&scales[row * 2 + block].to_le_bytes());
                let scale = [1., 0.5, -2., 0.25][row * 2 + block];
                for j in 0..16 {
                    let low = ((j + row + block) % 16) as u8;
                    let high = ((15 - j + row) % 16) as u8;
                    weights[offset + 2 + j] = low | (high << 4);
                    dense[row][block * 32 + j] = (low as f64 - 8.) * scale;
                    dense[row][block * 32 + j + 16] = (high as f64 - 8.) * scale;
                }
            }
        }
        let mut x = vec![777f32; 127];
        for j in 0..64 {
            x[j * 2] = (j as f32 - 30.) / 8.;
        }
        let expected: Vec<f64> = dense
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, w)| w * x[j * 2] as f64)
                    .sum()
            })
            .collect();
        let mut b = vec![weights, f32_bytes(&x), vec![0; 8]];
        k.interpret(&mut b, 50_000).unwrap();
        for (actual, expected) in f32_values(&b[2]).iter().zip(expected) {
            assert!(
                (*actual as f64 - expected).abs() < 0.0001,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn q4_nibbles_match_frozen_c_oracle_exactly() {
        // Recorded by invoking the original dequantize_row_q4_0 function from
        // the uploaded archive, not by this crate's FP16 or quantized decoder.
        let packed = include_bytes!("../fixtures/q4_0_reference_input.bin");
        let expected = include_bytes!("../fixtures/q4_0_reference_f32.bin");
        let expected = f32_values(expected);
        let k = compiled(KernelKind::Q4MatVec {
            rows: 4,
            columns: 64,
            row_stride_bytes: 36,
            block_stride_bytes: 18,
            vector_stride: 1,
        });
        // A basis vector probes every column across all rows. This checks the
        // half-scale and low/high nibble order without sharing a dot formula.
        for column in 0..64 {
            let mut x = vec![0f32; 64];
            x[column] = 1.0;
            let mut buffers = vec![packed.to_vec(), f32_bytes(&x), vec![0; 16]];
            k.interpret(&mut buffers, 100_000).unwrap();
            for (row, actual) in f32_values(&buffers[2]).iter().enumerate() {
                assert_eq!(
                    *actual,
                    expected[row * 64 + column],
                    "row {row}, column {column}"
                );
            }
        }
    }
    #[test]
    fn rejects_invalid_shapes_overflow_and_aliasing() {
        for kind in [
            KernelKind::AddF32 {
                elements: 0,
                lhs_stride: 1,
                rhs_stride: 1,
                output_stride: 1,
            },
            KernelKind::ScaleF32 {
                elements: 2,
                input_stride: u32::MAX,
                output_stride: 1,
                scale: 1.,
            },
            KernelKind::ScaleF32 {
                elements: 2,
                input_stride: 1,
                output_stride: 0,
                scale: 1.,
            },
            KernelKind::Q4MatVec {
                rows: 1,
                columns: 31,
                row_stride_bytes: 18,
                block_stride_bytes: 18,
                vector_stride: 1,
            },
            KernelKind::Q4MatVec {
                rows: 2,
                columns: 64,
                row_stride_bytes: 18,
                block_stride_bytes: 18,
                vector_stride: 1,
            },
        ] {
            assert!(generate(&KernelSpec { kind }, PtxTarget::default()).is_err());
        }
    }
    #[test]
    fn bad_buffers_and_instruction_budget_fail_explicitly() {
        let k = compiled(KernelKind::SumF32 {
            elements: 4,
            input_stride: 1,
        });
        assert!(k.validate_buffers(&[vec![0; 12], vec![0; 4]]).is_err());
        assert!(k.validate_buffers(&[vec![0; 16]]).is_err());
        let mut b = vec![f32_bytes(&[1., 2., 3., 4.]), vec![0; 4]];
        assert!(k.interpret(&mut b, 1).is_err());
    }
    #[test]
    fn target_version_is_not_inferred_from_device_family() {
        assert!(
            PtxTarget {
                sm: 89,
                isa_major: 7,
                isa_minor: 0
            }
            .validate()
            .is_err()
        );
        assert!(
            PtxTarget {
                sm: 89,
                isa_major: 7,
                isa_minor: 8
            }
            .validate()
            .is_ok()
        );
        assert!(
            PtxTarget {
                sm: 999,
                isa_major: 8,
                isa_minor: 0
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn malformed_typed_program_cannot_be_emitted() {
        let mut p = Program::default();
        let a = p.value(Operand::F32(1.));
        let b = p.value(1.into());
        p.binary(Binary::Add, a, b);
        assert!(p.validate().is_err());
        let mut p = Program::default();
        p.label();
        assert!(p.validate().is_err());
    }

    #[test]
    fn f32_matvec_strides_broadcast_and_tail_match_independent_dot() {
        for (rows, cols, matrix_stride, vector_stride) in
            [(3u32, 5u32, 13u32, 2u32), (259, 7, 17, 1), (2, 5, 0, 0)]
        {
            let spec = KernelSpec {
                kind: KernelKind::MatVecF32 {
                    rows,
                    columns: cols,
                    row_stride: matrix_stride,
                    column_stride: 2,
                    vector_stride,
                    output_stride: 3,
                },
            };
            let sizes = spec.buffer_bytes().unwrap();
            let matrix: Vec<f32> = (0..sizes[0] / 4)
                .map(|i| (i as i32 % 19 - 9) as f32 * 0.125)
                .collect();
            let vector: Vec<f32> = (0..sizes[1] / 4)
                .map(|i| (i as i32 % 7 - 3) as f32 * 0.25)
                .collect();
            let mut expected = vec![-991.0f32; sizes[2] / 4 + 5];
            for row in 0..rows as usize {
                expected[row * 3] = (0..cols as usize)
                    .map(|col| {
                        matrix[row * matrix_stride as usize + col * 2] as f64
                            * vector[col * vector_stride as usize] as f64
                    })
                    .sum::<f64>() as f32;
            }
            let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let mut buffers = vec![
                bytes(&matrix),
                bytes(&vector),
                bytes(&vec![-991.0; expected.len()]),
            ];
            let generated = generate(&spec, PtxTarget::default()).unwrap();
            generated.interpret(&mut buffers, 1_000_000).unwrap();
            let actual: Vec<f32> = buffers[2]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn f32_matvec_rejects_invalid_storage_and_output_aliasing() {
        let base = KernelKind::MatVecF32 {
            rows: 2,
            columns: 3,
            row_stride: 3,
            column_stride: 1,
            vector_stride: 1,
            output_stride: 1,
        };
        for field in 0..6 {
            let mut kind = base.clone();
            let KernelKind::MatVecF32 {
                rows,
                columns,
                row_stride,
                column_stride,
                vector_stride,
                output_stride,
            } = &mut kind
            else {
                unreachable!()
            };
            match field {
                0 => *rows = 0,
                1 => *columns = 0,
                2 => *row_stride = u32::MAX,
                3 => *column_stride = u32::MAX,
                4 => *vector_stride = u32::MAX,
                _ => *output_stride = 0,
            }
            assert!(generate(&KernelSpec { kind }, PtxTarget::default()).is_err());
        }
    }
}
