//! Small concrete kernel family, authored only as typed Rust IR nodes.
use crate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorLayout {
    pub length: u32,
    pub offset: u32,
    pub stride: u32,
}
impl VectorLayout {
    pub fn contiguous(length: u32) -> Self {
        Self {
            length,
            offset: 0,
            stride: 1,
        }
    }
    pub fn storage_elements(self) -> Result<u32> {
        if self.length == 0 {
            return Err(Error("empty vector".into()));
        }
        (self.length - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.offset))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error("vector address overflows U32".into()))
    }
    fn at(self, index: Expr) -> Expr {
        u(self.offset).binary(Binary::Add, index.binary(Binary::Mul, u(self.stride)))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: u32,
    pub cols: u32,
    pub offset: u32,
    pub row_stride: u32,
    pub col_stride: u32,
}
impl MatrixLayout {
    pub fn contiguous(rows: u32, cols: u32) -> Self {
        Self {
            rows,
            cols,
            offset: 0,
            row_stride: cols,
            col_stride: 1,
        }
    }
    pub fn storage_elements(self) -> Result<u32> {
        if self.rows == 0 || self.cols == 0 {
            return Err(Error("empty matrix".into()));
        }
        (self.rows - 1)
            .checked_mul(self.row_stride)
            .and_then(|n| {
                (self.cols - 1)
                    .checked_mul(self.col_stride)
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(self.offset))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error("matrix address overflows U32".into()))
    }
    fn at(self, row: Expr, col: Expr) -> Expr {
        u(self.offset)
            .binary(Binary::Add, row.binary(Binary::Mul, u(self.row_stride)))
            .binary(Binary::Add, col.binary(Binary::Mul, u(self.col_stride)))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pointwise {
    Add,
    Mul,
    Silu,
    Scale,
}

/// A safe device-dispatch plan. Construction is restricted to the built-in
/// builders whose address arithmetic and per-invocation write ownership are
/// established from checked layouts. Arbitrary public IR stays available for
/// code generation and bounded interpretation, but cannot become this type.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    kernel: Kernel,
}
impl DispatchPlan {
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
    pub fn pointwise(
        kind: Pointwise,
        lhs: VectorLayout,
        rhs: Option<VectorLayout>,
        output: VectorLayout,
        scale: f32,
    ) -> Result<Self> {
        Ok(Self {
            kernel: pointwise(kind, lhs, rhs, output, scale)?,
        })
    }
    pub fn row_sum(input: MatrixLayout) -> Result<Self> {
        Ok(Self {
            kernel: row_sum(input)?,
        })
    }
    pub fn rms_norm(input: MatrixLayout, epsilon: f32) -> Result<Self> {
        Ok(Self {
            kernel: rms_norm(input, epsilon)?,
        })
    }
    pub fn softmax(input: MatrixLayout, scale: f32) -> Result<Self> {
        Ok(Self {
            kernel: softmax(input, scale)?,
        })
    }
    pub fn matmul(lhs: MatrixLayout, rhs: MatrixLayout) -> Result<Self> {
        Ok(Self {
            kernel: matmul(lhs, rhs)?,
        })
    }
    pub fn q4_0_matvec(
        rows: u32,
        cols: u32,
        offset_bytes: u32,
        row_stride_bytes: u32,
        input: VectorLayout,
    ) -> Result<Self> {
        Ok(Self {
            kernel: q4_0_matvec(rows, cols, offset_bytes, row_stride_bytes, input)?,
        })
    }
    pub fn conformance_corpus() -> Result<Vec<Self>> {
        // conformance_kernels uses only the private checked built-in builders.
        Ok(conformance_kernels()?
            .into_iter()
            .map(|kernel| Self { kernel })
            .collect())
    }
}

fn u(value: u32) -> Expr {
    Expr::Uint(value)
}
fn f(value: f32) -> Expr {
    Expr::Float(value)
}
fn v(name: &str) -> Expr {
    Expr::var(name)
}
fn declare(name: &str, ty: Scalar, value: Expr) -> Stmt {
    Stmt::Declare {
        name: name.into(),
        ty,
        value,
    }
}
fn assign(name: &str, value: Expr) -> Stmt {
    Stmt::Assign {
        name: name.into(),
        value,
    }
}
fn store(buffer: &str, index: Expr, value: Expr) -> Stmt {
    Stmt::Store {
        buffer: buffer.into(),
        index,
        value,
    }
}
fn loop_(index: &str, end: u32, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        index: index.into(),
        end: u(end),
        body,
    }
}
fn buffer(name: &str, scalar: Scalar, access: Access, min_elements: u32) -> Buffer {
    Buffer {
        name: name.into(),
        scalar,
        access,
        min_elements,
    }
}
fn new(
    name: &str,
    invocations: u32,
    buffers: Vec<Buffer>,
    body: Vec<Stmt>,
    source: &str,
) -> Result<Kernel> {
    let kernel = Kernel {
        version: IR_VERSION,
        name: name.into(),
        invocations,
        workgroup_size: 64,
        buffers,
        body,
        source_refs: vec![source.into()],
    };
    kernel.validate()?;
    Ok(kernel)
}

pub fn pointwise(
    kind: Pointwise,
    lhs: VectorLayout,
    rhs: Option<VectorLayout>,
    output: VectorLayout,
    scale: f32,
) -> Result<Kernel> {
    if lhs.length != output.length || output.stride == 0 {
        return Err(Error("pointwise output shape/stride mismatch".into()));
    }
    let mut buffers = vec![buffer(
        "lhs",
        Scalar::F32,
        Access::Read,
        lhs.storage_elements()?,
    )];
    let left = Expr::load("lhs", lhs.at(Expr::GlobalId));
    let value = match kind {
        Pointwise::Add | Pointwise::Mul => {
            let rhs = rhs.ok_or_else(|| Error("binary pointwise requires rhs".into()))?;
            if rhs.length != lhs.length {
                return Err(Error("pointwise input length mismatch".into()));
            }
            buffers.push(buffer(
                "rhs",
                Scalar::F32,
                Access::Read,
                rhs.storage_elements()?,
            ));
            left.binary(
                if kind == Pointwise::Add {
                    Binary::Add
                } else {
                    Binary::Mul
                },
                Expr::load("rhs", rhs.at(Expr::GlobalId)),
            )
        }
        Pointwise::Silu => left.clone().binary(
            Binary::Div,
            f(1.0).binary(Binary::Add, left.unary(Unary::Neg).unary(Unary::Exp)),
        ),
        Pointwise::Scale => {
            if !scale.is_finite() {
                return Err(Error("nonfinite scale".into()));
            }
            left.binary(Binary::Mul, f(scale))
        }
    };
    buffers.push(buffer(
        "dst",
        Scalar::F32,
        Access::ReadWrite,
        output.storage_elements()?,
    ));
    new(
        match kind {
            Pointwise::Add => "add_f32",
            Pointwise::Mul => "mul_f32",
            Pointwise::Silu => "silu_f32",
            Pointwise::Scale => "scale_f32",
        },
        lhs.length,
        buffers,
        vec![store("dst", output.at(Expr::GlobalId), value)],
        "ggml/src/ggml-cpu/ops.cpp: pointwise operation semantics",
    )
}

pub fn row_sum(input: MatrixLayout) -> Result<Kernel> {
    new(
        "row_sum_f32",
        input.rows,
        vec![
            buffer("src", Scalar::F32, Access::Read, input.storage_elements()?),
            buffer("dst", Scalar::F32, Access::ReadWrite, input.rows),
        ],
        vec![
            declare("sum", Scalar::F32, f(0.0)),
            loop_(
                "col",
                input.cols,
                vec![assign(
                    "sum",
                    v("sum").binary(
                        Binary::Add,
                        Expr::load("src", input.at(Expr::GlobalId, v("col"))),
                    ),
                )],
            ),
            store("dst", Expr::GlobalId, v("sum")),
        ],
        "ggml/src/ggml-cpu/ops.cpp: SUM_ROWS",
    )
}

pub fn rms_norm(input: MatrixLayout, epsilon: f32) -> Result<Kernel> {
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(Error("RMS epsilon must be finite and positive".into()));
    }
    let count = input
        .rows
        .checked_mul(input.cols)
        .ok_or_else(|| Error("RMS output overflow".into()))?;
    new(
        "rms_norm_f32",
        input.rows,
        vec![
            buffer("src", Scalar::F32, Access::Read, input.storage_elements()?),
            buffer("dst", Scalar::F32, Access::ReadWrite, count),
        ],
        vec![
            declare("sum", Scalar::F32, f(0.0)),
            loop_(
                "col",
                input.cols,
                vec![
                    declare(
                        "value",
                        Scalar::F32,
                        Expr::load("src", input.at(Expr::GlobalId, v("col"))),
                    ),
                    assign(
                        "sum",
                        v("sum").binary(Binary::Add, v("value").binary(Binary::Mul, v("value"))),
                    ),
                ],
            ),
            declare(
                "factor",
                Scalar::F32,
                v("sum")
                    .binary(Binary::Div, f(input.cols as f32))
                    .binary(Binary::Add, f(epsilon))
                    .unary(Unary::Rsqrt),
            ),
            loop_(
                "col",
                input.cols,
                vec![store(
                    "dst",
                    Expr::GlobalId
                        .binary(Binary::Mul, u(input.cols))
                        .binary(Binary::Add, v("col")),
                    Expr::load("src", input.at(Expr::GlobalId, v("col")))
                        .binary(Binary::Mul, v("factor")),
                )],
            ),
        ],
        "ggml/src/ggml-cpu/ops.cpp: RMS_NORM",
    )
}

pub fn softmax(input: MatrixLayout, scale: f32) -> Result<Kernel> {
    if !scale.is_finite() {
        return Err(Error("nonfinite softmax scale".into()));
    }
    let count = input
        .rows
        .checked_mul(input.cols)
        .ok_or_else(|| Error("softmax output overflow".into()))?;
    let value =
        |col: Expr| Expr::load("src", input.at(Expr::GlobalId, col)).binary(Binary::Mul, f(scale));
    let exp = |col: Expr| {
        value(col)
            .binary(Binary::Sub, v("maximum"))
            .unary(Unary::Exp)
    };
    new(
        "softmax_f32",
        input.rows,
        vec![
            buffer("src", Scalar::F32, Access::Read, input.storage_elements()?),
            buffer("dst", Scalar::F32, Access::ReadWrite, count),
        ],
        vec![
            declare("maximum", Scalar::F32, value(u(0))),
            loop_(
                "col",
                input.cols,
                vec![assign(
                    "maximum",
                    Expr::Max(Box::new(v("maximum")), Box::new(value(v("col")))),
                )],
            ),
            declare("sum", Scalar::F32, f(0.0)),
            loop_(
                "col",
                input.cols,
                vec![assign("sum", v("sum").binary(Binary::Add, exp(v("col"))))],
            ),
            loop_(
                "col",
                input.cols,
                vec![store(
                    "dst",
                    Expr::GlobalId
                        .binary(Binary::Mul, u(input.cols))
                        .binary(Binary::Add, v("col")),
                    exp(v("col")).binary(Binary::Div, v("sum")),
                )],
            ),
        ],
        "ggml/src/ggml-cpu/ops.cpp: stable SOFT_MAX finite unmasked rows; masking/all-infinite rows pending",
    )
}

pub fn matmul(lhs: MatrixLayout, rhs: MatrixLayout) -> Result<Kernel> {
    if lhs.cols != rhs.rows {
        return Err(Error("matmul inner dimensions mismatch".into()));
    }
    let count = lhs
        .rows
        .checked_mul(rhs.cols)
        .ok_or_else(|| Error("matmul output overflow".into()))?;
    new(
        "matmul_f32",
        count,
        vec![
            buffer("lhs", Scalar::F32, Access::Read, lhs.storage_elements()?),
            buffer("rhs", Scalar::F32, Access::Read, rhs.storage_elements()?),
            buffer("dst", Scalar::F32, Access::ReadWrite, count),
        ],
        vec![
            declare(
                "row",
                Scalar::U32,
                Expr::GlobalId.binary(Binary::Div, u(rhs.cols)),
            ),
            declare(
                "col",
                Scalar::U32,
                Expr::GlobalId.binary(Binary::Rem, u(rhs.cols)),
            ),
            declare("sum", Scalar::F32, f(0.0)),
            loop_(
                "inner",
                lhs.cols,
                vec![assign(
                    "sum",
                    v("sum").binary(
                        Binary::Add,
                        Expr::load("lhs", lhs.at(v("row"), v("inner")))
                            .binary(Binary::Mul, Expr::load("rhs", rhs.at(v("inner"), v("col")))),
                    ),
                )],
            ),
            store("dst", Expr::GlobalId, v("sum")),
        ],
        "ggml/src/ggml-cpu/ops.cpp: scalar matrix product; row/column strides explicit",
    )
}

fn byte_load(byte: Expr) -> Expr {
    Expr::load("packed", byte.clone().binary(Binary::Div, u(4)))
        .binary(
            Binary::Shr,
            byte.binary(Binary::Rem, u(4)).binary(Binary::Mul, u(8)),
        )
        .binary(Binary::BitAnd, u(255))
}

/// GGML Q4_0 blocks are 18 bytes: little-endian F16 scale followed by 16
/// packed bytes. The low nibbles are elements 0..15; high are 16..31.
/// Offsets/row padding and input-vector strides are supported. Storage is U32
/// with zero padding to a whole word; no unsafe byte reinterpretation occurs.
pub fn q4_0_matvec(
    rows: u32,
    cols: u32,
    offset_bytes: u32,
    row_stride_bytes: u32,
    input: VectorLayout,
) -> Result<Kernel> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(32) || input.length != cols {
        return Err(Error(
            "Q4_0 requires nonzero rows, columns divisible by 32 and matching input".into(),
        ));
    }
    let row_bytes = (cols / 32)
        .checked_mul(18)
        .ok_or_else(|| Error("Q4_0 row size overflow".into()))?;
    if row_stride_bytes < row_bytes {
        return Err(Error("Q4_0 row stride is shorter than row payload".into()));
    }
    let bytes = (rows - 1)
        .checked_mul(row_stride_bytes)
        .and_then(|n| n.checked_add(offset_bytes))
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or_else(|| Error("Q4_0 address overflow".into()))?;
    let words = bytes
        .checked_add(3)
        .ok_or_else(|| Error("Q4_0 padding overflow".into()))?
        / 4;
    let base = u(offset_bytes)
        .binary(
            Binary::Add,
            Expr::GlobalId.binary(Binary::Mul, u(row_stride_bytes)),
        )
        .binary(Binary::Add, v("block").binary(Binary::Mul, u(18)));
    let scale_bits = byte_load(v("base")).binary(
        Binary::BitOr,
        byte_load(v("base").binary(Binary::Add, u(1))).binary(Binary::Shl, u(8)),
    );
    let packed = byte_load(
        v("base")
            .binary(Binary::Add, u(2))
            .binary(Binary::Add, v("lane")),
    );
    let lo = v("byte")
        .binary(Binary::BitAnd, u(15))
        .cast(Scalar::I32)
        .binary(Binary::Sub, Expr::Int(8))
        .cast(Scalar::F32);
    let hi = v("byte")
        .binary(Binary::Shr, u(4))
        .cast(Scalar::I32)
        .binary(Binary::Sub, Expr::Int(8))
        .cast(Scalar::F32);
    let element = v("block")
        .binary(Binary::Mul, u(32))
        .binary(Binary::Add, v("lane"));
    new(
        "q4_0_matvec",
        rows,
        vec![
            buffer("packed", Scalar::U32, Access::Read, words),
            buffer(
                "vector",
                Scalar::F32,
                Access::Read,
                input.storage_elements()?,
            ),
            buffer("dst", Scalar::F32, Access::ReadWrite, rows),
        ],
        vec![
            declare("sum", Scalar::F32, f(0.0)),
            loop_(
                "block",
                cols / 32,
                vec![
                    declare("base", Scalar::U32, base),
                    declare(
                        "scale",
                        Scalar::F32,
                        Expr::HalfToFloat(Box::new(scale_bits)),
                    ),
                    loop_(
                        "lane",
                        16,
                        vec![
                            declare("byte", Scalar::U32, packed),
                            assign(
                                "sum",
                                v("sum").binary(
                                    Binary::Add,
                                    v("scale").binary(Binary::Mul, lo).binary(
                                        Binary::Mul,
                                        Expr::load("vector", input.at(element.clone())),
                                    ),
                                ),
                            ),
                            assign(
                                "sum",
                                v("sum").binary(
                                    Binary::Add,
                                    v("scale").binary(Binary::Mul, hi).binary(
                                        Binary::Mul,
                                        Expr::load(
                                            "vector",
                                            input.at(element.binary(Binary::Add, u(16))),
                                        ),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            store("dst", Expr::GlobalId, v("sum")),
        ],
        "ggml/src/ggml-common.h:block_q4_0; ggml/src/ggml-quants.c:dequantize_row_q4_0",
    )
}

pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}

/// A reproducible code-generation corpus, including an unaligned quantized
/// matrix and noncontiguous input vectors. This is not the full backend matrix.
pub fn conformance_kernels() -> Result<Vec<Kernel>> {
    Ok(vec![
        pointwise(
            Pointwise::Add,
            VectorLayout {
                length: 37,
                offset: 1,
                stride: 2,
            },
            Some(VectorLayout {
                length: 37,
                offset: 2,
                stride: 3,
            }),
            VectorLayout {
                length: 37,
                offset: 4,
                stride: 2,
            },
            1.0,
        )?,
        pointwise(
            Pointwise::Mul,
            VectorLayout::contiguous(65),
            Some(VectorLayout::contiguous(65)),
            VectorLayout::contiguous(65),
            1.0,
        )?,
        pointwise(
            Pointwise::Silu,
            VectorLayout::contiguous(33),
            None,
            VectorLayout::contiguous(33),
            1.0,
        )?,
        pointwise(
            Pointwise::Scale,
            VectorLayout::contiguous(15),
            None,
            VectorLayout::contiguous(15),
            0.125,
        )?,
        row_sum(MatrixLayout {
            rows: 3,
            cols: 7,
            offset: 2,
            row_stride: 24,
            col_stride: 3,
        })?,
        rms_norm(MatrixLayout::contiguous(3, 13), 1e-5)?,
        softmax(MatrixLayout::contiguous(3, 17), 0.125)?,
        matmul(
            MatrixLayout {
                rows: 3,
                cols: 5,
                offset: 1,
                row_stride: 12,
                col_stride: 2,
            },
            MatrixLayout {
                rows: 5,
                cols: 7,
                offset: 3,
                row_stride: 1,
                col_stride: 6,
            },
        )?,
        q4_0_matvec(
            3,
            64,
            3,
            41,
            VectorLayout {
                length: 64,
                offset: 1,
                stride: 2,
            },
        )?,
    ])
}
