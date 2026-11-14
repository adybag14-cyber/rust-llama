//! Checked decoder kernels expressed through the same typed PTX instruction tree.
//! All data-dependent indices are guarded against the declared buffer capacities.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum DecoderKernel {
    Embedding {
        vocabulary: u32,
        width: u32,
    },
    MatVec {
        rows: u32,
        columns: u32,
        half: bool,
    },
    Q8MatVec {
        rows: u32,
        columns: u32,
    },
    ActivationQ8 {
        columns: u32,
    },
    QuantizedDot {
        rows: u32,
        columns: u32,
        q4: bool,
    },
    RmsScale {
        width: u32,
        epsilon: f32,
    },
    RmsApply {
        width: u32,
    },
    /// Independent RMS normalization of contiguous attention heads, with one
    /// shared learned weight vector and the source scalar accumulation order.
    HeadRms {
        width: u32,
        head_dimension: u32,
        epsilon: f32,
    },
    /// Copy a validated contiguous interval without a host round trip.
    Slice {
        source_width: u32,
        offset: u32,
        width: u32,
    },
    Rope {
        width: u32,
        head_dimension: u32,
        context: u32,
    },
    RopeNeox {
        width: u32,
        head_dimension: u32,
        context: u32,
    },
    CacheWrite {
        width: u32,
        context: u32,
    },
    AttentionScores {
        heads: u32,
        kv_heads: u32,
        dimension: u32,
        context: u32,
        scale: f32,
    },
    SoftmaxStats {
        heads: u32,
        context: u32,
    },
    SoftmaxApply {
        heads: u32,
        context: u32,
    },
    AttentionValues {
        heads: u32,
        kv_heads: u32,
        dimension: u32,
        context: u32,
    },
    SiluGate {
        width: u32,
    },
}
fn size(dimensions: &[u32], element: usize) -> Result<usize, KernelError> {
    let mut result = element;
    for &n in dimensions {
        if n == 0 || n > u32::MAX - 255 {
            return Err(KernelError(
                "decoder dimensions must be positive and bounded".into(),
            ));
        }
        result = result
            .checked_mul(n as usize)
            .filter(|&x| x <= isize::MAX as usize && x <= u32::MAX as usize - 255)
            .ok_or_else(|| KernelError("decoder extent overflow".into()))?;
    }
    Ok(result)
}
fn attention_shape(
    heads: u32,
    kv_heads: u32,
    dimension: u32,
    context: u32,
) -> Result<(), KernelError> {
    size(&[heads, dimension], 4)?;
    size(&[kv_heads, dimension, context], 4)?;
    size(&[heads, context], 4)?;
    if !heads.is_multiple_of(kv_heads) {
        return Err(KernelError(
            "attention heads do not divide into KV groups".into(),
        ));
    }
    Ok(())
}
impl DecoderKernel {
    pub(super) fn buffer_bytes(&self) -> Result<Vec<usize>, KernelError> {
        use DecoderKernel::*;
        let uniform = 8;
        match *self {
            Embedding { vocabulary, width } => Ok(vec![
                size(&[vocabulary, width], 4)?,
                uniform,
                size(&[width], 4)?,
            ]),
            MatVec {
                rows,
                columns,
                half,
            } => Ok(vec![
                size(&[rows, columns], if half { 2 } else { 4 })?,
                size(&[columns], 4)?,
                size(&[rows], 4)?,
            ]),
            Q8MatVec { rows, columns } => {
                if !columns.is_multiple_of(32) || columns == 0 {
                    return Err(KernelError("Q8_0 requires complete 32-value blocks".into()));
                }
                Ok(vec![
                    size(&[rows, columns / 32, 34], 1)?,
                    size(&[columns], 4)?,
                    size(&[rows], 4)?,
                ])
            }
            ActivationQ8 { columns } => {
                if columns == 0 || !columns.is_multiple_of(32) {
                    return Err(KernelError(
                        "Q8 activation requires complete 32-value blocks".into(),
                    ));
                }
                Ok(vec![size(&[columns], 4)?, size(&[columns / 32, 33], 4)?])
            }
            QuantizedDot { rows, columns, q4 } => {
                if columns == 0 || !columns.is_multiple_of(32) {
                    return Err(KernelError(
                        "quantized dot requires complete 32-value blocks".into(),
                    ));
                }
                Ok(vec![
                    size(&[rows, columns / 32, if q4 { 18 } else { 34 }], 1)?,
                    size(&[columns / 32, 33], 4)?,
                    size(&[rows], 4)?,
                ])
            }
            RmsScale { width, epsilon } => {
                if !epsilon.is_finite() || epsilon <= 0.0 {
                    return Err(KernelError(
                        "RMS epsilon must be finite and positive".into(),
                    ));
                }
                Ok(vec![size(&[width], 4)?, 4])
            }
            RmsApply { width } => Ok(vec![
                size(&[width], 4)?,
                size(&[width], 4)?,
                4,
                size(&[width], 4)?,
            ]),
            HeadRms {
                width,
                head_dimension,
                epsilon,
            } => {
                size(&[width], 4)?;
                size(&[head_dimension], 4)?;
                if !width.is_multiple_of(head_dimension) || !epsilon.is_finite() || epsilon <= 0.0 {
                    return Err(KernelError(
                        "head RMS requires complete heads and positive finite epsilon".into(),
                    ));
                }
                Ok(vec![
                    size(&[width], 4)?,
                    size(&[head_dimension], 4)?,
                    size(&[width], 4)?,
                ])
            }
            Slice {
                source_width,
                offset,
                width,
            } => {
                let source = size(&[source_width], 4)?;
                let output = size(&[width], 4)?;
                if offset
                    .checked_add(width)
                    .is_none_or(|end| end > source_width)
                {
                    return Err(KernelError("slice exceeds declared source".into()));
                }
                Ok(vec![source, output])
            }
            Rope {
                width,
                head_dimension,
                context,
            }
            | RopeNeox {
                width,
                head_dimension,
                context,
            } => {
                size(&[width], 4)?;
                size(&[head_dimension], 4)?;
                if !head_dimension.is_multiple_of(2) || !width.is_multiple_of(head_dimension) {
                    return Err(KernelError(
                        "RoPE requires complete even-dimensional heads".into(),
                    ));
                }
                Ok(vec![
                    size(&[width], 4)?,
                    size(&[context, head_dimension], 4)?,
                    uniform,
                    size(&[width], 4)?,
                ])
            }
            CacheWrite { width, context } => Ok(vec![
                size(&[width], 4)?,
                uniform,
                size(&[context, width], 4)?,
            ]),
            AttentionScores {
                heads,
                kv_heads,
                dimension,
                context,
                scale,
            } => {
                attention_shape(heads, kv_heads, dimension, context)?;
                if !scale.is_finite() || scale == 0.0 {
                    return Err(KernelError(
                        "attention scale must be finite and nonzero".into(),
                    ));
                }
                Ok(vec![
                    size(&[heads, dimension], 4)?,
                    size(&[context, kv_heads, dimension], 4)?,
                    uniform,
                    size(&[heads, context], 4)?,
                ])
            }
            SoftmaxStats { heads, context } => Ok(vec![
                size(&[heads, context], 4)?,
                uniform,
                size(&[heads, 2], 4)?,
            ]),
            SoftmaxApply { heads, context } => Ok(vec![
                size(&[heads, context], 4)?,
                size(&[heads, 2], 4)?,
                uniform,
                size(&[heads, context], 4)?,
            ]),
            AttentionValues {
                heads,
                kv_heads,
                dimension,
                context,
            } => {
                attention_shape(heads, kv_heads, dimension, context)?;
                Ok(vec![
                    size(&[heads, context], 4)?,
                    size(&[context, kv_heads, dimension], 4)?,
                    uniform,
                    size(&[heads, dimension], 4)?,
                ])
            }
            SiluGate { width } => Ok(vec![
                size(&[width], 4)?,
                size(&[width], 4)?,
                size(&[width], 4)?,
            ]),
        }
    }
    pub(super) fn outputs(&self) -> u32 {
        use DecoderKernel::*;
        match *self {
            Embedding { width, .. }
            | RmsApply { width }
            | CacheWrite { width, .. }
            | SiluGate { width } => width,
            MatVec { rows, .. } | Q8MatVec { rows, .. } | QuantizedDot { rows, .. } => rows,
            ActivationQ8 { columns } => columns / 32,
            RmsScale { .. } => 1,
            HeadRms {
                width,
                head_dimension,
                ..
            } => width.checked_div(head_dimension).unwrap_or(0),
            Slice { width, .. } => width,
            Rope { width, .. } | RopeNeox { width, .. } => width / 2,
            AttentionScores { heads, context, .. } | SoftmaxApply { heads, context } => {
                heads.saturating_mul(context)
            }
            SoftmaxStats { heads, .. } => heads,
            AttentionValues {
                heads, dimension, ..
            } => heads.saturating_mul(dimension),
        }
    }
    pub(super) fn output_buffer(&self) -> usize {
        use DecoderKernel::*;
        match self {
            RmsScale { .. } | ActivationQ8 { .. } | Slice { .. } => 1,
            RmsApply { .. }
            | Rope { .. }
            | RopeNeox { .. }
            | AttentionScores { .. }
            | SoftmaxApply { .. }
            | AttentionValues { .. } => 3,
            _ => 2,
        }
    }
}
fn load(p: &mut Program, base: Reg, index: Reg, stride: u32, kind: LoadType) -> Reg {
    let address = p.address(base, index, stride);
    p.load(address, kind)
}
fn store(p: &mut Program, base: Reg, index: Reg, value: Reg) {
    let address = p.address(base, index, 4);
    p.push(Instruction::Store { value, address });
}
// A numeric failure must survive later zero weights and quantization. Infinity
// is converted to NaN so it cannot be mistaken for an intentional attention mask.
fn checked_store(p: &mut Program, base: Reg, index: Reg, value: Reg) {
    let value = p.finite_or_nan(value);
    store(p, base, index, value);
}
fn uniform(p: &mut Program, base: Reg, index: u32) -> Reg {
    let i = p.value(index.into());
    load(p, base, i, 4, LoadType::U32)
}
fn position(p: &mut Program, base: Reg, context: u32, done: usize) -> Reg {
    let pos = uniform(p, base, 1);
    p.guard_ge(pos, context, done);
    pos
}
fn begin_loop(p: &mut Program) -> (Reg, usize, usize) {
    let cursor = p.value(0u32.into());
    let start = p.label();
    let end = p.label();
    p.push(Instruction::Label(start));
    (cursor, start, end)
}
fn end_loop(p: &mut Program, cursor: Reg, start: usize, end: usize) {
    p.assign_binary(cursor, Binary::Add, cursor, 1u32);
    p.push(Instruction::Branch {
        label: start,
        pred: None,
    });
    p.push(Instruction::Label(end));
}
fn add64(p: &mut Program, sum: Reg, value: Reg) {
    let value = p.convert(Conversion::F32ToF64, value);
    p.assign_binary(sum, Binary::Add, sum, value);
}
fn scalar_exp(p: &mut Program, value: Reg) -> Reg {
    // PTX ex2.approx plus F32 range reduction changes activation quantization
    // bins in real Q8 decoders. Reduce and evaluate in F64, then round once.
    // The reduced interval is [-ln(2)/2, ln(2)/2]; degree 12 has <2e-16
    // truncation error there. Explicit .rn operations prevent contraction.
    let result = p.value(Operand::F32(f32::NAN));
    let underflow = p.label();
    let overflow = p.label();
    let end = p.label();
    p.guard_ge(value, Operand::F32(89.0), overflow);
    p.guard_ge(Operand::F32(-104.0), value, underflow);
    p.guard_nonfinite(value, end);
    let x = p.convert(Conversion::F32ToF64, value);
    let scaled = p.binary(Binary::Mul, x, Operand::F64(std::f64::consts::LOG2_E));
    let n = p.convert(Conversion::F64ToI32Nearest, scaled);
    let nf = p.convert(Conversion::I32ToF64, n);
    let reduced = p.binary(Binary::Mul, nf, Operand::F64(std::f64::consts::LN_2));
    let reduced = p.binary(Binary::Sub, x, reduced);
    let biased = p.binary(Binary::Add, n, Operand::I32(1023));
    let biased = p.convert(Conversion::I32ToU64, biased);
    let bits = p.binary(Binary::Shl, biased, 52u32);
    let scale = p.convert(Conversion::U64BitsToF64, bits);
    let mut polynomial = p.value(Operand::F64(1.0 / 479_001_600.0));
    for factorial in [
        39_916_800.0,
        3_628_800.0,
        362_880.0,
        40_320.0,
        5_040.0,
        720.0,
        120.0,
        24.0,
        6.0,
        2.0,
        1.0,
        1.0,
    ] {
        polynomial = p.binary(Binary::Mul, polynomial, reduced);
        polynomial = p.binary(Binary::Add, polynomial, Operand::F64(1.0 / factorial));
    }
    let product = p.binary(Binary::Mul, polynomial, scale);
    let rounded = p.convert(Conversion::F64ToF32, product);
    p.push(Instruction::Move {
        dst: result,
        value: rounded.into(),
    });
    p.push(Instruction::Branch {
        label: end,
        pred: None,
    });
    p.push(Instruction::Label(underflow));
    p.push(Instruction::Move {
        dst: result,
        value: Operand::F32(0.0),
    });
    p.push(Instruction::Branch {
        label: end,
        pred: None,
    });
    p.push(Instruction::Label(overflow));
    p.push(Instruction::Move {
        dst: result,
        value: Operand::F32(f32::INFINITY),
    });
    p.push(Instruction::Label(end));
    result
}
fn sse2_policy() -> bool {
    llama_quant::cpu_math::native_math_policy() == llama_quant::cpu_math::MathPolicy::Sse2
}
fn sse2_exp(p: &mut Program, x: Reg) -> Reg {
    // Frozen ggml vec.h no-FMA SSE2 polynomial. A lane outside the normal
    // exponent range takes the same split-scale branch as its source vector.
    // Normal lanes have identical arithmetic in either vector branch.
    let c = |bits| Operand::F32(f32::from_bits(bits));
    let result = p.value(Operand::F32(f32::NAN));
    let zero = p.label();
    let infinity = p.label();
    let extreme = p.label();
    let end = p.label();
    p.guard_ge(x, Operand::F32(134.0), infinity);
    p.guard_ge(Operand::F32(-134.0), x, zero);
    p.guard_nonfinite(x, end);
    let z = p.binary(Binary::Mul, x, c(0x3fb8_aa3b));
    let z = p.binary(Binary::Add, z, c(0x4b40_0000));
    let n = p.binary(Binary::Sub, z, c(0x4b40_0000));
    p.guard_ge(n, Operand::F32(193.0), infinity);
    p.guard_ge(Operand::F32(-193.0), n, zero);
    let first = p.binary(Binary::Mul, n, c(0x3f31_7200));
    let reduced = p.binary(Binary::Sub, x, first);
    let second = p.binary(Binary::Mul, n, c(0x35bf_be8e));
    let reduced = p.binary(Binary::Sub, reduced, second);
    let exponent = p.convert(Conversion::F32BitsToU32, z);
    let exponent = p.binary(Binary::Shl, exponent, 23u32);
    let scale_bits = p.binary(Binary::Add, exponent, 0x3f80_0000u32);
    let scale = p.convert(Conversion::U32BitsToF32, scale_bits);
    let squared = p.binary(Binary::Mul, reduced, reduced);
    let first = p.binary(Binary::Mul, c(0x3c07_2010), reduced);
    let first = p.binary(Binary::Add, first, c(0x3d2b_9f17));
    let second = p.binary(Binary::Mul, c(0x3e2a_af33), reduced);
    let second = p.binary(Binary::Add, second, c(0x3eff_fedb));
    let polynomial = p.binary(Binary::Mul, first, squared);
    let polynomial = p.binary(Binary::Add, polynomial, second);
    let polynomial = p.binary(Binary::Mul, polynomial, squared);
    let last = p.binary(Binary::Mul, c(0x3f7f_fff6), reduced);
    let polynomial = p.binary(Binary::Add, polynomial, last);
    p.guard_ge(n, Operand::F32(127.0), extreme);
    p.guard_ge(Operand::F32(-127.0), n, extreme);
    let product = p.binary(Binary::Mul, polynomial, scale);
    p.assign_binary(result, Binary::Add, product, scale);
    p.push(Instruction::Branch {
        label: end,
        pred: None,
    });
    p.push(Instruction::Label(extreme));
    let shift = p.value(0u32.into());
    let positive = p.label();
    p.guard_ge(n, Operand::F32(0.0), positive);
    p.push(Instruction::Move {
        dst: shift,
        value: Operand::U32(0x8200_0000),
    });
    p.push(Instruction::Label(positive));
    let high = p.binary(Binary::Add, shift, 0x7f00_0000u32);
    let high = p.convert(Conversion::U32BitsToF32, high);
    let low = p.binary(Binary::Sub, exponent, shift);
    let low = p.convert(Conversion::U32BitsToF32, low);
    let product = p.binary(Binary::Mul, low, polynomial);
    let product = p.binary(Binary::Add, product, low);
    p.assign_binary(result, Binary::Mul, product, high);
    p.push(Instruction::Branch {
        label: end,
        pred: None,
    });
    p.push(Instruction::Label(zero));
    p.push(Instruction::Move {
        dst: result,
        value: Operand::F32(0.0),
    });
    p.push(Instruction::Branch {
        label: end,
        pred: None,
    });
    p.push(Instruction::Label(infinity));
    p.push(Instruction::Move {
        dst: result,
        value: Operand::F32(f32::INFINITY),
    });
    p.push(Instruction::Label(end));
    result
}
fn exp(p: &mut Program, value: Reg) -> Reg {
    if sse2_policy() {
        sse2_exp(p, value)
    } else {
        scalar_exp(p, value)
    }
}
fn silu_exp(p: &mut Program, value: Reg, index: Reg, width: u32) -> Reg {
    if !sse2_policy() {
        return scalar_exp(p, value);
    }
    if width.is_multiple_of(4) {
        return sse2_exp(p, value);
    }
    let result = p.value(Operand::F32(f32::NAN));
    let tail = p.label();
    let end = p.label();
    p.guard_ge(index, width / 4 * 4, tail);
    let vector = sse2_exp(p, value);
    p.push(Instruction::Move {
        dst: result,
        value: vector.into(),
    });
    p.push(Instruction::Branch {
        label: end,
        pred: None,
    });
    p.push(Instruction::Label(tail));
    let scalar = scalar_exp(p, value);
    p.push(Instruction::Move {
        dst: result,
        value: scalar.into(),
    });
    p.push(Instruction::Label(end));
    result
}
fn dot(
    p: &mut Program,
    a: Reg,
    b: Reg,
    length: u32,
    a_stride: u32,
    b_stride: u32,
    half: bool,
) -> Reg {
    let sum = p.value(Operand::F64(0.0));
    let (cursor, start, end) = begin_loop(p);
    p.guard_ge(cursor, length, end);
    let mut av = load(
        p,
        a,
        cursor,
        a_stride,
        if half { LoadType::F16 } else { LoadType::F32 },
    );
    if half {
        av = p.convert(Conversion::HalfToF32, av);
    }
    let bv = load(p, b, cursor, b_stride, LoadType::F32);
    let product = p.binary(Binary::Mul, av, bv);
    add64(p, sum, product);
    end_loop(p, cursor, start, end);
    p.convert(Conversion::F64ToF32, sum)
}
pub(super) fn emit(kind: &DecoderKernel, p: &mut Program, b: &[Reg], i: Reg, done: usize) {
    use DecoderKernel::*;
    match *kind {
        Embedding { vocabulary, width } => {
            let token = uniform(p, b[1], 0);
            p.guard_ge(token, vocabulary, done);
            let row = p.binary(Binary::Mul, token, width);
            let col = p.binary(Binary::Add, row, i);
            let value = load(p, b[0], col, 4, LoadType::F32);
            checked_store(p, b[2], i, value);
        }
        MatVec {
            rows: _,
            columns,
            half,
        } => {
            let row = p.address(b[0], i, columns * if half { 2 } else { 4 });
            let value = dot(p, row, b[1], columns, if half { 2 } else { 4 }, 4, half);
            checked_store(p, b[2], i, value);
        }
        Q8MatVec { columns, .. } => {
            let row = p.address(b[0], i, columns / 32 * 34);
            let sum = p.value(Operand::F64(0.0));
            let (block, start, end) = begin_loop(p);
            p.guard_ge(block, columns / 32, end);
            let block_address = p.address(row, block, 34);
            let half = p.load(block_address, LoadType::F16);
            let scale = p.convert(Conversion::HalfToF32, half);
            let first = p.binary(Binary::Mul, block, 32u32);
            let (cursor, inner_start, inner_end) = begin_loop(p);
            p.guard_ge(cursor, 32u32, inner_end);
            let byte = p.binary(Binary::Add, cursor, 2u32);
            let packed = load(p, block_address, byte, 1, LoadType::U8);
            let signed = p.convert(Conversion::U32ToI32, packed);
            let negative = p.label();
            let converted = p.label();
            p.guard_ge(packed, 128u32, negative);
            p.push(Instruction::Branch {
                label: converted,
                pred: None,
            });
            p.push(Instruction::Label(negative));
            p.assign_binary(signed, Binary::Sub, signed, Operand::I32(256));
            p.push(Instruction::Label(converted));
            let quant = p.convert(Conversion::I32ToF32, signed);
            let weight = p.binary(Binary::Mul, quant, scale);
            let column = p.binary(Binary::Add, first, cursor);
            let x = load(p, b[1], column, 4, LoadType::F32);
            let value = p.binary(Binary::Mul, weight, x);
            add64(p, sum, value);
            end_loop(p, cursor, inner_start, inner_end);
            end_loop(p, block, start, end);
            let value = p.convert(Conversion::F64ToF32, sum);
            checked_store(p, b[2], i, value);
        }
        ActivationQ8 { .. } => {
            let invalid = p.label();
            let row = p.address(b[0], i, 32 * 4);
            let output = p.address(b[1], i, 33 * 4);
            let maximum = p.value(Operand::F32(0.0));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, 32u32, end);
            let x = load(p, row, cursor, 4, LoadType::F32);
            p.guard_nonfinite(x, invalid);
            let negative = p.binary(Binary::Mul, x, Operand::F32(-1.0));
            let absolute = p.binary(Binary::Max, x, negative);
            p.assign_binary(maximum, Binary::Max, maximum, absolute);
            end_loop(p, cursor, start, end);
            let scale = p.binary(Binary::Div, maximum, Operand::F32(127.0));
            let half = p.convert(Conversion::F32ToHalf, scale);
            let rounded_scale = p.convert(Conversion::HalfToF32, half);
            p.guard_nonfinite(rounded_scale, invalid);
            p.push(Instruction::Store {
                value: rounded_scale,
                address: output,
            });
            let inverse = p.value(Operand::F32(0.0));
            let zero = p.label();
            p.guard_ge(Operand::F32(0.0), scale, zero);
            p.assign_binary(inverse, Binary::Div, Operand::F32(1.0), scale);
            p.push(Instruction::Label(zero));
            // The CPU codec rejects nonzero scales whose reciprocal overflows.
            p.guard_nonfinite(inverse, invalid);
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, 32u32, end);
            let x = load(p, row, cursor, 4, LoadType::F32);
            let x = p.binary(Binary::Mul, x, inverse);
            let negative = p.binary(Binary::Mul, x, Operand::F32(-1.0));
            let absolute = p.binary(Binary::Max, x, negative);
            let signed = p.convert(Conversion::F32ToI32Trunc, absolute);
            let integral = p.convert(Conversion::I32ToF32, signed);
            let fraction = p.binary(Binary::Sub, absolute, integral);
            // Adding 0.5 before truncation rounds values immediately below a
            // half-integer up. Compare the exact fractional part instead.
            let round_up = p.label();
            let rounded = p.label();
            p.guard_ge(fraction, Operand::F32(0.5), round_up);
            p.push(Instruction::Branch {
                label: rounded,
                pred: None,
            });
            p.push(Instruction::Label(round_up));
            p.assign_binary(signed, Binary::Add, signed, Operand::I32(1));
            p.push(Instruction::Label(rounded));
            let positive = p.label();
            p.guard_ge(x, Operand::F32(0.0), positive);
            p.assign_binary(signed, Binary::Sub, Operand::I32(0), signed);
            p.push(Instruction::Label(positive));
            let quant = p.convert(Conversion::I32ToF32, signed);
            let offset = p.binary(Binary::Add, cursor, 1u32);
            store(p, output, offset, quant);
            end_loop(p, cursor, start, end);
            p.push(Instruction::Branch {
                label: done,
                pred: None,
            });
            p.push(Instruction::Label(invalid));
            // Fully overwrite the block. Even a zero packed weight then sees
            // a NaN scale, so an invalid activation cannot become finite again.
            let poison = p.value(Operand::F32(f32::NAN));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, 33u32, end);
            store(p, output, cursor, poison);
            end_loop(p, cursor, start, end);
        }
        QuantizedDot { columns, q4, .. } => {
            let block_bytes = if q4 { 18 } else { 34 };
            let row = p.address(b[0], i, columns / 32 * block_bytes);
            let sum = p.value(Operand::F32(0.0));
            let (block, start, end) = begin_loop(p);
            p.guard_ge(block, columns / 32, end);
            let weights = p.address(row, block, block_bytes);
            let activations = p.address(b[1], block, 33 * 4);
            let half = p.load(weights, LoadType::F16);
            let dx = p.convert(Conversion::HalfToF32, half);
            let dy = p.load(activations, LoadType::F32);
            let integer_sum = p.value(Operand::F32(0.0));
            let (cursor, inner_start, inner_end) = begin_loop(p);
            p.guard_ge(cursor, if q4 { 16u32 } else { 32u32 }, inner_end);
            let offset = p.binary(Binary::Add, cursor, 2u32);
            let packed = load(p, weights, offset, 1, LoadType::U8);
            if q4 {
                let low = p.binary(Binary::And, packed, 15u32);
                let high = p.binary(Binary::Shr, packed, 4u32);
                for (value, delta) in [(low, 1u32), (high, 17u32)] {
                    let q = p.convert(Conversion::U32ToI32, value);
                    let q = p.binary(Binary::Sub, q, Operand::I32(8));
                    let q = p.convert(Conversion::I32ToF32, q);
                    let column = p.binary(Binary::Add, cursor, delta);
                    let activation = load(p, activations, column, 4, LoadType::F32);
                    let product = p.binary(Binary::Mul, q, activation);
                    p.assign_binary(integer_sum, Binary::Add, integer_sum, product);
                }
            } else {
                let signed = p.convert(Conversion::U32ToI32, packed);
                let negative = p.label();
                let converted = p.label();
                p.guard_ge(packed, 128u32, negative);
                p.push(Instruction::Branch {
                    label: converted,
                    pred: None,
                });
                p.push(Instruction::Label(negative));
                p.assign_binary(signed, Binary::Sub, signed, Operand::I32(256));
                p.push(Instruction::Label(converted));
                let q = p.convert(Conversion::I32ToF32, signed);
                let column = p.binary(Binary::Add, cursor, 1u32);
                let activation = load(p, activations, column, 4, LoadType::F32);
                let product = p.binary(Binary::Mul, q, activation);
                p.assign_binary(integer_sum, Binary::Add, integer_sum, product);
            }
            end_loop(p, cursor, inner_start, inner_end);
            let contribution = if q4 {
                let value = p.binary(Binary::Mul, integer_sum, dx);
                p.binary(Binary::Mul, value, dy)
            } else {
                let scale = p.binary(Binary::Mul, dx, dy);
                p.binary(Binary::Mul, integer_sum, scale)
            };
            p.assign_binary(sum, Binary::Add, sum, contribution);
            end_loop(p, block, start, end);
            checked_store(p, b[2], i, sum);
        }
        RmsScale { width, epsilon } => {
            let sum = p.value(Operand::F64(0.0));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, width, end);
            let value = load(p, b[0], cursor, 4, LoadType::F32);
            let square = p.binary(Binary::Mul, value, value);
            add64(p, sum, square);
            end_loop(p, cursor, start, end);
            let mean = p.binary(Binary::Div, sum, Operand::F64(f64::from(width)));
            let mean = p.convert(Conversion::F64ToF32, mean);
            let value = p.binary(Binary::Add, mean, Operand::F32(epsilon));
            let root = p.unary(Unary::Sqrt, value);
            let value = p.binary(Binary::Div, Operand::F32(1.0), root);
            let invalid = p.label();
            let valid = p.label();
            p.guard_nonfinite(value, invalid);
            p.guard_ge(Operand::F32(0.0), value, invalid);
            p.push(Instruction::Branch {
                label: valid,
                pred: None,
            });
            p.push(Instruction::Label(invalid));
            p.push(Instruction::Move {
                dst: value,
                value: Operand::F32(f32::NAN),
            });
            p.push(Instruction::Label(valid));
            store(p, b[1], i, value);
        }
        RmsApply { .. } => {
            let input = load(p, b[0], i, 4, LoadType::F32);
            let weight = load(p, b[1], i, 4, LoadType::F32);
            let scale = p.load(b[2], LoadType::F32);
            let value = p.binary(Binary::Mul, input, scale);
            let value = p.binary(Binary::Mul, value, weight);
            checked_store(p, b[3], i, value);
        }
        HeadRms {
            head_dimension,
            epsilon,
            ..
        } => {
            let row = p.binary(Binary::Mul, i, head_dimension);
            let sum = p.value(Operand::F64(0.0));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, head_dimension, end);
            let offset = p.binary(Binary::Add, row, cursor);
            let value = load(p, b[0], offset, 4, LoadType::F32);
            let square = p.binary(Binary::Mul, value, value);
            add64(p, sum, square);
            end_loop(p, cursor, start, end);
            let mean = p.binary(Binary::Div, sum, Operand::F64(f64::from(head_dimension)));
            let mean = p.convert(Conversion::F64ToF32, mean);
            let mean = p.binary(Binary::Add, mean, Operand::F32(epsilon));
            let root = p.unary(Unary::Sqrt, mean);
            let scale = p.binary(Binary::Div, Operand::F32(1.0), root);
            // A zero scale from overflow is a failure, even when subsequent
            // multiplication by zero weights could otherwise hide it.
            let invalid = p.label();
            let valid = p.label();
            p.guard_nonfinite(scale, invalid);
            p.guard_ge(Operand::F32(0.0), scale, invalid);
            p.push(Instruction::Branch {
                label: valid,
                pred: None,
            });
            p.push(Instruction::Label(invalid));
            p.push(Instruction::Move {
                dst: scale,
                value: Operand::F32(f32::NAN),
            });
            p.push(Instruction::Label(valid));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, head_dimension, end);
            let offset = p.binary(Binary::Add, row, cursor);
            let value = load(p, b[0], offset, 4, LoadType::F32);
            let weight = load(p, b[1], cursor, 4, LoadType::F32);
            let value = p.binary(Binary::Mul, value, scale);
            let value = p.binary(Binary::Mul, value, weight);
            checked_store(p, b[2], offset, value);
            end_loop(p, cursor, start, end);
        }
        Slice { offset, .. } => {
            let source = p.binary(Binary::Add, i, offset);
            let value = load(p, b[0], source, 4, LoadType::F32);
            checked_store(p, b[1], i, value);
        }
        Rope {
            head_dimension,
            context,
            ..
        }
        | RopeNeox {
            head_dimension,
            context,
            ..
        } => {
            let pos = position(p, b[2], context, done);
            let column = p.binary(Binary::Mul, i, 2u32);
            let offset = p.binary(Binary::Rem, column, head_dimension);
            let phase = p.binary(Binary::Mul, pos, head_dimension);
            let phase = p.binary(Binary::Add, phase, offset);
            let phase1 = p.binary(Binary::Add, phase, 1u32);
            let c = load(p, b[1], phase, 4, LoadType::F32);
            let s = load(p, b[1], phase1, 4, LoadType::F32);
            let (column, next) = if matches!(kind, RopeNeox { .. }) {
                let head = p.binary(Binary::Div, i, head_dimension / 2);
                let head = p.binary(Binary::Mul, head, head_dimension);
                let pair = p.binary(Binary::Rem, i, head_dimension / 2);
                let first = p.binary(Binary::Add, head, pair);
                let second = p.binary(Binary::Add, first, head_dimension / 2);
                (first, second)
            } else {
                let next = p.binary(Binary::Add, column, 1u32);
                (column, next)
            };
            let x = load(p, b[0], column, 4, LoadType::F32);
            let y = load(p, b[0], next, 4, LoadType::F32);
            let xc = p.binary(Binary::Mul, x, c);
            let ys = p.binary(Binary::Mul, y, s);
            let xs = p.binary(Binary::Mul, x, s);
            let yc = p.binary(Binary::Mul, y, c);
            let a = p.binary(Binary::Sub, xc, ys);
            let z = p.binary(Binary::Add, xs, yc);
            checked_store(p, b[3], column, a);
            checked_store(p, b[3], next, z);
        }
        CacheWrite { width, context } => {
            let pos = position(p, b[1], context, done);
            let row = p.binary(Binary::Mul, pos, width);
            let offset = p.binary(Binary::Add, row, i);
            let value = load(p, b[0], i, 4, LoadType::F32);
            checked_store(p, b[2], offset, value);
        }
        AttentionScores {
            heads,
            kv_heads,
            dimension,
            context,
            scale,
        } => {
            let pos = position(p, b[2], context, done);
            let count = p.binary(Binary::Add, pos, 1u32);
            let head = p.binary(Binary::Div, i, context);
            let slot = p.binary(Binary::Rem, i, context);
            let masked = p.label();
            p.guard_ge(slot, count, masked);
            let q = p.address(b[0], head, dimension * 4);
            let kv_head = p.binary(Binary::Div, head, heads / kv_heads);
            let row = p.binary(Binary::Mul, slot, kv_heads);
            let row = p.binary(Binary::Add, row, kv_head);
            let k = p.address(b[1], row, dimension * 4);
            let value = dot(p, q, k, dimension, 4, 4, false);
            let value = p.binary(Binary::Mul, value, Operand::F32(scale));
            checked_store(p, b[3], i, value);
            p.push(Instruction::Branch {
                label: done,
                pred: None,
            });
            p.push(Instruction::Label(masked));
            let value = p.value(Operand::F32(f32::NEG_INFINITY));
            store(p, b[3], i, value);
        }
        SoftmaxStats { context, .. } => {
            let invalid = p.label();
            let pos = position(p, b[1], context, done);
            let count = p.binary(Binary::Add, pos, 1u32);
            let row = p.address(b[0], i, context * 4);
            let maximum = p.value(Operand::F32(f32::NEG_INFINITY));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, count, end);
            let value = load(p, row, cursor, 4, LoadType::F32);
            // Only the visible prefix is visited: -Inf outside it is a mask,
            // while any nonfinite visible score represents a numeric failure.
            p.guard_nonfinite(value, invalid);
            p.assign_binary(maximum, Binary::Max, maximum, value);
            end_loop(p, cursor, start, end);
            let sum = p.value(Operand::F64(0.0));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, count, end);
            if sse2_policy() {
                // Original non-flash rows are padded to 256 columns. Sum
                // groups of four in F32, filling masked final lanes with zero,
                // then add the group's value to the F64 row accumulator.
                let mut lanes = Vec::with_capacity(4);
                for lane in 0..4u32 {
                    let value = p.value(Operand::F32(0.0));
                    let masked = p.label();
                    let slot = p.binary(Binary::Add, cursor, lane);
                    p.guard_ge(slot, count, masked);
                    let score = load(p, row, slot, 4, LoadType::F32);
                    let shifted = p.binary(Binary::Sub, score, maximum);
                    let exponential = sse2_exp(p, shifted);
                    p.push(Instruction::Move {
                        dst: value,
                        value: exponential.into(),
                    });
                    p.push(Instruction::Label(masked));
                    lanes.push(value);
                }
                let left = p.binary(Binary::Add, lanes[0], lanes[1]);
                let right = p.binary(Binary::Add, lanes[2], lanes[3]);
                let group = p.binary(Binary::Add, left, right);
                add64(p, sum, group);
                p.assign_binary(cursor, Binary::Add, cursor, 3u32);
            } else {
                let value = load(p, row, cursor, 4, LoadType::F32);
                let shifted = p.binary(Binary::Sub, value, maximum);
                let value = scalar_exp(p, shifted);
                add64(p, sum, value);
            }
            end_loop(p, cursor, start, end);
            // Preserve the original row's F64 reciprocal before rounding to F32.
            let inverse = p.binary(Binary::Div, Operand::F64(1.0), sum);
            let inverse = p.convert(Conversion::F64ToF32, inverse);
            let j = p.binary(Binary::Mul, i, 2u32);
            let k = p.binary(Binary::Add, j, 1u32);
            store(p, b[2], j, maximum);
            store(p, b[2], k, inverse);
            p.push(Instruction::Branch {
                label: done,
                pred: None,
            });
            p.push(Instruction::Label(invalid));
            let j = p.binary(Binary::Mul, i, 2u32);
            let k = p.binary(Binary::Add, j, 1u32);
            let poison = p.value(Operand::F32(f32::NAN));
            store(p, b[2], j, poison);
            store(p, b[2], k, poison);
        }
        SoftmaxApply { context, .. } => {
            let pos = position(p, b[2], context, done);
            let count = p.binary(Binary::Add, pos, 1u32);
            let head = p.binary(Binary::Div, i, context);
            let slot = p.binary(Binary::Rem, i, context);
            let masked = p.label();
            p.guard_ge(slot, count, masked);
            let j = p.binary(Binary::Mul, head, 2u32);
            let k = p.binary(Binary::Add, j, 1u32);
            let maximum = load(p, b[1], j, 4, LoadType::F32);
            let inverse = load(p, b[1], k, 4, LoadType::F32);
            let value = load(p, b[0], i, 4, LoadType::F32);
            let shifted = p.binary(Binary::Sub, value, maximum);
            let value = exp(p, shifted);
            let value = p.binary(Binary::Mul, value, inverse);
            checked_store(p, b[3], i, value);
            p.push(Instruction::Branch {
                label: done,
                pred: None,
            });
            p.push(Instruction::Label(masked));
            let zero = p.value(Operand::F32(0.0));
            store(p, b[3], i, zero);
        }
        AttentionValues {
            heads,
            kv_heads,
            dimension,
            context,
        } => {
            let pos = position(p, b[2], context, done);
            let count = p.binary(Binary::Add, pos, 1u32);
            let head = p.binary(Binary::Div, i, dimension);
            let col = p.binary(Binary::Rem, i, dimension);
            let kv_head = p.binary(Binary::Div, head, heads / kv_heads);
            let kv_col = p.binary(Binary::Mul, kv_head, dimension);
            let kv_col = p.binary(Binary::Add, kv_col, col);
            let v = p.address(b[1], kv_col, 4);
            let scores = p.address(b[0], head, context * 4);
            let sum = p.value(Operand::F64(0.0));
            let (cursor, start, end) = begin_loop(p);
            p.guard_ge(cursor, count, end);
            let a = load(p, v, cursor, kv_heads * dimension * 4, LoadType::F32);
            let w = load(p, scores, cursor, 4, LoadType::F32);
            let product = p.binary(Binary::Mul, a, w);
            add64(p, sum, product);
            end_loop(p, cursor, start, end);
            let value = p.convert(Conversion::F64ToF32, sum);
            checked_store(p, b[3], i, value);
        }
        SiluGate { width } => {
            let gate = load(p, b[0], i, 4, LoadType::F32);
            let up = load(p, b[1], i, 4, LoadType::F32);
            let negative = p.binary(Binary::Mul, gate, Operand::F32(-1.0));
            let value = silu_exp(p, negative, i, width);
            let denominator = p.binary(Binary::Add, value, Operand::F32(1.0));
            let silu = p.binary(Binary::Div, gate, denominator);
            let value = p.binary(Binary::Mul, silu, up);
            checked_store(p, b[2], i, value);
        }
    }
}
