use crate::*;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    OpenCl,
    Wgsl,
    Glsl,
    Msl,
}
impl Target {
    pub const ALL: [Self; 4] = [Self::OpenCl, Self::Wgsl, Self::Glsl, Self::Msl];
    pub fn extension(self) -> &'static str {
        match self {
            Self::OpenCl => "cl",
            Self::Wgsl => "wgsl",
            Self::Glsl => "comp",
            Self::Msl => "metal",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenCl => "opencl-c-1.2",
            Self::Wgsl => "wgsl",
            Self::Glsl => "vulkan-glsl-450",
            Self::Msl => "metal-shading-language",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub target: Target,
    pub entry_point: String,
    pub source: String,
    pub sha256: String,
    pub ir_version: u32,
}

fn scalar(ty: Scalar, target: Target) -> &'static str {
    match (ty, target) {
        (Scalar::F32, Target::Wgsl) => "f32",
        (Scalar::U32, Target::Wgsl) => "u32",
        (Scalar::I32, Target::Wgsl) => "i32",
        (Scalar::F32, _) => "float",
        (Scalar::U32, _) => "uint",
        (Scalar::I32, _) => "int",
        (Scalar::Bool, _) => "bool",
    }
}

fn expr(e: &Expr, target: Target) -> String {
    let render = |value: &Expr| expr(value, target);
    match e {
        Expr::Float(value) => {
            // Exact decimal round-trip of the authored f32, explicitly single precision.
            let text = format!("{value:?}");
            format!("{text}f")
        }
        Expr::Uint(value) => format!("{value}u"),
        Expr::Int(value) => format!("{value}"),
        Expr::Bool(value) => value.to_string(),
        Expr::Var(name) => format!("v_{name}"),
        Expr::GlobalId => "ir_gid".into(),
        Expr::Load { buffer, index } => format!("buf_{buffer}[{}]", render(index)),
        Expr::Binary { op, left, right } => {
            let symbol = match op {
                Binary::Add => "+",
                Binary::Sub => "-",
                Binary::Mul => "*",
                Binary::Div => "/",
                Binary::Rem => "%",
                Binary::BitAnd => "&",
                Binary::BitOr => "|",
                Binary::Xor => "^",
                Binary::Shl => "<<",
                Binary::Shr => ">>",
                Binary::Eq => "==",
                Binary::Ne => "!=",
                Binary::Lt => "<",
                Binary::Le => "<=",
                Binary::Gt => ">",
                Binary::Ge => ">=",
                Binary::And => "&&",
                Binary::Or => "||",
            };
            format!("({} {symbol} {})", render(left), render(right))
        }
        Expr::Unary { op, value } => {
            let v = render(value);
            match op {
                Unary::Neg => format!("(-{v})"),
                Unary::Not => format!("(!{v})"),
                Unary::Exp => format!("exp({v})"),
                Unary::Sqrt => format!("sqrt({v})"),
                Unary::Rsqrt => format!(
                    "{}({v})",
                    match target {
                        Target::Glsl => "inversesqrt",
                        Target::Wgsl => "inverseSqrt",
                        _ => "rsqrt",
                    }
                ),
                Unary::Tanh => format!("tanh({v})"),
                Unary::Abs => format!(
                    "{}({v})",
                    if target == Target::OpenCl {
                        "fabs"
                    } else {
                        "abs"
                    }
                ),
            }
        }
        Expr::Cast { ty, value } => {
            if target == Target::OpenCl {
                format!("(({})({}))", scalar(*ty, target), render(value))
            } else {
                format!("{}({})", scalar(*ty, target), render(value))
            }
        }
        Expr::Bitcast { ty, value } => {
            let v = render(value);
            match target {
                Target::Wgsl => format!("bitcast<{}>({v})", scalar(*ty, target)),
                Target::Msl => format!("as_type<{}>({v})", scalar(*ty, target)),
                Target::OpenCl => format!("as_{}({v})", scalar(*ty, target)),
                // All built-in kernels bitcast U32 to F32. Generic casts with
                // other source types are resolved by helper overloads below.
                Target::Glsl => format!("ir_bits_{}({v})", scalar(*ty, target)),
            }
        }
        Expr::Select { condition, yes, no } => {
            if target == Target::Wgsl {
                format!(
                    "select({}, {}, {})",
                    render(no),
                    render(yes),
                    render(condition)
                )
            } else {
                format!("({} ? {} : {})", render(condition), render(yes), render(no))
            }
        }
        Expr::Max(a, b) => format!("max({}, {})", render(a), render(b)),
        Expr::Min(a, b) => format!("min({}, {})", render(a), render(b)),
        Expr::HalfToFloat(value) => format!("ir_half({})", render(value)),
    }
}

fn statements(body: &[Stmt], target: Target, level: usize, output: &mut String) {
    let indent = "    ".repeat(level);
    for statement in body {
        match statement {
            Stmt::Declare { name, ty, value } => {
                if target == Target::Wgsl {
                    writeln!(
                        output,
                        "{indent}var v_{name}: {} = {};",
                        scalar(*ty, target),
                        expr(value, target)
                    )
                    .unwrap();
                } else {
                    writeln!(
                        output,
                        "{indent}{} v_{name} = {};",
                        scalar(*ty, target),
                        expr(value, target)
                    )
                    .unwrap();
                }
            }
            Stmt::Assign { name, value } => {
                writeln!(output, "{indent}v_{name} = {};", expr(value, target)).unwrap();
            }
            Stmt::Store {
                buffer,
                index,
                value,
            } => {
                writeln!(
                    output,
                    "{indent}buf_{buffer}[{}] = {};",
                    expr(index, target),
                    expr(value, target)
                )
                .unwrap();
            }
            Stmt::For { index, end, body } => {
                // IR range bounds are evaluated once on loop entry. A scoped
                // temporary keeps mutation of referenced outer variables from
                // changing generated shader behavior relative to the interpreter.
                writeln!(output, "{indent}{{").unwrap();
                let loop_indent = format!("{indent}    ");
                let bound = format!("ir_bound_{level}");
                if target == Target::Wgsl {
                    writeln!(
                        output,
                        "{loop_indent}let {bound}: u32 = {};",
                        expr(end, target)
                    )
                    .unwrap();
                    writeln!(output,"{loop_indent}for (var v_{index}: u32 = 0u; v_{index} < {bound}; v_{index} = v_{index} + 1u) {{").unwrap();
                } else {
                    writeln!(output, "{loop_indent}uint {bound} = {};", expr(end, target)).unwrap();
                    writeln!(
                        output,
                        "{loop_indent}for (uint v_{index} = 0u; v_{index} < {bound}; ++v_{index}) {{"
                    )
                    .unwrap();
                }
                statements(body, target, level + 2, output);
                writeln!(output, "{loop_indent}}}").unwrap();
                writeln!(output, "{indent}}}").unwrap();
            }
            Stmt::If { condition, yes, no } => {
                writeln!(output, "{indent}if ({}) {{", expr(condition, target)).unwrap();
                statements(yes, target, level + 1, output);
                if no.is_empty() {
                    writeln!(output, "{indent}}}").unwrap();
                } else {
                    writeln!(output, "{indent}}} else {{").unwrap();
                    statements(no, target, level + 1, output);
                    writeln!(output, "{indent}}}").unwrap();
                }
            }
        }
    }
}

fn half_helper(target: Target, output: &mut String) {
    // The conversion semantics are authored as Rust expression nodes. Emitters
    // generate syntax; no original OpenCL/Metal/WGSL helper is embedded here.
    let h = Expr::var("h");
    let sign = h.clone().binary(Binary::BitAnd, Expr::Uint(0x8000));
    let exponent = h
        .clone()
        .binary(Binary::Shr, Expr::Uint(10))
        .binary(Binary::BitAnd, Expr::Uint(31));
    let mantissa = h.binary(Binary::BitAnd, Expr::Uint(1023));
    let signed_subnormal = Expr::Select {
        condition: Box::new(Expr::var("sign").binary(Binary::Eq, Expr::Uint(0))),
        yes: Box::new(Expr::Float(1.0)),
        no: Box::new(Expr::Float(-1.0)),
    }
    .binary(Binary::Mul, Expr::var("mantissa").cast(Scalar::F32))
    .binary(Binary::Mul, Expr::Float(2f32.powi(-24)));
    let special = Expr::var("sign")
        .binary(Binary::Shl, Expr::Uint(16))
        .binary(Binary::BitOr, Expr::Uint(0x7f800000))
        .binary(
            Binary::BitOr,
            Expr::var("mantissa").binary(Binary::Shl, Expr::Uint(13)),
        )
        .bitcast(Scalar::F32);
    let normal = Expr::var("sign")
        .binary(Binary::Shl, Expr::Uint(16))
        .binary(
            Binary::BitOr,
            Expr::var("exponent")
                .binary(Binary::Add, Expr::Uint(112))
                .binary(Binary::Shl, Expr::Uint(23)),
        )
        .binary(
            Binary::BitOr,
            Expr::var("mantissa").binary(Binary::Shl, Expr::Uint(13)),
        )
        .bitcast(Scalar::F32);
    let result = Expr::Select {
        condition: Box::new(Expr::var("exponent").binary(Binary::Eq, Expr::Uint(0))),
        yes: Box::new(signed_subnormal),
        no: Box::new(Expr::Select {
            condition: Box::new(Expr::var("exponent").binary(Binary::Eq, Expr::Uint(31))),
            yes: Box::new(special),
            no: Box::new(normal),
        }),
    };
    if target == Target::Wgsl {
        output.push_str("fn ir_half(v_h: u32) -> f32 {\n");
    } else {
        output.push_str("float ir_half(uint v_h) {\n");
    }
    statements(
        &[
            Stmt::Declare {
                name: "sign".into(),
                ty: Scalar::U32,
                value: sign,
            },
            Stmt::Declare {
                name: "exponent".into(),
                ty: Scalar::U32,
                value: exponent,
            },
            Stmt::Declare {
                name: "mantissa".into(),
                ty: Scalar::U32,
                value: mantissa,
            },
        ],
        target,
        1,
        output,
    );
    writeln!(output, "    return {};\n}}\n", expr(&result, target)).unwrap();
}

pub fn emit(kernel: &Kernel, target: Target) -> Result<Artifact> {
    kernel.validate()?;
    let mut output = String::new();
    if target == Target::Glsl {
        output.push_str("#version 450\n");
    }
    writeln!(output,"// Generated from Rust kernel IR v{}. Do not hand edit.\n// Kernel: {}. No foreign kernel template input.",kernel.version,kernel.name).unwrap();
    match target {
        Target::Msl => output.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n"),
        Target::Glsl => {
            // Overloads encode explicit bit reinterpretation semantics.
            output.push_str("float ir_bits_float(uint x) { return uintBitsToFloat(x); }\nfloat ir_bits_float(int x) { return intBitsToFloat(x); }\nfloat ir_bits_float(float x) { return x; }\nuint ir_bits_uint(float x) { return floatBitsToUint(x); }\nuint ir_bits_uint(uint x) { return x; }\nuint ir_bits_uint(int x) { return uint(x); }\nint ir_bits_int(float x) { return floatBitsToInt(x); }\nint ir_bits_int(int x) { return x; }\nint ir_bits_int(uint x) { return int(x); }\n");
        }
        _ => {}
    }
    half_helper(target, &mut output);
    match target {
        Target::Wgsl => {
            for (binding, b) in kernel.buffers.iter().enumerate() {
                writeln!(
                    output,
                    "@group(0) @binding({binding}) var<storage, {}> buf_{}: array<{}>;",
                    if b.access == Access::Read {
                        "read"
                    } else {
                        "read_write"
                    },
                    b.name,
                    scalar(b.scalar, target)
                )
                .unwrap();
            }
            writeln!(output,"\n@compute @workgroup_size({}, 1, 1)\nfn {}(@builtin(global_invocation_id) ir_global: vec3<u32>) {{\n    let ir_gid: u32 = ir_global.x;",kernel.workgroup_size,kernel.name).unwrap();
        }
        Target::Glsl => {
            writeln!(
                output,
                "layout(local_size_x = {}, local_size_y = 1, local_size_z = 1) in;",
                kernel.workgroup_size
            )
            .unwrap();
            for (binding, b) in kernel.buffers.iter().enumerate() {
                writeln!(output,"layout(std430, set = 0, binding = {binding}) {}buffer ir_Block_{binding} {{ {} buf_{}[]; }};",if b.access==Access::Read {"readonly "}else{""},scalar(b.scalar,target),b.name).unwrap();
            }
            output.push_str("\nvoid main() {\n    uint ir_gid = gl_GlobalInvocationID.x;\n");
        }
        Target::OpenCl => {
            writeln!(
                output,
                "__attribute__((reqd_work_group_size({}, 1, 1)))",
                kernel.workgroup_size
            )
            .unwrap();
            write!(output, "__kernel void {}(", kernel.name).unwrap();
            for (index, b) in kernel.buffers.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write!(
                    output,
                    "__global {}{} *buf_{}",
                    if b.access == Access::Read {
                        "const "
                    } else {
                        ""
                    },
                    scalar(b.scalar, target),
                    b.name
                )
                .unwrap();
            }
            output.push_str(") {\n    uint ir_gid = (uint)get_global_id(0);\n");
        }
        Target::Msl => {
            write!(output, "kernel void {}(", kernel.name).unwrap();
            for (index, b) in kernel.buffers.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write!(
                    output,
                    "device {}{} *buf_{} [[buffer({index})]]",
                    if b.access == Access::Read {
                        "const "
                    } else {
                        ""
                    },
                    scalar(b.scalar, target),
                    b.name
                )
                .unwrap();
            }
            output.push_str(", uint ir_gid [[thread_position_in_grid]]) {\n");
        }
    }
    writeln!(
        output,
        "    if (ir_gid >= {}u) {{ return; }}",
        kernel.invocations
    )
    .unwrap();
    statements(&kernel.body, target, 1, &mut output);
    output.push_str("}\n");
    let sha256 = sha256_hex(output.as_bytes());
    Ok(Artifact {
        target,
        entry_point: if target == Target::Glsl {
            "main".into()
        } else {
            kernel.name.clone()
        },
        source: output,
        sha256,
        ir_version: kernel.version,
    })
}
