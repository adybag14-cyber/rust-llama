use llama_tensor::{DType, GluOp, Graph, Op, Tensor, TensorDesc, TensorError, Triangle, UnaryOp};
use std::sync::atomic::AtomicBool;

struct Reader {
    bytes: &'static [u8],
    at: usize,
}
impl Reader {
    fn take(&mut self, n: usize) -> &'static [u8] {
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        out
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn i64(&mut self) -> i64 {
        i64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn string(&mut self) -> &'static str {
        let n = self.u32() as usize;
        std::str::from_utf8(self.take(n)).unwrap()
    }
    fn tensor(&mut self, strided: bool) -> Tensor {
        let dtype = DType::from_id(self.u32()).unwrap();
        let shape = std::array::from_fn(|_| self.u64());
        let desc = if strided {
            TensorDesc::with_strides(dtype, shape, std::array::from_fn(|_| self.u64())).unwrap()
        } else {
            TensorDesc::contiguous(dtype, shape).unwrap()
        };
        let bytes = self.u64();
        Tensor::from_bytes(desc, self.take(bytes).to_vec()).unwrap()
    }
}
fn op(tag: &str, ints: &[i64], floats: &[f32], input_count: usize) -> Op {
    let shape = || std::array::from_fn(|d| ints[d] as usize);
    match tag {
        "cumsum" => Op::CumSum,
        "sum" => Op::Sum,
        "sum_rows" => Op::SumRows,
        "mean" => Op::Mean,
        "sqr" => Op::Square,
        "sqrt" => Op::Sqrt,
        "log" => Op::Log,
        "sin" => Op::Sin,
        "cos" => Op::Cos,
        "norm" => Op::Norm { epsilon: floats[0] },
        "argmax" => Op::ArgMax,
        "count_equal" => Op::CountEqual,
        "repeat_back" => Op::RepeatBack { shape: shape() },
        "group_norm" => Op::GroupNorm {
            groups: ints[0] as usize,
            epsilon: floats[0],
        },
        "l2_norm" => Op::L2Norm { epsilon: floats[0] },
        "out_prod" => Op::OutProd,
        "mul_mat" => Op::MatMul,
        "diag" => Op::Diag,
        "pad" => Op::Pad {
            left: std::array::from_fn(|d| ints[d] as i32),
            right: std::array::from_fn(|d| ints[d + 4] as i32),
            circular: ints[8] != 0,
        },
        "reflect" => Op::PadReflect1d {
            left: ints[0] as usize,
            right: ints[1] as usize,
        },
        "roll" => Op::Roll {
            shifts: std::array::from_fn(|d| ints[d] as i32),
        },
        "arange" => Op::Arange {
            start: floats[0],
            stop: floats[1],
            step: floats[2],
        },
        "timestep" => Op::TimestepEmbedding {
            dimensions: ints[0] as usize,
            max_period: ints[1] as usize,
        },
        "tri" => Op::Tri {
            triangle: [
                Triangle::UpperDiagonal,
                Triangle::Upper,
                Triangle::LowerDiagonal,
                Triangle::Lower,
            ][ints[0] as usize],
        },
        "fill" => Op::Fill { value: floats[0] },
        "leaky_relu" => Op::LeakyRelu {
            negative_slope: floats[0],
        },
        "silu_back" => Op::SiluBack,
        "get_rows_back" => Op::GetRowsBack {
            rows: ints[0] as usize,
        },
        "rms_norm_back" => Op::RmsNormBack { epsilon: floats[0] },
        "softmax_back" => Op::SoftmaxBack { scale: floats[0] },
        "view" => Op::View {
            shape: shape(),
            strides: std::array::from_fn(|d| ints[d + 4] as usize),
            offset: ints[8] as usize,
        },
        "unary" => Op::Unary(UnaryOp::from_id(ints[0] as u32).unwrap()),
        "xielu" => Op::Xielu {
            alpha_n: floats[0],
            alpha_p: floats[1],
            beta: floats[2],
            epsilon: floats[3],
        },
        "glu" => Op::Glu {
            kind: GluOp::from_id(ints[0] as u32).unwrap(),
            swap: ints[1] != 0,
        },
        "glu_packed" => Op::GluPacked {
            kind: GluOp::from_id(ints[0] as u32).unwrap(),
            swap: ints[1] != 0,
        },
        "swiglu_oai" if input_count == 2 => Op::SwiGluOai {
            alpha: floats[0],
            limit: floats[1],
        },
        "swiglu_oai" => Op::SwiGluOaiPacked {
            alpha: floats[0],
            limit: floats[1],
            swap: ints[0] != 0,
        },
        "swiglu_clamp" if input_count == 2 => Op::SwiGluClamp { limit: floats[0] },
        "swiglu_clamp" => Op::SwiGluClampPacked {
            limit: floats[0],
            swap: ints[0] != 0,
        },
        _ => panic!("fixture operation {tag}"),
    }
}
fn evaluate(op: Op, inputs: Vec<Tensor>) -> llama_tensor::Result<Tensor> {
    let mut graph = Graph::new();
    let ids = inputs
        .into_iter()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(op, &ids)?;
    graph.evaluate(target)
}
fn tensor(shape: [usize; 4]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(
        shape,
        &(0..n).map(|i| i as f32 * 0.125 - 1.0).collect::<Vec<_>>(),
    )
    .unwrap()
}

fn final_conversion_error(dtype: DType, x: f32) -> f32 {
    // Each final nearest-even conversion contributes at most half of the
    // adjacent output spacing. F32 retains the unchanged base comparison gate.
    if dtype == DType::F32 || !x.is_finite() {
        return 0.0;
    }
    let (bits, decode): (u16, fn(u16) -> f32) = match dtype {
        DType::F16 => (llama_quant::f32_to_f16(x.abs()), llama_quant::f16_to_f32),
        DType::BF16 => (llama_quant::f32_to_bf16(x.abs()), llama_quant::bf16_to_f32),
        _ => return 0.0,
    };
    let center = decode(bits);
    let below = decode(bits.saturating_sub(1));
    let above = decode(bits + 1);
    let mut spacing = center - below;
    if above.is_finite() {
        spacing = spacing.max(above - center);
    }
    0.5 * spacing
}

#[test]
fn extended_cpu_graphs_match_independent_native_outputs() {
    let bytes = include_bytes!("fixtures/cpu-extended-reference.bin");
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGTR0002");
    let cases = r.u32();
    let mut exact_values = 0usize;
    let mut approximate_values = 0usize;
    let mut maximum_error = 0.0_f32;
    let mut differing_transcendental_values = 0usize;
    let mut conversion_envelope_values = 0usize;
    let mut maximum_f32_error = 0.0_f32;
    let mut failures = Vec::new();
    for _ in 0..cases {
        let name = r.string();
        let tag = r.string();
        let approximate = r.u32() != 0;
        let ni = r.u32();
        let ints = (0..ni).map(|_| r.i64()).collect::<Vec<_>>();
        let nf = r.u32();
        let floats = (0..nf).map(|_| r.f32()).collect::<Vec<_>>();
        let nt = r.u32() as usize;
        let inputs = (0..nt).map(|_| r.tensor(true)).collect::<Vec<_>>();
        let expected = r.tensor(false);
        let actual = match evaluate(op(tag, &ints, &floats, nt), inputs) {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!("{name}: {error}"));
                continue;
            }
        };
        assert_eq!(actual.shape(), expected.shape(), "{name}");
        assert_eq!(actual.dtype(), expected.dtype(), "{name}");
        if expected.dtype() == DType::I32 {
            assert_eq!(
                actual.to_i32().unwrap(),
                expected.to_i32().unwrap(),
                "{name}"
            );
        } else if expected.dtype() == DType::I64 {
            assert_eq!(
                actual.to_i64().unwrap(),
                expected.to_i64().unwrap(),
                "{name}"
            );
        } else {
            let dtype = actual.dtype();
            let actual = actual.to_f32().unwrap();
            let expected = expected.to_f32().unwrap();
            let mut case_bad = 0;
            for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                if approximate {
                    approximate_values += 1;
                } else {
                    exact_values += 1;
                }
                let equal_bits = a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
                if !equal_bits {
                    let error = (a - b).abs();
                    if approximate {
                        differing_transcendental_values += 1;
                        maximum_error = maximum_error.max(error);
                        if dtype == DType::F32 {
                            maximum_f32_error = maximum_f32_error.max(error);
                        }
                    }
                    let base = 2e-6 + 2e-6 * b.abs();
                    let conversion =
                        final_conversion_error(dtype, a) + final_conversion_error(dtype, b);
                    let matches =
                        approximate && a.is_finite() && b.is_finite() && error <= base + conversion;
                    if matches && error > base {
                        conversion_envelope_values += 1;
                    }
                    if !matches {
                        case_bad += 1;
                        if case_bad <= 2 {
                            failures.push(format!(
                                "{name}[{i}]: Rust {a:?} native {b:?}, delta {error}"
                            ));
                        }
                    }
                }
            }
            if case_bad > 0 {
                failures.push(format!("{name}: {case_bad} mismatches"));
            }
        }
    }
    assert_eq!(r.at, bytes.len());
    println!(
        "{cases} native graph cases; {exact_values} bit-exact checked values; {approximate_values} transcendental values; {differing_transcendental_values} transcendental bit differences; {conversion_envelope_values} require final 16-bit rounding envelope; max F32 error {maximum_f32_error}; max all-dtype error {maximum_error}. Exact platform ERF parity remains open."
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn checked_extension_parameters_and_native_undefined_addresses_are_errors() {
    let a = tensor([4, 3, 2, 1]);
    for op in [
        Op::RepeatBack {
            shape: [3, 1, 1, 1],
        },
        Op::GroupNorm {
            groups: 0,
            epsilon: 1e-5,
        },
        Op::GroupNorm {
            groups: 2,
            epsilon: -1.,
        },
        Op::L2Norm { epsilon: f32::NAN },
        Op::Pad {
            left: [-9, 0, 0, 0],
            right: [10, 0, 0, 0],
            circular: false,
        },
        Op::Pad {
            left: [5, 0, 0, 0],
            right: [0, 0, 0, 0],
            circular: true,
        },
        Op::PadReflect1d { left: 4, right: 0 },
        Op::Roll {
            shifts: [4, 0, 0, 0],
        },
        Op::TimestepEmbedding {
            dimensions: 4,
            max_period: 0,
        },
        Op::ArgMax,
        Op::Diag,
        Op::Tri {
            triangle: Triangle::Lower,
        },
        Op::Xielu {
            alpha_n: 0.,
            alpha_p: 0.,
            beta: f32::NAN,
            epsilon: -1e-6,
        },
        Op::GluPacked {
            kind: GluOp::SWIGLU_OAI,
            swap: false,
        },
        Op::View {
            shape: [4, 3, 2, 1],
            strides: [4, 16, 48, 96],
            offset: 4,
        },
    ] {
        assert!(evaluate(op.clone(), vec![a.clone()]).is_err(), "{op:?}");
    }
    for op in [
        Op::Arange {
            start: 0.,
            stop: 1.,
            step: 0.,
        },
        Op::Arange {
            start: 1.,
            stop: 0.,
            step: -1.,
        },
        Op::Arange {
            start: 0.,
            stop: f32::MAX,
            step: f32::MIN_POSITIVE,
        },
    ] {
        assert!(evaluate(op, vec![]).is_err());
    }
    let integers = Tensor::from_i32([2, 3, 2, 1], &[1; 12]).unwrap();
    assert!(matches!(
        evaluate(Op::CountEqual, vec![integers.clone(), integers]),
        Err(TensorError::Bounds)
    ));
    assert!(
        evaluate(
            Op::GetRowsBack { rows: 2 },
            vec![
                tensor([4, 3, 1, 1]),
                Tensor::from_i32([3, 1, 1, 1], &[0, 2, 1]).unwrap()
            ]
        )
        .is_err()
    );
    assert!(
        evaluate(
            Op::SwiGluOai {
                alpha: 1.,
                limit: -1.
            },
            vec![a.clone(), a.clone()]
        )
        .is_err()
    );
    let mut graph = Graph::new();
    let input = graph.input(a);
    assert!(
        graph
            .op(
                Op::Arange {
                    start: 0.,
                    stop: 1.,
                    step: 0.1
                },
                &[input]
            )
            .is_err()
    );
}
#[test]
fn packed_rhs_matmul_is_not_requantized_and_scalar_nonfinite_is_preserved() {
    // Q8_0's prepacked scale and quants intentionally do not use the canonical
    // scale that requantizing its decoded values would choose.
    let mut weight = Vec::new();
    weight.extend_from_slice(&llama_quant::f32_to_f16(0.25).to_le_bytes());
    weight.extend_from_slice(&[0x98; 16]);
    let a = Tensor::from_bytes(
        TensorDesc::contiguous(DType::Q4_0, [32, 1, 1, 1]).unwrap(),
        weight.clone(),
    )
    .unwrap();
    let mut activation = Vec::new();
    activation.extend_from_slice(&llama_quant::f32_to_f16(0.75).to_le_bytes());
    activation.extend((0..32).map(|i| (i as i8 - 16) as u8));
    let b = Tensor::from_bytes(
        TensorDesc::contiguous(DType::Q8_0, [32, 1, 1, 1]).unwrap(),
        activation.clone(),
    )
    .unwrap();
    let expected = llama_quant::dot_packed(DType::Q4_0, &weight, &activation, 32).unwrap();
    assert_eq!(
        evaluate(Op::MatMul, vec![a, b]).unwrap().to_f32().unwrap()[0].to_bits(),
        expected.to_bits()
    );
    let a = Tensor::from_f32([2, 1, 1, 1], &[1., 1.]).unwrap();
    let b = Tensor::from_f32([2, 1, 1, 1], &[f32::INFINITY, 1.]).unwrap();
    assert_eq!(
        evaluate(Op::MatMul, vec![a, b]).unwrap().to_f32().unwrap(),
        [f32::INFINITY]
    );
}
#[test]
fn views_zero_input_nodes_and_i64_storage_obey_graph_contract() {
    let a = tensor([8, 4, 1, 1]);
    let output = evaluate(
        Op::View {
            shape: [4, 2, 1, 1],
            strides: [4, 32, 128, 128],
            offset: 8,
        },
        vec![a.clone()],
    )
    .unwrap();
    assert!(output.shares_storage(&a));
    assert_eq!(
        output.to_f32().unwrap(),
        [-0.75, -0.625, -0.5, -0.375, 0.25, 0.375, 0.5, 0.625]
    );
    let original = [i64::MIN, i64::MAX, (1i64 << 55) + 7, -4];
    let i = Tensor::from_i64([2, 2, 1, 1], &original).unwrap();
    assert_eq!(
        i.permute([1, 0, 2, 3]).unwrap().to_i64().unwrap(),
        [original[0], original[2], original[1], original[3]]
    );
    let mut graph = Graph::new();
    let target = graph
        .op(
            Op::Arange {
                start: 0.,
                stop: 2.,
                step: 0.25,
            },
            &[],
        )
        .unwrap();
    assert!(matches!(
        graph.evaluate_with_cancel(target, &AtomicBool::new(true)),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(
        graph.evaluate(target).unwrap().to_f32().unwrap(),
        [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75]
    );
}

#[test]
fn empty_and_extreme_descriptors_do_not_bypass_checked_arithmetic() {
    let empty = Tensor::from_f32([3, 2, 1, 0], &[]).unwrap();
    let result = evaluate(
        Op::GroupNorm {
            groups: 1,
            epsilon: 1e-5,
        },
        vec![empty],
    )
    .unwrap();
    assert!(result.to_f32().unwrap().is_empty());
    let extreme = Tensor::from_bytes(
        TensorDesc::with_strides(DType::F32, [usize::MAX / 16, 32, 2, 0], [4, 0, 0, 0]).unwrap(),
        vec![],
    )
    .unwrap();
    assert!(matches!(
        evaluate(
            Op::GroupNorm {
                groups: 1,
                epsilon: 1e-5
            },
            vec![extreme]
        ),
        Err(TensorError::Overflow)
    ));
    let half = Tensor::from_bytes(
        TensorDesc::contiguous(DType::F16, [2, 2, 1, 1]).unwrap(),
        vec![0; 8],
    )
    .unwrap();
    assert!(matches!(
        evaluate(Op::Sum, vec![half.permute([1, 0, 2, 3]).unwrap()]),
        Err(TensorError::Unsupported(_))
    ));
    let bf16 = Tensor::from_bytes(
        TensorDesc::contiguous(DType::BF16, [4, 1, 1, 1]).unwrap(),
        vec![0; 8],
    )
    .unwrap();
    assert!(matches!(
        evaluate(
            Op::SwiGluClampPacked {
                limit: 1.,
                swap: false
            },
            vec![bf16]
        ),
        Err(TensorError::Unsupported(_))
    ));
}
