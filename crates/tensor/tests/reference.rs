use llama_tensor::{
    DType, GluOp, Graph, Op, OpCode, RopeMode, Tensor, TensorDesc, TensorError, UnaryOp,
};
use std::sync::atomic::{AtomicBool, Ordering};

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn string(&mut self) -> String {
        let n = self.u32();
        std::str::from_utf8(self.take(n)).unwrap().to_string()
    }
    fn tensor(&mut self) -> Tensor {
        let dtype = DType::from_id(self.u32() as u32).unwrap();
        let ne = std::array::from_fn(|_| self.u32());
        let n = self.u32();
        Tensor::from_bytes(
            TensorDesc::contiguous(dtype, ne).unwrap(),
            self.take(n).to_vec(),
        )
        .unwrap()
    }
}
#[test]
fn cpu_dag_matches_65_frozen_ggml_graph_cases() {
    let mut r = Reader {
        b: include_bytes!("fixtures/cpu-reference.bin"),
        p: 0,
    };
    assert_eq!(r.take(8), b"GGTR0001");
    let count = r.u32();
    assert_eq!(count, 65);
    for _ in 0..count {
        let name = r.string();
        let tag = r.string();
        let ni = r.u32();
        let ints = (0..ni).map(|_| r.u32()).collect::<Vec<_>>();
        let nf = r.u32();
        let floats = (0..nf).map(|_| r.f32()).collect::<Vec<_>>();
        let na = r.u32();
        let inputs = (0..na).map(|_| r.tensor()).collect::<Vec<_>>();
        let expected = r.tensor();
        let op = match tag.as_str() {
            "dup" => Op::Dup,
            "sqr" => Op::Square,
            "sqrt" => Op::Sqrt,
            "log" => Op::Log,
            "sin" => Op::Sin,
            "cos" => Op::Cos,
            "cont" => Op::Contiguous,
            "add" => Op::Add,
            "sub" => Op::Sub,
            "mul" => Op::Mul,
            "div" => Op::Div,
            "add1" => Op::AddScalar,
            "sum" => Op::Sum,
            "sum_rows" => Op::SumRows,
            "mean" => Op::Mean,
            "scale" => Op::Scale {
                scale: floats[0],
                bias: floats[1],
            },
            "repeat" => Op::Repeat {
                shape: expected.shape(),
            },
            "concat" => Op::Concat { axis: ints[0] },
            "mul_mat" => Op::MatMul,
            "softmax" => Op::Softmax { scale: floats[0] },
            "norm" => Op::Norm { epsilon: floats[0] },
            "rms_norm" => Op::RmsNorm { epsilon: floats[0] },
            "unary" => Op::Unary(UnaryOp::from_id(ints[0] as u32).unwrap()),
            "glu" => Op::Glu {
                kind: GluOp::from_id(ints[0] as u32).unwrap(),
                swap: ints[1] != 0,
            },
            "clamp" => Op::Clamp {
                min: floats[0],
                max: floats[1],
            },
            "mask" => Op::DiagMask {
                past: ints[0],
                zero: ints[1] != 0,
            },
            "reshape" => Op::Reshape {
                shape: expected.shape(),
            },
            "transpose" => Op::Transpose,
            "permute" => Op::Permute {
                axes: ints.clone().try_into().unwrap(),
            },
            "rope" => Op::Rope {
                dimensions: ints[0],
                mode: if ints[1] == 0 {
                    RopeMode::Normal
                } else {
                    RopeMode::NeoX
                },
                positions: ints[2..].iter().map(|&v| v as i32).collect(),
                freq_base: floats[0],
                freq_scale: floats[1],
            },
            "get_rows" => Op::GetRows,
            _ => panic!("unknown fixture {tag}"),
        };
        let mut graph = Graph::new();
        let ids = inputs
            .into_iter()
            .map(|x| graph.input(x))
            .collect::<Vec<_>>();
        let target = graph.op(op, &ids).unwrap();
        let actual = graph
            .evaluate(target)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(actual.shape(), expected.shape(), "{name} shape");
        assert_eq!(actual.dtype(), expected.dtype(), "{name} dtype");
        if actual.dtype() == DType::I32 {
            assert_eq!(
                actual.to_i32().unwrap(),
                expected.to_i32().unwrap(),
                "{name} exact I32 values"
            );
        }
        for (i, (&x, &y)) in actual
            .to_f32()
            .unwrap()
            .iter()
            .zip(expected.to_f32().unwrap().iter())
            .enumerate()
        {
            let tolerance = 3e-6_f32 + 3e-6 * y.abs();
            assert!(
                x.to_bits() == y.to_bits()
                    || (x.is_nan() && y.is_nan())
                    || (x - y).abs() <= tolerance,
                "{name} output {i}: actual {x}, expected {y}, tolerance {tolerance}"
            );
        }
        let repeated = graph.evaluate(target).unwrap().to_f32().unwrap();
        assert_eq!(actual.to_f32().unwrap(), repeated, "graph reuse {name}");
    }
    assert_eq!(r.p, r.b.len());
}
#[test]
fn views_keep_storage_alive_and_do_not_alias_output_writes() {
    let original =
        Tensor::from_f32([4, 3, 1, 1], &(0..12).map(|v| v as f32).collect::<Vec<_>>()).unwrap();
    let desc = TensorDesc::with_strides(DType::F32, [2, 2, 1, 1], [4, 16, 48, 48]).unwrap();
    let view = original.view(desc, 4).unwrap();
    assert!(view.shares_storage(&original));
    drop(original);
    assert_eq!(view.to_f32().unwrap(), [1.0, 2.0, 5.0, 6.0]);
    let copy = view.contiguous().unwrap();
    assert!(!copy.shares_storage(&view));
    assert_eq!(copy.to_f32().unwrap(), view.to_f32().unwrap());
    assert!(copy.desc().is_contiguous());
    assert!(!view.desc().is_contiguous());
    assert!(matches!(
        view.reshape([4, 1, 1, 1]),
        Err(TensorError::Layout(_))
    ));
}
#[test]
fn quantized_views_use_block_strides_and_preserve_bytes() {
    let values = (0..128).map(|i| i as f32 - 64.0).collect::<Vec<_>>();
    let bytes = llama_quant::quantize(DType::Q4_0, &values).unwrap();
    let t = Tensor::from_bytes(
        TensorDesc::contiguous(DType::Q4_0, [64, 2, 1, 1]).unwrap(),
        bytes.clone(),
    )
    .unwrap();
    assert_eq!(t.desc().nb, [18, 36, 72, 72]);
    let view = t
        .view(
            TensorDesc::with_strides(DType::Q4_0, [32, 2, 1, 1], [18, 36, 72, 72]).unwrap(),
            18,
        )
        .unwrap();
    let expected = [
        llama_quant::dequantize(DType::Q4_0, &bytes[18..36], 32).unwrap(),
        llama_quant::dequantize(DType::Q4_0, &bytes[54..72], 32).unwrap(),
    ]
    .concat();
    assert_eq!(view.to_f32().unwrap(), expected);
    assert_eq!(view.contiguous().unwrap().to_f32().unwrap(), expected);
    assert!(matches!(
        t.permute([1, 0, 2, 3]),
        Err(TensorError::Unsupported(_))
    ));
}
#[test]
fn checked_layouts_reject_overflow_invalid_blocks_and_bounds() {
    assert!(TensorDesc::contiguous(DType::F64, [usize::MAX, 2, 1, 1]).is_err());
    assert!(TensorDesc::contiguous(DType::Q4_0, [31, 1, 1, 1]).is_err());
    assert!(TensorDesc::with_strides(DType::F32, [4, 1, 1, 1], [2, 16, 16, 16]).is_err());
    let t = Tensor::from_f32([4, 1, 1, 1], &[0.0; 4]).unwrap();
    assert!(matches!(
        t.view(
            TensorDesc::contiguous(DType::F32, [2, 1, 1, 1]).unwrap(),
            12
        ),
        Err(TensorError::Bounds)
    ));
    assert!(matches!(
        t.view(TensorDesc::contiguous(DType::F32, [1, 1, 1, 1]).unwrap(), 1),
        Err(TensorError::Layout(_))
    ));
    assert!(t.permute([0, 0, 2, 3]).is_err());
    assert!(t.reshape([3, 1, 1, 1]).is_err());
    let zero = Tensor::from_f32([0, 2, 1, 1], &[]).unwrap();
    assert!(zero.to_f32().unwrap().is_empty());
}
#[test]
fn cancellation_and_unrelated_branches_do_not_change_inputs() {
    let input = Tensor::from_f32([4, 1, 1, 1], &[1., 2., 3., 4.]).unwrap();
    let mut graph = Graph::new();
    let a = graph.input(input.clone());
    let pending = graph.op(Op::Unary(UnaryOp::GELU_ERF), &[a]).unwrap();
    let target = graph.op(Op::Square, &[a]).unwrap();
    assert!(matches!(
        graph.evaluate(pending),
        Err(TensorError::Unsupported(_))
    ));
    assert_eq!(
        graph.evaluate(target).unwrap().to_f32().unwrap(),
        [1., 4., 9., 16.]
    );
    let cancel = AtomicBool::new(true);
    assert!(matches!(
        graph.evaluate_with_cancel(target, &cancel),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(input.to_f32().unwrap(), [1., 2., 3., 4.]);
    cancel.store(false, Ordering::Relaxed);
    assert_eq!(
        graph
            .evaluate_with_cancel(target, &cancel)
            .unwrap()
            .to_f32()
            .unwrap(),
        [1., 4., 9., 16.]
    );
}
#[test]
fn full_registries_are_exact_but_pending_ops_are_not_execution_claims() {
    assert_eq!(OpCode::ALL.len(), 102);
    assert_eq!(UnaryOp::ALL.len(), 22);
    assert_eq!(GluOp::ALL.len(), 7);
    assert_eq!(OpCode::MUL_MAT.id(), 29);
    assert_eq!(OpCode::GLU.id(), 101);
    assert_eq!(OpCode::from_id(102), None);
    assert_eq!(UnaryOp::from_id(22), None);
}
#[test]
fn invalid_shapes_types_indices_and_softmax_are_errors() {
    let mut g = Graph::new();
    let a = g.input(Tensor::from_f32([4, 2, 1, 1], &[1.; 8]).unwrap());
    let b = g.input(Tensor::from_f32([3, 1, 1, 1], &[1.; 3]).unwrap());
    let target = g.op(Op::Add, &[a, b]).unwrap();
    assert!(g.evaluate(target).is_err());
    assert!(g.op(Op::Add, &[a]).is_err());
    let i = g.input(Tensor::from_i32([2, 1, 1, 1], &[0, 2]).unwrap());
    let rows = g.op(Op::GetRows, &[a, i]).unwrap();
    assert!(matches!(g.evaluate(rows), Err(TensorError::Bounds)));
    let all_masked = g.input(Tensor::from_f32([4, 1, 1, 1], &[f32::NEG_INFINITY; 4]).unwrap());
    let sm = g.op(Op::Softmax { scale: 1. }, &[all_masked]).unwrap();
    assert!(matches!(g.evaluate(sm), Err(TensorError::Parameters(_))));
    let integer = g.op(Op::Square, &[i]).unwrap();
    assert!(matches!(g.evaluate(integer), Err(TensorError::Type { .. })));
}

#[test]
fn node_ids_are_bound_to_their_graph() {
    let mut first = Graph::new();
    let first_id = first.input(Tensor::from_f32([1; 4], &[2.0]).unwrap());
    let mut second = Graph::new();
    let second_id = second.input(Tensor::from_f32([1; 4], &[9.0]).unwrap());
    assert_eq!(first_id.index(), second_id.index());
    assert!(matches!(
        first.evaluate(second_id),
        Err(TensorError::InvalidNode(_))
    ));
    assert!(matches!(
        first.op(Op::Square, &[second_id]),
        Err(TensorError::InvalidNode(_))
    ));
    assert_eq!(first.evaluate(first_id).unwrap().to_f32().unwrap(), [2.0]);
}
