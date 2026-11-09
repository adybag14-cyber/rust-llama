use llama_tensor::{DType, Graph, Op, SortOrder, Tensor, TensorDesc, TensorError};
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

fn evaluate(op: Op, inputs: Vec<Tensor>) -> llama_tensor::Result<Tensor> {
    let mut graph = Graph::new();
    let ids = inputs
        .into_iter()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(op, &ids)?;
    graph.evaluate(target)
}
fn operation(tag: &str, ints: &[i64]) -> Op {
    match tag {
        "add_id" => Op::AddId,
        "mul_mat_id" => Op::MatMulId,
        "set_rows" => Op::SetRows,
        "argsort" => Op::Argsort {
            order: if ints[0] == 0 {
                SortOrder::Ascending
            } else {
                SortOrder::Descending
            },
        },
        "top_k" => Op::TopK {
            count: ints[0] as usize,
        },
        "cast" => Op::Cast {
            dtype: DType::from_id(ints[0] as u32).unwrap(),
        },
        "acc" => Op::Acc {
            strides: std::array::from_fn(|i| ints[i] as usize),
            offset: ints[3] as usize,
        },
        "set" => Op::Set {
            strides: std::array::from_fn(|i| ints[i] as usize),
            offset: ints[3] as usize,
        },
        "top_k_mask" => Op::TopKMask {
            with_scores: ints[0] != 0,
            with_blocks: ints[1] != 0,
            with_extra: ints[2] != 0,
        },
        _ => panic!("unknown fixture operation {tag}"),
    }
}
#[test]
#[cfg(any(target_env = "msvc", all(target_os = "linux", target_env = "gnu")))]
fn indexing_graphs_match_independent_platform_native_bytes() {
    #[cfg(target_env = "msvc")]
    let bytes = include_bytes!("fixtures/cpu-indexing-msvc-reference.bin");
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let bytes = include_bytes!("fixtures/cpu-indexing-gnu-reference.bin");
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGTR0002");
    let cases = r.u32();
    let mut compared_bytes = 0;
    let mut failures = Vec::new();
    for _ in 0..cases {
        let name = r.string();
        let tag = r.string();
        assert_eq!(
            r.u32(),
            0,
            "all indexing outputs require exact native bytes"
        );
        let ni = r.u32();
        let ints = (0..ni).map(|_| r.i64()).collect::<Vec<_>>();
        let nf = r.u32();
        let floats = (0..nf).map(|_| r.f32()).collect::<Vec<_>>();
        assert!(floats.is_empty());
        let nt = r.u32();
        let tensors = (0..nt).map(|_| r.tensor(true)).collect::<Vec<_>>();
        let expected = r.tensor(false);
        let got = match evaluate(operation(tag, &ints), tensors) {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!("{name}: error {error}"));
                continue;
            }
        };
        assert_eq!(got.dtype(), expected.dtype(), "{name} dtype");
        assert_eq!(got.shape(), expected.shape(), "{name} shape");
        let expected = expected.to_bytes().unwrap();
        let actual = got.to_bytes().unwrap();
        compared_bytes += expected.len();
        if actual != expected {
            let at = actual
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap();
            failures.push(format!(
                "{name}: byte {at}: got {:?}, expected {:?}",
                &actual[at..(at + 16).min(actual.len())],
                &expected[at..(at + 16).min(expected.len())]
            ));
        }
    }
    assert_eq!(r.at, bytes.len());
    assert_eq!(cases, 890);
    println!(
        "native indexing cases={cases}, exact compared bytes={compared_bytes}, failures={}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{}\n{}",
        failures.len(),
        failures
            .iter()
            .take(32)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
fn f32_tensor(shape: [usize; 4]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(
        shape,
        &(0..n).map(|i| i as f32 * 0.25 - 1.0).collect::<Vec<_>>(),
    )
    .unwrap()
}
#[test]
fn row_writes_are_transactional_and_preserve_aliases_and_layout() {
    let storage = f32_tensor([6, 4, 1, 1]);
    let base = storage
        .view(
            TensorDesc::with_strides(DType::F32, [4, 4, 1, 1], [4, 24, 96, 96]).unwrap(),
            0,
        )
        .unwrap();
    let original = base.to_bytes().unwrap();
    let values = f32_tensor([4, 3, 1, 1]);
    let indices = Tensor::from_i64([3, 1, 1, 1], &[2, 2, 0]).unwrap();
    let result = evaluate(Op::SetRows, vec![base.clone(), values.clone(), indices]).unwrap();
    assert_eq!(result.desc(), base.desc());
    assert!(!result.shares_storage(&base));
    assert_eq!(base.to_bytes().unwrap(), original);
    assert_eq!(
        &result.to_f32().unwrap()[8..12],
        &values.to_f32().unwrap()[4..8]
    );
    let bad = Tensor::from_i64([3, 1, 1, 1], &[2, 0, i64::MAX]).unwrap();
    assert_eq!(
        evaluate(Op::SetRows, vec![base.clone(), values, bad]).unwrap_err(),
        TensorError::Bounds
    );
    assert_eq!(base.to_bytes().unwrap(), original);
    assert!(base.shares_storage(&storage));
    let huge = Tensor::from_bytes(
        TensorDesc::with_strides(
            DType::F32,
            [0, usize::MAX, usize::MAX, usize::MAX],
            [4, 0, 0, 0],
        )
        .unwrap(),
        Vec::new(),
    )
    .unwrap();
    assert!(huge.to_bytes().unwrap().is_empty());
    assert!(huge.to_f32().unwrap().is_empty());
    assert!(huge.contiguous().unwrap().to_bytes().unwrap().is_empty());
}
#[test]
fn undefined_cpu_inputs_and_missing_converters_are_explicit_errors() {
    let base = f32_tensor([8, 2, 1, 1]);
    let values = f32_tensor([4, 2, 1, 1]);
    for op in [
        Op::Acc {
            strides: [4, 8, 8],
            offset: 63,
        },
        Op::Set {
            strides: [usize::MAX, 8, 8],
            offset: 0,
        },
        Op::Acc {
            strides: [32, 64, 64],
            offset: 32,
        },
    ] {
        assert!(evaluate(op, vec![base.clone(), values.clone()]).is_err());
    }
    let ids = Tensor::from_i32([2, 1, 1, 1], &[0, -2]).unwrap();
    assert_eq!(
        evaluate(
            Op::MatMulId,
            vec![f32_tensor([8, 3, 2, 1]), f32_tensor([8, 2, 1, 1]), ids]
        )
        .unwrap_err(),
        TensorError::Bounds
    );
    let q8 = Tensor::from_bytes(
        TensorDesc::contiguous(DType::Q8_K, [256, 1, 1, 1]).unwrap(),
        llama_quant::quantize(DType::Q8_K, &[0.0; 256]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        evaluate(Op::Cast { dtype: DType::F32 }, vec![q8]),
        Err(TensorError::Unsupported(_))
    ));
    assert!(matches!(
        evaluate(
            Op::Cast {
                dtype: DType::IQ2_XXS
            },
            vec![f32_tensor([256, 1, 1, 1])]
        ),
        Err(TensorError::Unsupported(_))
    ));
    let nan = Tensor::from_f32([2, 1, 1, 1], &[f32::NAN, 1.0]).unwrap();
    assert!(matches!(
        evaluate(
            Op::Argsort {
                order: SortOrder::Ascending
            },
            vec![nan.clone()]
        ),
        Err(TensorError::Parameters(_))
    ));
    assert!(evaluate(Op::Cast { dtype: DType::I32 }, vec![nan]).is_err());
    assert!(evaluate(Op::TopK { count: 9 }, vec![base]).is_err());
}
#[test]
fn indexed_operations_observe_graph_cancellation_before_any_write() {
    let base = f32_tensor([8, 2, 1, 1]);
    let before = base.to_bytes().unwrap();
    let mut graph = Graph::new();
    let a = graph.input(base.clone());
    let b = graph.input(f32_tensor([8, 1, 1, 1]));
    let id = graph.input(Tensor::from_i32([1, 1, 1, 1], &[0]).unwrap());
    let node = graph.op(Op::SetRows, &[a, b, id]).unwrap();
    assert_eq!(
        graph
            .evaluate_with_cancel(node, &AtomicBool::new(true))
            .unwrap_err(),
        TensorError::Cancelled
    );
    assert_eq!(base.to_bytes().unwrap(), before);
}
