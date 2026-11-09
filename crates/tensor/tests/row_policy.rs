#![cfg(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    ),
    not(target_feature = "crt-static")
))]

use llama_tensor::{DType, GluOp, Graph, Op, Tensor, TensorDesc, UnaryOp};

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

fn operation(tag: &str, ints: &[i64], floats: &[f32]) -> Op {
    match tag {
        "unary" => Op::Unary(UnaryOp::from_id(ints[0] as u32).unwrap()),
        "glu" => Op::Glu {
            kind: GluOp::from_id(ints[0] as u32).unwrap(),
            swap: ints[1] != 0,
        },
        "glu_packed" => Op::GluPacked {
            kind: GluOp::from_id(ints[0] as u32).unwrap(),
            swap: ints[1] != 0,
        },
        "norm" => Op::Norm { epsilon: floats[0] },
        "rms_norm" => Op::RmsNorm { epsilon: floats[0] },
        "softmax" => Op::Softmax { scale: floats[0] },
        _ => panic!("unexpected source operation {tag}"),
    }
}
fn compare_fixture(bytes: &'static [u8], expected_cases: u32, expected_bytes: usize) {
    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(reader.take(8), b"GGTR0002");
    let cases = reader.u32();
    let mut total_bytes = 0;
    let mut bad_cases = 0;
    let mut bad_values = 0;
    let mut finite_bad = 0;
    let mut details = Vec::new();
    for _ in 0..cases {
        let name = reader.string();
        let tag = reader.string();
        assert_eq!(reader.u32(), 0, "exact native-byte reference");
        let ni = reader.u32();
        let ints = (0..ni).map(|_| reader.i64()).collect::<Vec<_>>();
        let nf = reader.u32();
        let floats = (0..nf).map(|_| reader.f32()).collect::<Vec<_>>();
        let nt = reader.u32();
        let inputs = (0..nt).map(|_| reader.tensor(true)).collect::<Vec<_>>();
        let expected = reader.tensor(false);
        let result = match evaluate(operation(tag, &ints, &floats), inputs) {
            Ok(value) => value,
            Err(error) => {
                bad_cases += 1;
                details.push(format!("{name}: {error}"));
                continue;
            }
        };
        assert_eq!(result.dtype(), expected.dtype(), "{name}");
        assert_eq!(result.shape(), expected.shape(), "{name}");
        let dtype = result.dtype();
        let actual = result.to_bytes().unwrap();
        let expected = expected.to_bytes().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}");
        total_bytes += actual.len();
        let width = if dtype == DType::F32 { 4 } else { 2 };
        let mut case_bad = 0;
        let mut case_finite = 0;
        let mut first = None;
        for (index, (a, b)) in actual
            .chunks_exact(width)
            .zip(expected.chunks_exact(width))
            .enumerate()
        {
            if a == b {
                continue;
            }
            case_bad += 1;
            let decode = |raw: &[u8]| match dtype {
                DType::F32 => f32::from_le_bytes(raw.try_into().unwrap()),
                DType::F16 => llama_quant::f16_to_f32(u16::from_le_bytes(raw.try_into().unwrap())),
                DType::BF16 => {
                    llama_quant::bf16_to_f32(u16::from_le_bytes(raw.try_into().unwrap()))
                }
                _ => panic!("unexpected native output dtype"),
            };
            let av = decode(a);
            let bv = decode(b);
            if av.is_finite() && bv.is_finite() {
                case_finite += 1;
            }
            if first.is_none() {
                first = Some(format!(
                    "index {index}: Rust {a:02x?} ({av:?}), native {b:02x?} ({bv:?})"
                ));
            }
        }
        if case_bad > 0 {
            bad_cases += 1;
            bad_values += case_bad;
            finite_bad += case_finite;
            if details.len() < 80 {
                details.push(format!(
                    "{name}: {case_bad} differing values ({case_finite} finite); {}",
                    first.unwrap()
                ));
            }
        }
    }
    assert_eq!(reader.at, bytes.len());
    assert_eq!(cases, expected_cases);
    assert_eq!(total_bytes, expected_bytes);
    println!(
        "{cases} source-native tensor row cases; output bytes {total_bytes}; differing cases {bad_cases}; differing values {bad_values}; finite differing values {finite_bad}"
    );
    assert_eq!(bad_cases, 0, "{}", details.join("\n"));
}
#[test]
fn native_tensor_row_policies_and_system_erf_match_original_graph_bytes() {
    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
    compare_fixture(
        include_bytes!("fixtures/row-policy/msvc/tensor-row-policy-reference.bin"),
        935,
        2_894_848,
    );
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    compare_fixture(
        include_bytes!("fixtures/row-policy/gnu/tensor-row-policy-reference.bin"),
        935,
        2_894_848,
    );
}

#[test]
fn explicit_sse2_quick_prefix_matches_native_gnu_on_both_hosts() {
    use llama_quant::cpu_math::{MathPolicy, gelu_quick_f32_with_policy};
    let bytes = include_bytes!("fixtures/row-policy/gnu/tensor-row-policy-reference.bin");
    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(reader.take(8), b"GGTR0002");
    let count = reader.u32();
    let mut checked = 0;
    for _ in 0..count {
        let name = reader.string();
        let tag = reader.string();
        assert_eq!(reader.u32(), 0);
        let ni = reader.u32();
        let ints = (0..ni).map(|_| reader.i64()).collect::<Vec<_>>();
        let nf = reader.u32();
        for _ in 0..nf {
            reader.f32();
        }
        let nt = reader.u32();
        let inputs = (0..nt).map(|_| reader.tensor(true)).collect::<Vec<_>>();
        let expected = reader.tensor(false);
        if tag != "unary"
            || inputs[0].dtype() != DType::F32
            || ints[0] != UnaryOp::GELU_QUICK as i64
        {
            continue;
        }
        let mut values = inputs[0].to_f32().unwrap();
        let expected = expected.to_f32().unwrap();
        let width = inputs[0].shape()[0];
        let prefix = width / 4 * 4;
        for (row_index, (row, expected)) in values
            .chunks_exact_mut(width)
            .zip(expected.chunks_exact(width))
            .enumerate()
        {
            gelu_quick_f32_with_policy(row, MathPolicy::Sse2);
            for (column, (actual, expected)) in
                row[..prefix].iter().zip(&expected[..prefix]).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{name} row{row_index} column{column}"
                );
            }
            checked += prefix;
        }
    }
    assert_eq!(reader.at, bytes.len());
    assert!(checked > 1_000);
    println!("Portable SSE2 quick GELU prefix: {checked} exact GNU source output words");
    // Scalar tails call the current host CRT. Native whole-row graph fixtures
    // and the separate original-inline scalar oracle check that distinct gate.
}

#[test]
fn row_policy_errors_are_checked_and_input_storage_is_unchanged() {
    let original = Tensor::from_f32([2, 1, 1, 1], &[0.0, f32::NEG_INFINITY]).unwrap();
    let before = original.to_bytes().unwrap();
    let mut graph = Graph::new();
    let input = graph.input(original.clone());
    let target = graph.op(Op::Softmax { scale: 0.0 }, &[input]).unwrap();
    assert!(
        graph.evaluate(target).is_err(),
        "zero times infinity cannot be silently ignored"
    );
    assert_eq!(original.to_bytes().unwrap(), before);
    assert_eq!(graph.evaluate(input).unwrap().to_bytes().unwrap(), before);

    let odd = Tensor::from_f32([3, 1, 1, 1], &[1.0, 2.0, 3.0]).unwrap();
    let changed_tail = Tensor::from_f32([3, 1, 1, 1], &[1.0, 2.0, 999.0]).unwrap();
    let operation = Op::GluPacked {
        kind: GluOp::SWIGLU,
        swap: false,
    };
    let result = evaluate(operation.clone(), vec![odd]).unwrap();
    assert_eq!(result.shape(), [1, 1, 1, 1]);
    assert_eq!(
        result.to_bytes().unwrap(),
        evaluate(operation, vec![changed_tail])
            .unwrap()
            .to_bytes()
            .unwrap()
    );

    let empty = Tensor::from_f32([0, 3, 1, 1], &[]).unwrap();
    let result = evaluate(Op::Unary(UnaryOp::SILU), vec![empty.clone()]).unwrap();
    assert_eq!(result.shape(), [0, 3, 1, 1]);
    assert!(result.to_bytes().unwrap().is_empty());
    let result = evaluate(
        Op::GluPacked {
            kind: GluOp::GEGLU_QUICK,
            swap: true,
        },
        vec![empty],
    )
    .unwrap();
    assert_eq!(result.shape(), [0, 3, 1, 1]);
    assert!(result.to_bytes().unwrap().is_empty());
}

#[test]
fn odd_packed_glu_preserves_source_flooring_and_vector_row_boundaries() {
    #[cfg(target_os = "windows")]
    compare_fixture(
        include_bytes!("fixtures/row-policy/odd-glu/msvc/odd-glu-reference.bin"),
        72,
        14_208,
    );
    #[cfg(target_os = "linux")]
    compare_fixture(
        include_bytes!("fixtures/row-policy/odd-glu/gnu/odd-glu-reference.bin"),
        72,
        14_208,
    );
}
