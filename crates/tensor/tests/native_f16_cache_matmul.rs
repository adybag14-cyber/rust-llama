use llama_tensor::{DType, Graph, Op, Tensor, TensorDesc};

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

fn operation(tag: &str, _ints: &[i64], _floats: &[f32]) -> Op {
    assert_eq!(tag, "mul_mat");
    Op::MatMul
}
fn compare_fixture(bytes: &'static [u8]) {
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
    assert_eq!(cases, 125);
    assert_eq!(total_bytes, 980_616);
    println!(
        "{cases} source-native F16 cache matmul cases; output bytes {total_bytes}; differing cases {bad_cases}; differing values {bad_values}; finite differing values {finite_bad}"
    );
    assert_eq!(bad_cases, 0, "{}", details.join("\n"));
}
#[test]
fn native_cache_matmul_keeps_unrounded_f32_rhs_and_strided_f16_weights() {
    // Independently captured original MSVC and GNU files are byte-identical.
    // The two provenance records retain both captures without duplicating data.
    compare_fixture(include_bytes!(
        "fixtures/f16-cache/f16-cache-matmul-reference.bin"
    ));
}
