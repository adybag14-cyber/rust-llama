//! Independent original GGML extended softmax outputs, including sink/ALiBi modes.
#![cfg(any(
    all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
use llama_tensor::{DType, Graph, Op, Tensor, TensorDesc, TensorError};
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
    fn wide(&mut self) -> usize {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
            .try_into()
            .unwrap()
    }
    fn string(&mut self) -> &'static str {
        let n = self.u32() as usize;
        std::str::from_utf8(self.take(n)).unwrap()
    }
    fn tensor(&mut self) -> Tensor {
        let dtype = DType::from_id(self.u32()).unwrap();
        let ne = std::array::from_fn(|_| self.wide());
        let nb = std::array::from_fn(|_| self.wide());
        let n = self.wide();
        Tensor::from_bytes(
            TensorDesc::with_strides(dtype, ne, nb).unwrap(),
            self.take(n).to_vec(),
        )
        .unwrap()
    }
}
#[cfg(target_os = "windows")]
const MAIN: &[u8] = include_bytes!("fixtures/softmax/msvc/softmax-ext-reference.bin");
#[cfg(target_os = "linux")]
const MAIN: &[u8] = include_bytes!("fixtures/softmax/gnu/softmax-ext-reference.bin");
#[cfg(target_os = "windows")]
const EDGES: &[u8] = include_bytes!("fixtures/softmax/msvc/edges/softmax-ext-reference.bin");
#[cfg(target_os = "linux")]
const EDGES: &[u8] = include_bytes!("fixtures/softmax/gnu/edges/softmax-ext-reference.bin");
fn compare(bytes: &'static [u8]) -> (u32, usize, usize, usize, usize) {
    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(reader.take(8), b"GGSM0001");
    let count = reader.u32();
    let mut exact = 0;
    let mut compared = 0;
    let mut excluded = 0;
    let mut retained = 0;
    let mut failures = Vec::new();
    for _ in 0..count {
        let name = reader.string();
        let op = reader.string();
        assert!(matches!(reader.u32(), 1 | 3));
        assert_eq!(reader.u32(), 6);
        let p: [usize; 6] = std::array::from_fn(|_| reader.wide());
        let f = |i: usize| f32::from_bits(p[i].try_into().unwrap());
        let n = reader.u32();
        let inputs = (0..n).map(|_| reader.tensor()).collect::<Vec<_>>();
        let expected = reader.tensor();
        let operation = match op {
            "softmax_ext" => Op::SoftmaxExt {
                scale: f(0),
                max_bias: f(1),
                with_mask: p[2] != 0,
                with_sinks: p[3] != 0,
            },
            "softmax_back" => {
                assert_eq!(f(1), 0.0);
                Op::SoftmaxBack { scale: f(0) }
            }
            _ => panic!("unknown original operation"),
        };
        let original = inputs
            .iter()
            .map(|v| v.to_bytes().unwrap())
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let ids = inputs
            .iter()
            .cloned()
            .map(|v| graph.input(v))
            .collect::<Vec<_>>();
        let target = graph.op(operation, &ids).unwrap();
        let result = graph.evaluate(target);
        let wanted = expected.to_bytes().unwrap();
        if p[5] != 0 {
            excluded += 1;
            retained += wanted.len();
            if !matches!(result, Err(TensorError::Unsupported(_))) {
                failures.push(format!(
                    "{name}: expected explicit checked boundary; got {result:?}"
                ));
            }
        } else {
            match result {
                Err(e) => failures.push(format!("{name}: {e}")),
                Ok(actual) => {
                    assert_eq!(actual.desc(), expected.desc(), "{name}: descriptor");
                    let got = actual.to_bytes().unwrap();
                    exact += 1;
                    compared += wanted.len();
                    if got != wanted {
                        let differences = got
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .zip(wanted.as_chunks::<4>().0.iter())
                            .enumerate()
                            .filter(|(_, (a, b))| a != b)
                            .take(8)
                            .map(|(i, (a, b))| {
                                format!(
                                    "word {i}: {:08x} != {:08x}",
                                    u32::from_le_bytes(*a),
                                    u32::from_le_bytes(*b)
                                )
                            })
                            .collect::<Vec<_>>();
                        failures.push(format!("{name}: {}", differences.join(", ")));
                    }
                    assert!(
                        inputs.iter().all(|v| !actual.shares_storage(v)),
                        "{name}: output aliases input"
                    );
                }
            }
        }
        for (input, before) in inputs.iter().zip(original) {
            assert_eq!(input.to_bytes().unwrap(), before, "{name}: input mutation");
        }
    }
    assert_eq!(reader.at, bytes.len());
    eprintln!(
        "Original softmax graphs={count}, exact={exact}, compared={compared}, excluded={excluded}, retained={retained}, failures={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    (count, exact, compared, excluded, retained)
}
#[test]
fn masks_sinks_alibi_broadcast_and_zero_bias_backward_match_original_words() {
    assert_eq!(compare(MAIN), (2122, 2122, 7_279_720, 0, 0));
}
#[test]
fn empty_ieee_sink_dominance_and_checked_boundaries_match_source() {
    assert_eq!(compare(EDGES), (267, 252, 371_160, 15, 15_120));
}

#[test]
fn intermediate_head_and_batch_divisors_use_periodic_mask_broadcast() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/softmax/msvc/broadcast/softmax-ext-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/softmax/gnu/broadcast/softmax-ext-reference.bin");
    assert_eq!(compare(bytes), (80, 80, 2_365_920, 0, 0));
}
