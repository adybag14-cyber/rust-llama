//! Independent original forward rotary outputs, not inverse-derived expectations.
#![cfg(any(
    all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
use llama_tensor::{
    DType, Graph, Op, RopeExtMode, RopeExtOptions, Tensor, TensorDesc, TensorError,
};
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
const MAIN: &[u8] = include_bytes!("fixtures/rope-forward/msvc/rope-forward-reference.bin");
#[cfg(target_os = "linux")]
const MAIN: &[u8] = include_bytes!("fixtures/rope-forward/gnu/rope-forward-reference.bin");
#[cfg(target_os = "windows")]
const EDGES: &[u8] = include_bytes!("fixtures/rope-forward/msvc/edges/rope-forward-reference.bin");
#[cfg(target_os = "linux")]
const EDGES: &[u8] = include_bytes!("fixtures/rope-forward/gnu/edges/rope-forward-reference.bin");

fn compare(bytes: &'static [u8]) -> (u32, usize, usize, usize, usize) {
    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(reader.take(8), b"GGRF0001");
    let count = reader.u32();
    let mut exact_graphs = 0;
    let mut exact_bytes = 0;
    let mut excluded_graphs = 0;
    let mut retained_bytes = 0;
    let mut failures = Vec::new();
    let mut modes = std::collections::BTreeSet::new();
    for _ in 0..count {
        let name = reader.string();
        assert_eq!(reader.string(), "rope_forward");
        assert!(matches!(reader.u32(), 1 | 3));
        assert_eq!(reader.u32(), 16);
        let parameters: [usize; 16] = std::array::from_fn(|_| reader.wide());
        modes.insert(parameters[1]);
        let f = |i: usize| f32::from_bits(parameters[i].try_into().unwrap());
        let mode = match parameters[1] {
            0 => RopeExtMode::Normal,
            2 => RopeExtMode::NeoX,
            8 => RopeExtMode::Multi,
            24 => RopeExtMode::Vision,
            40 => RopeExtMode::Interleaved,
            _ => panic!("unknown source mode"),
        };
        let options = RopeExtOptions {
            dimensions: parameters[0],
            mode,
            context_original: parameters[2],
            frequency_base: f(3),
            frequency_scale: f(4),
            extension_factor: f(5),
            attention_factor: f(6),
            beta_fast: f(7),
            beta_slow: f(8),
            sections: std::array::from_fn(|i| parameters[9 + i]),
            offset: parameters[13],
            truncate: parameters[14] != 0,
        };
        let input_count = reader.u32();
        assert!(matches!(input_count, 2 | 3));
        let inputs = (0..input_count)
            .map(|_| reader.tensor())
            .collect::<Vec<_>>();
        let expected = reader.tensor();
        let expected_bytes = expected.to_bytes().unwrap();
        let before = inputs
            .iter()
            .map(|v| v.to_bytes().unwrap())
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let ids = inputs
            .iter()
            .cloned()
            .map(|v| graph.input(v))
            .collect::<Vec<_>>();
        let target = graph
            .op(
                Op::RopeExt {
                    options,
                    with_factors: input_count == 3,
                },
                &ids,
            )
            .unwrap();
        let result = graph.evaluate(target);
        if parameters[15] != 0 {
            // The sealed original capture marks successful native graphs at
            // the declared finite-input/parameter/product boundary.
            excluded_graphs += 1;
            retained_bytes += expected_bytes.len();
            if !matches!(result, Err(TensorError::Unsupported(_))) {
                failures.push(format!(
                    "{name}: expected explicit unsupported, got {result:?}"
                ));
            }
        } else {
            match result {
                Err(error) => failures.push(format!("{name}: {error}")),
                Ok(actual) => {
                    assert!(
                        inputs.iter().all(|v| !actual.shares_storage(v)),
                        "{name}: aliased output"
                    );
                    if actual.desc() != expected.desc() {
                        failures.push(format!("{name}: output descriptor mismatch"));
                    } else {
                        exact_graphs += 1;
                        exact_bytes += expected_bytes.len();
                        let actual_bytes = actual.to_bytes().unwrap();
                        if actual_bytes != expected_bytes {
                            let differing = actual_bytes
                                .iter()
                                .zip(&expected_bytes)
                                .enumerate()
                                .filter(|(_, (a, b))| a != b)
                                .take(10)
                                .map(|(i, (a, b))| format!("byte {i}: {a:02x} != {b:02x}"))
                                .collect::<Vec<_>>();
                            failures.push(format!("{name}: {}", differing.join(", ")));
                        }
                    }
                }
            }
        }
        for (input, original) in inputs.iter().zip(before) {
            assert_eq!(input.to_bytes().unwrap(), original, "{name}: input mutated");
        }
    }
    assert_eq!(reader.at, bytes.len());
    assert_eq!(modes.len(), 5);
    eprintln!(
        "RoPE forward original graphs={count}, exact graphs={exact_graphs}, exact bytes={exact_bytes}, native-success exclusions={excluded_graphs}, retained bytes={retained_bytes}, failures={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    (
        count,
        exact_graphs,
        exact_bytes,
        excluded_graphs,
        retained_bytes,
    )
}
#[test]
fn original_forward_normal_neox_yarn_and_multi_axis_words_match() {
    assert_eq!(compare(MAIN), (780, 780, 1_031_360, 0, 0));
}
#[test]
fn native_empty_copied_ieee_and_explicit_finite_boundaries_are_retained() {
    assert_eq!(compare(EDGES), (112, 69, 133_380, 43, 5_216));
}
