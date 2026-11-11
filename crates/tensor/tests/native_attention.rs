//! Independent original GATED_DELTA_NET and DSV4 hyper-connection graphs.
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
fn compare(bytes: &'static [u8]) -> (u32, usize, usize, usize) {
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGHC0001");
    let count = r.u32();
    let mut failures = Vec::new();
    let mut compared = 0;
    let mut rejected = 0;
    let mut rejected_bytes = 0;
    for _ in 0..count {
        let name = r.string();
        let tag = r.string();
        assert!(matches!(r.u32(), 1 | 3));
        let nparams = r.u32();
        let params = (0..nparams).map(|_| r.wide()).collect::<Vec<_>>();
        let ninputs = r.u32();
        let inputs = (0..ninputs).map(|_| r.tensor()).collect::<Vec<_>>();
        let expected = r.tensor();
        let op = match tag {
            "gated_delta_net" => Op::GatedDeltaNet {
                snapshots: params[0],
            },
            "dsv4_hc_comb" => Op::Dsv4HcComb {
                epsilon: f32::from_bits(params[0] as u32),
                iterations: params[1],
            },
            "dsv4_hc_pre" => Op::Dsv4HcPre,
            "dsv4_hc_pre_gated" => Op::Dsv4HcPreGated {
                scale: f32::from_bits(params[0] as u32),
            },
            "dsv4_hc_post" => Op::Dsv4HcPost {
                with_comb: params[0] != 0,
            },
            _ => panic!("unknown native operation {tag}"),
        };
        let before = inputs
            .iter()
            .map(|t| t.to_bytes().unwrap())
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let ids = inputs
            .iter()
            .cloned()
            .map(|t| graph.input(t))
            .collect::<Vec<_>>();
        let target = graph.op(op, &ids).unwrap();
        let result = graph.evaluate(target);
        for (tensor, old) in inputs.iter().zip(before) {
            assert_eq!(tensor.to_bytes().unwrap(), old, "{name}: mutated input");
        }
        if name.starts_with("unsupported_") {
            rejected += 1;
            rejected_bytes += expected.to_bytes().unwrap().len();
            if !matches!(result, Err(TensorError::Unsupported(_))) {
                failures.push(format!(
                    "{name}: expected explicit source-success boundary, got {result:?}"
                ));
            }
            continue;
        }
        let actual = match result {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!("{name}: {error}"));
                continue;
            }
        };
        assert!(
            inputs.iter().all(|input| !actual.shares_storage(input)),
            "{name}: aliased output"
        );
        if actual.desc() != expected.desc() {
            failures.push(format!(
                "{name}: descriptor {:?} != {:?}",
                actual.desc(),
                expected.desc()
            ));
            continue;
        }
        let actual = actual.to_bytes().unwrap();
        let expected = expected.to_bytes().unwrap();
        compared += expected.len();
        if actual != expected {
            let differences = actual
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected.as_chunks::<4>().0.iter())
                .enumerate()
                .filter(|(_, (a, e))| a != e)
                .take(8)
                .map(|(i, (a, e))| {
                    format!(
                        "word {i}: {:08x} != {:08x}",
                        u32::from_le_bytes(*a),
                        u32::from_le_bytes(*e)
                    )
                })
                .collect::<Vec<_>>();
            failures.push(format!("{name}: {}", differences.join(", ")));
        }
    }
    assert_eq!(r.at, bytes.len());
    eprintln!(
        "original graphs={count}, exact output bytes={compared}, explicit source-success rejections={rejected}, retained rejected bytes={rejected_bytes}, mismatches={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    (count, compared, rejected, rejected_bytes)
}
#[test]
fn gated_delta_and_hyperconnections_match_native_graph_bytes() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/gated-delta-hyperconnection/msvc/attention-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/gated-delta-hyperconnection/gnu/attention-reference.bin");
    assert_eq!(compare(bytes), (867, 5_185_540, 0, 0));
}
#[test]
fn native_geometry_and_nonfinite_domains_are_retained() {
    #[cfg(target_os = "windows")]
    let bytes =
        include_bytes!("fixtures/gated-delta-hyperconnection/msvc/edges/attention-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes =
        include_bytes!("fixtures/gated-delta-hyperconnection/gnu/edges/attention-reference.bin");
    let (cases, bytes, rejected, rejected_bytes) = compare(bytes);
    assert_eq!(cases, 89);
    assert_eq!(rejected, 73);
    assert_eq!(bytes + rejected_bytes, 40_244);
}
