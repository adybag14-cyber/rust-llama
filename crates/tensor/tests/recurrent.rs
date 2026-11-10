//! Independent original-GGML graph captures, including complete recurrent state.
//! Expected data comes only from original native graphs and stays immutable.
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
    fn take(&mut self, count: usize) -> &'static [u8] {
        let out = &self.bytes[self.at..self.at + count];
        self.at += count;
        out
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> usize {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
            .try_into()
            .unwrap()
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn string(&mut self) -> &'static str {
        let count = self.u32() as usize;
        std::str::from_utf8(self.take(count)).unwrap()
    }
    fn tensor(&mut self) -> Tensor {
        let dtype = DType::from_id(self.u32()).unwrap();
        let shape = std::array::from_fn(|_| self.u64());
        let strides = std::array::from_fn(|_| self.u64());
        let length = self.u64();
        Tensor::from_bytes(
            TensorDesc::with_strides(dtype, shape, strides).unwrap(),
            self.take(length).to_vec(),
        )
        .unwrap()
    }
}

struct Counts {
    graphs: u32,
    compared_bytes: usize,
    checked_boundaries: Vec<&'static str>,
    by_operation: std::collections::BTreeMap<&'static str, u32>,
}
fn compare(bytes: &'static [u8]) -> Counts {
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGRC0001");
    let count = r.u32();
    let mut failures = Vec::new();
    let mut compared_bytes = 0;
    let mut checked_boundaries = Vec::new();
    let mut by_operation = std::collections::BTreeMap::new();
    for _ in 0..count {
        let name = r.string();
        let tag = r.string();
        let threads = r.u32();
        assert!(threads == 1 || threads == 3);
        let snapshots = r.u32() as usize;
        let scale = r.f32();
        let input_count = r.u32();
        let inputs = (0..input_count).map(|_| r.tensor()).collect::<Vec<_>>();
        let expected = r.tensor();
        let op = match tag {
            "ssm_conv" => Op::SsmConv,
            "ssm_scan" => Op::SsmScan { snapshots },
            "rwkv_wkv6" => Op::RwkvWkv6,
            "gated_linear_attn" => Op::GatedLinearAttn { scale },
            "rwkv_wkv7" => Op::RwkvWkv7,
            _ => panic!("unknown reference op {tag}"),
        };
        *by_operation.entry(tag).or_insert(0_u32) += 1;
        let original = inputs
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
        let actual = match graph.evaluate(target) {
            Ok(value) => value,
            Err(TensorError::Unsupported(
                "recurrent arithmetic with competing NaN payloads is unsupported",
            )) if name.starts_with("nan_payloads_") || name.starts_with("nan_infinity_") => {
                checked_boundaries.push(name);
                for (input, before) in inputs.iter().zip(original) {
                    assert_eq!(
                        input.to_bytes().unwrap(),
                        before,
                        "{name}: rejected execution mutated input"
                    );
                }
                continue;
            }
            Err(error) => {
                failures.push(format!("{name}: {error}"));
                continue;
            }
        };
        assert_eq!(
            actual.desc(),
            expected.desc(),
            "{name}: dtype/shape/strides"
        );
        let expected_bytes = expected.to_bytes().unwrap();
        let actual_bytes = actual.to_bytes().unwrap();
        compared_bytes += expected_bytes.len();
        if actual_bytes != expected_bytes {
            let differing = actual_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected_bytes.as_chunks::<4>().0)
                .enumerate()
                .filter(|(_, (got, want))| got != want)
                .map(|(i, (got, want))| {
                    format!(
                        "word {i}: {:08x} != {:08x}",
                        u32::from_le_bytes(*got),
                        u32::from_le_bytes(*want)
                    )
                })
                .take(12)
                .collect::<Vec<_>>();
            failures.push(format!("{name}: {}", differing.join(", ")));
        }
        for (input, before) in inputs.iter().zip(original) {
            assert_eq!(
                input.to_bytes().unwrap(),
                before,
                "{name}: input was mutated"
            );
            assert!(
                !actual.shares_storage(input),
                "{name}: output aliases input"
            );
        }
    }

    assert_eq!(r.at, bytes.len());
    println!(
        "original recurrent graphs={count}, exact output/state bytes={compared_bytes}, checked NaN boundaries={checked_boundaries:?}, mismatches={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Counts {
        graphs: count,
        compared_bytes,
        checked_boundaries,
        by_operation,
    }
}
#[test]
fn recurrent_graphs_match_frozen_native_shapes_strides_and_all_output_state_words() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/recurrent/msvc/recurrent-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/recurrent/gnu/recurrent-reference.bin");
    let counts = compare(bytes);
    assert_eq!(counts.graphs, 944);
    assert_eq!(counts.compared_bytes, 1_145_804);
    assert_eq!(
        counts.checked_boundaries,
        [
            "nan_payloads_ssm_conv",
            "nan_payloads_ssm_scan",
            "nan_payloads_rwkv_wkv6",
            "nan_payloads_gated_linear_attn",
            "nan_payloads_rwkv_wkv7",
        ]
    );
    for (name, expected) in [
        ("ssm_conv", 157),
        ("ssm_scan", 424),
        ("rwkv_wkv6", 121),
        ("gated_linear_attn", 108),
        ("rwkv_wkv7", 134),
    ] {
        assert_eq!(counts.by_operation[name], expected);
    }
}
#[test]
fn recurrent_raw_ieee_and_signaling_nan_controls_match_native_exactly() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/recurrent/msvc/recurrent-ieee-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/recurrent/gnu/recurrent-ieee-reference.bin");
    let counts = compare(bytes);
    assert_eq!(counts.graphs, 450);
    assert_eq!(counts.compared_bytes, 132_336);
    assert!(counts.checked_boundaries.is_empty());
    for (name, expected) in [
        ("ssm_conv", 42),
        ("ssm_scan", 102),
        ("rwkv_wkv6", 102),
        ("gated_linear_attn", 87),
        ("rwkv_wkv7", 117),
    ] {
        assert_eq!(counts.by_operation[name], expected);
    }
}
