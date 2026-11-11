//! Independently captured original GGML windows, relative positions and losses.
#![cfg(any(
    all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
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
fn compare(bytes: &'static [u8]) -> (u32, usize, std::collections::BTreeMap<&'static str, usize>) {
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGVL0001");
    let count = r.u32();
    let mut failures = Vec::new();
    let mut compared = 0;
    let mut counts = std::collections::BTreeMap::new();
    for _ in 0..count {
        let name = r.string();
        let op = r.string();
        let threads = r.u32();
        assert!(matches!(threads, 1 | 3));
        let n = r.u32();
        let args = (0..n).map(|_| r.wide()).collect::<Vec<_>>();
        let n = r.u32();
        let inputs = (0..n).map(|_| r.tensor()).collect::<Vec<_>>();
        let expected = r.tensor();
        let operation = match op {
            "win_part" => Op::WindowPartition { window: args[0] },
            "win_unpart" => Op::WindowUnpartition {
                width: args[0],
                height: args[1],
                window: args[2],
            },
            "get_rel_pos" => Op::GetRelativePosition {
                query: args[0],
                key: args[1],
            },
            "add_rel_pos" => Op::AddRelativePosition,
            "solve_tri" => Op::SolveTriangular,
            "cross_entropy_loss" => Op::CrossEntropyLoss,
            "cross_entropy_loss_back" => Op::CrossEntropyLossBack,
            _ => panic!("unknown native operation {op}"),
        };
        *counts.entry(op).or_insert(0usize) += 1;
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
        let target = graph.op(operation, &ids).unwrap();
        let actual = match graph.evaluate(target) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        assert_eq!(actual.desc(), expected.desc(), "{name}: descriptor");
        let actual_bytes = actual.to_bytes().unwrap();
        let expected_bytes = expected.to_bytes().unwrap();
        compared += expected_bytes.len();
        if actual_bytes != expected_bytes {
            let differs = actual_bytes
                .iter()
                .zip(&expected_bytes)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(12)
                .map(|(at, (a, b))| format!("byte {at}: {a:02x} != {b:02x}"))
                .collect::<Vec<_>>();
            failures.push(format!("{name}: {}", differs.join(", ")));
        }
        for (input, bytes) in inputs.iter().zip(before) {
            assert_eq!(input.to_bytes().unwrap(), bytes, "{name}: mutated input");
            assert!(
                !actual.shares_storage(input),
                "{name}: output aliases input"
            );
        }
    }
    assert_eq!(r.at, bytes.len());
    println!(
        "original graphs={count}, exact output bytes={compared}, mismatches={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    (count, compared, counts)
}
#[test]
fn windows_relative_positions_solve_and_losses_match_every_native_output_byte() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/vision-loss/msvc/vision-loss-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/vision-loss/gnu/vision-loss-reference.bin");
    let (count, compared, counts) = compare(bytes);
    assert_eq!(count, 920);
    assert_eq!(compared, 1_135_798);
    for (op, n) in [
        ("win_part", 151),
        ("win_unpart", 152),
        ("get_rel_pos", 83),
        ("add_rel_pos", 65),
        ("solve_tri", 73),
        ("cross_entropy_loss", 201),
        ("cross_entropy_loss_back", 195),
    ] {
        assert_eq!(counts[op], n);
    }
}

#[test]
fn relative_addition_and_triangular_solve_preserve_native_raw_ieee_words() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/vision-loss/msvc/ieee/vision-loss-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/vision-loss/gnu/ieee/vision-loss-reference.bin");
    let (count, compared, counts) = compare(bytes);
    assert_eq!(count, 111);
    assert_eq!(compared, 25_992);
    assert_eq!(counts["add_rel_pos"], 48);
    assert_eq!(counts["solve_tri"], 63);
}

#[test]
fn competing_nan_operand_policies_cover_variable_geometry() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/vision-loss/msvc/policy/vision-loss-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/vision-loss/gnu/policy/vision-loss-reference.bin");
    let (count, compared, counts) = compare(bytes);
    assert_eq!(count, 222);
    assert_eq!(compared, 83_424);
    assert_eq!(counts["add_rel_pos"], 96);
    assert_eq!(counts["solve_tri"], 126);
}

#[test]
fn one_key_query_width_specialization_preserves_native_nan_payloads() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/vision-loss/msvc/width/vision-loss-reference.bin");
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/vision-loss/gnu/width/vision-loss-reference.bin");
    let (graphs, bytes, counts) = compare(bytes);
    assert_eq!(graphs, 168);
    assert_eq!(bytes, 26_208);
    assert_eq!(counts["add_rel_pos"], 168);
}
