//! Independent original Lightning Indexer graphs and zero-mask score controls.
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
fn run(inputs: &[Tensor]) -> Result<Tensor, TensorError> {
    let mut graph = Graph::new();
    let nodes = inputs
        .iter()
        .cloned()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(Op::LightningIndexer, &nodes)?;
    graph.evaluate(target)
}
fn compare(bytes: &'static [u8]) -> (u32, usize, usize, usize, usize) {
    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(reader.take(8), b"GGLI0001");
    let count = reader.u32();
    let mut exact_graphs = 0;
    let mut exact_bytes = 0;
    let mut rejected_graphs = 0;
    let mut retained_bytes = 0;
    let mut failures = Vec::new();
    let mut types = std::collections::BTreeSet::new();
    for _ in 0..count {
        let name = reader.string();
        assert_eq!(reader.string(), "lightning_indexer");
        assert!(matches!(reader.u32(), 1 | 3));
        assert_eq!(reader.u32(), 0);
        assert_eq!(reader.u32(), 4);
        let inputs = (0..4).map(|_| reader.tensor()).collect::<Vec<_>>();
        types.insert(inputs[1].dtype() as u32);
        let original_score = reader.tensor();
        let original_masked = reader.tensor();
        assert_eq!(original_score.desc(), original_masked.desc());
        // This admission classification comes from a second original native
        // graph stored in the fixture, never from a Rust-generated answer.
        let unsupported = original_score
            .to_f32()
            .unwrap()
            .iter()
            .any(|v| !v.is_finite());
        let before = inputs
            .iter()
            .map(|t| t.to_bytes().unwrap())
            .collect::<Vec<_>>();
        for (masked, expected) in [(false, original_score), (true, original_masked)] {
            let mut args = inputs.clone();
            if !masked {
                let desc = *args[3].desc();
                args[3] = Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap();
            }
            let result = run(&args);
            let expected_bytes = expected.to_bytes().unwrap();
            if unsupported {
                rejected_graphs += 1;
                retained_bytes += expected_bytes.len();
                if !matches!(result, Err(TensorError::Unsupported(_))) {
                    failures.push(format!(
                        "{name} masked={masked}: expected source-success boundary, got {result:?}"
                    ));
                }
                continue;
            }
            let actual = match result {
                Ok(value) => value,
                Err(error) => {
                    failures.push(format!("{name} masked={masked}: {error}"));
                    continue;
                }
            };
            assert!(
                args.iter().all(|a| !actual.shares_storage(a)),
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
            exact_graphs += 1;
            exact_bytes += expected_bytes.len();
            let actual_bytes = actual.to_bytes().unwrap();
            if actual_bytes != expected_bytes {
                let diffs = actual_bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected_bytes.as_chunks::<4>().0)
                    .enumerate()
                    .filter(|(_, (a, e))| a != e)
                    .take(6)
                    .map(|(i, (a, e))| {
                        format!(
                            "word {i}: {:08x} != {:08x}",
                            u32::from_le_bytes(*a),
                            u32::from_le_bytes(*e)
                        )
                    })
                    .collect::<Vec<_>>();
                failures.push(format!("{name} masked={masked}: {}", diffs.join(", ")));
            }
        }
        for (input, expected) in inputs.iter().zip(before) {
            assert_eq!(input.to_bytes().unwrap(), expected, "{name}: mutated input");
        }
    }
    assert_eq!(reader.at, bytes.len());
    if count == 652 {
        assert_eq!(types.len(), 29);
    }
    eprintln!(
        "lightning cases={count}, exact native graphs={exact_graphs}, exact bytes={exact_bytes}, rejected native-success graphs={rejected_graphs}, retained bytes={retained_bytes}, failures={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    (
        count,
        exact_graphs,
        exact_bytes,
        rejected_graphs,
        retained_bytes,
    )
}
#[test]
fn every_native_key_decoder_and_all_half_mask_words_match() {
    assert_eq!(
        compare(include_bytes!(
            "fixtures/lightning-indexer/lightning-reference.bin"
        )),
        (652, 1304, 2_802_864, 0, 0)
    );
}
#[test]
fn native_empty_ieee_and_nonfinite_score_domains_are_preserved() {
    assert_eq!(
        compare(include_bytes!(
            "fixtures/lightning-indexer/edges/lightning-reference.bin"
        )),
        (51, 78, 2784, 24, 1152)
    );
}
