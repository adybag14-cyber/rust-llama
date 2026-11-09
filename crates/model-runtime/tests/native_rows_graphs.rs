use llama_model_runtime::math::native_rows::{silu_f32_in_place, softmax_exp_sum_f32_in_place};
struct Reader {
    raw: &'static [u8],
    at: usize,
}
impl Reader {
    fn take(&mut self, n: usize) -> &'static [u8] {
        let s = &self.raw[self.at..self.at + n];
        self.at += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> usize {
        u64::from_le_bytes(self.take(8).try_into().unwrap()) as usize
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn string(&mut self) -> &'static str {
        let n = self.u32() as usize;
        std::str::from_utf8(self.take(n)).unwrap()
    }
}
#[test]
#[cfg(any(
    target_env = "msvc",
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
fn actual_original_graphs_preserve_row_boundaries_scaling_and_platform_policy() {
    #[cfg(target_env = "msvc")]
    let bytes = include_bytes!("fixtures/native-rows/row-math-msvc-reference.bin");
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    let bytes = include_bytes!("fixtures/native-rows/row-math-gnu-reference.bin");
    let mut r = Reader { raw: bytes, at: 0 };
    assert_eq!(r.take(8), b"GGTR0002");
    let count = r.u32();
    let mut compared = 0;
    let mut failures = Vec::new();
    for _ in 0..count {
        let name = r.string();
        let operation = r.string();
        assert_eq!(r.u32(), 0);
        assert_eq!(r.u32(), 0);
        assert_eq!(r.u32(), 1);
        let scale = r.f32();
        assert_eq!(r.u32(), 1);
        assert_eq!(r.u32(), 0);
        let shape: [usize; 4] = std::array::from_fn(|_| r.u64());
        let strides: [usize; 4] = std::array::from_fn(|_| r.u64());
        let length = r.u64();
        let raw = r.take(length);
        assert_eq!(r.u32(), 0);
        let output_shape: [usize; 4] = std::array::from_fn(|_| r.u64());
        assert_eq!(shape, output_shape);
        let length = r.u64();
        let expected = r.take(length);
        let mut output = Vec::new();
        for i3 in 0..shape[3] {
            for i2 in 0..shape[2] {
                for i1 in 0..shape[1] {
                    let start = i3 * strides[3] + i2 * strides[2] + i1 * strides[1];
                    let mut row = (0..shape[0])
                        .map(|i| {
                            let p = start + i * strides[0];
                            f32::from_le_bytes(raw[p..p + 4].try_into().unwrap())
                        })
                        .collect::<Vec<_>>();
                    if operation == "silu" {
                        silu_f32_in_place(&mut row);
                    } else {
                        assert_eq!(operation, "softmax");
                        for value in &mut row {
                            *value *= scale;
                        }
                        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let sum = softmax_exp_sum_f32_in_place(&mut row, maximum);
                        let reciprocal = (1.0 / sum) as f32;
                        for value in &mut row {
                            *value *= reciprocal;
                        }
                    }
                    for value in row {
                        output.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        compared += output.len() / 4;
        if output != expected {
            let first = output
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap()
                / 4
                * 4;
            failures.push(format!(
                "{name}: got {:08x}, expected {:08x}",
                u32::from_le_bytes(output[first..first + 4].try_into().unwrap()),
                u32::from_le_bytes(expected[first..first + 4].try_into().unwrap())
            ));
        }
    }
    assert_eq!(r.at, bytes.len());
    println!(
        "original graph cases={count}, exact F32 output checks={compared}, failures={}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{}\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
