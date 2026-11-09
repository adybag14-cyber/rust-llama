use llama_model_runtime::math::native_rows::{
    MathPolicy, center_f32_and_variance, center_f32_and_variance_with_policy,
};
struct Reader {
    bytes: &'static [u8],
    at: usize,
}
impl Reader {
    fn take(&mut self, n: usize) -> &'static [u8] {
        let s = &self.bytes[self.at..self.at + n];
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
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn string(&mut self) -> &'static str {
        let n = self.u32() as usize;
        std::str::from_utf8(self.take(n)).unwrap()
    }
}
#[test]
fn original_scalar_and_sse2_centered_variance_match_all_primitive_bits() {
    let captures: [(MathPolicy, &[u8]); 2] = [
        (
            MathPolicy::Scalar,
            include_bytes!("fixtures/native-rows/variance-msvc-reference.bin"),
        ),
        (
            MathPolicy::Sse2,
            include_bytes!("fixtures/native-rows/variance-gnu-reference.bin"),
        ),
    ];
    let mut failures = Vec::new();
    let mut outputs = 0;
    let mut variances = 0;
    let mut finite_failures = 0;
    for (policy, bytes) in captures {
        let mut r = Reader { bytes, at: 0 };
        assert_eq!(r.take(8), b"GGCVAR01");
        let count = r.u32();
        for _ in 0..count {
            let name = r.string();
            let n = r.u32() as usize;
            let mean = r.f32();
            let expected_variance = r.f64();
            let mut row = (0..n).map(|_| r.f32()).collect::<Vec<_>>();
            let expected = (0..n).map(|_| r.f32()).collect::<Vec<_>>();
            let actual = center_f32_and_variance_with_policy(&mut row, mean, policy);
            for (i, (&a, &e)) in row.iter().zip(&expected).enumerate() {
                outputs += 1;
                if a.to_bits() != e.to_bits() {
                    if !(a.is_nan() && e.is_nan()) {
                        finite_failures += 1;
                    }
                    failures.push(format!(
                        "{policy:?} {name} centered[{i}] got {:08x}, expected {:08x}",
                        a.to_bits(),
                        e.to_bits()
                    ));
                    break;
                }
            }
            variances += 1;
            if actual.to_bits() != expected_variance.to_bits() {
                if !(actual.is_nan() && expected_variance.is_nan()) {
                    finite_failures += 1;
                }
                failures.push(format!(
                    "{policy:?} {name} variance got {:016x}, expected {:016x}",
                    actual.to_bits(),
                    expected_variance.to_bits()
                ));
            }
        }
        assert_eq!(r.at, bytes.len());
        assert_eq!(count, 1393);
    }
    println!(
        "centered outputs checked={outputs}, F64 variances={variances}, finite mismatches={finite_failures}, all mismatches={}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{}",
        failures
            .iter()
            .take(80)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
#[test]
#[cfg(any(
    target_env = "msvc",
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
fn actual_original_norm_graphs_match_native_policy_and_rounding() {
    #[cfg(target_env = "msvc")]
    let bytes = include_bytes!("fixtures/native-rows/norm-msvc-reference.bin");
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    let bytes = include_bytes!("fixtures/native-rows/norm-gnu-reference.bin");
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGTR0002");
    let count = r.u32();
    let mut compared = 0;
    let mut failures = Vec::new();
    for _ in 0..count {
        let name = r.string();
        assert_eq!(r.string(), "norm");
        assert_eq!(r.u32(), 0);
        assert_eq!(r.u32(), 0);
        assert_eq!(r.u32(), 1);
        let eps = r.f32();
        assert_eq!(r.u32(), 1);
        assert_eq!(r.u32(), 0);
        let shape: [usize; 4] = std::array::from_fn(|_| r.u64());
        let strides: [usize; 4] = std::array::from_fn(|_| r.u64());
        let length = r.u64();
        let raw = r.take(length);
        assert_eq!(r.u32(), 0);
        let out_shape: [usize; 4] = std::array::from_fn(|_| r.u64());
        assert_eq!(shape, out_shape);
        let length = r.u64();
        let expected = r.take(length);
        let mut output = Vec::new();
        for i3 in 0..shape[3] {
            for i2 in 0..shape[2] {
                for i1 in 0..shape[1] {
                    let offset = i3 * strides[3] + i2 * strides[2] + i1 * strides[1];
                    let mut row = (0..shape[0])
                        .map(|i| {
                            let at = offset + i * strides[0];
                            f32::from_le_bytes(raw[at..at + 4].try_into().unwrap())
                        })
                        .collect::<Vec<_>>();
                    let sum = row.iter().map(|&x| x as f64).sum::<f64>() as f32;
                    let mean = sum / shape[0] as f32;
                    let variance = center_f32_and_variance(&mut row, mean) as f32;
                    let scale = 1.0 / (variance + eps).sqrt();
                    for value in row {
                        output.extend_from_slice(&(value * scale).to_le_bytes());
                    }
                }
            }
        }
        compared += output.len() / 4;
        if output != expected {
            let at = output
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap()
                / 4
                * 4;
            failures.push(format!(
                "{name}: got {:08x}, expected {:08x}",
                u32::from_le_bytes(output[at..at + 4].try_into().unwrap()),
                u32::from_le_bytes(expected[at..at + 4].try_into().unwrap())
            ));
        }
    }
    assert_eq!(r.at, bytes.len());
    assert_eq!(count, 312);
    println!(
        "actual NORM graphs={count}, F32 outputs={compared}, failures={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
