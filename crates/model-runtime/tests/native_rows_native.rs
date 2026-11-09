use llama_model_runtime::math::native_rows::sse2_exp_f32x4;
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
use llama_model_runtime::math::native_rows::{
    MathPolicy, silu_f32_with_policy, softmax_exp_sum_f32_with_policy,
};
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
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
fn independent_original_sse2_routines_match_every_output_bit() {
    let bytes = include_bytes!("fixtures/native-rows/gnu-sse2-math-reference.bin");
    let mut r = Reader { raw: bytes, at: 0 };
    assert_eq!(r.take(8), b"GGSSE201");
    let count = r.u32();
    let mut failures = Vec::new();
    let mut outputs = 0;
    let mut non_nan_differences = 0;
    let mut nan_differences = 0;
    let mut sum_differences = 0;
    for _ in 0..count {
        let name = r.string();
        let kind = r.u32();
        let n = r.u32() as usize;
        let maximum = r.f32();
        let sum = r.f64();
        let mut input = (0..n).map(|_| r.f32()).collect::<Vec<_>>();
        let expected = (0..n).map(|_| r.f32()).collect::<Vec<_>>();
        let actual_sum = match kind {
            0 => {
                for row in input.as_chunks_mut::<4>().0 {
                    *row = sse2_exp_f32x4(*row);
                }
                0.0
            }
            1 => {
                silu_f32_with_policy(&mut input, MathPolicy::Sse2);
                0.0
            }
            2 => softmax_exp_sum_f32_with_policy(&mut input, maximum, MathPolicy::Sse2),
            _ => panic!("bad kind"),
        };
        outputs += n;
        let mut first = None;
        for (i, (&a, &e)) in input.iter().zip(&expected).enumerate() {
            if a.to_bits() != e.to_bits() {
                if a.is_nan() && e.is_nan() {
                    nan_differences += 1;
                } else {
                    non_nan_differences += 1;
                }
                first.get_or_insert(format!(
                    "{name}[{i}] got {:08x} expected {:08x}",
                    a.to_bits(),
                    e.to_bits()
                ));
            }
        }
        if let Some(value) = first {
            failures.push(value);
        }
        if actual_sum.to_bits() != sum.to_bits() {
            sum_differences += 1;
            failures.push(format!(
                "{name} sum got {:016x} expected {:016x}",
                actual_sum.to_bits(),
                sum.to_bits()
            ));
        }
    }
    assert_eq!(r.at, bytes.len());
    println!(
        "cases={count}, outputs={outputs}, non-NaN differences={non_nan_differences}, NaN bit differences={nan_differences}, F64 sum differences={sum_differences}"
    );
    assert!(
        failures.is_empty(),
        "{}\n{}",
        failures.len(),
        failures.into_iter().take(30).collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn explicit_sse2_prefix_matches_original_bits_on_either_host() {
    let bytes = include_bytes!("fixtures/native-rows/gnu-sse2-math-reference.bin");
    let mut r = Reader { raw: bytes, at: 0 };
    assert_eq!(r.take(8), b"GGSSE201");
    let cases = r.u32();
    let mut compared = 0;
    for _ in 0..cases {
        let name = r.string();
        let kind = r.u32();
        let n = r.u32() as usize;
        let _maximum = r.f32();
        let _sum = r.f64();
        let input = (0..n).map(|_| r.f32()).collect::<Vec<_>>();
        let expected = (0..n).map(|_| r.f32()).collect::<Vec<_>>();
        if kind != 0 {
            continue;
        }
        for (block, expected) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected.as_chunks::<4>().0)
        {
            assert_eq!(
                sse2_exp_f32x4(*block).map(f32::to_bits),
                expected.map(f32::to_bits),
                "{name}"
            );
            compared += 4;
        }
    }
    assert_eq!(r.at, bytes.len());
    println!("portable SSE2 polynomial exact outputs={compared}");
}
