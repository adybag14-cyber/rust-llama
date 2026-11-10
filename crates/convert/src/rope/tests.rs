use super::*;

#[test]
fn parameter_errors_are_transactional() {
    let mut output = [0x5a_u8; 256];
    for dimension in [-1, i64::MIN] {
        let p = Parameters::new(dimension);
        assert_eq!(encoded_len(&p), Err(RopeError::NegativeDimension));
        assert_eq!(
            generate_into(&p, &mut output),
            Err(RopeError::NegativeDimension)
        );
        assert_eq!(output, [0x5a; 256]);
    }
    for dimension in [0, 1, 128] {
        for zero in [0.0, -0.0] {
            let mut p = Parameters::new(dimension);
            p.low_frequency_factor = zero;
            let error = RopeError::ZeroFrequencyFactor("low_frequency_factor");
            assert_eq!(generate_into(&p, &mut output), Err(error));
            p.low_frequency_factor = 1.0;
            p.high_frequency_factor = zero;
            let error = RopeError::ZeroFrequencyFactor("high_frequency_factor");
            assert_eq!(generate_into(&p, &mut output), Err(error));
            assert_eq!(output, [0x5a; 256]);
        }
    }
    for field in 0..5 {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut p = Parameters::new(128);
            match field {
                0 => p.theta = value,
                1 => p.factor = value,
                2 => p.low_frequency_factor = value,
                3 => p.high_frequency_factor = value,
                _ => p.original_context_length = value,
            }
            assert!(matches!(
                generate_into(&p, &mut output),
                Err(RopeError::NonFiniteParameter(_))
            ));
            assert_eq!(output, [0x5a; 256]);
        }
    }
}

#[test]
fn dimension_and_length_arithmetic_is_checked_without_allocation() {
    assert_eq!(encoded_len(&Parameters::new(0)), Ok(0));
    assert_eq!(encoded_len(&Parameters::new(1)), Ok(4));
    assert_eq!(encoded_len(&Parameters::new(2)), Ok(4));
    assert_eq!(encoded_len(&Parameters::new(3)), Ok(8));
    assert_eq!(
        encoded_len(&Parameters::new(i64::MAX)),
        Err(RopeError::SizeOverflow)
    );
    assert_eq!(
        encoded_len(&Parameters::new(i64::MAX / 2 + 1)),
        Err(RopeError::SizeOverflow)
    );
}

#[test]
fn unavailable_profile_is_an_explicit_atomic_error() {
    let p = Parameters::new(128);
    for profile in [
        SourceProfile::WindowsTorch211Numpy246,
        SourceProfile::LinuxTorch211Numpy226,
        SourceProfile::PortableFinite,
    ] {
        if profile != SourceProfile::current() || profile == SourceProfile::PortableFinite {
            assert_eq!(
                encoded_len_with_profile(&p, profile),
                Err(RopeError::UnsupportedProfile(profile)),
            );
            let mut output = [0x91_u8; 256];
            assert_eq!(
                generate_into_with_profile(&p, profile, &mut output),
                Err(RopeError::UnsupportedProfile(profile)),
            );
            assert_eq!(output, [0x91; 256]);
        }
    }
}

#[cfg(any(
    all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
    all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
))]
mod native {
    use super::*;

    struct Reader<'a> {
        bytes: &'a [u8],
        at: usize,
    }
    impl<'a> Reader<'a> {
        fn take(&mut self, count: usize) -> &'a [u8] {
            let end = self.at.checked_add(count).expect("fixture offset");
            let out = self.bytes.get(self.at..end).expect("complete fixture");
            self.at = end;
            out
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take(4).try_into().unwrap())
        }
        fn i64(&mut self) -> i64 {
            i64::from_le_bytes(self.take(8).try_into().unwrap())
        }
        fn f64(&mut self) -> f64 {
            f64::from_le_bytes(self.take(8).try_into().unwrap())
        }
    }

    fn compare(
        bytes: &[u8],
        expected_cases: u32,
        expected_rejections: u32,
        expected_output_bytes: usize,
    ) {
        let mut reader = Reader { bytes, at: 0 };
        assert_eq!(reader.take(8), b"HLR30001");
        let count = reader.u32();
        assert_eq!(count, expected_cases);
        let mut rejected = 0;
        let mut output_bytes = 0;
        let mut stage_words = 0;
        for _ in 0..count {
            let name_len = reader.u32() as usize;
            let p = Parameters {
                dimension: reader.i64(),
                theta: reader.f64(),
                factor: reader.f64(),
                low_frequency_factor: reader.f64(),
                high_frequency_factor: reader.f64(),
                original_context_length: reader.f64(),
            };
            let status = reader.u32();
            let n = reader.u32() as usize;
            let name = std::str::from_utf8(reader.take(name_len)).unwrap();
            if status != 0 {
                assert_eq!(status, 1);
                assert_eq!(n, 0);
                let error = encoded_len(&p).unwrap_err();
                assert!(
                    matches!(
                        error,
                        RopeError::NegativeDimension | RopeError::ZeroFrequencyFactor(_)
                    ),
                    "{name}: {error}"
                );
                let mut sentinel = [0x79; 8];
                assert_eq!(generate_into(&p, &mut sentinel), Err(error), "{name}");
                assert_eq!(sentinel, [0x79; 8]);
                rejected += 1;
                continue;
            }
            let stage_size = n.checked_mul(4).unwrap();
            let stages: [&[u8]; 7] = std::array::from_fn(|_| reader.take(stage_size));
            let branches = reader.take(n);
            // These are independent OS/double-pow counterexamples, not expected
            // implementations. Their original bytes remain in the fixture.
            reader.take(n.checked_mul(8).unwrap());
            assert_eq!(encoded_len(&p).unwrap(), stage_size, "{name}");
            let mut output = vec![0xa6_u8; stage_size + 16];
            generate_into(&p, &mut output[..stage_size]).unwrap();
            assert_eq!(&output[..stage_size], stages[6], "{name}: complete tensor");
            assert_eq!(&output[stage_size..], &[0xa6; 16], "{name}: guard");
            output_bytes += stage_size;
            let arithmetic = Arithmetic::new(&p, SourceProfile::current(), n);
            let vector_end = n / 32 * 32;
            for (index, &expected_branch) in branches.iter().enumerate() {
                let (arange, exponent, power) = arithmetic.power(index, vector_end);
                let frequency = 1.0_f32 / power;
                let (branch, wavelength, smooth, factor) = arithmetic.factor(frequency);
                assert_eq!(branch, expected_branch, "{name}: branch at {index}");
                let actual = [
                    arange, exponent, power, frequency, wavelength, smooth, factor,
                ];
                for (stage, (&value, expected)) in actual.iter().zip(stages).enumerate() {
                    if stage == 5 && branch != 2 {
                        continue; // the off-branch smooth value is a capture sentinel
                    }
                    let begin = index * 4;
                    let expected =
                        u32::from_le_bytes(expected[begin..begin + 4].try_into().unwrap());
                    assert_eq!(
                        value.to_bits(),
                        expected,
                        "{name}: stage {stage}, element {index}"
                    );
                    stage_words += 1;
                }
            }
        }
        assert_eq!(reader.at, bytes.len());
        assert_eq!(rejected, expected_rejections);
        assert_eq!(output_bytes, expected_output_bytes);
        println!(
            "{count} original cases; {rejected} original errors; {output_bytes} exact output bytes; {stage_words} exact diagnostic stage words"
        );
    }

    #[test]
    fn original_structured_torch_generator_and_all_seven_stages_are_exact() {
        #[cfg(target_os = "windows")]
        let bytes = include_bytes!("../../tests/fixtures/rope/windows-structured.bin");
        #[cfg(target_os = "linux")]
        let bytes = include_bytes!("../../tests/fixtures/rope/linux-structured.bin");
        compare(bytes, 579, 3, 90_880);
    }

    #[test]
    fn original_random_extreme_and_nan_generator_stages_are_exact() {
        #[cfg(target_os = "windows")]
        let bytes = include_bytes!("../../tests/fixtures/rope/windows-edge.bin");
        #[cfg(target_os = "linux")]
        let bytes = include_bytes!("../../tests/fixtures/rope/linux-edge.bin");
        compare(bytes, 1336, 28, 446_584);
    }

    #[test]
    fn short_long_and_partial_component_outputs_are_atomic_errors() {
        let p = Parameters::new(128);
        for length in [0, 1, 127, 128, 255, 257, 512] {
            let mut output = vec![0x3b; length];
            assert_eq!(
                generate_into(&p, &mut output),
                Err(RopeError::OutputLength {
                    expected: 256,
                    actual: length
                }),
            );
            assert!(output.iter().all(|&byte| byte == 0x3b));
        }
        let mut empty = [];
        generate_into(&Parameters::new(0), &mut empty).unwrap();
    }
    #[test]
    fn original_large_index_arange_and_power_windows_are_exact() {
        #[cfg(target_os = "windows")]
        let bytes = include_bytes!("../../tests/fixtures/rope/windows-index.bin");
        #[cfg(target_os = "linux")]
        let bytes = include_bytes!("../../tests/fixtures/rope/linux-index.bin");
        let mut reader = Reader { bytes, at: 0 };
        assert_eq!(reader.take(8), b"HIDX0001");
        let count = reader.u32();
        assert_eq!(count, 852);
        let mut words = 0;
        let mut direct_integer_cast_differences = 0;
        for case in 0..count {
            let dimension = reader.i64();
            let start = u64::from_le_bytes(reader.take(8).try_into().unwrap()) as usize;
            let theta = reader.f64();
            let n = reader.u32() as usize;
            let stages: [&[u8]; 3] = std::array::from_fn(|_| reader.take(n * 4));
            let mut p = Parameters::new(dimension);
            p.theta = theta;
            let elements = encoded_len(&p).unwrap() / 4;
            let arithmetic = Arithmetic::new(&p, SourceProfile::current(), elements);
            assert!(start.checked_add(n).unwrap() <= elements);
            for offset in 0..n {
                let index = start + offset;
                let (arange, exponent, power) = arithmetic.power(index, elements / 32 * 32);
                for (stage, (&actual, expected)) in
                    [arange, exponent, power].iter().zip(stages).enumerate()
                {
                    let begin = offset * 4;
                    let expected =
                        u32::from_le_bytes(expected[begin..begin + 4].try_into().unwrap());
                    assert_eq!(
                        actual.to_bits(),
                        expected,
                        "indexed case {case}, dimension {dimension}, index {index}, stage {stage}"
                    );
                    if stage == 0 && ((index * 2) as f32).to_bits() != expected {
                        direct_integer_cast_differences += 1;
                    }
                    words += 1;
                }
            }
        }
        assert_eq!(reader.at, bytes.len());
        assert_eq!(words, 72_270);
        assert!(
            direct_integer_cast_differences > 0,
            "the independent fixture must expose vector-base rounding"
        );
        println!(
            "{count} original operator windows; {words} exact stage words; {direct_integer_cast_differences} direct-cast counterexamples"
        );
    }
}
