use llama_checkpoint::DType as SourceType;
use llama_convert::codec::{
    CodecError, SourceProfile, WorkingType, encode_into, encode_into_with_profile, encoded_len,
    prepare_into, prepare_into_with_profile, prepared_len, working_type,
};
use llama_quant::DType as TargetType;

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn bytes(&mut self, length: usize) -> &'a [u8] {
        let end = self.at.checked_add(length).unwrap();
        let result = self.data.get(self.at..end).unwrap();
        self.at = end;
        result
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn usize(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.bytes(8).try_into().unwrap())).unwrap()
    }
}
fn source(id: u32) -> SourceType {
    [
        SourceType::F64,
        SourceType::F32,
        SourceType::Bf16,
        SourceType::F16,
        SourceType::I64,
        SourceType::I32,
        SourceType::I16,
        SourceType::U8,
        SourceType::I8,
        SourceType::Bool,
        SourceType::F8E4M3,
        SourceType::F8E5M2,
        SourceType::F8E8M0,
        SourceType::U16,
        SourceType::U32,
        SourceType::U64,
        SourceType::C64,
    ][id as usize]
}
fn compare(data: &[u8], profile: SourceProfile) {
    let mut reader = Reader { data, at: 0 };
    assert_eq!(reader.bytes(8), b"HCOD0001");
    let count = reader.u32();
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut rejected = 0usize;
    let mut output_bytes = 0usize;
    for _ in 0..count {
        let kind = reader.u32();
        let dtype = source(reader.u32());
        let force = reader.u32() != 0;
        let target = reader.u32();
        let width = reader.usize();
        let input_len = reader.usize();
        let expected_len = reader.usize();
        let status = reader.u32();
        let expected_work = reader.u32();
        let name_len = reader.u32() as usize;
        let name = std::str::from_utf8(reader.bytes(name_len)).unwrap();
        let input = reader.bytes(input_len);
        let expected = reader.bytes(expected_len);
        let mut output = vec![0xa5; expected_len];
        let result = if kind == 0 {
            prepare_into_with_profile(dtype, input, force, profile, &mut output).map(|work| {
                if status == 0 {
                    assert_eq!(
                        work,
                        if expected_work == 0 {
                            WorkingType::F16
                        } else {
                            WorkingType::F32
                        },
                        "{name}"
                    );
                    assert_eq!(
                        prepared_len(dtype, input.len(), force).unwrap().1,
                        expected.len(),
                        "{name}"
                    );
                }
            })
        } else {
            let work = if dtype == SourceType::F16 {
                WorkingType::F16
            } else {
                WorkingType::F32
            };
            encode_into_with_profile(
                work,
                input,
                width,
                TargetType::from_id(target).unwrap(),
                profile,
                &mut output,
            )
        };
        if status == 0 {
            if (result.is_err() || output != expected) && failures.len() < 20 {
                let first = output.iter().zip(expected).position(|(a, b)| a != b);
                failures.push(format!(
                    "{name}: result {result:?}, first {first:?}, got/expected {:?}",
                    first.map(|i| (
                        &output[i.saturating_sub(4)..(i + 8).min(output.len())],
                        &expected[i.saturating_sub(4)..(i + 8).min(expected.len())]
                    ))
                ));
            }
            compared += 1;
            output_bytes += expected.len();
        } else {
            let good = match status {
                1 => matches!(result, Err(CodecError::RowWidth { .. })),
                2 => matches!(result, Err(CodecError::UnsupportedSource(_))),
                3 => matches!(result, Err(CodecError::Shape { .. })),
                _ => false,
            };
            if !good && failures.len() < 20 {
                failures.push(format!(
                    "{name}: expected original status {status}, got {result:?}"
                ));
            }
            assert!(
                output.iter().all(|x| *x == 0xa5),
                "{name}: error mutated output"
            );
            rejected += 1;
        }
    }
    assert_eq!(reader.at, data.len());
    println!(
        "{profile:?}: {count} cases, {compared} successes/{rejected} source errors, {output_bytes} checked output bytes"
    );
    assert!(failures.is_empty(), "{}\n", failures.join("\n"));
}
#[test]
fn original_torch_numpy_windows_bytes() {
    compare(
        include_bytes!("fixtures/codec/windows-reference.bin"),
        SourceProfile::WindowsTorch211Numpy246,
    );
}
#[test]
fn original_torch_numpy_linux_bytes() {
    compare(
        include_bytes!("fixtures/codec/linux-reference.bin"),
        SourceProfile::LinuxTorch211Numpy226,
    );
}

#[test]
fn checked_errors_leave_output_unchanged() {
    let mut out = [0xa5; 32];
    let before = out;
    assert!(matches!(
        prepare_into(SourceType::U16, &[0, 0], false, &mut out),
        Err(CodecError::UnsupportedSource(_))
    ));
    assert_eq!(out, before);
    assert!(matches!(
        prepare_into(SourceType::F16, &[0], false, &mut out),
        Err(CodecError::InputLength { .. })
    ));
    assert_eq!(out, before);
    assert!(matches!(
        prepare_into(SourceType::F16, &[0, 0], true, &mut out),
        Err(CodecError::OutputLength { .. })
    ));
    assert_eq!(out, before);
    assert!(matches!(
        encoded_len(WorkingType::F32, 4 * 31, 31, TargetType::Q8_0),
        Err(CodecError::RowWidth { .. })
    ));
    assert!(matches!(
        encoded_len(WorkingType::F32, 4 * 31, 32, TargetType::Q8_0),
        Err(CodecError::Shape { .. })
    ));
    assert!(matches!(
        encoded_len(WorkingType::F32, 0, 0, TargetType::BF16),
        Err(CodecError::Shape { .. })
    ));
    assert_eq!(
        encoded_len(WorkingType::F32, 0, 0, TargetType::F16).unwrap(),
        0
    );
    assert_eq!(
        encoded_len(WorkingType::F32, 0, 256, TargetType::Q8_0).unwrap(),
        0
    );
    assert!(matches!(
        prepared_len(SourceType::U8, usize::MAX, false),
        Err(CodecError::SizeOverflow)
    ));
    assert!(matches!(
        encoded_len(WorkingType::F16, usize::MAX - 1, 1, TargetType::F32),
        Err(CodecError::SizeOverflow)
    ));
    assert!(matches!(
        encode_into(WorkingType::F32, &[0; 4], 1, TargetType::Q4_0, &mut out),
        Err(CodecError::UnsupportedTarget(_))
    ));
    assert_eq!(out, before);
    for dtype in [
        SourceType::F4,
        SourceType::F6E2M3,
        SourceType::F6E3M2,
        SourceType::F8E4M3Fnuz,
        SourceType::F8E5M2Fnuz,
        SourceType::U16,
        SourceType::U32,
        SourceType::U64,
        SourceType::C64,
    ] {
        assert!(matches!(
            working_type(dtype),
            Err(CodecError::UnsupportedSource(_))
        ));
    }
}

#[test]
fn unmeasured_nonfinite_profile_rejects_entire_input_transactionally() {
    let mut input = vec![0u8; 512 * 4];
    input[511 * 4..].copy_from_slice(&0x7fc12345u32.to_le_bytes());
    let mut out =
        vec![0xa5; encoded_len(WorkingType::F32, input.len(), 256, TargetType::Q8_0).unwrap()];
    let before = out.clone();
    assert!(matches!(
        encode_into_with_profile(
            WorkingType::F32,
            &input,
            256,
            TargetType::Q8_0,
            SourceProfile::PortableFinite,
            &mut out
        ),
        Err(CodecError::UnverifiedNonfiniteProfile { index: 511 })
    ));
    assert_eq!(out, before);
    input[511 * 4..].copy_from_slice(&1.0f32.to_le_bytes());
    encode_into_with_profile(
        WorkingType::F32,
        &input,
        256,
        TargetType::Q8_0,
        SourceProfile::PortableFinite,
        &mut out,
    )
    .unwrap();
}
#[test]
fn promotion_and_numpy_cast_are_distinct_and_noop_bits_are_preserved() {
    let raw = 0x7c01u16.to_le_bytes();
    let mut no_op = [0u8; 2];
    let mut torch = [0u8; 4];
    let mut numpy = [0u8; 4];
    prepare_into(SourceType::F16, &raw, false, &mut no_op).unwrap();
    prepare_into_with_profile(
        SourceType::F16,
        &raw,
        true,
        SourceProfile::WindowsTorch211Numpy246,
        &mut torch,
    )
    .unwrap();
    encode_into(WorkingType::F16, &raw, 1, TargetType::F32, &mut numpy).unwrap();
    assert_eq!(no_op, raw);
    assert_eq!(u32::from_le_bytes(torch), 0x7fc02000);
    assert_eq!(u32::from_le_bytes(numpy), 0x7f802000);
    let mut narrow = [0u8; 2];
    encode_into(WorkingType::F32, &numpy, 1, TargetType::F16, &mut narrow).unwrap();
    assert_eq!(narrow, raw);
    prepare_into_with_profile(
        SourceType::F16,
        &raw,
        true,
        SourceProfile::LinuxTorch211Numpy226,
        &mut torch,
    )
    .unwrap();
    assert_eq!(u32::from_le_bytes(torch), 0x7fff_ffff);
    let before = torch;
    assert!(matches!(
        prepare_into_with_profile(
            SourceType::F16,
            &raw,
            true,
            SourceProfile::PortableFinite,
            &mut torch
        ),
        Err(CodecError::UnverifiedNonfiniteProfile { index: 0 })
    ));
    assert_eq!(torch, before);
}

#[test]
fn original_nan_reductions_and_special_cast_tails() {
    compare(
        include_bytes!("fixtures/codec/windows-reductions-reference.bin"),
        SourceProfile::WindowsTorch211Numpy246,
    );
    compare(
        include_bytes!("fixtures/codec/linux-reductions-reference.bin"),
        SourceProfile::LinuxTorch211Numpy226,
    );
}

fn compare_promotion(data: &[u8], profile: SourceProfile) {
    let mut reader = Reader { data, at: 0 };
    assert_eq!(reader.bytes(8), b"HCAT0001");
    let count = reader.u32();
    let mut total = 0usize;
    for _ in 0..count {
        let name_len = reader.u32() as usize;
        let components = reader.u32();
        let output_type = reader.u32();
        let expected_len = reader.usize();
        let name = std::str::from_utf8(reader.bytes(name_len)).unwrap();
        let mut output = Vec::new();
        for _ in 0..components {
            let dtype = if reader.u32() == 0 {
                SourceType::F16
            } else {
                SourceType::F32
            };
            let rows = reader.usize();
            let width = reader.usize();
            let length = reader.usize();
            let raw = reader.bytes(length);
            assert_eq!(
                rows.checked_mul(width).unwrap() * (dtype.bits() as usize / 8),
                raw.len()
            );
            let (_, size) = prepared_len(dtype, raw.len(), output_type == 1).unwrap();
            let mut component = vec![0xa5; size];
            prepare_into_with_profile(dtype, raw, output_type == 1, profile, &mut component)
                .unwrap();
            output.extend(component);
        }
        let expected = reader.bytes(expected_len);
        if output != expected {
            let first = output.iter().zip(expected).position(|(a, b)| a != b);
            panic!(
                "{name}, {profile:?}: first mismatch {first:?}, got {:?}, expected {:?}",
                first.map(|i| &output[i.saturating_sub(4)..(i + 8).min(output.len())]),
                first.map(|i| &expected[i.saturating_sub(4)..(i + 8).min(expected.len())])
            );
        }
        total += expected.len();
    }
    assert_eq!(reader.at, data.len());
    println!("{profile:?}: {count} actual mixed QKV cats, {total} exact output bytes");
}
#[test]
fn actual_mixed_qkv_concatenation_and_post_permutation_tails() {
    compare_promotion(
        include_bytes!("fixtures/codec/windows-promotion-reference.bin"),
        SourceProfile::WindowsTorch211Numpy246,
    );
    compare_promotion(
        include_bytes!("fixtures/codec/linux-promotion-reference.bin"),
        SourceProfile::LinuxTorch211Numpy226,
    );
}

fn compare_half_scalar(data: &[u8], profile: SourceProfile) {
    let mut reader = Reader { data, at: 0 };
    assert_eq!(reader.bytes(8), b"HSCL0001");
    assert_eq!(reader.u32(), 65536);
    for bits in 0..=u16::MAX {
        let mut actual = [0xa5; 4];
        prepare_into_with_profile(
            SourceType::F16,
            &bits.to_le_bytes(),
            true,
            profile,
            &mut actual,
        )
        .unwrap();
        assert_eq!(
            actual,
            reader.bytes(4),
            "{profile:?}, scalar half bits {bits:04x}"
        );
    }
    let promoted = &data[12..12 + 4 * 65536];
    let direct_cast_control = reader.bytes(4 * 65536);
    // This is original-to-original evidence only, not a second Rust capability.
    assert_eq!(promoted, direct_cast_control);
    assert_eq!(reader.at, data.len());
}
#[test]
fn every_half_pattern_through_one_element_source_promotion() {
    compare_half_scalar(
        include_bytes!("fixtures/codec/windows-half-scalar-reference.bin"),
        SourceProfile::WindowsTorch211Numpy246,
    );
    compare_half_scalar(
        include_bytes!("fixtures/codec/linux-half-scalar-reference.bin"),
        SourceProfile::LinuxTorch211Numpy226,
    );
}
