use llama_quant::{
    DType, QuantError, dequantize, dequantize_into, f8_e4m3_to_f32, f16_to_f32, f32_to_bf16,
    f32_to_f16, quantize, quantize_into,
};

struct Reader<'a> {
    bytes: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.bytes[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
}
#[test]
fn all_active_decoders_match_independent_frozen_c_reference() {
    let mut r = Reader {
        bytes: include_bytes!("fixtures/dequant-reference.bin"),
        p: 0,
    };
    assert_eq!(r.take(8), b"GGQD0001");
    let count = r.u32();
    assert_eq!(count, DType::ALL.len());
    let mut seen = Vec::new();
    for _ in 0..count {
        let t = DType::from_id(r.u32() as u32).unwrap();
        let n = r.u32();
        let size = r.u32();
        assert_eq!(t.row_bytes(n).unwrap(), size);
        let bytes = r.take(size);
        let reference = r.take(n * 4);
        let actual = dequantize(t, bytes, n).unwrap();
        for (i, (&x, raw)) in actual
            .iter()
            .zip(reference.as_chunks::<4>().0.iter())
            .enumerate()
        {
            let expected = f32::from_le_bytes(*raw);
            assert!(
                x.to_bits() == expected.to_bits() || (x.is_nan() && expected.is_nan()),
                "{t:?} value {i}: actual {x:?} ({:08x}), expected {expected:?} ({:08x})",
                x.to_bits(),
                expected.to_bits()
            );
        }
        assert!(!seen.contains(&t));
        seen.push(t);
    }
    assert_eq!(r.p, r.bytes.len());
}
#[test]
fn reference_encoders_match_frozen_c_bytes() {
    let mut r = Reader {
        bytes: include_bytes!("fixtures/quant-reference.bin"),
        p: 0,
    };
    assert_eq!(r.take(8), b"GGQE0001");
    let count = r.u32();
    assert_eq!(count, 122);
    for record in 0..count {
        let t = DType::from_id(r.u32() as u32).unwrap();
        let n = r.u32();
        let size = r.u32();
        let input = r
            .take(n * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>();
        let expected = r.take(size);
        let got = quantize(t, &input).unwrap();
        if got != expected {
            let pos = got.iter().zip(expected).position(|(x, y)| x != y).unwrap();
            panic!(
                "{t:?} encoder record {record}, byte {pos}: actual {:02x}, expected {:02x}",
                got[pos], expected[pos]
            );
        }
    }
    assert_eq!(r.p, r.bytes.len());
}
#[test]
fn numeric_ids_preserve_every_reserved_hole() {
    for id in [4, 5, 31, 32, 33, 36, 37, 38, 44, u32::MAX] {
        assert_eq!(DType::from_id(id), Err(QuantError::InvalidType(id)));
    }
    for t in DType::ALL {
        assert_eq!(DType::from_id(t.id()), Ok(t));
    }
    assert_eq!(
        (DType::Q8_1.block_size(), DType::Q8_1.type_size()),
        (32, 36)
    );
    assert_eq!(
        (DType::Q1_0.block_size(), DType::Q1_0.type_size()),
        (128, 18)
    );
    assert_eq!(
        (DType::NVFP4.block_size(), DType::NVFP4.type_size()),
        (64, 36)
    );
}
#[test]
fn malformed_rows_are_rejected_before_output_is_changed() {
    for t in DType::ALL {
        let mut out = vec![123.5; t.block_size()];
        let expected = out.clone();
        for size in [0, t.type_size() - 1, t.type_size() + 1] {
            let input = vec![0; size];
            assert!(matches!(
                dequantize_into(t, &input, &mut out),
                Err(QuantError::InvalidLength { .. })
            ));
            assert_eq!(out, expected);
        }
        assert_eq!(dequantize(t, &[], 0).unwrap(), Vec::<f32>::new());
        if t.block_size() > 1 {
            assert!(matches!(
                dequantize(t, &vec![0; t.type_size()], t.block_size() - 1),
                Err(QuantError::NonMultiple { .. })
            ));
        }
    }
    assert_eq!(
        DType::F64.row_bytes(usize::MAX),
        Err(QuantError::SizeOverflow)
    );
}
#[test]
fn half_ieee_boundaries_and_ties_are_explicit() {
    assert_eq!(f16_to_f32(1), 2.0f32.powi(-24));
    assert_eq!(f16_to_f32(0x3ff), 1023.0 * 2.0f32.powi(-24));
    assert_eq!(f16_to_f32(0x400), 2.0f32.powi(-14));
    assert_eq!(f16_to_f32(0x8000).to_bits(), 0x80000000);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(65520.0), 0x7c00);
    assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
    assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
    assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0);
    assert_eq!(
        f32_to_f16(f32::from_bits(2.0f32.powi(-25).to_bits() + 1)),
        1
    );
    for bits in 0..=u16::MAX {
        let v = f16_to_f32(bits);
        if !v.is_nan() {
            assert_eq!(f32_to_f16(v), bits);
        }
    }
}
#[test]
fn bf16_and_fp8_edge_cases_are_ieee_correct() {
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f808000)), 0x3f80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f818000)), 0x3f82);
    assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x7fffffff)), 0x7fff);
    assert_eq!(f8_e4m3_to_f32(0x7e), 448.0);
    assert_eq!(f8_e4m3_to_f32(1), 1.0 / 512.0);
    assert_eq!(f8_e4m3_to_f32(0x80).to_bits(), 0x80000000);
    assert!(f8_e4m3_to_f32(0x7f).is_nan());
    assert!(f8_e4m3_to_f32(0xff).is_nan());
}
#[test]
fn quantizer_fails_clearly_without_mutating_the_destination() {
    let mut x = vec![1.0; 32];
    x[15] = f32::NAN;
    let mut out = vec![0x55; 18];
    let original = out.clone();
    assert_eq!(
        quantize_into(DType::Q4_0, &x, &mut out),
        Err(QuantError::NonFiniteInput { index: 15 })
    );
    assert_eq!(out, original);
    x.fill(f32::MAX);
    assert_eq!(
        quantize_into(DType::Q4_0, &x, &mut out),
        Err(QuantError::UnrepresentableScale { block: 0 })
    );
    assert_eq!(out, original);
    assert!(matches!(
        quantize(DType::I64, &[0.0; 256]),
        Err(QuantError::UnsupportedQuantization(DType::I64))
    ));
}
#[test]
fn multiblock_decode_does_not_interleave_or_bleed_rows() {
    let a = vec![2.0; 32];
    let b = vec![-4.0; 32];
    let mut bytes = quantize(DType::Q8_0, &a).unwrap();
    bytes.extend(quantize(DType::Q8_0, &b).unwrap());
    let y = dequantize(DType::Q8_0, &bytes, 64).unwrap();
    assert_eq!(&y[..32], dequantize(DType::Q8_0, &bytes[..34], 32).unwrap());
    assert_eq!(&y[32..], dequantize(DType::Q8_0, &bytes[34..], 32).unwrap());
}
