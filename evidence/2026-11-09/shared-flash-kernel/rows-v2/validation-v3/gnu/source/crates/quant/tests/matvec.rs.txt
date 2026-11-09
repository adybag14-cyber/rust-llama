use llama_quant::{
    DType, QuantError, dequantize, dot_packed, matvec, matvec_activation_dtype, quantize,
};
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let x = &self.b[self.p..self.p + n];
        self.p += n;
        x
    }
    fn word(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        self.take(n * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect()
    }
}
#[test]
fn native_matvec_matches_generic_c_and_real_ggml_graphs() {
    let mut r = Reader {
        b: include_bytes!("fixtures/matvec-reference.bin"),
        p: 0,
    };
    assert_eq!(r.take(8), b"GGQM0001");
    let records = r.word();
    assert_eq!(records, 162);
    for record in 0..records {
        let t = DType::from_id(r.word() as u32).unwrap();
        let rows = r.word();
        let cols = r.word();
        let size = r.word();
        let packed = r.take(size);
        let input = r.floats(cols);
        let generic = r.floats(rows);
        let native = r.floats(rows);
        let actual = matvec(t, packed, rows, cols, &input).unwrap();
        for i in 0..rows {
            if t.block_size() > 1 {
                assert_eq!(
                    actual[i].to_bits(),
                    generic[i].to_bits(),
                    "{t:?} record {record}, row {i}: scalar generic dot mismatch"
                );
            }
            let tolerance = 2e-5_f32 * (1.0 + native[i].abs());
            assert!(
                (actual[i] - native[i]).abs() <= tolerance,
                "{t:?} record {record}, row {i}, cols {cols}: actual {}, native {}, tolerance {tolerance}",
                actual[i],
                native[i]
            );
        }
    }
    assert_eq!(r.p, r.b.len());
}
#[test]
fn activation_quantization_is_not_replaced_with_a_dense_dot() {
    let weights = (0..32)
        .map(|i| (i % 7) as f32 * 0.7 - 2.0)
        .collect::<Vec<_>>();
    let input = (0..32).map(|i| (i as f32 * 0.31).sin()).collect::<Vec<_>>();
    let packed = quantize(DType::Q4_0, &weights).unwrap();
    let decoded = dequantize(DType::Q4_0, &packed, 32).unwrap();
    let dense = decoded.iter().zip(&input).map(|(x, y)| x * y).sum::<f32>();
    let native = matvec(DType::Q4_0, &packed, 1, 32, &input).unwrap()[0];
    assert!(
        (dense - native).abs() > 1e-4,
        "regression vector must exercise activation rounding"
    );
    let activation = quantize(DType::Q8_0, &input).unwrap();
    assert_eq!(
        native.to_bits(),
        dot_packed(DType::Q4_0, &packed, &activation, 32)
            .unwrap()
            .to_bits()
    );
}
#[test]
fn matvec_rejects_non_weight_types_and_malformed_rows() {
    assert_eq!(
        matvec_activation_dtype(DType::Q8_K),
        Err(QuantError::UnsupportedMatVec(DType::Q8_K))
    );
    assert!(matches!(
        matvec(DType::Q4_0, &[0; 18], 1, 32, &[1.; 31]),
        Err(QuantError::InvalidVectorLength { .. })
    ));
    assert!(matches!(
        matvec(DType::Q4_0, &[0; 17], 1, 32, &[1.; 32]),
        Err(QuantError::InvalidLength { .. })
    ));
    assert!(matches!(
        matvec(DType::Q4_0, &[], 0, 32, &[1.; 32]),
        Err(QuantError::InvalidMatrixShape { .. })
    ));
    assert!(matches!(
        matvec(DType::Q4_0, &[], 1, 31, &[1.; 31]),
        Err(QuantError::NonMultiple { .. })
    ));
    assert!(matches!(
        matvec(DType::Q4_0, &[0; 18], 1, 32, &[f32::NAN; 32]),
        Err(QuantError::NonFiniteInput { index: 0 })
    ));
    assert!(dot_packed(DType::Q4_0, &[0; 18], &[0; 33], 32).is_err());
}
