use llama_quant::{
    DType, QuantError, dequantize, quantize, quantize_with_weights, quantize_with_weights_into,
    requires_importance_weights,
};
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let r = &self.b[self.p..self.p + n];
        self.p += n;
        r
    }
    fn word(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        self.take(n * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect()
    }
}
#[test]
fn weighted_and_iq4_encoders_match_unmodified_c_bytes() {
    let mut r = Reader {
        b: include_bytes!("fixtures/weighted-encoder-reference.bin"),
        p: 0,
    };
    assert_eq!(r.take(8), b"GGQW0001");
    let count = r.word();
    assert_eq!(count, 1820);
    for record in 0..count {
        let dtype = DType::from_id(r.word() as u32).unwrap();
        let weighted = r.word() != 0;
        let elements = r.word();
        let bytes = r.word();
        let input = r.floats(elements);
        let importance = if weighted {
            Some(r.floats(elements))
        } else {
            None
        };
        let expected = r.take(bytes);
        let actual = if let Some(weights) = &importance {
            quantize_with_weights(dtype, &input, weights)
        } else {
            quantize(dtype, &input)
        }
        .unwrap_or_else(|e| panic!("record {record}, {dtype:?}, weighted={weighted}: {e}"));
        assert_eq!(actual.len(), expected.len());
        if let Some((offset, (a, b))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            panic!(
                "record {record}, {dtype:?}, weighted={weighted}, elements {elements}, byte {offset}: actual {a:02x}, source {b:02x}"
            );
        }
        if let Some(weights) = importance {
            let mut output = vec![0xa5; bytes];
            quantize_with_weights_into(dtype, &input, &weights, &mut output).unwrap();
            assert_eq!(output, expected);
        }
    }
    assert_eq!(r.p, r.b.len());
}
#[test]
fn required_importance_is_explicit_and_validation_is_atomic() {
    for dtype in [DType::IQ2_XXS, DType::IQ2_XS, DType::IQ1_S] {
        assert!(requires_importance_weights(dtype));
        assert_eq!(
            quantize(dtype, &[1.0; 256]),
            Err(QuantError::ImportanceWeightsRequired(dtype))
        );
    }
    for dtype in [
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::IQ1_S,
        DType::IQ1_M,
        DType::IQ2_XXS,
        DType::IQ2_XS,
        DType::IQ2_S,
        DType::IQ3_XXS,
        DType::IQ3_S,
        DType::IQ4_NL,
        DType::IQ4_XS,
    ] {
        let n = dtype.block_size() * 2;
        let input = vec![1.0; n];
        let mut weights = vec![1.0; n];
        let mut output = vec![0xa5; dtype.type_size() * 2];
        weights[n - 1] = -1.0;
        assert_eq!(
            quantize_with_weights_into(dtype, &input, &weights, &mut output),
            Err(QuantError::InvalidImportanceWeight { index: n - 1 })
        );
        assert!(output.iter().all(|&b| b == 0xa5));
        weights[n - 1] = f32::NAN;
        assert_eq!(
            quantize_with_weights_into(dtype, &input, &weights, &mut output),
            Err(QuantError::InvalidImportanceWeight { index: n - 1 })
        );
        assert!(output.iter().all(|&b| b == 0xa5));
        assert_eq!(
            quantize_with_weights(dtype, &input, &weights[..n - 1]),
            Err(QuantError::InvalidImportanceLength {
                expected: n,
                actual: n - 1
            })
        );
        let mut invalid = input.clone();
        invalid[n - 1] = f32::INFINITY;
        weights[n - 1] = 1.0;
        assert_eq!(
            quantize_with_weights_into(dtype, &invalid, &weights, &mut output),
            Err(QuantError::NonFiniteInput { index: n - 1 })
        );
        assert!(output.iter().all(|&b| b == 0xa5));
    }
    assert_eq!(
        quantize_with_weights(DType::I64, &[1.0], &[1.0]),
        Err(QuantError::UnsupportedWeightedQuantization(DType::I64))
    );
}
#[test]
fn uninitialized_source_iq4_nl_zero_payload_is_canonicalized() {
    for value in [0.0, -0.0, 1e-20, -1e-20] {
        let encoded = quantize(DType::IQ4_NL, &[value; 32]).unwrap();
        assert_eq!(encoded, vec![0; 18]);
        assert!(
            dequantize(DType::IQ4_NL, &encoded, 32)
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }
}

#[test]
fn iq_numeric_failures_and_malformed_buffers_are_atomic() {
    for dtype in [
        DType::IQ1_S,
        DType::IQ1_M,
        DType::IQ2_XXS,
        DType::IQ2_XS,
        DType::IQ2_S,
        DType::IQ3_XXS,
        DType::IQ3_S,
        DType::IQ4_NL,
        DType::IQ4_XS,
    ] {
        let n = dtype.block_size() * 2;
        let mut values = vec![1.0; n];
        let mut weights = vec![1.0; n];
        let mut destination = vec![0xa5; dtype.type_size() * 2];
        values[n - 1] = f32::MAX;
        assert!(
            matches!(
                quantize_with_weights_into(dtype, &values, &weights, &mut destination),
                Err(QuantError::UnrepresentableScale { .. })
            ),
            "{dtype:?} must reject overflow before changing destination"
        );
        assert!(destination.iter().all(|&v| v == 0xa5));
        values[n - 1] = 1.0;
        weights[n - 1] = f32::MAX;
        assert!(
            quantize_with_weights_into(dtype, &values, &weights, &mut destination).is_err(),
            "{dtype:?} must reject overflowing weighted fits"
        );
        assert!(destination.iter().all(|&v| v == 0xa5));
        weights[n - 1] = 1.0;
        assert!(matches!(
            quantize_with_weights_into(
                dtype,
                &values[..n - 1],
                &weights[..n - 1],
                &mut destination
            ),
            Err(QuantError::NonMultiple { .. })
        ));
        assert!(destination.iter().all(|&v| v == 0xa5));
        assert!(matches!(
            quantize_with_weights_into(dtype, &values, &weights, &mut destination[..1]),
            Err(QuantError::InvalidLength { .. })
        ));
        assert!(destination.iter().all(|&v| v == 0xa5));
    }
}
#[test]
fn codec_and_source_model_quantizer_capabilities_are_distinct() {
    let mut model_formats = 0;
    let mut reference_codecs = 0;
    for id in 0..64 {
        let Ok(dtype) = DType::from_id(id) else {
            continue;
        };
        let values = vec![1.0; dtype.block_size()];
        let weights = vec![1.0; values.len()];
        let integer = matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64);
        let helper = matches!(dtype, DType::F64 | DType::Q8_1 | DType::Q8_K);
        let result = quantize(dtype, &values);
        if requires_importance_weights(dtype) {
            assert_eq!(result, Err(QuantError::ImportanceWeightsRequired(dtype)));
        } else if integer {
            assert_eq!(result, Err(QuantError::UnsupportedQuantization(dtype)));
        } else {
            assert_eq!(result.unwrap().len(), dtype.type_size());
            reference_codecs += 1;
        }
        let weighted = quantize_with_weights(dtype, &values, &weights);
        if integer || helper {
            assert_eq!(
                weighted,
                Err(QuantError::UnsupportedWeightedQuantization(dtype))
            );
        } else {
            assert_eq!(weighted.unwrap().len(), dtype.type_size());
            model_formats += 1;
        }
    }
    assert_eq!(reference_codecs, 29);
    assert_eq!(model_formats, 29);
}
