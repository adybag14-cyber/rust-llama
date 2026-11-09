use llama_quant::{
    DType, QuantError, dot_packed, matvec, matvec_activation_dtype, quantize, quantize_into,
};

const EXTRA_TYPES: [DType; 20] = [
    DType::Q1_0,
    DType::Q2_0,
    DType::MXFP4,
    DType::NVFP4,
    DType::Q2_K,
    DType::Q3_K,
    DType::Q4_K,
    DType::Q5_K,
    DType::Q6_K,
    DType::IQ2_XXS,
    DType::IQ2_XS,
    DType::IQ2_S,
    DType::IQ3_XXS,
    DType::IQ3_S,
    DType::IQ1_S,
    DType::IQ1_M,
    DType::IQ4_NL,
    DType::IQ4_XS,
    DType::TQ1_0,
    DType::TQ2_0,
];
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let result = &self.bytes[self.position..self.position + n];
        self.position += n;
        result
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
fn all_twenty_remaining_dots_match_both_independent_c_oracles_bitwise() {
    let mut reader = Reader {
        bytes: include_bytes!("fixtures/matvec-extended-reference.bin"),
        position: 0,
    };
    assert_eq!(reader.take(8), b"GGQX0001");
    let count = reader.word();
    assert_eq!(count, 480);
    let mut covered = std::collections::HashSet::new();
    for record in 0..count {
        let dtype = DType::from_id(reader.word() as u32).unwrap();
        let rows = reader.word();
        let cols = reader.word();
        let weight_size = reader.word();
        let activation_size = reader.word();
        let weights = reader.take(weight_size);
        let input = reader.floats(cols);
        let activation = reader.take(activation_size);
        let generic = reader.floats(rows);
        let native = reader.floats(rows);
        let encoded = quantize(matvec_activation_dtype(dtype).unwrap(), &input).unwrap();
        assert_eq!(
            encoded, activation,
            "{dtype:?} activation bytes in case {record}"
        );
        let actual = matvec(dtype, weights, rows, cols, &input).unwrap();
        for row in 0..rows {
            assert_eq!(
                actual[row].to_bits(),
                generic[row].to_bits(),
                "{dtype:?} original generic C case {record}, row {row}, width {cols}: {} != {}",
                actual[row],
                generic[row]
            );
            assert_eq!(
                actual[row].to_bits(),
                native[row].to_bits(),
                "{dtype:?} original GGML graph case {record}, row {row}, width {cols}: {} != {}",
                actual[row],
                native[row]
            );
            let stride = dtype.row_bytes(cols).unwrap();
            let direct = dot_packed(
                dtype,
                &weights[row * stride..(row + 1) * stride],
                activation,
                cols,
            )
            .unwrap();
            assert_eq!(direct.to_bits(), actual[row].to_bits());
        }
        covered.insert(dtype);
    }
    assert_eq!(covered.len(), EXTRA_TYPES.len());
    assert!(EXTRA_TYPES.iter().all(|t| covered.contains(t)));
    assert_eq!(reader.position, reader.bytes.len());
}

#[test]
fn q8k_activation_encoder_matches_original_c_including_extreme_finite_values() {
    let mut reader = Reader {
        bytes: include_bytes!("fixtures/q8k-reference.bin"),
        position: 0,
    };
    assert_eq!(reader.take(8), b"GGQ80001");
    let count = reader.word();
    assert_eq!(count, 30);
    for record in 0..count {
        let elements = reader.word();
        let bytes = reader.word();
        let input = reader.floats(elements);
        let expected = reader.take(bytes);
        let actual = quantize(DType::Q8_K, &input).unwrap();
        assert_eq!(
            actual, expected,
            "Q8_K byte mismatch in independent case {record}"
        );
        for block in actual.as_chunks::<292>().0.iter() {
            for group in 0..16 {
                let sum = block[4 + group * 16..20 + group * 16]
                    .iter()
                    .map(|x| *x as i8 as i16)
                    .sum::<i16>();
                assert_eq!(
                    i16::from_le_bytes(block[260 + group * 2..262 + group * 2].try_into().unwrap()),
                    sum
                );
            }
        }
    }
    assert_eq!(reader.position, reader.bytes.len());
}

#[test]
fn q8k_preflight_is_atomic_and_zero_blocks_are_canonical() {
    let mut output = vec![0xa5; 584];
    let mut input = vec![1.0; 512];
    input[256..].fill(f32::from_bits(1));
    assert_eq!(
        quantize_into(DType::Q8_K, &input, &mut output),
        Err(QuantError::UnrepresentableScale { block: 1 })
    );
    assert_eq!(output, vec![0xa5; 584]);
    input[300] = f32::INFINITY;
    assert_eq!(
        quantize_into(DType::Q8_K, &input, &mut output),
        Err(QuantError::NonFiniteInput { index: 300 })
    );
    assert_eq!(output, vec![0xa5; 584]);
    input.fill(-0.0);
    quantize_into(DType::Q8_K, &input, &mut output).unwrap();
    assert_eq!(output, vec![0; 584]);
    assert!(matches!(
        quantize(DType::Q8_K, &[1.0; 255]),
        Err(QuantError::NonMultiple { .. })
    ));
    assert!(matches!(
        quantize_into(DType::Q8_K, &[1.0; 256], &mut [0; 291]),
        Err(QuantError::InvalidLength { .. })
    ));
}

#[test]
fn every_extended_matvec_path_rejects_malformed_dimensions_and_packing() {
    for dtype in EXTRA_TYPES {
        let elements = dtype.block_size();
        let input = vec![1.0; elements];
        let weights = vec![0; dtype.type_size()];
        let activation_type = matvec_activation_dtype(dtype).unwrap();
        let activation = quantize(activation_type, &input).unwrap();
        assert!(
            matches!(
                matvec(dtype, &weights[..weights.len() - 1], 1, elements, &input),
                Err(QuantError::InvalidLength { .. })
            ),
            "{dtype:?}"
        );
        assert!(
            matches!(
                matvec(dtype, &weights, 1, elements, &input[..elements - 1]),
                Err(QuantError::InvalidVectorLength { .. })
            ),
            "{dtype:?}"
        );
        assert!(
            matches!(
                matvec(dtype, &weights, usize::MAX, elements, &input),
                Err(QuantError::SizeOverflow)
            ),
            "{dtype:?}"
        );
        assert!(
            dot_packed(
                dtype,
                &weights,
                &activation[..activation.len() - 1],
                elements
            )
            .is_err(),
            "{dtype:?}"
        );
        assert!(
            dot_packed(dtype, &weights, &activation, elements - 1).is_err(),
            "{dtype:?}"
        );
        let mut invalid = input;
        invalid[elements - 1] = f32::NAN;
        assert_eq!(
            matvec(dtype, &weights, 1, elements, &invalid),
            Err(QuantError::NonFiniteInput {
                index: elements - 1
            }),
            "{dtype:?}"
        );
    }
}

#[test]
fn native_weight_dot_capabilities_distinguish_all_twenty_nine_supported_types() {
    let excluded = [
        DType::Q8_1,
        DType::Q8_K,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F64,
    ];
    for dtype in DType::ALL {
        assert_eq!(
            matvec_activation_dtype(dtype).is_ok(),
            !excluded.contains(&dtype),
            "{dtype:?}"
        );
    }
}
