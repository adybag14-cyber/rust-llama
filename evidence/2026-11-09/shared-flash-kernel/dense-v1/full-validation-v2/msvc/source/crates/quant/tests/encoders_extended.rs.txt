use llama_quant::{
    DType, QuantError, convert_f32_to_integer, convert_f32_to_integer_into, quantize, quantize_into,
};

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> &'a [u8] {
        let result = &self.bytes[self.position..self.position + count];
        self.position += count;
        result
    }
    fn word(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
    fn floats(&mut self, count: usize) -> Vec<f32> {
        self.take(count * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect()
    }
}
#[test]
fn extended_reference_encoders_match_original_c_bytes() {
    let mut reader = Reader {
        bytes: include_bytes!("fixtures/encoder-extended-reference.bin"),
        position: 0,
    };
    assert_eq!(reader.take(8), b"GGQE0001");
    let count = reader.word();
    assert_eq!(count, 525);
    for record in 0..count {
        let dtype = DType::from_id(reader.word() as u32).unwrap();
        let elements = reader.word();
        let bytes = reader.word();
        let input = reader.floats(elements);
        let expected = reader.take(bytes);
        let actual = quantize(dtype, &input).unwrap_or_else(|error| {
            panic!("record {record}, {dtype:?}, {elements} elements: {error}")
        });
        assert_eq!(actual.len(), expected.len());
        if let Some((offset, (actual, expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            panic!(
                "record {record}, {dtype:?}, {elements} elements, byte {offset}: actual {actual:02x}, original C {expected:02x}"
            );
        }
        let mut destination = vec![0xa5; bytes];
        quantize_into(dtype, &input, &mut destination).unwrap();
        assert_eq!(
            destination, actual,
            "record {record}: owned and destination encoders differ"
        );
    }
    assert_eq!(reader.position, reader.bytes.len());
}
#[test]
fn integer_assignment_matches_original_c_in_its_defined_domain() {
    let mut reader = Reader {
        bytes: include_bytes!("fixtures/integer-conversion-reference.bin"),
        position: 0,
    };
    assert_eq!(reader.take(8), b"GGQI0001");
    let count = reader.word();
    assert_eq!(count, 3);
    for record in 0..count {
        let dtype = DType::from_id(reader.word() as u32).unwrap();
        let elements = reader.word();
        let bytes = reader.word();
        let input = reader.floats(elements);
        let expected = reader.take(bytes);
        assert_eq!(
            convert_f32_to_integer(dtype, &input).unwrap(),
            expected,
            "record {record}: {dtype:?}"
        );
        let mut destination = vec![0xa5; bytes];
        convert_f32_to_integer_into(dtype, &input, &mut destination).unwrap();
        assert_eq!(destination, expected);
        assert_eq!(
            quantize(dtype, &input),
            Err(QuantError::UnsupportedQuantization(dtype))
        );
    }
    assert_eq!(reader.position, reader.bytes.len());
}
#[test]
fn extended_packed_validation_is_atomic() {
    for dtype in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::TQ1_0,
        DType::TQ2_0,
        DType::MXFP4,
        DType::NVFP4,
    ] {
        let mut input = vec![1.0; dtype.block_size() * 2];
        let mut destination = vec![0xa5; dtype.type_size() * 2];
        input[dtype.block_size()] = f32::NAN;
        assert_eq!(
            quantize_into(dtype, &input, &mut destination),
            Err(QuantError::NonFiniteInput {
                index: dtype.block_size()
            })
        );
        assert!(
            destination.iter().all(|&b| b == 0xa5),
            "{dtype:?} changed output on invalid later block"
        );
        assert!(quantize(dtype, &input[..input.len() - 1]).is_err());
        input[dtype.block_size()] = 1.0;
        assert!(
            quantize_into(dtype, &input, &mut destination[..dtype.type_size() * 2 - 1]).is_err()
        );
    }
    for dtype in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::TQ1_0,
        DType::TQ2_0,
    ] {
        let mut input = vec![1.0; 512];
        input[256..].fill(f32::MAX);
        let mut destination = vec![0x5a; dtype.type_size() * 2];
        assert!(
            matches!(
                quantize_into(dtype, &input, &mut destination),
                Err(QuantError::UnrepresentableScale { block: 1 })
            ),
            "{dtype:?}"
        );
        assert!(destination.iter().all(|&b| b == 0x5a));
    }
    assert!(matches!(
        quantize(DType::MXFP4, &[f32::from_bits(1); 32]),
        Err(QuantError::UnrepresentableScale { block: 0 })
    ));
}
#[test]
fn integer_conversion_rejects_undefined_source_assignments_atomically() {
    for (dtype, outside) in [
        (DType::I8, 128.0),
        (DType::I16, 32768.0),
        (DType::I32, 2147483648.0),
    ] {
        let mut destination = vec![0xa5; dtype.type_size() * 2];
        assert_eq!(
            convert_f32_to_integer_into(dtype, &[1.0, outside], &mut destination),
            Err(QuantError::IntegerOutOfRange { dtype, index: 1 })
        );
        assert!(destination.iter().all(|&b| b == 0xa5));
        assert_eq!(
            convert_f32_to_integer_into(dtype, &[1.0, f32::NAN], &mut destination),
            Err(QuantError::NonFiniteInput { index: 1 })
        );
        assert!(destination.iter().all(|&b| b == 0xa5));
    }
    assert_eq!(
        convert_f32_to_integer(DType::I64, &[1.0]),
        Err(QuantError::UnsupportedIntegerConversion(DType::I64))
    );
}
