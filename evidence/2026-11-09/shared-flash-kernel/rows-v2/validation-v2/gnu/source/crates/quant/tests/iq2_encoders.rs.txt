use llama_quant::{
    DType, QuantError, quantize, quantize_into, quantize_with_weights, quantize_with_weights_into,
};
struct Reader<'a> {
    bytes: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let r = &self.bytes[self.p..self.p + n];
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
fn iq2_encoders_match_original_lattice_search_and_packing() {
    let mut reader = Reader {
        bytes: include_bytes!("fixtures/iq2-encoder-reference.bin"),
        p: 0,
    };
    assert_eq!(reader.take(8), b"GGQW0001");
    let count = reader.word();
    assert_eq!(count, 624);
    for record in 0..count {
        let dtype = DType::from_id(reader.word() as u32).unwrap();
        let weighted = reader.word() != 0;
        let elements = reader.word();
        let bytes = reader.word();
        let values = reader.floats(elements);
        let weights = if weighted {
            Some(reader.floats(elements))
        } else {
            None
        };
        let expected = reader.take(bytes);
        let actual = if let Some(w) = &weights {
            quantize_with_weights(dtype, &values, w)
        } else {
            quantize(dtype, &values)
        }
        .unwrap_or_else(|e| panic!("record {record}, {dtype:?}, weighted={weighted}: {e}"));
        assert_eq!(actual.len(), expected.len());
        if let Some((offset, (actual, expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            panic!(
                "record {record}, {dtype:?}, weighted={weighted}, elements={elements}, byte {offset}: actual {actual:02x}, C {expected:02x}"
            );
        }
        let mut destination = vec![0xa5; bytes];
        if let Some(w) = weights {
            quantize_with_weights_into(dtype, &values, &w, &mut destination).unwrap();
        } else {
            quantize_into(dtype, &values, &mut destination).unwrap();
        }
        assert_eq!(destination, expected);
    }
    assert_eq!(reader.p, reader.bytes.len());
}
#[test]
fn iq2_invalid_later_blocks_cannot_change_destination() {
    {
        let dtype = DType::IQ2_S;
        let mut values = vec![1.0; 512];
        let mut destination = vec![0xa5; dtype.type_size() * 2];
        values[400] = f32::NAN;
        assert_eq!(
            quantize_into(dtype, &values, &mut destination),
            Err(QuantError::NonFiniteInput { index: 400 })
        );
        assert!(destination.iter().all(|&v| v == 0xa5));
        values[400] = f32::MAX;
        assert_eq!(
            quantize_into(dtype, &values, &mut destination),
            Err(QuantError::UnrepresentableScale { block: 1 })
        );
        assert!(destination.iter().all(|&v| v == 0xa5));
        assert!(quantize(dtype, &values[..511]).is_err());
    }
}
