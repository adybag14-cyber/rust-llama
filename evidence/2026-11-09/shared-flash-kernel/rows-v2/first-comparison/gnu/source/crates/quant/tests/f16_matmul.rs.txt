//! Original GGML graph outputs for F16 at SIMD and conversion chunk boundaries.
//! These fixtures pin the actual frozen MSVC fallback, including batched inputs.
use llama_quant::{DType, matvec};

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
fn f16_reduction_matches_original_batched_graphs_bit_for_bit() {
    let mut reader = Reader {
        bytes: include_bytes!("fixtures/f16-matmul-reference.bin"),
        position: 0,
    };
    assert_eq!(reader.take(8), b"GGQH0001");
    let records = reader.word();
    assert_eq!(records, 68);
    let mut output_values = 0;
    let mut bert_shapes = 0;
    for record in 0..records {
        let rows = reader.word();
        let cols = reader.word();
        let batch = reader.word();
        let packed_bytes = reader.word();
        assert_eq!(packed_bytes, rows * cols * 2);
        let weights = reader.take(packed_bytes);
        let activations = reader.floats(cols * batch);
        let expected = reader.floats(rows * batch);
        for (column, input) in activations.chunks_exact(cols).enumerate() {
            let actual = matvec(DType::F16, weights, rows, cols, input).unwrap();
            for (row, (actual, expected)) in actual
                .iter()
                .zip(&expected[column * rows..(column + 1) * rows])
                .enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "record {record}, {rows} x {cols}, batch {batch}, column {column}, row {row}: actual {actual}, native {expected}"
                );
                output_values += 1;
            }
        }
        if rows == 384 && matches!(cols, 384 | 1536) && batch == 7 {
            bert_shapes += 1;
        }
    }
    assert_eq!(bert_shapes, 2);
    assert_eq!(output_values, 6_168);
    assert_eq!(reader.position, reader.bytes.len());
}
