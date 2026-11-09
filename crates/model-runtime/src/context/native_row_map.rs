//! Source fused KV row selection for a unified dense Llama cache.
//!
//! Logical columns are independent of the physical K/V extent. Shared cells
//! contribute once per selected sequence to the source padding upper bound,
//! then appear once in the ascending physical-index map. Negative entries pad
//! the map; the shared flash kernel preserves their branch-specific behavior.
use super::{Cell, NativeBatchToken, SequenceSet, invalid};
use crate::Result;

pub(super) struct RowMap {
    pub columns: usize,
    pub queries: usize,
    pub batches: usize,
    pub indices: Vec<i32>,
}

fn selected(tokens: &[NativeBatchToken]) -> SequenceSet {
    let mut ids = SequenceSet::new();
    for token in tokens {
        for &sequence in &token.sequences {
            ids.insert(sequence);
        }
    }
    ids
}

/// Unified source splitting is simple caller order. Equal-length contiguous
/// runs of one sequence become individual logical Q batches; shared/ragged
/// sequence sets use one union slice. No input or heap allocation is needed.
pub(super) fn slices(tokens: &[NativeBatchToken]) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    let count = selected(tokens).len();
    if count <= 1 {
        return 1;
    }
    if !tokens.len().is_multiple_of(count) {
        return 1;
    }
    let per_slice = tokens.len() / count;
    if tokens.chunks_exact(per_slice).all(|run| {
        run.iter()
            .all(|token| token.sequences.len() == 1 && token.sequences[0] == run[0].sequences[0])
    }) {
        count
    } else {
        1
    }
}

pub(super) fn maximum_heap_bytes(
    tokens: &[NativeBatchToken],
    microbatch_tokens: usize,
    capacity: usize,
) -> Result<usize> {
    if microbatch_tokens == 0 {
        return Err(invalid("zero row-map microbatch limit"));
    }
    let maximum = tokens
        .chunks(microbatch_tokens)
        .map(slices)
        .max()
        .unwrap_or(0);
    capacity
        .checked_mul(maximum)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| invalid("native row-map size overflow"))
}

fn slice_ids(
    tokens: &[NativeBatchToken],
    batches: usize,
    index: usize,
    all: SequenceSet,
) -> SequenceSet {
    if batches == 1 {
        all
    } else {
        [tokens[index * (tokens.len() / batches)].sequences[0]]
            .into_iter()
            .collect()
    }
}
fn intersection_count(left: &SequenceSet, right: &SequenceSet) -> usize {
    left.0
        .iter()
        .zip(right.0)
        .map(|(&a, b)| (a & b).count_ones() as usize)
        .sum()
}

impl RowMap {
    pub fn new(cells: &[Cell], tokens: &[NativeBatchToken]) -> Result<Self> {
        if cells.is_empty() || tokens.is_empty() {
            return Err(invalid(
                "native row map requires a nonempty cache and query batch",
            ));
        }
        let batches = slices(tokens);
        let queries = tokens.len() / batches;
        let all = selected(tokens);
        let used = cells
            .iter()
            .rposition(Cell::occupied)
            .map_or(0, |index| index + 1);
        let mut maximum = 0usize;
        for batch in 0..batches {
            let ids = slice_ids(tokens, batches, batch, all);
            let memberships = cells[..used].iter().try_fold(0usize, |count, cell| {
                count
                    .checked_add(intersection_count(&cell.sequences, &ids))
                    .ok_or_else(|| invalid("native row-map membership count overflow"))
            })?;
            maximum = maximum.max(memberships.min(used));
        }
        let columns = maximum
            .checked_add(255)
            .map(|count| count / 256 * 256)
            .ok_or_else(|| invalid("native row-map padding overflow"))?
            .max(256)
            .min(cells.len());
        let count = columns
            .checked_mul(batches)
            .ok_or_else(|| invalid("native row-map shape overflow"))?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count)
            .map_err(|_| invalid("native row-map allocation failed"))?;
        indices.resize(count, -1);
        for batch in 0..batches {
            let ids = slice_ids(tokens, batches, batch, all);
            let destination = &mut indices[batch * columns..(batch + 1) * columns];
            let mut written = 0usize;
            for (index, cell) in cells[..used].iter().enumerate() {
                if cell.occupied() && intersection_count(&cell.sequences, &ids) != 0 {
                    let target = destination
                        .get_mut(written)
                        .ok_or_else(|| invalid("native row-map membership bound exceeded"))?;
                    *target = i32::try_from(index)
                        .map_err(|_| invalid("native physical row exceeds I32"))?;
                    written += 1;
                }
            }
        }
        Ok(Self {
            columns,
            queries,
            batches,
            indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Input, OutputSelection};

    fn token(sequences: &[i32]) -> NativeBatchToken {
        NativeBatchToken {
            sequences: sequences.to_vec(),
            position: 0,
            input: Input::Token(1),
            output: OutputSelection::None,
        }
    }
    #[test]
    fn source_slice_conditions_and_full_batch_bound() {
        assert_eq!(
            slices(&[token(&[0]), token(&[0]), token(&[255]), token(&[255])]),
            2
        );
        assert_eq!(slices(&[token(&[0]), token(&[255]), token(&[0])]), 1);
        assert_eq!(slices(&[token(&[0, 255]), token(&[0, 255])]), 1);
        let tokens = vec![
            token(&[0]),
            token(&[0]),
            token(&[0]),
            token(&[0]),
            token(&[63]),
            token(&[64]),
            token(&[127]),
            token(&[255]),
        ];
        assert_eq!(maximum_heap_bytes(&tokens, 4, 768).unwrap(), 4 * 768 * 4);
        assert!(maximum_heap_bytes(&tokens, 0, 768).is_err());
        assert!(maximum_heap_bytes(&tokens, 4, usize::MAX).is_err());
    }
    fn occupied(cells: &mut [Cell], start: usize, length: usize, position: i32, sequences: &[i32]) {
        for (offset, cell) in cells[start..start + length].iter_mut().enumerate() {
            *cell = Cell {
                position: position + offset as i32,
                shift: 0,
                sequences: sequences.iter().copied().collect(),
            };
        }
    }
    fn source_recipe(case: u32) -> (Vec<Cell>, Vec<NativeBatchToken>) {
        let mut cells = vec![Cell::default(); 1024];
        let tokens = match case {
            0 => {
                occupied(&mut cells, 256, 264, 256, &[0]);
                occupied(&mut cells, 0, 1, 520, &[0]);
                vec![token(&[0])]
            }
            1 => {
                occupied(&mut cells, 0, 260, 0, &[0, 1]);
                occupied(&mut cells, 260, 300, 0, &[2]);
                occupied(&mut cells, 560, 1, 260, &[0, 1]);
                vec![token(&[0, 1])]
            }
            2 => {
                let mut tokens = Vec::new();
                for (group, id) in [63, 64, 127, 128, 255].into_iter().enumerate() {
                    occupied(&mut cells, group, 1, 0, &[id]);
                    occupied(&mut cells, 5 + group * 2, 2, 1, &[id]);
                    tokens.extend([token(&[id]), token(&[id])]);
                }
                tokens
            }
            3 => {
                occupied(&mut cells, 0, 96, 0, &[0, 255]);
                occupied(&mut cells, 96, 1, 96, &[0]);
                occupied(&mut cells, 97, 1, 96, &[255]);
                occupied(&mut cells, 98, 1, 97, &[0, 255]);
                vec![token(&[0]), token(&[255]), token(&[0, 255])]
            }
            4 => {
                occupied(&mut cells, 0, 3, 0, &[0, 2]);
                occupied(&mut cells, 3, 64, 3, &[0]);
                occupied(&mut cells, 67, 64, 3, &[2]);
                let mut tokens = vec![token(&[0]); 64];
                tokens.extend(vec![token(&[2]); 64]);
                tokens
            }
            _ => panic!("unknown original row-map recipe"),
        };
        (cells, tokens)
    }

    #[cfg(all(
        target_arch = "x86_64",
        any(
            all(target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", target_env = "gnu")
        )
    ))]
    #[test]
    fn original_row_map_bytes_and_logical_query_geometry() {
        let data = include_bytes!("../../tests/fixtures/native-row-map-reference.bin");
        assert_eq!(&data[..8], b"LLKVR001");
        let mut cursor = 8;
        fn word(data: &[u8], cursor: &mut usize) -> u32 {
            let value = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
            *cursor += 4;
            value
        }
        assert_eq!(word(data, &mut cursor), 5);
        for case in 0..5 {
            assert_eq!(word(data, &mut cursor), case);
            let columns = word(data, &mut cursor) as usize;
            let queries = word(data, &mut cursor) as usize;
            let batches = word(data, &mut cursor) as usize;
            let count = word(data, &mut cursor) as usize;
            let (cells, tokens) = source_recipe(case);
            let map = RowMap::new(&cells, &tokens).unwrap();
            assert_eq!(
                (map.columns, map.queries, map.batches),
                (columns, queries, batches),
                "original geometry case {case}"
            );
            assert_eq!(map.indices.len(), count);
            for (index, actual) in map.indices.iter().enumerate() {
                assert_eq!(
                    *actual as u32,
                    word(data, &mut cursor),
                    "original map case {case} column {index}"
                );
            }
        }
        assert_eq!(cursor, data.len());
    }
}
