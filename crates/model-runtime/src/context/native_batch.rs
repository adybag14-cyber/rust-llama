//! Frozen dense-KV microbatch order from llama_batch_allocr.
//!
//! Non-unified admission currently has one sequence ID per token. Within that
//! domain the source sequence-set map reduces to per-sequence index queues.
//! Selecting the next consecutive sequence searches its remaining indices after
//! the last selection point, preserving source scan order without rescanning the
//! complete caller batch for every microbatch.
use super::{NativeBatchToken, invalid};
use crate::Result;
use std::mem::size_of;

pub(super) struct Microbatch {
    /// Indices into the original caller batch, grouped in source sequence order.
    pub indices: Vec<usize>,
    pub first_stream: usize,
    pub streams: usize,
    pub queries_per_stream: usize,
}

pub(super) struct Planner<'a> {
    batch: &'a [NativeBatchToken],
    limit: usize,
    unified: bool,
    embeddings_all: bool,
    simple_cursor: usize,
    // Fixed, bounded stack descriptors; heap index payload is charged explicitly.
    groups: [Vec<usize>; 256],
    cursors: [usize; 256],
}

impl<'a> Planner<'a> {
    pub fn maximum_chunk_tokens(
        batch_tokens: usize,
        limit: usize,
        unified: bool,
        embeddings_all: bool,
    ) -> Result<usize> {
        if limit == 0 {
            return Err(invalid("native microbatch limit must be positive"));
        }
        // split_equal stops selecting sequence sets only after size > n_ubatch,
        // so its first one-token-per-stream expansion can exceed n_ubatch by one.
        let maximum = if unified || embeddings_all || limit >= 256 {
            limit
        } else {
            limit
                .checked_add(1)
                .ok_or_else(|| invalid("native microbatch upper bound overflow"))?
        };
        Ok(batch_tokens.min(maximum))
    }

    pub fn heap_bytes(
        batch_tokens: usize,
        limit: usize,
        unified: bool,
        embeddings_all: bool,
    ) -> Result<usize> {
        let indices = if unified { 0 } else { batch_tokens };
        indices
            .checked_add(Self::maximum_chunk_tokens(
                batch_tokens,
                limit,
                unified,
                embeddings_all,
            )?)
            .and_then(|count| count.checked_mul(size_of::<usize>()))
            .ok_or_else(|| invalid("native microbatch planner size overflow"))
    }

    pub fn new(
        batch: &'a [NativeBatchToken],
        limit: usize,
        unified: bool,
        embeddings_all: bool,
    ) -> Result<Self> {
        Self::maximum_chunk_tokens(batch.len(), limit, unified, embeddings_all)?;
        let mut groups: [Vec<usize>; 256] = std::array::from_fn(|_| Vec::new());
        if !unified {
            let mut counts = [0usize; 256];
            for item in batch {
                if item.sequences.len() != 1 || !(0..256).contains(&item.sequences[0]) {
                    return Err(invalid(
                        "separate native microbatch requires one bounded sequence",
                    ));
                }
                counts[item.sequences[0] as usize] += 1;
            }
            for (group, &count) in groups.iter_mut().zip(&counts) {
                group
                    .try_reserve_exact(count)
                    .map_err(|_| invalid("native sequence index allocation failed"))?;
            }
            for (index, item) in batch.iter().enumerate() {
                groups[item.sequences[0] as usize].push(index);
            }
        }
        Ok(Self {
            batch,
            limit,
            unified,
            embeddings_all,
            simple_cursor: 0,
            groups,
            cursors: [0; 256],
        })
    }

    pub fn next(&mut self) -> Result<Option<Microbatch>> {
        if self.unified {
            if self.simple_cursor == self.batch.len() {
                return Ok(None);
            }
            let length = self.limit.min(self.batch.len() - self.simple_cursor);
            let mut indices = Vec::new();
            indices
                .try_reserve_exact(length)
                .map_err(|_| invalid("native microbatch index allocation failed"))?;
            indices.extend(self.simple_cursor..self.simple_cursor + length);
            self.simple_cursor += length;
            return Ok(Some(Microbatch {
                indices,
                first_stream: 0,
                streams: 1,
                queries_per_stream: length,
            }));
        }
        let first = self
            .groups
            .iter()
            .enumerate()
            .filter_map(|(stream, group)| {
                group
                    .get(self.cursors[stream])
                    .map(|&index| (index, stream))
            })
            .min();
        let Some((mut scan_index, first_stream)) = first else {
            return Ok(None);
        };
        let mut streams = 1;
        if !self.embeddings_all {
            while first_stream + streams < 256 && streams <= self.limit {
                let next_stream = first_stream + streams;
                let remaining = &self.groups[next_stream][self.cursors[next_stream]..];
                let after = remaining.partition_point(|&index| index <= scan_index);
                let Some(&next_index) = remaining.get(after) else {
                    break;
                };
                scan_index = next_index;
                streams += 1;
            }
        }
        let available = (first_stream..first_stream + streams)
            .map(|stream| self.groups[stream].len() - self.cursors[stream])
            .min()
            .ok_or_else(|| invalid("native microbatch has no sequence"))?;
        let queries_per_stream = available.min((self.limit / streams).max(1));
        let length = streams
            .checked_mul(queries_per_stream)
            .ok_or_else(|| invalid("native microbatch token count overflow"))?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(length)
            .map_err(|_| invalid("native microbatch index allocation failed"))?;
        for stream in first_stream..first_stream + streams {
            let begin = self.cursors[stream];
            let end = begin + queries_per_stream;
            indices.extend_from_slice(&self.groups[stream][begin..end]);
            self.cursors[stream] = end;
        }
        Ok(Some(Microbatch {
            indices,
            first_stream,
            streams,
            queries_per_stream,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Input, OutputSelection};

    struct Reader<'a>(&'a [u8]);
    impl Reader<'_> {
        fn u32(&mut self) -> usize {
            let (bytes, tail) = self.0.split_at(4);
            self.0 = tail;
            u32::from_le_bytes(bytes.try_into().unwrap()) as usize
        }
        fn text(&mut self) -> String {
            let len = self.u32();
            let (value, tail) = self.0.split_at(len);
            self.0 = tail;
            String::from_utf8(value.to_vec()).unwrap()
        }
    }

    #[test]
    fn both_original_compilers_select_the_same_native_input_order() {
        let bytes = include_bytes!("../../tests/fixtures/native-microbatch-order.bin");
        assert_eq!(&bytes[..8], b"NMBO0001");
        let mut reader = Reader(&bytes[8..]);
        let count = reader.u32();
        assert_eq!(count, 12);
        for _ in 0..count {
            let name = reader.text();
            let limit = reader.u32();
            let unified = reader.u32() != 0;
            let embeddings = reader.u32() != 0;
            let length = reader.u32();
            let batch = (0..length)
                .map(|index| NativeBatchToken {
                    sequences: vec![reader.u32() as i32],
                    position: index as i32,
                    input: Input::Token(index as u32),
                    output: OutputSelection::None,
                })
                .collect::<Vec<_>>();
            let mut planner = Planner::new(&batch, limit, unified, embeddings).unwrap();
            let chunks = reader.u32();
            for _ in 0..chunks {
                let expected_len = reader.u32();
                let expected = (0..expected_len).map(|_| reader.u32()).collect::<Vec<_>>();
                let actual = planner.next().unwrap().unwrap();
                assert_eq!(actual.indices, expected, "{name}");
                assert_eq!(
                    actual.indices.len(),
                    actual.streams * actual.queries_per_stream,
                    "{name}"
                );
                assert!(
                    actual.indices.len()
                        <= Planner::maximum_chunk_tokens(batch.len(), limit, unified, embeddings,)
                            .unwrap(),
                    "{name}"
                );
                for (stream, group) in actual
                    .indices
                    .chunks_exact(actual.queries_per_stream)
                    .enumerate()
                {
                    assert!(
                        group.iter().all(
                            |&i| batch[i].sequences[0] as usize == actual.first_stream + stream
                        ),
                        "{name}"
                    );
                }
            }
            assert!(planner.next().unwrap().is_none(), "{name}");
        }
        assert!(reader.0.is_empty());
    }
    #[test]
    fn unified_order_is_contiguous_and_empty_invalid_domains_are_checked() {
        let batch = (0..11)
            .map(|index| NativeBatchToken {
                sequences: vec![index % 3],
                position: index,
                input: Input::Token(1),
                output: OutputSelection::None,
            })
            .collect::<Vec<_>>();
        for embeddings in [false, true] {
            let mut planner = Planner::new(&batch, 4, true, embeddings).unwrap();
            for indices in [&[0, 1, 2, 3][..], &[4, 5, 6, 7], &[8, 9, 10]] {
                let actual = planner.next().unwrap().unwrap();
                assert_eq!(actual.indices, indices);
                assert_eq!(actual.streams, 1);
                assert_eq!(actual.first_stream, 0);
            }
            assert!(planner.next().unwrap().is_none());
        }
        assert!(Planner::new(&batch, 0, true, false).is_err());
        assert!(Planner::heap_bytes(usize::MAX, 1, false, false).is_err());
    }
}
