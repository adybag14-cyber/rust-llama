//! Deterministic finite-value index sorting for the frozen IQ1 fitting policy.
//!
//! Equal values retain the order produced by median-of-three partitioning,
//! a short-range maximum-selection finish, and removal of trailing pivot ties.
//! This independent value-based implementation is checked against 640 calls
//! to the actual native comparator. No platform runtime or pointer sort is used.
pub(super) fn order(values: &[f32], indices: &mut [usize]) {
    debug_assert_eq!(values.len(), indices.len());
    debug_assert!(values.len() <= 32);
    for (index, destination) in indices.iter_mut().enumerate() {
        *destination = index;
    }
    if indices.len() >= 2 {
        partition(values, indices, 0, indices.len() - 1);
    }
}
fn partition(values: &[f32], indices: &mut [usize], first: usize, last: usize) {
    if last - first < 8 {
        for end in (first + 1..=last).rev() {
            let maximum = (first + 1..=end).fold(first, |best, candidate| {
                if values[indices[candidate]] > values[indices[best]] {
                    candidate
                } else {
                    best
                }
            });
            indices.swap(maximum, end);
        }
        return;
    }
    let middle = first + (last - first).div_ceil(2);
    for (a, b) in [(first, middle), (first, last), (middle, last)] {
        if values[indices[a]] > values[indices[b]] {
            indices.swap(a, b);
        }
    }
    // Copying the finite pivot value makes moving its index irrelevant.
    let pivot = values[indices[middle]];
    let mut lower = first + 1;
    let mut upper = last - 1;
    loop {
        while lower <= last && values[indices[lower]] <= pivot {
            lower += 1;
        }
        while upper > first && values[indices[upper]] > pivot {
            upper -= 1;
        }
        if lower > upper {
            break;
        }
        indices.swap(lower, upper);
        lower += 1;
        upper -= 1;
    }
    while upper > first && values[indices[upper]] == pivot {
        upper -= 1;
    }
    if upper > first {
        partition(values, indices, first, upper);
    }
    if lower < last {
        partition(values, indices, lower, last);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn equal_value_order_matches_original_iq1_native_sort() {
        let bytes = include_bytes!("../tests/fixtures/iq1-sort-reference.bin");
        let mut position = 8;
        assert_eq!(&bytes[..8], b"GGQS0001");
        fn word(bytes: &[u8], position: &mut usize) -> usize {
            let value =
                u32::from_le_bytes(bytes[*position..*position + 4].try_into().unwrap()) as usize;
            *position += 4;
            value
        }
        let count = word(bytes, &mut position);
        assert_eq!(count, 640);
        for record in 0..count {
            let elements = word(bytes, &mut position);
            let values: Vec<f32> = (0..elements)
                .map(|_| f32::from_bits(word(bytes, &mut position) as u32))
                .collect();
            let expected: Vec<usize> = (0..elements).map(|_| word(bytes, &mut position)).collect();
            let mut actual = vec![0; elements];
            super::order(&values, &mut actual);
            assert_eq!(
                actual, expected,
                "sort record {record}, {elements} elements"
            );
        }
        assert_eq!(position, bytes.len());
    }
}
