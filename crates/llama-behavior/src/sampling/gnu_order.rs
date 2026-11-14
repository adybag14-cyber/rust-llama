//! GNU/Linux native sampler tie ordering. Independently expressed Rust policy
//! shared in algorithm with tensor/src/native_sort.rs. The frozen GNU 11.4.0
//! oracle verifies both stages of typical + distribution on 512 tied logits.
//! No GNU header, C/C++ object, or foreign runtime is included.

use super::sort_heap_repair;

pub(super) fn sort<T: Copy>(v: &mut [T], key: impl Fn(&T) -> f32 + Copy) {
    if v.len() < 2 {
        return;
    }
    let depth = (usize::BITS - 1 - v.len().leading_zeros()) as usize * 2;
    gnu_partition_ranges(v, 0, v.len(), depth, key);
    // Once all partitions have at most 16 elements, insertion is stable within
    // comparator equivalence classes and preserves the partition's tie order.
    for next in 1..v.len() {
        let value = v[next];
        let mut at = next;
        while at > 0 && key(&value) < key(&v[at - 1]) {
            v[at] = v[at - 1];
            at -= 1;
        }
        v[at] = value;
    }
}
fn gnu_partition_ranges<T: Copy>(
    v: &mut [T],
    lo: usize,
    mut hi: usize,
    mut depth: usize,
    key: impl Fn(&T) -> f32 + Copy,
) {
    while hi - lo > 16 {
        if depth == 0 {
            let part = &mut v[lo..hi];
            for root in (0..part.len() / 2).rev() {
                let value = part[root];
                sort_heap_repair(part, root, part.len(), value, key);
            }
            for end in (1..part.len()).rev() {
                let value = part[end];
                part[end] = part[0];
                sort_heap_repair(part, 0, end, value, key);
            }
            return;
        }
        depth -= 1;
        let (a, b, c) = (lo + 1, lo + (hi - lo) / 2, hi - 1);
        let chosen = if key(&v[a]) < key(&v[b]) {
            if key(&v[b]) < key(&v[c]) {
                b
            } else if key(&v[a]) < key(&v[c]) {
                c
            } else {
                a
            }
        } else if key(&v[a]) < key(&v[c]) {
            a
        } else if key(&v[b]) < key(&v[c]) {
            c
        } else {
            b
        };
        v.swap(lo, chosen);
        let pivot = v[lo];
        let mut left = lo + 1;
        let mut right = hi;
        let cut = loop {
            while key(&v[left]) < key(&pivot) {
                left += 1;
            }
            right -= 1;
            while key(&pivot) < key(&v[right]) {
                right -= 1;
            }
            if left >= right {
                break left;
            }
            v.swap(left, right);
            left += 1;
        };
        gnu_partition_ranges(v, cut, hi, depth, key);
        hi = cut;
    }
}
