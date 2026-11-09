//! Sorting policies whose tied indices are observable in original GGML.
//!
//! The MSVC helpers adapt the already attributed behavior-crate implementation.
//! Copyright (c) Microsoft Corporation. Apache-2.0 WITH LLVM-exception.
//! See THIRD_PARTY_NOTICES.md and LICENSE-MICROSOFT-STL.txt.
//! GNU policy below is independently expressed Rust and verified against the
//! frozen source compiled with GNU 11.4.0/libstdc++; no GNU header is included.
use crate::{Result, SortOrder, TensorError};

pub(super) fn check_platform() -> Result<()> {
    if cfg!(target_env = "msvc") || cfg!(all(target_os = "linux", target_env = "gnu")) {
        Ok(())
    } else {
        Err(TensorError::Unsupported(
            "native sort tie ordering is verified for MSVC and GNU/Linux only",
        ))
    }
}

pub(super) fn full(values: &mut [i32], keys: &[f32], order: SortOrder) {
    let key = |id: &i32| match order {
        SortOrder::Ascending => keys[*id as usize],
        SortOrder::Descending => -keys[*id as usize],
    };
    if cfg!(target_env = "msvc") {
        msvc_sort_range(values, 0, values.len(), values.len(), key);
    } else {
        gnu_sort(values, key);
    }
}

pub(super) fn partial(values: &mut [i32], count: usize, keys: &[f32]) {
    if count == 0 {
        return;
    }
    let key = |id: &i32| -keys[*id as usize];
    for root in (0..count / 2).rev() {
        let value = values[root];
        sort_heap_repair(values, root, count, value, key);
    }
    for candidate in count..values.len() {
        if key(&values[candidate]) < key(&values[0]) {
            let value = values[candidate];
            values[candidate] = values[0];
            sort_heap_repair(values, 0, count, value, key);
        }
    }
    for end in (1..count).rev() {
        let value = values[end];
        values[end] = values[0];
        sort_heap_repair(values, 0, end, value, key);
    }
}

fn median_three<T: Copy>(v: &mut [T], a: usize, b: usize, c: usize, key: impl Fn(&T) -> f32) {
    if key(&v[b]) < key(&v[a]) {
        v.swap(a, b);
    }
    if key(&v[c]) < key(&v[b]) {
        v.swap(b, c);
        if key(&v[b]) < key(&v[a]) {
            v.swap(a, b);
        }
    }
}
fn sort_heap_repair<T: Copy>(
    v: &mut [T],
    root: usize,
    end: usize,
    item: T,
    key: impl Fn(&T) -> f32,
) {
    let mut hole = root;
    while hole * 2 + 1 < end {
        let left = hole * 2 + 1;
        let right = left + 1;
        let child = if right < end && key(&v[right]) >= key(&v[left]) {
            right
        } else {
            left
        };
        v[hole] = v[child];
        hole = child;
    }
    while hole > root {
        let parent = (hole - 1) / 2;
        if key(&v[parent]) >= key(&item) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = item;
}
fn msvc_sort_range<T: Copy>(
    v: &mut [T],
    mut lo: usize,
    mut hi: usize,
    mut budget: usize,
    key: impl Fn(&T) -> f32 + Copy,
) {
    loop {
        if hi - lo <= 32 {
            for next in lo + usize::from(lo < hi)..hi {
                let item = v[next];
                let mut at = next;
                while at > lo && key(&item) < key(&v[at - 1]) {
                    v[at] = v[at - 1];
                    at -= 1;
                }
                v[at] = item;
            }
            return;
        }
        if budget == 0 {
            let part = &mut v[lo..hi];
            for root in (0..part.len() / 2).rev() {
                let item = part[root];
                sort_heap_repair(part, root, part.len(), item, key);
            }
            for end in (1..part.len()).rev() {
                let item = part[end];
                part[end] = part[0];
                sort_heap_repair(part, 0, end, item, key);
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let last = hi - 1;
        if last - lo > 40 {
            let step = (hi - lo) / 8;
            median_three(v, lo, lo + step, lo + 2 * step, key);
            median_three(v, mid - step, mid, mid + step, key);
            median_three(v, last - 2 * step, last - step, last, key);
            median_three(v, lo + step, mid, last - step, key);
        } else {
            median_three(v, lo, mid, last, key);
        }
        let mut equal_lo = mid;
        let mut equal_hi = mid + 1;
        while equal_lo > lo && key(&v[equal_lo - 1]) == key(&v[equal_lo]) {
            equal_lo -= 1;
        }
        while equal_hi < hi && key(&v[equal_hi]) == key(&v[equal_lo]) {
            equal_hi += 1;
        }
        let mut scan_hi = equal_hi;
        let mut scan_lo = equal_lo;
        loop {
            while scan_hi < hi {
                if key(&v[equal_lo]) < key(&v[scan_hi]) {
                    scan_hi += 1;
                } else if key(&v[scan_hi]) < key(&v[equal_lo]) {
                    break;
                } else {
                    v.swap(equal_hi, scan_hi);
                    equal_hi += 1;
                    scan_hi += 1;
                }
            }
            while scan_lo > lo {
                let prev = scan_lo - 1;
                if key(&v[prev]) < key(&v[equal_lo]) {
                    scan_lo -= 1;
                } else if key(&v[equal_lo]) < key(&v[prev]) {
                    break;
                } else {
                    equal_lo -= 1;
                    v.swap(equal_lo, prev);
                    scan_lo -= 1;
                }
            }
            if scan_lo == lo && scan_hi == hi {
                break;
            }
            if scan_lo == lo {
                if equal_hi != scan_hi {
                    v.swap(equal_lo, equal_hi);
                }
                equal_hi += 1;
                v.swap(equal_lo, scan_hi);
                equal_lo += 1;
                scan_hi += 1;
            } else if scan_hi == hi {
                scan_lo -= 1;
                equal_lo -= 1;
                v.swap(scan_lo, equal_lo);
                equal_hi -= 1;
                v.swap(equal_lo, equal_hi);
            } else {
                scan_lo -= 1;
                v.swap(scan_hi, scan_lo);
                scan_hi += 1;
            }
        }
        budget = (budget / 2) + (budget / 4);
        if equal_lo - lo < hi - equal_hi {
            msvc_sort_range(v, lo, equal_lo, budget, key);
            lo = equal_hi;
        } else {
            msvc_sort_range(v, equal_hi, hi, budget, key);
            hi = equal_lo;
        }
    }
}

fn gnu_sort<T: Copy>(v: &mut [T], key: impl Fn(&T) -> f32 + Copy) {
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
