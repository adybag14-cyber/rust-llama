//! Frozen x64 C++ vocabulary iteration order, used only when inferred special
//! spellings are ambiguous. MSVC and libstdc++ choose different first entries.
//! Ordinary token lookup remains an ordered Rust map.
use std::cmp::Reverse;
use std::collections::BTreeMap;

pub(super) struct NativeOrder {
    first_in_bucket: Vec<i32>,
}
impl NativeOrder {
    pub(super) fn new(tokens: &BTreeMap<String, i32>) -> Self {
        let count = bucket_count(tokens.len());
        let mut first = vec![i32::MAX; count];
        let mut buckets = vec![0; tokens.len()];
        for (text, &id) in tokens {
            let bucket = (string_hash(text.as_bytes()) % count as u64) as usize;
            buckets[id as usize] = bucket;
            first[bucket] = first[bucket].min(id);
        }
        let first_in_bucket = buckets.into_iter().map(|bucket| first[bucket]).collect();
        Self { first_in_bucket }
    }
    pub(super) fn rank(&self, id: i32) -> (i32, Reverse<i32>) {
        let first = self.first_in_bucket[id as usize];
        (if cfg!(windows) { first } else { -first }, Reverse(id))
    }
}

pub(super) fn first_matching(
    tokens: &BTreeMap<String, i32>,
    spellings: &[&str],
    order: &mut Option<NativeOrder>,
) -> Option<i32> {
    let mut ids = spellings
        .iter()
        .filter_map(|text| tokens.get(*text).copied());
    let first = ids.next()?;
    let Some(second) = ids.next() else {
        return Some(first);
    };
    let native = order.get_or_insert_with(|| NativeOrder::new(tokens));
    Some(
        std::iter::once(first)
            .chain(std::iter::once(second))
            .chain(ids)
            .min_by_key(|&id| native.rank(id))
            .unwrap(),
    )
}

fn bucket_count(entries: usize) -> usize {
    if cfg!(windows) {
        entries.max(8).next_power_of_two()
    } else {
        PRIME_BUCKETS[PRIME_BUCKETS.partition_point(|&prime| prime < entries)]
    }
}
fn string_hash(bytes: &[u8]) -> u64 {
    if cfg!(windows) {
        bytes.iter().fold(14_695_981_039_346_656_037, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(1_099_511_628_211)
        })
    } else {
        const M: u64 = 0xc6a4_a793_5bd1_e995;
        let mut hash = 0xc70f_6907 ^ (bytes.len() as u64).wrapping_mul(M);
        let (chunks, tail) = bytes.as_chunks::<8>();
        for bytes in chunks {
            let mut k = u64::from_le_bytes(*bytes).wrapping_mul(M);
            k ^= k >> 47;
            k = k.wrapping_mul(M);
            hash = (hash ^ k).wrapping_mul(M);
        }
        if !tail.is_empty() {
            let mut padded = [0; 8];
            padded[..tail.len()].copy_from_slice(tail);
            hash = (hash ^ u64::from_le_bytes(padded)).wrapping_mul(M);
        }
        hash ^= hash >> 47;
        hash = hash.wrapping_mul(M);
        hash ^ (hash >> 47)
    }
}
// Captured from libstdc++'s native reserve policy, through the supported one
// million vocabulary entries. This is compatibility data, not primality search.
const PRIME_BUCKETS: &[usize] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    103, 109, 113, 127, 137, 139, 149, 157, 167, 179, 193, 199, 211, 227, 241, 257, 277, 293, 313,
    337, 359, 383, 409, 439, 467, 503, 541, 577, 619, 661, 709, 761, 823, 887, 953, 1031, 1109,
    1193, 1289, 1381, 1493, 1613, 1741, 1879, 2029, 2179, 2357, 2549, 2753, 2971, 3209, 3469, 3739,
    4027, 4349, 4703, 5087, 5503, 5953, 6427, 6949, 7517, 8123, 8783, 9497, 10273, 11113, 12011,
    12983, 14033, 15173, 16411, 17749, 19183, 20753, 22447, 24281, 26267, 28411, 30727, 33223,
    35933, 38873, 42043, 45481, 49201, 53201, 57557, 62233, 67307, 72817, 78779, 85229, 92203,
    99733, 107897, 116731, 126271, 136607, 147793, 159871, 172933, 187091, 202409, 218971, 236897,
    256279, 277261, 299951, 324503, 351061, 379787, 410857, 444487, 480881, 520241, 562841, 608903,
    658753, 712697, 771049, 834181, 902483, 976369, 1056323,
];

#[cfg(test)]
mod tests {
    use super::*;
    use llama_behavior::json;
    use llama_conformance::digest::sha256;
    #[test]
    fn native_reserve_hash_and_iteration_order() {
        let reference = json::parse(include_str!(
            "../../../fixtures/tokenizer/bpe/map-reference.json"
        ))
        .unwrap();
        let platform = if cfg!(windows) { "windows" } else { "linux" };
        for group in reference.get(platform).unwrap().as_array().unwrap() {
            let size = group.get("count").unwrap().as_u64().unwrap() as usize;
            let keys: Vec<_> = (0..size)
                .map(|id| format!("key_{id}:é汉🦀{}", if id % 37 == 0 { "\0tail" } else { "" }))
                .collect();
            let map = keys
                .iter()
                .enumerate()
                .map(|(id, key)| (key.clone(), id as i32))
                .collect();
            let native = NativeOrder::new(&map);
            let mut order: Vec<_> = (0..size as i32).collect();
            order.sort_by_key(|&id| native.rank(id));
            let mut frame = (size as u32).to_le_bytes().to_vec();
            frame.extend((bucket_count(size) as u32).to_le_bytes());
            for key in keys {
                frame.extend(string_hash(key.as_bytes()).to_le_bytes());
            }
            for id in order {
                frame.extend(id.to_le_bytes());
            }
            assert_eq!(
                sha256(&frame),
                group.get("sha256").unwrap().as_str().unwrap(),
                "native map with {size} entries"
            );
        }
        if !cfg!(windows) {
            let primes: Vec<_> = reference
                .get("prime_buckets")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(PRIME_BUCKETS, primes);
        }
    }
}
