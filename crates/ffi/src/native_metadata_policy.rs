//! Native x64 metadata map and scalar formatting policies.
//!
//! The frozen MSVC STL and GNU libstdc++ use different string hashes, growth
//! schedules and list iteration orders. This module implements their observed
//! default unordered-map semantics in Rust. It does not call a C++ library.
//! The independent corpus includes duplicate keys, rehash boundaries, a million
//! entries, and raw IEEE float patterns. Allocation payloads and work are bounded.
use std::fmt::Write;
use std::mem::size_of;

type Result<T> = std::result::Result<T, String>;
const NONE: usize = usize::MAX;
const MAX_KEYS: usize = 1_000_000;
const GNU_GROWTH: &[usize] = &[
    13, 29, 59, 127, 257, 541, 1109, 2357, 5087, 10273, 20753, 42043, 85229, 172933, 351061,
    712697, 1447153,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Profile {
    Msvc,
    Gnu,
}
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub max_bytes: usize,
    pub max_work: u64,
}
pub(crate) struct Order {
    pub indices: Vec<usize>,
    pub peak_bytes: usize,
    pub result_bytes: usize,
    pub work_units: u64,
}
#[derive(Clone, Copy)]
struct Node {
    prev: usize,
    next: usize,
    input: usize,
    hash: u64,
}
struct NativeMap<'a> {
    keys: &'a [&'a [u8]],
    nodes: Vec<Node>,
    buckets: Vec<usize>,
    head: usize,
    tail: usize,
    profile: Profile,
    limits: Limits,
    work: u64,
    peak: usize,
}
fn bytes(count: usize, unit: usize) -> Result<usize> {
    count
        .checked_mul(unit)
        .ok_or_else(|| "metadata policy size overflow".into())
}
fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| "metadata policy size overflow".into())
}
fn allocated<T>(count: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| "metadata policy allocation failed")?;
    Ok(result)
}
impl<'a> NativeMap<'a> {
    fn new(keys: &'a [&'a [u8]], profile: Profile, limits: Limits) -> Result<Self> {
        if keys.len() > MAX_KEYS {
            return Err("metadata policy exceeds one million input keys".into());
        }
        let bucket_count = if profile == Profile::Msvc { 8 } else { 1 };
        let required = add(
            size_of::<Self>(),
            add(
                bytes(keys.len(), size_of::<Node>())?,
                bytes(bucket_count, size_of::<usize>())?,
            )?,
        )?;
        if required > limits.max_bytes {
            return Err("metadata policy byte limit".into());
        }
        let nodes = allocated(keys.len())?;
        let mut buckets = allocated(bucket_count)?;
        buckets.resize(bucket_count, NONE);
        Ok(Self {
            keys,
            nodes,
            buckets,
            head: NONE,
            tail: NONE,
            profile,
            limits,
            work: 0,
            peak: required,
        })
    }
    fn charge_work(&mut self, amount: u64) -> Result<()> {
        self.work = self
            .work
            .checked_add(amount)
            .ok_or("metadata policy work overflow")?;
        if self.work > self.limits.max_work {
            return Err("metadata policy work limit".into());
        }
        Ok(())
    }
    fn charge_bytes(&mut self, extra: usize) -> Result<()> {
        let resident = add(
            size_of::<Self>(),
            add(
                bytes(self.nodes.capacity(), size_of::<Node>())?,
                bytes(self.buckets.capacity(), size_of::<usize>())?,
            )?,
        )?;
        let total = add(resident, extra)?;
        if total > self.limits.max_bytes {
            return Err("metadata policy byte limit".into());
        }
        self.peak = self.peak.max(total);
        Ok(())
    }
    fn bucket(&self, hash: u64) -> usize {
        (hash % self.buckets.len() as u64) as usize
    }
    fn order(&mut self) -> Result<Vec<usize>> {
        self.charge_work(self.nodes.len() as u64)?;
        self.charge_bytes(bytes(self.nodes.len(), size_of::<usize>())?)?;
        let mut result = allocated(self.nodes.len())?;
        let mut node = self.head;
        while node != NONE {
            result.push(self.nodes[node].input);
            node = self.nodes[node].next;
        }
        Ok(result)
    }
    /// Insert an existing node into an otherwise valid bucket/list table.
    /// New MSVC buckets are appended; new GNU buckets are prepended. Within a
    /// populated bucket, every distinct insertion precedes the previous first.
    fn link(&mut self, node: usize) {
        let bucket = self.bucket(self.nodes[node].hash);
        let first = self.buckets[bucket];
        let before = if first != NONE {
            first
        } else if self.profile == Profile::Gnu {
            self.head
        } else {
            NONE
        };
        let after = if before == NONE {
            self.tail
        } else {
            self.nodes[before].prev
        };
        self.nodes[node].prev = after;
        self.nodes[node].next = before;
        if after == NONE {
            self.head = node;
        } else {
            self.nodes[after].next = node;
        }
        if before == NONE {
            self.tail = node;
        } else {
            self.nodes[before].prev = node;
        }
        self.buckets[bucket] = node;
    }
    fn grow(&mut self) -> Result<()> {
        let old_count = self.buckets.len();
        let next = match self.profile {
            Profile::Msvc => old_count
                .checked_mul(if old_count < 512 { 8 } else { 2 })
                .ok_or("metadata bucket growth overflow")?,
            Profile::Gnu => *GNU_GROWTH
                .iter()
                .find(|&&n| n > old_count)
                .ok_or("metadata GNU bucket growth exceeds measured domain")?,
        };
        let scratch_bytes = bytes(self.nodes.len(), size_of::<usize>())?;
        self.charge_bytes(add(scratch_bytes, bytes(next, size_of::<usize>())?)?)?;
        self.charge_work(add(self.nodes.len(), next)? as u64)?;
        let mut order = allocated(self.nodes.len())?;
        let mut node = self.head;
        while node != NONE {
            order.push(node);
            node = self.nodes[node].next;
        }
        let mut buckets = allocated(next)?;
        buckets.resize(next, NONE);
        self.buckets = buckets;
        self.head = NONE;
        self.tail = NONE;
        for node in order {
            self.link(node);
        }
        Ok(())
    }
    fn insert(&mut self, input: usize) -> Result<(u64, bool)> {
        let key = self.keys[input];
        self.charge_work(key.len() as u64 + 1)?;
        let hash = string_hash(key, self.profile);
        self.insert_hash(input, hash)
    }
    fn insert_hash(&mut self, input: usize, hash: u64) -> Result<(u64, bool)> {
        let key = self.keys[input];
        let bucket = self.bucket(hash);
        let mut node = self.buckets[bucket];
        while node != NONE && self.bucket(self.nodes[node].hash) == bucket {
            self.charge_work(1)?;
            if self.nodes[node].hash == hash {
                let other = self.keys[self.nodes[node].input];
                self.charge_work(other.len().min(key.len()) as u64 + 1)?;
                if other == key {
                    return Ok((hash, false));
                }
            }
            node = self.nodes[node].next;
        }
        let grow = self.nodes.len() + 1 > self.buckets.len()
            || (self.profile == Profile::Gnu && self.nodes.is_empty());
        if grow {
            self.grow()?;
        }
        let node = self.nodes.len();
        self.nodes.push(Node {
            prev: NONE,
            next: NONE,
            input,
            hash,
        });
        self.link(node);
        Ok((hash, grow))
    }
}
pub(crate) fn default_map_order(keys: &[&[u8]], profile: Profile, limits: Limits) -> Result<Order> {
    let mut map = NativeMap::new(keys, profile, limits)?;
    for input in 0..keys.len() {
        map.insert(input)?;
    }
    let indices = map.order()?;
    let result_bytes = bytes(indices.capacity(), size_of::<usize>())?;
    Ok(Order {
        indices,
        peak_bytes: map.peak,
        result_bytes,
        work_units: map.work,
    })
}
/// Frozen std::hash<uintptr_t> and default pointer-map list ordering.
pub(crate) fn pointer_map_order(keys: &[usize], profile: Profile) -> Result<Vec<usize>> {
    if keys.len() > 256 {
        return Err("native adapter set exceeds 256 handles".into());
    }
    let words: Vec<_> = keys.iter().map(|key| (*key as u64).to_le_bytes()).collect();
    let refs: Vec<_> = words.iter().map(|word| word.as_slice()).collect();
    let mut map = NativeMap::new(
        &refs,
        profile,
        Limits {
            max_bytes: 128 * 1024,
            max_work: 1_000_000,
        },
    )?;
    for (index, key) in keys.iter().enumerate() {
        let hash = match profile {
            Profile::Msvc => string_hash(&words[index], Profile::Msvc),
            Profile::Gnu => *key as u64,
        };
        map.insert_hash(index, hash)?;
    }
    map.order()
}

/// Frozen std::unordered_set<int32_t> as copied into a legacy batch token.
/// MSVC hashes the four-byte object representation and its copy reinserts
/// iteration order into the retained bucket count. GNU uses the integer hash
/// and preserves list order during copy. The independent 569-case/profile
/// corpus covers every size through 256, rehashes, collisions and duplicates.
pub(crate) fn sequence_set_order(keys: &[i32], profile: Profile) -> Result<Vec<i32>> {
    if keys.len() > 256 || keys.iter().any(|&key| !(0..256).contains(&key)) {
        return Err("native encoder sequence set exceeds the measured I32 domain".into());
    }
    let words: Vec<_> = keys.iter().map(|key| key.to_le_bytes()).collect();
    let refs: Vec<_> = words.iter().map(|word| word.as_slice()).collect();
    let mut map = NativeMap::new(
        &refs,
        profile,
        Limits {
            max_bytes: 128 * 1024,
            max_work: 1_000_000,
        },
    )?;
    for (index, key) in keys.iter().enumerate() {
        let hash = match profile {
            Profile::Msvc => string_hash(&words[index], Profile::Msvc),
            Profile::Gnu => *key as u64,
        };
        map.insert_hash(index, hash)?;
    }
    if profile == Profile::Msvc {
        let mut nodes = allocated(map.nodes.len())?;
        let mut node = map.head;
        while node != NONE {
            nodes.push(node);
            node = map.nodes[node].next;
        }
        map.buckets.fill(NONE);
        map.head = NONE;
        map.tail = NONE;
        for node in nodes {
            map.link(node);
        }
    }
    Ok(map.order()?.into_iter().map(|index| keys[index]).collect())
}

fn string_hash(bytes: &[u8], profile: Profile) -> u64 {
    match profile {
        Profile::Msvc => bytes
            .iter()
            .fold(14_695_981_039_346_656_037, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
            }),
        Profile::Gnu => {
            const M: u64 = 0xc6a4_a793_5bd1_e995;
            let mut hash = 0xc70f_6907 ^ (bytes.len() as u64).wrapping_mul(M);
            let (chunks, tail) = bytes.as_chunks::<8>();
            for bytes in chunks {
                let mut word = u64::from_le_bytes(*bytes).wrapping_mul(M);
                word ^= word >> 47;
                word = word.wrapping_mul(M);
                hash = (hash ^ word).wrapping_mul(M);
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
}
pub(crate) fn format_f32(value: f32, profile: Profile) -> Result<String> {
    // std::to_string(float) promotes to double; that conversion quiets sNaNs.
    // Preserve this explicitly instead of relying on target FP exceptions.
    let bits = value.to_bits();
    if value.is_nan() {
        let negative = bits >> 31 != 0;
        return Ok(if profile == Profile::Msvc && bits == 0xffc0_0000 {
            "-nan(ind)"
        } else if negative {
            "-nan"
        } else {
            "nan"
        }
        .into());
    }
    format_f64(value as f64, profile)
}
pub(crate) fn format_f64(value: f64, profile: Profile) -> Result<String> {
    let bits = value.to_bits();
    if value.is_nan() {
        let sign = if bits >> 63 != 0 { "-" } else { "" };
        let spelling = if profile == Profile::Msvc {
            if bits == 0xfff8_0000_0000_0000 {
                return Ok("-nan(ind)".into());
            }
            if bits & (1 << 51) == 0 {
                "nan(snan)"
            } else {
                "nan"
            }
        } else {
            "nan"
        };
        return Ok(format!("{sign}{spelling}"));
    }
    let mut text = String::new();
    text.try_reserve_exact(317)
        .map_err(|_| "metadata float formatting allocation")?;
    write!(text, "{value:.6}").map_err(|_| "metadata float formatting failed")?;
    if text.len() > 317 {
        return Err("metadata float format exceeds its IEEE bound".into());
    }
    Ok(text)
}

#[cfg(test)]
#[path = "native_metadata_policy_tests.rs"]
mod tests;
