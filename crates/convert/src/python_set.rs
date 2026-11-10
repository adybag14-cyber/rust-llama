//! Captured CPython ASCII-set iteration for named chat-template metadata.
//! Native references use PYTHONHASHSEED=0: Python 3.14/SipHash13 on Windows,
//! Python 3.10/SipHash24 on GNU. This is explicit source-build policy, not
//! detection of a Python runtime and not a promise about random hash seeds.
//! Set probing/layout follows the PSF-licensed CPython setobject.c algorithm.
use crate::codec::SourceProfile;
use crate::{Error, Result, reserve};
fn round(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[2] = v[2].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(13) ^ v[0];
    v[3] = v[3].rotate_left(16) ^ v[2];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[1]);
    v[0] = v[0].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(17) ^ v[2];
    v[3] = v[3].rotate_left(21) ^ v[0];
    v[2] = v[2].rotate_left(32);
}
fn hash(bytes: &[u8], c: usize, d: usize) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut v = [
        0x736f6d6570736575,
        0x646f72616e646f6d,
        0x6c7967656e657261,
        0x7465646279746573,
    ];
    let (chunks, tail) = bytes.as_chunks::<8>();
    for chunk in chunks {
        let word = u64::from_le_bytes(*chunk);
        v[3] ^= word;
        for _ in 0..c {
            round(&mut v);
        }
        v[0] ^= word;
    }
    let mut last = (bytes.len() as u64) << 56;
    for (index, &byte) in tail.iter().enumerate() {
        last |= u64::from(byte) << (index * 8);
    }
    v[3] ^= last;
    for _ in 0..c {
        round(&mut v);
    }
    v[0] ^= last;
    v[2] ^= 0xff;
    for _ in 0..d {
        round(&mut v);
    }
    let out = v[0] ^ v[1] ^ v[2] ^ v[3];
    if out == u64::MAX { u64::MAX - 1 } else { out }
}
fn table(size: usize) -> Result<Vec<Option<(u64, usize)>>> {
    let mut out = Vec::new();
    reserve(&mut out, size, "chat template set buckets")?;
    out.resize(size, None);
    Ok(out)
}
fn insert(table: &mut [Option<(u64, usize)>], entry: (u64, usize), names: &[String]) -> bool {
    let mask = table.len() - 1;
    let mut position = (entry.0 as usize) & mask;
    let mut perturb = entry.0;
    loop {
        let count = if position + 9 <= mask { 10 } else { 1 };
        for slot in &mut table[position..position + count] {
            match slot {
                None => {
                    *slot = Some(entry);
                    return true;
                }
                Some(old) if old.0 == entry.0 && names[old.1] == names[entry.1] => return false,
                _ => {}
            }
        }
        perturb >>= 5;
        position = position
            .wrapping_mul(5)
            .wrapping_add(1)
            .wrapping_add(perturb as usize)
            & mask;
    }
}
pub(crate) fn ascii_order(names: &[String], profile: SourceProfile) -> Result<Vec<usize>> {
    let mut order = Vec::new();
    reserve(&mut order, names.len(), "chat template set order")?;
    if names.len() <= 1 {
        order.extend(0..names.len());
        return Ok(order);
    }
    let (c, d) = match profile {
        SourceProfile::WindowsTorch211Numpy246 => (1, 3),
        SourceProfile::LinuxTorch211Numpy226 => (2, 4),
        SourceProfile::PortableFinite => {
            return Err(Error::Unsupported(
                "multiple chat template names require a captured Python hash profile",
            ));
        }
    };
    let mut buckets = table(8)?;
    let mut used = 0usize;
    for (index, name) in names.iter().enumerate() {
        if !name.is_ascii() {
            return Err(Error::Invalid("chat set key must be sanitized ASCII"));
        }
        if insert(&mut buckets, (hash(name.as_bytes(), c, d), index), names) {
            used += 1;
            if used.checked_mul(5).ok_or(Error::Limit("chat set size"))? >= (buckets.len() - 1) * 3
            {
                let minimum = used
                    .checked_mul(if used > 50000 { 2 } else { 4 })
                    .ok_or(Error::Limit("chat set growth"))?;
                let mut size = 8usize;
                while size <= minimum {
                    size = size
                        .checked_mul(2)
                        .ok_or(Error::Limit("chat set buckets"))?;
                }
                let mut grown = table(size)?;
                for entry in buckets.iter().flatten() {
                    insert(&mut grown, *entry, names);
                }
                buckets = grown;
            }
        }
    }
    order.extend(buckets.into_iter().flatten().map(|(_, index)| index));
    Ok(order)
}
