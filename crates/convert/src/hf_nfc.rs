//! Bounded NFC from Tokenizers 0.22.2's pinned Unicode 9.0.0 dependency.
//! Canonical tables come from unicode-normalization-alignments 0.1.12.
//! Every allocation is fallible, and decomposition, stable ordering and
//! composition consume the converter's existing work budget.
#[path = "hf_nfc_data.rs"]
mod data;

use crate::{Error, Result, string};

#[derive(Clone, Copy)]
struct Unit {
    ch: char,
    class: u8,
}
fn charge(work: &mut usize, count: usize) -> Result<()> {
    *work = work
        .checked_sub(count)
        .ok_or(Error::Limit("HF tokenizer work budget"))?;
    Ok(())
}
fn combining_class(ch: char) -> u8 {
    data::COMBINING_CLASS
        .binary_search_by_key(&(ch as u32), |row| row.0)
        .map_or(0, |index| data::COMBINING_CLASS[index].1)
}
fn decomposition(ch: char) -> Option<&'static [char]> {
    data::DECOMPOSITION
        .binary_search_by_key(&(ch as u32), |row| row.0)
        .ok()
        .map(|index| data::DECOMPOSITION[index].1)
}
fn hangul_len(ch: char) -> Option<usize> {
    let cp = ch as u32;
    (0xac00..0xac00 + 11172).contains(&cp).then(|| {
        if (cp - 0xac00).is_multiple_of(28) {
            2
        } else {
            3
        }
    })
}
fn compose(first: char, second: char) -> Option<char> {
    let (a, b) = (first as u32, second as u32);
    if (0x1100..0x1100 + 19).contains(&a) && (0x1161..0x1161 + 21).contains(&b) {
        return char::from_u32(0xac00 + ((a - 0x1100) * 21 + b - 0x1161) * 28);
    }
    if (0xac00..0xac00 + 11172).contains(&a)
        && (a - 0xac00).is_multiple_of(28)
        && (0x11a8..0x11a7 + 28).contains(&b)
    {
        return char::from_u32(a + b - 0x11a7);
    }
    let key = (u64::from(a) << 32) | u64::from(b);
    data::COMPOSITION
        .binary_search_by_key(&key, |row| row.0)
        .ok()
        .map(|index| data::COMPOSITION[index].1)
}

/// Stable merge ordering uses an explicitly reserved scratch slice. Charging
/// each merge/output pass also bounds an adversarial run of combining marks.
fn reorder(values: &mut [Unit], scratch: &mut [Unit], work: &mut usize) -> Result<()> {
    let mut width = 1;
    while width < values.len() {
        let mut start = 0;
        while start < values.len() {
            let mid = start.saturating_add(width).min(values.len());
            let end = mid.saturating_add(width).min(values.len());
            let (mut left, mut right) = (start, mid);
            for output in &mut scratch[start..end] {
                charge(work, 1)?;
                if right == end || (left < mid && values[left].class <= values[right].class) {
                    *output = values[left];
                    left += 1;
                } else {
                    *output = values[right];
                    right += 1;
                }
            }
            start = end;
        }
        charge(work, values.len())?;
        values.copy_from_slice(&scratch[..values.len()]);
        width = width.saturating_mul(2);
    }
    Ok(())
}

pub(crate) fn normalize(input: &str, max_bytes: usize, work: &mut usize) -> Result<String> {
    if input.len() > max_bytes {
        return Err(Error::Limit("HF NFC working bytes"));
    }
    charge(work, input.len())?;
    if input.is_ascii() {
        return string(input);
    }
    let mut count = 0usize;
    for ch in input.chars() {
        let length = hangul_len(ch).unwrap_or_else(|| decomposition(ch).map_or(1, <[char]>::len));
        count = count
            .checked_add(length)
            .ok_or(Error::Limit("HF NFC decomposition size"))?;
    }
    // Two Unit buffers and the maximum UTF-8 output coexist. Reserve only the
    // longest nonstarter run for scratch, but preflight this conservative bound.
    let bounded_bytes = count
        .checked_mul(2 * std::mem::size_of::<Unit>() + 4)
        .ok_or(Error::Limit("HF NFC working bytes"))?;
    if bounded_bytes > max_bytes {
        return Err(Error::Limit("HF NFC working bytes"));
    }
    charge(work, count)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("HF NFC decomposition allocation"))?;
    let mut append = |ch| {
        values.push(Unit {
            ch,
            class: combining_class(ch),
        })
    };
    for ch in input.chars() {
        if hangul_len(ch).is_some() {
            let index = ch as u32 - 0xac00;
            append(char::from_u32(0x1100 + index / 588).expect("Hangul leading Jamo"));
            append(char::from_u32(0x1161 + (index % 588) / 28).expect("Hangul vowel Jamo"));
            if !index.is_multiple_of(28) {
                append(char::from_u32(0x11a7 + index % 28).expect("Hangul trailing Jamo"));
            }
        } else if let Some(chars) = decomposition(ch) {
            for &ch in chars {
                append(ch);
            }
        } else {
            append(ch);
        }
    }
    let mut longest = 0usize;
    let mut run = 0usize;
    for value in &values {
        charge(work, 1)?;
        run = if value.class == 0 { 0 } else { run + 1 };
        longest = longest.max(run);
    }
    let mut scratch = Vec::new();
    if longest > 1 {
        scratch
            .try_reserve_exact(longest)
            .map_err(|_| Error::Limit("HF NFC ordering allocation"))?;
        scratch.resize(longest, Unit { ch: '\0', class: 0 });
    }
    let mut at = 0;
    while at < values.len() {
        if values[at].class == 0 {
            at += 1;
        }
        let begin = at;
        while at < values.len() && values[at].class != 0 {
            at += 1;
        }
        if at - begin > 1 {
            reorder(&mut values[begin..at], &mut scratch, work)?;
        }
    }
    let mut used = 0usize;
    let mut starter: Option<usize> = None;
    let mut previous_class = 0;
    for index in 0..values.len() {
        charge(work, 1)?;
        let value = values[index];
        let composite = starter
            .filter(|_| previous_class == 0 || previous_class < value.class)
            .and_then(|start| compose(values[start].ch, value.ch));
        if let Some(ch) = composite {
            values[starter.expect("composition has a starter")].ch = ch;
        } else {
            if value.class == 0 {
                starter = Some(used);
            }
            values[used] = value;
            used += 1;
            previous_class = value.class;
        }
    }
    let output_bytes = values[..used].iter().try_fold(0usize, |n, value| {
        n.checked_add(value.ch.len_utf8())
            .ok_or(Error::Limit("HF NFC output bytes"))
    })?;
    charge(work, output_bytes)?;
    let mut output = String::new();
    output
        .try_reserve_exact(output_bytes)
        .map_err(|_| Error::Limit("HF NFC output allocation"))?;
    for value in &values[..used] {
        output.push(value.ch);
    }
    Ok(output)
}
