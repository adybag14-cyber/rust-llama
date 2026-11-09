//! Frozen Unicode properties, case maps, normalization and byte/UTF-8 codecs.
//! These functions use the complete source tables, independent of Rust's
//! evolving Unicode version. Source surrogate-codepoint behavior is retained.
//! BPE regular-expression splitting is a separate tokenizer requirement.
#[path = "unicode_data.rs"]
mod data;
use std::{fmt, sync::OnceLock};

pub const MAX_CODEPOINTS: u32 = 0x110000;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flags(pub u16);
impl Flags {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002;
    pub const LETTER: u16 = 0x0004;
    pub const SEPARATOR: u16 = 0x0008;
    pub const ACCENT_MARK: u16 = 0x0010;
    pub const PUNCTUATION: u16 = 0x0020;
    pub const SYMBOL: u16 = 0x0040;
    pub const CONTROL: u16 = 0x0080;
    pub const WHITESPACE: u16 = 0x0100;
    pub const LOWERCASE: u16 = 0x0200;
    pub const UPPERCASE: u16 = 0x0400;
    pub const NFD: u16 = 0x0800;
    pub fn contains(self, flag: u16) -> bool {
        self.0 & flag != 0
    }
    pub fn category(self) -> u16 {
        self.0 & 0xff
    }
}
static FLAGS: OnceLock<Vec<Flags>> = OnceLock::new();
pub fn flags(codepoint: u32) -> Flags {
    if codepoint >= MAX_CODEPOINTS {
        return Flags(Flags::UNDEFINED);
    }
    FLAGS.get_or_init(|| {
        let mut values = vec![Flags(Flags::UNDEFINED); MAX_CODEPOINTS as usize];
        for pair in data::RANGE_FLAGS.windows(2) {
            values[pair[0].0 as usize..pair[1].0 as usize].fill(Flags(pair[0].1));
        }
        for &cp in data::WHITESPACE {
            values[cp as usize].0 |= Flags::WHITESPACE;
        }
        for &(_, cp) in data::LOWERCASE {
            values[cp as usize].0 |= Flags::LOWERCASE;
        }
        for &(_, cp) in data::UPPERCASE {
            values[cp as usize].0 |= Flags::UPPERCASE;
        }
        for &(_, _, cp) in data::NFD {
            values[cp as usize].0 |= Flags::NFD;
        }
        values
    })[codepoint as usize]
}
pub fn lowercase(cp: u32) -> u32 {
    data::LOWERCASE
        .binary_search_by_key(&cp, |&(first, _)| first)
        .map_or(cp, |index| data::LOWERCASE[index].1)
}
/// Single-codepoint uppercase mapping from the complete frozen source table.
pub fn uppercase(cp: u32) -> u32 {
    data::UPPERCASE
        .binary_search_by_key(&cp, |&(first, _)| first)
        .map_or(cp, |index| data::UPPERCASE[index].1)
}
/// Source's one-codepoint NFD base mapping; this is intentionally not the
/// full canonical-decomposition stream used internally by NFC.
pub fn nfd_base(cp: u32) -> u32 {
    let end = data::NFD.partition_point(|&(first, _, _)| first <= cp);
    if end == 0 {
        return cp;
    }
    let (first, last, mapped) = data::NFD[end - 1];
    if first <= cp && cp <= last {
        mapped
    } else {
        cp
    }
}
pub fn normalize_nfd(codepoints: &[u32]) -> Vec<u32> {
    codepoints.iter().copied().map(nfd_base).collect()
}
pub fn combining_class(cp: u32) -> u8 {
    data::COMBINING_CLASS
        .binary_search_by_key(&cp, |&(first, _)| first)
        .map_or(0, |index| data::COMBINING_CLASS[index].1)
}
fn decompose(cp: u32, output: &mut Vec<u32>, pending: &mut Vec<u32>) {
    pending.push(cp);
    while let Some(cp) = pending.pop() {
        if (0xac00..0xac00 + 11172).contains(&cp) {
            let index = cp - 0xac00;
            output.push(0x1100 + index / 588);
            output.push(0x1161 + (index % 588) / 28);
            if !index.is_multiple_of(28) {
                output.push(0x11a7 + index % 28);
            }
        } else if let Ok(index) =
            data::DECOMPOSITION.binary_search_by_key(&cp, |&(first, _, _)| first)
        {
            let (_, first, second) = data::DECOMPOSITION[index];
            if second != 0 {
                pending.push(second);
            }
            pending.push(first);
        } else {
            output.push(cp);
        }
    }
}
fn compose(first: u32, second: u32) -> Option<u32> {
    if (0x1100..0x1100 + 19).contains(&first) && (0x1161..0x1161 + 21).contains(&second) {
        return Some(0xac00 + ((first - 0x1100) * 21 + second - 0x1161) * 28);
    }
    if (0xac00..0xac00 + 11172).contains(&first)
        && (first - 0xac00).is_multiple_of(28)
        && (0x11a8..0x11a7 + 28).contains(&second)
    {
        return Some(first + second - 0x11a7);
    }
    let key = (u64::from(first) << 32) | u64::from(second);
    data::COMPOSITION
        .binary_search_by_key(&key, |&(key, _)| key)
        .ok()
        .map(|index| data::COMPOSITION[index].1)
}
pub fn normalize_nfc(codepoints: &[u32]) -> Vec<u32> {
    let mut decomposed = Vec::with_capacity(codepoints.len());
    let mut pending = Vec::new();
    for &cp in codepoints {
        decompose(cp, &mut decomposed, &mut pending);
    }
    let mut index = 0;
    while index < decomposed.len() {
        if combining_class(decomposed[index]) == 0 {
            index += 1;
        }
        let begin = index;
        while index < decomposed.len() && combining_class(decomposed[index]) != 0 {
            index += 1;
        }
        decomposed[begin..index].sort_by_key(|&cp| combining_class(cp));
    }
    let mut result = Vec::with_capacity(decomposed.len());
    let mut starter = 0;
    let mut previous_class = 0;
    for cp in decomposed {
        let current_class = combining_class(cp);
        let composite =
            if !result.is_empty() && (previous_class == 0 || previous_class < current_class) {
                compose(result[starter], cp)
            } else {
                None
            };
        if let Some(value) = composite {
            result[starter] = value;
        } else {
            if current_class == 0 {
                starter = result.len();
            }
            result.push(cp);
            previous_class = current_class;
        }
    }
    result
}
pub fn is_han(cp: u32) -> bool {
    matches!(cp, 0x4e00..=0x9fff | 0x3400..=0x4dbf | 0x20000..=0x2a6df
        | 0x2a700..=0x2b73f | 0x2b740..=0x2b81f | 0x2b820..=0x2ceaf
        | 0x2ceb0..=0x2ebef | 0xf900..=0xfaff | 0x2f800..=0x2fa1f)
}
/// WordPiece has a distinct frozen range list, including the documented
/// 0x2B920 boundary inherited from the original HF tokenizer behavior.
pub fn is_wordpiece_chinese(cp: u32) -> bool {
    matches!(cp, 0x4e00..=0x9fff | 0x3400..=0x4dbf | 0x20000..=0x2a6df
        | 0x2a700..=0x2b73f | 0x2b740..=0x2b81f | 0x2b920..=0x2ceaf
        | 0xf900..=0xfaff | 0x2f800..=0x2fa1f)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidUtf8,
    InvalidCodepoint,
    NotByteEncoding,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidUtf8 => "invalid frozen-source UTF-8 sequence",
            Self::InvalidCodepoint => "codepoint exceeds U+10FFFF",
            Self::NotByteEncoding => "not a frozen byte-encoding codepoint",
        })
    }
}
impl std::error::Error for Error {}
pub fn len_utf8(first: u8) -> usize {
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4][(first >> 4) as usize]
}
/// Encode exactly as the source, including its surrogate-codepoint bytes.
/// Use a byte vector because those source results are not always valid Rust str.
pub fn encode_utf8(cp: u32) -> Result<Vec<u8>, Error> {
    Ok(match cp {
        0..=0x7f => vec![cp as u8],
        0x80..=0x7ff => vec![0xc0 | (cp >> 6) as u8, 0x80 | (cp & 0x3f) as u8],
        0x800..=0xffff => vec![
            0xe0 | (cp >> 12) as u8,
            0x80 | ((cp >> 6) & 0x3f) as u8,
            0x80 | (cp & 0x3f) as u8,
        ],
        0x10000..=0x10ffff => vec![
            0xf0 | (cp >> 18) as u8,
            0x80 | ((cp >> 12) & 0x3f) as u8,
            0x80 | ((cp >> 6) & 0x3f) as u8,
            0x80 | (cp & 0x3f) as u8,
        ],
        _ => return Err(Error::InvalidCodepoint),
    })
}
/// Offset advances only on success. Three-byte surrogate sequences are accepted
/// by the source; overlong, truncated and out-of-range encodings are rejected.
pub fn decode_next(bytes: &[u8], offset: &mut usize) -> Result<u32, Error> {
    let first = *bytes.get(*offset).ok_or(Error::InvalidUtf8)?;
    let (length, mask, minimum) = if first & 0x80 == 0 {
        (1, 0x7f, 0)
    } else if first & 0x40 == 0 {
        return Err(Error::InvalidUtf8);
    } else if first & 0x20 == 0 {
        (2, 0x1f, 0x80)
    } else if first & 0x10 == 0 {
        (3, 0x0f, 0x800)
    } else if first & 0x08 == 0 {
        (4, 0x07, 0x10000)
    } else {
        return Err(Error::InvalidUtf8);
    };
    let end = offset.checked_add(length).ok_or(Error::InvalidUtf8)?;
    let chunk = bytes.get(*offset..end).ok_or(Error::InvalidUtf8)?;
    let mut cp = u32::from(first & mask);
    for &byte in &chunk[1..] {
        if byte & 0xc0 != 0x80 {
            return Err(Error::InvalidUtf8);
        }
        cp = (cp << 6) | u32::from(byte & 0x3f);
    }
    if cp < minimum || cp > 0x10ffff {
        return Err(Error::InvalidUtf8);
    }
    *offset = end;
    Ok(cp)
}
pub fn from_utf8(bytes: &[u8]) -> Vec<u32> {
    let mut result = Vec::with_capacity(bytes.len());
    let mut offset = 0;
    while offset < bytes.len() {
        match decode_next(bytes, &mut offset) {
            Ok(cp) => result.push(cp),
            Err(_) => {
                offset += 1;
                result.push(0xfffd);
            }
        }
    }
    result
}
pub fn flags_from_utf8(bytes: &[u8]) -> Result<Flags, Error> {
    if bytes.is_empty() {
        return Ok(Flags(Flags::UNDEFINED));
    }
    decode_next(bytes, &mut 0).map(flags)
}
pub fn byte_to_codepoint(byte: u8) -> u32 {
    match byte {
        0..=32 => 256 + u32::from(byte),
        127..=160 => 162 + u32::from(byte),
        173 => 323,
        _ => u32::from(byte),
    }
}
pub fn byte_to_utf8(byte: u8) -> Vec<u8> {
    encode_utf8(byte_to_codepoint(byte)).expect("byte encoding is in U+0000..U+0143")
}
pub fn codepoint_to_byte(cp: u32) -> Result<u8, Error> {
    Ok(match cp {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => cp as u8,
        0x100..=0x120 => (cp - 256) as u8,
        0x121..=0x142 => (cp - 162) as u8,
        0x143 => 173,
        _ => return Err(Error::NotByteEncoding),
    })
}
pub fn utf8_to_byte(bytes: &[u8]) -> Result<u8, Error> {
    let mut offset = 0;
    let cp = decode_next(bytes, &mut offset)?;
    if offset != bytes.len() {
        return Err(Error::NotByteEncoding);
    }
    codepoint_to_byte(cp)
}
