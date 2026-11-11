//! Frozen quantize/common scalar parsing, authored in safe Rust.
//! Source ZIP: 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
//! Copyright (c) 2023-2026 The ggml authors. MIT license.
use crate::{
    Error, FileType, MetadataOverride, OverrideValue, Result, TensorTypeOverride, invalid,
};
use llama_quant::DType;

pub(crate) const MAX_SCALAR_BYTES: usize = 16 * 1024;
pub(crate) fn c_prefix(text: &str) -> &str {
    &text[..text.find('\0').unwrap_or(text.len())]
}
fn checked_text(text: &str) -> Result<&str> {
    if text.len() > MAX_SCALAR_BYTES {
        return Err(Error::Limit("CLI scalar exceeds 16384 bytes".into()));
    }
    Ok(c_prefix(text))
}
pub(crate) fn c_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 11 | 12)
}
/// Prefix decimal conversion. No digits returns None; overflow is a checked error.
pub(crate) fn integer_prefix(text: &str, bits: u32) -> Result<Option<i64>> {
    let bytes = checked_text(text)?.as_bytes();
    let mut i = 0;
    while bytes.get(i).is_some_and(|b| c_space(*b)) {
        i += 1;
    }
    let negative = bytes.get(i) == Some(&b'-');
    if matches!(bytes.get(i), Some(b'+' | b'-')) {
        i += 1;
    }
    let start = i;
    let limit = (1u64 << (bits - 1)) - u64::from(!negative);
    let mut value = 0u64;
    while let Some(&b @ b'0'..=b'9') = bytes.get(i) {
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(b - b'0')))
            .filter(|v| *v <= limit)
            .ok_or_else(|| invalid("CLI integer prefix is outside its C integer range"))?;
        i += 1;
    }
    if i == start {
        return Ok(None);
    }
    Ok(Some(if negative {
        if value == 1u64 << 63 {
            i64::MIN
        } else {
            -(value as i64)
        }
    } else {
        value as i64
    }))
}
pub(crate) const FILE_TYPES: &[(&str, u32)] = &[
    ("Q1_0", 40),
    ("Q2_0", 41),
    ("Q4_0", 2),
    ("Q4_1", 3),
    ("MXFP4_MOE", 38),
    ("Q5_0", 8),
    ("Q5_1", 9),
    ("IQ2_XXS", 19),
    ("IQ2_XS", 20),
    ("IQ2_S", 28),
    ("IQ2_M", 29),
    ("IQ1_S", 24),
    ("IQ1_M", 31),
    ("TQ1_0", 36),
    ("TQ2_0", 37),
    ("Q2_K", 10),
    ("Q2_K_S", 21),
    ("IQ3_XXS", 23),
    ("IQ3_S", 26),
    ("IQ3_M", 27),
    ("Q3_K", 12),
    ("IQ3_XS", 22),
    ("Q3_K_S", 11),
    ("Q3_K_M", 12),
    ("Q3_K_L", 13),
    ("IQ4_NL", 25),
    ("IQ4_XS", 30),
    ("Q4_K", 15),
    ("Q4_K_S", 14),
    ("Q4_K_M", 15),
    ("Q5_K", 17),
    ("Q5_K_S", 16),
    ("Q5_K_M", 17),
    ("Q6_K", 18),
    ("Q8_0", 7),
    ("F16", 1),
    ("BF16", 32),
    ("F32", 0),
    ("COPY", 0),
];
/// File type plus the original table spelling used in default output filenames.
pub fn parse_file_type(text: &str) -> Result<(FileType, &'static str)> {
    let text = checked_text(text)?;
    let named = FILE_TYPES
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(text));
    let selected = if let Some(row) = named {
        Some(row)
    } else {
        integer_prefix(text, 32)
            .ok()
            .flatten()
            .and_then(|id| FILE_TYPES.iter().find(|(_, n)| i64::from(*n) == id))
    };
    let &(name, id) = selected.ok_or_else(|| invalid(format!("unknown file type {text:?}")))?;
    Ok((FileType::from_id(id)?, name))
}
pub fn parse_dtype(text: &str) -> Result<DType> {
    let text = checked_text(text)?;
    DType::ALL
        .into_iter()
        .find(|dtype| format!("{dtype:?}").eq_ignore_ascii_case(text))
        .ok_or_else(|| invalid(format!("invalid ggml_type {text:?}")))
}
pub fn parse_tensor_type(text: &str) -> Result<TensorTypeOverride> {
    let text = checked_text(text)?;
    let (pattern, dtype) = text
        .split_once('=')
        .ok_or_else(|| invalid("malformed tensor type"))?;
    if pattern.is_empty() || dtype.is_empty() {
        return Err(invalid("tensor type needs a nonempty name and type"));
    }
    Ok(TensorTypeOverride {
        pattern: pattern.bytes().map(|b| b.to_ascii_lowercase()).collect(),
        dtype: parse_dtype(dtype)?,
    })
}
pub fn parse_prune_layers(text: &str) -> Result<Vec<i32>> {
    let text = checked_text(text)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(text.bytes().filter(|b| *b == b',').count() + 1)
        .map_err(|_| Error::Limit("pruned layer list allocation".into()))?;
    for token in text.split(',') {
        let id = integer_prefix(token, 32)?
            .filter(|v| *v >= 0)
            .ok_or_else(|| invalid(format!("invalid layer id {token:?}")))?;
        result.push(id as i32);
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}
pub fn parse_metadata_override(text: &str) -> Result<MetadataOverride> {
    let text = checked_text(text)?;
    let (key, value) = text
        .split_once('=')
        .ok_or_else(|| invalid("malformed KV override"))?;
    if key.len() >= 128 {
        return Err(invalid("KV override key must be shorter than 128 bytes"));
    }
    let value = if let Some(v) = value.strip_prefix("int:") {
        OverrideValue::Integer(integer_prefix(v, if cfg!(windows) { 32 } else { 64 })?.unwrap_or(0))
    } else if let Some(v) = value.strip_prefix("float:") {
        OverrideValue::Float(float_prefix(v)?)
    } else if let Some(v) = value.strip_prefix("bool:") {
        OverrideValue::Bool(match v {
            "true" => true,
            "false" => false,
            _ => return Err(invalid("KV override boolean must be true or false")),
        })
    } else if let Some(v) = value.strip_prefix("str:") {
        if v.len() > 127 {
            return Err(invalid("KV override string exceeds 127 bytes"));
        }
        OverrideValue::String(v.as_bytes().to_vec())
    } else {
        return Err(invalid("invalid KV override type"));
    };
    Ok(MetadataOverride {
        key: key.as_bytes().to_vec(),
        value,
    })
}
fn starts(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes
        .get(..prefix.len())
        .is_some_and(|v| v.eq_ignore_ascii_case(prefix))
}
fn nan_bits(body: &[u8], negative: bool) -> u64 {
    let payload = body
        .strip_prefix(b"(")
        .and_then(|rest| rest.iter().position(|b| *b == b')').map(|n| &rest[..n]));
    let sign = u64::from(negative) << 63;
    if cfg!(windows) {
        let value = match payload {
            Some(v) if v.eq_ignore_ascii_case(b"snan") => 0x7ff0_0000_0000_0001,
            Some(v) if v.eq_ignore_ascii_case(b"ind") => 0xfff8_0000_0000_0000,
            _ => 0x7fff_ffff_ffff_ffff,
        };
        return sign | value;
    }
    let value = payload
        .and_then(|v| {
            if v.is_empty() || v.iter().any(|b| !b.is_ascii_alphanumeric() && *b != b'_') {
                return None;
            }
            let (radix, digits) = if starts(v, b"0x") {
                (16, &v[2..])
            } else if v.starts_with(b"0") {
                (8, v)
            } else {
                (10, v)
            };
            if digits.is_empty() {
                return None;
            }
            let mut result = 0u64;
            for &b in digits {
                let digit = hex_digit(b).filter(|d| *d < radix)?;
                result = result
                    .saturating_mul(u64::from(radix))
                    .saturating_add(u64::from(digit));
            }
            Some(result)
        })
        .unwrap_or(0);
    sign | 0x7ff8_0000_0000_0000 | (value & 0x000f_ffff_ffff_ffff)
}
fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}
fn exponent(bytes: &[u8], index: usize) -> Option<i64> {
    let mut i = index;
    let negative = bytes.get(i) == Some(&b'-');
    if matches!(bytes.get(i), Some(b'+' | b'-')) {
        i += 1;
    }
    let start = i;
    let mut value = 0i64;
    while let Some(&b @ b'0'..=b'9') = bytes.get(i) {
        value = (value * 10 + i64::from(b - b'0')).min(1_000_000);
        i += 1;
    }
    (i != start).then_some(if negative { -value } else { value })
}
fn rounded_right(value: u64, shift: i64, sticky: bool) -> u64 {
    if shift <= 0 {
        return value << (-shift as u32);
    }
    if shift > 64 {
        return 0;
    }
    if shift == 64 {
        return u64::from(value > 1u64 << 63 || (value == 1u64 << 63 && sticky));
    }
    let shift = shift as u32;
    let high = value >> shift;
    let remainder = value & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    high + u64::from(remainder > half || (remainder == half && (sticky || high & 1 != 0)))
}
fn hex_float(body: &[u8], negative: bool) -> Option<f64> {
    let mut i = 2;
    let mut dot = false;
    let mut digits = 0usize;
    let mut fractional = 0i64;
    let mut total_bits = 0i64;
    let mut kept = 0i64;
    let mut significand = 0u64;
    let mut sticky = false;
    while let Some(&b) = body.get(i) {
        if b == b'.' && !dot {
            dot = true;
            i += 1;
            continue;
        }
        let Some(d) = hex_digit(b) else {
            break;
        };
        digits += 1;
        fractional += i64::from(dot);
        for bit in (0..4).rev() {
            let value = (d >> bit) & 1;
            if total_bits == 0 && value == 0 {
                continue;
            }
            total_bits += 1;
            if kept < 64 {
                significand = (significand << 1) | u64::from(value);
                kept += 1;
            } else {
                sticky |= value != 0;
            }
        }
        i += 1;
    }
    if digits == 0 {
        return None;
    }
    let sign = u64::from(negative) << 63;
    if total_bits == 0 {
        return Some(f64::from_bits(sign));
    }
    let power = if matches!(body.get(i), Some(b'p' | b'P')) {
        exponent(body, i + 1).unwrap_or(0)
    } else {
        0
    };
    let mut high_exponent = total_bits - 1 + power - 4 * fractional;
    if high_exponent > 1023 {
        return Some(f64::from_bits(sign | 0x7ff0_0000_0000_0000));
    }
    let unit = (high_exponent - 52).max(-1074);
    let rounded = rounded_right(significand, unit - high_exponent + kept - 1, sticky);
    // Correct IEEE ties-to-even rounding is intentional. UCRT loses the
    // carry into DBL_MIN for a measured hex subnormal boundary; public F32
    // metadata remains byte-exact, including signed zero. See correction receipt.
    if high_exponent < -1022 {
        return Some(f64::from_bits(sign | rounded));
    }
    let mantissa = if rounded == 1u64 << 53 {
        high_exponent += 1;
        rounded >> 1
    } else {
        rounded
    };
    let magnitude = if high_exponent > 1023 {
        0x7ff0_0000_0000_0000
    } else {
        ((high_exponent + 1023) as u64) << 52 | (mantissa & 0x000f_ffff_ffff_ffff)
    };
    Some(f64::from_bits(sign | magnitude))
}
fn float_prefix(text: &str) -> Result<f64> {
    let text = checked_text(text)?;
    let bytes = text.as_bytes();
    let mut start = 0;
    while bytes.get(start).is_some_and(|b| c_space(*b)) {
        start += 1;
    }
    let negative = bytes.get(start) == Some(&b'-');
    let mut i = start;
    if matches!(bytes.get(i), Some(b'+' | b'-')) {
        i += 1;
    }
    let body = &bytes[i..];
    if starts(body, b"inf") {
        return Ok(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if starts(body, b"nan") {
        return Ok(f64::from_bits(nan_bits(&body[3..], negative)));
    }
    if starts(body, b"0x")
        && let Some(value) = hex_float(body, negative)
    {
        return Ok(value);
    }
    let mut digits = 0;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
        digits += 1;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return Ok(0.0);
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        let rollback = i;
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let begin = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == begin {
            i = rollback;
        }
    }
    text[start..i]
        .parse()
        .map_err(|_| invalid("invalid CLI floating prefix"))
}
