//! Exact Unicode properties from each pinned CPython source environment.
//! PortableFinite admits only the common ASCII metadata domain.
use crate::codec::SourceProfile;
use crate::{Error, Result};
#[path = "python_unicode_data.rs"]
mod data;
pub(crate) const ALPHA: u8 = 1;
pub(crate) const DECIMAL: u8 = 2;
pub(crate) const LOWER: u8 = 4;
pub(crate) const UPPER: u8 = 8;
pub(crate) const TITLE: u8 = 16;
pub(crate) const SPACE: u8 = 32;
pub(crate) const ALNUM: u8 = 64;
fn tables(profile: SourceProfile) -> (&'static [(u32, u32, u8)], &'static [data::Map]) {
    match profile {
        SourceProfile::LinuxTorch211Numpy226 => (data::LINUX_RANGES, data::LINUX_MAPS),
        _ => (data::WINDOWS_RANGES, data::WINDOWS_MAPS),
    }
}
fn admitted(cp: u32, profile: SourceProfile) -> Result<()> {
    if cp > 0x10ffff {
        return Err(Error::Invalid("Python Unicode codepoint range"));
    }
    if profile == SourceProfile::PortableFinite && cp > 127 {
        return Err(Error::Unsupported(
            "non-ASCII metadata requires a captured Python Unicode profile",
        ));
    }
    Ok(())
}
pub(crate) fn flags(cp: u32, profile: SourceProfile) -> Result<u8> {
    admitted(cp, profile)?;
    let ranges = tables(profile).0;
    let index = ranges.partition_point(|(_, end, _)| *end < cp);
    Ok(ranges
        .get(index)
        .filter(|(start, _, _)| *start <= cp)
        .map_or(0, |(_, _, flags)| *flags))
}
fn mapping(cp: u32, profile: SourceProfile) -> Result<Option<&'static data::Map>> {
    admitted(cp, profile)?;
    let maps = tables(profile).1;
    Ok(maps
        .binary_search_by_key(&cp, |m| m.cp)
        .ok()
        .map(|i| &maps[i]))
}
pub(crate) fn decimal(cp: u32, profile: SourceProfile) -> Result<Option<u8>> {
    Ok(mapping(cp, profile)?
        .filter(|m| m.decimal >= 0)
        .map(|m| m.decimal as u8))
}
pub(crate) fn is_lower(value: &str, profile: SourceProfile) -> Result<bool> {
    let mut any = false;
    for cp in value.chars().map(u32::from) {
        let flags = flags(cp, profile)?;
        if flags & (UPPER | TITLE) != 0 {
            return Ok(false);
        }
        any |= flags & LOWER != 0;
    }
    Ok(any)
}
pub(crate) fn all(value: &str, flag: u8, profile: SourceProfile) -> Result<bool> {
    if value.is_empty() {
        return Ok(false);
    }
    for cp in value.chars().map(u32::from) {
        if flags(cp, profile)? & flag == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(crate) fn strip(value: &str, profile: SourceProfile) -> Result<&str> {
    let mut start = 0;
    let mut end = value.len();
    for (index, ch) in value.char_indices() {
        if flags(ch as u32, profile)? & SPACE == 0 {
            start = index;
            break;
        }
        start = index + ch.len_utf8();
    }
    for (index, ch) in value.char_indices().rev() {
        if flags(ch as u32, profile)? & SPACE == 0 {
            end = index + ch.len_utf8();
            break;
        }
        end = index;
    }
    Ok(&value[start.min(end)..end])
}
fn append(out: &mut String, cps: &[u32]) -> Result<()> {
    for &cp in cps {
        let ch =
            char::from_u32(cp).ok_or(Error::Invalid("Unicode expansion contains surrogate"))?;
        out.try_reserve(ch.len_utf8())
            .map_err(|_| Error::Limit("Unicode text expansion"))?;
        out.push(ch);
    }
    Ok(())
}
pub(crate) fn upper(value: &str, profile: SourceProfile) -> Result<String> {
    let mut out = String::new();
    out.try_reserve_exact(value.len())
        .map_err(|_| Error::Limit("uppercase output"))?;
    for ch in value.chars() {
        if let Some(map) = mapping(ch as u32, profile)? {
            append(&mut out, map.upper)?;
        } else {
            out.try_reserve(ch.len_utf8())
                .map_err(|_| Error::Limit("Unicode text growth"))?;
            out.push(ch);
        }
    }
    Ok(out)
}
/// Search an ASCII needle in CPython's lowercased text without allocating the
/// lowercased copy. Contextual final Sigma only changes non-ASCII characters,
/// which cannot match any part of this explicitly ASCII search domain.
pub(crate) fn lower_contains_ascii(
    value: &str,
    needle: &str,
    profile: SourceProfile,
) -> Result<bool> {
    if !needle.is_ascii() {
        return Err(Error::Invalid("lowercase search requires an ASCII needle"));
    }
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return Ok(true);
    }
    let mut prefix = Vec::new();
    prefix
        .try_reserve_exact(needle.len())
        .map_err(|_| Error::Limit("lowercase search pattern"))?;
    prefix.resize(needle.len(), 0);
    for i in 1..needle.len() {
        let mut j = prefix[i - 1];
        while j > 0 && needle[i] != needle[j] {
            j = prefix[j - 1];
        }
        if needle[i] == needle[j] {
            j += 1;
        }
        prefix[i] = j;
    }
    let mut matched = 0;
    for ch in value.chars() {
        let unchanged = [ch as u32];
        let lowered = mapping(ch as u32, profile)?.map_or(unchanged.as_slice(), |m| m.lower);
        for &cp in lowered {
            while matched > 0 && cp != u32::from(needle[matched]) {
                matched = prefix[matched - 1];
            }
            if cp == u32::from(needle[matched]) {
                matched += 1;
            }
            if matched == needle.len() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Metadata.id_to_title calls str.title only on entirely lowercase words.
/// That source precondition avoids the context-sensitive uppercase Sigma rule.
pub(crate) fn title_lower_word(value: &str, profile: SourceProfile) -> Result<String> {
    if !is_lower(value, profile)? {
        return Err(Error::Invalid("title helper requires a lowercase word"));
    }
    let mut out = String::new();
    out.try_reserve_exact(value.len())
        .map_err(|_| Error::Limit("title output"))?;
    let mut previous_cased = false;
    for ch in value.chars() {
        let cp = ch as u32;
        if let Some(map) = mapping(cp, profile)? {
            append(&mut out, if previous_cased { map.lower } else { map.title })?;
        } else {
            out.try_reserve(ch.len_utf8())
                .map_err(|_| Error::Limit("Unicode text growth"))?;
            out.push(ch);
        }
        previous_cased = flags(cp, profile)? & (LOWER | UPPER | TITLE) != 0;
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use llama_checkpoint::{Limits, Number, Value, json};
    fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
        v.as_object().unwrap().get(k).unwrap()
    }
    fn num(v: &Value) -> u32 {
        match v {
            Value::Number(Number::Integer(n)) => n.as_u64().unwrap() as u32,
            _ => panic!("number"),
        }
    }
    fn text(v: &Value) -> Option<&str> {
        v.as_text().unwrap().as_str()
    }
    #[test]
    fn exhaustive_original_unicode_flags_and_decimal_values() {
        for (profile, raw) in [
            (
                SourceProfile::WindowsTorch211Numpy246,
                include_bytes!("../tests/fixtures/python-unicode/windows-reference.json")
                    .as_slice(),
            ),
            (
                SourceProfile::LinuxTorch211Numpy226,
                include_bytes!("../tests/fixtures/python-unicode/linux-reference.json").as_slice(),
            ),
        ] {
            let source = json::parse(raw, Limits::default()).unwrap();
            let ranges = get(&source, "ranges").as_array().unwrap();
            let mut at = 0;
            let maps = get(&source, "maps").as_array().unwrap();
            let mut mi = 0;
            for cp in 0..=0x10ffff {
                while at < ranges.len() && num(&ranges[at].as_array().unwrap()[1]) < cp {
                    at += 1;
                }
                let expected = if at < ranges.len() {
                    let row = ranges[at].as_array().unwrap();
                    if num(&row[0]) <= cp {
                        num(&row[2]) as u8
                    } else {
                        0
                    }
                } else {
                    0
                };
                assert_eq!(flags(cp, profile).unwrap(), expected, "{profile:?}/{cp:X}");
                while mi < maps.len() && num(&maps[mi].as_array().unwrap()[0]) < cp {
                    mi += 1;
                }
                let expected = if mi < maps.len() && num(&maps[mi].as_array().unwrap()[0]) == cp {
                    let value = &maps[mi].as_array().unwrap()[4];
                    if matches!(value, Value::Null) {
                        None
                    } else {
                        Some(num(value) as u8)
                    }
                } else {
                    None
                };
                assert_eq!(
                    decimal(cp, profile).unwrap(),
                    expected,
                    "{profile:?}/{cp:X}"
                );
            }
        }
    }
    #[test]
    fn original_full_case_expansions_for_metadata_domain() {
        for (profile, raw) in [
            (
                SourceProfile::WindowsTorch211Numpy246,
                include_bytes!("../tests/fixtures/python-unicode/windows-strings.json").as_slice(),
            ),
            (
                SourceProfile::LinuxTorch211Numpy226,
                include_bytes!("../tests/fixtures/python-unicode/linux-strings.json").as_slice(),
            ),
        ] {
            let source = json::parse(raw, Limits::default()).unwrap();
            for case in get(&source, "cases").as_array().unwrap() {
                let Some(input) = text(get(case, "text")) else {
                    continue;
                };
                assert_eq!(
                    upper(input, profile).unwrap(),
                    text(get(case, "upper")).unwrap(),
                    "{input:?}"
                );
                let lowered = text(get(case, "lower")).unwrap();
                for needle in [
                    "",
                    "a",
                    "aa",
                    "abc",
                    "e",
                    "k",
                    "i",
                    "ss",
                    "qwen3-reranker",
                    "sequenceclassification",
                ] {
                    assert_eq!(
                        lower_contains_ascii(input, needle, profile).unwrap(),
                        lowered.contains(needle),
                        "{input:?} {needle:?}"
                    );
                }
                let lower = get(case, "islower") == &Value::Bool(true);
                assert_eq!(is_lower(input, profile).unwrap(), lower, "{input:?}");
                if lower {
                    assert_eq!(
                        title_lower_word(input, profile).unwrap(),
                        text(get(case, "title")).unwrap(),
                        "{input:?}"
                    );
                }
            }
        }
    }
}
