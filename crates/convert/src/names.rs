//! Frozen model-ID and title heuristics, using the captured Python Unicode policy.
use crate::codec::SourceProfile;
use crate::python_unicode as py;
use crate::{Error, Result, reserve, string};
#[derive(Debug)]
pub(crate) struct Parts {
    pub name: String,
    pub organization: Option<String>,
    pub basename: Option<String>,
    pub finetune: Option<String>,
    pub version: Option<String>,
    pub size: Option<String>,
}
const BASE: u8 = 1;
const SIZE: u8 = 2;
const FINE: u8 = 4;
const VERSION: u8 = 8;
const TYPE: u8 = 16;
fn equal(ch: char, ascii: char) -> bool {
    ch.eq_ignore_ascii_case(&ascii)
        || (ascii == 'i' && matches!(ch, 'İ' | 'ı'))
        || (ascii == 's' && ch == 'ſ')
        || (ascii == 'k' && ch == 'K')
}
fn word(value: &str, expected: &str) -> bool {
    let mut left = value.chars();
    for ch in expected.chars() {
        if left.next().is_none_or(|v| !equal(v, ch)) {
            return false;
        }
    }
    left.next().is_none()
}
fn decimal(ch: char, profile: SourceProfile) -> Result<bool> {
    Ok(py::flags(ch as u32, profile)? & py::DECIMAL != 0)
}
fn digits(chars: &[char], at: &mut usize, profile: SourceProfile) -> Result<bool> {
    let start = *at;
    while *at < chars.len() && decimal(chars[*at], profile)? {
        *at += 1;
    }
    Ok(*at > start)
}
fn version(chars: &[char], profile: SourceProfile) -> Result<bool> {
    let mut at = 0;
    if chars.first().is_some_and(|c| equal(*c, 'v')) {
        at = 1;
    } else if chars.len() >= 4
        && chars[..4]
            .iter()
            .zip("iter".chars())
            .all(|(a, b)| equal(*a, b))
    {
        at = 4;
    }
    if !digits(chars, &mut at, profile)? {
        return Ok(false);
    }
    while at < chars.len() {
        if chars[at] != '.' {
            return Ok(false);
        }
        at += 1;
        if !digits(chars, &mut at, profile)? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn quant_type(chars: &[char], profile: SourceProfile) -> Result<bool> {
    let mut at = 0;
    if chars.first().is_some_and(|c| equal(*c, 'i')) {
        at += 1;
    }
    if chars.get(at).is_some_and(|c| equal(*c, 'q')) {
        at += 1;
        if chars
            .get(at)
            .is_none_or(|c| decimal(*c, profile).ok() != Some(true))
        {
            return Ok(false);
        }
        at += 1;
        while at < chars.len() {
            if chars[at] != '_' {
                return Ok(false);
            }
            at += 1;
            let Some(ch) = chars.get(at) else {
                return Ok(false);
            };
            if *ch != '_' && py::flags(*ch as u32, profile)? & py::ALNUM == 0 {
                return Ok(false);
            }
            at += 1;
        }
        return Ok(true);
    }
    at = usize::from(chars.first().is_some_and(|c| equal(*c, 'b')));
    if chars.get(at).is_none_or(|c| !equal(*c, 'f')) {
        return Ok(false);
    }
    at += 1;
    if chars.get(at).is_some_and(|c| equal(*c, 'p')) {
        at += 1;
    }
    Ok(chars
        .get(at..)
        .is_some_and(|v| v == ['1', '6'] || v == ['3', '2']))
}
fn size_type(chars: &[char], value: &str, profile: SourceProfile) -> Result<bool> {
    if ["small", "mini", "medium", "large", "xl", "xxl"]
        .iter()
        .any(|w| word(value, w))
    {
        return Ok(true);
    }
    let mut at = 0;
    if chars.first().is_some_and(|c| equal(*c, 'a')) {
        at = 1;
    } else {
        let mut end = 0;
        if digits(chars, &mut end, profile)? && chars.get(end).is_some_and(|c| equal(*c, 'x')) {
            at = end + 1;
        }
    }
    if !digits(chars, &mut at, profile)? {
        return Ok(false);
    }
    if chars.get(at).is_some_and(|c| matches!(c, '.' | '_')) {
        at += 1;
        if !digits(chars, &mut at, profile)? {
            return Ok(false);
        }
    }
    if chars
        .get(at)
        .is_none_or(|c| !['k', 'm', 'b', 't'].iter().any(|s| equal(*c, *s)))
    {
        return Ok(false);
    }
    at += 1;
    if chars
        .get(at)
        .is_some_and(|c| decimal(*c, profile).ok() == Some(true))
    {
        at += 1;
    }
    Ok(at == chars.len())
}
fn chars(value: &str) -> Result<Vec<char>> {
    let mut result = Vec::new();
    reserve(&mut result, value.chars().count(), "model name codepoints")?;
    result.extend(value.chars());
    Ok(result)
}
fn numeric(value: &str, profile: SourceProfile) -> Result<Option<f64>> {
    let mut text = String::new();
    text.try_reserve_exact(value.len())
        .map_err(|_| Error::Limit("size label number"))?;
    for ch in value.chars() {
        if let Some(digit) = py::decimal(ch as u32, profile)? {
            text.push(char::from(b'0' + digit));
        } else if ch == '.' {
            text.push(ch);
        } else {
            return Ok(None);
        }
    }
    Ok(text.parse::<f64>().ok())
}
fn join(parts: &[(String, u8)], kind: u8, exclude: u8, unique: bool) -> Result<Option<String>> {
    let mut output = String::new();
    let mut seen: Vec<&str> = Vec::new();
    for (name, flags) in parts {
        if flags & kind == 0 || flags & exclude != 0 {
            continue;
        }
        if unique && seen.contains(&name.as_str()) {
            continue;
        }
        if unique {
            reserve(&mut seen, 1, "name size-label deduplication")?;
            seen.push(name);
        }
        let bytes = name.len() + usize::from(!output.is_empty());
        output
            .try_reserve(bytes)
            .map_err(|_| Error::Limit("model name parts"))?;
        if !output.is_empty() {
            output.push('-');
        }
        output.push_str(name);
    }
    Ok((!output.is_empty()).then_some(output))
}
pub(crate) fn components(
    model_id: &str,
    total_params: u64,
    profile: SourceProfile,
) -> Result<Parts> {
    for cp in model_id.chars() {
        py::flags(cp as u32, profile)?;
    }
    if model_id.contains(' ') {
        return Ok(Parts {
            name: string(model_id)?,
            organization: None,
            basename: None,
            finetune: None,
            version: None,
            size: None,
        });
    }
    let (organization, name) = match model_id.split_once('/') {
        Some((org, name)) => (
            if org.starts_with('.') {
                None
            } else {
                Some(string(org)?)
            },
            name,
        ),
        None => (None, model_id),
    };
    let mut parts = Vec::new();
    for value in name.split('-').filter(|s| !s.is_empty()) {
        if parts.len() >= 4096 {
            return Err(Error::Limit("model ID component count"));
        }
        reserve(&mut parts, 1, "model ID components")?;
        parts.push((string(value)?, 0u8));
    }
    for (index, (value, annotation)) in parts.iter_mut().enumerate() {
        let codes = chars(value)?;
        if version(&codes, profile)? {
            *annotation |= VERSION;
        } else if quant_type(&codes, profile)? {
            *annotation |= TYPE;
            *value = py::upper(value, profile)?;
        } else if index > 0 && size_type(&codes, value, profile)? {
            *value = value.replace('_', ".");
            let mut codes = chars(value)?;
            if let Some(last) = codes.pop_if(|c| decimal(*c, profile).ok() == Some(true)) {
                let suffix = codes.pop().expect("size has suffix");
                codes.push('.');
                codes.push(last);
                codes.push(suffix);
            }
            if codes.len() > 1 && decimal(codes[codes.len() - 2], profile)? {
                let last = codes.last_mut().expect("nonempty");
                if matches!(*last, 'k' | 'm' | 'b' | 't') {
                    *last = last.to_ascii_uppercase();
                }
            }
            value.clear();
            for ch in codes {
                value.push(ch);
            }
            if total_params != 0 {
                let (last_index, last) = value.char_indices().last().expect("size has text");
                if let Some(number) = numeric(&value[..last_index], profile)? {
                    let power = " KMBT".find(last).map_or(-1, |i| i as i32);
                    let label = number * 1000f64.powi(power);
                    let threshold = ((u128::from(total_params) * 7) / 8) as f64;
                    if (label - total_params as f64).abs() > threshold {
                        *annotation |= FINE;
                        if last.is_ascii() {
                            value.replace_range(
                                last_index..,
                                &last.to_ascii_lowercase().to_string(),
                            );
                        } else {
                            // Accepted size suffixes are ASCII or IGNORECASE's
                            // Kelvin sign; Python lower supplies that exact map.
                            if last == 'K' {
                                value.replace_range(last_index.., "k");
                            }
                        }
                    }
                }
            }
            if *annotation == 0 {
                *annotation |= SIZE;
            }
        } else if index > 0
            && ["chat", "instruct", "vision", "lora"]
                .iter()
                .any(|w| word(value, w))
        {
            *annotation |= FINE;
        }
    }
    let mut number_size = false;
    for (name, annotation) in &parts {
        if annotation & SIZE != 0 {
            for ch in name.chars() {
                number_size |= decimal(ch, profile)?;
            }
        }
    }
    if number_size {
        for (name, annotation) in &mut parts {
            if *annotation & SIZE != 0 && py::all(name, py::ALPHA, profile)? {
                *annotation &= !SIZE;
            }
        }
    }
    let mut at_start = true;
    for (name, annotation) in &mut parts {
        if at_start
            && ((*annotation == 0
                && py::flags(name.chars().next().expect("nonempty") as u32, profile)? & py::ALPHA
                    != 0)
                || *annotation & VERSION != 0)
        {
            *annotation |= BASE;
        } else {
            at_start = false;
            if *annotation == 0 {
                *annotation |= FINE;
            }
        }
    }
    for (_, annotation) in parts.iter_mut().rev() {
        if *annotation & BASE != 0 && annotation.count_ones() > 1 {
            *annotation &= !BASE;
        } else {
            break;
        }
    }
    let size = join(&parts, SIZE, 0, true)?;
    let finetune = join(&parts, FINE, 0, false)?;
    let version = join(&parts, VERSION, BASE, false)?;
    let basename = if size.is_none() && finetune.is_none() && version.is_none() {
        None
    } else {
        join(&parts, BASE, 0, false)?
    };
    Ok(Parts {
        name: string(name)?,
        organization,
        basename,
        finetune,
        version,
        size,
    })
}
pub(crate) fn title(value: &str, profile: SourceProfile) -> Result<String> {
    for cp in value.chars() {
        py::flags(cp as u32, profile)?;
    }
    let value = value.replace('-', " ");
    let mut output = String::new();
    output
        .try_reserve_exact(value.len())
        .map_err(|_| Error::Limit("model title"))?;
    let mut start = None;
    let mut words = Vec::new();
    for (index, ch) in value.char_indices() {
        if py::flags(ch as u32, profile)? & py::SPACE != 0 {
            if let Some(start) = start.take() {
                reserve(&mut words, 1, "title words")?;
                words.push(&value[start..index]);
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(start) = start {
        reserve(&mut words, 1, "title words")?;
        words.push(&value[start..]);
    }
    for word in words {
        let chars = chars(word)?;
        let numeric = chars
            .first()
            .is_some_and(|c| decimal(*c, profile).ok() == Some(true));
        let numeric_version = chars.first() == Some(&'v') && version(&chars, profile)?;
        let selected = if py::is_lower(word, profile)? && !numeric && !numeric_version {
            py::title_lower_word(word, profile)?
        } else {
            string(word)?
        };
        output
            .try_reserve(selected.len() + usize::from(!output.is_empty()))
            .map_err(|_| Error::Limit("model title bytes"))?;
        if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(&selected);
    }
    Ok(output)
}
