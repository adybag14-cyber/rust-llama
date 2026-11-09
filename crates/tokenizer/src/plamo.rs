//! Frozen PLaMo suffix automaton, integer-scored backward DP and PLaMo-3 cuts.
use crate::{Error, Result, Token, invalid, parse_byte_token, unicode};
use std::collections::BTreeMap;
const INVALID_SCORE: i32 = -20_000_000;
const UNKNOWN_SCORE: i32 = -10_000_000;
const MAX_ROWS: usize = 8 * 1024 * 1024;
#[derive(Debug, Clone, Copy)]
struct Row {
    length: usize,
    token: i32,
    score: i32,
    piece: usize,
}
#[derive(Debug, Clone)]
pub(super) struct Plamo {
    bytes: [i32; 256],
    transitions: BTreeMap<(u32, usize), usize>,
    rows: Vec<Row>,
    pre_segment: bool,
}
fn ends(text: &[u8]) -> Result<Vec<usize>> {
    let mut ends = Vec::new();
    let mut position = 0;
    while position < text.len() {
        unicode::decode_next(text, &mut position).map_err(|e| invalid(e.to_string()))?;
        ends.push(position);
    }
    Ok(ends)
}
impl Plamo {
    pub(super) fn new(vocabulary: &[Token], pre_segment: bool) -> Result<Self> {
        let mut bytes = [0; 256];
        let mut scores: BTreeMap<Vec<u8>, Option<f32>> = BTreeMap::new();
        let mut token_ids = BTreeMap::new();
        let mut suffix_bytes = 0usize;
        for (id, token) in vocabulary.iter().enumerate() {
            token_ids.insert(token.text.as_bytes().to_vec(), id as i32);
            if token.attributes & 32 != 0 {
                if let Some(byte) = parse_byte_token(&token.text) {
                    bytes[byte as usize] = id as i32;
                }
                continue;
            }
            let text = token.text.as_bytes();
            if !scores.contains_key(text) {
                suffix_bytes = suffix_bytes
                    .checked_add(text.len())
                    .ok_or_else(|| invalid("PLaMo suffix bytes overflow"))?;
            }
            if suffix_bytes > 256 * 1024 * 1024 || scores.len() >= MAX_ROWS {
                return Err(Error::Limit(
                    "PLaMo suffix dictionary exceeds bounded memory".into(),
                ));
            }
            scores.insert(text.to_vec(), Some(token.score));
            for start in ends(text)?
                .into_iter()
                .take_while(|&start| start < text.len())
            {
                let suffix = &text[start..];
                if !scores.contains_key(suffix) {
                    suffix_bytes = suffix_bytes
                        .checked_add(suffix.len())
                        .ok_or_else(|| invalid("PLaMo suffix bytes overflow"))?;
                    if suffix_bytes > 256 * 1024 * 1024 || scores.len() >= MAX_ROWS {
                        return Err(Error::Limit(
                            "PLaMo suffix dictionary exceeds bounded memory".into(),
                        ));
                    }
                    scores.insert(suffix.to_vec(), None);
                }
            }
        }
        if bytes.contains(&0) {
            return Err(invalid("PLaMo requires all 256 byte tokens at nonzero IDs"));
        }
        let mut suffixes: Vec<_> = scores.keys().cloned().collect();
        suffixes.push(Vec::new());
        suffixes.sort_by(|left, right| left.iter().rev().cmp(right.iter().rev()));
        let mut suffix_ids = BTreeMap::new();
        let mut transitions = BTreeMap::new();
        let mut row_count = 0usize;
        for suffix in &suffixes {
            suffix_ids.insert(suffix.clone(), row_count);
            if !suffix.is_empty() {
                let mut first_end = 0;
                let first = unicode::decode_next(suffix, &mut first_end)
                    .map_err(|e| invalid(e.to_string()))?;
                let remaining = *suffix_ids
                    .get(&suffix[first_end..])
                    .ok_or_else(|| invalid("PLaMo reverse suffix order is inconsistent"))?;
                transitions.insert((first, remaining), row_count);
                row_count += ends(suffix)?
                    .iter()
                    .filter(|&&end| scores.contains_key(&suffix[..end]))
                    .count();
            }
            row_count += 1;
            if row_count > MAX_ROWS {
                return Err(Error::Limit("PLaMo table exceeds 8 million rows".into()));
            }
        }
        let mut rows = Vec::with_capacity(row_count);
        for suffix in &suffixes {
            for (index, end) in ends(suffix)?.into_iter().enumerate().rev() {
                let piece = &suffix[..end];
                let Some(score) = scores.get(piece) else {
                    continue;
                };
                let score = match score {
                    None => INVALID_SCORE,
                    Some(score) if score.is_nan() => INVALID_SCORE,
                    Some(score) => {
                        // The original 1e4 literal promotes the source f32 to
                        // double before multiplying and rounding away from zero.
                        let rounded = (f64::from(*score) * 1e4).round();
                        if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
                            return Err(invalid("PLaMo scaled score exceeds i32"));
                        }
                        rounded as i32
                    }
                };
                rows.push(Row {
                    length: index + 1,
                    token: token_ids.get(piece).copied().unwrap_or(-1),
                    score,
                    piece: *suffix_ids
                        .get(piece)
                        .ok_or_else(|| invalid("PLaMo prefix lacks suffix ID"))?,
                });
            }
            rows.push(Row {
                length: 1,
                token: -1,
                score: UNKNOWN_SCORE,
                piece: 0,
            });
        }
        if rows.len() != row_count {
            return Err(invalid("PLaMo table row count differs from suffix plan"));
        }
        Ok(Self {
            bytes,
            transitions,
            rows,
            pre_segment,
        })
    }
    pub(super) fn tokenize(&self, input: &[u8], output: &mut Vec<i32>) -> Result<()> {
        let text = unicode::from_utf8(input);
        let mut work = 0usize;
        if !self.pre_segment {
            let text = if text.first() == Some(&0xfeff) {
                &text[1..]
            } else {
                &text
            };
            return self.encode(text, output, &mut work);
        }
        let mut cuts = vec![false; text.len() + 1];
        let prefix = [60, 124, 112, 108, 97, 109, 111, 58];
        let mut i = 0;
        while i + prefix.len() <= text.len() {
            if text[i..i + prefix.len()] != prefix {
                i += 1;
                continue;
            }
            let mut j = i + prefix.len();
            while j < text.len()
                && j - (i + prefix.len()) < 64
                && text[j] != 124
                && !(0x1c..=0x1f).contains(&text[j])
                && !unicode::flags(text[j]).contains(unicode::Flags::WHITESPACE)
            {
                j += 1;
            }
            if j + 1 < text.len() && text[j] == 124 && text[j + 1] == 62 {
                cuts[i] = true;
                cuts[j + 2] = true;
                i = j + 2;
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < text.len() {
            let cp = text[i];
            let mut run = 1;
            while i + run < text.len() && text[i + run] == cp && !cuts[i + run] {
                run += 1;
            }
            if (cp != 10 && run >= 4) || (cp == 32 && run >= 2) {
                cuts[i] = true;
                cuts[i + run] = true;
            }
            i += run;
        }
        let mut start = 0;
        for end in 0..=text.len() {
            let boundary = end < text.len() && text[end] == 0xee00;
            if end == text.len() || cuts[end] || boundary {
                if start < end {
                    self.encode(&text[start..end], output, &mut work)?;
                }
                start = end + usize::from(boundary);
            }
        }
        Ok(())
    }
    fn step(work: &mut usize) -> Result<()> {
        *work += 1;
        if *work > 128 * 1024 * 1024 {
            Err(Error::Limit("PLaMo table work budget exhausted".into()))
        } else {
            Ok(())
        }
    }
    fn encode(&self, text: &[u32], output: &mut Vec<i32>, work: &mut usize) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let mut costs = vec![1i64 << 60; text.len() + 1];
        costs[text.len()] = 0;
        let mut path = vec![(0usize, 0i32); text.len()];
        let mut suffix = 0;
        for i in (0..text.len()).rev() {
            let cp = text[i];
            for row in &self.rows[suffix..] {
                Self::step(work)?;
                suffix = self.transitions.get(&(cp, row.piece)).copied().unwrap_or(0);
                if suffix > 0 || row.score == UNKNOWN_SCORE {
                    break;
                }
            }
            for row in &self.rows[suffix..] {
                Self::step(work)?;
                if row.score > INVALID_SCORE {
                    let end = i
                        .checked_add(row.length)
                        .filter(|&end| end <= text.len())
                        .ok_or_else(|| invalid("PLaMo matched row exceeds input"))?;
                    let candidate = costs[end] - i64::from(row.score);
                    if candidate < costs[i] {
                        costs[i] = candidate;
                        path[i] = (row.length, row.token);
                    }
                }
                if row.score == UNKNOWN_SCORE {
                    break;
                }
            }
        }
        let mut position = 0;
        while position < text.len() {
            let (length, token) = path[position];
            if length == 0 {
                return Err(invalid("PLaMo path failed to make progress"));
            }
            if token >= 0 {
                output.push(token);
            } else {
                for byte in
                    unicode::encode_utf8(text[position]).map_err(|e| invalid(e.to_string()))?
                {
                    output.push(self.bytes[byte as usize]);
                }
            }
            position += length;
        }
        Ok(())
    }
    pub(super) fn newline(&self) -> Result<i32> {
        let mut output = Vec::new();
        self.tokenize(b"\n", &mut output)?;
        Ok(output.first().copied().unwrap_or(3))
    }
}
