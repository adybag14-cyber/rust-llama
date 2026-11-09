//! SentencePiece-style scored byte-pair tokenizer from frozen llama-vocab.cpp.
//! This is llama.cpp's SPM algorithm, not a replacement with a generic unigram
//! tokenizer. Other tokenizer families remain explicit unsupported work.
#![forbid(unsafe_code)]

use llama_gguf::Value;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    fmt,
};

pub const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
    Limit(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid tokenizer: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported tokenizer: {s}"),
            Self::Limit(s) => write!(f, "tokenizer limit: {s}"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

#[derive(Debug, Clone)]
struct Token {
    text: String,
    score: f32,
    kind: i32,
}
#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokens: Vec<Token>,
    lookup: BTreeMap<String, i32>,
    byte_ids: [Option<i32>; 256],
    special: Vec<i32>,
    pub bos_id: i32,
    pub eos_id: i32,
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_space_prefix: bool,
    eog: BTreeSet<i32>,
}
impl Tokenizer {
    pub fn from_metadata(m: &BTreeMap<String, Value>) -> Result<Self> {
        let family = m
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("missing tokenizer.ggml.model"))?;
        if family != "llama" {
            return Err(Error::Unsupported(format!(
                "family {family}; implemented family is llama/SPM"
            )));
        }
        let words = m
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("missing token array"))?;
        if words.is_empty() || words.len() > 1_000_000 {
            return Err(invalid("vocabulary size must be 1..=1000000"));
        }
        let scores = match m.get("tokenizer.ggml.scores") {
            Some(v) => Some(
                v.as_array()
                    .ok_or_else(|| invalid("scores is not an array"))?,
            ),
            None => None,
        };
        let kinds = match m.get("tokenizer.ggml.token_type") {
            Some(v) => Some(
                v.as_array()
                    .ok_or_else(|| invalid("token_type is not an array"))?,
            ),
            None => None,
        };
        if scores.is_some_and(|v| v.len() < words.len())
            || kinds.is_some_and(|v| v.len() < words.len())
        {
            return Err(invalid("scores/token types shorter than vocabulary"));
        }
        let mut tokens = Vec::with_capacity(words.len());
        let mut lookup = BTreeMap::new();
        let mut byte_ids = [None; 256];
        for (i, w) in words.iter().enumerate() {
            let mut text = w
                .as_str()
                .ok_or_else(|| invalid("non-string vocabulary entry"))?
                .to_owned();
            if text.is_empty() {
                text = format!("[EMPTY_{i}]")
            }
            let score = match scores {
                Some(s) => {
                    s[i].as_f64()
                        .or_else(|| s[i].as_i64().map(|x| x as f64))
                        .ok_or_else(|| invalid("invalid token score"))? as f32
                }
                None => 0.,
            };
            if !score.is_finite() {
                return Err(invalid("non-finite token score"));
            }
            let kind = match kinds {
                Some(k) => {
                    i32::try_from(k[i].as_i64().ok_or_else(|| invalid("invalid token type"))?)
                        .map_err(|_| invalid("token type exceeds i32"))?
                }
                None => 1,
            };
            if !(0..=6).contains(&kind) {
                return Err(invalid("token type outside 0..=6"));
            }
            if kind == 6 {
                let byte = parse_byte_token(&text)
                    .ok_or_else(|| invalid(format!("invalid byte token {text}")))?;
                byte_ids[byte as usize] = Some(i as i32);
            }
            if lookup.insert(text.clone(), i as i32).is_some() {
                return Err(invalid(format!("duplicate vocabulary token {text}")));
            }
            tokens.push(Token { text, score, kind });
        }
        let bool_value = |key: &str, default: bool| -> Result<bool> {
            match m.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| invalid(format!("{key} is not BOOL"))),
            }
        };
        let id = |key: &str, default: i32| -> Result<i32> {
            let id = match m.get(key) {
                None => default,
                Some(v) => i32::try_from(
                    v.as_i64()
                        .ok_or_else(|| invalid(format!("{key} is not integer")))?,
                )
                .map_err(|_| invalid("token ID exceeds i32"))?,
            };
            if id < 0 || id as usize >= tokens.len() {
                Err(invalid(format!("{key} is outside vocabulary")))
            } else {
                Ok(id)
            }
        };
        let bos_id = id("tokenizer.ggml.bos_token_id", 1)?;
        let eos_id = id("tokenizer.ggml.eos_token_id", 2)?;
        let add_bos = bool_value("tokenizer.ggml.add_bos_token", true)?;
        let add_eos = bool_value("tokenizer.ggml.add_eos_token", false)?;
        let add_space_prefix = bool_value("tokenizer.ggml.add_space_prefix", true)?;
        if bool_value("tokenizer.ggml.clean_spaces", false)? {
            return Err(Error::Unsupported("clean_spaces detokenization".into()));
        }
        if m.contains_key("tokenizer.ggml.precompiled_charsmap") {
            return Err(Error::Unsupported("precompiled normalization map".into()));
        }
        let mut eog = BTreeSet::from([eos_id]);
        for key in [
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
            "tokenizer.ggml.fim_pad_token_id",
            "tokenizer.ggml.fim_rep_token_id",
            "tokenizer.ggml.fim_sep_token_id",
        ] {
            if m.contains_key(key) {
                eog.insert(id(key, eos_id)?);
            }
        }
        for (i, t) in tokens.iter_mut().enumerate() {
            if matches!(
                t.text.as_str(),
                "<|eot_id|>"
                    | "<|eot|>"
                    | "<|im_end|>"
                    | "<|end|>"
                    | "<|return|>"
                    | "<|call|>"
                    | "<|flush|>"
                    | "<|calls|>"
                    | "<end_of_turn>"
                    | "<|endoftext|>"
                    | "</s>"
                    | "<|eom_id|>"
                    | "<EOT>"
                    | "_<EOT>"
                    | "[EOT]"
                    | "[EOS]"
                    | "<|end_of_text|>"
                    | "<end_of_utterance>"
                    | "<eos>"
                    | "<turn|>"
                    | "<|tool_response>"
                    | "<｜end▁of▁sentence｜>"
                    | "[e~["
                    | "<|ifm|im_end|>"
            ) {
                eog.insert(i as i32);
                t.kind = 3;
            }
            if matches!(
                t.text.as_str(),
                "<|channel|>" | "<|message|>" | "<|start|>" | "<|constrain|>"
            ) {
                t.kind = 4;
            }
        }
        if (lookup.contains_key("<|return|>") || lookup.contains_key("<|flush|>"))
            && (lookup.contains_key("<|call|>") || lookup.contains_key("<|calls|>"))
            && let Some(&id) = lookup.get("<|end|>")
        {
            eog.remove(&id);
            tokens[id as usize].kind = 4;
        }
        if (lookup.contains_key("<rfl:parameter name=\"")
            || lookup.contains_key("<|tool_response>")
            || lookup.contains_key("<|plamo:eos|>"))
            && let Some(&id) = lookup.get("</s>")
        {
            eog.remove(&id);
            tokens[id as usize].kind = 1;
        }
        let mut special: Vec<i32> = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t.kind, 2..=4))
            .map(|(i, _)| i as i32)
            .collect();
        special.sort_by(|&a, &b| {
            tokens[b as usize]
                .text
                .len()
                .cmp(&tokens[a as usize].text.len())
                .then(a.cmp(&b))
        });
        for (b, id) in byte_ids.iter_mut().enumerate() {
            if id.is_none() {
                *id = lookup.get(&format!("<0x{b:02X}>")).copied().or_else(|| {
                    std::str::from_utf8(&[b as u8])
                        .ok()
                        .and_then(|s| lookup.get(s).copied())
                });
            }
        }
        Ok(Self {
            tokens,
            lookup,
            byte_ids,
            special,
            bos_id,
            eos_id,
            add_bos,
            add_eos,
            add_space_prefix,
            eog,
        })
    }
    pub fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }
    /// Normalized vocabulary entry; kind uses frozen LLAMA_TOKEN_TYPE values.
    pub fn token_info(&self, id: i32) -> Option<(&str, f32, i32)> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .map(|t| (t.text.as_str(), t.score, t.kind))
    }
    pub fn is_end_of_generation(&self, id: i32) -> bool {
        self.eog.contains(&id)
    }
    /// Match llama_tokenize's independent add_special and parse_special flags.
    /// USER_DEFINED strings are partitioned even when parse_special is false.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        if text.len() > MAX_INPUT_BYTES {
            return Err(Error::Limit("input exceeds 4 MiB".into()));
        }
        enum Fragment<'a> {
            Text(&'a str),
            Token(i32),
        }
        let mut fragments = if text.is_empty() {
            vec![]
        } else {
            vec![Fragment::Text(text)]
        };
        for &id in &self.special {
            let token = &self.tokens[id as usize];
            if !parse_special && matches!(token.kind, 2 | 3) {
                continue;
            }
            let mut next = Vec::new();
            for fragment in fragments {
                match fragment {
                    Fragment::Token(id) => next.push(Fragment::Token(id)),
                    Fragment::Text(mut part) => {
                        while let Some(i) = part.find(&token.text) {
                            if i > 0 {
                                next.push(Fragment::Text(&part[..i]))
                            }
                            next.push(Fragment::Token(id));
                            part = &part[i + token.text.len()..];
                        }
                        if !part.is_empty() {
                            next.push(Fragment::Text(part))
                        }
                    }
                }
            }
            fragments = next;
        }
        let mut output = Vec::new();
        if add_special && self.add_bos {
            output.push(self.bos_id)
        }
        let mut previous_special = true;
        for fragment in fragments {
            match fragment {
                Fragment::Token(id) => {
                    output.push(id);
                    previous_special = true;
                }
                Fragment::Text(text) => {
                    let mut escaped = String::with_capacity(text.len());
                    if self.add_space_prefix && previous_special {
                        escaped.push('▁')
                    }
                    escaped.push_str(&text.replace(' ', "▁"));
                    self.spm(&escaped, &mut output)?;
                    previous_special = false;
                }
            }
        }
        if add_special && self.add_eos {
            output.push(self.eos_id)
        }
        Ok(output)
    }
    fn spm(&self, text: &str, out: &mut Vec<i32>) -> Result<()> {
        #[derive(Debug)]
        struct Symbol {
            start: usize,
            len: usize,
            prev: Option<usize>,
            next: Option<usize>,
        }
        let mut symbols: Vec<Symbol> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, c))| Symbol {
                start,
                len: c.len_utf8(),
                prev: i.checked_sub(1),
                next: Some(i + 1),
            })
            .collect();
        if symbols.is_empty() {
            return Ok(());
        }
        symbols.last_mut().unwrap().next = None;
        let mut heap = BinaryHeap::new();
        let add = |left: Option<usize>,
                   right: Option<usize>,
                   symbols: &[Symbol],
                   heap: &mut BinaryHeap<Bigram>| {
            if let (Some(l), Some(r)) = (left, right) {
                let len = symbols[l].len + symbols[r].len;
                let s = &text[symbols[l].start..symbols[l].start + len];
                if let Some(&id) = self.lookup.get(s) {
                    heap.push(Bigram {
                        left: l,
                        right: r,
                        len,
                        score: self.tokens[id as usize].score,
                    });
                }
            }
        };
        for i in 1..symbols.len() {
            add(Some(i - 1), Some(i), &symbols, &mut heap)
        }
        while let Some(b) = heap.pop() {
            if symbols[b.left].len == 0
                || symbols[b.right].len == 0
                || symbols[b.left].len + symbols[b.right].len != b.len
                || symbols[b.left].next != Some(b.right)
            {
                continue;
            }
            let next = symbols[b.right].next;
            symbols[b.left].len = b.len;
            symbols[b.left].next = next;
            symbols[b.right].len = 0;
            if let Some(n) = next {
                symbols[n].prev = Some(b.left)
            }
            add(symbols[b.left].prev, Some(b.left), &symbols, &mut heap);
            add(Some(b.left), next, &symbols, &mut heap);
        }
        let mut current = Some(0);
        while let Some(i) = current {
            let s = &symbols[i];
            let part = &text[s.start..s.start + s.len];
            if let Some(&id) = self.lookup.get(part) {
                out.push(id)
            } else {
                for &b in part.as_bytes() {
                    out.push(
                        self.byte_ids[b as usize]
                            .ok_or_else(|| invalid(format!("missing byte fallback 0x{b:02X}")))?,
                    )
                }
            }
            current = s.next;
        }
        Ok(())
    }
    pub fn token_piece(&self, id: i32, special: bool) -> Result<Vec<u8>> {
        let token = usize::try_from(id)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .ok_or_else(|| invalid(format!("token ID {id} is outside vocabulary")))?;
        Ok(match token.kind {
            2 | 3 if !special => vec![],
            2..=4 => token.text.as_bytes().to_vec(),
            1 => token.text.replace('▁', " ").into_bytes(),
            6 => {
                vec![parse_byte_token(&token.text).ok_or_else(|| invalid("malformed byte token"))?]
            }
            _ => vec![],
        })
    }
    /// Byte-exact detokenization, including byte tokens which can form partial UTF-8.
    pub fn detokenize_bytes(
        &self,
        tokens: &[i32],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>> {
        if tokens.len() > MAX_INPUT_BYTES {
            return Err(Error::Limit("token sequence too long".into()));
        }
        let mut tokens = tokens;
        let mut remove_space = self.add_space_prefix;
        if remove_special && self.add_bos && tokens.first() == Some(&self.bos_id) {
            remove_space = false;
            tokens = &tokens[1..]
        }
        if remove_special && self.add_eos && tokens.last() == Some(&self.eos_id) {
            tokens = &tokens[..tokens.len() - 1]
        }
        let mut result = Vec::new();
        for &id in tokens {
            let mut piece = self.token_piece(id, unparse_special)?;
            if remove_space && piece.first() == Some(&b' ') {
                piece.remove(0);
            }
            remove_space = false;
            if result
                .len()
                .checked_add(piece.len())
                .is_none_or(|n| n > MAX_INPUT_BYTES * 16)
            {
                return Err(Error::Limit("decoded output exceeds 64 MiB".into()));
            }
            result.extend(piece)
        }
        Ok(result)
    }
}
fn parse_byte_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}
#[derive(Debug, Clone, Copy)]
struct Bigram {
    left: usize,
    right: usize,
    len: usize,
    score: f32,
}
impl PartialEq for Bigram {
    fn eq(&self, b: &Self) -> bool {
        self.left == b.left && self.score == b.score
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Bigram {
    fn cmp(&self, b: &Self) -> Ordering {
        self.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.left.cmp(&self.left))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> Tokenizer {
        let mut m = BTreeMap::new();
        m.insert("tokenizer.ggml.model".into(), Value::String("llama".into()));
        let mut tokens = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
        let mut kinds = vec![2, 3, 3];
        for b in 0..256 {
            tokens.push(format!("<0x{b:02X}>"));
            kinds.push(6)
        }
        for t in ["▁", "a", "b", "c", "ab", "bc", "▁ab", "[user]"] {
            tokens.push(t.into());
            kinds.push(if t == "[user]" { 4 } else { 1 });
        }
        let mut scores = vec![-1.; tokens.len()];
        scores[263] = 2.;
        scores[264] = 2.;
        scores[265] = 3.;
        m.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array {
                element_type: 8,
                values: tokens.into_iter().map(Value::String).collect(),
            },
        );
        m.insert(
            "tokenizer.ggml.scores".into(),
            Value::Array {
                element_type: 6,
                values: scores.into_iter().map(Value::F32).collect(),
            },
        );
        m.insert(
            "tokenizer.ggml.token_type".into(),
            Value::Array {
                element_type: 5,
                values: kinds.into_iter().map(Value::I32).collect(),
            },
        );
        Tokenizer::from_metadata(&m).unwrap()
    }
    #[test]
    fn score_and_left_tie_break_match_spm() {
        assert_eq!(
            tokenizer().tokenize("abc", false, false).unwrap(),
            vec![265, 262]
        );
    }
    #[test]
    fn unknown_unicode_uses_exact_utf8_bytes() {
        let t = tokenizer();
        let ids = t.tokenize("界🦀", false, false).unwrap();
        assert_eq!(
            t.detokenize_bytes(&ids, false, false).unwrap(),
            "界🦀".as_bytes()
        );
    }
    #[test]
    fn special_flags_are_independent() {
        let t = tokenizer();
        assert_eq!(t.tokenize("", true, false).unwrap(), vec![1]);
        assert!(t.tokenize("", false, true).unwrap().is_empty());
        assert_eq!(t.tokenize("<s>a", false, true).unwrap(), vec![1, 259, 260]);
        assert_eq!(
            t.tokenize("[user]a", false, false).unwrap(),
            vec![266, 259, 260]
        );
        assert!(!t.tokenize("<s>", false, false).unwrap().contains(&1));
    }
    #[test]
    fn pieces_preserve_partial_utf8_and_reject_invalid_ids() {
        let t = tokenizer();
        assert_eq!(t.token_piece(3 + 0xe2, false).unwrap(), vec![0xe2]);
        assert!(t.token_piece(-1, false).is_err());
        assert_eq!(t.token_piece(1, false).unwrap(), Vec::<u8>::new());
        assert_eq!(t.token_piece(1, true).unwrap(), b"<s>");
    }
    #[test]
    fn bos_detokenize_follows_reference_space_policy() {
        let t = tokenizer();
        assert_eq!(
            t.detokenize_bytes(&[1, 259, 260], true, false).unwrap(),
            b" a"
        );
        assert_eq!(t.detokenize_bytes(&[259, 260], true, false).unwrap(), b"a");
    }
}
