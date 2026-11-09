//! Source-ranked BPE with the frozen metadata-selected pretokenizers.
use crate::{Error, Result, Tokenizer, find_bytes, invalid, replace_bytes, unicode, unicode_regex};
use llama_gguf::Value;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap},
    fmt::Write,
};

#[derive(Debug, Clone)]
pub(super) struct Config {
    pub pre: Pre,
    pub ignore_merges: bool,
    pub clean_spaces: bool,
    pub add_bos: bool,
    pub add_sep: bool,
    pub escape_whitespaces: bool,
    pub lowercase: bool,
    pub bos_null: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            pre: Pre::Default,
            ignore_merges: false,
            clean_spaces: true,
            add_bos: false,
            add_sep: false,
            escape_whitespaces: false,
            lowercase: true,
            bos_null: false,
        }
    }
}
include!("bpe_profiles.rs");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Model {
    Standard,
    Whitespace,
    HybridDna,
}
#[derive(Debug, Clone)]
pub(super) struct Bpe {
    pub config: Config,
    pub metaspace_first: bool,
    pub gemma_defaults: bool,
    pub hybrid_dna: bool,
    model: Model,
    dna_auto_tags: bool,
    patterns: Vec<unicode_regex::Pattern>,
    byte_encode: bool,
    ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
}
impl Bpe {
    pub(super) fn from_metadata(metadata: &BTreeMap<String, Value>, model: &str) -> Result<Self> {
        let pre = if model == "gemma4" {
            "gemma4"
        } else {
            match metadata.get("tokenizer.ggml.pre") {
                None => "",
                Some(value) => value
                    .as_str()
                    .ok_or_else(|| invalid("tokenizer.ggml.pre is not a string"))?
                    .split('\0')
                    .next()
                    .unwrap(),
            }
        };
        let config = Config::profile(pre)?;
        let (expressions, byte_encode) = config.pre.expressions();
        let patterns = expressions
            .iter()
            .map(|expression| unicode_regex::Pattern::compile(expression))
            .collect::<Result<_>>()?;
        let merges = match metadata.get("tokenizer.ggml.merges") {
            Some(Value::Array {
                element_type: 8,
                values,
            }) => values.as_slice(),
            Some(_) => return Err(invalid("tokenizer.ggml.merges is not a string array")),
            None if pre == "kimi-k2" => &[],
            None => return Err(invalid("missing tokenizer.ggml.merges")),
        };
        if merges.len() > 4_000_000 {
            return Err(Error::Limit("BPE merge count exceeds 4000000".into()));
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let text = merge
                .as_str()
                .ok_or_else(|| invalid("non-string BPE merge"))?
                .split('\0')
                .next()
                .unwrap()
                .as_bytes();
            // The original reader searches for the first ASCII space after byte
            // zero. Missing separators produce the empty pair; duplicates keep
            // their first rank.
            let pair = if let Some(position) = text
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(i, &b)| (b == b' ').then_some(i))
            {
                (text[..position].to_vec(), text[position + 1..].to_vec())
            } else {
                (Vec::new(), Vec::new())
            };
            ranks.entry(pair).or_insert(rank);
        }
        let metaspace_first =
            match metadata.get("tokenizer.ggml.metaspace.prepend_scheme") {
                None => false,
                Some(value) => match value.as_str().map(|text| text.split('\0').next().unwrap()) {
                    Some("never") => false,
                    Some("first") => true,
                    Some("always") => return Err(Error::Unsupported(
                        "BPE metaspace prepend_scheme always (also rejected by the frozen source)"
                            .into(),
                    )),
                    _ => return Err(invalid("unknown BPE metaspace prepend_scheme")),
                },
            };
        let dna_auto_tags = match metadata.get("tokenizer.ggml.dna_auto_tags") {
            None => false,
            Some(value) => value
                .as_bool()
                .ok_or_else(|| invalid("tokenizer.ggml.dna_auto_tags is not BOOL"))?,
        };
        Ok(Self {
            config,
            metaspace_first,
            gemma_defaults: model == "gemma4",
            hybrid_dna: model == "hybriddna",
            model: match model {
                "whitespace" => Model::Whitespace,
                "hybriddna" => Model::HybridDna,
                _ => Model::Standard,
            },
            dna_auto_tags,
            patterns,
            byte_encode,
            ranks,
        })
    }
    pub(super) fn newline(&self, vocab: &Tokenizer) -> i32 {
        if self.model == Model::Whitespace {
            return -1;
        }
        if self.config.pre == Pre::Soprano {
            return vocab.lookup.get(" ").copied().unwrap_or(-1);
        }
        let mut ids = Vec::new();
        self.plain(vocab, b"\n", &mut ids)
            .ok()
            .and_then(|()| ids.first().copied())
            .unwrap_or(-1)
    }
    pub(super) fn tokenize(
        &self,
        vocab: &Tokenizer,
        text: &[u8],
        out: &mut Vec<i32>,
    ) -> Result<()> {
        match self.model {
            Model::Standard => self.plain(vocab, text, out),
            Model::Whitespace => {
                let mut segment = Vec::new();
                for cp in unicode::from_utf8(text) {
                    if unicode::flags(cp).contains(unicode::Flags::WHITESPACE) {
                        if !segment.is_empty() {
                            self.plain(vocab, &segment, out)?;
                            segment.clear();
                        }
                    } else {
                        let cp = if vocab.lowercase {
                            unicode::lowercase(cp)
                        } else {
                            cp
                        };
                        segment
                            .extend(unicode::encode_utf8(cp).map_err(|e| invalid(e.to_string()))?);
                    }
                }
                if !segment.is_empty() {
                    self.plain(vocab, &segment, out)?;
                }
                Ok(())
            }
            Model::HybridDna => self.dna(vocab, text, out),
        }
    }
    fn plain(&self, vocab: &Tokenizer, text: &[u8], out: &mut Vec<i32>) -> Result<()> {
        let normalized;
        let text = if self.config.pre == Pre::Nemotron35Embd {
            normalized = replace_bytes(
                &replace_bytes(text, "[\u{200c}".as_bytes(), b"["),
                "<\u{200c}".as_bytes(),
                b"<",
            );
            normalized.as_slice()
        } else if self.config.pre == Pre::Soprano {
            let mut result = Vec::new();
            let mut space = false;
            for cp in unicode::from_utf8(text) {
                if unicode::flags(cp).contains(unicode::Flags::WHITESPACE) {
                    if !space {
                        result.push(b' ');
                    }
                    space = true;
                } else {
                    if cp == 0x130 {
                        result.extend("i\u{307}".as_bytes());
                    } else {
                        result.extend(
                            unicode::encode_utf8(unicode::lowercase(cp))
                                .map_err(|e| invalid(e.to_string()))?,
                        );
                    }
                    space = false;
                }
            }
            normalized = result;
            normalized.as_slice()
        } else {
            text
        };
        let mut words = unicode_regex::split(text, &self.patterns, self.byte_encode)?;
        if self.config.pre == Pre::Mmbert {
            let mut expanded = Vec::new();
            for word in words {
                if word.iter().all(|&b| b == b'\n') {
                    expanded.push(word);
                    continue;
                }
                let word = if word.starts_with("▁".as_bytes()) {
                    word
                } else {
                    let mut prefixed = "▁".as_bytes().to_vec();
                    prefixed.extend(word);
                    prefixed
                };
                let mut start = 0;
                while start < word.len() {
                    let end = find_bytes(&word[start + 3..], "▁".as_bytes())
                        .map_or(word.len(), |offset| start + 3 + offset);
                    expanded.push(word[start..end].to_vec());
                    start = end;
                }
            }
            words = expanded;
        }
        for word in words {
            self.merge(vocab, &word, out)?;
        }
        Ok(())
    }
    fn merge(&self, vocab: &Tokenizer, word: &[u8], out: &mut Vec<i32>) -> Result<()> {
        let direct = (self.config.ignore_merges
            || (matches!(self.config.pre, Pre::Gemma4 | Pre::Mmbert)
                && word.iter().all(|&b| b == b'\n')))
        .then(|| lookup(vocab, word))
        .flatten();
        if let Some(id) = direct {
            out.push(id);
            return Ok(());
        }
        if word.len() > 1 && word.contains(&b' ') {
            return Err(invalid(
                "unescaped ASCII space in a BPE merge symbol (rejected by the frozen source)",
            ));
        }
        #[derive(Debug)]
        struct Symbol {
            start: usize,
            len: usize,
            prev: Option<usize>,
            next: Option<usize>,
        }
        let mut symbols = Vec::new();
        let mut offset = 0;
        while offset < word.len() {
            let len = unicode::len_utf8(word[offset]).min(word.len() - offset);
            let index = symbols.len();
            symbols.push(Symbol {
                start: offset,
                len,
                prev: index.checked_sub(1),
                next: Some(index + 1),
            });
            offset += len;
        }
        let Some(last) = symbols.last_mut() else {
            return Ok(());
        };
        last.next = None;
        let mut heap = BinaryHeap::new();
        let add = |left: Option<usize>,
                   right: Option<usize>,
                   symbols: &[Symbol],
                   heap: &mut BinaryHeap<Pair>| {
            if let (Some(l), Some(r)) = (left, right) {
                let a = &word[symbols[l].start..symbols[l].start + symbols[l].len];
                let b = &word[symbols[r].start..symbols[r].start + symbols[r].len];
                if let Some(&rank) = self.ranks.get(&(a.to_vec(), b.to_vec())) {
                    heap.push(Pair {
                        left: l,
                        right: r,
                        len: a.len() + b.len(),
                        rank,
                    });
                }
            }
        };
        for i in 1..symbols.len() {
            add(Some(i - 1), Some(i), &symbols, &mut heap);
        }
        while let Some(pair) = heap.pop() {
            let left = &symbols[pair.left];
            let right = &symbols[pair.right];
            if left.len == 0
                || right.len == 0
                || left.len + right.len != pair.len
                || left.next != Some(pair.right)
            {
                continue;
            }
            let next = right.next;
            symbols[pair.left].len = pair.len;
            symbols[pair.left].next = next;
            symbols[pair.right].len = 0;
            if let Some(index) = next {
                symbols[index].prev = Some(pair.left);
            }
            add(
                symbols[pair.left].prev,
                Some(pair.left),
                &symbols,
                &mut heap,
            );
            add(Some(pair.left), next, &symbols, &mut heap);
        }
        let mut current = Some(0);
        while let Some(index) = current {
            let symbol = &symbols[index];
            let bytes = &word[symbol.start..symbol.start + symbol.len];
            if let Some(id) = lookup(vocab, bytes) {
                out.push(id);
            } else if self.config.pre == Pre::Soprano {
                if vocab.unk_id < 0 {
                    return Err(invalid("Soprano fallback requires UNK"));
                }
                out.push(vocab.unk_id);
            } else {
                for &byte in bytes {
                    let id = if self.byte_encode {
                        lookup(vocab, &[byte])
                    } else {
                        vocab.lookup.get(&format!("<0x{byte:02X}>")).copied()
                    };
                    // The source deliberately drops bytes absent from this
                    // fallback vocabulary, including encoded UTF-8 lead bytes.
                    if let Some(id) = id {
                        out.push(id);
                    }
                }
            }
            current = symbol.next;
        }
        Ok(())
    }
    pub(super) fn decode_piece(&self, text: &str) -> Vec<u8> {
        if self.config.pre == Pre::Soprano {
            return text.as_bytes().to_vec();
        }
        if self.config.escape_whitespaces {
            return text.replace('▁', " ").into_bytes();
        }
        let mut output = Vec::new();
        for cp in text.chars().map(u32::from) {
            if let Ok(byte) = unicode::codepoint_to_byte(cp) {
                output.push(byte);
            } else {
                let mut fallback = String::from("[UNK_BYTE_0x");
                for byte in unicode::encode_utf8(cp).expect("valid input scalar") {
                    write!(fallback, "{byte:02x}").unwrap();
                }
                fallback.push_str(text);
                fallback.push(']');
                output.extend(fallback.as_bytes());
            }
        }
        output
    }
    fn dna(&self, vocab: &Tokenizer, text: &[u8], out: &mut Vec<i32>) -> Result<()> {
        let (Some(&begin), Some(&end), Some(&unknown)) = (
            vocab.lookup.get("<dna>"),
            vocab.lookup.get("</dna>"),
            vocab.lookup.get("<oov>"),
        ) else {
            return self.plain(vocab, text, out);
        };
        let dna = Dna {
            begin,
            end,
            unknown,
            pad: if vocab.lookup.contains_key("ANNNNN\u{e000}") {
                'N'
            } else {
                'A'
            },
        };
        let mut position = 0;
        while position < text.len() {
            let start = find_bytes(&text[position..], b"<dna>").map(|offset| position + offset);
            let end = find_bytes(&text[position..], b"</dna>").map(|offset| position + offset);
            match (start, end) {
                (None, None) => {
                    self.dna_text(vocab, &dna, &text[position..], out)?;
                    break;
                }
                (None, Some(end)) => {
                    dna.tokenize(vocab, &text[position..end], false, true, out);
                    position = end + 6;
                }
                (Some(start), Some(end)) if end < start => {
                    dna.tokenize(vocab, &text[position..end], false, true, out);
                    position = end + 6;
                }
                (Some(start), end) => {
                    self.dna_text(vocab, &dna, &text[position..start], out)?;
                    let content = start + 5;
                    if let Some(end) = end {
                        dna.tokenize(vocab, &text[content..end], true, true, out);
                        position = end + 6;
                    } else {
                        dna.tokenize(vocab, &text[content..], true, false, out);
                        break;
                    }
                }
            }
        }
        Ok(())
    }
    fn dna_text(&self, vocab: &Tokenizer, dna: &Dna, raw: &[u8], out: &mut Vec<i32>) -> Result<()> {
        if !self.dna_auto_tags {
            return self.plain(vocab, raw, out);
        }
        let text = normalize_spaced_dna(raw);
        let bytes = text.as_slice();
        let mut previous = 0;
        let mut i = 0;
        while i < bytes.len() {
            if !iupac(bytes[i]) {
                i += 1;
                continue;
            }
            let mut j = i;
            let mut ambiguous = 0;
            while j < bytes.len() && iupac(bytes[j]) {
                ambiguous += usize::from(!acgt(bytes[j].to_ascii_uppercase()));
                j += 1;
            }
            let len = j - i;
            if (ambiguous == 0 && len >= 6) || (ambiguous > 0 && len >= 12 && 5 * ambiguous <= len)
            {
                self.plain(vocab, &text[previous..i], out)?;
                dna.tokenize(vocab, &text[i..j], true, true, out);
                previous = j;
            }
            i = j;
        }
        self.plain(vocab, &text[previous..], out)
    }
}
fn lookup(vocab: &Tokenizer, bytes: &[u8]) -> Option<i32> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|text| vocab.lookup.get(text))
        .copied()
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pair {
    left: usize,
    right: usize,
    len: usize,
    rank: usize,
}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}
struct Dna {
    begin: i32,
    end: i32,
    unknown: i32,
    pad: char,
}
impl Dna {
    fn tokenize(
        &self,
        vocab: &Tokenizer,
        text: &[u8],
        has_start: bool,
        has_end: bool,
        out: &mut Vec<i32>,
    ) {
        if has_start {
            out.push(self.begin);
        }
        let codepoints = unicode::from_utf8(text);
        let mut first = 0;
        let mut last = codepoints.len();
        while first < last && unicode::flags(codepoints[first]).contains(unicode::Flags::WHITESPACE)
        {
            first += 1;
        }
        while last > first
            && unicode::flags(codepoints[last - 1]).contains(unicode::Flags::WHITESPACE)
        {
            last -= 1;
        }
        for chunk in codepoints[first..last].chunks(6) {
            let mut kmer = String::new();
            let mut valid = true;
            for &cp in chunk {
                if cp > 127 || !acgt((cp as u8).to_ascii_uppercase()) {
                    valid = false;
                    break;
                }
                kmer.push(char::from((cp as u8).to_ascii_uppercase()));
            }
            if valid {
                while kmer.len() < 6 {
                    kmer.push(self.pad);
                }
                kmer.push('\u{e000}');
                out.push(vocab.lookup.get(&kmer).copied().unwrap_or(self.unknown));
            } else {
                out.push(self.unknown);
            }
        }
        if has_end {
            out.push(self.end);
        }
    }
}
fn acgt(byte: u8) -> bool {
    matches!(byte, b'A' | b'C' | b'G' | b'T')
}
fn iupac(byte: u8) -> bool {
    b"ACGTRYSWKMBDHVN".contains(&byte.to_ascii_uppercase())
}
fn normalize_spaced_dna(text: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(text.len());
    let mut position = 0;
    let blank = |byte| matches!(byte, b' ' | b'\t');
    while position < text.len() {
        if acgt(text[position]) && (position == 0 || !text[position - 1].is_ascii_alphabetic()) {
            let mut match_end = None;
            let mut chunk = position;
            let mut count = 1;
            loop {
                let mut end = chunk;
                while end < text.len() && end - chunk < 5 && acgt(text[end]) {
                    end += 1;
                }
                if end == chunk {
                    break;
                }
                if count >= 6 && (end == text.len() || !text[end].is_ascii_alphabetic()) {
                    match_end = Some(end);
                }
                let mut next = end;
                while next < text.len() && blank(text[next]) {
                    next += 1;
                }
                if next == end {
                    break;
                }
                chunk = next;
                count += 1;
            }
            if let Some(end) = match_end {
                let sequence: Vec<u8> = text[position..end]
                    .iter()
                    .copied()
                    .filter(|&b| !blank(b))
                    .collect();
                if sequence.len() >= 12 {
                    output.extend(sequence);
                } else {
                    output.extend_from_slice(&text[position..end]);
                }
                position = end;
                continue;
            }
        }
        output.push(text[position]);
        position += 1;
    }
    output
}
