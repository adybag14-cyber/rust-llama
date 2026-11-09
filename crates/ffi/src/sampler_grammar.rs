use crate::sampler_vocab::{self, SamplerVocab};
use crate::*;
use llama_behavior::native_grammar::NativeGrammar;
use llama_behavior::native_regex::NativeRegex;
use llama_behavior::sampling::{Candidates, MAX_CANDIDATES};
use std::ffi::{CStr, c_char};
use std::ops::Range;
use std::sync::Arc;

const MAX_TRIGGER_BUFFER: usize = 1024 * 1024;
const MAX_PATTERNS: usize = 128;
#[derive(Clone)]
struct Pattern {
    source: Vec<u8>,
    regex: NativeRegex,
}
impl Pattern {
    fn parse(source: Vec<u8>) -> Result<Self> {
        let regex = NativeRegex::parse(&source).map_err(|e| e.to_string())?;
        Ok(Self { source, regex })
    }
    fn find(&self, input: &[u8]) -> Result<Option<usize>> {
        let full = if self.source.first() == Some(&b'^') && self.source.last() == Some(&b'$') {
            self.regex.full_match(input).map_err(|e| e.to_string())?
        } else {
            None
        };
        let found = match full {
            Some(m) => Some(m),
            None => self.regex.find(input).map_err(|e| e.to_string())?,
        };
        Ok(found.map(|m| {
            m.captures
                .iter()
                .flatten()
                .find(|range| !range.is_empty())
                .map_or(m.range.start, |range| range.start)
        }))
    }
}
#[derive(Clone)]
pub(crate) struct GrammarState {
    engine: Option<NativeGrammar>,
    vocab: Option<Arc<SamplerVocab>>,
    lazy: bool,
    awaiting: bool,
    patterns: Vec<Pattern>,
    triggers: Vec<i32>,
    buffer: Vec<u8>,
    positions: Vec<(i32, Range<usize>)>,
}
impl GrammarState {
    fn empty() -> Self {
        Self {
            engine: None,
            vocab: None,
            lazy: false,
            awaiting: false,
            patterns: Vec::new(),
            triggers: Vec::new(),
            buffer: Vec::new(),
            positions: Vec::new(),
        }
    }
    pub fn unconstrained(&self) -> bool {
        self.engine.is_none() || self.awaiting
    }
    pub fn apply(&self, candidates: &mut Candidates) -> Result<()> {
        if self.unconstrained() {
            return Ok(());
        }
        let engine = self.engine.as_ref().expect("active engine");
        let vocab = self.vocab.as_ref().expect("active vocabulary");
        for token in &mut candidates.data {
            let piece = vocab.piece(token.id)?;
            if !engine
                .allows(token.id, piece, vocab.is_eog(token.id))
                .map_err(|e| e.to_string())?
            {
                token.logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
    pub fn accept(&mut self, token: i32) -> Result<()> {
        if self.engine.is_none() {
            return Ok(());
        }
        let mut next = self.clone();
        next.accept_inner(token)?;
        *self = next;
        Ok(())
    }
    fn accept_inner(&mut self, token: i32) -> Result<()> {
        let vocab = self.vocab.as_ref().expect("grammar vocabulary");
        let piece = vocab.piece(token)?;
        let engine = self.engine.as_mut().expect("grammar engine");
        if !self.awaiting {
            return engine
                .accept(token, piece, vocab.is_eog(token))
                .map_err(|e| e.to_string());
        }
        if self.triggers.contains(&token) {
            engine
                .accept(token, piece, false)
                .map_err(|e| e.to_string())?;
            self.awaiting = false;
            self.buffer.clear();
            self.positions.clear();
            return Ok(());
        }
        let start = self.buffer.len();
        let end = start
            .checked_add(piece.len())
            .ok_or("lazy grammar trigger buffer overflow")?;
        if end > MAX_TRIGGER_BUFFER || self.positions.len() >= MAX_CANDIDATES {
            return Err("lazy grammar trigger buffer exceeds its memory limit".into());
        }
        self.positions.push((token, start..end));
        self.buffer.extend_from_slice(piece);
        for pattern in &self.patterns {
            if let Some(start) = pattern.find(&self.buffer)? {
                for (token, range) in &self.positions {
                    if range.end <= start {
                        continue;
                    }
                    let from = range.start.max(start);
                    engine
                        .accept(*token, &self.buffer[from..range.end], false)
                        .map_err(|e| e.to_string())?;
                }
                self.awaiting = false;
                self.buffer.clear();
                self.positions.clear();
                return Ok(());
            }
        }
        Ok(())
    }
    pub fn reset(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.reset();
        }
        self.awaiting = self.lazy;
        self.buffer.clear();
        self.positions.clear();
    }
}

/// In frozen MSVC regex_replace, "$0" is literal text; the source's "\\$0"
/// replacement therefore produces "\\$0" for every metacharacter. Preserve the
/// measured result rather than silently repairing the original word escaping.
fn word_pattern(words: &[Vec<u8>]) -> Vec<u8> {
    let mut pattern = br"[\s\S]*?(".to_vec();
    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            pattern.push(b'|');
        }
        for &byte in word {
            if b".^$|()*+?[]{}\\".contains(&byte) {
                pattern.extend_from_slice(br"\$0");
            } else {
                pattern.push(byte);
            }
        }
    }
    pattern.extend_from_slice(br")[\s\S]*");
    pattern
}
unsafe fn strings(values: *const *const c_char, n: usize) -> Result<Vec<Vec<u8>>> {
    if n > MAX_PATTERNS {
        return Err("lazy grammar exceeds 128 trigger patterns".into());
    }
    let input = unsafe { input_slice(values, n)? };
    let mut out = Vec::with_capacity(n);
    let mut bytes = 0usize;
    for &pointer in input {
        if pointer.is_null() {
            return Err("null lazy grammar trigger string".into());
        }
        let value = unsafe { CStr::from_ptr(pointer) }.to_bytes();
        bytes = bytes
            .checked_add(value.len())
            .ok_or("lazy grammar pattern length overflow")?;
        if bytes > MAX_TRIGGER_BUFFER {
            return Err("lazy grammar patterns exceed one MiB".into());
        }
        out.push(value.to_vec());
    }
    Ok(out)
}
pub(crate) enum Triggers {
    None,
    Words {
        strings: *const *const c_char,
        count: usize,
        tokens: *const i32,
        token_count: usize,
    },
    Patterns {
        strings: *const *const c_char,
        count: usize,
        tokens: *const i32,
        token_count: usize,
    },
}
pub(crate) unsafe fn create(
    vocab: *const VocabHandle,
    source: *const c_char,
    root: *const c_char,
    triggers: Triggers,
) -> Result<GrammarState> {
    if source.is_null() || unsafe { CStr::from_ptr(source) }.to_bytes().is_empty() {
        return Ok(GrammarState::empty());
    }
    let source = unsafe { c_string(source)? };
    let root = unsafe { c_string(root)? };
    let vocab = unsafe { reference(vocab)? };
    let mut token_error = None;
    let engine = NativeGrammar::parse(source, root, |text| {
        match vocab.tokenizer.tokenize_bytes(text.as_bytes(), false, true) {
            Ok(tokens) if tokens.len() == 1 => Some(tokens[0]),
            Ok(_) => None,
            Err(error) => {
                token_error = Some(error.to_string());
                None
            }
        }
    })
    .map_err(|e| token_error.unwrap_or_else(|| e.to_string()))?;
    let snapshot = sampler_vocab::snapshot(vocab, true)?;
    let (lazy, patterns, tokens) = match triggers {
        Triggers::None => (false, Vec::new(), Vec::new()),
        Triggers::Words {
            strings: raw,
            count,
            tokens,
            token_count,
        } => {
            if token_count > MAX_CANDIDATES {
                return Err("lazy grammar trigger token budget exceeded".into());
            }
            let tokens = unsafe { input_slice(tokens, token_count)? }.to_vec();
            // Source only enters the legacy word branch for a non-null, nonempty list.
            let patterns = if !raw.is_null() && count > 0 {
                let words = unsafe { strings(raw, count)? };
                vec![Pattern::parse(word_pattern(&words))?]
            } else {
                Vec::new()
            };
            (true, patterns, tokens)
        }
        Triggers::Patterns {
            strings: raw,
            count,
            tokens,
            token_count,
        } => {
            if token_count > MAX_CANDIDATES {
                return Err("lazy grammar trigger token budget exceeded".into());
            }
            let tokens = unsafe { input_slice(tokens, token_count)? }.to_vec();
            let patterns = unsafe { strings(raw, count)? }
                .into_iter()
                .map(Pattern::parse)
                .collect::<Result<Vec<_>>>()?;
            (true, patterns, tokens)
        }
    };
    Ok(GrammarState {
        engine: Some(engine),
        vocab: Some(snapshot),
        lazy,
        awaiting: lazy,
        patterns,
        triggers: tokens,
        buffer: Vec::new(),
        positions: Vec::new(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_word_escaping_and_capture_start() {
        assert_eq!(
            word_pattern(&[b"a+b[c]".to_vec()]),
            br"[\s\S]*?(a\$0b\$0c\$0)[\s\S]*"
        );
        let p = Pattern::parse(br"prefix\s*(!)".to_vec()).unwrap();
        assert_eq!(p.find(b"prefix !hello").unwrap(), Some(7));
        let p = Pattern::parse(br"()(x)".to_vec()).unwrap();
        assert_eq!(p.find(b"abx").unwrap(), Some(2));
    }
}
