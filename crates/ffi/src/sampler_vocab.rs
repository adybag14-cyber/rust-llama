use crate::*;
use llama_behavior::sampling::MAX_CANDIDATES;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_char};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct SamplerVocab {
    pub pieces: Vec<Vec<u8>>,
    pub eog: Vec<bool>,
    pub fallback: i32,
}
impl SamplerVocab {
    pub fn piece(&self, id: i32) -> Result<&[u8]> {
        let index = usize::try_from(id).map_err(|_| "negative sampler token ID")?;
        self.pieces
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| "sampler token lies outside vocabulary".into())
    }
    pub fn is_eog(&self, id: i32) -> bool {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.eog.get(i))
            .copied()
            .unwrap_or(false)
    }
}
pub(crate) fn snapshot(vocab: &VocabHandle, special: bool) -> Result<Arc<SamplerVocab>> {
    let n = vocab.tokenizer.vocabulary_size();
    if n > MAX_CANDIDATES {
        return Err("sampler vocabulary exceeds one million tokens".into());
    }
    let mut pieces = Vec::with_capacity(n);
    let mut eog = Vec::with_capacity(n);
    let mut bytes = 0usize;
    for id in 0..n {
        let piece = vocab
            .tokenizer
            .token_piece(id as i32, special)
            .map_err(|e| e.to_string())?;
        bytes = bytes
            .checked_add(piece.len())
            .ok_or("sampler vocabulary size overflow")?;
        if bytes > 256 * 1024 * 1024 {
            return Err("sampler vocabulary pieces exceed 256 MiB".into());
        }
        pieces.push(piece);
        eog.push(vocab.tokenizer.is_end_of_generation(id as i32));
    }
    let fallback = if vocab.special["eot"] >= 0 {
        vocab.special["eot"]
    } else {
        vocab.special["eos"]
    };
    Ok(Arc::new(SamplerVocab {
        pieces,
        eog,
        fallback,
    }))
}

/// Preserve source byte truncation, overlapping token heads, and per-breaker
/// suffix tokenization. The returned values exclude the map's head token.
pub(crate) unsafe fn dry_breakers(
    vocab: *const VocabHandle,
    breakers: *const *const c_char,
    count: usize,
) -> Result<BTreeMap<i32, Vec<Vec<i32>>>> {
    let mut processed = BTreeMap::<i32, Vec<Vec<i32>>>::new();
    if breakers.is_null() || count == 0 {
        return Ok(processed);
    }
    let input = unsafe { input_slice(breakers, count)? };
    let vocab = unsafe { reference(vocab)? };
    let tokenizer = &vocab.tokenizer;
    if tokenizer.vocabulary_size() > MAX_CANDIDATES {
        return Err("DRY vocabulary exceeds token budget".into());
    }
    let mut words = Vec::with_capacity(tokenizer.vocabulary_size());
    let mut word_bytes = 0usize;
    for id in 0..tokenizer.vocabulary_size() {
        let word = tokenizer
            .detokenize_bytes(&[id as i32], false, true)
            .map_err(|e| e.to_string())?;
        word_bytes = word_bytes
            .checked_add(word.len())
            .ok_or("DRY vocabulary size overflow")?;
        if word_bytes > 256 * 1024 * 1024 {
            return Err("DRY vocabulary text exceeds 256 MiB".into());
        }
        words.push(word);
    }
    let mut seen = BTreeSet::new();
    let (mut entries, mut tail_tokens) = (0usize, 0usize);
    for &raw in input {
        if raw.is_null() {
            continue;
        }
        // Caller promises NUL termination; truncation is in bytes, not Unicode.
        let bytes = unsafe { CStr::from_ptr(raw) }.to_bytes();
        if bytes.is_empty() {
            continue;
        }
        let breaker = &bytes[..bytes.len().min(40)];
        if !seen.insert(breaker.to_vec()) {
            continue;
        }
        if seen.len() > 64 {
            break;
        }
        let mut cache = vec![None::<Vec<i32>>; breaker.len() + 1];
        for (id, word) in words.iter().enumerate() {
            let contains = word.windows(breaker.len()).any(|w| w == breaker);
            let mut tails = Vec::new();
            if contains {
                tails.push(Vec::new());
            } else {
                for pos in 0..word.len() {
                    if word[pos] != breaker[0] {
                        continue;
                    }
                    let mut i = 1;
                    while i < breaker.len() && pos + i < word.len() && word[pos + i] == breaker[i] {
                        i += 1;
                    }
                    if i < breaker.len() && pos + i < word.len() {
                        continue;
                    }
                    if cache[i].is_none() {
                        let mut tokens = tokenizer
                            .tokenize_bytes(&breaker[i..], false, false)
                            .map_err(|e| e.to_string())?;
                        tokens.truncate(20);
                        cache[i] = Some(tokens);
                    }
                    tails.push(cache[i].as_ref().expect("cached suffix").clone());
                }
            }
            if tails.is_empty() {
                continue;
            }
            let stored = processed.entry(id as i32).or_default();
            for tail in tails {
                if stored.contains(&tail) {
                    continue;
                }
                entries = entries
                    .checked_add(1)
                    .ok_or("DRY processed breaker overflow")?;
                tail_tokens = tail_tokens
                    .checked_add(tail.len())
                    .ok_or("DRY tail token count overflow")?;
                if entries > MAX_CANDIDATES || tail_tokens > 8 * MAX_CANDIDATES {
                    return Err("DRY processed breaker budget exceeded".into());
                }
                stored.push(tail);
            }
        }
    }
    Ok(processed)
}
