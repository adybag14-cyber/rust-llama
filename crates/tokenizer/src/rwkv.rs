//! Frozen RWKV escaped-byte vocabulary and longest-prefix tokenizer.
use crate::{Result, Token, byte_trie::Trie, invalid};
#[derive(Debug, Clone)]
pub(super) struct Rwkv {
    trie: Trie,
}
pub(super) fn unescape(escaped: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut remaining = 0u8;
    let mut hex = 0u8;
    for &byte in escaped {
        if remaining != 0 {
            let signed = i16::from(byte as i8);
            let value = if signed >= i16::from(b'a') {
                signed - i16::from(b'a') + 10
            } else {
                signed - i16::from(b'0')
            } as u8;
            hex = hex.wrapping_shl(4).wrapping_add(value);
            remaining -= 1;
            if remaining == 0 {
                output.push(hex);
                hex = 0;
            }
        } else if escaping {
            match byte {
                b't' => output.push(b'\t'),
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b'x' => remaining = 2,
                byte => output.push(byte),
            }
            escaping = false;
        } else if byte == b'\\' {
            escaping = true;
        } else {
            output.push(byte);
        }
    }
    output
}
impl Rwkv {
    pub(super) fn new(vocabulary: &[Token]) -> Result<Self> {
        let mut trie = Trie::new();
        for (id, token) in vocabulary.iter().enumerate() {
            trie.insert(&unescape(token.text.as_bytes()), id as i32)?;
        }
        Ok(Self { trie })
    }
    pub(super) fn tokenize(&self, input: &[u8], unknown: i32, output: &mut Vec<i32>) -> Result<()> {
        let mut position = 0;
        while position < input.len() {
            let Some(mut node) = self.trie.next(0, input[position]) else {
                output.push(unknown);
                position += 1;
                continue;
            };
            let mut best = None;
            let mut end = position;
            loop {
                if let Some(id) = self.trie.value(node) {
                    best = Some((id, end + 1));
                }
                end += 1;
                if end == input.len() {
                    break;
                }
                match self.trie.next(node, input[end]) {
                    Some(next) => node = next,
                    None => break,
                }
            }
            let (id, end) = best.ok_or_else(|| {
                invalid(
                    "RWKV vocabulary has a nonterminal byte prefix; source would fail to advance",
                )
            })?;
            output.push(id);
            position = end;
        }
        Ok(())
    }
    pub(super) fn newline(&self) -> Result<i32> {
        let mut output = Vec::new();
        self.tokenize(b"\n", -1, &mut output)?;
        output
            .first()
            .copied()
            .ok_or_else(|| invalid("RWKV vocabulary has no newline token"))
    }
}
