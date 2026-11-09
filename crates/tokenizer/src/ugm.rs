//! Source Unigram trie/Viterbi and its precompiled SentencePiece normalization.
use crate::{Error, Result, Token, Tokenizer, byte_trie::Trie, invalid, unicode};
use llama_gguf::Value;
use std::collections::BTreeMap;
const MAX_NORMALIZED_BYTES: usize = 16 * 1024 * 1024;
#[derive(Debug, Clone)]
pub(super) struct Ugm {
    tokens: Trie,
    user_defined: Trie,
    xcda: Vec<u32>,
    replacements: Vec<u8>,
    unknown_score: f32,
    nfc: bool,
    remove_extra_whitespaces: bool,
}
impl Ugm {
    pub(super) fn from_metadata(
        metadata: &BTreeMap<String, Value>,
        vocabulary: &[Token],
    ) -> Result<Self> {
        let mut tokens = Trie::new();
        let mut user_defined = Trie::new();
        let mut min_score = f32::MAX;
        for (id, token) in vocabulary.iter().enumerate() {
            if token.attributes & 4 != 0 {
                min_score = min_score.min(token.score);
            }
            if token.attributes & (4 | 16 | 2) != 0 {
                tokens.insert(token.text.as_bytes(), id as i32)?;
            }
            if token.attributes & 16 != 0 {
                user_defined.insert(token.text.as_bytes(), id as i32)?;
            }
        }
        let flag = |key: &str, default: bool| -> Result<bool> {
            metadata.get(key).map_or(Ok(default), |v| {
                v.as_bool()
                    .ok_or_else(|| invalid(format!("{key} is not BOOL")))
            })
        };
        let mut xcda = Vec::new();
        let mut replacements = Vec::new();
        if let Some(value) = metadata.get("tokenizer.ggml.precompiled_charsmap") {
            let Value::Array {
                element_type,
                values,
            } = value
            else {
                return Err(invalid("precompiled_charsmap is not an array"));
            };
            if !matches!(element_type, 0 | 1) {
                return Err(invalid("precompiled_charsmap must contain INT8 or UINT8"));
            }
            if values.len() > 64 * 1024 * 1024 {
                return Err(Error::Limit("precompiled_charsmap exceeds 64 MiB".into()));
            }
            let bytes: Vec<u8> = values
                .iter()
                .map(|value| match (*element_type, value) {
                    (0, Value::U8(byte)) => Ok(*byte),
                    (1, Value::I8(byte)) => Ok(*byte as u8),
                    _ => Err(invalid(
                        "precompiled_charsmap element type differs from array header",
                    )),
                })
                .collect::<Result<_>>()?;
            let header: [u8; 4] = bytes
                .get(..4)
                .ok_or_else(|| invalid("precompiled_charsmap missing length header"))?
                .try_into()
                .unwrap();
            let blob_size = u32::from_le_bytes(header) as usize;
            let end = 4usize
                .checked_add(blob_size)
                .ok_or_else(|| invalid("precompiled_charsmap length overflow"))?;
            if end >= bytes.len() {
                return Err(invalid(
                    "precompiled_charsmap replacement section is absent",
                ));
            }
            xcda = bytes[4..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            replacements.extend_from_slice(&bytes[end..]);
        }
        Ok(Self {
            tokens,
            user_defined,
            xcda,
            replacements,
            unknown_score: min_score - 10.0,
            nfc: flag("tokenizer.ggml.normalizer.nfc", false)?,
            remove_extra_whitespaces: flag("tokenizer.ggml.remove_extra_whitespaces", false)?,
        })
    }
    fn node(&self, index: u32) -> Result<u32> {
        self.xcda
            .get(index as usize)
            .copied()
            .ok_or_else(|| invalid("XCDA node index is outside charsmap"))
    }
    fn base(&self, index: u32) -> Result<u32> {
        let value = self.node(index)?;
        Ok((value >> 10) << ((value & (1 << 9)) >> 6))
    }
    fn normalized_prefix<'a>(&'a self, input: &'a [u8]) -> Result<(&'a [u8], usize)> {
        let protected = self.user_defined.longest_path(input);
        // The frozen helper returns the longest traversed prefix, even if it
        // stops before a terminal user-defined token.
        if protected > 0 {
            return Ok((&input[..protected], protected));
        }
        let mut longest = 0;
        let mut replacement = 0;
        if !self.xcda.is_empty() {
            let mut index = self.base(0)?;
            for (offset, &byte) in input.iter().enumerate() {
                if byte == 0 {
                    break;
                }
                index ^= u32::from(byte);
                let node = self.node(index)?;
                if node & ((1 << 31) | 255) != u32::from(byte) {
                    break;
                }
                let leaf = (node >> 8) & 1 != 0;
                index ^= self.base(index)?;
                if leaf {
                    longest = offset + 1;
                    replacement = (self.node(index)? & 0x7fff_ffff) as usize;
                }
            }
        }
        if longest > 0 {
            let bytes = self
                .replacements
                .get(replacement..)
                .ok_or_else(|| invalid("charsmap replacement is outside string section"))?;
            let end = bytes
                .iter()
                .position(|&byte| byte == 0)
                .ok_or_else(|| invalid("unterminated charsmap replacement"))?;
            return Ok((&bytes[..end], longest));
        }
        let mut offset = 0;
        if unicode::decode_next(input, &mut offset).is_ok() {
            Ok((&input[..offset], offset))
        } else {
            Ok(("\u{fffd}".as_bytes(), 1))
        }
    }
    pub(super) fn normalize(&self, input: &[u8], prepend: bool, nfc: bool) -> Result<Vec<u8>> {
        let canonical;
        let input = if nfc {
            canonical = unicode::normalize_nfc(&unicode::from_utf8(input))
                .into_iter()
                .try_fold(Vec::new(), |mut bytes, cp| {
                    bytes.extend(unicode::encode_utf8(cp).map_err(|e| invalid(e.to_string()))?);
                    Ok::<_, Error>(bytes)
                })?;
            canonical.as_slice()
        } else {
            input
        };
        let mut output = Vec::new();
        let mut offset = 0;
        let mut prepended = false;
        let mut non_whitespace = false;
        while offset < input.len() {
            let (normalized, consumed) = self.normalized_prefix(&input[offset..])?;
            for &byte in normalized {
                if output.len() > MAX_NORMALIZED_BYTES - 4 {
                    return Err(Error::Limit(
                        "Unigram normalized input exceeds 16 MiB".into(),
                    ));
                }
                if byte != b' ' {
                    if !non_whitespace {
                        non_whitespace = true;
                        if (prepend && !prepended) || self.remove_extra_whitespaces {
                            output.extend_from_slice("▁".as_bytes());
                            prepended = true;
                        }
                    }
                    output.push(byte);
                } else {
                    non_whitespace = false;
                    if !self.remove_extra_whitespaces {
                        output.extend_from_slice("▁".as_bytes());
                    }
                }
            }
            offset += consumed;
        }
        Ok(output)
    }
    pub(super) fn tokenize(
        &self,
        vocabulary: &Tokenizer,
        input: &[u8],
        output: &mut Vec<i32>,
    ) -> Result<()> {
        self.tokenize_with(vocabulary, input, vocabulary.unk_id, self.nfc, output)
    }
    pub(super) fn newline(&self, vocabulary: &Tokenizer) -> Result<i32> {
        let mut tokens = Vec::new();
        self.tokenize_with(vocabulary, b"\n", 2, false, &mut tokens)?;
        Ok(tokens.first().copied().unwrap_or(0))
    }
    fn tokenize_with(
        &self,
        vocabulary: &Tokenizer,
        input: &[u8],
        unknown: i32,
        nfc: bool,
        output: &mut Vec<i32>,
    ) -> Result<()> {
        let normalized = self.normalize(input, vocabulary.add_space_prefix, nfc)?;
        if normalized.is_empty() {
            return Ok(());
        }
        #[derive(Clone, Copy)]
        struct Best {
            token: i32,
            offset: usize,
            score: f64,
        }
        let mut best = vec![
            Best {
                token: unknown,
                offset: 0,
                score: -f64::MAX
            };
            normalized.len() + 1
        ];
        best[0].score = 0.0;
        let mut input_offset = 0;
        let mut work = 0usize;
        while input_offset < normalized.len() {
            let nbytes =
                unicode::len_utf8(normalized[input_offset]).min(normalized.len() - input_offset);
            let current = best[input_offset];
            let mut single_found = false;
            let mut node = 0;
            let mut prefix = input_offset;
            while prefix < normalized.len() {
                work += 1;
                if work > 128 * 1024 * 1024 {
                    return Err(Error::Limit("Unigram trie work budget exhausted".into()));
                }
                let Some(next) = self.tokens.next(node, normalized[prefix]) else {
                    break;
                };
                node = next;
                prefix += 1;
                if let Some(token_id) = self.tokens.value(node) {
                    if prefix - input_offset == nbytes {
                        single_found = true;
                    }
                    let token = &vocabulary.tokens[token_id as usize];
                    let score = current.score
                        + if token.attributes & 16 != 0 {
                            0.0
                        } else {
                            f64::from(token.score)
                        };
                    if score > best[prefix].score {
                        best[prefix] = Best {
                            token: token_id,
                            offset: input_offset,
                            score,
                        };
                    }
                }
            }
            if !single_found {
                let end = input_offset + nbytes;
                let score = current.score + f64::from(self.unknown_score);
                if score > best[end].score {
                    best[end] = Best {
                        token: unknown,
                        offset: input_offset,
                        score,
                    };
                }
            }
            input_offset += nbytes;
        }
        let start = output.len();
        let mut previous_unknown = false;
        let mut offset = normalized.len();
        loop {
            let step = best[offset];
            let is_unknown = step.token == unknown;
            if !(previous_unknown && is_unknown) {
                output.push(step.token);
            }
            if step.offset == 0 {
                break;
            }
            if step.offset >= offset {
                return Err(invalid("Unigram path failed to make progress"));
            }
            previous_unknown = is_unknown;
            offset = step.offset;
        }
        output[start..].reverse();
        Ok(())
    }
}
