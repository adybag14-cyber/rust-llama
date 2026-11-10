//! Frozen llama.cpp tokenizers: SPM, BPE, WordPiece, Unigram, RWKV,
//! PLaMo-2/3 and the source test family, plus no-vocabulary metadata.
//! Algorithms, Unicode properties, byte decoding, pretokenizer selection and
//! metadata normalization follow the frozen source under explicit resource bounds.
#![forbid(unsafe_code)]

mod bpe;
mod byte_trie;
mod plamo;
#[cfg(test)]
mod raw_name_reference;
mod rwkv;
mod ugm;
#[cfg(test)]
mod ugm_reference;
pub mod unicode;
mod unicode_regex;
mod vocab_order;

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
    attributes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum VocabType {
    None = 0,
    Spm = 1,
    Bpe = 2,
    Wpm = 3,
    Ugm = 4,
    Rwkv = 5,
    Plamo2 = 6,
    Test = 7,
    Plamo3 = 8,
}
#[derive(Debug, Clone)]
pub struct Tokenizer {
    family: VocabType,
    token_type_count: u32,
    suppressed_tokens: Vec<i32>,
    bpe: Option<bpe::Bpe>,
    ugm: Option<ugm::Ugm>,
    rwkv: Option<rwkv::Rwkv>,
    plamo: Option<plamo::Plamo>,
    max_token_len: usize,
    lowercase: bool,
    strip_accents: bool,
    pub clean_spaces: bool,
    tokens: Vec<Token>,
    lookup: BTreeMap<String, i32>,
    byte_ids: [Option<i32>; 256],
    special: Vec<i32>,
    pub bos_id: i32,
    pub eos_id: i32,
    pub eot_id: i32,
    pub eom_id: i32,
    pub fim_pre_id: i32,
    pub fim_suf_id: i32,
    pub fim_mid_id: i32,
    pub fim_pad_id: i32,
    pub fim_rep_id: i32,
    pub fim_sep_id: i32,
    pub unk_id: i32,
    pub sep_id: i32,
    pub pad_id: i32,
    pub mask_id: i32,
    newline_id: i32,
    pub add_sep: bool,
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_space_prefix: bool,
    eog: BTreeSet<i32>,
}
impl Tokenizer {
    fn without_vocabulary(metadata: &BTreeMap<String, Value>) -> Result<Self> {
        let key = metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .map(|architecture| format!("{architecture}.vocab_size"));
        let count = key
            .as_ref()
            .and_then(|key| metadata.get(key))
            .map(|value| match value {
                Value::U32(n) => Ok(u64::from(*n)),
                _ => Err(invalid("dummy vocabulary size is not UINT32")),
            })
            .transpose()?
            .unwrap_or(0);
        if count > 1_000_000 {
            return Err(Error::Limit(
                "dummy vocabulary exceeds 1000000 entries".into(),
            ));
        }
        Ok(Self {
            family: VocabType::None,
            token_type_count: metadata
                .get("tokenizer.ggml.token_type_count")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            suppressed_tokens: Vec::new(),
            bpe: None,
            ugm: None,
            rwkv: None,
            plamo: None,
            max_token_len: 0,
            lowercase: true,
            strip_accents: true,
            clean_spaces: false,
            tokens: vec![
                Token {
                    text: String::new(),
                    score: 0.0,
                    kind: 0,
                    attributes: 0
                };
                count as usize
            ],
            lookup: BTreeMap::new(),
            byte_ids: [None; 256],
            special: Vec::new(),
            bos_id: -1,
            eos_id: -1,
            eot_id: -1,
            eom_id: -1,
            unk_id: -1,
            sep_id: -1,
            pad_id: -1,
            mask_id: -1,
            newline_id: -1,
            fim_pre_id: -1,
            fim_suf_id: -1,
            fim_mid_id: -1,
            fim_pad_id: -1,
            fim_rep_id: -1,
            fim_sep_id: -1,
            add_sep: false,
            add_bos: false,
            add_eos: false,
            add_space_prefix: false,
            eog: BTreeSet::new(),
        })
    }
    pub fn from_metadata(m: &BTreeMap<String, Value>) -> Result<Self> {
        let model = m
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("missing tokenizer.ggml.model"))?
            .split('\0')
            .next()
            .unwrap();
        if let Some(value) = m.get("tokenizer.ggml.pre")
            && !matches!(value, Value::String(_))
        {
            return Err(invalid("tokenizer.ggml.pre is not STRING"));
        }
        if let Some(value) = m.get("tokenizer.ggml.token_type_count")
            && !matches!(value, Value::U32(_))
        {
            return Err(invalid("tokenizer.ggml.token_type_count is not UINT32"));
        }
        if matches!(model, "none" | "no_vocab") {
            return Self::without_vocabulary(m);
        }
        let family = match model {
            "llama" => VocabType::Spm,
            "bert" => VocabType::Wpm,
            "t5" => VocabType::Ugm,
            "rwkv" => VocabType::Rwkv,
            "plamo2" => VocabType::Plamo2,
            "plamo3" => VocabType::Plamo3,
            "test" => VocabType::Test,
            "gpt2" | "hybriddna" | "whitespace" | "gemma4" => VocabType::Bpe,
            other => {
                return Err(Error::Unsupported(format!(
                    "unknown tokenizer family {other}"
                )));
            }
        };
        let bpe = if family == VocabType::Bpe {
            Some(bpe::Bpe::from_metadata(m, model)?)
        } else {
            None
        };
        let words = typed_array(m, "tokenizer.ggml.tokens", &[8])?
            .ok_or_else(|| invalid("missing token array"))?;
        if words.is_empty() || words.len() > 1_000_000 {
            return Err(invalid("vocabulary size must be 1..=1000000"));
        }
        let scores = typed_array(m, "tokenizer.ggml.scores", &[5, 6])?;
        let kinds = typed_array(m, "tokenizer.ggml.token_type", &[5])?;
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
                .split('\0')
                .next()
                .unwrap()
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
            if family == VocabType::Spm && !score.is_finite() {
                return Err(invalid("non-finite SentencePiece merge score"));
            }
            let kind = match kinds {
                Some(k) => {
                    i32::try_from(k[i].as_i64().ok_or_else(|| invalid("invalid token type"))?)
                        .map_err(|_| invalid("token type exceeds i32"))?
                }
                None => 1,
            };
            // Unknown source token-type integers map to UNDEFINED attributes.
            let kind = if (0..=6).contains(&kind) { kind } else { 0 };
            if kind == 6 {
                match parse_byte_token(&text) {
                    Some(byte) => byte_ids[byte as usize] = Some(i as i32),
                    None if matches!(
                        family,
                        VocabType::Rwkv | VocabType::Test | VocabType::Plamo2 | VocabType::Plamo3
                    ) => {}
                    None => return Err(invalid(format!("invalid byte token {text}"))),
                }
            }
            if lookup.insert(text.clone(), i as i32).is_some() {
                return Err(invalid(format!("duplicate vocabulary token {text}")));
            }
            tokens.push(Token {
                text,
                score,
                kind,
                attributes: attributes_from_kind(kind),
            });
        }
        let ugm = if family == VocabType::Ugm {
            Some(ugm::Ugm::from_metadata(m, &tokens)?)
        } else {
            None
        };
        let rwkv = if family == VocabType::Rwkv {
            Some(rwkv::Rwkv::new(&tokens)?)
        } else {
            None
        };
        let plamo = if matches!(family, VocabType::Plamo2 | VocabType::Plamo3) {
            Some(plamo::Plamo::new(&tokens, family == VocabType::Plamo3)?)
        } else {
            None
        };
        let bool_value = |key: &str, default: bool| -> Result<bool> {
            match m.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| invalid(format!("{key} is not BOOL"))),
            }
        };
        let optional_id = |key: &str, default: i32| -> Result<i32> {
            let fallback = if default >= 0 && (default as usize) < tokens.len() {
                default
            } else {
                -1
            };
            let Some(value) = m.get(key) else {
                return Ok(fallback);
            };
            let Value::U32(value) = value else {
                return Err(invalid(format!("{key} is not UINT32")));
            };
            let value = *value;
            if value as u64 >= tokens.len() as u64 {
                // The frozen loader ignores out-of-range optional overrides
                // for every family, including the UINT32_MAX sentinel in
                // pretrained SPM metadata, and retains its source default.
                Ok(fallback)
            } else {
                Ok(value as i32)
            }
        };
        let bos_id = optional_id(
            "tokenizer.ggml.bos_token_id",
            match family {
                VocabType::Spm => 1,
                VocabType::None | VocabType::Ugm | VocabType::Rwkv | VocabType::Test => -1,
                VocabType::Plamo2 | VocabType::Plamo3 => 1,
                VocabType::Wpm => 101,
                VocabType::Bpe => {
                    if bpe.as_ref().unwrap().gemma_defaults || bpe.as_ref().unwrap().config.bos_null
                    {
                        -1
                    } else {
                        11
                    }
                }
            },
        )?;
        let eos_id = optional_id(
            "tokenizer.ggml.eos_token_id",
            match family {
                VocabType::Spm => 2,
                VocabType::Ugm => 1,
                VocabType::None | VocabType::Rwkv | VocabType::Test => -1,
                VocabType::Plamo2 | VocabType::Plamo3 => 2,
                VocabType::Wpm => -1,
                VocabType::Bpe => {
                    if bpe.as_ref().unwrap().gemma_defaults {
                        -1
                    } else {
                        11
                    }
                }
            },
        )?;
        let unk_id = optional_id(
            "tokenizer.ggml.unknown_token_id",
            match family {
                VocabType::Spm => 0,
                VocabType::Ugm => 2,
                VocabType::None | VocabType::Rwkv | VocabType::Test => -1,
                VocabType::Plamo2 | VocabType::Plamo3 => 0,
                VocabType::Wpm => 100,
                VocabType::Bpe => -1,
            },
        )?;
        let sep_id = optional_id(
            "tokenizer.ggml.seperator_token_id",
            if family == VocabType::Wpm { 102 } else { -1 },
        )?;
        let pad_id = optional_id(
            "tokenizer.ggml.padding_token_id",
            if matches!(family, VocabType::Wpm | VocabType::Ugm) {
                0
            } else if matches!(family, VocabType::Plamo2 | VocabType::Plamo3) {
                3
            } else {
                -1
            },
        )?;
        let mask_id = optional_id(
            "tokenizer.ggml.mask_token_id",
            if family == VocabType::Wpm { 103 } else { -1 },
        )?;
        let mut add_bos = bool_value(
            "tokenizer.ggml.add_bos_token",
            matches!(family, VocabType::Spm | VocabType::Wpm)
                || bpe.as_ref().is_some_and(|b| b.config.add_bos),
        )?;
        if bpe
            .as_ref()
            .is_some_and(|b| b.config.pre == bpe::Pre::Gemma4)
        {
            add_bos = true;
        }
        let add_eos = bool_value("tokenizer.ggml.add_eos_token", family == VocabType::Ugm)?;
        let add_sep = bool_value(
            "tokenizer.ggml.add_sep_token",
            family == VocabType::Wpm || bpe.as_ref().is_some_and(|b| b.config.add_sep),
        )?;
        let add_space_prefix =
            bool_value("tokenizer.ggml.add_space_prefix", family == VocabType::Spm)?;
        let clean_spaces = bool_value(
            "tokenizer.ggml.clean_spaces",
            family == VocabType::Wpm || bpe.as_ref().is_some_and(|b| b.config.clean_spaces),
        )?;
        let lowercase = bool_value(
            "tokenizer.ggml.normalizer.lowercase",
            bpe.as_ref().is_none_or(|b| b.config.lowercase),
        )?;
        let strip_accents = bool_value("tokenizer.ggml.normalizer.strip_accents", lowercase)?;
        // These keys are typed and validated by the source for every family,
        // including families whose tokenizer does not consume the setting.
        bool_value("tokenizer.ggml.normalizer.nfc", false)?;
        bool_value("tokenizer.ggml.remove_extra_whitespaces", false)?;
        bool_value("tokenizer.ggml.dna_auto_tags", false)?;
        if let Some(value) = m.get("tokenizer.ggml.metaspace.prepend_scheme") {
            match value.as_str().map(|text| text.split('\0').next().unwrap()) {
                Some("never" | "first" | "always") => {}
                Some(other) => {
                    return Err(invalid(format!("unknown metaspace prepend scheme {other}")));
                }
                None => return Err(invalid("metaspace prepend scheme is not STRING")),
            }
        }

        let token_type_count = m
            .get("tokenizer.ggml.token_type_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let suppressed_tokens = typed_array(m, "tokenizer.ggml.suppress_tokens", &[5])?
            .unwrap_or(&[])
            .iter()
            .filter_map(|value| {
                let id = value.as_i64().unwrap() as i32;
                (id >= 0 && (id as usize) < tokens.len()).then_some(id)
            })
            .collect();
        let mut native_order = None;
        let mut inferred_control = Vec::new();
        let mut resolve =
            |key: &str, deprecated: Option<&str>, spellings: &[&str]| -> Result<i32> {
                let configured = optional_id(key, -1)?;
                let configured = match deprecated {
                    Some(key) => optional_id(key, configured)?,
                    None => configured,
                };
                if configured >= 0 {
                    return Ok(configured);
                }
                if let Some(id) = vocab_order::first_matching(&lookup, spellings, &mut native_order)
                {
                    inferred_control.push(id);
                    Ok(id)
                } else {
                    Ok(-1)
                }
            };
        let eot_id = resolve(
            "tokenizer.ggml.eot_token_id",
            None,
            &[
                "<|eot_id|>",
                "<|im_end|>",
                "<|end|>",
                "<end_of_turn>",
                "<|endoftext|>",
                "<|end_of_text|>",
                "<EOT>",
                "_<EOT>",
                "[EOT]",
                "<｜end▁of▁sentence｜>",
                "<end_of_utterance>",
            ],
        )?;
        let eom_id = resolve("tokenizer.ggml.eom_token_id", None, &["<|eom_id|>"])?;
        let fim_pre_id = resolve(
            "tokenizer.ggml.fim_pre_token_id",
            Some("tokenizer.ggml.prefix_token_id"),
            &[
                "<|fim_prefix|>",
                "<fim-prefix>",
                "<fim_prefix>",
                "<｜fim▁begin｜>",
                "<PRE>",
                "▁<PRE>",
                "<|code_prefix|>",
                "<|prefix|>",
                "<|plamo:fim_prefix|>",
            ],
        )?;
        let fim_suf_id = resolve(
            "tokenizer.ggml.fim_suf_token_id",
            Some("tokenizer.ggml.suffix_token_id"),
            &[
                "<|fim_suffix|>",
                "<fim-suffix>",
                "<fim_suffix>",
                "<｜fim▁hole｜>",
                "<SUF>",
                "▁<SUF>",
                "<|code_suffix|>",
                "<|suffix|>",
                "<|plamo:fim_suffix|>",
            ],
        )?;
        let fim_mid_id = resolve(
            "tokenizer.ggml.fim_mid_token_id",
            Some("tokenizer.ggml.middle_token_id"),
            &[
                "<|fim_middle|>",
                "<fim-middle>",
                "<fim_middle>",
                "<｜fim▁end｜>",
                "<MID>",
                "▁<MID>",
                "<|code_middle|>",
                "<|middle|>",
                "<|plamo:fim_middle|>",
            ],
        )?;
        let fim_pad_id = resolve(
            "tokenizer.ggml.fim_pad_token_id",
            None,
            &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"],
        )?;
        let fim_rep_id = resolve(
            "tokenizer.ggml.fim_rep_token_id",
            None,
            &[
                "<|fim_repo|>",
                "<|repo_name|>",
                "<fim-repo>",
                "<REPO>",
                "<reponame>",
            ],
        )?;
        let fim_sep_id = resolve(
            "tokenizer.ggml.fim_sep_token_id",
            None,
            &["<|file_sep|>", "<|plamo:file_separator|>"],
        )?;
        let mut eog: BTreeSet<_> = [eos_id, eot_id, eom_id, fim_pad_id, fim_rep_id, fim_sep_id]
            .into_iter()
            .filter(|id| *id >= 0)
            .collect();
        for token in inferred_control {
            tokens[token as usize].attributes |= 8;
            tokens[token as usize].kind = 3;
        }
        for token in &mut tokens {
            if token.attributes & 8 != 0 && token.text.contains("unused") {
                token.attributes |= 2;
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
                t.attributes |= 8;
            }
            if matches!(
                t.text.as_str(),
                "<|channel|>" | "<|message|>" | "<|start|>" | "<|constrain|>"
            ) {
                t.kind = 4;
                t.attributes = 16;
            }
        }
        if (lookup.contains_key("<|return|>") || lookup.contains_key("<|flush|>"))
            && (lookup.contains_key("<|call|>") || lookup.contains_key("<|calls|>"))
            && let Some(&id) = lookup.get("<|end|>")
        {
            eog.remove(&id);
            tokens[id as usize].kind = 4;
            tokens[id as usize].attributes = 16;
        }
        if (lookup.contains_key("<rfl:parameter name=\"")
            || lookup.contains_key("<|tool_response>")
            || lookup.contains_key("<|plamo:eos|>"))
            && let Some(&id) = lookup.get("</s>")
        {
            eog.remove(&id);
            tokens[id as usize].kind = 1;
            tokens[id as usize].attributes = 4;
        }
        if bpe
            .as_ref()
            .is_some_and(|b| b.config.pre == bpe::Pre::Flint)
        {
            for text in ["<|user|>", "<|observation|>"] {
                if let Some(&id) = lookup.get(text) {
                    eog.insert(id);
                    tokens[id as usize].attributes |= 8;
                    tokens[id as usize].kind = 3;
                }
            }
        }
        {
            // The native loader retrieves a first-NUL byte string. Invalid
            // UTF-8 must not hide ASCII model-name markers. Match the C-locale
            // lowercasing without allocating or changing other bytes.
            let name = m
                .get("general.name")
                .and_then(Value::as_bytes)
                .unwrap_or_default()
                .split(|byte| *byte == 0)
                .next()
                .unwrap_or_default();
            let name_contains = |needle: &[u8]| {
                name.windows(needle.len())
                    .any(|part| part.eq_ignore_ascii_case(needle))
            };
            let pre = m
                .get("tokenizer.ggml.pre")
                .and_then(Value::as_str)
                .unwrap_or("");
            let arch = m
                .get("general.architecture")
                .and_then(Value::as_str)
                .unwrap_or("");
            if ["jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert"]
                .iter()
                .any(|x| pre.contains(x))
                || ["nomic-bert-moe", "jina-bert-v3"]
                    .iter()
                    .any(|x| arch.contains(x))
            {
                if let Some(&mask) = lookup.get("<mask>") {
                    tokens[mask as usize].attributes |= 128;
                }
            } else if name_contains(b"phi-3") || name_contains(b"phi3") {
                for token in &mut tokens {
                    if token.attributes & (1 | 8 | 16) != 0 || token.text == "</s>" {
                        token.attributes |= 256;
                    }
                }
                for text in ["<unk>", "<s>", "<|endoftext|>"] {
                    if let Some(&id) = lookup.get(text) {
                        tokens[id as usize].attributes &= !256;
                    }
                }
            } else if name_contains(b"modern-bert")
                && let Some(&mask) = lookup.get("[MASK]")
            {
                tokens[mask as usize].attributes |= 128;
            }
        }
        if bpe.as_ref().is_some_and(|b| b.hybrid_dna)
            && let Some(&oov) = lookup.get("<oov>")
        {
            for token in &mut tokens[oov as usize + 1..] {
                if token.text.len() > 3 && token.text.ends_with('\u{e000}') {
                    token.text.truncate(token.text.len() - 3);
                    if let Some(last) = token.text.rfind(|c| c != 'N') {
                        token.text.truncate(last + 1);
                    }
                }
            }
        }
        let max_token_len = tokens
            .iter()
            .map(|token| token.text.len())
            .max()
            .unwrap_or(0);
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
        let mut result = Self {
            family,
            token_type_count,
            suppressed_tokens,
            bpe,
            ugm,
            rwkv,
            plamo,
            max_token_len,
            lowercase,
            strip_accents,
            clean_spaces,
            unk_id,
            sep_id,
            pad_id,
            mask_id,
            newline_id: -1,
            add_sep,
            tokens,
            lookup,
            byte_ids,
            special,
            bos_id,
            eos_id,
            eot_id,
            eom_id,
            fim_pre_id,
            fim_suf_id,
            fim_mid_id,
            fim_pad_id,
            fim_rep_id,
            fim_sep_id,
            add_bos,
            add_eos,
            add_space_prefix,
            eog,
        };
        result.newline_id = match family {
            VocabType::None => -1,
            VocabType::Spm => result.byte_ids[10].unwrap_or(-1),
            VocabType::Wpm => 0,
            VocabType::Ugm => result.ugm.as_ref().unwrap().newline(&result)?,
            VocabType::Rwkv => result.rwkv.as_ref().unwrap().newline()?,
            VocabType::Plamo2 | VocabType::Plamo3 => result.plamo.as_ref().unwrap().newline()?,
            VocabType::Test => 10 % result.tokens.len() as i32,
            VocabType::Bpe => result.bpe.as_ref().unwrap().newline(&result),
        };
        Ok(result)
    }
    /// Source newline token, resolved before optional special-ID/normalizer overrides.
    pub fn newline_id(&self) -> i32 {
        self.newline_id
    }
    pub fn metaspace_prepend_first(&self) -> bool {
        self.bpe.as_ref().is_some_and(|bpe| bpe.metaspace_first)
    }
    /// Model-specific IDs in native order; invalid IDs were dropped on load.
    pub fn suppressed_tokens(&self) -> &[i32] {
        &self.suppressed_tokens
    }
    pub fn token_type_count(&self) -> u32 {
        self.token_type_count
    }
    pub fn vocab_type(&self) -> VocabType {
        self.family
    }
    pub fn token_attributes(&self, id: i32) -> Option<u32> {
        if self.family == VocabType::None {
            return None;
        }
        usize::try_from(id)
            .ok()
            .and_then(|index| self.tokens.get(index))
            .map(|token| token.attributes)
    }
    pub fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }
    /// Normalized vocabulary entry; kind uses frozen LLAMA_TOKEN_TYPE values.
    pub fn token_info(&self, id: i32) -> Option<(&str, f32, i32)> {
        if self.family == VocabType::None {
            return None;
        }
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
        self.tokenize_bytes(text.as_bytes(), add_special, parse_special)
    }
    /// Byte-preserving input matching the source C API, including malformed
    /// UTF-8 and byte-truncated DRY breakers. Special literals partition first.
    pub fn tokenize_bytes(
        &self,
        text: &[u8],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        if self.family == VocabType::None {
            return Err(Error::Unsupported("model has no tokenizer".into()));
        }
        if text.len() > MAX_INPUT_BYTES {
            return Err(Error::Limit("input exceeds 4 MiB".into()));
        }
        enum Fragment<'a> {
            Text(&'a [u8]),
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
                        while let Some(i) = find_bytes(part, token.text.as_bytes()) {
                            let mut left = &part[..i];
                            if token.attributes & 128 != 0 {
                                while left
                                    .last()
                                    .is_some_and(|&b| matches!(b, b' ' | b'\t'..=b'\r'))
                                {
                                    left = &left[..left.len() - 1];
                                }
                            }
                            if !left.is_empty() {
                                next.push(Fragment::Text(left));
                            }
                            next.push(Fragment::Token(id));
                            part = &part[i + token.text.len()..];
                            if token.attributes & 256 != 0 {
                                while part
                                    .first()
                                    .is_some_and(|&b| matches!(b, b' ' | b'\t'..=b'\r'))
                                {
                                    part = &part[1..];
                                }
                            }
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
        if add_special
            && !matches!(self.family, VocabType::Rwkv | VocabType::Test)
            && (self.family == VocabType::Wpm || self.add_bos)
        {
            if self.bos_id < 0 {
                return Err(invalid("add_special requires a valid BOS token"));
            }
            output.push(self.bos_id)
        }
        let mut previous_special = true;
        for (fragment_index, fragment) in fragments.into_iter().enumerate() {
            match fragment {
                Fragment::Token(id) => {
                    output.push(id);
                    previous_special = true;
                }
                Fragment::Text(text) => {
                    if let Some(plamo) = &self.plamo {
                        plamo.tokenize(text, &mut output)?;
                        continue;
                    }
                    if self.family == VocabType::Test {
                        for chunk in text.chunks(5) {
                            let hash = chunk.iter().fold(0u64, |hash, &byte| {
                                hash.wrapping_mul(31).wrapping_add(u64::from(byte))
                            });
                            output.push((hash % self.tokens.len() as u64) as i32);
                        }
                        continue;
                    }
                    if let Some(rwkv) = &self.rwkv {
                        rwkv.tokenize(text, self.unk_id, &mut output)?;
                        continue;
                    }
                    if let Some(ugm) = &self.ugm {
                        ugm.tokenize(self, text, &mut output)?;
                        continue;
                    }
                    if let Some(bpe) = &self.bpe {
                        let mut prepared = if bpe.config.escape_whitespaces {
                            escape_spaces(text)
                        } else {
                            text.to_vec()
                        };
                        if bpe.metaspace_first
                            && fragment_index == 0
                            && !prepared.starts_with("▁".as_bytes())
                        {
                            prepared.splice(0..0, "▁".bytes());
                        }
                        bpe.tokenize(self, &prepared, &mut output)?;
                        continue;
                    }
                    if self.family == VocabType::Wpm {
                        self.wordpiece(text, &mut output)?;
                        continue;
                    }
                    let mut escaped = Vec::with_capacity(text.len());
                    if self.add_space_prefix && previous_special {
                        escaped.extend("▁".as_bytes())
                    }
                    escaped.extend(escape_spaces(text));
                    self.spm(&escaped, &mut output)?;
                    previous_special = false;
                }
            }
        }
        if add_special && !matches!(self.family, VocabType::Rwkv | VocabType::Test) {
            if self.family == VocabType::Wpm {
                if self.sep_id < 0 {
                    return Err(invalid("add_special requires a valid SEP token"));
                }
                output.push(self.sep_id);
            } else if self.add_eos {
                if self.eos_id < 0 {
                    return Err(invalid("add_special requires a valid EOS token"));
                }
                output.push(self.eos_id);
            }
        }
        Ok(output)
    }
    fn wordpiece(&self, text: &[u8], out: &mut Vec<i32>) -> Result<()> {
        use unicode::Flags;
        let mut words = vec![Vec::new()];
        for original in unicode::from_utf8(text) {
            let cp = if self.strip_accents {
                unicode::nfd_base(original)
            } else {
                original
            };
            let flags = unicode::flags(cp);
            if flags.contains(Flags::WHITESPACE) {
                if !words.last().unwrap().is_empty() {
                    words.push(Vec::new());
                }
                continue;
            }
            if cp == 0 || cp == 0xfffd || flags.contains(Flags::CONTROL) {
                continue;
            }
            if self.strip_accents && flags.contains(Flags::ACCENT_MARK) {
                continue;
            }
            let cased = if self.lowercase {
                unicode::lowercase(cp)
            } else {
                cp
            };
            let character = unicode::encode_utf8(cased).map_err(|e| invalid(e.to_string()))?;
            if flags.contains(Flags::PUNCTUATION)
                || (cp < 0x7f && flags.contains(Flags::SYMBOL))
                || unicode::is_wordpiece_chinese(cp)
            {
                if !words.last().unwrap().is_empty() {
                    words.push(Vec::new());
                }
                words.last_mut().unwrap().extend(character);
                words.push(Vec::new());
            } else {
                words.last_mut().unwrap().extend(character);
            }
        }
        for word in words.into_iter().filter(|word| !word.is_empty()) {
            let mut prefixed = "▁".as_bytes().to_vec();
            prefixed.extend(word);
            let word = prefixed;
            let initial_len = out.len();
            let mut start = 0;
            while start < word.len() {
                let largest = word
                    .len()
                    .min(start.saturating_add(self.max_token_len).saturating_add(1));
                let found = (start + 1..=largest).rev().find_map(|end| {
                    word.get(start..end)
                        .and_then(|text| std::str::from_utf8(text).ok())
                        .and_then(|text| self.lookup.get(text))
                        .map(|&id| (end, id))
                });
                if let Some((end, id)) = found {
                    out.push(id);
                    start = end;
                } else {
                    out.truncate(initial_len);
                    break;
                }
            }
            if out.len() == initial_len {
                if self.unk_id < 0 {
                    return Err(invalid(
                        "unmatched WordPiece word requires a valid UNK token",
                    ));
                }
                out.push(self.unk_id);
            }
        }
        Ok(())
    }
    fn spm(&self, text: &[u8], out: &mut Vec<i32>) -> Result<()> {
        #[derive(Debug)]
        struct Symbol {
            start: usize,
            len: usize,
            prev: Option<usize>,
            next: Option<usize>,
        }
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let length = unicode::len_utf8(text[offset]).min(text.len() - offset);
            let index = symbols.len();
            symbols.push(Symbol {
                start: offset,
                len: length,
                prev: index.checked_sub(1),
                next: Some(index + 1),
            });
            offset += length;
        }
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
                if let Some(&id) = std::str::from_utf8(s).ok().and_then(|s| self.lookup.get(s)) {
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
            if let Some(&id) = std::str::from_utf8(part)
                .ok()
                .and_then(|s| self.lookup.get(s))
            {
                out.push(id)
            } else {
                for &b in part {
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
        if self.family == VocabType::None {
            return Err(Error::Unsupported("model has no token pieces".into()));
        }
        let token = usize::try_from(id)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .ok_or_else(|| invalid(format!("token ID {id} is outside vocabulary")))?;
        if !special && token.attributes & (1 | 8) != 0 {
            return Ok(Vec::new());
        }
        if self.family == VocabType::Rwkv {
            return Ok(rwkv::unescape(token.text.as_bytes()));
        }
        if matches!(self.family, VocabType::Plamo2 | VocabType::Plamo3) {
            return Ok(if token.attributes & 32 != 0 {
                parse_byte_token(&token.text)
                    .map_or_else(|| token.text.as_bytes().to_vec(), |byte| vec![byte])
            } else {
                token.text.as_bytes().to_vec()
            });
        }
        if self.family == VocabType::Test {
            return Ok(format!("{id:x}").into_bytes());
        }
        Ok(match token.kind {
            2 | 3 if !special => vec![],
            2..=4 => token.text.as_bytes().to_vec(),
            1 => {
                if let Some(bpe) = &self.bpe {
                    bpe.decode_piece(&token.text)
                } else {
                    token.text.replace('▁', " ").into_bytes()
                }
            }
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
        if self.family == VocabType::None {
            return Ok(Vec::new());
        }
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
        let mut remove_first_space = self.metaspace_prepend_first();
        for &id in tokens {
            let mut piece = self.token_piece(id, unparse_special)?;
            if (remove_space || remove_first_space) && piece.first() == Some(&b' ') {
                piece.remove(0);
            }
            remove_space = false;
            if unparse_special || self.token_attributes(id).unwrap_or(0) & (1 | 8) == 0 {
                remove_first_space = false;
            }
            if result
                .len()
                .checked_add(piece.len())
                .is_none_or(|n| n > MAX_INPUT_BYTES * 16)
            {
                return Err(Error::Limit("decoded output exceeds 64 MiB".into()));
            }
            result.extend(piece)
        }
        if self.clean_spaces {
            cleanup_spaces(&mut result);
        }
        Ok(result)
    }
}

fn typed_array<'a>(
    metadata: &'a BTreeMap<String, Value>,
    key: &str,
    types: &[u32],
) -> Result<Option<&'a [Value]>> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let Value::Array {
        element_type,
        values,
    } = value
    else {
        return Err(invalid(format!("{key} is not ARRAY")));
    };
    if !types.contains(element_type) || values.iter().any(|v| v.type_id() != *element_type) {
        return Err(invalid(format!("{key} has invalid array element type")));
    }
    Ok(Some(values))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if let (Ok(text), Ok(pattern)) = (std::str::from_utf8(haystack), std::str::from_utf8(needle)) {
        return text.find(pattern);
    }
    haystack
        .windows(needle.len())
        .position(|part| part == needle)
}
fn replace_bytes(input: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut input = input;
    while let Some(index) = find_bytes(input, from) {
        output.extend_from_slice(&input[..index]);
        output.extend_from_slice(to);
        input = &input[index + from.len()..];
    }
    output.extend_from_slice(input);
    output
}
fn escape_spaces(input: &[u8]) -> Vec<u8> {
    replace_bytes(input, b" ", "▁".as_bytes())
}

fn attributes_from_kind(kind: i32) -> u32 {
    match kind {
        1 => 4,
        2 => 1,
        3 => 8,
        4 => 16,
        5 => 2,
        6 => 32,
        _ => 0,
    }
}
// Preserve the source's three ordered in-place passes, including which
// apostrophe contractions intentionally keep their preceding space.
fn cleanup_spaces(text: &mut Vec<u8>) {
    let mut total = usize::from(!text.is_empty());
    for i in 1..text.len() {
        let x = text[i];
        if text[i - 1] == b' ' && matches!(x, b'?' | b'!' | b'.' | b',') {
            total -= 1;
        }
        text[total] = x;
        total += 1;
    }
    text.truncate(total);
    let length = text.len();
    total = usize::from(length != 0);
    let mut i = 1;
    while i < length {
        let x = text[i];
        if x == b'\'' && i + 1 < length && text[i - 1] == b' ' && text[i + 1] == b' ' {
            total -= 1;
            i += 1;
            text[i] = 0;
        }
        text[total] = x;
        total += 1;
        i += 1;
    }
    text.truncate(total);
    let length = text.len();
    total = usize::from(length != 0);
    for i in 1..length {
        let x = text[i];
        if text[i - 1] == b' ' && x == b'\'' && i + 1 < length {
            let x1 = text[i + 1];
            if matches!(x1, b's' | b'm')
                || (i + 2 < length
                    && ((x1 == b'r' && text[i + 2] == b'e') || (x1 == b'v' && text[i + 2] == b'e')))
            {
                total -= 1;
            }
        }
        text[total] = x;
        total += 1;
    }
    text.truncate(total);
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
