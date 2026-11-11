//! Deterministic HF Tokenizers BPE preparation used by the frozen converter.
//! Source: pinned Tokenizers 0.22.2 (Apache-2.0), with independently captured
//! Unicode properties. This is separate from llama.cpp's runtime pretokenizers.
#[path = "hf_nfc.rs"]
pub(crate) mod nfc;
use crate::hf_schema::Normalization;
use crate::hf_unicode as unicode;
use crate::util::{object, optional_text, text};
use crate::{Error, Limits, Result, reserve, string};
use llama_checkpoint::{Number, Object, Value as Json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Added<'a> {
    pub content: &'a str,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}
impl<'a> Added<'a> {
    pub fn plain(content: &'a str, special: bool) -> Self {
        Self {
            content,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: !special,
            special,
        }
    }
    pub fn from_json(value: &'a Json, serialized: bool, force_special: bool) -> Result<Self> {
        if !serialized && let Json::Text(_) = value {
            return Ok(Self::plain(text(value)?, force_special));
        }
        let value = object(value)?;
        let content = value
            .get("content")
            .map(text)
            .transpose()?
            .ok_or(Error::Invalid("added token content is absent"))?;
        let special = force_special || flag(value, "special", false, serialized)?;
        Ok(Self {
            content,
            single_word: flag(value, "single_word", false, serialized)?,
            lstrip: flag(value, "lstrip", false, serialized)?,
            rstrip: flag(value, "rstrip", false, serialized)?,
            normalized: flag(value, "normalized", !special, serialized)?,
            special,
        })
    }
}
fn flag(value: &Object, key: &str, default: bool, required: bool) -> Result<bool> {
    match value.get(key) {
        Some(Json::Bool(v)) => Ok(*v),
        None if !required => Ok(default),
        _ => Err(Error::Invalid(
            "tokenizer boolean field is absent or has the wrong type",
        )),
    }
}
fn model_flag(value: &Object, key: &str, default: bool) -> Result<bool> {
    match value.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(Json::Bool(v)) => Ok(*v),
        _ => Err(Error::Invalid("BPE boolean field has the wrong type")),
    }
}
pub(crate) fn id(value: &Json) -> Result<u32> {
    if let Json::Number(Number::Integer(value)) = value {
        value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(Error::Invalid("tokenizer ID is outside UINT32"))
    } else {
        Err(Error::Invalid("tokenizer ID is not an unsigned integer"))
    }
}
#[derive(Clone, Copy, Debug)]
enum Stage {
    Byte { prefix: bool, regex: bool },
    Digits { individual: bool },
    QwenSplit,
}
#[derive(Clone, Copy)]
struct Merge {
    left: u32,
    right: u32,
    rank: usize,
    result: u32,
}
struct Pattern<'a> {
    content: &'a str,
    normalized_content: Option<String>,
    normalized: bool,
    special: bool,
    id: u32,
}

pub(crate) struct Backend<'a> {
    pub base_size: usize,
    base: Vec<(&'a str, u32)>,
    pub by_id: Vec<Option<&'a str>>,
    pub added: Vec<(u32, Added<'a>)>,
    patterns: Vec<Pattern<'a>>,
    merges: Vec<Merge>,
    prefix: Option<&'a str>,
    suffix: Option<&'a str>,
    unk: Option<&'a str>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    no_merges: bool,
    stages: Vec<Stage>,
    direct_byte_level: bool,
    executable: bool,
    normalization: Normalization,
    byte_decoder: bool,
    post: crate::hf_schema::Post<'a>,
    pub split_special_tokens: bool,
    pub clean_up_spaces: bool,
    max_vocabulary: usize,
    max_bytes: usize,
    operations_left: usize,
}
impl<'a> Backend<'a> {
    pub fn from_json(raw: &'a Json, limits: &Limits) -> Result<Self> {
        let root = object(raw)?;
        if let Some(value) = root.get("version")
            && text(value)? != "1.0"
        {
            return Err(Error::Invalid("unknown tokenizer JSON version"));
        }
        let model = object(
            root.get("model")
                .ok_or(Error::Invalid("missing tokenizer model"))?,
        )?;
        if let Some(value) = model.get("type")
            && text(value)? != "BPE"
        {
            return Err(Error::Unsupported(
                "HF converter tokenizer model is not BPE",
            ));
        }
        let entries = object(
            model
                .get("vocab")
                .ok_or(Error::Invalid("missing BPE vocabulary"))?,
        )?;
        if entries.len() > limits.max_vocabulary {
            return Err(Error::Limit("HF base vocabulary count"));
        }
        let mut base = Vec::new();
        reserve(&mut base, entries.len(), "HF vocabulary descriptors")?;
        let mut maximum = 0usize;
        for (token, value) in entries.entries() {
            let token = token
                .as_str()
                .ok_or(Error::Invalid("HF token is not a Unicode scalar string"))?;
            let id = id(value)?;
            maximum = maximum.max(id as usize);
            if id as usize >= limits.max_vocabulary {
                return Err(Error::Limit("HF vocabulary ID"));
            }
            base.push((token, id));
        }
        base.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let base_size = base.len();
        let mut by_id = Vec::new();
        let count = if base_size == 0 { 0 } else { maximum + 1 };
        reserve(&mut by_id, count, "HF vocabulary reverse map")?;
        by_id.resize(count, None);
        for &(token, id) in &base {
            if by_id[id as usize].replace(token).is_some() {
                return Err(Error::Unsupported(
                    "duplicate HF base IDs have source hash-order-dependent reverse mappings",
                ));
            }
        }
        let prefix = optional_text(model.get("continuing_subword_prefix"))?;
        let suffix = optional_text(model.get("end_of_word_suffix"))?;
        let unk = optional_text(model.get("unk_token"))?;
        let byte_fallback = model_flag(model, "byte_fallback", false)?;
        let fuse_unk = model_flag(model, "fuse_unk", false)?;
        let ignore_merges = model_flag(model, "ignore_merges", false)?;
        let no_merges = match model.get("dropout") {
            None | Some(Json::Null) => false,
            Some(Json::Number(n)) => {
                let value = n.as_f64()?;
                if !(0.0..=1.0).contains(&value) {
                    return Err(Error::Invalid("invalid BPE dropout"));
                }
                if value != 0.0 && value != 1.0 {
                    return Err(Error::Unsupported("stochastic HF BPE dropout conversion"));
                }
                value == 1.0
            }
            _ => return Err(Error::Invalid("invalid BPE dropout type")),
        };
        let raw_merges = model
            .get("merges")
            .and_then(Json::as_array)
            .ok_or(Error::Invalid("missing BPE merges array"))?;
        if raw_merges.len() > limits.json.max_json_nodes {
            return Err(Error::Limit("HF merge count"));
        }
        let mut merges = Vec::new();
        reserve(&mut merges, raw_merges.len(), "HF merge descriptors")?;
        let lookup = |token: &str| {
            base.binary_search_by(|p| p.0.cmp(token))
                .ok()
                .map(|i| base[i].1)
        };
        for (rank, value) in raw_merges.iter().enumerate() {
            let (left, right) = if let Some(pair) = value.as_array() {
                if pair.len() != 2 {
                    return Err(Error::Invalid("BPE merge tuple must have two strings"));
                }
                (text(&pair[0])?, text(&pair[1])?)
            } else {
                let value = text(value)?;
                // Legacy parsing ignores the version-header line.
                if value.starts_with("#version") {
                    continue;
                }
                let mut fields = value.split(' ');
                let left = fields.next().unwrap_or("");
                let right = fields
                    .next()
                    .ok_or(Error::Invalid("invalid legacy BPE merge"))?;
                if fields.next().is_some() {
                    return Err(Error::Invalid("invalid legacy BPE merge"));
                }
                (left, right)
            };
            let left_id = lookup(left).ok_or(Error::Invalid("left merge token is absent"))?;
            let right_id = lookup(right).ok_or(Error::Invalid("right merge token is absent"))?;
            let tail = right
                .get(prefix.map_or(0, str::len)..)
                .ok_or(Error::Invalid(
                    "BPE continuing prefix splits a UTF-8 character or exceeds token",
                ))?;
            let length = left
                .len()
                .checked_add(tail.len())
                .filter(|n| *n <= limits.max_working_bytes)
                .ok_or(Error::Limit("HF merge text"))?;
            let mut joined = String::new();
            joined
                .try_reserve_exact(length)
                .map_err(|_| Error::Limit("HF merge text"))?;
            joined.push_str(left);
            joined.push_str(tail);
            let result = lookup(&joined).ok_or(Error::Invalid("merged token is absent"))?;
            merges.push(Merge {
                left: left_id,
                right: right_id,
                rank,
                result,
            });
        }
        // The original map keeps the last declaration for a repeated pair.
        merges.sort_unstable_by_key(|m| (m.left, m.right, m.rank));
        let mut unique: Vec<Merge> = Vec::new();
        reserve(&mut unique, merges.len(), "HF unique merges")?;
        for item in merges {
            if let Some(previous) = unique
                .last_mut()
                .filter(|p| p.left == item.left && p.right == item.right)
            {
                *previous = item;
            } else {
                unique.push(item);
            }
        }
        let mut stages = Vec::new();
        let normalization = crate::hf_schema::normalizer(root.get("normalizer"), 0)?;
        crate::hf_schema::pretokenizer(root.get("pre_tokenizer"), 0)?;
        let stage_execution = parse_stages(root.get("pre_tokenizer"), &mut stages, 0)?;
        // Fast constructors inspect the serialized root, not a flattened
        // sequence. A one-child Sequence must keep its saved child settings.
        let direct_byte_level = root
            .get("pre_tokenizer")
            .and_then(Json::as_object)
            .and_then(|stage| stage.get("type"))
            .map(text)
            .transpose()?
            == Some("ByteLevel");
        let executable = normalization != Normalization::Unsupported && stage_execution;
        let byte_decoder = crate::hf_schema::decoder(root.get("decoder"), 0)? == "ByteLevel";
        let post = crate::hf_schema::Post::parse(root.get("post_processor"), 0)?;
        crate::hf_schema::settings(root)?;
        let mut result = Self {
            base_size,
            base,
            by_id,
            added: Vec::new(),
            patterns: Vec::new(),
            merges: unique,
            prefix,
            suffix,
            unk,
            fuse_unk,
            byte_fallback,
            ignore_merges,
            no_merges,
            stages,
            direct_byte_level,
            executable,
            normalization,
            byte_decoder,
            post,
            split_special_tokens: false,
            clean_up_spaces: false,
            max_vocabulary: limits.max_vocabulary,
            max_bytes: limits.max_working_bytes,
            operations_left: limits.max_tokenizer_operations,
        };
        if let Some(values) = root.get("added_tokens") {
            let values = values
                .as_array()
                .ok_or(Error::Invalid("added_tokens must be an array"))?;
            let mut added = Vec::new();
            reserve(
                &mut added,
                values.len(),
                "serialized added token descriptors",
            )?;
            for value in values {
                let object = object(value)?;
                id(object
                    .get("id")
                    .ok_or(Error::Invalid("serialized added token ID is absent"))?)?;
                added.push(Added::from_json(value, true, false)?);
            }
            result.add_tokens(&added)?;
        }
        Ok(result)
    }
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .rev()
            .find(|(_, a)| a.content == token)
            .map(|(id, _)| *id)
            .or_else(|| {
                self.base
                    .binary_search_by(|v| v.0.cmp(token))
                    .ok()
                    .map(|i| self.base[i].1)
            })
    }
    pub fn add_tokens(&mut self, tokens: &[Added<'a>]) -> Result<()> {
        // Source special patterns precede ordinary patterns; their first
        // normalized/non-normalized membership persists across flag updates.
        for token in tokens {
            if token.special
                && !token.content.is_empty()
                && !self
                    .patterns
                    .iter()
                    .any(|p| p.special && p.content == token.content)
            {
                reserve(&mut self.patterns, 1, "special token patterns")?;
                self.patterns.push(Pattern {
                    content: token.content,
                    normalized_content: None,
                    normalized: token.normalized,
                    special: true,
                    id: 0,
                });
            }
        }
        let mut next = self
            .added
            .iter()
            .map(|p| p.0 as usize)
            .max()
            .map_or(self.base_size, |n| n.saturating_add(1).max(self.base_size));
        let mut existing = Vec::new();
        reserve(
            &mut existing,
            self.added.len() + tokens.len(),
            "added token update set",
        )?;
        existing.extend(self.added.iter().map(|p| p.1));
        for token in tokens {
            if token.content.is_empty() || existing.contains(token) {
                continue;
            }
            let id = if let Some(id) = self.token_id(token.content) {
                id
            } else {
                if next >= self.max_vocabulary {
                    return Err(Error::Limit("added-token ID"));
                }
                let id = next as u32;
                next += 1;
                id
            };
            let index = id as usize;
            if index >= self.by_id.len() {
                let growth = index + 1 - self.by_id.len();
                reserve(&mut self.by_id, growth, "added reverse map")?;
                self.by_id.resize(index + 1, None);
            }
            if self.by_id[index].is_some_and(|v| v != token.content) {
                return Err(Error::Unsupported(
                    "added ID collides with a noncontiguous base vocabulary",
                ));
            }
            self.by_id[index] = Some(token.content);
            if let Some(entry) = self.added.iter_mut().find(|p| p.0 == id) {
                entry.1 = *token;
            } else {
                reserve(&mut self.added, 1, "added token descriptors")?;
                self.added.push((id, *token));
            }
            if !self
                .patterns
                .iter()
                .any(|p| p.special && p.content == token.content)
            {
                reserve(&mut self.patterns, 1, "ordinary token patterns")?;
                self.patterns.push(Pattern {
                    content: token.content,
                    normalized_content: None,
                    normalized: token.normalized,
                    special: false,
                    id,
                });
            }
            existing.push(*token);
        }
        for index in 0..self.patterns.len() {
            let token = self.patterns[index].content;
            self.patterns[index].id = self
                .token_id(token)
                .ok_or(Error::Invalid("pattern has no added token ID"))?;
            if self.patterns[index].normalized && self.normalization == Normalization::Nfc {
                self.patterns[index].normalized_content = Some(nfc::normalize(
                    token,
                    self.max_bytes,
                    &mut self.operations_left,
                )?);
            }
        }
        self.added.sort_unstable_by_key(|p| p.0);
        Ok(())
    }
    pub fn override_direct_prefix(
        &mut self,
        config: Option<&Object>,
        is_llama: bool,
    ) -> Result<()> {
        let get = |key| config.and_then(|c| c.get(key));
        if is_llama && get("add_prefix_space").is_some_and(|v| !matches!(v, Json::Null)) {
            return Err(Error::Unsupported(
                "LlamaTokenizerFast explicit prefix selects slow SentencePiece reconstruction",
            ));
        }
        if !is_llama
            && self.direct_byte_level
            && self.stages.len() == 1
            && let Stage::Byte { prefix, .. } = &mut self.stages[0]
        {
            // The base fast class catches a failed pretokenizer rebuild.
            // Null/non-Boolean values preserve the saved stage. Llama passes
            // None by default, so its saved prefix is preserved as well.
            match get("add_prefix_space") {
                None => *prefix = false,
                Some(Json::Bool(v)) => *prefix = *v,
                _ => {}
            }
        }
        self.split_special_tokens = match get("split_special_tokens") {
            None => false,
            Some(Json::Bool(v)) => *v,
            _ => return Err(Error::Invalid("HF split_special_tokens type")),
        };
        self.clean_up_spaces = match get("clean_up_tokenization_spaces") {
            None => false,
            Some(Json::Bool(v)) => *v,
            _ => return Err(Error::Invalid("HF cleanup flag type")),
        };
        Ok(())
    }
    fn charge(&mut self, count: usize) -> Result<()> {
        self.operations_left = self
            .operations_left
            .checked_sub(count)
            .ok_or(Error::Limit("HF tokenizer work budget"))?;
        Ok(())
    }
    fn check_bytes(&self, count: usize) -> Result<()> {
        if count > self.max_bytes {
            Err(Error::Limit("HF tokenizer working bytes"))
        } else {
            Ok(())
        }
    }
    fn merge(&self, left: u32, right: u32) -> Option<Merge> {
        self.merges
            .binary_search_by_key(&(left, right), |m| (m.left, m.right))
            .ok()
            .map(|i| self.merges[i])
    }
    fn bpe(&mut self, input: &str, output: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }
        if self.ignore_merges
            && let Some(id) = self
                .base
                .binary_search_by(|v| v.0.cmp(input))
                .ok()
                .map(|i| self.base[i].1)
        {
            reserve(output, 1, "BPE output IDs")?;
            output.push(id);
            return Ok(());
        }
        let count = input.chars().count();
        self.charge(input.len())?;
        let mut symbols = Vec::new();
        reserve(&mut symbols, input.len(), "BPE symbols")?;
        let mut pending = None;
        for (index, ch) in input.chars().enumerate() {
            let mut local = [0u8; 4];
            let scalar = ch.encode_utf8(&mut local);
            let prefix = if index > 0 {
                self.prefix.unwrap_or("")
            } else {
                ""
            };
            let suffix = if index + 1 == count {
                self.suffix.unwrap_or("")
            } else {
                ""
            };
            let bytes = prefix
                .len()
                .checked_add(scalar.len())
                .and_then(|n| n.checked_add(suffix.len()))
                .ok_or(Error::Limit("BPE symbol text"))?;
            self.check_bytes(bytes)?;
            let mut symbol = String::new();
            symbol
                .try_reserve_exact(bytes)
                .map_err(|_| Error::Limit("BPE symbol text"))?;
            symbol.push_str(prefix);
            symbol.push_str(scalar);
            symbol.push_str(suffix);
            if let Some(id) = self
                .base
                .binary_search_by(|v| v.0.cmp(&symbol))
                .ok()
                .map(|i| self.base[i].1)
            {
                if let Some(id) = pending.take() {
                    symbols.push(id);
                }
                symbols.push(id);
            } else {
                let mut fallback = Vec::new();
                if self.byte_fallback {
                    reserve(&mut fallback, symbol.len(), "BPE byte fallback")?;
                    for byte in symbol.bytes() {
                        let hex = b"0123456789ABCDEF";
                        let name = [
                            b'<',
                            b'0',
                            b'x',
                            hex[(byte >> 4) as usize],
                            hex[(byte & 15) as usize],
                            b'>',
                        ];
                        let name = std::str::from_utf8(&name).expect("ASCII byte token");
                        let Some(id) = self
                            .base
                            .binary_search_by(|v| v.0.cmp(name))
                            .ok()
                            .map(|i| self.base[i].1)
                        else {
                            fallback.clear();
                            break;
                        };
                        fallback.push(id);
                    }
                    if !fallback.is_empty() {
                        reserve(&mut symbols, fallback.len(), "fallback symbols")?;
                        symbols.extend(fallback);
                        continue;
                    }
                }
                if let Some(unknown) = self.unk {
                    let id = self
                        .base
                        .binary_search_by(|v| v.0.cmp(unknown))
                        .ok()
                        .map(|i| self.base[i].1)
                        .ok_or(Error::Invalid(
                            "unknown token is absent from BPE vocabulary",
                        ))?;
                    if !self.fuse_unk {
                        if let Some(previous) = pending.replace(id) {
                            symbols.push(previous);
                        }
                    } else {
                        pending = Some(id);
                    }
                }
            }
        }
        if let Some(id) = pending {
            symbols.push(id);
        }
        if !self.no_merges {
            loop {
                self.charge(symbols.len())?;
                let mut selected: Option<(usize, Merge)> = None;
                for (at, pair) in symbols.windows(2).enumerate() {
                    if let Some(candidate) = self.merge(pair[0], pair[1])
                        && selected.is_none_or(|(_, old)| candidate.rank < old.rank)
                    {
                        selected = Some((at, candidate));
                    }
                }
                let Some((at, item)) = selected else {
                    break;
                };
                symbols[at] = item.result;
                symbols.remove(at + 1);
            }
        }
        reserve(output, symbols.len(), "BPE output IDs")?;
        output.extend(symbols);
        Ok(())
    }
    pub fn pieces(&mut self, text: &str) -> Result<Vec<String>> {
        if !self.executable {
            return Err(Error::Unsupported(
                "HF normalization/pretokenization pipeline has no verified converter execution",
            ));
        }
        self.check_bytes(text.len())?;
        self.charge(text.len())?;
        let mut pieces = Vec::new();
        if !text.is_empty() {
            reserve(&mut pieces, 1, "pretokenized fragments")?;
            pieces.push(string(text)?);
        }
        for stage_index in 0..self.stages.len() {
            let stage = self.stages[stage_index];
            let mut next = Vec::new();
            for piece in pieces {
                match stage {
                    Stage::QwenSplit => {
                        self.charge(piece.len())?;
                        let mut start = 0;
                        while start < piece.len() {
                            let end = qwen_regex_end(&piece, start);
                            self.push_piece(&mut next, &piece[start..end])?;
                            start = end;
                        }
                    }
                    Stage::Digits { individual } => {
                        let mut start = 0;
                        let mut prior = None;
                        for (index, ch) in piece.char_indices() {
                            let digit = unicode::flags(ch as u32) & unicode::DIGIT != 0;
                            let split =
                                index > start && (prior != Some(digit) || (digit && individual));
                            if split {
                                self.push_piece(&mut next, &piece[start..index])?;
                                start = index;
                            }
                            prior = Some(digit);
                        }
                        if start < piece.len() {
                            self.push_piece(&mut next, &piece[start..])?;
                        }
                    }
                    Stage::Byte { prefix, regex } => {
                        let mut piece = piece;
                        if prefix && !piece.starts_with(' ') {
                            self.check_bytes(
                                piece
                                    .len()
                                    .checked_add(1)
                                    .ok_or(Error::Limit("prefix space"))?,
                            )?;
                            piece
                                .try_reserve(1)
                                .map_err(|_| Error::Limit("prefix space"))?;
                            piece.insert(0, ' ');
                        }
                        if regex {
                            let mut start = 0;
                            while start < piece.len() {
                                let end = byte_regex_end(&piece, start);
                                self.push_byte_piece(&mut next, &piece[start..end])?;
                                start = end;
                            }
                        } else {
                            self.push_byte_piece(&mut next, &piece)?;
                        }
                    }
                }
            }
            pieces = next;
        }
        Ok(pieces)
    }
    fn push_piece(&mut self, pieces: &mut Vec<String>, text: &str) -> Result<()> {
        self.charge(text.len())?;
        reserve(pieces, 1, "pretokenizer fragments")?;
        pieces.push(string(text)?);
        Ok(())
    }
    fn push_byte_piece(&mut self, pieces: &mut Vec<String>, text: &str) -> Result<()> {
        let cap = text
            .len()
            .checked_mul(2)
            .ok_or(Error::Limit("byte pretokenizer size"))?;
        self.check_bytes(cap)?;
        self.charge(cap)?;
        let mut piece = String::new();
        piece
            .try_reserve_exact(cap)
            .map_err(|_| Error::Limit("byte pretokenizer allocation"))?;
        for byte in text.bytes() {
            piece.push(BYTE_CHAR[byte as usize]);
        }
        reserve(pieces, 1, "byte fragments")?;
        pieces.push(piece);
        Ok(())
    }
    fn split_added(
        &mut self,
        input: &str,
        normalized: bool,
    ) -> Result<Vec<(usize, usize, Option<u32>)>> {
        let mut result = Vec::new();
        let mut cursor = 0;
        let mut search = 0;
        while search < input.len() {
            let mut best: Option<(usize, usize, usize)> = None;
            let work = (input.len() - search)
                .checked_mul(self.patterns.len())
                .ok_or(Error::Limit("added-token matching work"))?;
            self.charge(work)?;
            // Native automaton orders special patterns before ordinary ones.
            for special in [true, false] {
                for (index, pattern) in self.patterns.iter().enumerate() {
                    if pattern.special != special || pattern.normalized != normalized {
                        continue;
                    }
                    let content = pattern
                        .normalized_content
                        .as_deref()
                        .unwrap_or(pattern.content);
                    if let Some(found) = input[search..].find(content) {
                        let start = search + found;
                        let end = start + content.len();
                        if best.is_none_or(|(a, b, _)| start < a || (start == a && end > b)) {
                            best = Some((start, end, index));
                        }
                    }
                }
            }
            let Some((mut start, mut end, index)) = best else {
                break;
            };
            search = end;
            let pattern = &self.patterns[index];
            let token = self
                .added
                .iter()
                .find(|v| v.0 == pattern.id)
                .ok_or(Error::Invalid("added matcher ID"))?
                .1;
            if self.split_special_tokens
                && self
                    .patterns
                    .iter()
                    .any(|p| p.special && p.content == token.content)
            {
                continue;
            }
            if token.single_word {
                let left = input[..start]
                    .chars()
                    .next_back()
                    .is_some_and(|ch| unicode::flags(ch as u32) & unicode::WORD != 0);
                let right = input[end..]
                    .chars()
                    .next()
                    .is_some_and(|ch| unicode::flags(ch as u32) & unicode::WORD != 0);
                if left || right {
                    continue;
                }
            }
            if token.lstrip {
                let trim = input[..start]
                    .trim_end_matches(|ch: char| {
                        unicode::flags(ch as u32) & unicode::ADDED_SPACE != 0
                    })
                    .len();
                start = trim.max(cursor);
            }
            if token.rstrip {
                end += input[end..].len()
                    - input[end..]
                        .trim_start_matches(|ch: char| {
                            unicode::flags(ch as u32) & unicode::ADDED_SPACE != 0
                        })
                        .len();
            }
            if start < cursor || end < start {
                return Err(Error::Invalid("overlapping added-token whitespace spans"));
            }
            if cursor < start {
                reserve(&mut result, 1, "added-token spans")?;
                result.push((cursor, start, None));
            }
            reserve(&mut result, 1, "added-token spans")?;
            result.push((start, end, Some(pattern.id)));
            cursor = end;
        }
        if cursor < input.len() {
            reserve(&mut result, 1, "added-token spans")?;
            result.push((cursor, input.len(), None));
        }
        Ok(result)
    }
    pub fn encode(&mut self, input: &str) -> Result<Vec<u32>> {
        if !self.executable {
            return Err(Error::Unsupported(
                "HF normalization/pretokenization pipeline has no verified converter execution",
            ));
        }
        self.check_bytes(input.len())?;
        let mut output = Vec::new();
        for (start, end, id) in self.split_added(input, false)? {
            if let Some(id) = id {
                reserve(&mut output, 1, "added output IDs")?;
                output.push(id);
                continue;
            }
            let fragment = &input[start..end];
            let normalized = if self.normalization == Normalization::Nfc {
                Some(nfc::normalize(
                    fragment,
                    self.max_bytes,
                    &mut self.operations_left,
                )?)
            } else {
                None
            };
            let fragment = normalized.as_deref().unwrap_or(fragment);
            for (a, b, id) in self.split_added(fragment, true)? {
                if let Some(id) = id {
                    reserve(&mut output, 1, "added output IDs")?;
                    output.push(id);
                    continue;
                }
                for piece in self.pieces(&fragment[a..b])? {
                    self.bpe(&piece, &mut output)?;
                }
            }
        }
        Ok(output)
    }
    pub fn set_llama_post(
        &mut self,
        bos: Option<(&str, u32)>,
        eos: Option<(&str, u32)>,
    ) -> Result<()> {
        self.post = crate::hf_schema::Post::Class {
            bos: bos
                .map(|(name, id)| crate::hf_schema::class_piece(name, id))
                .transpose()?,
            eos: eos
                .map(|(name, id)| crate::hf_schema::class_piece(name, id))
                .transpose()?,
        };
        Ok(())
    }
    pub fn encode_with_special(&mut self, input: &str, special: bool) -> Result<Vec<u32>> {
        let ids = self.encode(input)?;
        self.post
            .apply(ids, special, &mut self.operations_left, self.max_bytes)
    }
    pub fn decode(&mut self, ids: &[u32]) -> Result<String> {
        if !self.byte_decoder {
            return Err(Error::Unsupported("HF decoder execution is not ByteLevel"));
        }
        let mut raw = Vec::new();
        for &id in ids {
            let text = self
                .by_id
                .get(id as usize)
                .and_then(|s| *s)
                .ok_or(Error::Invalid("decoded ID has no token"))?;
            self.charge(text.len())?;
            let all = text.chars().all(|ch| BYTE_CHAR.contains(&ch));
            let bytes = if all {
                text.chars().count()
            } else {
                text.len()
            };
            self.check_bytes(
                raw.len()
                    .checked_add(bytes)
                    .ok_or(Error::Limit("decoded bytes"))?,
            )?;
            reserve(&mut raw, bytes, "HF decoded bytes")?;
            if all {
                for ch in text.chars() {
                    raw.push(
                        BYTE_CHAR
                            .iter()
                            .position(|v| *v == ch)
                            .expect("mapped character") as u8,
                    );
                }
            } else {
                raw.extend_from_slice(text.as_bytes());
            }
        }
        let mut output = String::new();
        let mut remaining = raw.as_slice();
        while !remaining.is_empty() {
            match std::str::from_utf8(remaining) {
                Ok(valid) => {
                    output
                        .try_reserve(valid.len())
                        .map_err(|_| Error::Limit("decoded UTF-8"))?;
                    output.push_str(valid);
                    break;
                }
                Err(error) => {
                    let valid = std::str::from_utf8(&remaining[..error.valid_up_to()])
                        .expect("valid prefix");
                    output
                        .try_reserve(valid.len() + 3)
                        .map_err(|_| Error::Limit("decoded UTF-8"))?;
                    output.push_str(valid);
                    output.push('\u{fffd}');
                    let consumed = error.valid_up_to()
                        + error
                            .error_len()
                            .unwrap_or(remaining.len() - error.valid_up_to());
                    remaining = &remaining[consumed..];
                }
            }
        }
        self.check_bytes(output.len())?;
        if self.clean_up_spaces {
            for (from, to) in [
                (" .", "."),
                (" ?", "?"),
                (" !", "!"),
                (" ,", ","),
                (" ' ", "'"),
                (" n't", "n't"),
                (" 'm", "'m"),
                (" 's", "'s"),
                (" 've", "'ve"),
                (" 're", "'re"),
            ] {
                if output.contains(from) {
                    output = output.replace(from, to);
                }
            }
        }
        Ok(output)
    }
}
fn parse_stages(value: Option<&Json>, stages: &mut Vec<Stage>, depth: usize) -> Result<bool> {
    if depth > 16 || stages.len() > 64 {
        return Err(Error::Limit("HF pretokenizer sequence depth/count"));
    }
    let Some(value) = value.filter(|v| !matches!(v, Json::Null)) else {
        return Ok(true);
    };
    let value = object(value)?;
    let kind = text(
        value
            .get("type")
            .ok_or(Error::Invalid("pretokenizer type"))?,
    )?;
    match kind {
        "Sequence" => {
            let values = value
                .get("pretokenizers")
                .and_then(Json::as_array)
                .ok_or(Error::Invalid("pretokenizer sequence"))?;
            let mut supported = true;
            for value in values {
                supported &= parse_stages(Some(value), stages, depth + 1)?;
            }
            Ok(supported)
        }
        "ByteLevel" => {
            reserve(stages, 1, "HF pretokenizer stages")?;
            let prefix = flag(value, "add_prefix_space", false, true)?;
            flag(value, "trim_offsets", false, true)?;
            stages.push(Stage::Byte {
                prefix,
                regex: flag(value, "use_regex", true, false)?,
            });
            Ok(true)
        }
        "Digits" => {
            reserve(stages, 1, "HF pretokenizer stages")?;
            stages.push(Stage::Digits {
                individual: flag(value, "individual_digits", false, true)?,
            });
            Ok(true)
        }
        "Split" => {
            let expression = value
                .get("pattern")
                .and_then(Json::as_object)
                .and_then(|p| p.get("Regex"))
                .map(text)
                .transpose()?;
            let behavior = value.get("behavior").map(text).transpose()?;
            let invert = flag(value, "invert", false, true)?;
            if expression == Some(crate::hf_schema::QWEN_PATTERN)
                && behavior == Some("Isolated")
                && !invert
            {
                reserve(stages, 1, "HF pretokenizer stages")?;
                stages.push(Stage::QwenSplit);
                Ok(true)
            } else {
                Ok(false)
            }
        }
        "Metaspace" | "Whitespace" | "WhitespaceSplit" | "BertPreTokenizer" | "Punctuation"
        | "CharDelimiterSplit" | "UnicodeScripts" | "FixedLength" => Ok(false),
        _ => Err(Error::Unsupported("HF pretokenizer schema")),
    }
}
const fn byte_chars() -> [char; 256] {
    let mut result = ['\0'; 256];
    let mut extra = 0;
    let mut i = 0;
    while i < 256 {
        let direct = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174);
        let cp = if direct {
            i as u32
        } else {
            let cp = 256 + extra;
            extra += 1;
            cp
        };
        result[i] = match char::from_u32(cp) {
            Some(c) => c,
            None => '\0',
        };
        i += 1;
    }
    result
}
const BYTE_CHAR: [char; 256] = byte_chars();
fn byte_regex_end(text: &str, start: usize) -> usize {
    let rest = &text[start..];
    for suffix in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.starts_with(suffix) {
            return start + suffix.len();
        }
    }
    let optional = usize::from(rest.starts_with(' '));
    for group in [unicode::LETTER, unicode::NUMBER, 0] {
        let mut end = start + optional;
        let first = end;
        for ch in text[end..].chars() {
            let flags = unicode::flags(ch as u32);
            let member = if group == 0 {
                flags & (unicode::SPACE | unicode::LETTER | unicode::NUMBER) == 0
            } else {
                flags & group != 0
            };
            if !member {
                break;
            }
            end += ch.len_utf8();
        }
        if end > first {
            return end;
        }
    }
    let mut end = start;
    let mut last = start;
    for ch in rest.chars() {
        if unicode::flags(ch as u32) & unicode::SPACE == 0 {
            break;
        }
        last = end;
        end += ch.len_utf8();
    }
    if end > start {
        if end < text.len() && last > start {
            return last;
        }
        return end;
    }
    // All scalar values are covered by Letter, Number, Space or Other.
    start
        + rest
            .chars()
            .next()
            .expect("nonempty regex remainder")
            .len_utf8()
}

/// Exact reviewed Qwen Oniguruma alternative order. The native casefold
/// capture over every Unicode scalar adds U+017F to ASCII S/s only.
pub(crate) fn qwen_regex_end(text: &str, start: usize) -> usize {
    let rest = &text[start..];
    if let Some(tail) = rest.strip_prefix('\'') {
        for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
            let mut chars = tail.chars();
            let mut end = start + 1;
            let matched = suffix.chars().all(|expected| {
                let Some(ch) = chars.next() else { return false };
                let folded = if ch == '\u{17f}' {
                    's'
                } else {
                    ch.to_ascii_lowercase()
                };
                if folded != expected {
                    return false;
                }
                end += ch.len_utf8();
                true
            });
            if matched {
                return end;
            }
        }
    }
    let first = rest.chars().next().expect("nonempty Qwen regex remainder");
    let flags = unicode::flags(first as u32);
    let prefix =
        if flags & (unicode::LETTER | unicode::NUMBER) == 0 && first != '\r' && first != '\n' {
            first.len_utf8()
        } else {
            0
        };
    let mut end = start + prefix;
    let begin = end;
    for ch in text[end..].chars() {
        if unicode::flags(ch as u32) & unicode::LETTER == 0 {
            break;
        }
        end += ch.len_utf8();
    }
    if end > begin {
        return end;
    }
    if flags & unicode::NUMBER != 0 {
        return start + first.len_utf8();
    }
    let mut end = start + usize::from(first == ' ');
    let begin = end;
    for ch in text[end..].chars() {
        if unicode::flags(ch as u32) & (unicode::SPACE | unicode::LETTER | unicode::NUMBER) != 0 {
            break;
        }
        end += ch.len_utf8();
    }
    if end > begin {
        while matches!(text.as_bytes().get(end), Some(b'\r' | b'\n')) {
            end += 1;
        }
        return end;
    }
    let (mut end, mut last, mut newline) = (start, start, start);
    for ch in rest.chars() {
        if unicode::flags(ch as u32) & unicode::SPACE == 0 {
            break;
        }
        last = end;
        end += ch.len_utf8();
        if matches!(ch, '\r' | '\n') {
            newline = end;
        }
    }
    if newline > start {
        return newline;
    }
    if end > start {
        return if end < text.len() && last > start {
            last
        } else {
            end
        };
    }
    // Every scalar is Letter, Number, Space or Other.
    start + first.len_utf8()
}
