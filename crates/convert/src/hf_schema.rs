//! Tokenizers 0.22.2 JSON contracts and ID-only postprocessing.
//! The original Serde wrapper dispatch (including legacy untagged variants) is
//! observable during metadata-only LlamaHfVocab loading. Parsing a pipeline does
//! not imply that its normalization or regex execution is implemented.
use crate::hf_backend::id;
use crate::util::{object, optional_text, text};
use crate::{Error, Result, reserve};
use llama_checkpoint::{Number, Object, Value as Json};

pub(crate) const BLOOM_PATTERN: &str = r" ?[^(\s|[.,!?…。，、।۔،])]+";

pub(crate) const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Normalization {
    Identity,
    Nfc,
    Unsupported,
}
impl Normalization {
    fn identity(value: bool) -> Self {
        if value {
            Self::Identity
        } else {
            Self::Unsupported
        }
    }
    fn then(self, next: Self) -> Self {
        match (self, next) {
            (Self::Unsupported, _) | (_, Self::Unsupported) => Self::Unsupported,
            (Self::Nfc, _) | (_, Self::Nfc) => Self::Nfc,
            _ => Self::Identity,
        }
    }
}

pub(crate) const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn required<'a>(value: &'a Object, key: &str) -> Result<&'a Json> {
    value
        .get(key)
        .ok_or(Error::Invalid("missing tokenizer schema field"))
}
fn bool_value(value: &Json) -> Result<bool> {
    match value {
        Json::Bool(v) => Ok(*v),
        _ => Err(Error::Invalid("tokenizer schema boolean")),
    }
}
fn boolean(value: &Object, key: &str) -> Result<bool> {
    bool_value(required(value, key)?)
}
fn optional_bool(value: Option<&Json>) -> Result<Option<bool>> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(v) => bool_value(v).map(Some),
    }
}
fn usize_value(value: &Json) -> Result<usize> {
    match value {
        Json::Number(Number::Integer(value)) => value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(Error::Invalid("tokenizer schema usize")),
        _ => Err(Error::Invalid("tokenizer schema usize")),
    }
}
fn array(value: &Json) -> Result<&[Json]> {
    value
        .as_array()
        .ok_or(Error::Invalid("tokenizer schema array"))
}
fn character(value: &Json) -> Result<char> {
    let mut chars = text(value)?.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => Ok(ch),
        _ => Err(Error::Invalid(
            "tokenizer schema requires one Unicode scalar",
        )),
    }
}
fn tag(value: &Object) -> Option<&str> {
    value.get("type").and_then(|v| text(v).ok())
}
fn marker(value: &Object, name: &str) -> Result<()> {
    if tag(value) == Some(name) {
        Ok(())
    } else {
        Err(Error::Invalid("tokenizer schema type marker"))
    }
}
fn choice<'a>(value: &'a Json, values: &[&str]) -> Result<&'a str> {
    let s = text(value)?;
    if values.contains(&s) {
        Ok(s)
    } else {
        Err(Error::Invalid("tokenizer schema enum value"))
    }
}
fn nested(depth: usize) -> Result<()> {
    if depth > 16 {
        Err(Error::Limit("HF tokenizer schema nesting"))
    } else {
        Ok(())
    }
}

/// A finite, explicitly reviewed regex domain. Unknown valid Oniguruma syntax
/// is rejected as unsupported, never executed as a literal or another regex.
fn regex(pattern: &str) -> Result<()> {
    if pattern == LLAMA3_PATTERN
        || pattern == QWEN_PATTERN
        || pattern == BLOOM_PATTERN
        || pattern == "x+"
    {
        return Ok(());
    }
    if pattern.chars().all(|ch| {
        !matches!(
            ch,
            '\\' | '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$'
        )
    }) {
        return Ok(());
    }
    Err(Error::Unsupported(
        "HF Oniguruma expression outside the reviewed converter domain",
    ))
}
fn pattern(value: &Json) -> Result<()> {
    let value = object(value)?;
    if value.len() != 1 {
        return Err(Error::Invalid("tokenizer pattern enum"));
    }
    let (name, value) = value
        .entries()
        .next()
        .ok_or(Error::Invalid("tokenizer pattern enum"))?;
    match name.as_str() {
        Some("String") => {
            text(value)?;
            Ok(())
        }
        Some("Regex") => regex(text(value)?),
        _ => Err(Error::Invalid("tokenizer pattern enum")),
    }
}
fn replacement(value: &Object) -> Result<()> {
    pattern(required(value, "pattern")?)?;
    text(required(value, "content")?)?;
    Ok(())
}
fn byte_level(value: &Object) -> Result<()> {
    marker(value, "ByteLevel")?;
    boolean(value, "add_prefix_space")?;
    boolean(value, "trim_offsets")?;
    if let Some(value) = value.get("use_regex") {
        bool_value(value)?;
    }
    Ok(())
}
fn metaspace(value: &Object) -> Result<()> {
    marker(value, "Metaspace")?;
    character(required(value, "replacement")?)?;
    let scheme = value
        .get("prepend_scheme")
        .map(|v| choice(v, &["first", "never", "always"]))
        .transpose()?
        .unwrap_or("always");
    if optional_bool(value.get("add_prefix_space"))? == Some(false) && scheme != "never" {
        return Err(Error::Invalid(
            "Metaspace prefix contradicts prepend scheme",
        ));
    }
    optional_bool(value.get("split"))?;
    optional_text(value.get("str_rep"))?;
    Ok(())
}
fn alternate<T>(names: &[&str], mut parse: impl FnMut(&str) -> Result<T>) -> Result<T> {
    let mut unsupported = None;
    for name in names {
        match parse(name) {
            Ok(value) => return Ok(value),
            Err(error @ Error::Limit(_)) => return Err(error),
            Err(error @ Error::Unsupported(_)) => unsupported = Some(error),
            Err(_) => {}
        }
    }
    Err(unsupported.unwrap_or(Error::Invalid("no compatible tokenizer schema variant")))
}

pub(crate) fn normalizer(value: Option<&Json>, depth: usize) -> Result<Normalization> {
    nested(depth)?;
    let Some(value) = value.filter(|v| !matches!(v, Json::Null)) else {
        return Ok(Normalization::Identity);
    };
    let value = object(value)?;
    const TAGS: &[&str] = &[
        "Bert",
        "Strip",
        "StripAccents",
        "NFC",
        "NFD",
        "NFKC",
        "NFKD",
        "Sequence",
        "Lowercase",
        "Nmt",
        "Precompiled",
        "Replace",
        "Prepend",
        "ByteLevel",
    ];
    if let Some(kind) = tag(value).filter(|s| TAGS.contains(s)) {
        return normalizer_variant(value, kind, depth);
    }
    // BertNormalizer is deliberately absent from the Tagged enum in 0.22.2.
    alternate(
        &[
            "Bert",
            "Strip",
            "StripAccents",
            "NFC",
            "NFD",
            "NFKC",
            "NFKD",
            "Sequence",
            "Lowercase",
            "Nmt",
            "Precompiled",
            "Replace",
            "Prepend",
            "ByteLevel",
        ],
        |kind| normalizer_variant(value, kind, depth),
    )
}
fn normalizer_variant(value: &Object, kind: &str, depth: usize) -> Result<Normalization> {
    match kind {
        "Bert" => {
            let clean = boolean(value, "clean_text")?;
            let cjk = boolean(value, "handle_chinese_chars")?;
            let lower = boolean(value, "lowercase")?;
            let accents = optional_bool(value.get("strip_accents"))?.unwrap_or(lower);
            Ok(Normalization::identity(
                !clean && !cjk && !lower && !accents,
            ))
        }
        "Strip" => {
            let left = boolean(value, "strip_left")?;
            let right = boolean(value, "strip_right")?;
            Ok(Normalization::identity(!left && !right))
        }
        "Sequence" => {
            let values = array(required(value, "normalizers")?)?;
            let mut execution = Normalization::Identity;
            for value in values {
                if matches!(value, Json::Null) {
                    return Err(Error::Invalid("null nested normalizer"));
                }
                execution = execution.then(normalizer(Some(value), depth + 1)?);
            }
            Ok(execution)
        }
        "Replace" => {
            replacement(value)?;
            Ok(Normalization::Unsupported)
        }
        "Prepend" => Ok(Normalization::identity(
            text(required(value, "prepend")?)?.is_empty(),
        )),
        "Precompiled" => {
            marker(value, "Precompiled")?;
            let map = text(required(value, "precompiled_charsmap")?)?;
            if map.is_empty() {
                return Err(Error::Invalid("empty precompiled normalizer map"));
            }
            Err(Error::Unsupported("Tokenizers precompiled normalizer map"))
        }
        "NFC" => {
            marker(value, kind)?;
            Ok(Normalization::Nfc)
        }
        "StripAccents" | "NFD" | "NFKC" | "NFKD" | "Lowercase" | "Nmt" | "ByteLevel" => {
            marker(value, kind)?;
            Ok(Normalization::Unsupported)
        }
        _ => Err(Error::Invalid("unknown normalizer variant")),
    }
}
pub(crate) fn pretokenizer(value: Option<&Json>, depth: usize) -> Result<()> {
    nested(depth)?;
    let Some(value) = value.filter(|v| !matches!(v, Json::Null)) else {
        return Ok(());
    };
    let value = object(value)?;
    match tag(value).ok_or(Error::Invalid("pretokenizer type"))? {
        "ByteLevel" => byte_level(value),
        "Digits" => {
            boolean(value, "individual_digits")?;
            Ok(())
        }
        "Metaspace" => metaspace(value),
        "Sequence" => {
            for child in array(required(value, "pretokenizers")?)? {
                if matches!(child, Json::Null) {
                    return Err(Error::Invalid("null nested pretokenizer"));
                }
                pretokenizer(Some(child), depth + 1)?;
            }
            Ok(())
        }
        "Split" => {
            pattern(required(value, "pattern")?)?;
            choice(
                required(value, "behavior")?,
                &[
                    "Removed",
                    "Isolated",
                    "MergedWithPrevious",
                    "MergedWithNext",
                    "Contiguous",
                ],
            )?;
            boolean(value, "invert")?;
            Ok(())
        }
        "Punctuation" => {
            if let Some(value) = value.get("behavior") {
                choice(
                    value,
                    &[
                        "Removed",
                        "Isolated",
                        "MergedWithPrevious",
                        "MergedWithNext",
                        "Contiguous",
                    ],
                )?;
            }
            Ok(())
        }
        "CharDelimiterSplit" => {
            character(required(value, "delimiter")?)?;
            Ok(())
        }
        "FixedLength" => {
            if let Some(value) = value.get("length") {
                usize_value(value)?;
            }
            Ok(())
        }
        "BertPreTokenizer" | "Whitespace" | "WhitespaceSplit" | "UnicodeScripts" => Ok(()),
        _ => Err(Error::Unsupported("HF pretokenizer schema")),
    }
}
pub(crate) fn decoder(value: Option<&Json>, depth: usize) -> Result<&'static str> {
    nested(depth)?;
    let Some(value) = value.filter(|v| !matches!(v, Json::Null)) else {
        return Ok("None");
    };
    let value = object(value)?;
    const TAGS: &[&str] = &[
        "BPEDecoder",
        "ByteLevel",
        "WordPiece",
        "Metaspace",
        "CTC",
        "Sequence",
        "Replace",
        "Fuse",
        "Strip",
        "ByteFallback",
    ];
    if let Some(kind) = tag(value).filter(|s| TAGS.contains(s)) {
        return decoder_variant(value, kind, depth);
    }
    alternate(TAGS, |kind| decoder_variant(value, kind, depth))
}
fn decoder_variant(value: &Object, kind: &str, depth: usize) -> Result<&'static str> {
    Ok(match kind {
        "BPEDecoder" => {
            text(required(value, "suffix")?)?;
            "BPEDecoder"
        }
        "ByteLevel" => {
            byte_level(value)?;
            "ByteLevel"
        }
        "WordPiece" => {
            text(required(value, "prefix")?)?;
            boolean(value, "cleanup")?;
            "WordPiece"
        }
        "Metaspace" => {
            metaspace(value)?;
            "Metaspace"
        }
        "CTC" => {
            text(required(value, "pad_token")?)?;
            text(required(value, "word_delimiter_token")?)?;
            boolean(value, "cleanup")?;
            "CTC"
        }
        "Sequence" => {
            marker(value, "Sequence")?;
            for child in array(required(value, "decoders")?)? {
                if matches!(child, Json::Null) {
                    return Err(Error::Invalid("null nested decoder"));
                }
                decoder(Some(child), depth + 1)?;
            }
            "Sequence"
        }
        "Replace" => {
            replacement(value)?;
            "Replace"
        }
        "Strip" => {
            character(required(value, "content")?)?;
            usize_value(required(value, "start")?)?;
            usize_value(required(value, "stop")?)?;
            "Strip"
        }
        "Fuse" => {
            marker(value, "Fuse")?;
            "Fuse"
        }
        "ByteFallback" => {
            marker(value, "ByteFallback")?;
            "ByteFallback"
        }
        _ => return Err(Error::Invalid("unknown decoder variant")),
    })
}
pub(crate) fn settings(root: &Object) -> Result<()> {
    if let Some(value) = root.get("padding").filter(|v| !matches!(v, Json::Null)) {
        let value = object(value)?;
        match required(value, "strategy")? {
            Json::Text(_) => {
                choice(required(value, "strategy")?, &["BatchLongest"])?;
            }
            other => {
                let strategy = object(other)?;
                if strategy.len() != 1 {
                    return Err(Error::Invalid("padding strategy"));
                }
                usize_value(required(strategy, "Fixed")?)?;
            }
        }
        choice(required(value, "direction")?, &["Left", "Right"])?;
        if let Some(value) = value
            .get("pad_to_multiple_of")
            .filter(|v| !matches!(v, Json::Null))
        {
            usize_value(value)?;
        }
        id(required(value, "pad_id")?)?;
        id(required(value, "pad_type_id")?)?;
        text(required(value, "pad_token")?)?;
    }
    if let Some(value) = root.get("truncation").filter(|v| !matches!(v, Json::Null)) {
        let value = object(value)?;
        if let Some(value) = value.get("direction") {
            choice(value, &["Left", "Right"])?;
        }
        usize_value(required(value, "max_length")?)?;
        usize_value(required(value, "stride")?)?;
        choice(
            required(value, "strategy")?,
            &["LongestFirst", "OnlyFirst", "OnlySecond"],
        )?;
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) enum Post<'a> {
    None,
    Bert {
        cls: u32,
        sep: u32,
        roberta: bool,
    },
    Template {
        single: &'a [Json],
        pair: &'a [Json],
        specials: &'a Object,
    },
    Sequence(Vec<Post<'a>>),
    Class {
        bos: Option<ClassPiece>,
        eos: Option<ClassPiece>,
    },
}
fn token_pair(value: &Json) -> Result<u32> {
    let pair = array(value)?;
    if pair.len() != 2 {
        return Err(Error::Invalid("postprocessor token tuple"));
    }
    text(&pair[0])?;
    id(&pair[1])
}
fn template(value: &Json) -> Result<&[Json]> {
    let pieces = array(value)?;
    for piece in pieces {
        let piece = object(piece)?;
        if piece.len() != 1 {
            return Err(Error::Invalid("postprocessor template piece"));
        }
        if let Some(value) = piece.get("Sequence") {
            let value = object(value)?;
            choice(required(value, "id")?, &["A", "B"])?;
            id(required(value, "type_id")?)?;
        } else if let Some(value) = piece.get("SpecialToken") {
            let value = object(value)?;
            text(required(value, "id")?)?;
            id(required(value, "type_id")?)?;
        } else {
            return Err(Error::Invalid("postprocessor template piece"));
        }
    }
    Ok(pieces)
}
fn special_tokens(value: &Json) -> Result<&Object> {
    let map = object(value)?;
    for (key, value) in map.entries() {
        if key.as_str().is_none() {
            return Err(Error::Invalid("postprocessor special map key"));
        }
        let value = object(value)?;
        text(required(value, "id")?)?;
        for value in array(required(value, "ids")?)? {
            id(value)?;
        }
        for value in array(required(value, "tokens")?)? {
            text(value)?;
        }
        // The source Deserialize path does not apply the builder's equal-length
        // or template-reference checks. Execution errors are checked separately.
    }
    Ok(map)
}
impl<'a> Post<'a> {
    pub(crate) fn parse(value: Option<&'a Json>, depth: usize) -> Result<Self> {
        nested(depth)?;
        let Some(value) = value.filter(|v| !matches!(v, Json::Null)) else {
            return Ok(Self::None);
        };
        let value = object(value)?;
        // PostProcessorWrapper is untagged: even known tags do not override the
        // Roberta/Bert field-based precedence. Malformed Roberta flags can select Bert.
        if let (Ok(cls), Ok(sep)) = (
            required(value, "cls").and_then(token_pair),
            required(value, "sep").and_then(token_pair),
        ) {
            let roberta = boolean(value, "trim_offsets").is_ok()
                && boolean(value, "add_prefix_space").is_ok();
            return Ok(Self::Bert { cls, sep, roberta });
        }
        if byte_level(value).is_ok() {
            return Ok(Self::None);
        }
        if let (Ok(single), Ok(pair), Ok(specials)) = (
            required(value, "single").and_then(template),
            required(value, "pair").and_then(template),
            required(value, "special_tokens").and_then(special_tokens),
        ) {
            return Ok(Self::Template {
                single,
                pair,
                specials,
            });
        }
        marker(value, "Sequence")?;
        let values = array(required(value, "processors")?)?;
        let mut output = Vec::new();
        reserve(&mut output, values.len(), "HF postprocessor descriptors")?;
        for value in values {
            if matches!(value, Json::Null) {
                return Err(Error::Invalid("null nested postprocessor"));
            }
            output.push(Self::parse(Some(value), depth + 1)?);
        }
        Ok(Self::Sequence(output))
    }
    pub(crate) fn apply(
        &self,
        input: Vec<u32>,
        special: bool,
        remaining: &mut usize,
        max_bytes: usize,
    ) -> Result<Vec<u32>> {
        let mut parts = Vec::new();
        reserve(&mut parts, 1, "HF postprocessor input")?;
        parts.push(input);
        let parts = self.parts(parts, special, remaining, max_bytes)?;
        let count = parts.iter().try_fold(0usize, |n, p| {
            n.checked_add(p.len())
                .ok_or(Error::Limit("HF postprocessor IDs"))
        })?;
        post_budget(count, parts.len(), remaining, max_bytes)?;
        let mut result = Vec::new();
        reserve(&mut result, count, "HF postprocessor merged IDs")?;
        for part in parts {
            result.extend(part);
        }
        Ok(result)
    }
    fn parts(
        &self,
        input: Vec<Vec<u32>>,
        special: bool,
        remaining: &mut usize,
        max_bytes: usize,
    ) -> Result<Vec<Vec<u32>>> {
        match self {
            Self::None => Ok(input),
            Self::Sequence(stages) => {
                let mut value = input;
                for stage in stages {
                    value = stage.parts(value, special, remaining, max_bytes)?;
                }
                Ok(value)
            }
            Self::Class { bos, eos } => {
                if input.len() != 1 {
                    return Err(Error::Invalid(
                        "single input required by converter class postprocessor",
                    ));
                }
                let pieces = [*bos, Some(ClassPiece::Sequence(0)), *eos];
                let mut count = 0usize;
                for piece in pieces.into_iter().flatten() {
                    let length = match piece {
                        ClassPiece::Token(_) => usize::from(special),
                        ClassPiece::Sequence(index) => input
                            .get(index)
                            .ok_or(Error::Invalid(
                                "class postprocessor references absent sequence",
                            ))?
                            .len(),
                    };
                    count = count
                        .checked_add(length)
                        .ok_or(Error::Limit("class postprocessor IDs"))?;
                }
                post_budget(count, 1, remaining, max_bytes)?;
                let mut result = Vec::new();
                reserve(&mut result, count, "class postprocessor IDs")?;
                for piece in pieces.into_iter().flatten() {
                    match piece {
                        ClassPiece::Token(id) => {
                            if special {
                                result.push(id);
                            }
                        }
                        ClassPiece::Sequence(index) => result.extend_from_slice(&input[index]),
                    }
                }
                let mut parts = Vec::new();
                reserve(&mut parts, 1, "class postprocessor output")?;
                parts.push(result);
                Ok(parts)
            }
            Self::Bert { cls, sep, roberta } => {
                if !special {
                    return Ok(input);
                }
                let count = input.iter().enumerate().try_fold(0usize, |n, (i, p)| {
                    n.checked_add(p.len())
                        .and_then(|v| v.checked_add(if i == 0 || *roberta { 2 } else { 1 }))
                        .ok_or(Error::Limit("BERT special IDs"))
                })?;
                post_budget(count, input.len(), remaining, max_bytes)?;
                let mut result = Vec::new();
                reserve(&mut result, input.len(), "BERT parts")?;
                for (i, part) in input.into_iter().enumerate() {
                    let extra = if i == 0 || *roberta { 2 } else { 1 };
                    let mut value = Vec::new();
                    reserve(&mut value, part.len() + extra, "BERT IDs")?;
                    if i == 0 {
                        value.push(*cls);
                    } else if *roberta {
                        value.push(*sep);
                    }
                    value.extend(part);
                    value.push(*sep);
                    result.push(value);
                }
                Ok(result)
            }
            Self::Template {
                single,
                pair,
                specials,
            } => {
                let template = match input.len() {
                    1 => single,
                    2 => pair,
                    _ => {
                        return Err(Error::Invalid(
                            "source TemplateProcessing panics outside one or two encodings",
                        ));
                    }
                };
                let mut selections: Vec<Selection<'_>> = Vec::new();
                reserve(&mut selections, template.len(), "HF template selections")?;
                let mut count = 0usize;
                for piece in *template {
                    let piece = object(piece)?;
                    let selection =
                        if let Some(value) = piece.get("Sequence") {
                            let value = object(value)?;
                            let i = usize::from(text(required(value, "id")?)? == "B");
                            Selection::Ids(input.get(i).ok_or(Error::Invalid(
                                "postprocessor references absent sequence",
                            ))?)
                        } else if special {
                            let value = object(required(piece, "SpecialToken")?)?;
                            let name = text(required(value, "id")?)?;
                            let token = object(specials.get(name).ok_or(Error::Invalid(
                                "postprocessor references absent special token",
                            ))?)?;
                            Selection::Json(array(required(token, "ids")?)?)
                        } else {
                            continue;
                        };
                    count = count
                        .checked_add(selection.len())
                        .ok_or(Error::Limit("HF template output IDs"))?;
                    selections.push(selection);
                }
                post_budget(count, selections.len(), remaining, max_bytes)?;
                let mut result = Vec::new();
                reserve(&mut result, selections.len(), "HF template parts")?;
                for selection in selections {
                    let mut value = Vec::new();
                    reserve(&mut value, selection.len(), "HF template IDs")?;
                    match selection {
                        Selection::Ids(ids) => value.extend_from_slice(ids),
                        Selection::Json(ids) => {
                            for item in ids {
                                value.push(id(item)?);
                            }
                        }
                    }
                    result.push(value);
                }
                Ok(result)
            }
        }
    }
}
/// LlamaTokenizerFast builds a TemplateProcessing string from configured token
/// text. Names such as "$A" become sequence references; colon/space names can
/// fail the original builder even when their token IDs are well formed.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ClassPiece {
    Token(u32),
    Sequence(usize),
}
pub(crate) fn class_piece(name: &str, id: u32) -> Result<ClassPiece> {
    if name.contains([':', ' ']) {
        return Err(Error::Invalid(
            "class special token cannot form the source template",
        ));
    }
    if let Some(reference) = name.strip_prefix('$') {
        return match reference {
            "" | "A" | "a" => Ok(ClassPiece::Sequence(0)),
            "B" | "b" => Ok(ClassPiece::Sequence(1)),
            value if value.parse::<u32>().is_ok() => Ok(ClassPiece::Sequence(0)),
            _ => Err(Error::Invalid(
                "class special token is an invalid template sequence",
            )),
        };
    }
    Ok(ClassPiece::Token(id))
}
enum Selection<'a> {
    Ids(&'a [u32]),
    Json(&'a [Json]),
}
impl Selection<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Ids(v) => v.len(),
            Self::Json(v) => v.len(),
        }
    }
}
fn post_budget(count: usize, parts: usize, remaining: &mut usize, max_bytes: usize) -> Result<()> {
    let bytes = count
        .checked_mul(8)
        .and_then(|n| {
            parts
                .checked_mul(2 * std::mem::size_of::<Vec<u32>>())
                .and_then(|p| n.checked_add(p))
        })
        .ok_or(Error::Limit("HF postprocessor working bytes"))?;
    if bytes > max_bytes {
        return Err(Error::Limit("HF postprocessor working bytes"));
    }
    *remaining = remaining
        .checked_sub(
            count
                .checked_add(parts)
                .ok_or(Error::Limit("HF postprocessor work"))?,
        )
        .ok_or(Error::Limit("HF postprocessor work"))?;
    Ok(())
}
