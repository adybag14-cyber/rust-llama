//! Frozen LlamaHfVocab and GPT-2-family vocabulary conversion.
//! AutoTokenizer admission is explicit; unsupported executable pipelines cannot
//! borrow another family's probe classification.
use crate::codec::SourceProfile;
use crate::config::Config;
use crate::hf_backend::{Added, Backend};
use crate::util::{JsonBudget, Metadata, integer, object, present, text, truth};
use crate::vocab::{Files, looks_special};
use crate::{Error, Limits, Result, bytes, reserve, string};
use llama_checkpoint::{Number, Object, SourceDirectory, Value as Json};
use llama_gguf::RawValue as Value;
use std::cmp::Ordering;
const SPECIAL: &[&str] = &[
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
];

#[derive(Debug)]
enum IndexKey {
    Integer { negative: bool, digits: String },
    Float(f64),
}
impl IndexKey {
    fn digits(negative: bool, digits: &str) -> Result<Self> {
        let digits = digits.trim_start_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        Ok(Self::Integer {
            negative: negative && digits != "0",
            digits: string(digits)?,
        })
    }
    fn from_json(value: &Json) -> Result<Self> {
        match value {
            Json::Number(Number::Integer(value)) => {
                Self::digits(value.is_negative(), value.digits())
            }
            Json::Number(Number::Float(value)) if value.is_finite() => Ok(Self::Float(*value)),
            Json::Bool(value) => Self::digits(false, if *value { "1" } else { "0" }),
            _ => Err(Error::Invalid(
                "legacy added token index is not a finite number",
            )),
        }
    }
    fn from_key(value: &str, profile: SourceProfile) -> Result<Self> {
        let value = crate::python_unicode::strip(value, profile)?;
        let (negative, input) = if let Some(value) = value.strip_prefix('-') {
            (true, value)
        } else {
            (false, value.strip_prefix('+').unwrap_or(value))
        };
        let mut digits = String::new();
        digits
            .try_reserve(input.len())
            .map_err(|_| Error::Limit("added-token integer key"))?;
        let mut previous = false;
        for ch in input.chars() {
            if ch == '_' {
                if !previous {
                    return Err(Error::Invalid("invalid added token integer key"));
                }
                previous = false;
                continue;
            }
            let value = crate::python_unicode::decimal(ch as u32, profile)?
                .ok_or(Error::Invalid("invalid added token integer key"))?;
            if digits.len() >= 4300 {
                return Err(Error::Limit("added token integer key digits"));
            }
            digits.push(char::from(b'0' + value));
            previous = true;
        }
        if !previous {
            return Err(Error::Invalid("invalid added token integer key"));
        }
        Self::digits(negative, &digits)
    }
    fn integer_cmp(an: bool, a: &str, bn: bool, b: &str) -> Ordering {
        if an != bn {
            return if an {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        let order = a.len().cmp(&b.len()).then_with(|| a.cmp(b));
        if an { order.reverse() } else { order }
    }
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (
                Self::Integer {
                    negative: a,
                    digits: x,
                },
                Self::Integer {
                    negative: b,
                    digits: y,
                },
            ) => Self::integer_cmp(*a, x, *b, y),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b).expect("finite index"),
            (Self::Integer { negative, digits }, Self::Float(value)) => {
                let floor = value.floor();
                let rendered = format!("{floor:.0}");
                let (neg, mag) = rendered
                    .strip_prefix('-')
                    .map_or((false, rendered.as_str()), |v| (v != "0", v));
                let cmp = Self::integer_cmp(*negative, digits, neg, mag);
                if cmp == Ordering::Equal && floor != *value {
                    Ordering::Less
                } else {
                    cmp
                }
            }
            (Self::Float(_), Self::Integer { .. }) => other.cmp(self).reverse(),
        }
    }
}
fn config_object(files: &Files) -> Result<Option<&Object>> {
    files.config.as_ref().map(object).transpose()
}
fn special_value<'a>(
    value: Option<&'a Json>,
    list: bool,
    from_map: bool,
) -> Result<Vec<Added<'a>>> {
    let mut output = Vec::new();
    let Some(value) = value.filter(|v| !matches!(v, Json::Null)) else {
        return Ok(output);
    };
    if list {
        let array = value
            .as_array()
            .ok_or(Error::Invalid("additional special tokens must be an array"))?;
        reserve(
            &mut output,
            array.len(),
            "additional special token descriptors",
        )?;
        for item in array {
            output.push(if matches!(item, Json::Text(_)) {
                Added::plain(text(item)?, true)
            } else {
                Added::from_json(item, false, from_map)?
            });
        }
    } else {
        reserve(&mut output, 1, "special token descriptor")?;
        if value.as_object().is_some() && !from_map {
            let value_object = object(value)?;
            if value_object.get("__type").map(text).transpose()? != Some("AddedToken") {
                return Err(Error::Invalid(
                    "tokenizer config special dictionary lacks AddedToken tag",
                ));
            }
        }
        output.push(if matches!(value, Json::Text(_)) {
            Added::plain(text(value)?, true)
        } else {
            Added::from_json(value, false, from_map)?
        });
    }
    Ok(output)
}
pub(crate) fn prepare<'a>(
    files: &'a Files,
    special_map: Option<&'a Json>,
    limits: &Limits,
    profile: SourceProfile,
    model_config: Option<&Config>,
) -> Result<(Backend<'a>, Vec<&'a str>)> {
    let raw = files
        .tokenizer
        .as_ref()
        .ok_or(Error::Invalid("tokenizer.json is absent"))?;
    let root = object(raw)?;
    let config = config_object(files)?;
    let class = match present(config.and_then(|o| o.get("tokenizer_class"))) {
        Some(value) => text(value)?,
        None => model_config
            .ok_or(Error::Unsupported(
                "tokenizer class inference requires a resolved model configuration",
            ))?
            .auto_tokenizer_fallback()?,
    };
    let is_llama = matches!(class, "LlamaTokenizer" | "LlamaTokenizerFast");
    let is_gpt2 = matches!(class, "GPT2Tokenizer" | "GPT2TokenizerFast");
    let is_qwen2 = matches!(class, "Qwen2Tokenizer" | "Qwen2TokenizerFast");
    let is_neox = matches!(class, "GPTNeoXTokenizer" | "GPTNeoXTokenizerFast");
    let is_bloom = matches!(class, "BloomTokenizer" | "BloomTokenizerFast");
    if !is_llama
        && !is_gpt2
        && !is_qwen2
        && !is_neox
        && !is_bloom
        && class != "PreTrainedTokenizerFast"
    {
        return Err(Error::Unsupported(
            "AutoTokenizer class outside reviewed dense converter ingestion",
        ));
    }
    if let Some(config) = config {
        for key in ["from_slow", "fast_tokenizer_files"] {
            if config.get(key).is_some_and(truth) {
                return Err(Error::Unsupported(
                    "alternate/slow tokenizer reconstruction",
                ));
            }
        }
        if let Some(path) = config
            .get("tokenizer_file")
            .filter(|v| !matches!(v, Json::Null))
            && text(path)? != "tokenizer.json"
        {
            return Err(Error::Unsupported(
                "tokenizer_file override requires separate retained sidecar selection",
            ));
        }
        for key in ["padding_side", "truncation_side"] {
            if let Some(value) = config.get(key)
                && !matches!(text(value)?, "left" | "right")
            {
                return Err(Error::Invalid("tokenizer side must be left or right"));
            }
        }
    }
    let mut backend = Backend::from_json(raw, limits)?;
    backend.override_direct_prefix(config, is_llama, is_bloom)?;
    // Python class defaults are applied before legacy map overrides.
    let mut attributes: Vec<Vec<Added<'a>>> = Vec::new();
    reserve(&mut attributes, SPECIAL.len(), "special token attributes")?;
    for (index, key) in SPECIAL.iter().enumerate() {
        let mut value = Vec::new();
        let default =
            if (is_qwen2 && matches!(index, 1 | 2 | 4)) || ((is_gpt2 || is_neox) && index < 3) {
                Some("<|endoftext|>")
            } else if is_llama || is_bloom {
                match index {
                    0 => Some("<s>"),
                    1 => Some("</s>"),
                    2 => Some("<unk>"),
                    4 if is_bloom => Some("<pad>"),
                    _ => None,
                }
            } else {
                None
            };
        if let Some(default) = default {
            reserve(&mut value, 1, "default special token")?;
            value.push(Added::plain(default, true));
        }
        if let Some(explicit) = config.and_then(|c| c.get(key)) {
            value = special_value(Some(explicit), index == 7, false)?;
        }
        attributes.push(value);
    }
    let decoder = config.and_then(|o| o.get("added_tokens_decoder"));
    if decoder.is_none()
        && let Some(map) = special_map
    {
        let map = object(map)?;
        for (index, key) in SPECIAL.iter().enumerate() {
            if let Some(value) = map.get(key) {
                let values = special_value(Some(value), index == 7, true)?;
                if index == 7 {
                    for item in values {
                        if !attributes[index].contains(&item) {
                            reserve(&mut attributes[index], 1, "legacy special union")?;
                            attributes[index].push(item);
                        }
                    }
                } else {
                    attributes[index] = values;
                }
            }
        }
    }
    if let Some(padding) = root.get("padding").filter(|v| !matches!(v, Json::Null)) {
        let padding = object(padding)?;
        if config.is_none_or(|c| c.get("pad_token").is_none()) {
            let token = text(
                padding
                    .get("pad_token")
                    .ok_or(Error::Invalid("padding pad_token is absent"))?,
            )?;
            attributes[4].clear();
            attributes[4].push(Added::plain(token, true));
        }
    }
    let mut pending: Vec<(IndexKey, usize, Added<'a>)> = Vec::new();
    if let Some(decoder) = decoder {
        for (index, (key, value)) in object(decoder)?.entries().enumerate() {
            let key = IndexKey::from_key(
                key.as_str()
                    .ok_or(Error::Invalid("added decoder key is not Unicode"))?,
                profile,
            )?;
            object(value)?;
            let token = Added::from_json(value, false, false)?;
            reserve(&mut pending, 1, "configured added token list")?;
            pending.push((key, index, token));
        }
    } else {
        if let Some(legacy) = files.added.as_ref() {
            for (value, index) in object(legacy)?.entries() {
                let value = value
                    .as_str()
                    .ok_or(Error::Invalid("legacy added token is not Unicode"))?;
                let special = attributes.iter().flatten().any(|a| a.content == value);
                let token = Added::plain(value, special);
                let index = IndexKey::from_json(index)?;
                reserve(&mut pending, 1, "legacy added token list")?;
                pending.push((index, pending.len(), token));
            }
        }
        // Tokenizer JSON metadata overwrites a legacy decoder entry at the
        // declared ID, while the backend already assigned its actual own IDs.
        if let Some(added) = root.get("added_tokens") {
            for value in added
                .as_array()
                .ok_or(Error::Invalid("serialized added tokens"))?
            {
                let item = object(value)?;
                let declared = item.get("id").ok_or(Error::Invalid("added token ID"))?;
                let index = IndexKey::from_json(declared)?;
                let token = Added::from_json(value, true, false)?;
                reserve(&mut pending, 1, "serialized added token updates")?;
                pending.push((index, pending.len(), token));
            }
        }
    }
    // Config special strings resolve to the saved AddedToken object by content.
    for values in &mut attributes {
        for token in values {
            if let Some((_, _, saved)) = pending.iter().rev().find(|p| p.2.content == token.content)
            {
                *token = *saved;
            }
        }
    }
    let mut specials = Vec::new();
    for token in attributes.iter().flatten() {
        if !token.content.is_empty() && !specials.contains(&token.content) {
            reserve(&mut specials, 1, "all special token names")?;
            specials.push(token.content);
        }
    }
    if let Some(extra) = config.and_then(|o| o.get("extra_special_tokens")) {
        for (_, value) in object(extra)?.entries() {
            let token = if matches!(value, Json::Text(_)) {
                Added::plain(text(value)?, true)
            } else {
                Added::from_json(value, false, false)?
            };
            if !token.content.is_empty() && !specials.contains(&token.content) {
                reserve(&mut specials, 1, "model special token names")?;
                specials.push(token.content);
            }
            reserve(&mut attributes, 1, "model special token attributes")?;
            let mut entry = Vec::new();
            reserve(&mut entry, 1, "model special token")?;
            entry.push(token);
            attributes.push(entry);
        }
    }
    pending.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut updates = Vec::new();
    reserve(
        &mut updates,
        pending.len() + specials.len(),
        "HF added token updates",
    )?;
    let mut at = 0;
    while at < pending.len() {
        let mut last = at;
        while last + 1 < pending.len() && pending[at].0.cmp(&pending[last + 1].0) == Ordering::Equal
        {
            last += 1;
        }
        let mut token = pending[last].2;
        token.special |= specials.contains(&token.content);
        updates.push(token);
        at = last + 1;
    }
    for values in &attributes {
        for &token in values {
            if backend
                .added
                .iter()
                .any(|(_, a)| a.content == token.content)
                || updates.iter().any(|a| a.content == token.content)
            {
                continue;
            }
            let mut token = token;
            token.special = true;
            updates.push(token);
        }
    }
    backend.add_tokens(&updates)?;
    if is_llama || is_neox {
        let enabled = |key: &str, default| config.and_then(|c| c.get(key)).map_or(default, truth);
        let resolve = |index: usize, enabled: bool| -> Result<Option<(&str, u32)>> {
            if !enabled {
                return Ok(None);
            }
            let name = attributes[index]
                .first()
                .ok_or(Error::Invalid("enabled class special token is absent"))?
                .content;
            let id = backend
                .token_id(name)
                .or_else(|| {
                    attributes[2]
                        .first()
                        .and_then(|v| backend.token_id(v.content))
                })
                .ok_or(Error::Invalid(
                    "class special token and unknown fallback have no ID",
                ))?;
            Ok(Some((name, id)))
        };
        let bos = resolve(0, enabled("add_bos_token", is_llama))?;
        let eos = resolve(1, enabled("add_eos_token", false))?;
        backend.set_llama_post(bos, eos)?;
    }
    Ok((backend, specials))
}
pub(crate) fn write(
    files: &Files,
    source: &mut SourceDirectory,
    config: &Config,
    budget: &mut JsonBudget,
    out: &mut Metadata,
    limits: &Limits,
    profile: SourceProfile,
) -> Result<(Option<usize>, bool)> {
    let map = source
        .read_optional("special_tokens_map.json")?
        .map(|raw| budget.parse(raw))
        .transpose()?;
    let raw = files
        .tokenizer
        .as_ref()
        .ok_or(Error::Invalid("tokenizer.json is absent"))?;
    let root = object(raw)?;
    let model = object(
        root.get("model")
            .ok_or(Error::Invalid("tokenizer model is absent"))?,
    )?;
    let gpt_family = matches!(
        config.family,
        crate::config::Architecture::Gpt2
            | crate::config::Architecture::GptNeoX
            | crate::config::Architecture::Bloom
            | crate::config::Architecture::Falcon
    );
    // These source subclasses directly use the GPT2 vocabulary routine,
    // whose native Tokenizers deserializer admits legacy untagged BPE.
    // Keep the existing mandatory-type boundary for other source paths.
    let bpe = match model.get("type") {
        Some(value) => text(value)? == "BPE",
        None if gpt_family => true,
        None => return Err(Error::Invalid("tokenizer model type is absent")),
    };
    let ignore = model.get("ignore_merges").is_some_and(truth);
    let fallback = model.get("byte_fallback").is_some_and(truth);
    let sequence = matches!(
        root.get("decoder")
            .and_then(Json::as_object)
            .and_then(|o| o.get("type"))
            .map(text),
        Some(Ok("Sequence"))
    );
    let llama_hf = !gpt_family && bpe && fallback && sequence && !(ignore && !fallback);
    let (mut backend, specials) = prepare(files, map.as_ref(), limits, profile, Some(config))?;
    let (mut tokens, mut scores, mut types) = (Vec::new(), Vec::new(), Vec::new());
    if llama_hf {
        let added = backend
            .added
            .iter()
            .filter(|p| p.0 as usize >= backend.base_size)
            .count();
        let bound = backend
            .base_size
            .checked_add(added)
            .filter(|n| *n <= limits.max_vocabulary)
            .ok_or(Error::Limit("Llama HF vocabulary size"))?;
        reserve(&mut tokens, bound, "HF tokens")?;
        reserve(&mut scores, bound, "HF scores")?;
        reserve(&mut types, bound, "HF types")?;
        for index in 0..backend.base_size {
            let token = backend
                .by_id
                .get(index)
                .and_then(|v| *v)
                .ok_or(Error::Invalid("noncontiguous Llama HF base IDs"))?;
            let byte = token.as_bytes();
            let byte_token = byte.len() == 6
                && byte.starts_with(b"<0x")
                && byte.ends_with(b">")
                && byte[3..5].iter().all(u8::is_ascii_hexdigit);
            let kind = if byte_token {
                6
            } else if specials.contains(&token) {
                3
            } else {
                1
            };
            tokens.push(Value::String(bytes(byte)?));
            scores.push(Value::F32(-1000.0));
            types.push(Value::I32(kind));
        }
        for &(id, token) in &backend.added {
            if (id as usize) < backend.base_size {
                continue;
            }
            tokens.push(Value::String(bytes(token.content.as_bytes())?));
            scores.push(Value::F32(-1000.0));
            types.push(Value::I32(if specials.contains(&token.content) {
                3
            } else {
                4
            }));
        }
        if tokens.len() != bound {
            return Err(Error::Invalid("Llama HF vocabulary count"));
        }
        out.string("tokenizer.ggml.model", "llama")?;
        out.string("tokenizer.ggml.pre", "default")?;
        arrays(out, tokens, Some(scores), types)?;
        Ok((Some(bound), false))
    } else {
        let bound = match config.get("vocab_size")? {
            Some(value) => usize::try_from(integer(value)?)
                .map_err(|_| Error::Invalid("negative GPT2 vocabulary size"))?,
            None => backend.by_id.iter().filter(|v| v.is_some()).count(),
        };
        if bound > limits.max_vocabulary {
            return Err(Error::Limit("GPT2 vocabulary size"));
        }
        if backend
            .by_id
            .iter()
            .enumerate()
            .any(|(i, v)| v.is_some() && i >= bound)
        {
            return Err(Error::Invalid("HF vocabulary ID exceeds configured size"));
        }
        let ids = backend.encode_with_special(crate::hf_probe::PROBE, true)?;
        let mut list = String::new();
        list.try_reserve(
            ids.len()
                .checked_mul(12)
                .and_then(|n| n.checked_add(2))
                .ok_or(Error::Limit("probe ID representation"))?,
        )
        .map_err(|_| Error::Limit("probe ID representation"))?;
        list.push('[');
        for (index, id) in ids.iter().enumerate() {
            if index > 0 {
                list.push_str(", ");
            }
            list.push_str(&id.to_string());
        }
        list.push(']');
        let hash = llama_conformance::digest::sha256(list.as_bytes());
        let kind = crate::hf_probe::HASHES
            .iter()
            .rev()
            .find(|(digest, _)| *digest == hash)
            .map(|(_, kind)| *kind)
            .ok_or(Error::Unsupported(
                "frozen converter does not recognize the HF tokenizer probe hash",
            ))?;
        reserve(&mut tokens, bound, "GPT2 output tokens")?;
        reserve(&mut types, bound, "GPT2 token types")?;
        for index in 0..bound {
            let token = backend.by_id.get(index).and_then(|v| *v);
            let Some(token) = token else {
                tokens.push(Value::String(format!("[PAD{index}]").into_bytes()));
                types.push(Value::I32(5));
                continue;
            };
            let added = backend
                .added
                .iter()
                .find(|p| p.0 as usize == index)
                .map(|p| p.1);
            let (mut token, kind) = if let Some(added) = added {
                let token = if added.normalized {
                    string(token)?
                } else {
                    let ids = backend.encode(token)?;
                    backend.decode(&ids)?
                };
                let kind = if added.special || looks_special(&token) {
                    3
                } else {
                    4
                };
                (token, kind)
            } else {
                (string(token)?, 1)
            };
            if kind == 4 {
                token = token.replace('▁', " ");
            }
            tokens.push(Value::String(token.into_bytes()));
            types.push(Value::I32(kind));
        }
        out.string("tokenizer.ggml.model", "gpt2")?;
        out.string("tokenizer.ggml.pre", kind)?;
        arrays(out, tokens, None, types)?;
        Ok((None, true))
    }
}
fn arrays(
    out: &mut Metadata,
    tokens: Vec<Value>,
    scores: Option<Vec<Value>>,
    types: Vec<Value>,
) -> Result<()> {
    if !tokens.is_empty() {
        out.put(
            "tokenizer.ggml.tokens",
            Value::Array {
                element_type: 8,
                values: tokens,
            },
        )?;
    }
    if let Some(scores) = scores
        && !scores.is_empty()
    {
        out.put(
            "tokenizer.ggml.scores",
            Value::Array {
                element_type: 6,
                values: scores,
            },
        )?;
    }
    if !types.is_empty() {
        out.put(
            "tokenizer.ggml.token_type",
            Value::Array {
                element_type: 5,
                values: types,
            },
        )?;
    }
    Ok(())
}
