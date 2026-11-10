//! Dense Llama vocabulary emission using retained sidecars.
use crate::codec::SourceProfile;
use crate::config::Config;
use crate::util::{JsonBudget, Metadata, integer, object, text, truth};
use crate::{Error, Limits, Result, reserve, string};
use crate::{sentencepiece, special};
use llama_checkpoint::{Number, SourceDirectory, Value as Json};
use llama_gguf::RawValue as Value;

pub(crate) struct Files {
    pub tokenizer: Option<Json>,
    pub config: Option<Json>,
    pub added: Option<Json>,
}
impl Files {
    pub fn load(source: &mut SourceDirectory, budget: &mut JsonBudget) -> Result<Self> {
        fn read(
            source: &mut SourceDirectory,
            budget: &mut JsonBudget,
            name: &str,
        ) -> Result<Option<Json>> {
            source
                .read_optional(name)?
                .map(|bytes| budget.parse(bytes))
                .transpose()
        }
        Ok(Self {
            tokenizer: read(source, budget, "tokenizer.json")?,
            config: read(source, budget, "tokenizer_config.json")?,
            added: read(source, budget, "added_tokens.json")?,
        })
    }
}
pub(crate) fn looks_special(value: &str) -> bool {
    matches!(value, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (value.starts_with("<|") && value.ends_with("|>"))
        || (value.starts_with("<｜") && value.ends_with("｜>"))
        || (value.starts_with("<unused") && value.ends_with('>'))
}
fn vocab_size(value: Option<&Json>, fallback: usize, max: usize) -> Result<usize> {
    let size = match value.filter(|v| truth(v)) {
        Some(value) => usize::try_from(integer(value)?)
            .map_err(|_| Error::Invalid("negative vocabulary size"))?,
        None => fallback,
    };
    if size > max {
        return Err(Error::Limit("conversion vocabulary size"));
    }
    Ok(size)
}
fn position(value: &Json, size: usize) -> Result<Option<usize>> {
    match value {
        Json::Number(Number::Integer(value)) => {
            if !value.is_negative() {
                return match value.as_u64() {
                    Some(v) if v < (size as u64) => Ok(Some(v as usize)),
                    _ => Ok(None),
                };
            }
            let signed = value.as_i64().ok_or(Error::Invalid(
                "negative added-token index exceeds vocabulary",
            ))?;
            let target = (size as i128) + i128::from(signed);
            if target < 0 {
                return Err(Error::Invalid(
                    "negative added-token index exceeds vocabulary",
                ));
            }
            Ok(Some(target as usize))
        }
        Json::Bool(value) => {
            if usize::from(*value) >= size {
                Ok(None)
            } else {
                Ok(Some(usize::from(*value)))
            }
        }
        Json::Number(Number::Float(value)) if *value >= size as f64 => Ok(None),
        _ => Err(Error::Invalid("added token list index is not an integer")),
    }
}
fn key_position(value: &str, size: usize, profile: SourceProfile) -> Result<Option<usize>> {
    let value = crate::python_unicode::strip(value, profile)?;
    let (negative, digits) = if let Some(rest) = value.strip_prefix('-') {
        (true, rest)
    } else {
        (false, value.strip_prefix('+').unwrap_or(value))
    };
    let mut count = 0usize;
    let mut number = 0usize;
    let mut overflow = false;
    let mut previous_digit = false;
    for ch in digits.chars() {
        if ch == '_' {
            if !previous_digit {
                return Err(Error::Invalid("invalid added token integer key"));
            }
            previous_digit = false;
            continue;
        }
        let digit = crate::python_unicode::decimal(ch as u32, profile)?
            .ok_or(Error::Invalid("invalid added token integer key"))?;
        count += 1;
        if count > 4300 {
            return Err(Error::Limit("Python decimal integer key digit limit"));
        }
        previous_digit = true;
        match number
            .checked_mul(10)
            .and_then(|n| n.checked_add(digit as usize))
        {
            Some(n) => number = n,
            None => overflow = true,
        }
    }
    if count == 0 || !previous_digit {
        return Err(Error::Invalid("invalid added token integer key"));
    }
    if !negative {
        return if overflow || number >= size {
            Ok(None)
        } else {
            Ok(Some(number))
        };
    }
    if number == 0 {
        return if size == 0 { Ok(None) } else { Ok(Some(0)) };
    }
    if overflow || number > size {
        return Err(Error::Invalid(
            "negative added token key exceeds vocabulary",
        ));
    }
    Ok(Some(size - number))
}
fn sentencepiece_vocab(
    raw: &[u8],
    files: &Files,
    config: &Config,
    out: &mut Metadata,
    limits: &Limits,
    profile: SourceProfile,
) -> Result<usize> {
    let parsed = sentencepiece::Model::parse(raw, limits.max_vocabulary, limits.max_proto_fields)?;
    let _source_special_getters = parsed.special_ids;
    let size = vocab_size(
        config.find(&["vocab_size_per_layer_input", "vocab_size"])?,
        parsed.pieces.len(),
        limits.max_vocabulary,
    )?;
    let mut tokens = Vec::new();
    let mut scores = Vec::new();
    let mut types = Vec::new();
    reserve(&mut tokens, size, "SPM output tokens")?;
    reserve(&mut scores, size, "SPM scores")?;
    reserve(&mut types, size, "SPM token types")?;
    for id in 0..size {
        if let Some(piece) = parsed.pieces.get(id) {
            tokens.push(Value::String(crate::bytes(piece.text.as_bytes())?));
            // GetScore crosses the native float -> Python float -> GGUF float
            // boundary. Signaling NaNs are quieted while sign/payload survive.
            let bits = piece.score.to_bits();
            let bits = if bits & 0x7f800000 == 0x7f800000 && bits & 0x007fffff != 0 {
                bits | 0x00400000
            } else {
                bits
            };
            scores.push(Value::F32(f32::from_bits(bits)));
            types.push(Value::I32(if piece.kind == 4 {
                1
            } else {
                piece.kind as i32
            }));
        } else {
            tokens.push(Value::String(format!("[PAD{id}]").into_bytes()));
            scores.push(Value::F32(-10000.0));
            types.push(Value::I32(5));
        }
    }
    if let Some(added) = files.added.as_ref() {
        for (name, id) in object(added)?.entries() {
            let Some(id) = position(id, size)? else {
                continue;
            };
            tokens[id] = Value::String(crate::bytes(
                name.as_str()
                    .ok_or(Error::Invalid("added token is not Unicode scalar text"))?
                    .as_bytes(),
            )?);
            scores[id] = Value::F32(-1000.0);
            types[id] = Value::I32(4);
        }
    }
    if let Some(config) = files.config.as_ref()
        && let Some(added) = object(config)?.get("added_tokens_decoder")
    {
        for (id, value) in object(added)?.entries() {
            let id = id
                .as_str()
                .ok_or(Error::Invalid("added token index is not Unicode"))?;
            let index = key_position(id, size, profile)?;
            let value = object(value)?;
            let token = text(
                value
                    .get("content")
                    .ok_or(Error::Invalid("added token content is absent"))?,
            )?;
            let Some(id) = index else {
                continue;
            };
            let special = value.get("special").is_some_and(truth) || looks_special(token);
            let token = if special {
                string(token)?
            } else {
                token.replace('▁', " ")
            };
            tokens[id] = Value::String(token.into_bytes());
            scores[id] = Value::F32(-1000.0);
            types[id] = Value::I32(if special { 3 } else { 4 });
        }
    }
    out.string("tokenizer.ggml.model", "llama")?;
    out.string("tokenizer.ggml.pre", "default")?;
    for (key, kind, values) in [
        ("tokenizer.ggml.tokens", 8, tokens),
        ("tokenizer.ggml.scores", 6, scores),
        ("tokenizer.ggml.token_type", 5, types),
    ] {
        if !values.is_empty() {
            out.put(
                key,
                Value::Array {
                    element_type: kind,
                    values,
                },
            )?;
        }
    }
    Ok(size)
}
pub(crate) fn write(
    source: &mut SourceDirectory,
    config: &Config,
    budget: &mut JsonBudget,
    out: &mut Metadata,
    limits: &Limits,
    profile: SourceProfile,
) -> Result<()> {
    let files = Files::load(source, budget)?;
    if files.tokenizer.is_none() && source.read_optional("tekken.json")?.is_some() {
        return Err(Error::Unsupported("Mistral Tekken conversion vocabulary"));
    }
    if let Some(value) = files
        .config
        .as_ref()
        .map(object)
        .transpose()?
        .and_then(|o| o.get("add_prefix_space"))
        .filter(|v| !matches!(v, Json::Null))
    {
        out.boolean("tokenizer.ggml.add_space_prefix", truth(value))?;
    }
    let (bound, load_merges) = if let Some(raw) = source.read_optional("tokenizer.model")? {
        (
            Some(sentencepiece_vocab(
                raw, &files, config, out, limits, profile,
            )?),
            false,
        )
    } else {
        crate::hf_bpe::write(&files, source, config, budget, out, limits, profile)?
    };
    special::write(
        source,
        config.root(),
        files.tokenizer.as_ref(),
        files.config.as_ref(),
        budget,
        out,
        special::SpecialOptions {
            bound,
            load_merges,
            types: special::DEFAULT_TYPES,
            defaults: &[],
            profile,
        },
    )?;
    if let Some(value) = config.get("vocab_size")? {
        if matches!(integer(value), Ok(32016)) {
            special::write(
                source,
                config.root(),
                files.tokenizer.as_ref(),
                files.config.as_ref(),
                budget,
                out,
                special::SpecialOptions {
                    bound: None,
                    load_merges: false,
                    types: &["prefix", "suffix", "middle", "eot"],
                    defaults: &[
                        ("prefix", 32007),
                        ("suffix", 32008),
                        ("middle", 32009),
                        ("eot", 32010),
                    ],
                    profile,
                },
            )?;
        }
        if matches!(integer(value), Ok(49152)) {
            out.boolean("tokenizer.ggml.add_bos_token", false)?;
        }
    }
    Ok(())
}
