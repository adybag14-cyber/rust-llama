//! Frozen SpecialVocab metadata precedence. Tokenizer execution is separate.
use crate::codec::SourceProfile;
use crate::util::{JsonBudget, Metadata, object, optional_text, text, truth};
use crate::{Error, Result, reserve, string};
use llama_checkpoint::{Object, SourceDirectory, Value as Json};
const TYPES: [&str; 7] = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"];
fn get<'a>(value: Option<&'a Json>, key: &str) -> Result<Option<&'a Json>> {
    value
        .map(object)
        .transpose()
        .map(|value| value.and_then(|v| v.get(key)))
}
fn json_string(value: &str) -> Result<Json> {
    Ok(Json::Text(llama_checkpoint::Text::try_from_str(value)?))
}
fn array(value: &Json) -> Result<&[Json]> {
    value
        .as_array()
        .ok_or(Error::Invalid("tokenizer field requires an array"))
}
fn meaningful(value: Option<&Json>) -> Option<&Json> {
    value.filter(|v| truth(v))
}
#[derive(Default)]
struct State {
    ids: Vec<(&'static str, u32)>,
    add: Vec<(&'static str, bool)>,
    types: Vec<&'static str>,
    merges: Vec<String>,
    nfc: Option<bool>,
    lowercase: Option<bool>,
    strip: Option<bool>,
    clean: Option<bool>,
    metaspace: Option<String>,
}
impl State {
    fn id(&mut self, kind: &'static str, value: Option<&Json>, bound: Option<usize>) -> Result<()> {
        let Some(value) = value else {
            return Ok(());
        };
        if !matches!(
            value,
            Json::Bool(_) | Json::Number(llama_checkpoint::Number::Integer(_))
        ) {
            return Ok(());
        }
        let negative = match value {
            Json::Number(llama_checkpoint::Number::Integer(v)) => {
                v.is_negative() && v.digits() != "0"
            }
            _ => false,
        };
        if negative {
            return Err(Error::Invalid("negative special token ID"));
        }
        let raw = match value {
            Json::Bool(value) => Some(u64::from(*value)),
            Json::Number(llama_checkpoint::Number::Integer(v)) => v.as_u64(),
            _ => unreachable!(),
        };
        if bound.is_some_and(|bound| raw.is_none_or(|raw| raw >= bound as u64)) {
            return Ok(());
        }
        if self.ids.iter().any(|(name, _)| *name == kind) {
            return Ok(());
        }
        let value = u32::try_from(raw.ok_or(Error::Invalid("special token ID overflow"))?)
            .map_err(|_| Error::Invalid("special token ID exceeds UINT32"))?;
        reserve(&mut self.ids, 1, "special token IDs")?;
        self.ids.push((kind, value));
        Ok(())
    }
    fn add(&mut self, kind: &'static str, value: bool, overwrite: bool) -> Result<()> {
        if let Some((_, old)) = self.add.iter_mut().find(|(name, _)| *name == kind) {
            if overwrite {
                *old = value;
            }
            return Ok(());
        }
        reserve(&mut self.add, 1, "special token flags")?;
        self.add.push((kind, value));
        Ok(())
    }
    fn normalizer(&mut self, value: &Json, depth: usize) -> Result<()> {
        if depth > 128 {
            return Err(Error::Limit("tokenizer normalizer nesting"));
        }
        let value = object(value)?;
        match optional_text(value.get("type"))? {
            Some("NFC") => self.nfc = Some(true),
            Some("Lowercase") => self.lowercase = Some(true),
            Some("StripAccents") => self.strip = Some(true),
            Some("BertNormalizer") => {
                if let Some(value) = value.get("lowercase") {
                    self.lowercase = if matches!(value, Json::Null) {
                        None
                    } else {
                        Some(truth(value))
                    };
                }
                if let Some(value) = value.get("strip_accents") {
                    self.strip = if matches!(value, Json::Null) {
                        None
                    } else {
                        Some(truth(value))
                    };
                }
            }
            Some("Sequence") => {
                if let Some(values) = value.get("normalizers") {
                    for value in array(values)? {
                        self.normalizer(value, depth + 1)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn pretokenizer(&mut self, value: &Json, depth: usize) -> Result<()> {
        if depth > 128 {
            return Err(Error::Limit("tokenizer pretokenizer nesting"));
        }
        let value = object(value)?;
        match optional_text(value.get("type"))? {
            Some("Metaspace") => {
                if let Some(value) = value.get("prepend_scheme") {
                    self.metaspace = optional_text(Some(value))?.map(string).transpose()?;
                }
            }
            Some("Sequence") => {
                if let Some(values) = value.get("pretokenizers") {
                    for value in array(values)? {
                        self.pretokenizer(value, depth + 1)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
struct ConfigView<'a> {
    base: Option<&'a Object>,
    overrides: Vec<(&'static str, Json)>,
}
impl<'a> ConfigView<'a> {
    fn new(value: Option<&'a Json>) -> Result<Self> {
        Ok(Self {
            base: value.map(object).transpose()?,
            overrides: Vec::new(),
        })
    }
    fn active(&self) -> bool {
        self.base.is_some_and(|v| !v.is_empty()) || !self.overrides.is_empty()
    }
    fn get(&self, key: &str) -> Option<&Json> {
        self.overrides
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
            .or_else(|| self.base.and_then(|v| v.get(key)))
    }
    fn set(&mut self, key: &'static str, value: Json) -> Result<()> {
        if let Some((_, old)) = self.overrides.iter_mut().find(|(k, _)| *k == key) {
            *old = value;
            return Ok(());
        }
        reserve(&mut self.overrides, 1, "tokenizer config overrides")?;
        self.overrides.push((key, value));
        Ok(())
    }
}
fn entry_id(value: Option<&Json>, kind: &str) -> Result<Option<Json>> {
    let Some(value) = value else {
        return Ok(None);
    };
    Ok(get(get(Some(value), kind)?, "id")?.cloned())
}
fn postprocess(state: &mut State, view: &mut ConfigView<'_>, tokenizer: &Json) -> Result<()> {
    let mut bos = meaningful(view.get("bos_token")).cloned();
    let mut cls = meaningful(view.get("cls_token")).cloned();
    let mut eos = meaningful(view.get("eos_token")).cloned();
    let mut sep = meaningful(view.get("sep_token")).cloned();
    if bos.is_none() && cls.is_some() && view.active() {
        bos = cls.clone();
        view.set("bos_token", bos.clone().expect("present"))?;
    }
    if eos.is_none() && sep.is_some() && view.active() {
        eos = sep.clone();
        view.set("eos_token", eos.clone().expect("present"))?;
    }
    let Some(post) = meaningful(get(Some(tokenizer), "post_processor")?) else {
        return Ok(());
    };
    let processors = match object(post)?.get("processors") {
        Some(value) => array(value)?,
        None => std::slice::from_ref(post),
    };
    for processor in processors {
        let processor = object(processor)?;
        match optional_text(processor.get("type"))? {
            Some("ByteLevel") => {
                state.add("bos", false, false)?;
                state.add("eos", false, false)?;
            }
            Some("RobertaProcessing") => {
                state.add("bos", true, true)?;
                state.add("eos", true, true)?;
                state.add("sep", true, true)?;
                if cls.is_none() && view.active() {
                    cls = match processor.get("cls") {
                        Some(v) => array(v)?.first().cloned(),
                        None => bos.clone(),
                    };
                    view.set(
                        "cls_token",
                        cls.clone().ok_or(Error::Invalid("empty Roberta CLS"))?,
                    )?;
                }
                if sep.is_none() && view.active() {
                    sep = match processor.get("sep") {
                        Some(v) => array(v)?.first().cloned(),
                        None => eos.clone(),
                    };
                    view.set(
                        "sep_token",
                        sep.clone().ok_or(Error::Invalid("empty Roberta SEP"))?,
                    )?;
                }
            }
            Some("TemplateProcessing") => {
                let single = processor
                    .get("single")
                    .map(array)
                    .transpose()?
                    .unwrap_or(&[]);
                let pair = processor.get("pair").map(array).transpose()?.unwrap_or(&[]);
                let mut first = None;
                let mut last = None;
                if single.len() > 1 {
                    first = entry_id(single.first(), "SpecialToken")?.filter(truth);
                    last = entry_id(single.last(), "SpecialToken")?.filter(truth);
                    if let Some(value) = first.as_ref() {
                        if !view.active() {
                            bos = Some(value.clone());
                        } else if Some(value) != bos.as_ref() && Some(value) != cls.as_ref() {
                            if bos.is_none() {
                                bos = Some(value.clone());
                                view.set("bos_token", value.clone())?;
                            }
                            if cls.is_none() {
                                cls = Some(value.clone());
                                view.set("cls_token", value.clone())?;
                            }
                        }
                        state.add(
                            "bos",
                            Some(value) == bos.as_ref() || Some(value) == cls.as_ref(),
                            true,
                        )?;
                    }
                    if let Some(value) = last.as_ref() {
                        if !view.active() {
                            eos = Some(value.clone());
                        } else if Some(value) != eos.as_ref() {
                            let fallback = if !state.types.contains(&"eot") {
                                Some(("eot", "eot_token"))
                            } else if !state.types.contains(&"eom") {
                                Some(("eom", "eom_token"))
                            } else {
                                None
                            };
                            if let Some((kind, key)) = fallback {
                                reserve(&mut state.types, 1, "special token kinds")?;
                                state.types.push(kind);
                                view.set(key, eos.clone().unwrap_or(Json::Null))?;
                            }
                            view.set("eos_token", value.clone())?;
                            eos = Some(value.clone());
                        }
                        state.add("eos", Some(value) == eos.as_ref(), true)?;
                    }
                }
                if !pair.is_empty() {
                    let start = usize::from(
                        first.is_some() && entry_id(pair.first(), "SpecialToken")? == first,
                    );
                    let stop = pair.len()
                        - usize::from(
                            last.is_some() && entry_id(pair.last(), "SpecialToken")? == last,
                        );
                    if let Some(middle) = pair.get(start..stop).filter(|v| !v.is_empty()) {
                        let a = entry_id(middle.first(), "Sequence")?;
                        let b = entry_id(middle.last(), "Sequence")?;
                        if a.as_ref().map(text).transpose()? == Some("A")
                            && b.as_ref().map(text).transpose()? == Some("B")
                            && middle.len() > 2
                        {
                            let middle = &middle[1..middle.len() - 1];
                            let first_sep = entry_id(middle.first(), "SpecialToken")?.filter(truth);
                            let mut add_sep = first_sep.as_ref().is_some_and(|v| {
                                (Some(v) == sep.as_ref() || Some(v) == eos.as_ref())
                                    && last.is_none()
                            });
                            if middle.len() == 2
                                && let Some(value) =
                                    entry_id(middle.get(1), "SpecialToken")?.filter(truth)
                                && (Some(&value) == sep.as_ref() || Some(&value) == eos.as_ref())
                            {
                                add_sep = true;
                            }
                            state.add("sep", add_sep, true)?;
                            if add_sep && sep.is_none() && view.active() {
                                view.set("sep_token", eos.clone().unwrap_or(Json::Null))?;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
fn merges(state: &mut State, tokenizer: Option<&Json>) -> Result<()> {
    let Some(values) = get(get(tokenizer, "model")?, "merges")?
        .and_then(Json::as_array)
        .filter(|v| !v.is_empty())
    else {
        return Ok(());
    };
    reserve(&mut state.merges, values.len(), "tokenizer merges")?;
    match &values[0] {
        Json::Text(_) => {
            for value in values {
                state.merges.push(string(text(value)?)?);
            }
        }
        Json::Array(_) => {
            for value in values {
                let parts = array(value)?;
                if parts.len() != 2 {
                    return Err(Error::Invalid("tokenizer merge pair length"));
                }
                let left = text(&parts[0])?;
                let right = text(&parts[1])?;
                let capacity = left
                    .len()
                    .checked_add(right.len())
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|n| n.checked_add(1))
                    .ok_or(Error::Limit("tokenizer merge bytes"))?;
                let mut output = String::new();
                output
                    .try_reserve_exact(capacity)
                    .map_err(|_| Error::Limit("tokenizer merge allocation"))?;
                for ch in left.chars() {
                    output.push(if ch == ' ' { 'Ġ' } else { ch });
                }
                output.push(' ');
                for ch in right.chars() {
                    output.push(if ch == ' ' { 'Ġ' } else { ch });
                }
                state.merges.push(output);
            }
        }
        _ => return Err(Error::Invalid("unknown tokenizer merges format")),
    }
    Ok(())
}
fn chat(out: &mut Metadata, value: &Json, profile: SourceProfile) -> Result<()> {
    if matches!(value, Json::Null) {
        return Ok(());
    }
    if let Json::Text(_) = value {
        return out.string("tokenizer.chat_template", text(value)?);
    }
    let choices = array(value)?;
    let mut names = Vec::new();
    let mut default = None;
    for choice in choices {
        let choice = object(choice)?;
        let name = choice.get("name").map(text).transpose()?.unwrap_or("");
        let Some(template) = choice.get("template").filter(|v| !matches!(v, Json::Null)) else {
            continue;
        };
        let mut cleaned = String::new();
        cleaned
            .try_reserve_exact(name.len())
            .map_err(|_| Error::Limit("chat template name"))?;
        for ch in name.chars() {
            cleaned.push(if ch.is_ascii_alphanumeric() { ch } else { '_' });
        }
        if cleaned.is_empty() {
            continue;
        }
        if cleaned == "default" {
            default = Some(template);
        } else {
            if !names.contains(&cleaned) {
                reserve(&mut names, 1, "chat template names")?;
                names.push(string(&cleaned)?);
            }
            out.string(
                &format!("tokenizer.chat_template.{cleaned}"),
                text(template)?,
            )?;
        }
    }
    if !names.is_empty() {
        let order = crate::python_set::ascii_order(&names, profile)?;
        out.strings(
            "tokenizer.chat_templates",
            order.into_iter().map(|i| names[i].clone()),
        )?;
    }
    if let Some(default) = default {
        out.string("tokenizer.chat_template", text(default)?)?;
    }
    Ok(())
}
pub(crate) struct SpecialOptions<'a> {
    pub bound: Option<usize>,
    pub load_merges: bool,
    pub types: &'a [&'static str],
    pub defaults: &'a [(&'static str, u32)],
    pub profile: SourceProfile,
}
pub(crate) const DEFAULT_TYPES: &[&str] = &TYPES;
pub(crate) fn write(
    source: &mut SourceDirectory,
    raw_config: &Object,
    tokenizer: Option<&Json>,
    tokenizer_config: Option<&Json>,
    budget: &mut JsonBudget,
    out: &mut Metadata,
    options: SpecialOptions<'_>,
) -> Result<()> {
    let SpecialOptions {
        bound,
        load_merges,
        types,
        defaults,
        profile,
    } = options;
    let mut state = State::default();
    reserve(&mut state.types, types.len(), "special token kinds")?;
    state.types.extend_from_slice(types);
    if load_merges {
        merges(&mut state, tokenizer)?;
    }
    if let Some(normalizer) = meaningful(get(tokenizer, "normalizer")?) {
        state.normalizer(normalizer, 0)?;
    }
    if let Some(pre) = meaningful(get(tokenizer, "pre_tokenizer")?) {
        state.pretokenizer(pre, 0)?;
    }
    let mut view = ConfigView::new(tokenizer_config)?;
    if let Some(tokenizer) = tokenizer.filter(|v| truth(v)) {
        postprocess(&mut state, &mut view, tokenizer)?;
    }
    let mut chat_value = None;
    if view.active() {
        chat_value = view.get("chat_template").cloned();
        if let Some(raw) = source.read_optional("chat_template.jinja")? {
            let value =
                std::str::from_utf8(raw).map_err(|_| Error::Invalid("chat template UTF-8"))?;
            let value = value.replace("\r\n", "\n").replace('\r', "\n");
            chat_value = Some(json_string(&value)?);
            if let Some(entries) = source.entries_under("additional_chat_templates")? {
                let mut selected = Vec::new();
                for entry in entries {
                    let name = entry
                        .as_str()
                        .ok_or(Error::Invalid("chat template filename is not Unicode"))?;
                    if name.ends_with(".jinja") {
                        reserve(&mut selected, 1, "chat template file names")?;
                        selected.push(string(name)?);
                    }
                }
                if !selected.is_empty() {
                    // File ordering is the native directory enumeration order,
                    // as in pathlib.glob; callers need stable input directories.
                    let default = chat_value.take().expect("jinja loaded");
                    let mut choices = Vec::new();
                    reserve(&mut choices, selected.len() + 1, "chat choices")?;
                    choices.push(chat_choice("default", default, budget)?);
                    for name in selected {
                        let raw =
                            source.read_required(&format!("additional_chat_templates/{name}"))?;
                        let template = std::str::from_utf8(raw)
                            .map_err(|_| Error::Invalid("chat template UTF-8"))?
                            .replace("\r\n", "\n")
                            .replace('\r', "\n");
                        choices.push(chat_choice(
                            &name[..name.len() - 6],
                            json_string(&template)?,
                            budget,
                        )?);
                    }
                    chat_value = Some(Json::Array(choices));
                }
            }
        } else if let Some(raw) = source.read_optional("chat_template.json")? {
            let parsed = budget.parse(raw)?;
            chat_value = object(&parsed)?.get("chat_template").cloned();
        }
        if let Some(value) = view.get("clean_up_tokenization_spaces")
            && let Json::Bool(value) = value
        {
            state.clean = Some(*value);
        }
        let kinds = state.types.clone();
        for kind in kinds {
            if let Some(Json::Bool(value)) = view.get(&format!("add_{kind}_token")) {
                state.add(kind, *value, false)?;
            }
            let Some(entry) = view.get(&format!("{kind}_token")) else {
                continue;
            };
            let content = match entry {
                Json::Text(_) => Some(text(entry)?),
                Json::Object(value) => value
                    .get("content")
                    .and_then(Json::as_text)
                    .and_then(|v| v.as_str()),
                _ => None,
            };
            let Some(content) = content else {
                continue;
            };
            let added = get(tokenizer, "added_tokens")?;
            let mut found = None;
            if let Some(added) = added.filter(|v| truth(v)) {
                for token in array(added)? {
                    let token = object(token)?;
                    if token
                        .get("content")
                        .and_then(Json::as_text)
                        .and_then(|v| v.as_str())
                        == Some(content)
                    {
                        found = token.get("id");
                        break;
                    }
                }
            }
            state.id(kind, found, bound)?;
        }
    }
    let kinds = state.types.clone();
    for kind in kinds {
        let key = format!("{kind}_token_id");
        let mut value = raw_config.get(&key);
        if value.is_none_or(|v| matches!(v, Json::Null))
            && let Some(nested) = raw_config.get("text_config")
        {
            value = object(nested)?.get(&key);
        }
        state.id(kind, value, bound)?;
    }
    if load_merges
        && state.merges.is_empty()
        && let Some(raw) = source.read_optional("merges.txt")?
    {
        let value = std::str::from_utf8(raw).map_err(|_| Error::Invalid("merges.txt UTF-8"))?;
        let mut lines = value.lines();
        if !value.lines().next().unwrap_or("").trim().starts_with('#') {
            lines = value.lines();
        } else {
            lines.next();
        }
        for line in lines {
            let mut parts = line.split_whitespace();
            let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
                continue;
            };
            if parts.next().is_some() {
                continue;
            }
            reserve(&mut state.merges, 1, "merges.txt lines")?;
            state.merges.push(format!("{a} {b}"));
        }
    }
    for &(kind, id) in defaults {
        if !state.ids.iter().any(|(name, _)| *name == kind) {
            reserve(&mut state.ids, 1, "default special IDs")?;
            state.ids.push((kind, id));
        }
    }
    out.strings("tokenizer.ggml.merges", state.merges)?;
    for (kind, id) in state.ids {
        if matches!(kind, "cls" | "prefix" | "suffix" | "middle") {
            continue;
        } // No frozen writer setter.
        let suffix = match kind {
            "unk" => "unknown",
            "sep" => "seperator",
            "pad" => "padding",
            "prefix" => "fim_pre",
            "suffix" => "fim_suf",
            "middle" => "fim_mid",
            other => other,
        };
        out.uint(&format!("tokenizer.ggml.{suffix}_token_id"), id)?;
    }
    for (kind, value) in state.add {
        if matches!(kind, "bos" | "eos" | "sep") {
            out.boolean(&format!("tokenizer.ggml.add_{kind}_token"), value)?;
        }
    }
    if let Some(value) = chat_value.filter(|v| matches!(v, Json::Text(_) | Json::Array(_))) {
        chat(out, &value, profile)?;
    }
    for (key, value) in [
        ("tokenizer.ggml.normalizer.nfc", state.nfc),
        ("tokenizer.ggml.normalizer.lowercase", state.lowercase),
        ("tokenizer.ggml.normalizer.strip_accents", state.strip),
        ("tokenizer.ggml.clean_spaces", state.clean),
    ] {
        if let Some(value) = value {
            out.boolean(key, value)?;
        }
    }
    if let Some(value) = state.metaspace {
        out.string("tokenizer.ggml.metaspace.prepend_scheme", &value)?;
    }
    Ok(())
}
fn chat_choice(name: &str, template: Json, budget: &mut JsonBudget) -> Result<Json> {
    let value = text(&template)?;
    let capacity = name
        .len()
        .checked_add(value.len())
        .and_then(|n| n.checked_mul(6))
        .and_then(|n| n.checked_add(32))
        .ok_or(Error::Limit("chat metadata JSON bytes"))?;
    let mut encoded = String::new();
    encoded
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Limit("chat metadata JSON allocation"))?;
    fn quoted(out: &mut String, value: &str) -> Result<()> {
        use std::fmt::Write;
        out.push('"');
        for ch in value.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c < ' ' => write!(out, "\\u{:04x}", c as u32)
                    .map_err(|_| Error::Limit("chat JSON formatting"))?,
                c => out.push(c),
            }
        }
        out.push('"');
        Ok(())
    }
    encoded.push_str("{\"name\":");
    quoted(&mut encoded, name)?;
    encoded.push_str(",\"template\":");
    quoted(&mut encoded, value)?;
    encoded.push('}');
    budget.parse(encoded.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn get<'a>(v: &'a Json, k: &str) -> &'a Json {
        v.as_object().unwrap().get(k).unwrap()
    }
    fn txt(v: &Json) -> &str {
        text(v).unwrap()
    }
    fn raw(v: &Json) -> Vec<u8> {
        txt(v)
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect()
    }
    #[test]
    fn original_special_vocab_ordered_header_bytes() {
        for (profile, data) in [
            (
                SourceProfile::WindowsTorch211Numpy246,
                include_bytes!("../tests/fixtures/special/windows-reference.json").as_slice(),
            ),
            (
                SourceProfile::LinuxTorch211Numpy226,
                include_bytes!("../tests/fixtures/special/linux-reference.json").as_slice(),
            ),
        ] {
            let reference =
                llama_checkpoint::json::parse(data, llama_checkpoint::Limits::default()).unwrap();
            let mut accepted = 0;
            for case in get(&reference, "cases").as_array().unwrap() {
                let name = txt(get(case, "name"));
                let folder = std::env::temp_dir().join(format!(
                    "llama-convert-special-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir(&folder).unwrap();
                for file in get(case, "files").as_array().unwrap() {
                    let path = folder.join(txt(get(file, "path")));
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    std::fs::write(path, raw(get(file, "hex"))).unwrap();
                }
                let result = (|| -> Result<Vec<u8>> {
                    let mut source =
                        SourceDirectory::open(&folder, llama_checkpoint::SourceLimits::default())?;
                    let mut budget = JsonBudget::new(llama_checkpoint::Limits::default());
                    let config = source
                        .read_optional("config.json")?
                        .map(|v| budget.parse(v))
                        .transpose()?
                        .unwrap_or(budget.parse(b"{}")?);
                    let tokenizer = source
                        .read_optional("tokenizer.json")?
                        .map(|v| budget.parse(v))
                        .transpose()?;
                    let tc = source
                        .read_optional("tokenizer_config.json")?
                        .map(|v| budget.parse(v))
                        .transpose()?;
                    let bound = match get(case, "n_vocab") {
                        Json::Null => None,
                        value => Some(crate::util::unsigned(value)? as usize),
                    };
                    let mut out = Metadata::default();
                    out.string("general.architecture", "llama")?;
                    write(
                        &mut source,
                        object(&config)?,
                        tokenizer.as_ref(),
                        tc.as_ref(),
                        &mut budget,
                        &mut out,
                        SpecialOptions {
                            bound,
                            load_merges: get(case, "load_merges") == &Json::Bool(true),
                            types: DEFAULT_TYPES,
                            defaults: &[],
                            profile,
                        },
                    )?;
                    let items = out
                        .0
                        .iter()
                        .map(|(k, v)| (k.as_bytes(), v))
                        .collect::<Vec<_>>();
                    let limits = llama_gguf::Limits::default();
                    let layout =
                        llama_gguf::write_raw_header(&mut std::io::sink(), &items, &[], &limits)?;
                    let mut bytes = Vec::new();
                    bytes
                        .try_reserve_exact(layout.data_offset as usize)
                        .map_err(|_| Error::Limit("test header allocation"))?;
                    llama_gguf::write_raw_header(&mut bytes, &items, &[], &limits)?;
                    bytes.truncate(layout.encoded_header_bytes as usize);
                    Ok(bytes)
                })();
                std::fs::remove_dir_all(&folder).unwrap();
                if get(case, "ok") == &Json::Bool(false) {
                    assert!(result.is_err(), "{profile:?}/{name}");
                } else {
                    let actual = result.unwrap_or_else(|e| panic!("{profile:?}/{name}: {e}"));
                    assert_eq!(actual, raw(get(case, "header_hex")), "{profile:?}/{name}");
                    accepted += 1;
                }
            }
            assert_eq!(accepted, 82);
        }
    }
}
