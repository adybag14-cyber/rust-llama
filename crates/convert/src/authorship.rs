//! Frozen model-card/config authorship precedence and ordered raw GGUF metadata.
//! Values remain borrowed graph nodes until overrides finish. Ignored aliases
//! and cycles therefore do not force expansion or premature type rejection.
use crate::codec::SourceProfile;
use crate::model_card::{Card, Collection, Node, Scalar};
use crate::util::{JsonBudget, Metadata, object, real, text, truth};
use crate::{Error, Limits, Result, bytes, names, reserve, string};
use llama_checkpoint::{Number, Object, SourceDirectory, Value as Json};
use llama_gguf::RawValue as Value;

#[derive(Debug)]
enum Atom<'a> {
    Json(&'a Json),
    Card(&'a Card, usize),
    Pair(&'a Card, usize, usize),
    Text(String),
}
impl<'a> Atom<'a> {
    fn null(&self) -> bool {
        matches!(self, Self::Json(Json::Null) | Self::Card(_, _))
            && match self {
                Self::Json(v) => matches!(v, Json::Null),
                Self::Card(c, i) => matches!(c.nodes[*i], Node::Scalar(Scalar::Null)),
                _ => false,
            }
    }
    fn text(&self) -> Option<&str> {
        match self {
            Self::Json(v) => v.as_text()?.as_str(),
            Self::Card(c, i) => match &c.nodes[*i] {
                Node::Scalar(Scalar::Text(v)) => Some(v),
                _ => None,
            },
            Self::Text(v) => Some(v),
            Self::Pair(..) => None,
        }
    }
    fn bytes(&self) -> Option<&[u8]> {
        if let Some(value) = self.text() {
            return Some(value.as_bytes());
        }
        match self {
            Self::Card(c, i) => match &c.nodes[*i] {
                Node::Scalar(Scalar::Bytes(v)) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }
    fn map(&self) -> bool {
        match self {
            Self::Json(Json::Object(_)) => true,
            Self::Card(c, i) => matches!(c.nodes[*i], Node::Mapping(Collection::Mapping, _)),
            _ => false,
        }
    }
    fn list(&self) -> bool {
        match self {
            Self::Json(Json::Array(_)) => true,
            Self::Card(c, i) => matches!(
                c.nodes[*i],
                Node::Sequence(Collection::Sequence, _)
                    | Node::Mapping(Collection::OrderedMap | Collection::Pairs, _)
            ),
            _ => false,
        }
    }
    fn get(&self, key: &str) -> Option<Atom<'a>> {
        match self {
            Self::Json(v) => (*v).as_object()?.get(key).map(Atom::Json),
            Self::Card(c, i) if self.map() => (*c).map_get(*i, key).map(|i| Atom::Card(c, i)),
            _ => None,
        }
    }
    fn items(&self) -> Result<Vec<Atom<'a>>> {
        let mut out = Vec::new();
        match self {
            Self::Json(Json::Array(v)) => {
                reserve(&mut out, v.len(), "metadata list")?;
                out.extend(v.iter().map(Atom::Json));
            }
            Self::Json(Json::Object(v)) => {
                reserve(&mut out, v.len(), "metadata mapping keys")?;
                for (key, _) in v.entries() {
                    out.push(Atom::Text(string(
                        key.as_str()
                            .ok_or(Error::Invalid("metadata key is not Unicode"))?,
                    )?));
                }
            }
            Self::Card(c, i) => match &c.nodes[*i] {
                Node::Sequence(Collection::Sequence, values) => {
                    reserve(&mut out, values.len(), "model-card list")?;
                    out.extend(values.iter().map(|&i| Atom::Card(c, i)));
                }
                Node::Mapping(Collection::OrderedMap | Collection::Pairs, values) => {
                    reserve(&mut out, values.len(), "model-card pairs")?;
                    out.extend(values.iter().map(|&(k, v)| Atom::Pair(c, k, v)));
                }
                Node::Mapping(_, values) => {
                    reserve(&mut out, values.len(), "model-card keys")?;
                    out.extend(values.iter().map(|&(k, _)| Atom::Card(c, k)));
                }
                _ => return self.text_items(),
            },
            Self::Pair(c, k, v) => {
                reserve(&mut out, 2, "YAML pair")?;
                out.push(Atom::Card(c, *k));
                out.push(Atom::Card(c, *v));
            }
            _ => return self.text_items(),
        }
        Ok(out)
    }
    fn text_items(&self) -> Result<Vec<Atom<'a>>> {
        let value = self
            .text()
            .ok_or(Error::Invalid("metadata value is not iterable"))?;
        let mut items = Vec::new();
        reserve(
            &mut items,
            value.chars().count(),
            "metadata string iteration",
        )?;
        for ch in value.chars() {
            let mut item = String::new();
            item.try_reserve_exact(ch.len_utf8())
                .map_err(|_| Error::Limit("metadata character"))?;
            item.push(ch);
            items.push(Atom::Text(item));
        }
        Ok(items)
    }
    fn contains(&self, key: &str) -> Result<bool> {
        if self.map() {
            return Ok(self.get(key).is_some());
        }
        if let Some(value) = self.text() {
            return Ok(value.contains(key));
        }
        if matches!(self,Self::Card(c,i) if matches!(c.nodes[*i],Node::Scalar(Scalar::Bytes(_)))) {
            return Err(Error::Invalid("text membership in a byte metadata value"));
        }
        Ok(self.items()?.iter().any(|v| v.text() == Some(key)))
    }
    fn field(&self, key: &str) -> Result<Option<Atom<'a>>> {
        if !self.contains(key)? {
            return Ok(None);
        }
        self.get(key)
            .map(Some)
            .ok_or(Error::Invalid("metadata entry is not a keyed mapping"))
    }
    fn integer(&self) -> Result<i32> {
        let value = match self {
            Self::Json(Json::Bool(v)) => return Ok(i32::from(*v)),
            Self::Json(Json::Number(Number::Integer(v))) => v.as_i64(),
            Self::Card(c, i) => match &c.nodes[*i] {
                Node::Scalar(Scalar::Bool(v)) => return Ok(i32::from(*v)),
                Node::Scalar(Scalar::Integer(v)) => v.parse::<i64>().ok(),
                _ => return Err(Error::Invalid("INT32 metadata requires an integer")),
            },
            _ => return Err(Error::Invalid("INT32 metadata requires an integer")),
        }
        .ok_or(Error::Invalid("INT32 metadata overflow"))?;
        i32::try_from(value).map_err(|_| Error::Invalid("INT32 metadata overflow"))
    }
    fn float(&self) -> Result<f32> {
        let value = match self {
            Self::Json(v) => real(v)?,
            Self::Card(c, i) => match &c.nodes[*i] {
                Node::Scalar(Scalar::Bool(v)) => f64::from(*v),
                Node::Scalar(Scalar::Integer(v)) => v
                    .parse::<f64>()
                    .map_err(|_| Error::Invalid("numeric metadata integer"))?,
                Node::Scalar(Scalar::Float(v)) => *v,
                _ => return Err(Error::Invalid("FLOAT32 metadata requires a number")),
            },
            _ => return Err(Error::Invalid("FLOAT32 metadata requires a number")),
        };
        let result = value as f32;
        if value.is_finite() && !result.is_finite() {
            return Err(Error::Invalid("FLOAT32 metadata overflow"));
        }
        Ok(result)
    }
    fn kind(&self) -> Result<u32> {
        if self.bytes().is_some() {
            return Ok(8);
        }
        Ok(match self {
            Self::Json(Json::Bool(_)) => 7,
            Self::Json(Json::Number(Number::Integer(_))) => 5,
            Self::Json(Json::Number(Number::Float(_))) => 6,
            Self::Json(Json::Array(_)) | Self::Pair(..) => 9,
            Self::Card(c, i) => match &c.nodes[*i] {
                Node::Scalar(Scalar::Bool(_)) => 7,
                Node::Scalar(Scalar::Integer(_)) => 5,
                Node::Scalar(Scalar::Float(_)) => 6,
                Node::Sequence(..)
                | Node::Mapping(Collection::OrderedMap | Collection::Pairs, _) => 9,
                _ => return Err(Error::Invalid("unsupported GGUF metadata element type")),
            },
            _ => return Err(Error::Invalid("unsupported GGUF metadata element type")),
        })
    }
    fn value(&self, depth: usize) -> Result<Value> {
        if depth >= 128 {
            return Err(Error::Limit("selected metadata array depth or cycle"));
        }
        Ok(match self.kind()? {
            8 => Value::String(bytes(self.bytes().expect("string kind"))?),
            7 => Value::Bool(self.integer()? != 0),
            5 => Value::I32(self.integer()?),
            6 => Value::F32(self.float()?),
            9 => array_value(&self.items()?, depth + 1)?
                .ok_or(Error::Invalid("nested empty metadata array"))?,
            _ => unreachable!(),
        })
    }
}
fn array_value(values: &[Atom<'_>], depth: usize) -> Result<Option<Value>> {
    let Some(first) = values.first() else {
        return Ok(None);
    };
    let kind = first.kind()?;
    let mut items = Vec::new();
    reserve(&mut items, values.len(), "GGUF metadata array")?;
    for value in values {
        if value.kind()? != kind {
            return Err(Error::Invalid("heterogeneous GGUF metadata array"));
        }
        items.push(value.value(depth)?);
    }
    Ok(Some(Value::Array {
        element_type: kind,
        values: items,
    }))
}
#[derive(Default)]
struct Fields<'a>(Vec<(&'static str, Atom<'a>)>);
impl<'a> Fields<'a> {
    fn get(&self, key: &str) -> Option<&Atom<'a>> {
        self.0
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, v)| v)
            .filter(|v| !v.null())
    }
    fn put(&mut self, key: &'static str, value: Atom<'a>) -> Result<()> {
        if let Some((_, current)) = self.0.iter_mut().find(|(name, _)| *name == key) {
            *current = value;
        } else {
            reserve(&mut self.0, 1, "authorship fields")?;
            self.0.push((key, value));
        }
        Ok(())
    }
    fn fill(&mut self, key: &'static str, value: Option<String>) -> Result<()> {
        if self.get(key).is_none()
            && let Some(value) = value
        {
            self.put(key, Atom::Text(value))?;
        }
        Ok(())
    }
}
enum Entry<'a> {
    Raw(Atom<'a>),
    Inferred(Fields<'a>),
}
enum Entries<'a> {
    Prepared(Vec<Entry<'a>>),
    Raw(Atom<'a>),
}
enum Values<'a> {
    Prepared(Vec<Atom<'a>>),
    Raw(Atom<'a>),
}
const ENTRY_KEYS: [&str; 9] = [
    "name",
    "author",
    "version",
    "organization",
    "description",
    "url",
    "doi",
    "uuid",
    "repo_url",
];
const STRING_FIELDS: [&str; 20] = [
    "name",
    "author",
    "version",
    "organization",
    "finetune",
    "basename",
    "description",
    "quantized_by",
    "size_label",
    "license",
    "license_name",
    "license_link",
    "url",
    "doi",
    "uuid",
    "repo_url",
    "source_url",
    "source_doi",
    "source_uuid",
    "source_repo_url",
];
const SAMPLING: [(&str, &str, bool); 12] = [
    ("sampling_sequence", "sequence", false),
    ("sampling_top_k", "top_k", true),
    ("sampling_top_p", "top_p", false),
    ("sampling_min_p", "min_p", false),
    ("sampling_xtc_probability", "xtc_probability", false),
    ("sampling_xtc_threshold", "xtc_threshold", false),
    ("sampling_temp", "temperature", false),
    ("sampling_penalty_last_n", "penalty_last_n", true),
    ("sampling_penalty_repeat", "penalty_repeat", false),
    ("sampling_mirostat", "mirostat", true),
    ("sampling_mirostat_tau", "mirostat_tau", false),
    ("sampling_mirostat_eta", "mirostat_eta", false),
];
fn output_key(field: &str) -> String {
    if let Some(rest) = field.strip_prefix("sampling_") {
        return format!("general.sampling.{rest}");
    }
    match field {
        "license_name" => String::from("general.license.name"),
        "license_link" => String::from("general.license.link"),
        field if field.starts_with("source_") => format!("general.source.{}", &field[7..]),
        _ => format!("general.{field}"),
    }
}
fn infer_id(
    fields: &mut Fields<'_>,
    value: &str,
    params: u64,
    profile: SourceProfile,
) -> Result<()> {
    let parts = names::components(value, params, profile)?;
    fields.fill("name", Some(names::title(&parts.name, profile)?))?;
    fields.fill(
        "organization",
        parts
            .organization
            .as_deref()
            .map(|v| names::title(v, profile))
            .transpose()?,
    )?;
    fields.fill("basename", parts.basename)?;
    fields.fill("finetune", parts.finetune)?;
    fields.fill("version", parts.version)?;
    fields.fill("size_label", parts.size)
}
fn model_entry<'a>(value: Atom<'a>, params: u64, profile: SourceProfile) -> Result<Entry<'a>> {
    if value.map() {
        return Ok(Entry::Raw(value));
    }
    let mut fields = Fields::default();
    if let Some(id) = value.text() {
        let url =
            id.starts_with("http://") || id.starts_with("https://") || id.starts_with("ssh://");
        let mut model_id = Some(id);
        if url {
            fields.put("repo_url", Atom::Text(string(id)?))?;
            model_id = id
                .strip_prefix("https://huggingface.co/")
                .or_else(|| id.strip_prefix("http://huggingface.co/"))
                .filter(|id| id.split('/').count() == 2 && id.split('/').all(|s| !s.is_empty()));
        }
        if let Some(id) = model_id {
            let parts = names::components(id, params, profile)?;
            fields.put("name", Atom::Text(names::title(&parts.name, profile)?))?;
            if let Some(org) = parts.organization.as_deref() {
                fields.put("organization", Atom::Text(names::title(org, profile)?))?;
                if !url {
                    fields.put(
                        "repo_url",
                        Atom::Text(format!("https://huggingface.co/{org}/{}", parts.name)),
                    )?;
                }
            }
            if let Some(version) = parts.version {
                fields.put("version", Atom::Text(version))?;
            }
        }
    }
    Ok(Entry::Inferred(fields))
}
fn card_entries<'a>(
    card: &'a Card,
    keys: &[&str],
    params: u64,
    profile: SourceProfile,
) -> Result<Option<Entries<'a>>> {
    let Some(id) = keys.iter().find_map(|k| card.get(k)) else {
        return Ok(None);
    };
    let atom = Atom::Card(card, id);
    let mut values = Vec::new();
    if atom.text().is_some() {
        reserve(&mut values, 1, "model-card entry")?;
        values.push(atom);
    } else if atom.list() {
        values = atom.items()?;
    }
    let mut entries = Vec::new();
    reserve(&mut entries, values.len(), "model-card entries")?;
    for value in values {
        entries.push(model_entry(value, params, profile)?);
    }
    Ok(Some(Entries::Prepared(entries)))
}
fn card_values<'a>(card: &'a Card, keys: &[&str]) -> Result<Option<Values<'a>>> {
    let mut result = None;
    for key in keys {
        let Some(id) = card.get(key) else {
            continue;
        };
        let atom = Atom::Card(card, id);
        if atom.null() {
            continue;
        }
        let values = result.get_or_insert_with(Vec::new);
        if atom.text().is_some() {
            reserve(values, 1, "model-card tag")?;
            values.push(atom);
        } else if atom.list() {
            let next = atom.items()?;
            reserve(values, next.len(), "model-card tags")?;
            values.extend(next);
        }
    }
    Ok(result.map(Values::Prepared))
}
fn write_entries(out: &mut Metadata, prefix: &str, entries: &Entries<'_>) -> Result<()> {
    match entries {
        Entries::Prepared(entries) => {
            out.uint(
                &format!("general.{prefix}.count"),
                u32::try_from(entries.len()).map_err(|_| Error::Limit("authorship entry count"))?,
            )?;
            for (index, entry) in entries.iter().enumerate() {
                for key in ENTRY_KEYS {
                    let owned;
                    let value = match entry {
                        Entry::Inferred(fields) => fields.get(key),
                        Entry::Raw(atom) => {
                            owned = atom.field(key)?;
                            owned.as_ref()
                        }
                    };
                    if let Some(value) = value {
                        let raw = value
                            .bytes()
                            .ok_or(Error::Invalid("authorship entry string type"))?;
                        out.bytes(&format!("general.{prefix}.{index}.{key}"), raw)?;
                    }
                }
            }
        }
        Entries::Raw(atom) => {
            let values = atom.items()?;
            out.uint(
                &format!("general.{prefix}.count"),
                u32::try_from(values.len()).map_err(|_| Error::Limit("authorship entry count"))?,
            )?;
            for (index, entry) in values.iter().enumerate() {
                for key in ENTRY_KEYS {
                    if let Some(value) = entry.field(key)? {
                        out.bytes(
                            &format!("general.{prefix}.{index}.{key}"),
                            value
                                .bytes()
                                .ok_or(Error::Invalid("authorship entry string type"))?,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}
fn write_values(out: &mut Metadata, key: &str, values: &Values<'_>) -> Result<()> {
    let owned;
    let values = match values {
        Values::Prepared(values) => values,
        Values::Raw(atom) => {
            // add_array skips any empty value before validating Sequence.
            if atom.map() && atom.items()?.is_empty() {
                return Ok(());
            }
            if atom.map() {
                return Err(Error::Invalid("GGUF metadata array requires a sequence"));
            }
            owned = atom.items()?;
            &owned
        }
    };
    if let Some(value) = array_value(values, 0)? {
        out.put(key, value)?;
    }
    Ok(())
}
fn size_label(params: u64) -> String {
    let (scale, suffix) = if params > 1_000_000_000_000 {
        (1e-12, "T")
    } else if params > 1_000_000_000 {
        (1e-9, "B")
    } else if params > 1_000_000 {
        (1e-6, "M")
    } else {
        (1e-3, "K")
    };
    let scaled = params as f64 * scale;
    let rounded = format!("{:.0}", scaled.round_ties_even());
    let precision = 2usize.saturating_sub(rounded.trim_start_matches('0').len());
    format!("{scaled:.precision$}{suffix}")
}
/// Names are bytes because original YAML !!binary reaches GGUF strings unchanged.
/// Filename generation performs its own checked text conversion only when needed.
#[derive(Debug)]
pub(crate) struct Naming {
    pub name: Vec<u8>,
    pub basename: Option<Vec<u8>>,
    pub finetune: Option<Vec<u8>>,
    pub version: Option<Vec<u8>>,
    pub size_label: Option<Vec<u8>>,
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn write(
    source: &mut SourceDirectory,
    config: &Object,
    folder_name: &str,
    model_name: Option<&str>,
    override_path: Option<&str>,
    params: u64,
    budget: &mut JsonBudget,
    limits: &Limits,
    profile: SourceProfile,
    out: &mut Metadata,
) -> Result<Naming> {
    let card = source
        .read_optional("README.md")?
        .map(|raw| Card::parse_frontmatter(raw, &limits.yaml))
        .transpose()?;
    let generation = if let Some(raw) = source.read_optional("generation_config.json")? {
        // The frozen loader ignores malformed JSON, but Unicode decode failures
        // and our configured input/resource limits remain visible.
        std::str::from_utf8(raw)
            .map_err(|_| Error::Invalid("generation_config.json is not UTF-8"))?;
        match budget.parse(raw) {
            Ok(v) => Some(v),
            Err(Error::Checkpoint(llama_checkpoint::Error::Invalid(_))) => None,
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    let overrides = override_path
        .map(|path| source.read_optional(path))
        .transpose()?
        .flatten()
        .map(|raw| budget.parse(raw))
        .transpose()?;
    let mut fields = Fields::default();
    let (mut base_models, mut datasets, mut tags, mut languages) = (None, None, None, None);
    if let Some(card) = &card {
        let pairs = [
            ("name", "name"),
            ("author", "author"),
            ("version", "version"),
            ("organization", "organization"),
            ("description", "description"),
            ("finetune", "finetune"),
            ("basename", "basename"),
            ("size_label", "size_label"),
            ("source_url", "url"),
            ("source_doi", "doi"),
            ("source_uuid", "uuid"),
            ("source_repo_url", "repo_url"),
            ("name", "model_name"),
            ("author", "model_author"),
            ("version", "model_version"),
            ("organization", "model_organization"),
            ("description", "model_description"),
            ("finetune", "model_finetune"),
            ("basename", "model_basename"),
            ("size_label", "model_size_label"),
            ("source_url", "model_url"),
            ("source_doi", "model_doi"),
            ("source_uuid", "model_uuid"),
            ("source_repo_url", "model_repo_url"),
            ("name", "model_name"),
            ("author", "model_creator"),
            ("basename", "model_type"),
            ("license", "license"),
            ("license_name", "license_name"),
            ("license_link", "license_link"),
        ];
        for (field, key) in pairs {
            if fields.get(field).is_none()
                && let Some(id) = card.get(key)
            {
                fields.put(field, Atom::Card(card, id))?;
            }
        }
        base_models = card_entries(
            card,
            &["base_model", "base_models", "base_model_sources"],
            params,
            profile,
        )?;
        datasets = card_entries(
            card,
            &["datasets", "dataset", "dataset_sources"],
            params,
            profile,
        )?;
        tags = card_values(card, &["tags", "pipeline_tag"])?;
        languages = card_values(card, &["languages", "language"])?;
    }
    if let Some(value) = config
        .get("_name_or_path")
        .filter(|v| !matches!(v, Json::Null))
    {
        let value = text(value)?;
        if value.matches('/').count() <= 1 {
            infer_id(&mut fields, value, params, profile)?;
        }
    }
    infer_id(&mut fields, folder_name, params, profile)?;
    if let Some(generation) = generation.as_ref().filter(|v| truth(v)) {
        let generation = object(generation)?;
        for (field, key, _) in SAMPLING {
            let value = generation.get(key).or_else(|| {
                if key == "penalty_repeat" {
                    generation.get("repetition_penalty")
                } else {
                    None
                }
            });
            if let Some(value) = value {
                fields.put(field, Atom::Json(value))?;
            }
        }
    }
    if let Some(overrides) = &overrides {
        let overrides = object(overrides)?;
        for field in STRING_FIELDS {
            if field == "license" {
                continue;
            } // Frozen override list intentionally omits it.
            if let Some(value) = overrides.get(&output_key(field)) {
                fields.put(field, Atom::Json(value))?;
            }
        }
        for (field, _, _) in SAMPLING {
            if let Some(value) = overrides.get(&output_key(field)) {
                fields.put(field, Atom::Json(value))?;
            }
        }
        if let Some(value) = overrides.get("general.base_models") {
            base_models = if matches!(value, Json::Null) {
                None
            } else {
                Some(Entries::Raw(Atom::Json(value)))
            };
        }
        if let Some(value) = overrides.get("general.datasets") {
            datasets = if matches!(value, Json::Null) {
                None
            } else {
                Some(Entries::Raw(Atom::Json(value)))
            };
        }
        if let Some(value) = overrides.get("general.tags") {
            tags = if matches!(value, Json::Null) {
                None
            } else {
                Some(Values::Raw(Atom::Json(value)))
            };
        }
        if let Some(value) = overrides.get("general.languages") {
            languages = if matches!(value, Json::Null) {
                None
            } else {
                Some(Values::Raw(Atom::Json(value)))
            };
        }
    }
    if let Some(name) = model_name {
        fields.put("name", Atom::Text(string(name)?))?;
    }
    fields.fill("name", Some(string(folder_name)?))?;
    if params > 0 {
        fields.fill("size_label", Some(size_label(params)))?;
    }
    for (field, _, integer) in SAMPLING {
        if let Some(value) = fields.get(field) {
            let key = output_key(field);
            if field == "sampling_sequence" {
                out.bytes(
                    &key,
                    value
                        .bytes()
                        .ok_or(Error::Invalid("sampling sequence requires a string"))?,
                )?;
            } else if integer {
                out.put(&key, Value::I32(value.integer()?))?;
            } else {
                out.float(&key, value.float()?)?;
            }
        }
    }
    for field in STRING_FIELDS {
        if let Some(value) = fields.get(field) {
            let key = output_key(field);
            if field == "license" && value.list() {
                let parts = value.items()?;
                let mut joined = String::new();
                for (index, part) in parts.iter().enumerate() {
                    let part = part
                        .text()
                        .ok_or(Error::Invalid("license list requires Unicode strings"))?;
                    joined
                        .try_reserve(part.len() + usize::from(index > 0))
                        .map_err(|_| Error::Limit("license list bytes"))?;
                    if index > 0 {
                        joined.push(',');
                    }
                    joined.push_str(part);
                }
                out.string(&key, &joined)?;
            } else {
                out.bytes(
                    &key,
                    value
                        .bytes()
                        .ok_or(Error::Invalid("authorship metadata string type"))?,
                )?;
            }
        }
    }
    if let Some(entries) = &base_models {
        write_entries(out, "base_model", entries)?;
    }
    if let Some(entries) = &datasets {
        write_entries(out, "dataset", entries)?;
    }
    if let Some(values) = &tags {
        write_values(out, "general.tags", values)?;
    }
    if let Some(values) = &languages {
        write_values(out, "general.languages", values)?;
    }
    let naming = |key: &str| -> Result<Option<Vec<u8>>> {
        fields
            .get(key)
            .map(|v| {
                bytes(
                    v.bytes()
                        .ok_or(Error::Invalid("naming field string type"))?,
                )
            })
            .transpose()
    };
    Ok(Naming {
        name: naming("name")?.ok_or(Error::Invalid("model name is absent"))?,
        basename: naming("basename")?,
        finetune: naming("finetune")?,
        version: naming("version")?,
        size_label: naming("size_label")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn get<'a>(value: &'a Json, key: &str) -> &'a Json {
        value.as_object().unwrap().get(key).unwrap()
    }
    fn raw(value: &Json) -> Vec<u8> {
        text(value)
            .unwrap()
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect()
    }
    #[test]
    fn original_authorship_precedence_and_raw_ordered_headers() {
        for (profile, data) in [
            (
                SourceProfile::WindowsTorch211Numpy246,
                include_bytes!("../tests/fixtures/authorship/windows-reference.json").as_slice(),
            ),
            (
                SourceProfile::LinuxTorch211Numpy226,
                include_bytes!("../tests/fixtures/authorship/linux-reference.json").as_slice(),
            ),
        ] {
            let reference =
                llama_checkpoint::json::parse(data, llama_checkpoint::Limits::default()).unwrap();
            let mut accepted = 0;
            let mut rejected = 0;
            for case in get(&reference, "cases").as_array().unwrap() {
                let name = text(get(case, "name")).unwrap();
                let folder = std::env::temp_dir().join(format!(
                    "llama-convert-authorship-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir(&folder).unwrap();
                for file in get(case, "files").as_array().unwrap() {
                    let dest = folder.join(text(get(file, "path")).unwrap());
                    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
                    std::fs::write(dest, raw(get(file, "hex"))).unwrap();
                }
                let result = (|| -> Result<Vec<u8>> {
                    let limits = Limits::default();
                    let mut source = SourceDirectory::open(&folder, limits.sidecars.clone())?;
                    let mut budget = JsonBudget::new(limits.json.clone());
                    let config = source
                        .read_optional("config.json")?
                        .map(|v| budget.parse(v))
                        .transpose()?
                        .unwrap_or(budget.parse(b"{}")?);
                    let mut out = Metadata::default();
                    out.string("general.architecture", "llama")?;
                    out.string("general.type", "model")?;
                    fn optional(v: &Json) -> Option<&str> {
                        if matches!(v, Json::Null) {
                            None
                        } else {
                            Some(text(v).unwrap())
                        }
                    }
                    write(
                        &mut source,
                        object(&config)?,
                        text(get(case, "folder_name"))?,
                        optional(get(case, "model_name")),
                        optional(get(case, "override_path")),
                        crate::util::unsigned(get(case, "total_params"))?,
                        &mut budget,
                        &limits,
                        profile,
                        &mut out,
                    )?;
                    let items = out
                        .0
                        .iter()
                        .map(|(k, v)| (k.as_bytes(), v))
                        .collect::<Vec<_>>();
                    let layout = llama_gguf::write_raw_header(
                        &mut std::io::sink(),
                        &items,
                        &[],
                        &limits.gguf,
                    )?;
                    let mut actual = Vec::new();
                    actual
                        .try_reserve_exact(layout.data_offset as usize)
                        .map_err(|_| Error::Limit("test output"))?;
                    llama_gguf::write_raw_header(&mut actual, &items, &[], &limits.gguf)?;
                    actual.truncate(layout.encoded_header_bytes as usize);
                    Ok(actual)
                })();
                std::fs::remove_dir_all(&folder).unwrap();
                if get(case, "ok") == &Json::Bool(false) {
                    assert!(result.is_err(), "{profile:?}/{name}: native rejected");
                    rejected += 1;
                } else {
                    let actual = result.unwrap_or_else(|e| panic!("{profile:?}/{name}: {e}"));
                    assert_eq!(actual, raw(get(case, "header_hex")), "{profile:?}/{name}");
                    accepted += 1;
                }
            }
            assert_eq!((accepted, rejected), (95, 17));
        }
    }
}
