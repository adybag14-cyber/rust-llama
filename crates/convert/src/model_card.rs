//! Model-card YAML events with the pinned PyYAML 1.1 scalar/merge policy.
//! Aliases are graph edges: ignored cyclic metadata is never eagerly expanded.
use crate::{Error, Result, reserve, string};
use llama_behavior::native_regex::NativeRegex;
use yaml_rust2::parser::{Event, Parser, Tag};
use yaml_rust2::scanner::{Scanner, TScalarStyle, TokenType};

#[derive(Clone, Debug)]
pub(crate) enum Scalar {
    Null,
    Bool(bool),
    Integer(String),
    Float(f64),
    Text(String),
    Bytes(Vec<u8>),
    Date(String),
    DateTime(String),
    Merge,
    Value,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Collection {
    Sequence,
    Mapping,
    OrderedMap,
    Pairs,
    Set,
}
#[derive(Debug)]
pub(crate) enum Node {
    Scalar(Scalar),
    Sequence(Collection, Vec<usize>),
    Mapping(Collection, Vec<(usize, usize)>),
}
#[derive(Debug)]
pub(crate) struct Card {
    pub nodes: Vec<Node>,
    pub root: Option<usize>,
}
#[derive(Clone, Debug)]
pub struct YamlLimits {
    pub max_input_bytes: usize,
    pub max_nodes: usize,
    pub max_depth: usize,
    pub max_scalar_bytes: usize,
    pub max_text_bytes: usize,
    pub max_aliases: usize,
    pub max_merge_edges: usize,
    pub max_integer_digits: usize,
}
impl Default for YamlLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 8 * 1024 * 1024,
            max_nodes: 200_000,
            max_depth: 128,
            max_scalar_bytes: 1024 * 1024,
            max_text_bytes: 16 * 1024 * 1024,
            max_aliases: 100_000,
            max_merge_edges: 1_000_000,
            max_integer_digits: 4300,
        }
    }
}
struct Budget<'a> {
    limits: &'a YamlLimits,
    text: usize,
    aliases: usize,
    edges: usize,
}
impl Budget<'_> {
    fn text(&mut self, n: usize) -> Result<()> {
        self.text = self
            .text
            .checked_add(n)
            .filter(|n| *n <= self.limits.max_text_bytes)
            .ok_or(Error::Limit("YAML scalar text"))?;
        if n > self.limits.max_scalar_bytes {
            return Err(Error::Limit("YAML individual scalar"));
        }
        Ok(())
    }
    fn edges(&mut self, n: usize) -> Result<()> {
        self.edges = self
            .edges
            .checked_add(n)
            .filter(|n| *n <= self.limits.max_merge_edges)
            .ok_or(Error::Limit("YAML materialized merge edges"))?;
        Ok(())
    }
}
#[derive(Debug)]
struct Frame {
    id: usize,
    key: Option<usize>,
}
impl Card {
    pub fn parse_frontmatter(input: &[u8], limits: &YamlLimits) -> Result<Self> {
        if input.len() > limits.max_input_bytes {
            return Err(Error::Limit("model-card input bytes"));
        }
        if limits.max_depth > 256 {
            return Err(Error::Limit("YAML configured depth"));
        }
        let input =
            std::str::from_utf8(input).map_err(|_| Error::Invalid("model card is not UTF-8"))?;
        // Python str.splitlines recognizes these boundaries, including CRLF as one.
        let mut lines = input
            .split([
                '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}',
                '\u{2029}',
            ])
            .peekable();
        if lines.next() != Some("---") {
            return Ok(Self {
                nodes: Vec::new(),
                root: None,
            });
        }
        let mut content = String::new();
        content
            .try_reserve(input.len() + 1)
            .map_err(|_| Error::Limit("frontmatter buffer"))?;
        // Normalize CRLF before line splitting, so a paired LF is not a blank line.
        let normalized = input.replace("\r\n", "\n");
        for line in normalized
            .split_terminator([
                '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}',
                '\u{2029}',
            ])
            .skip(1)
        {
            if line == "---" {
                break;
            }
            content.push_str(line);
            content.push('\n');
        }
        if content.is_empty() {
            content.push('\n');
        }
        let content = content.replace("- no\n", "- \"no\"\n").replace('\t', "  ");
        if content.len()
            > limits
                .max_input_bytes
                .checked_mul(2)
                .ok_or(Error::Limit("frontmatter expansion"))?
        {
            return Err(Error::Limit("normalized frontmatter bytes"));
        }
        Self::parse(&content, limits)
    }
    fn parse(content: &str, limits: &YamlLimits) -> Result<Self> {
        let mut budget = Budget {
            limits,
            text: 0,
            aliases: 0,
            edges: 0,
        };
        let mut anchors: Vec<String> = Vec::new();
        let scanner = Scanner::new(content.chars());
        let mut tokens = 0usize;
        for token in scanner {
            tokens = tokens
                .checked_add(1)
                .filter(|n| *n <= limits.max_nodes.saturating_mul(4).saturating_add(16))
                .ok_or(Error::Limit("YAML scanner tokens"))?;
            if let TokenType::Anchor(name) = token.1 {
                if anchors.contains(&name) {
                    return Err(Error::Invalid("duplicate YAML anchor"));
                }
                if anchors.len() >= limits.max_nodes {
                    return Err(Error::Limit("YAML anchor count"));
                }
                budget.text(name.len())?;
                reserve(&mut anchors, 1, "YAML anchor names")?;
                anchors.push(name);
            }
        }
        // Parser repeats lexical validation and reports a scanner error, if any.
        let patterns = Patterns::new()?;
        let mut parser = Parser::new_from_str(content);
        let mut card = Self {
            nodes: Vec::new(),
            root: None,
        };
        let mut frames: Vec<Frame> = Vec::new();
        let mut anchors: Vec<Option<usize>> = Vec::new();
        let mut documents = 0usize;
        loop {
            let (event, _) = parser
                .next_token()
                .map_err(|_| Error::Invalid("YAML syntax"))?;
            let (node, anchor, begin) = match event {
                Event::StreamStart | Event::DocumentEnd => continue,
                Event::StreamEnd => break,
                Event::DocumentStart => {
                    documents += 1;
                    if documents > 1 {
                        return Err(Error::Invalid("multiple YAML frontmatter documents"));
                    }
                    continue;
                }
                Event::Alias(anchor) => {
                    budget.aliases = budget
                        .aliases
                        .checked_add(1)
                        .filter(|n| *n <= limits.max_aliases)
                        .ok_or(Error::Limit("YAML aliases"))?;
                    let id = anchors
                        .get(anchor)
                        .copied()
                        .flatten()
                        .ok_or(Error::Invalid("undefined YAML alias"))?;
                    card.attach(id, &mut frames, &mut budget)?;
                    continue;
                }
                Event::Scalar(value, style, anchor, tag) => {
                    budget.text(value.len())?;
                    let scalar = patterns.scalar(value, style, tag, limits)?;
                    (Node::Scalar(scalar), anchor, false)
                }
                Event::SequenceStart(anchor, tag) => {
                    let kind = collection(tag, false)?;
                    (Node::Sequence(kind, Vec::new()), anchor, true)
                }
                Event::MappingStart(anchor, tag) => {
                    let kind = collection(tag, true)?;
                    (Node::Mapping(kind, Vec::new()), anchor, true)
                }
                Event::SequenceEnd | Event::MappingEnd => {
                    let frame = frames
                        .pop()
                        .ok_or(Error::Invalid("unbalanced YAML collection"))?;
                    if frame.key.is_some() {
                        return Err(Error::Invalid("missing YAML mapping value"));
                    }
                    continue;
                }
                Event::Nothing => return Err(Error::Invalid("unexpected YAML parser event")),
            };
            if card.nodes.len() >= limits.max_nodes {
                return Err(Error::Limit("YAML nodes"));
            }
            if begin && frames.len() >= limits.max_depth {
                return Err(Error::Limit("YAML nesting"));
            }
            reserve(&mut card.nodes, 1, "YAML node arena")?;
            let id = card.nodes.len();
            card.nodes.push(node);
            card.attach(id, &mut frames, &mut budget)?;
            if anchor != 0 {
                if anchor > limits.max_nodes {
                    return Err(Error::Limit("YAML anchor index"));
                }
                if anchors.len() <= anchor {
                    let n = anchor + 1 - anchors.len();
                    reserve(&mut anchors, n, "YAML anchor index")?;
                    anchors.resize(anchor + 1, None);
                }
                anchors[anchor] = Some(id);
            }
            if begin {
                reserve(&mut frames, 1, "YAML frame stack")?;
                frames.push(Frame { id, key: None });
            }
        }
        if !frames.is_empty() {
            return Err(Error::Invalid("unfinished YAML collection"));
        }
        // Source flatten_mapping removes merge edges before descending, which
        // permits safe ignored self-references without recursive alias expansion.
        for id in 0..card.nodes.len() {
            if matches!(card.nodes[id], Node::Mapping(..)) {
                card.flatten(id, 0, &mut budget)?;
            }
        }
        // Validate constructor-visible nodes before duplicate map values are
        // overwritten. Removed merge-key marker nodes are intentionally inert.
        let mut pending = Vec::new();
        if let Some(root) = card.root {
            pending.push(root);
        }
        let mut visited = Vec::new();
        reserve(&mut visited, card.nodes.len(), "YAML reachable-node marks")?;
        visited.resize(card.nodes.len(), false);
        while let Some(id) = pending.pop() {
            if visited[id] {
                continue;
            }
            visited[id] = true;
            match &card.nodes[id] {
                Node::Scalar(Scalar::Merge | Scalar::Value) => {
                    return Err(Error::Invalid("YAML merge/value tag outside a mapping key"));
                }
                Node::Sequence(_, items) => {
                    reserve(&mut pending, items.len(), "YAML graph walk")?;
                    pending.extend_from_slice(items);
                }
                Node::Mapping(_, pairs) => {
                    reserve(&mut pending, pairs.len() * 2, "YAML graph walk")?;
                    for &(k, v) in pairs {
                        pending.push(k);
                        pending.push(v);
                    }
                }
                _ => {}
            }
        }
        for id in (0..card.nodes.len()).rev() {
            card.finish_collection(id, &mut budget)?;
        }
        if card
            .root
            .is_none_or(|id| !matches!(card.nodes[id], Node::Mapping(Collection::Mapping, ..)))
        {
            card.root = None;
        }
        Ok(card)
    }
    fn attach(&mut self, id: usize, frames: &mut [Frame], budget: &mut Budget<'_>) -> Result<()> {
        if let Some(parent) = frames.last_mut() {
            match &mut self.nodes[parent.id] {
                Node::Sequence(_, values) => {
                    budget.edges(1)?;
                    reserve(values, 1, "YAML sequence")?;
                    values.push(id);
                }
                Node::Mapping(_, values) => {
                    if let Some(key) = parent.key.take() {
                        budget.edges(1)?;
                        reserve(values, 1, "YAML mapping")?;
                        values.push((key, id));
                    } else {
                        parent.key = Some(id);
                    }
                }
                _ => return Err(Error::Invalid("scalar YAML parent")),
            }
        } else if self.root.replace(id).is_some() {
            return Err(Error::Invalid("multiple YAML document roots"));
        }
        Ok(())
    }
    fn flatten(&mut self, id: usize, depth: usize, budget: &mut Budget<'_>) -> Result<()> {
        if depth > budget.limits.max_depth {
            return Err(Error::Limit("YAML merge depth"));
        }
        let mut merged = Vec::new();
        let mut at = 0usize;
        loop {
            let pair = match &self.nodes[id] {
                Node::Mapping(_, values) => values.get(at).copied(),
                _ => return Err(Error::Invalid("merge target is not a mapping")),
            };
            let Some((key, value)) = pair else {
                break;
            };
            if matches!(self.nodes[key], Node::Scalar(Scalar::Merge)) {
                if let Node::Mapping(_, values) = &mut self.nodes[id] {
                    values.remove(at);
                }
                // Change the removed tag after all merge users have been seen,
                // by retaining its node id for the final scalar validation.
                match &self.nodes[value] {
                    Node::Mapping(..) => {
                        self.flatten(value, depth + 1, budget)?;
                        let Node::Mapping(_, pairs) = &self.nodes[value] else {
                            unreachable!()
                        };
                        budget.edges(pairs.len())?;
                        reserve(&mut merged, pairs.len(), "YAML merged pairs")?;
                        merged.extend_from_slice(pairs);
                    }
                    Node::Sequence(_, items) => {
                        let mut sources = Vec::new();
                        reserve(&mut sources, items.len(), "YAML merge source list")?;
                        sources.extend_from_slice(items);
                        // Earlier maps in a merge sequence take precedence.
                        for source in sources.into_iter().rev() {
                            if !matches!(self.nodes[source], Node::Mapping(..)) {
                                return Err(Error::Invalid(
                                    "YAML merge sequence contains a non-mapping",
                                ));
                            }
                            self.flatten(source, depth + 1, budget)?;
                            let Node::Mapping(_, pairs) = &self.nodes[source] else {
                                unreachable!()
                            };
                            budget.edges(pairs.len())?;
                            reserve(&mut merged, pairs.len(), "YAML merged pairs")?;
                            merged.extend_from_slice(pairs);
                        }
                    }
                    _ => return Err(Error::Invalid("YAML merge value is not a map or sequence")),
                }
                // A merge-tag alias can be used as another key; it remains a
                // marker until final validation, which checks live references.
            } else {
                if matches!(self.nodes[key], Node::Scalar(Scalar::Value)) {
                    self.nodes[key] = Node::Scalar(Scalar::Text(string("=")?));
                }
                at += 1;
            }
        }
        if !merged.is_empty() {
            let Node::Mapping(_, pairs) = &mut self.nodes[id] else {
                unreachable!()
            };
            budget.edges(pairs.len())?;
            reserve(&mut merged, pairs.len(), "YAML explicit pairs")?;
            merged.extend_from_slice(pairs);
            *pairs = merged;
        }
        Ok(())
    }
    fn finish_collection(&mut self, id: usize, budget: &mut Budget<'_>) -> Result<()> {
        if matches!(self.nodes[id], Node::Mapping(..)) {
            let old = match &mut self.nodes[id] {
                Node::Mapping(_, pairs) => std::mem::take(pairs),
                _ => unreachable!(),
            };
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            reserve(&mut pairs, old.len(), "YAML resolved mapping")?;
            for (key, value) in old {
                if !matches!(self.nodes[key], Node::Scalar(_)) {
                    return Err(Error::Invalid("unhashable YAML mapping key"));
                }
                if let Some(pair) = pairs
                    .iter_mut()
                    .find(|(other, _)| self.key_equal(key, *other))
                {
                    pair.1 = value;
                } else {
                    pairs.push((key, value));
                }
            }
            if let Node::Mapping(_, values) = &mut self.nodes[id] {
                *values = pairs;
            }
        }
        if let Node::Sequence(kind, items) = &self.nodes[id]
            && matches!(kind, Collection::OrderedMap | Collection::Pairs)
        {
            let mut pairs = Vec::new();
            reserve(&mut pairs, items.len(), "YAML ordered pairs")?;
            for &item in items {
                let Node::Mapping(_, entries) = &self.nodes[item] else {
                    return Err(Error::Invalid("ordered YAML pair is not a map"));
                };
                if entries.len() != 1 {
                    return Err(Error::Invalid(
                        "ordered YAML pair does not contain exactly one entry",
                    ));
                }
                pairs.push(entries[0]);
            }
            budget.edges(pairs.len())?;
            self.nodes[id] = Node::Mapping(*kind, pairs);
        }
        Ok(())
    }
    fn key_equal(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        let (Node::Scalar(a), Node::Scalar(b)) = (&self.nodes[a], &self.nodes[b]) else {
            return false;
        };
        match (a, b) {
            (Scalar::Null, Scalar::Null) => true,
            (Scalar::Bool(a), Scalar::Bool(b)) => a == b,
            (Scalar::Integer(a), Scalar::Integer(b)) => a == b,
            (Scalar::Float(a), Scalar::Float(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Scalar::Text(a), Scalar::Text(b))
            | (Scalar::Date(a), Scalar::Date(b))
            | (Scalar::DateTime(a), Scalar::DateTime(b)) => a == b,
            (Scalar::Bytes(a), Scalar::Bytes(b)) => a == b,
            (Scalar::Bool(a), Scalar::Integer(b)) | (Scalar::Integer(b), Scalar::Bool(a)) => {
                b == if *a { "1" } else { "0" }
            }
            (Scalar::Bool(a), Scalar::Float(b)) | (Scalar::Float(b), Scalar::Bool(a)) => {
                f64::from(u8::from(*a)) == *b
            }
            (Scalar::Integer(a), Scalar::Float(b)) | (Scalar::Float(b), Scalar::Integer(a)) => {
                b.is_finite()
                    && b.fract() == 0.0
                    && (if *b == 0.0 {
                        a == "0"
                    } else {
                        a == &format!("{b:.0}")
                    })
            }
            _ => false,
        }
    }
    pub fn get(&self, name: &str) -> Option<usize> {
        self.root.and_then(|id| self.map_get(id, name))
    }
    pub fn map_get(&self, id: usize, name: &str) -> Option<usize> {
        let Node::Mapping(_, pairs) = self.nodes.get(id)? else {
            return None;
        };
        pairs.iter().find_map(|&(k, v)| match &self.nodes[k] {
            Node::Scalar(Scalar::Text(s)) if s == name => Some(v),
            _ => None,
        })
    }
    #[cfg(test)]
    pub fn scalar_text(&self, id: usize) -> Result<&str> {
        match self.nodes.get(id) {
            Some(Node::Scalar(Scalar::Text(s))) => Ok(s),
            _ => Err(Error::Invalid("model-card field requires a string")),
        }
    }
}
fn collection(tag: Option<Tag>, mapping: bool) -> Result<Collection> {
    let name = tag_name(tag)?;
    match (mapping, name.as_deref()) {
        (true, None | Some("map" | "!")) => Ok(Collection::Mapping),
        (false, None | Some("seq" | "!")) => Ok(Collection::Sequence),
        (true, Some("set")) => Ok(Collection::Set),
        (false, Some("omap")) => Ok(Collection::OrderedMap),
        (false, Some("pairs")) => Ok(Collection::Pairs),
        _ => Err(Error::Invalid(
            "YAML tag does not construct this collection",
        )),
    }
}
fn tag_name(tag: Option<Tag>) -> Result<Option<String>> {
    match tag {
        None => Ok(None),
        Some(Tag { handle, suffix }) if handle == "tag:yaml.org,2002:" => Ok(Some(suffix)),
        Some(Tag { handle, suffix }) if handle.is_empty() && suffix == "!" => Ok(Some(suffix)),
        _ => Err(Error::Invalid("unknown or unsafe YAML tag")),
    }
}
struct Patterns {
    float: NativeRegex,
    integer: NativeRegex,
    timestamp: NativeRegex,
}
impl Patterns {
    fn new() -> Result<Self> {
        let parse =
            |p: &[u8]| NativeRegex::parse(p).map_err(|_| Error::Invalid("built-in scalar pattern"));
        Ok(Self{
            float:parse(br"[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)")?,
            integer:parse(br"[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+")?,
            timestamp:parse(br"[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?")?,
        })
    }
    fn matches(pattern: &NativeRegex, value: &str) -> Result<bool> {
        pattern
            .full_match(value.as_bytes())
            .map(|m| m.is_some())
            .map_err(|_| Error::Limit("YAML scalar matching work"))
    }
    fn scalar(
        &self,
        value: String,
        style: TScalarStyle,
        tag: Option<Tag>,
        limits: &YamlLimits,
    ) -> Result<Scalar> {
        let tag = tag_name(tag)?.filter(|name| name != "!");
        let kind = if let Some(kind) = tag.as_deref() {
            kind
        } else if style != TScalarStyle::Plain {
            "str"
        } else if matches!(
            value.as_str(),
            "yes"
                | "Yes"
                | "YES"
                | "no"
                | "No"
                | "NO"
                | "true"
                | "True"
                | "TRUE"
                | "false"
                | "False"
                | "FALSE"
                | "on"
                | "On"
                | "ON"
                | "off"
                | "Off"
                | "OFF"
        ) {
            "bool"
        } else if Self::matches(&self.float, &value)? {
            "float"
        } else if Self::matches(&self.integer, &value)? {
            "int"
        } else if value == "<<" {
            "merge"
        } else if matches!(value.as_str(), "" | "~" | "null" | "Null" | "NULL") {
            "null"
        } else if Self::matches(&self.timestamp, &value)? {
            "timestamp"
        } else if value == "=" {
            "value"
        } else {
            "str"
        };
        Ok(match kind {
            "str" | "!" => Scalar::Text(value),
            "null" => Scalar::Null,
            "bool" => Scalar::Bool(match value.to_ascii_lowercase().as_str() {
                "yes" | "true" | "on" => true,
                "no" | "false" | "off" => false,
                _ => return Err(Error::Invalid("invalid YAML boolean")),
            }),
            "int" => Scalar::Integer(yaml_integer(&value, limits.max_integer_digits)?),
            "float" => Scalar::Float(yaml_float(&value)?),
            "binary" => Scalar::Bytes(base64(&value)?),
            "timestamp" => timestamp(&value)?,
            "merge" => Scalar::Merge,
            "value" => Scalar::Value,
            _ => return Err(Error::Invalid("unknown YAML scalar tag")),
        })
    }
}
fn yaml_integer(value: &str, max: usize) -> Result<String> {
    let value = value.replace('_', "");
    let (negative, body) = if let Some(v) = value.strip_prefix('-') {
        (true, v)
    } else {
        (false, value.strip_prefix('+').unwrap_or(&value))
    };
    if body.is_empty() {
        return Err(Error::Invalid("empty YAML integer"));
    }
    let (radix, digits) = if let Some(v) = body.strip_prefix("0b") {
        (2u32, v)
    } else if let Some(v) = body.strip_prefix("0x") {
        (16, v)
    } else if body.starts_with('0') && (body.len() > 1) {
        (8, body)
    } else {
        (10, body)
    };
    let mut result = vec![0u8];
    if digits.contains(':') && radix == 10 {
        for part in digits.split(':') {
            multiply_add(&mut result, 60, 0, max)?;
            let part = part
                .parse::<u32>()
                .map_err(|_| Error::Invalid("invalid YAML sexagesimal integer"))?;
            multiply_add(&mut result, 1, part, max)?;
        }
    } else {
        for ch in digits.chars() {
            let n = ch
                .to_digit(radix)
                .ok_or(Error::Invalid("invalid YAML integer digit"))?;
            multiply_add(&mut result, radix, n, max)?;
        }
    }
    if digits.is_empty() {
        return Err(Error::Invalid("empty YAML integer"));
    }
    let mut out = String::new();
    out.try_reserve(result.len() + 1)
        .map_err(|_| Error::Limit("YAML integer text"))?;
    if negative && !(result.len() == 1 && result[0] == 0) {
        out.push('-');
    }
    for digit in result.into_iter().rev() {
        out.push(char::from(b'0' + digit));
    }
    Ok(out)
}
fn multiply_add(value: &mut Vec<u8>, factor: u32, mut carry: u32, max: usize) -> Result<()> {
    for digit in value.iter_mut() {
        let n = u32::from(*digit) * factor + carry;
        *digit = (n % 10) as u8;
        carry = n / 10;
    }
    while carry > 0 {
        if value.len() >= max {
            return Err(Error::Limit("YAML integer digits"));
        }
        reserve(value, 1, "YAML integer allocation")?;
        value.push((carry % 10) as u8);
        carry /= 10;
    }
    Ok(())
}
fn yaml_float(value: &str) -> Result<f64> {
    let value = value.replace('_', "").to_ascii_lowercase();
    let (sign, body) = if let Some(v) = value.strip_prefix('-') {
        (-1.0, v)
    } else {
        (1.0, value.strip_prefix('+').unwrap_or(&value))
    };
    if body == ".inf" {
        return Ok(sign * f64::INFINITY);
    }
    if body == ".nan" {
        return Ok(f64::from_bits(0xfff8_0000_0000_0000));
    }
    if body.contains(':') {
        let mut sum = 0.0;
        let mut base = 1.0;
        for part in body.rsplit(':') {
            sum += part
                .parse::<f64>()
                .map_err(|_| Error::Invalid("invalid YAML sexagesimal float"))?
                * base;
            base *= 60.0;
        }
        Ok(sign * sum)
    } else {
        body.parse::<f64>()
            .map(|v| sign * v)
            .map_err(|_| Error::Invalid("invalid YAML float"))
    }
}
fn base64(value: &str) -> Result<Vec<u8>> {
    if !value.is_ascii() {
        return Err(Error::Invalid("YAML binary is not ASCII"));
    }
    let mut output = Vec::new();
    reserve(&mut output, value.len() / 4 * 3, "YAML binary")?;
    let mut bits = 0u32;
    let mut count = 0;
    let mut padding = 0;
    for byte in value.bytes() {
        if byte == b'=' {
            padding += 1;
            continue;
        }
        let n = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue,
        };
        if padding > 0 {
            break;
        }
        bits = (bits << 6) | u32::from(n);
        count += 1;
        if count == 4 {
            output.extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]);
            bits = 0;
            count = 0;
        }
    }
    match count {
        0 => {}
        2 if padding >= 2 => output.push((bits >> 4) as u8),
        3 if padding >= 1 => output.extend_from_slice(&[(bits >> 10) as u8, (bits >> 2) as u8]),
        _ => return Err(Error::Invalid("invalid YAML base64 padding")),
    }
    Ok(output)
}
fn timestamp(value: &str) -> Result<Scalar> {
    // PyYAML's constructor accepts one-digit month/day even though implicit
    // date-only resolution requires two; explicit !!timestamp uses this path.
    let pattern=NativeRegex::parse(br"([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?:(?:[Tt]|[ \t]+)([0-9]{1,2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]*))?(?:[ \t]*(Z|([-+])([0-9]{1,2})(?::([0-9]{2}))?))?)?")
        .map_err(|_|Error::Invalid("built-in timestamp pattern"))?;
    let m = pattern
        .full_match(value.as_bytes())
        .map_err(|_| Error::Limit("timestamp matching work"))?
        .ok_or(Error::Invalid("invalid YAML timestamp"))?;
    let cap = |index: usize| {
        m.captures
            .get(index)
            .and_then(|x| x.as_ref())
            .map(|r| &value[r.clone()])
    };
    let n = |index: usize| {
        cap(index)
            .unwrap_or("0")
            .parse::<u32>()
            .map_err(|_| Error::Invalid("timestamp component"))
    };
    let (year, month, day) = (n(0)?, n(1)?, n(2)?);
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    if year == 0 || day == 0 || day > days {
        return Err(Error::Invalid("invalid calendar date"));
    }
    let date = format!("{year:04}-{month:02}-{day:02}");
    if cap(3).is_none() {
        return Ok(Scalar::Date(date));
    }
    let (hour, minute, second) = (n(3)?, n(4)?, n(5)?);
    if hour > 23 || minute > 59 || second > 59 {
        return Err(Error::Invalid("invalid clock time"));
    }
    let mut out = format!("{date}T{hour:02}:{minute:02}:{second:02}");
    if let Some(frac) = cap(6) {
        let fraction = &frac[..frac.len().min(6)];
        let value = if fraction.is_empty() {
            0
        } else {
            fraction
                .parse::<u32>()
                .map_err(|_| Error::Invalid("timestamp fraction"))?
                * 10u32.pow((6 - fraction.len()) as u32)
        };
        if value != 0 {
            out.push_str(&format!(".{value:06}"));
        }
    }
    if cap(7).is_some() {
        let offset = n(9)?
            .checked_mul(60)
            .and_then(|x| x.checked_add(n(10).ok()?))
            .ok_or(Error::Invalid("timestamp offset"))?;
        if offset >= 24 * 60 {
            return Err(Error::Invalid("timestamp offset exceeds one day"));
        }
        let sign = if cap(8) == Some("-") && offset != 0 {
            '-'
        } else {
            '+'
        };
        out.push_str(&format!("{sign}{:02}:{:02}", offset / 60, offset % 60));
    }
    Ok(Scalar::DateTime(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_checkpoint::{Number, Value as Json};
    fn get<'a>(v: &'a Json, k: &str) -> &'a Json {
        v.as_object().unwrap().get(k).unwrap()
    }
    fn text(v: &Json) -> &str {
        v.as_text().unwrap().as_str().unwrap()
    }
    fn array(v: &Json) -> &[Json] {
        v.as_array().unwrap()
    }
    fn index(v: &Json) -> usize {
        match v {
            Json::Number(Number::Integer(n)) => n.as_u64().unwrap() as usize,
            _ => panic!("index"),
        }
    }
    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }
    fn compare(
        expected: &Json,
        id: usize,
        card: &Card,
        nodes: &[Json],
        seen: &mut [Option<usize>],
        name: &str,
    ) {
        if let Some(r) = expected.as_object().unwrap().get("ref") {
            let ri = index(r);
            if let Some(previous) = seen[ri] {
                assert_eq!(previous, id, "{name}: alias owner");
                return;
            }
            seen[ri] = Some(id);
            let e = &nodes[ri];
            match text(get(e, "kind")) {
                "map" => {
                    let Node::Mapping(Collection::Mapping, pairs) = &card.nodes[id] else {
                        panic!("{name}: expected map: {:?}", card.nodes[id])
                    };
                    let entries = array(get(e, "entries"));
                    assert_eq!(entries.len(), pairs.len(), "{name}: map count");
                    for (entry, &(k, v)) in entries.iter().zip(pairs) {
                        let p = array(entry);
                        compare(&p[0], k, card, nodes, seen, name);
                        compare(&p[1], v, card, nodes, seen, name);
                    }
                }
                "list" => {
                    let items = array(get(e, "items"));
                    match &card.nodes[id] {
                        Node::Sequence(Collection::Sequence, actual) => {
                            assert_eq!(items.len(), actual.len(), "{name}: sequence count");
                            for (item, &actual) in items.iter().zip(actual) {
                                compare(item, actual, card, nodes, seen, name);
                            }
                        }
                        Node::Mapping(Collection::OrderedMap | Collection::Pairs, pairs) => {
                            assert_eq!(items.len(), pairs.len(), "{name}: pairs count");
                            for (item, &(k, v)) in items.iter().zip(pairs) {
                                let expected_pair = &nodes[index(get(item, "ref"))];
                                assert_eq!(text(get(expected_pair, "kind")), "tuple");
                                let tuple = array(get(expected_pair, "items"));
                                assert_eq!(tuple.len(), 2);
                                compare(&tuple[0], k, card, nodes, seen, name);
                                compare(&tuple[1], v, card, nodes, seen, name);
                            }
                        }
                        _ => panic!("{name}: expected list: {:?}", card.nodes[id]),
                    }
                }
                "set" => {
                    let Node::Mapping(Collection::Set, actual) = &card.nodes[id] else {
                        panic!("{name}: set")
                    };
                    let items = array(get(e, "items"));
                    assert_eq!(items.len(), actual.len());
                    // Set order is not part of Python's or the converter's metadata contract.
                    for item in items {
                        let value = text(get(item, "value"));
                        let &(id,_)=actual.iter().find(|(k,_)|matches!(&card.nodes[*k],Node::Scalar(Scalar::Text(s))if s==value)).unwrap();
                        compare(item, id, card, nodes, seen, name);
                    }
                }
                k => panic!("{name}: unhandled reference container {k}"),
            }
            return;
        }
        let Node::Scalar(actual) = &card.nodes[id] else {
            panic!("{name}: expected scalar")
        };
        match (text(get(expected, "kind")), actual) {
            ("null", Scalar::Null) => {}
            ("bool", Scalar::Bool(actual)) => {
                assert_eq!(get(expected, "value"), &Json::Bool(*actual), "{name}")
            }
            ("text", Scalar::Text(actual))
            | ("int", Scalar::Integer(actual))
            | ("date", Scalar::Date(actual))
            | ("datetime", Scalar::DateTime(actual)) => {
                assert_eq!(text(get(expected, "value")), actual, "{name}")
            }
            ("float", Scalar::Float(actual)) => assert_eq!(
                hex(text(get(expected, "bits_le"))),
                actual.to_bits().to_le_bytes(),
                "{name}"
            ),
            ("bytes", Scalar::Bytes(actual)) => {
                assert_eq!(&hex(text(get(expected, "hex"))), actual, "{name}")
            }
            (kind, _) => panic!("{name}: expected {kind}, got {actual:?}"),
        }
    }
    #[test]
    fn frozen_model_card_frontmatter_graphs() {
        let reference = llama_checkpoint::json::parse(
            include_bytes!("../tests/fixtures/model-card/reference.json"),
            llama_checkpoint::Limits::default(),
        )
        .unwrap();
        let mut accepted = 0;
        let mut rejected = 0;
        for case in array(get(&reference, "cases")) {
            let name = text(get(case, "name"));
            let result = Card::parse_frontmatter(
                text(get(case, "readme")).as_bytes(),
                &YamlLimits::default(),
            );
            if get(case, "ok") == &Json::Bool(false) {
                assert!(result.is_err(), "{name}: source rejected");
                rejected += 1;
                continue;
            }
            let card = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let graph = get(case, "graph");
            let nodes = array(get(graph, "nodes"));
            let root = get(graph, "root");
            if let Some(id) = card.root {
                let mut seen = vec![None; nodes.len()];
                compare(root, id, &card, nodes, &mut seen, name);
            } else {
                let expected = &nodes[index(get(root, "ref"))];
                assert_eq!(text(get(expected, "kind")), "map", "{name}");
                assert!(
                    array(get(expected, "entries")).is_empty(),
                    "{name}: nonempty root lost"
                );
            }
            accepted += 1;
        }
        assert_eq!((accepted, rejected), (117, 16));
    }
    #[test]
    fn bounded_aliases_merges_depth_and_scalar_inputs() {
        let base = YamlLimits::default();
        assert!(matches!(
            Card::parse_frontmatter(
                b"---\nname: abc\n---\n",
                &YamlLimits {
                    max_input_bytes: 4,
                    ..base.clone()
                }
            ),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            Card::parse_frontmatter(
                b"---\na: [x, y]\n---\n",
                &YamlLimits {
                    max_nodes: 3,
                    ..base.clone()
                }
            ),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            Card::parse_frontmatter(
                b"---\na: &a x\nb: *a\nc: *a\n---\n",
                &YamlLimits {
                    max_aliases: 1,
                    ..base.clone()
                }
            ),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            Card::parse_frontmatter(
                b"---\na: [[[[x]]]]\n---\n",
                &YamlLimits {
                    max_depth: 3,
                    ..base.clone()
                }
            ),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            Card::parse_frontmatter(
                b"---\na: abcdef\n---\n",
                &YamlLimits {
                    max_scalar_bytes: 3,
                    ..base.clone()
                }
            ),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            Card::parse_frontmatter(
                b"---\na: &a {x: 1, y: 2}\nb: {<<: *a}\n---\n",
                &YamlLimits {
                    max_merge_edges: 4,
                    ..base
                }
            ),
            Err(Error::Limit(_))
        ));
    }
    #[test]
    fn ignored_cycles_remain_graph_edges() {
        let card = Card::parse_frontmatter(
            b"---\nunused: &x [*x]\nname: Fixture\n---\n",
            &YamlLimits::default(),
        )
        .unwrap();
        let id = card.get("unused").unwrap();
        let Node::Sequence(_, items) = &card.nodes[id] else {
            panic!("sequence")
        };
        assert_eq!(items, &[id]);
        assert_eq!(
            card.scalar_text(card.get("name").unwrap()).unwrap(),
            "Fixture"
        );
    }
}
