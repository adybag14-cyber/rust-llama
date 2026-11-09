//! Bounded JSON parsing and deterministic serialization without a foreign
//! runtime or native dependency. Duplicate keys and non-finite numbers are
//! rejected. Integer lexemes preserve signed/unsigned 64-bit identity; fractions
//! and exponents use finite IEEE-754 binary64. Arbitrary precision arithmetic
//! remains part of the browser math work package.

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Integer(i64),
    Unsigned(u64),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Self> {
        self.as_object()?.get(key)
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        if let Self::Array(value) = self {
            Some(value)
        } else {
            None
        }
    }
    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        if let Self::Object(value) = self {
            Some(value)
        } else {
            None
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::Integer(value) => Some(*value as f64),
            Self::Unsigned(value) => Some(*value as f64),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => return Some(*value),
            Self::Unsigned(value) => return i64::try_from(*value).ok(),
            _ => {}
        }
        let n = self.as_f64()?;
        if n.is_finite()
            && n.fract() == 0.0
            && n >= i64::MIN as f64
            && n < 9_223_372_036_854_775_808.0
        {
            Some(n as i64)
        } else {
            None
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Integer(value) => return u64::try_from(*value).ok(),
            Self::Unsigned(value) => return Some(*value),
            _ => {}
        }
        let n = self.as_f64()?;
        if n.is_finite() && n.fract() == 0.0 && (0.0..18_446_744_073_709_551_616.0).contains(&n) {
            Some(n as u64)
        } else {
            None
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn object(entries: impl IntoIterator<Item = (impl Into<String>, Value)>) -> Self {
        Self::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}
impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Self::Integer(value as i64)
    }
}
impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self::Integer(value as i64)
    }
}
impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}
impl From<u64> for Value {
    fn from(value: u64) -> Self {
        if let Ok(value) = i64::try_from(value) {
            Self::Integer(value)
        } else {
            Self::Unsigned(value)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_bytes: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_depth: 128,
            max_nodes: 1_000_000,
        }
    }
}

pub fn parse(input: &str) -> Result<Value> {
    parse_with_limits(input, Limits::default())
}

pub fn parse_with_limits(input: &str, limits: Limits) -> Result<Value> {
    if input.len() > limits.max_bytes {
        return Err(Error::new("JSON input exceeds byte limit"));
    }
    if limits.max_depth > 256 {
        return Err(Error::new("JSON recursion limit cannot exceed 256"));
    }
    let mut parser = Parser {
        input,
        pos: 0,
        nodes: 0,
        limits,
    };
    let value = parser.value(0)?;
    parser.space();
    if parser.pos != input.len() {
        return Err(parser.error("trailing data"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
    nodes: usize,
    limits: Limits,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> Error {
        Error::new(format!("JSON byte {}: {message}", self.pos))
    }
    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }
    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }
    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > self.limits.max_depth {
            return Err(self.error("nesting exceeds depth limit"));
        }
        self.nodes += 1;
        if self.nodes > self.limits.max_nodes {
            return Err(self.error("node limit exceeded"));
        }
        self.space();
        match self.peek() {
            Some(b'n') => {
                self.keyword("null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.keyword("true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.keyword("false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => {
                self.pos += 1;
                self.space();
                let mut values = Vec::new();
                if self.consume(b']') {
                    return Ok(Value::Array(values));
                }
                loop {
                    values.push(self.value(depth + 1)?);
                    self.space();
                    if self.consume(b']') {
                        break;
                    }
                    if !self.consume(b',') {
                        return Err(self.error("expected comma or closing bracket"));
                    }
                }
                Ok(Value::Array(values))
            }
            Some(b'{') => {
                self.pos += 1;
                self.space();
                let mut values = BTreeMap::new();
                if self.consume(b'}') {
                    return Ok(Value::Object(values));
                }
                loop {
                    self.space();
                    if self.peek() != Some(b'"') {
                        return Err(self.error("object key must be a string"));
                    }
                    let key = self.string()?;
                    self.space();
                    if !self.consume(b':') {
                        return Err(self.error("expected colon"));
                    }
                    let value = self.value(depth + 1)?;
                    if values.insert(key, value).is_some() {
                        return Err(self.error("duplicate object key"));
                    }
                    self.space();
                    if self.consume(b'}') {
                        break;
                    }
                    if !self.consume(b',') {
                        return Err(self.error("expected comma or closing brace"));
                    }
                }
                Ok(Value::Object(values))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("expected a JSON value")),
        }
    }
    fn keyword(&mut self, word: &str) -> Result<()> {
        if self.input[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.error("invalid literal"))
        }
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        self.consume(b'-');
        if self.consume(b'0') {
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("leading zero in number"));
            }
        } else {
            if !matches!(self.peek(), Some(b'1'..=b'9')) {
                return Err(self.error("expected integer digits"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if self.consume(b'.') {
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected fraction digits"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected exponent digits"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let lexeme = &self.input[start..self.pos];
        if !lexeme.contains(['.', 'e', 'E']) && lexeme != "-0" {
            if let Ok(value) = lexeme.parse::<i64>() {
                return Ok(Value::Integer(value));
            }
            if let Ok(value) = lexeme.parse::<u64>() {
                return Ok(Value::Unsigned(value));
            }
            return Err(self.error("integer outside signed/unsigned 64-bit range"));
        }
        let value = lexeme
            .parse::<f64>()
            .map_err(|_| self.error("invalid number"))?;
        if !value.is_finite() {
            return Err(self.error("number outside finite binary64 range"));
        }
        Ok(Value::Number(value))
    }
    fn hex4(&mut self) -> Result<u16> {
        let mut value = 0u16;
        for _ in 0..4 {
            let nibble = match self.peek() {
                Some(b @ b'0'..=b'9') => b - b'0',
                Some(b @ b'a'..=b'f') => b - b'a' + 10,
                Some(b @ b'A'..=b'F') => b - b'A' + 10,
                _ => return Err(self.error("expected four hexadecimal digits")),
            };
            self.pos += 1;
            value = (value << 4) | nibble as u16;
        }
        Ok(value)
    }
    fn string(&mut self) -> Result<String> {
        if !self.consume(b'"') {
            return Err(self.error("expected string"));
        }
        let mut result = String::new();
        loop {
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(result);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.pos += 1;
                    match escape {
                        b'"' => result.push('"'),
                        b'\\' => result.push('\\'),
                        b'/' => result.push('/'),
                        b'b' => result.push('\u{8}'),
                        b'f' => result.push('\u{c}'),
                        b'n' => result.push('\n'),
                        b'r' => result.push('\r'),
                        b't' => result.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let scalar = if (0xd800..=0xdbff).contains(&first) {
                                if !self.consume(b'\\') || !self.consume(b'u') {
                                    return Err(self.error("high surrogate requires low surrogate"));
                                }
                                let second = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&second) {
                                    return Err(self.error("invalid low surrogate"));
                                }
                                0x10000
                                    + (((first as u32 - 0xd800) << 10) | (second as u32 - 0xdc00))
                            } else if (0xdc00..=0xdfff).contains(&first) {
                                return Err(self.error("unpaired low surrogate"));
                            } else {
                                first as u32
                            };
                            result.push(
                                char::from_u32(scalar)
                                    .ok_or_else(|| self.error("invalid Unicode scalar"))?,
                            );
                        }
                        _ => return Err(self.error("unknown escape")),
                    }
                }
                Some(0..=31) => return Err(self.error("unescaped control character")),
                _ => {
                    let c = self.input[self.pos..]
                        .chars()
                        .next()
                        .expect("peek guaranteed a character");
                    self.pos += c.len_utf8();
                    result.push(c);
                }
            }
        }
    }
}

pub fn stringify(value: &Value) -> Result<String> {
    stringify_with_limits(value, Limits::default())
}

pub fn stringify_with_limits(value: &Value, limits: Limits) -> Result<String> {
    if limits.max_depth > 256 {
        return Err(Error::new("JSON recursion limit cannot exceed 256"));
    }
    let mut output = String::new();
    let mut nodes = 0;
    write_value(value, &mut output, 0, &mut nodes, limits)?;
    Ok(output)
}

fn write_value(
    value: &Value,
    out: &mut String,
    depth: usize,
    nodes: &mut usize,
    limits: Limits,
) -> Result<()> {
    if depth > limits.max_depth {
        return Err(Error::new("JSON nesting exceeds depth limit"));
    }
    *nodes += 1;
    if *nodes > limits.max_nodes {
        return Err(Error::new("JSON node limit exceeded"));
    }
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Integer(value) => {
            write!(out, "{value}").expect("String write");
        }
        Value::Unsigned(value) => {
            write!(out, "{value}").expect("String write");
        }
        Value::Number(value) => {
            if !value.is_finite() {
                return Err(Error::new("cannot serialize non-finite JSON number"));
            }
            // Preserve the floating-number variant even when its value happens
            // to be integral (e.g. -125.0), including the sign of negative zero.
            write!(out, "{value:?}").expect("writing to String cannot fail");
        }
        Value::String(value) => write_string(value, out, limits.max_bytes)?,
        Value::Array(values) => {
            out.push('[');
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(value, out, depth + 1, nodes, limits)?;
            }
            out.push(']');
        }
        Value::Object(values) => {
            out.push('{');
            for (i, (key, value)) in values.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out, limits.max_bytes)?;
                out.push(':');
                write_value(value, out, depth + 1, nodes, limits)?;
            }
            out.push('}');
        }
    }
    if out.len() > limits.max_bytes {
        return Err(Error::new("JSON output exceeds byte limit"));
    }
    Ok(())
}

fn write_string(value: &str, out: &mut String, max_bytes: usize) -> Result<()> {
    if value.len() > max_bytes.saturating_sub(out.len()) {
        return Err(Error::new("JSON output exceeds byte limit"));
    }
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\u{0}'..='\u{1f}' => {
                write!(out, "\\u{:04x}", c as u32).expect("String write");
            }
            _ => out.push(c),
        }
        if out.len() > max_bytes {
            return Err(Error::new("JSON output exceeds byte limit"));
        }
    }
    out.push('"');
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_surrogates_and_escape_roundtrip() {
        let value =
            parse(r#"{"text":"A\u00a3\ud83e\udd80界\u0000\n","array":[true,false,null,-1.25e2]}"#)
                .unwrap();
        assert_eq!(value.get("text").unwrap().as_str(), Some("A£🦀界\0\n"));
        assert_eq!(parse(&stringify(&value).unwrap()).unwrap(), value);
    }
    #[test]
    fn rejects_malformed_ambiguous_or_non_finite_input() {
        for input in [
            "",
            "01",
            "-",
            "+1",
            "1.",
            "1e",
            "1e400",
            "true false",
            "[1,]",
            "{\"a\":1,}",
            "{\"a\":1,\"a\":2}",
            "\"\\uD800\"",
            "\"\\uDC00\"",
            "\"\\uD800\\u0041\"",
            "\"\\x41\"",
            "\"\n\"",
        ] {
            assert!(parse(input).is_err(), "accepted malformed {input:?}");
        }
        assert!(stringify(&Value::Number(f64::NAN)).is_err());
    }
    #[test]
    fn enforces_limits_and_handles_negative_zero() {
        let limits = Limits {
            max_bytes: 20,
            max_depth: 2,
            max_nodes: 5,
        };
        assert!(parse_with_limits("[[[0]]]", limits).is_err());
        assert!(parse_with_limits("[1,2,3,4,5]", limits).is_err());
        assert!(parse_with_limits("\"abcdefghijklmnopqrstuv\"", limits).is_err());
        assert!(parse("-0").unwrap().as_f64().unwrap().is_sign_negative());
        assert!(
            Value::Number(9_223_372_036_854_775_808.0)
                .as_i64()
                .is_none()
        );
    }
    #[test]
    fn deterministic_key_order() {
        assert_eq!(
            stringify(&parse("{\"z\":3,\"a\":1}").unwrap()).unwrap(),
            "{\"a\":1,\"z\":3}"
        );
    }
    #[test]
    fn integers_preserve_identity_beyond_binary64_precision() {
        for text in [
            "9007199254740993",
            "-9007199254740993",
            "9223372036854775807",
            "-9223372036854775808",
            "18446744073709551615",
        ] {
            let value = parse(text).unwrap();
            assert_eq!(stringify(&value).unwrap(), text);
        }
        assert_eq!(
            parse("9007199254740993").unwrap().as_u64(),
            Some(9007199254740993)
        );
        assert!(parse("18446744073709551616").is_err());
        assert!(parse("-9223372036854775809").is_err());
    }
}
