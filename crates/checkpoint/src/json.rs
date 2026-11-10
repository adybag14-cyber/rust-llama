//! The frozen converter uses Python json.loads rather than a canonical-format
//! validator. This bounded parser preserves its duplicate-key, number and
//! escaped-surrogate behavior. It is not a general Python interpreter.
use crate::{Budget, Error, Limits, Result, reserve};
use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Text(pub(crate) Vec<u8>);
impl Text {
    pub fn try_from_str(value: &str) -> Result<Self> {
        let mut bytes = Vec::new();
        reserve(&mut bytes, value.len(), "text allocation")?;
        bytes.extend_from_slice(value.as_bytes());
        Ok(Self(bytes))
    }
    /// Canonical extended UTF-8 preserves unpaired escaped surrogate codepoints.
    /// Use as_str when a consumer requires Unicode scalar values.
    pub fn as_wtf8(&self) -> &[u8] {
        &self.0
    }
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }
    pub fn codepoints(&self) -> impl Iterator<Item = u32> + '_ {
        let mut position = 0;
        std::iter::from_fn(move || {
            let (value, length) = decode(&self.0[position..])?;
            position += length;
            Some(value)
        })
    }
    pub(crate) fn copy(&self, budget: &mut Budget) -> Result<Self> {
        budget.text(self.0.len())?;
        let mut bytes = Vec::new();
        reserve(&mut bytes, self.0.len(), "text copy allocation")?;
        bytes.extend_from_slice(&self.0);
        Ok(Self(bytes))
    }
    pub(crate) fn from_bytes(bytes: &[u8], budget: &mut Budget) -> Result<Self> {
        if bytes.len() > budget.limits.max_string_bytes {
            return Err(Error::Limit("string bytes"));
        }
        budget.text(bytes.len())?;
        let mut output = Vec::new();
        reserve(&mut output, bytes.len(), "text allocation")?;
        output.extend_from_slice(bytes);
        Ok(Self(output))
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn push_codepoint(output: &mut Vec<u8>, value: u32) -> Result<()> {
        let (bytes, length) = encode(value);
        reserve(output, length, "text codepoint allocation")?;
        output.extend_from_slice(&bytes[..length]);
        Ok(())
    }
    pub(crate) fn from_codepoints(
        values: impl IntoIterator<Item = u32>,
        budget: &mut Budget,
    ) -> Result<Self> {
        let mut output = Vec::new();
        for value in values {
            if value > 0x10ffff {
                return Err(Error::Invalid("text codepoint outside Unicode range"));
            }
            let (bytes, length) = encode(value);
            if output
                .len()
                .checked_add(length)
                .is_none_or(|n| n > budget.limits.max_string_bytes)
            {
                return Err(Error::Limit("string bytes"));
            }
            budget.text(length)?;
            reserve(&mut output, length, "text codepoint allocation")?;
            output.extend_from_slice(&bytes[..length]);
        }
        Ok(Self(output))
    }
}
impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"")?;
        for code in self.codepoints() {
            match code {
                0x22 => f.write_str("\\\"")?,
                0x5c => f.write_str("\\\\")?,
                0x00..=0x1f | 0xd800..=0xdfff => write!(f, "\\u{code:04x}")?,
                value => {
                    if let Some(ch) = char::from_u32(value) {
                        write!(f, "{ch}")?;
                    }
                }
            }
        }
        f.write_str("\"")
    }
}
fn encode(value: u32) -> ([u8; 4], usize) {
    if value < 0x80 {
        ([value as u8, 0, 0, 0], 1)
    } else if value < 0x800 {
        (
            [
                (0xc0 | (value >> 6)) as u8,
                (0x80 | (value & 63)) as u8,
                0,
                0,
            ],
            2,
        )
    } else if value < 0x10000 {
        (
            [
                (0xe0 | (value >> 12)) as u8,
                (0x80 | ((value >> 6) & 63)) as u8,
                (0x80 | (value & 63)) as u8,
                0,
            ],
            3,
        )
    } else {
        (
            [
                (0xf0 | (value >> 18)) as u8,
                (0x80 | ((value >> 12) & 63)) as u8,
                (0x80 | ((value >> 6) & 63)) as u8,
                (0x80 | (value & 63)) as u8,
            ],
            4,
        )
    }
}
fn decode(bytes: &[u8]) -> Option<(u32, usize)> {
    let first = *bytes.first()?;
    let length = if first < 0x80 {
        1
    } else if first < 0xe0 {
        2
    } else if first < 0xf0 {
        3
    } else {
        4
    };
    if bytes.len() < length {
        return None;
    }
    let mask = [0, 0x7f, 0x1f, 0x0f, 0x07][length];
    let mut value = u32::from(first & mask);
    for &byte in &bytes[1..length] {
        value = (value << 6) | u32::from(byte & 63);
    }
    Some((value, length))
}

/// Arbitrary-size JSON integer, bounded by input/decoded-text limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Integer {
    negative: bool,
    digits: Vec<u8>,
}
impl Integer {
    pub fn is_negative(&self) -> bool {
        self.negative
    }
    pub fn digits(&self) -> &str {
        std::str::from_utf8(&self.digits).expect("Integer stores only ASCII decimal digits")
    }
    pub fn as_u64(&self) -> Option<u64> {
        if self.negative {
            return None;
        }
        self.digits.iter().try_fold(0u64, |n, &d| {
            n.checked_mul(10)?.checked_add(u64::from(d - b'0'))
        })
    }
    pub fn as_i64(&self) -> Option<i64> {
        let n = self.digits.iter().try_fold(0u64, |n, &d| {
            n.checked_mul(10)?.checked_add(u64::from(d - b'0'))
        })?;
        if self.negative {
            if n == (i64::MAX as u64) + 1 {
                Some(i64::MIN)
            } else {
                i64::try_from(n).ok()?.checked_neg()
            }
        } else {
            i64::try_from(n).ok()
        }
    }
    fn parse(bytes: &[u8], budget: &mut Budget) -> Result<Self> {
        let negative = bytes.first() == Some(&b'-');
        let digits = if negative { &bytes[1..] } else { bytes };
        if digits.len() > budget.limits.max_integer_digits {
            return Err(Error::Limit("JSON integer digits"));
        }
        budget.text(digits.len())?;
        let mut output = Vec::new();
        reserve(&mut output, digits.len(), "integer allocation")?;
        output.extend_from_slice(digits);
        Ok(Self {
            negative: negative && digits != b"0",
            digits: output,
        })
    }
    pub(crate) fn from_u64(mut value: u64, budget: &mut Budget) -> Result<Self> {
        let mut scratch = [0u8; 20];
        let mut start = scratch.len();
        loop {
            start -= 1;
            scratch[start] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        budget.text(scratch.len() - start)?;
        let mut digits = Vec::new();
        reserve(&mut digits, scratch.len() - start, "integer allocation")?;
        digits.extend_from_slice(&scratch[start..]);
        Ok(Self {
            negative: false,
            digits,
        })
    }
    fn magnitude_cmp(&self, other: &Self) -> Ordering {
        self.digits
            .len()
            .cmp(&other.digits.len())
            .then(self.digits.cmp(&other.digits))
    }
    fn combine(&self, other: &Self, subtract: bool, budget: &mut Budget) -> Result<Self> {
        let other_negative = other.negative ^ (subtract && other.digits != b"0");
        let same_sign = self.negative == other_negative;
        let capacity = self
            .digits
            .len()
            .max(other.digits.len())
            .checked_add(1)
            .ok_or(Error::Limit("integer arithmetic size"))?;
        budget.text(capacity)?;
        let mut out = Vec::new();
        reserve(&mut out, capacity, "integer arithmetic allocation")?;
        let negative;
        if same_sign {
            negative = self.negative;
            let mut carry = 0u8;
            let mut a = self.digits.iter().rev();
            let mut b = other.digits.iter().rev();
            loop {
                let x = a.next();
                let y = b.next();
                if x.is_none() && y.is_none() && carry == 0 {
                    break;
                }
                let digit = x.map_or(0, |x| x - b'0') + y.map_or(0, |x| x - b'0') + carry;
                out.push(b'0' + digit % 10);
                carry = digit / 10;
            }
        } else {
            let (large, small, sign) = if self.magnitude_cmp(other) == Ordering::Less {
                (other, self, other_negative)
            } else {
                (self, other, self.negative)
            };
            negative = sign;
            let mut borrow = 0i16;
            let mut smaller = small.digits.iter().rev();
            for &digit in large.digits.iter().rev() {
                let mut value = i16::from(digit - b'0')
                    - i16::from(smaller.next().map_or(0, |x| x - b'0'))
                    - borrow;
                borrow = i16::from(value < 0);
                if value < 0 {
                    value += 10;
                }
                out.push(b'0' + value as u8);
            }
        }
        if out.is_empty() {
            out.push(b'0');
        }
        while out.len() > 1 && out.last() == Some(&b'0') {
            out.pop();
        }
        out.reverse();
        Ok(Self {
            negative: negative && out != b"0",
            digits: out,
        })
    }
    fn as_float(&self) -> Result<f64> {
        let value = self
            .digits()
            .parse::<f64>()
            .map_err(|_| Error::Invalid("integer to float conversion"))?;
        if !value.is_finite() {
            return Err(Error::Invalid(
                "integer is too large for source float arithmetic",
            ));
        }
        Ok(if self.negative { -value } else { value })
    }
}
#[derive(Debug, Clone)]
pub enum Number {
    Integer(Integer),
    Float(f64),
}
impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(a), Self::Integer(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}
impl Number {
    pub fn as_integer(&self) -> Option<&Integer> {
        if let Self::Integer(n) = self {
            Some(n)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::Integer(n) => n.as_float(),
            Self::Float(n) => Ok(*n),
        }
    }
    pub(crate) fn add_u64(&self, n: u64, budget: &mut Budget) -> Result<Self> {
        match self {
            Self::Integer(value) => Ok(Self::Integer(value.combine(
                &Integer::from_u64(n, budget)?,
                false,
                budget,
            )?)),
            Self::Float(value) => Ok(Self::Float(n as f64 + value)),
        }
    }
    pub(crate) fn subtract(&self, other: &Self, budget: &mut Budget) -> Result<Self> {
        match (self, other) {
            (Self::Integer(a), Self::Integer(b)) => Ok(Self::Integer(a.combine(b, true, budget)?)),
            _ => Ok(Self::Float(self.as_f64()? - other.as_f64()?)),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Object(pub(crate) Vec<Member>);
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Member {
    pub key: Text,
    pub value: Value,
    order: usize,
}
impl Object {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn entries(&self) -> impl Iterator<Item = (&Text, &Value)> {
        self.0.iter().map(|m| (&m.key, &m.value))
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0
            .iter()
            .find(|m| m.key.0 == key.as_bytes())
            .map(|m| &m.value)
    }
    pub(crate) fn remove(&mut self, key: &str) -> Option<Value> {
        let index = self.0.iter().position(|m| m.key.0 == key.as_bytes())?;
        Some(self.0.remove(index).value)
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    Text(Text),
    Array(Vec<Value>),
    Object(Object),
}
impl Value {
    pub fn as_text(&self) -> Option<&Text> {
        if let Self::Text(x) = self {
            Some(x)
        } else {
            None
        }
    }
    pub fn as_object(&self) -> Option<&Object> {
        if let Self::Object(x) = self {
            Some(x)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        if let Self::Array(x) = self {
            Some(x)
        } else {
            None
        }
    }
    pub(crate) fn numeric(self, budget: &mut Budget) -> Result<Number> {
        match self {
            Self::Number(n) => Ok(n),
            Self::Bool(v) => Ok(Number::Integer(Integer::from_u64(u64::from(v), budget)?)),
            _ => Err(Error::Invalid(
                "tensor offsets do not support source numeric arithmetic",
            )),
        }
    }
}
pub fn parse(bytes: &[u8], limits: Limits) -> Result<Value> {
    parse_with_usage(bytes, limits).map(|(value, _)| value)
}
/// Return actual parser work, including discarded duplicate values. Callers
/// parsing several sidecars can supply the remaining budgets before each parse.
pub fn parse_with_usage(bytes: &[u8], limits: Limits) -> Result<(Value, crate::Usage)> {
    let mut budget = Budget::new(limits)?;
    budget.header(bytes.len() as u64)?;
    let value = parse_charged(bytes, &mut budget)?;
    Ok((value, budget.used))
}
pub(crate) fn parse_charged(bytes: &[u8], budget: &mut Budget) -> Result<Value> {
    std::str::from_utf8(bytes).map_err(|_| Error::Invalid("JSON header is not UTF-8"))?;
    let mut parser = Parser {
        bytes,
        at: 0,
        budget,
    };
    let result = parser.value(0)?;
    parser.space();
    if parser.at != parser.bytes.len() {
        return Err(Error::Invalid("trailing JSON data"));
    }
    Ok(result)
}
struct Parser<'a, 'b> {
    bytes: &'a [u8],
    at: usize,
    budget: &'b mut Budget,
}
impl Parser<'_, '_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }
    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }
    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn literal(&mut self, text: &[u8]) -> Result<()> {
        if self.bytes.get(self.at..self.at.saturating_add(text.len())) != Some(text) {
            return Err(Error::Invalid("invalid JSON literal"));
        }
        self.at += text.len();
        Ok(())
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > self.budget.limits.max_json_depth {
            return Err(Error::Limit("JSON depth"));
        }
        self.budget.node()?;
        self.space();
        match self.peek() {
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'N') => {
                self.literal(b"NaN")?;
                Ok(Value::Number(Number::Float(f64::NAN)))
            }
            Some(b'I') => {
                self.literal(b"Infinity")?;
                Ok(Value::Number(Number::Float(f64::INFINITY)))
            }
            Some(b'"') => Ok(Value::Text(self.string()?)),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(Error::Invalid("expected JSON value")),
        }
    }
    fn array(&mut self, depth: usize) -> Result<Value> {
        self.at += 1;
        let mut values = Vec::new();
        self.space();
        if self.take(b']') {
            return Ok(Value::Array(values));
        }
        loop {
            let value = self.value(depth + 1)?;
            reserve(&mut values, 1, "JSON array allocation")?;
            values.push(value);
            self.space();
            if self.take(b']') {
                break;
            }
            if !self.take(b',') {
                return Err(Error::Invalid("JSON array separator"));
            }
        }
        Ok(Value::Array(values))
    }
    fn object(&mut self, depth: usize) -> Result<Value> {
        self.at += 1;
        let mut members = Vec::new();
        self.space();
        if self.take(b'}') {
            return Ok(Value::Object(Object(members)));
        }
        loop {
            self.space();
            self.budget.node()?;
            let key = self.string()?;
            self.space();
            if !self.take(b':') {
                return Err(Error::Invalid("JSON object colon"));
            }
            let value = self.value(depth + 1)?;
            let order = members.len();
            reserve(&mut members, 1, "JSON object allocation")?;
            members.push(Member { key, value, order });
            self.space();
            if self.take(b'}') {
                break;
            }
            if !self.take(b',') {
                return Err(Error::Invalid("JSON object separator"));
            }
        }
        // Python updates the last value without moving the first key position.
        // Sorting in place avoids quadratic hostile-key insertion or an
        // additional infallibly allocated hash table.
        members.sort_unstable_by(|a, b| a.key.cmp(&b.key).then(a.order.cmp(&b.order)));
        members.dedup_by(|later, earlier| {
            if later.key == earlier.key {
                earlier.value = std::mem::replace(&mut later.value, Value::Null);
                true
            } else {
                false
            }
        });
        members.sort_unstable_by_key(|member| member.order);
        Ok(Value::Object(Object(members)))
    }
    fn hex4(&mut self) -> Result<u32> {
        let bytes = self
            .bytes
            .get(self.at..self.at.saturating_add(4))
            .ok_or(Error::Invalid("truncated Unicode escape"))?;
        let mut value = 0;
        for &byte in bytes {
            value = (value << 4)
                | u32::from(match byte {
                    b'0'..=b'9' => byte - b'0',
                    b'a'..=b'f' => byte - b'a' + 10,
                    b'A'..=b'F' => byte - b'A' + 10,
                    _ => return Err(Error::Invalid("invalid Unicode escape")),
                });
        }
        self.at += 4;
        Ok(value)
    }
    fn append_text(&mut self, output: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
        if output
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.budget.limits.max_string_bytes)
        {
            return Err(Error::Limit("string bytes"));
        }
        self.budget.text(bytes.len())?;
        reserve(output, bytes.len(), "JSON string allocation")?;
        output.extend_from_slice(bytes);
        Ok(())
    }
    fn string(&mut self) -> Result<Text> {
        if !self.take(b'"') {
            return Err(Error::Invalid("JSON object key/string expected"));
        }
        let mut output = Vec::new();
        loop {
            let start = self.at;
            while self
                .peek()
                .is_some_and(|b| b != b'"' && b != b'\\' && b >= 0x20)
            {
                self.at += 1;
            }
            let stop = self.at;
            self.append_text(&mut output, &self.bytes[start..stop])?;
            match self.peek() {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(Text(output));
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escape = self.peek().ok_or(Error::Invalid("truncated JSON escape"))?;
                    self.at += 1;
                    let value = match escape {
                        b'"' => b'"' as u32,
                        b'\\' => b'\\' as u32,
                        b'/' => b'/' as u32,
                        b'b' => 8,
                        b'f' => 12,
                        b'n' => 10,
                        b'r' => 13,
                        b't' => 9,
                        b'u' => {
                            let first = self.hex4()?;
                            if (0xd800..=0xdbff).contains(&first)
                                && self.bytes.get(self.at..self.at.saturating_add(2))
                                    == Some(b"\\u")
                            {
                                let saved = self.at;
                                self.at += 2;
                                let second = self.hex4()?;
                                if (0xdc00..=0xdfff).contains(&second) {
                                    0x10000 + ((first - 0xd800) << 10) + second - 0xdc00
                                } else {
                                    self.at = saved;
                                    first
                                }
                            } else {
                                first
                            }
                        }
                        _ => return Err(Error::Invalid("unknown JSON escape")),
                    };
                    let (bytes, length) = encode(value);
                    self.append_text(&mut output, &bytes[..length])?;
                }
                _ => return Err(Error::Invalid("control byte or unterminated JSON string")),
            }
        }
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.at;
        self.take(b'-');
        if self.peek() == Some(b'I') {
            if self.at == start {
                return Err(Error::Invalid("invalid Infinity token"));
            }
            self.literal(b"Infinity")?;
            return Ok(Value::Number(Number::Float(f64::NEG_INFINITY)));
        }
        if !self.take(b'0') {
            if !self.peek().is_some_and(|v| matches!(v, b'1'..=b'9')) {
                return Err(Error::Invalid("invalid JSON number"));
            }
            while self.peek().is_some_and(|v| v.is_ascii_digit()) {
                self.at += 1;
            }
        }
        let mut floating = false;
        if self.take(b'.') {
            floating = true;
            let digits = self.at;
            while self.peek().is_some_and(|v| v.is_ascii_digit()) {
                self.at += 1;
            }
            if digits == self.at {
                return Err(Error::Invalid("fraction has no digits"));
            }
        }
        if self.peek().is_some_and(|v| v == b'e' || v == b'E') {
            floating = true;
            self.at += 1;
            if self.peek().is_some_and(|v| v == b'+' || v == b'-') {
                self.at += 1;
            }
            let digits = self.at;
            while self.peek().is_some_and(|v| v.is_ascii_digit()) {
                self.at += 1;
            }
            if digits == self.at {
                return Err(Error::Invalid("exponent has no digits"));
            }
        }
        let token = &self.bytes[start..self.at];
        let number = if floating {
            let text = std::str::from_utf8(token).map_err(|_| Error::Invalid("number encoding"))?;
            Number::Float(
                text.parse()
                    .map_err(|_| Error::Invalid("JSON float conversion"))?,
            )
        } else {
            Number::Integer(Integer::parse(token, self.budget)?)
        };
        Ok(Value::Number(number))
    }
}
