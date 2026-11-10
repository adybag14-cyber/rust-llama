//! Ordered metadata and cumulative auxiliary JSON parsing.
use crate::{Error, Result, reserve, string};
use llama_checkpoint::{Limits as JsonLimits, Number, Object, Usage, Value as Json};
use llama_gguf::RawValue as Value;

pub(crate) struct JsonBudget {
    limits: JsonLimits,
    used: Usage,
}
impl JsonBudget {
    pub(crate) fn new(limits: JsonLimits) -> Self {
        Self {
            limits,
            used: Usage::default(),
        }
    }
    pub(crate) fn parse(&mut self, bytes: &[u8]) -> Result<Json> {
        let mut remaining = self.limits.clone();
        remaining.max_total_header_bytes = remaining
            .max_total_header_bytes
            .checked_sub(self.used.header_bytes)
            .ok_or(Error::Limit("auxiliary JSON bytes"))?;
        remaining.max_json_nodes = remaining
            .max_json_nodes
            .checked_sub(self.used.json_nodes)
            .ok_or(Error::Limit("auxiliary JSON nodes"))?;
        remaining.max_text_bytes = remaining
            .max_text_bytes
            .checked_sub(self.used.text_bytes)
            .ok_or(Error::Limit("auxiliary JSON text"))?;
        let (value, actual) = llama_checkpoint::json::parse_with_usage(bytes, remaining)?;
        self.used.header_bytes += actual.header_bytes;
        self.used.json_nodes += actual.json_nodes;
        self.used.text_bytes += actual.text_bytes;
        Ok(value)
    }
}
pub(crate) fn object(value: &Json) -> Result<&Object> {
    value
        .as_object()
        .ok_or(Error::Invalid("expected a JSON object"))
}
pub(crate) fn text(value: &Json) -> Result<&str> {
    value
        .as_text()
        .and_then(|t| t.as_str())
        .ok_or(Error::Invalid("expected a Unicode scalar string"))
}
pub(crate) fn optional_text(value: Option<&Json>) -> Result<Option<&str>> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(value) => text(value).map(Some),
    }
}
pub(crate) fn integer(value: &Json) -> Result<i64> {
    match value {
        Json::Number(Number::Integer(n)) => n
            .as_i64()
            .ok_or(Error::Invalid("integer exceeds signed 64-bit range")),
        Json::Bool(value) => Ok(i64::from(*value)),
        _ => Err(Error::Invalid("integer metadata requires an integer")),
    }
}
pub(crate) fn unsigned(value: &Json) -> Result<u64> {
    match value {
        Json::Number(Number::Integer(n)) => n
            .as_u64()
            .ok_or(Error::Invalid("integer exceeds unsigned 64-bit range")),
        Json::Bool(value) => Ok(u64::from(*value)),
        _ => Err(Error::Invalid("integer metadata requires an integer")),
    }
}
pub(crate) fn u32_value(value: &Json) -> Result<u32> {
    u32::try_from(unsigned(value)?).map_err(|_| Error::Invalid("UINT32 metadata overflow"))
}
pub(crate) fn real(value: &Json) -> Result<f64> {
    match value {
        Json::Number(n) => Ok(n.as_f64()?),
        Json::Bool(value) => Ok(f64::from(*value)),
        _ => Err(Error::Invalid("float metadata requires a number")),
    }
}
pub(crate) fn f32_value(value: &Json) -> Result<f32> {
    let value = real(value)?;
    let narrowed = value as f32;
    if value.is_finite() && !narrowed.is_finite() {
        return Err(Error::Invalid("FLOAT32 metadata overflow"));
    }
    Ok(narrowed)
}
pub(crate) fn truth(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(value) => *value,
        Json::Number(Number::Integer(n)) => n.digits() != "0",
        Json::Number(Number::Float(value)) => *value != 0.0,
        Json::Text(text) => !text.as_wtf8().is_empty(),
        Json::Array(values) => !values.is_empty(),
        Json::Object(values) => !values.is_empty(),
    }
}
pub(crate) fn present(value: Option<&Json>) -> Option<&Json> {
    value.filter(|v| !matches!(v, Json::Null))
}
#[derive(Debug, Default)]
pub(crate) struct Metadata(pub(crate) Vec<(String, Value)>);
impl Metadata {
    pub(crate) fn put(&mut self, key: &str, value: Value) -> Result<()> {
        if let Some((_, current)) = self.0.iter_mut().find(|(name, _)| name == key) {
            *current = value;
            return Ok(());
        }
        reserve(&mut self.0, 1, "metadata entries allocation")?;
        self.0.push((string(key)?, value));
        Ok(())
    }
    pub(crate) fn string(&mut self, key: &str, value: &str) -> Result<()> {
        // Frozen GGUFWriter.add_string skips empty strings.
        if value.is_empty() {
            return Ok(());
        }
        self.bytes(key, value.as_bytes())
    }
    pub(crate) fn bytes(&mut self, key: &str, value: &[u8]) -> Result<()> {
        if value.is_empty() {
            return Ok(());
        }
        self.put(key, Value::String(crate::bytes(value)?))
    }
    pub(crate) fn boolean(&mut self, key: &str, value: bool) -> Result<()> {
        self.put(key, Value::Bool(value))
    }
    pub(crate) fn uint(&mut self, key: &str, value: u32) -> Result<()> {
        self.put(key, Value::U32(value))
    }
    pub(crate) fn float(&mut self, key: &str, value: f32) -> Result<()> {
        self.put(key, Value::F32(value))
    }
    pub(crate) fn json_uint(&mut self, key: &str, value: &Json) -> Result<()> {
        self.uint(key, u32_value(value)?)
    }
    pub(crate) fn json_float(&mut self, key: &str, value: &Json) -> Result<()> {
        self.float(key, f32_value(value)?)
    }
    pub(crate) fn json_number_or_array(&mut self, key: &str, value: &Json) -> Result<()> {
        if !matches!(value, Json::Array(_)) {
            return self.json_uint(key, value);
        }
        let values = value.as_array().ok_or(Error::Invalid("number array"))?;
        if values.is_empty() {
            return Ok(());
        }
        let mut converted = Vec::new();
        reserve(&mut converted, values.len(), "metadata numeric array")?;
        let kind = match values[0] {
            Json::Bool(_) => 7,
            Json::Number(Number::Integer(_)) => 5,
            Json::Number(Number::Float(_)) => 6,
            _ => return Err(Error::Invalid("numeric metadata array element type")),
        };
        for item in values {
            let item_kind = match item {
                Json::Bool(_) => 7,
                Json::Number(Number::Integer(_)) => 5,
                Json::Number(Number::Float(_)) => 6,
                _ => return Err(Error::Invalid("numeric metadata array element type")),
            };
            if item_kind != kind {
                return Err(Error::Invalid("heterogeneous GGUF metadata array"));
            }
            converted.push(match kind {
                7 => Value::Bool(truth(item)),
                5 => Value::I32(
                    i32::try_from(integer(item)?)
                        .map_err(|_| Error::Invalid("INT32 metadata overflow"))?,
                ),
                _ => Value::F32(f32_value(item)?),
            });
        }
        self.put(
            key,
            Value::Array {
                element_type: kind,
                values: converted,
            },
        )
    }
    pub(crate) fn strings(
        &mut self,
        key: &str,
        values: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let mut items = Vec::new();
        for value in values {
            reserve(&mut items, 1, "string metadata array")?;
            items.push(Value::String(value.into_bytes()));
        }
        if items.is_empty() {
            return Ok(());
        }
        self.put(
            key,
            Value::Array {
                element_type: 8,
                values: items,
            },
        )
    }
}
