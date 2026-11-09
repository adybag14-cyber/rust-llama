//! Frozen server embedding request and response semantics.
//!
//! The source of the wire contract is tools/server/server-context.cpp,
//! server-common.cpp and server-task.cpp in the pinned archive. Text and token
//! input shapes are preserved; media inputs require a multimodal runtime.
//! Encoding is bounded before any response headers are written.
use crate::provider::{EmbeddingInfo, InferenceProvider, ProviderError};
use llama_behavior::json::{self, Value};

pub(crate) struct Prepared {
    pub inputs: Vec<Vec<i32>>,
    pub normalization: i32,
    openai: bool,
    base64: bool,
    model: String,
    prompt_tokens: u64,
}

fn integer(value: &Value) -> bool {
    matches!(value, Value::Integer(_) | Value::Unsigned(_))
}

/// Preserve the source JSON library's arithmetic conversions: integer casts
/// retain the low 32 bits, booleans map to 0/1, and floats truncate toward zero.
/// Out-of-range floating casts remain explicit errors because the native result
/// is undefined, unlike the measured native two's-complement integer conversion.
fn signed_parameter(value: &Value, name: &str) -> Result<i32, ProviderError> {
    match value {
        Value::Integer(value) => return Ok(*value as i32),
        Value::Unsigned(value) => return Ok(*value as i32),
        Value::Bool(value) => return Ok(i32::from(*value)),
        _ => {}
    }
    let value = value
        .as_f64()
        .ok_or_else(|| ProviderError::invalid(format!("{name} must be numeric")))?;
    let truncated = value.trunc();
    if !value.is_finite() || truncated < f64::from(i32::MIN) || truncated > f64::from(i32::MAX) {
        return Err(ProviderError::invalid(format!("{name} must fit in i32")));
    }
    Ok(truncated as i32)
}

fn mixed(input: &Value, provider: &dyn InferenceProvider) -> Result<Vec<i32>, ProviderError> {
    if let Some(text) = input.as_str() {
        return provider.tokenize(text, true);
    }
    let parts = input.as_array().ok_or_else(|| {
        ProviderError::invalid("prompt_string must be a string or an array of text and token IDs")
    })?;
    let mut tokens = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(text) = part.as_str() {
            tokens.extend(provider.tokenize(text, index == 0)?);
        } else {
            tokens.push(signed_parameter(part, "token ID")?);
        }
    }
    Ok(tokens)
}

fn tokenize_entry(
    input: &Value,
    provider: &dyn InferenceProvider,
) -> Result<Vec<i32>, ProviderError> {
    if let Some(object) = input.as_object()
        && let Some(content) = object.get("content")
    {
        let content = content
            .as_array()
            .ok_or_else(|| ProviderError::invalid("\"content\" must be an array"))?;
        let mut text = String::new();
        for part in content {
            let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
            match kind {
                "text" | "tool_reference" => {
                    // The reference json_value helper uses its empty-string
                    // fallback for a missing or incorrectly typed text field.
                    if let Some(piece) = part.get("text").and_then(Value::as_str) {
                        text.push_str(piece);
                    }
                }
                "image_url" | "input_audio" | "input_video" => {
                    return Err(ProviderError::unsupported(
                        "Media embeddings require a multimodal runtime",
                    ));
                }
                _ => return Err(ProviderError::invalid("unsupported content[].type")),
            }
        }
        return provider.tokenize(&text, true);
    }
    if input.as_str().is_some() {
        return mixed(input, provider);
    }
    if let Some(parts) = input.as_array() {
        let has_number = parts.iter().any(integer);
        let has_string = parts.iter().any(|part| part.as_str().is_some());
        if has_number && has_string {
            return mixed(input, provider);
        }
        if parts.iter().all(integer) {
            return parts
                .iter()
                .map(|part| signed_parameter(part, "token ID"))
                .collect();
        }
    }
    if let Some(prompt) = input.get("prompt_string") {
        if input.get("multimodal_data").is_some() {
            return Err(ProviderError::unsupported(
                "Multimodal data provided, but this model does not support multimodal requests",
            ));
        }
        return mixed(prompt, provider);
    }
    Err(ProviderError::invalid(
        "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens",
    ))
}

pub(crate) fn prepare(
    body: &Value,
    openai: bool,
    info: EmbeddingInfo,
    provider: &dyn InferenceProvider,
    max_inputs: usize,
    max_response_bytes: usize,
) -> Result<Prepared, ProviderError> {
    if body.as_object().is_none() {
        return Err(ProviderError::invalid("Request body must be an object"));
    }
    let (input, openai) = if let Some(input) = body.get("input") {
        (input, openai)
    } else if let Some(content) = body.get("content") {
        // Even /v1/embeddings returns the native shape for this legacy key.
        (content, false)
    } else {
        return Err(ProviderError::invalid(
            "\"input\" or \"content\" must be provided",
        ));
    };
    let base64 = match body.get("encoding_format") {
        None => false,
        Some(Value::String(format)) if format == "float" => false,
        Some(Value::String(format)) if format == "base64" => true,
        _ => {
            return Err(ProviderError::invalid(
                "The format to return the embeddings in. Can be either float or base64",
            ));
        }
    };
    let normalization = body
        .get("embd_normalize")
        .map(|value| signed_parameter(value, "embd_normalize"))
        .transpose()?
        .unwrap_or(info.normalization);
    let entries: Vec<&Value> = match input.as_array() {
        Some(parts) if !parts.iter().any(integer) => parts.iter().collect(),
        _ => vec![input],
    };
    if entries.is_empty() {
        return Err(ProviderError::invalid("\"input\" must not be empty"));
    }
    if entries.len() > max_inputs {
        return Err(ProviderError::new(
            413,
            "request_too_large",
            "Embedding input count exceeds the configured limit",
        ));
    }
    if info.dimensions == 0 || info.context_length == 0 || info.vocabulary_size == 0 {
        return Err(ProviderError::internal(
            "Provider returned invalid embedding dimensions",
        ));
    }
    let mut inputs = Vec::with_capacity(entries.len());
    let mut prompt_tokens = 0_u64;
    let mut raw_bytes = 0_usize;
    for entry in entries {
        let tokens = tokenize_entry(entry, provider)?;
        if tokens.is_empty() {
            return Err(ProviderError::invalid("Input content cannot be empty"));
        }
        if tokens.len() > info.context_length {
            return Err(ProviderError::invalid(format!(
                "Input too large: {} tokens exceed the context limit of {}",
                tokens.len(),
                info.context_length,
            )));
        }
        if tokens
            .iter()
            .any(|&token| token < 0 || token as usize >= info.vocabulary_size)
        {
            return Err(ProviderError::invalid("Input contains an invalid token ID"));
        }
        let rows = if info.pooled { 1 } else { tokens.len() };
        let bytes = rows
            .checked_mul(info.dimensions)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(response_too_large)?;
        raw_bytes = raw_bytes
            .checked_add(bytes)
            .ok_or_else(response_too_large)?;
        if raw_bytes > max_response_bytes {
            return Err(response_too_large());
        }
        prompt_tokens = prompt_tokens
            .checked_add(tokens.len() as u64)
            .ok_or_else(|| ProviderError::invalid("Embedding token count overflow"))?;
        inputs.push(tokens);
    }
    // This is an output label, as in the original single-model embedding route.
    // It does not select or swap the loaded model.
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| provider.models().first().map(|model| model.id.clone()))
        .unwrap_or_else(|| "provider".into());
    Ok(Prepared {
        inputs,
        normalization,
        openai,
        base64,
        model,
        prompt_tokens,
    })
}

fn response_too_large() -> ProviderError {
    ProviderError::new(
        413,
        "response_too_large",
        "Embedding output exceeds the configured response limit",
    )
}

/// Accumulates one complete bounded response. Partial JSON is never sent on
/// provider failure, cancellation or overflow.
pub(crate) struct Response {
    body: String,
    limit: usize,
    openai: bool,
    base64: bool,
    count: usize,
}

impl Response {
    pub(crate) fn new(request: &Prepared, limit: usize) -> Result<Self, ProviderError> {
        let mut response = Self {
            body: String::with_capacity(limit.min(8192)),
            limit,
            openai: request.openai,
            base64: request.base64,
            count: 0,
        };
        if request.openai {
            let model = json::stringify(&Value::String(request.model.clone()))
                .map_err(|error| ProviderError::internal(error.to_string()))?;
            response.append(&format!(
                "{{\"object\":\"list\",\"model\":{model},\"usage\":{{\"prompt_tokens\":{},\"total_tokens\":{}}},\"data\":[",
                request.prompt_tokens, request.prompt_tokens,
            ))?;
        } else {
            response.append("[")?;
        }
        Ok(response)
    }

    fn append(&mut self, text: &str) -> Result<(), ProviderError> {
        if text.len() > self.limit.saturating_sub(self.body.len()) {
            return Err(response_too_large());
        }
        self.body.push_str(text);
        Ok(())
    }

    fn vector(&mut self, values: &[f32]) -> Result<(), ProviderError> {
        self.append("[")?;
        for (index, &value) in values.iter().enumerate() {
            if index != 0 {
                self.append(",")?;
            }
            // Debug formatting preserves a decimal marker and negative zero.
            // The original JSON serializer also widens f32 values to f64.
            self.append(&format!("{:?}", f64::from(value)))?;
        }
        self.append("]")
    }

    fn base64_vector(&mut self, values: &[f32]) -> Result<(), ProviderError> {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut bytes = values.iter().flat_map(|value| value.to_ne_bytes());
        self.append("\"")?;
        while let Some(first) = bytes.next() {
            let second = bytes.next();
            let third = bytes.next();
            let quartet = [
                ALPHABET[(first >> 2) as usize],
                ALPHABET[(((first & 3) << 4) | (second.unwrap_or(0) >> 4)) as usize],
                second.map_or(b'=', |second| {
                    ALPHABET[(((second & 15) << 2) | (third.unwrap_or(0) >> 6)) as usize]
                }),
                third.map_or(b'=', |third| ALPHABET[(third & 63) as usize]),
            ];
            // All four bytes come from the ASCII alphabet or padding above.
            self.append(std::str::from_utf8(&quartet).unwrap())?;
        }
        self.append("\"")
    }

    pub(crate) fn push(
        &mut self,
        vectors: &[Vec<f32>],
        info: EmbeddingInfo,
        input_tokens: usize,
    ) -> Result<(), ProviderError> {
        let expected_rows = if info.pooled { 1 } else { input_tokens };
        if vectors.len() != expected_rows
            || vectors
                .iter()
                .any(|row| row.len() != info.dimensions || row.iter().any(|x| !x.is_finite()))
        {
            return Err(ProviderError::internal(
                "Provider returned invalid embedding shape or nonfinite values",
            ));
        }
        if self.count != 0 {
            self.append(",")?;
        }
        self.append(&format!("{{\"index\":{},\"embedding\":", self.count))?;
        if self.openai {
            if self.base64 {
                self.base64_vector(&vectors[0])?;
                self.append(",\"encoding_format\":\"base64\"")?;
            } else {
                self.vector(&vectors[0])?;
            }
            self.append(",\"object\":\"embedding\"}")?;
        } else {
            self.append("[")?;
            for (index, vector) in vectors.iter().enumerate() {
                if index != 0 {
                    self.append(",")?;
                }
                self.vector(vector)?;
            }
            self.append("]}")?;
        }
        self.count += 1;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<String, ProviderError> {
        self.append(if self.openai { "]}" } else { "]" })?;
        Ok(self.body)
    }
}

/// common/common.cpp::common_embd_normalize. Products for the Euclidean
/// reduction are rounded to f32 before widening; the reciprocal is narrowed
/// once, then every output uses a single f32 multiplication.
pub(crate) fn normalize(values: &mut [f32], mode: i32) -> Result<(), ProviderError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ProviderError::internal("Nonfinite raw embedding"));
    }
    let sum = match mode {
        -1 => 1.0,
        0 => {
            values
                .iter()
                .fold(0.0_f64, |max, value| max.max(f64::from(value.abs())))
                / 32760.0
        }
        2 => values
            .iter()
            .map(|value| f64::from(value * value))
            .sum::<f64>()
            .sqrt(),
        _ => values
            .iter()
            .map(|value| f64::from(value.abs()).powf(f64::from(mode)))
            .sum::<f64>()
            .powf(1.0 / f64::from(mode)),
    };
    let scale = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    for value in values {
        *value *= scale;
        if !value.is_finite() {
            return Err(ProviderError::internal("Nonfinite normalized embedding"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(value: &Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect()
    }

    fn check(raw: &Value, expected: &Value, mode: i32, compared: &mut usize) {
        let mut actual = row(raw);
        let expected = row(expected);
        normalize(&mut actual, mode).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
            let ulps = actual.to_bits().abs_diff(expected.to_bits());
            let allowed = if cfg!(all(
                target_os = "windows",
                target_arch = "x86_64",
                target_env = "msvc"
            )) {
                0
            } else {
                2
            };
            assert!(
                ulps <= allowed,
                "normalization {mode}: {actual} != {expected}, {ulps} ULPs"
            );
            *compared += 1;
        }
    }

    #[test]
    fn normalization_matches_original_server_from_identical_native_raw_inputs() {
        let fixture = json::parse(include_str!(
            "../../../fixtures/inference/embedding-http-bge.json"
        ))
        .unwrap();
        let mut compared = 0;
        for group in fixture.get("cases_by_pooling").unwrap().as_array().unwrap() {
            if group.get("pooling").unwrap().as_str() == Some("none") {
                continue;
            }
            let cases = group.get("cases").unwrap().as_array().unwrap();
            let find = |name| {
                cases
                    .iter()
                    .find(|case| case.get("name").and_then(Value::as_str) == Some(name))
                    .unwrap()
                    .get("response")
                    .unwrap()
                    .get("body")
                    .unwrap()
            };
            let raw = find("raw_batch").as_array().unwrap();
            let normalized = find("oai_batch").get("data").unwrap().as_array().unwrap();
            for (raw, normalized) in raw.iter().zip(normalized) {
                check(
                    &raw.get("embedding").unwrap().as_array().unwrap()[0],
                    normalized.get("embedding").unwrap(),
                    2,
                    &mut compared,
                );
            }
            if group.get("pooling").unwrap().as_str() == Some("model") {
                for (name, mode) in [("norm_0", 0), ("norm_1", 1), ("norm_3", 3)] {
                    check(
                        &raw[0].get("embedding").unwrap().as_array().unwrap()[0],
                        &find(name).as_array().unwrap()[0]
                            .get("embedding")
                            .unwrap()
                            .as_array()
                            .unwrap()[0],
                        mode,
                        &mut compared,
                    );
                }
            }
        }
        assert_eq!(compared, 4608);
    }
}
