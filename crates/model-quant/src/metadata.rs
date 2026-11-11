//! Frozen model-file admission, separate from the general tensor type planner.
//! Sources: src/llama-model-loader.cpp, src/llama-model.cpp and the eight
//! admitted src/models/*.cpp files. Original MSVC/GNU captures were sealed
//! before this module; see tests/fixtures/metadata/provenance.json.
//! Copyright (c) 2023-2026 The ggml authors. MIT license.
use crate::{Error, ModelDescriptor, OverrideValue, QuantizeParams, Result, invalid};
use llama_gguf::{GgufSet, Value};
use std::collections::HashSet;

const FILE_ARCHITECTURES: &[&str] = &[
    "llama", "qwen2", "qwen3", "gpt2", "gptneox", "bert", "t5", "bloom",
];
const MAX_LAYERS: usize = 512;
const MAX_EXPERTS: u32 = 1024;

fn c_prefix(value: &[u8]) -> &[u8] {
    &value[..value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len())]
}
fn required<T>(value: Option<T>, key: &str) -> Result<T> {
    value.ok_or_else(|| invalid(format!("missing model key {key}")))
}
struct Metadata<'a> {
    set: &'a GgufSet,
    params: &'a QuantizeParams,
}
impl<'a> Metadata<'a> {
    fn override_for(&self, key: &str) -> Option<&'a OverrideValue> {
        self.params
            .kv_overrides
            .iter()
            .find(|value| c_prefix(&value.key) == key.as_bytes())
            .map(|value| &value.value)
    }
    fn u32(&self, key: &str) -> Result<Option<u32>> {
        if let Some(OverrideValue::Integer(value)) = self.override_for(key) {
            // Native assignment from int64_t to uint32_t is reduction modulo
            // 2^32. Apply field/resource bounds after this source-defined cast.
            return Ok(Some(*value as u32));
        }
        match self.set.metadata().get(key) {
            Some(Value::U32(value)) => Ok(Some(*value)),
            Some(_) => Err(invalid(format!("{key} must be scalar UINT32"))),
            None => Ok(None),
        }
    }
    fn f32(&self, key: &str) -> Result<Option<f32>> {
        if let Some(OverrideValue::Float(value)) = self.override_for(key) {
            return Ok(Some(*value as f32));
        }
        match self.set.metadata().get(key) {
            Some(Value::F32(value)) => Ok(Some(*value)),
            Some(_) => Err(invalid(format!("{key} must be scalar FLOAT32"))),
            None => Ok(None),
        }
    }
    fn boolean(&self, key: &str) -> Result<Option<bool>> {
        if let Some(OverrideValue::Bool(value)) = self.override_for(key) {
            return Ok(Some(*value));
        }
        match self.set.metadata().get(key) {
            Some(Value::Bool(value)) => Ok(Some(*value)),
            Some(_) => Err(invalid(format!("{key} must be scalar BOOL"))),
            None => Ok(None),
        }
    }
    fn string(&self, key: &str) -> Result<Option<&'a [u8]>> {
        if let Some(OverrideValue::String(value)) = self.override_for(key) {
            return Ok(Some(c_prefix(value)));
        }
        match self.set.metadata().get(key) {
            Some(value) => value
                .as_bytes()
                .map(|bytes| Some(c_prefix(bytes)))
                .ok_or_else(|| invalid(format!("{key} must be scalar STRING"))),
            None => Ok(None),
        }
    }
    // get_arr ignores scalar overrides. Optional missing or scalar metadata
    // returns false, while an array with the wrong element type is an error.
    fn array(&self, key: &str, required: bool, types: &[u32]) -> Result<Option<&'a [Value]>> {
        match self.set.metadata().get(key) {
            Some(Value::Array {
                element_type,
                values,
            }) => {
                if !types.contains(element_type)
                    || values.iter().any(|value| value.type_id() != *element_type)
                {
                    return Err(invalid(format!("{key} has the wrong array element type")));
                }
                Ok(Some(values))
            }
            _ if required => Err(invalid(format!("missing array key {key}"))),
            _ => Ok(None),
        }
    }
    fn strings(&self, key: &str, required: bool) -> Result<Option<&'a [Value]>> {
        self.array(key, required, &[8])
    }
    fn layers(&self, key: &str, count: usize, result: &mut [u32; MAX_LAYERS]) -> Result<()> {
        // The original checks existence before consulting overrides, and arrays
        // bypass overrides entirely. Do not introduce an absent key here.
        let Some(value) = self.set.metadata().get(key) else {
            return Ok(());
        };
        match value {
            Value::Array { .. } => {
                let values = self
                    .array(key, true, &[4, 5, 7])?
                    .ok_or_else(|| invalid(format!("missing array key {key}")))?;
                if values.len() != count {
                    return Err(invalid(format!(
                        "{key} array length {} differs from block count {count}",
                        values.len()
                    )));
                }
                for (target, value) in result[..count].iter_mut().zip(values) {
                    *target = match value {
                        Value::U32(value) => *value,
                        Value::I32(value) => *value as u32,
                        Value::Bool(value) => u32::from(*value),
                        _ => return Err(invalid(format!("{key} array element type"))),
                    };
                }
            }
            _ => result[..count].fill(required(self.u32(key)?, key)?),
        }
        Ok(())
    }
    fn hadamard(&self, arch: &str) -> Result<()> {
        let tied = self.boolean("prism.hadamard.tied_output")?.unwrap_or(false);
        let Some(version) = self.u32("prism.hadamard.version")? else {
            return if tied {
                Err(invalid("prism.hadamard.tied_output requires version 2"))
            } else {
                Ok(())
            };
        };
        if !matches!(version, 1 | 2) || (version == 2) != tied {
            return Err(invalid(
                "invalid prism.hadamard version/tied_output combination",
            ));
        }
        let has_output = self.set.tensor("output.weight").is_some();
        if tied && has_output {
            return Err(invalid(
                "tied Hadamard output requires output.weight to be absent",
            ));
        }
        let block_key = "prism.hadamard.block_size";
        let block = required(self.u32(block_key)?, block_key)?;
        if !block.is_power_of_two() {
            return Err(invalid(
                "Hadamard block size must be a positive power of two",
            ));
        }
        for (key, expected) in [
            (
                "prism.hadamard.transform",
                b"normalized-sylvester-walsh-hadamard".as_slice(),
            ),
            ("prism.hadamard.axis", b"input-last-dimension".as_slice()),
        ] {
            if required(self.string(key)?, key)? != expected {
                return Err(invalid(format!("unsupported {key}")));
            }
        }
        let sign_key = "prism.hadamard.sign_mode";
        let sign_mode = required(self.string(sign_key)?, sign_key)?;
        if !matches!(sign_mode, b"identity" | b"explicit") {
            return Err(invalid("unsupported Hadamard sign mode"));
        }
        let names_key = "prism.hadamard.weight_names";
        let names = required(self.strings(names_key, true)?, names_key)?;
        if names.is_empty() {
            return Err(invalid("Hadamard weight names are empty"));
        }
        if sign_mode == b"explicit" {
            let width_key = "prism.hadamard.sign_widths";
            let value_key = "prism.hadamard.sign_values";
            let widths = required(self.array(width_key, true, &[4, 5])?, width_key)?;
            let signs = required(self.array(value_key, true, &[4, 5])?, value_key)?;
            if widths.is_empty() {
                return Err(invalid("explicit Hadamard signs require nonempty widths"));
            }
            let mut seen = HashSet::new();
            seen.try_reserve(widths.len())
                .map_err(|_| Error::Limit("Hadamard sign-width bookkeeping".into()))?;
            let mut offset = 0usize;
            for width in widths {
                let width = signed_element(width)?;
                if width <= 0 || !(width as u32).is_multiple_of(block) || !seen.insert(width) {
                    return Err(invalid("invalid or duplicate Hadamard sign width"));
                }
                let end = offset
                    .checked_add(width as usize)
                    .filter(|end| *end <= signs.len())
                    .ok_or_else(|| invalid("Hadamard sign width exceeds values"))?;
                for value in &signs[offset..end] {
                    if !matches!(signed_element(value)?, -1 | 1) {
                        return Err(invalid("Hadamard sign values must be +/-1"));
                    }
                }
                offset = end;
            }
            if offset != signs.len() {
                return Err(invalid("Hadamard sign value length mismatch"));
            }
        }
        self.boolean("prism.hadamard.gdn_v_grouped")?;
        if !matches!(arch, "llama" | "qwen3") {
            return Err(invalid(
                "Hadamard transforms are not verified by this source architecture",
            ));
        }
        let mut weights = HashSet::new();
        weights
            .try_reserve(names.len())
            .map_err(|_| Error::Limit("Hadamard weight-name bookkeeping".into()))?;
        for name in names {
            let name = c_prefix(
                name.as_bytes()
                    .ok_or_else(|| invalid("Hadamard weight name is not a string"))?,
            );
            if !foldable(name) || !weights.insert(name) {
                return Err(invalid("unverified or duplicate Hadamard weight"));
            }
        }
        let inverse = self
            .strings("prism.hadamard.inverse_weight_names", false)?
            .unwrap_or(&[]);
        let mut latent_token = false;
        for name in inverse {
            let name = c_prefix(
                name.as_bytes()
                    .ok_or_else(|| invalid("Hadamard inverse name is not a string"))?,
            );
            if name != b"token_embd.weight" || latent_token || weights.contains(name) {
                return Err(invalid("unverified or duplicate Hadamard inverse weight"));
            }
            latent_token = true;
        }
        if tied && !latent_token {
            return Err(invalid(
                "tied Hadamard output requires a latent token embedding",
            ));
        }
        if !tied && latent_token && !has_output {
            return Err(invalid(
                "a tied Hadamard output requires version 2 and tied_output",
            ));
        }
        Ok(())
    }
}
fn signed_element(value: &Value) -> Result<i32> {
    match value {
        Value::U32(value) => Ok(*value as i32),
        Value::I32(value) => Ok(*value),
        _ => Err(invalid(
            "Hadamard signs require UINT32 or INT32 array elements",
        )),
    }
}
fn foldable(name: &[u8]) -> bool {
    if name == b"output.weight" {
        return true;
    }
    let Some(tail) = name.strip_prefix(b"blk.") else {
        return false;
    };
    let digits = tail.iter().take_while(|byte| byte.is_ascii_digit()).count();
    if digits == 0 || tail.get(digits) != Some(&b'.') {
        return false;
    }
    [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_qkv",
        "attn_gate",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_up_exps",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
        "ssm_out",
    ]
    .iter()
    .any(|kind| {
        tail[digits + 1..]
            .strip_suffix(b".weight")
            .is_some_and(|value| value == kind.as_bytes())
    })
}

/// Scalar getter used for general.file_type. Arrays are never scalar metadata.
pub(crate) fn metadata_u32(
    set: &GgufSet,
    params: &QuantizeParams,
    key: &str,
    default: Option<u32>,
) -> Result<u32> {
    required(Metadata { set, params }.u32(key)?.or(default), key)
}

/// Admit only the file-driver families whose source metadata contracts are
/// implemented here. The broad low-level tensor planner has separate scope.
pub(crate) fn model_descriptor(set: &GgufSet, params: &QuantizeParams) -> Result<ModelDescriptor> {
    let metadata = Metadata { set, params };
    let arch = required(
        metadata.string("general.architecture")?,
        "general.architecture",
    )?;
    let arch = std::str::from_utf8(arch)
        .map_err(|_| Error::Unsupported("non-UTF-8 file architecture".into()))?;
    if !FILE_ARCHITECTURES.contains(&arch) {
        return Err(Error::Unsupported(format!(
            "file quantization architecture {arch}; admitted families are {}",
            FILE_ARCHITECTURES.join(", ")
        )));
    }
    // The original loader reads this for every conversion, not only only_copy.
    metadata.u32("general.file_type")?;
    metadata.hadamard(arch)?;
    metadata.string("general.name")?;
    let key = |suffix: &str| format!("{arch}.{suffix}");
    required(
        metadata.u32(&key("context_length"))?,
        &key("context_length"),
    )?;
    let n_embd = required(
        metadata.u32(&key("embedding_length"))?,
        &key("embedding_length"),
    )?;
    for suffix in [
        "embedding_length_out",
        "pooling_type",
        "classifier.pooling_type",
    ] {
        metadata.u32(&key(suffix))?;
    }
    for suffix in ["attention.causal", "classifier_free_guidance"] {
        metadata.boolean(&key(suffix))?;
    }
    if let Some(activation) = metadata.string(&key("classifier.activation"))?
        && !matches!(activation, b"gelu" | b"silu" | b"tanh")
    {
        return Err(invalid("unsupported classifier activation"));
    }
    let n_layer = required(metadata.u32(&key("block_count"))?, &key("block_count"))?;
    if n_layer == 0 || n_layer as usize > MAX_LAYERS {
        return Err(invalid(
            "model block count is outside the frozen 1..=512 limit",
        ));
    }
    let nextn = metadata.u32(&key("nextn_predict_layers"))?.unwrap_or(0);
    if nextn > n_layer {
        return Err(invalid("nextn layer count exceeds total block count"));
    }
    let n_expert = metadata.u32(&key("expert_count"))?.unwrap_or(0);
    let mut expert_used = [0; MAX_LAYERS];
    metadata.layers(
        &key("expert_used_count"),
        n_layer as usize,
        &mut expert_used,
    )?;
    let groups = metadata.u32(&key("expert_group_count"))?.unwrap_or(0);
    let group_used = metadata.u32(&key("expert_group_used_count"))?.unwrap_or(0);
    let used_max = expert_used[..n_layer as usize]
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if n_expert > MAX_EXPERTS
        || used_max > n_expert
        || (n_expert > 0 && (used_max == 0 || groups >= n_expert))
        || (n_expert == 0 && (used_max != 0 || groups != 0))
        || (n_expert > 0
            && groups > 1
            && (!n_expert.is_multiple_of(groups) || group_used == 0 || group_used >= groups))
    {
        return Err(invalid("invalid frozen expert routing/group geometry"));
    }
    let mut feed_forward = [0; MAX_LAYERS];
    let mut heads = [0; MAX_LAYERS];
    metadata.layers(
        &key("feed_forward_length"),
        n_layer as usize,
        &mut feed_forward,
    )?;
    metadata.layers(&key("attention.head_count"), n_layer as usize, &mut heads)?;
    let mut kv_heads = heads;
    metadata.layers(
        &key("attention.head_count_kv"),
        n_layer as usize,
        &mut kv_heads,
    )?;
    metadata.boolean(&key("rope.scaling.finetuned"))?;
    metadata.u32(&key("rope.scaling.original_context_length"))?;
    metadata.f32(&key("rope.freq_base"))?;
    let scaling = metadata
        .string(&key("rope.scaling.type"))?
        .unwrap_or(b"linear");
    if !matches!(scaling, b"none" | b"linear" | b"yarn" | b"longrope") {
        return Err(invalid("unknown RoPE scaling type"));
    }
    let section = metadata
        .string(&key("rope.section_order"))?
        .unwrap_or(b"tyxz");
    // None of these eight frozen families constructs an M-RoPE model.
    if section != b"tyxz" {
        return Err(invalid(
            "unknown RoPE section order or M-RoPE-only ordering",
        ));
    }
    if metadata.f32(&key("rope.scaling.factor"))?.is_none() {
        metadata.f32(&key("rope.scale_linear"))?;
    }
    for suffix in ["rope.scaling.attn_factor", "rope.scaling.alpha"] {
        metadata.f32(&key(suffix))?;
    }
    let (head_k, head_v) = if let Some(default) = n_embd.checked_div(heads[0]) {
        let head_k = metadata
            .u32(&key("attention.key_length"))?
            .unwrap_or(default);
        let head_v = metadata
            .u32(&key("attention.value_length"))?
            .unwrap_or(default);
        let rotary = metadata
            .u32(&key("rope.dimension_count"))?
            .unwrap_or(head_k);
        if arch == "llama" && rotary != head_k {
            return Err(invalid(
                "Llama rotary dimension differs from key head width",
            ));
        }
        (head_k, head_v)
    } else {
        (0, 0)
    };
    for suffix in [
        "attention.key_length_swa",
        "attention.value_length_swa",
        "rope.dimension_count_swa",
    ] {
        metadata.u32(&key(suffix))?;
    }
    metadata.strings(&key("classifier.output_labels"), false)?;
    metadata.strings(&key("decision.labels"), false)?;

    let epsilon = if matches!(arch, "llama" | "qwen2" | "qwen3" | "t5") {
        "attention.layer_norm_rms_epsilon"
    } else {
        "attention.layer_norm_epsilon"
    };
    required(metadata.f32(&key(epsilon))?, &key(epsilon))?;
    match arch {
        "llama" => {
            if metadata.u32(&key("vocab_size"))?.is_none()
                && let Some(value) = set.metadata().get("tokenizer.ggml.tokens")
                && !matches!(value, Value::Array { .. })
            {
                return Err(invalid("tokenizer.ggml.tokens must be an array"));
            }
        }
        "gptneox" => {
            required(
                metadata.boolean(&key("use_parallel_residual"))?,
                &key("use_parallel_residual"),
            )?;
        }
        "t5" => {
            required(
                metadata.u32(&key("attention.relative_buckets_count"))?,
                &key("attention.relative_buckets_count"),
            )?;
            metadata.u32(&key("decoder_start_token_id"))?;
            metadata.u32(&key("decoder_block_count"))?;
        }
        _ => {}
    }
    Ok(ModelDescriptor {
        architecture: arch.into(),
        n_embd,
        n_ff: feed_forward[0],
        n_layer,
        n_head: heads[0],
        n_head_kv: kv_heads[0],
        n_expert,
        n_embd_head_k: head_k,
        n_embd_head_v: head_v,
        llama_70b: Some(
            arch == "llama" && n_expert != 8 && n_layer - nextn == 80 && heads[0] != kv_heads[0],
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileType, MetadataOverride, quantize_file};
    use llama_behavior::json::{self, Value as Json};
    use llama_gguf::{Limits, RawTensorDescriptor, RawValue, write_raw_header};
    use llama_quant::DType;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
        value
            .get(key)
            .unwrap_or_else(|| panic!("fixture field {key}"))
    }
    fn text(value: &Json) -> &str {
        value.as_str().expect("fixture string")
    }
    fn array(value: &Json) -> &[Json] {
        value.as_array().expect("fixture array")
    }
    fn integer(value: &Json) -> i64 {
        value.as_i64().expect("fixture integer")
    }
    fn document(name: &str) -> Json {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/metadata")
            .join(name);
        json::parse(&fs::read_to_string(path).expect("pinned metadata fixture"))
            .expect("fixture JSON")
    }
    fn raw_value(kind: &str, value: &Json) -> RawValue {
        if let Some(scalar) = kind.strip_suffix('s') {
            let element_type = match scalar {
                "u32" => 4,
                "i32" => 5,
                "f32" => 6,
                "bool" => 7,
                "str" => 8,
                _ => panic!("fixture array kind"),
            };
            return RawValue::Array {
                element_type,
                values: array(value).iter().map(|v| raw_value(scalar, v)).collect(),
            };
        }
        match kind {
            "u32" => RawValue::U32(integer(value) as u32),
            "i32" => RawValue::I32(integer(value) as i32),
            "f32" => RawValue::F32(value.as_f64().expect("fixture float") as f32),
            "bool" => RawValue::Bool(value.as_bool().expect("fixture boolean")),
            "str" => RawValue::String(text(value).as_bytes().to_vec()),
            _ => panic!("fixture scalar kind"),
        }
    }
    fn input(case: &Json) -> Vec<u8> {
        let metadata: Vec<_> = array(field(case, "metadata"))
            .iter()
            .map(|entry| {
                (
                    text(field(entry, "key")).as_bytes().to_vec(),
                    raw_value(text(field(entry, "type")), field(entry, "value")),
                )
            })
            .collect();
        let names = [
            b"token_embd.weight".as_slice(),
            b"blk.0.attn_v.weight".as_slice(),
            b"output.weight".as_slice(),
        ];
        let count = if field(case, "output_weight").as_bool().unwrap() {
            3
        } else {
            2
        };
        let dimensions = [256, 4];
        let tensors: Vec<_> = names[..count]
            .iter()
            .map(|name| RawTensorDescriptor {
                name: *name,
                dimensions: &dimensions,
                dtype: DType::F32,
            })
            .collect();
        let metadata: Vec<_> = metadata
            .iter()
            .map(|(key, value)| (key.as_slice(), value))
            .collect();
        let mut bytes = Vec::new();
        let layout = write_raw_header(&mut bytes, &metadata, &tensors, &Limits::default()).unwrap();
        for index in 0..count {
            for value in 0..1024 {
                let value = ((value * 73 + (index + 1) * 101) % 2053) as i32 - 1026;
                bytes.extend_from_slice(&(value as f32 / 4096.0).to_le_bytes());
            }
        }
        assert_eq!(bytes.len() as u64, layout.file_bytes);
        bytes
    }
    fn params(case: &Json) -> QuantizeParams {
        let value = field(case, "params");
        let mut params = QuantizeParams {
            ftype: FileType::from_id(integer(field(value, "ftype")) as u32).unwrap(),
            nthread: integer(field(value, "nthread")) as i32,
            only_copy: field(value, "only_copy").as_bool().unwrap(),
            ..QuantizeParams::default()
        };
        if let Some(overrides) = value.get("kv_overrides") {
            params.kv_overrides = array(overrides)
                .iter()
                .map(|item| MetadataOverride {
                    key: text(field(item, "key")).as_bytes().to_vec(),
                    value: match text(field(item, "type")) {
                        "int" => OverrideValue::Integer(integer(field(item, "value"))),
                        "float" => OverrideValue::Float(field(item, "value").as_f64().unwrap()),
                        "bool" => OverrideValue::Bool(field(item, "value").as_bool().unwrap()),
                        "str" => {
                            OverrideValue::String(text(field(item, "value")).as_bytes().to_vec())
                        }
                        _ => panic!("fixture override type"),
                    },
                })
                .collect();
        }
        params
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "llama-quant-metadata-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn replay(corpus: &str, count: usize, success_count: usize, unsupported_count: usize) {
        let cases = document(&format!("{corpus}-cases.json"));
        let profile = if cfg!(target_os = "windows") {
            "msvc"
        } else {
            "gnu"
        };
        let observations = document(&format!("{corpus}-{profile}.json"));
        assert_eq!(array(&cases).len(), count);
        assert_eq!(array(&observations).len(), count);
        let dir = Directory::new();
        let mut failures = Vec::new();
        let (mut accepted, mut rejected, mut unsupported) = (0, 0, 0);
        for (case, reference) in array(&cases).iter().zip(array(&observations)) {
            let id = text(field(case, "id"));
            assert_eq!(id, text(field(reference, "id")));
            let bytes = input(case);
            let expected_input = field(reference, "input");
            assert_eq!(
                bytes.len() as i64,
                integer(field(expected_input, "bytes")),
                "{id}"
            );
            assert_eq!(
                llama_conformance::digest::sha256(&bytes),
                text(field(expected_input, "sha256")),
                "Rust fixture construction must reproduce independently captured input bytes: {id}"
            );
            let input = dir.0.join(format!("{id}.input.gguf"));
            let output = dir.0.join(format!("{id}.output.gguf"));
            fs::write(&input, bytes).unwrap();
            let result = quantize_file(&input, &output, &params(case));
            if text(field(reference, "scope")) == "unsupported-family" {
                unsupported += 1;
                if !matches!(result, Err(Error::Unsupported(_))) || output.exists() {
                    failures.push(format!(
                        "{id}: expected explicit file-scope rejection; {result:?}"
                    ));
                }
                continue;
            }
            if text(field(reference, "native_outcome")) != "accepted" {
                rejected += 1;
                if result.is_ok() || output.exists() {
                    failures.push(format!(
                        "{id}: expected checked native rejection; {result:?}"
                    ));
                }
                continue;
            }
            let report = match result {
                Ok(report) => report,
                Err(error) => {
                    failures.push(format!(
                        "{id}: original accepted but Rust rejected: {error}"
                    ));
                    continue;
                }
            };
            accepted += 1;
            let expected_output = field(reference, "output");
            let data = fs::read(&output).unwrap();
            let hash = llama_conformance::digest::sha256(&data);
            if report.outputs.len() != 1
                || !report.written
                || data.len() as i64 != integer(field(expected_output, "bytes"))
                || hash != text(field(expected_output, "sha256"))
            {
                failures.push(format!(
                    "{id}: complete output differs from original: {} bytes, {hash}",
                    data.len()
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} mismatches:\n{}",
            failures.len(),
            failures.join("\n")
        );
        assert_eq!(accepted, success_count);
        assert_eq!(unsupported, unsupported_count);
        assert_eq!(accepted + rejected + unsupported, count);
        println!(
            "metadata {corpus}/{profile}: {accepted} exact complete files, {rejected} safe native rejections, {unsupported} explicit family exclusions"
        );
    }
    #[test]
    fn independent_frozen_metadata_admission_and_complete_copies() {
        replay("admission", 346, 140, 4);
    }
    #[test]
    fn independent_nextn_changes_real_file_precision_classification() {
        replay("classification", 10, 10, 0);
    }
}
