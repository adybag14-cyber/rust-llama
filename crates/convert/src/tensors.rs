//! Ordered source contributions, dense transforms and bounded output buffers.
use crate::codec::{self, SourceProfile, WorkingType};
use crate::config::Architecture;
use crate::config::Config;
use crate::frozen::TensorName;
use crate::{Error, Limits, Options, OutputType, Result, reserve, string};
use llama_checkpoint::{Checkpoint, DType as SourceType, TypedLayout};
use llama_quant::DType;
use std::borrow::Cow;
use std::io::Read;

#[derive(Debug)]
struct Contribution {
    name: String,
    file: usize,
    tensor: usize,
    order: usize,
}
#[derive(Debug)]
pub(crate) struct Component {
    file: usize,
    tensor: usize,
    source: SourceType,
    input_bytes: usize,
    pub shape: Vec<u64>,
    heads: Option<usize>,
    transform: crate::gpt::Transform,
}
#[derive(Debug)]
pub(crate) struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub encoded_bytes: usize,
    work: WorkingType,
    parts: Vec<Component>,
}
#[derive(Debug)]
pub(crate) struct Plan {
    pub tensors: Vec<Tensor>,
    pub output_type: OutputType,
    pub parameters: u64,
}
impl Plan {
    pub fn new(
        checkpoint: &Checkpoint,
        config: &Config,
        options: &Options,
        limits: &Limits,
    ) -> Result<Self> {
        let mut values = contributions(checkpoint, config.family, limits)?;
        let output_type = if options.output_type == OutputType::Auto {
            let mut guessed = OutputType::F16;
            for value in &values {
                let layout = layout(checkpoint, value)?;
                if layout.shape.len() < 2 {
                    continue;
                }
                match layout.dtype {
                    SourceType::Bf16 => {
                        guessed = OutputType::Bf16;
                        break;
                    }
                    SourceType::F16 => {
                        guessed = OutputType::F16;
                        break;
                    }
                    _ => {}
                }
            }
            guessed
        } else {
            options.output_type
        };
        if options.vocab_only {
            return Ok(Self {
                tensors: Vec::new(),
                output_type,
                parameters: 0,
            });
        }
        let mut fusion = Vec::new();
        reserve(&mut fusion, config.layers, "fusion layer descriptors")?;
        fusion.resize_with(config.layers, Fusion::default);
        if options.fuse_qkv {
            for value in &values {
                let Some(mapped) =
                    map_name(&value.name, config.layers, config.family.tensor_names())?
                else {
                    continue;
                };
                if let Some((layer, component, bias)) = qkv_slot(&mapped.name)
                    && let Some(entry) = fusion.get_mut(layer)
                {
                    if bias {
                        entry.bias_mask |= 1 << component;
                    } else {
                        entry.weight_mask |= 1 << component;
                    }
                }
            }
        }
        let mut tensors = Vec::new();
        reserve(&mut tensors, values.len(), "output tensor descriptors")?;
        for value in values.drain(..) {
            if [
                ".attention.masked_bias",
                ".attention.bias",
                ".rotary_emb.inv_freq",
            ]
            .iter()
            .any(|suffix| value.name.ends_with(suffix))
            {
                continue;
            }
            let layout = layout(checkpoint, &value)?;
            if layout.shape.len() > 4 {
                return Err(Error::Unsupported("GGUF rank exceeds four"));
            }
            // Preserved FP8/ModelOpt is a separate conversion contract, not an F32 fallback.
            if layout.dtype == SourceType::F8E4M3 && options.output_type == OutputType::Auto {
                return Err(Error::Unsupported("automatic preserved FP8 conversion"));
            }
            codec::working_type(layout.dtype)?;
            let raw_name = if matches!(config.architecture.as_str(), "LlamaModel" | "Qwen2Model") {
                let mut name = string("model.")?;
                name.try_reserve(value.name.len())
                    .map_err(|_| Error::Limit("tensor name"))?;
                name.push_str(&value.name);
                name
            } else if config.family == Architecture::Bloom {
                // BloomModel removes every literal transformer. occurrence
                // only after the base contribution dictionary is built.
                // Normalizing earlier would hide canonical-name collisions.
                String::from_utf8(replace_bytes(value.name.as_bytes(), b"transformer.", b"")?)
                    .map_err(|_| Error::Invalid("BLOOM tensor name is not Unicode"))?
            } else {
                string(&value.name)?
            };
            let mapped = map_name(&raw_name, config.layers, config.family.tensor_names())?.ok_or(
                Error::Invalid("source tensor has no mapping for the frozen architecture"),
            )?;
            if mapped.kind.contains("EXP")
                || raw_name.ends_with(".lora_a")
                || raw_name.ends_with(".lora_b")
            {
                return Err(Error::Unsupported(
                    "expert or adapter tensor transformation",
                ));
            }
            let heads = if config.family != Architecture::Llama {
                None
            } else if raw_name.ends_with("q_proj.weight") || raw_name.ends_with("q_proj.bias") {
                Some(config.head_count()?)
            } else if raw_name.ends_with("k_proj.weight") || raw_name.ends_with("k_proj.bias") {
                Some(config.kv_head_count()?.unwrap_or(config.head_count()?))
            } else {
                None
            };
            if let Some(heads) = heads {
                let rows = *layout
                    .shape
                    .first()
                    .ok_or(Error::Invalid("cannot permute a scalar Q/K tensor"))?;
                let group = (heads as u64)
                    .checked_mul(2)
                    .ok_or(Error::Limit("head grouping"))?;
                if group == 0 || !rows.is_multiple_of(group) {
                    return Err(Error::Invalid(
                        "Q/K first dimension is not divisible by twice its head count",
                    ));
                }
            }
            let input_bytes = usize::try_from(layout.bytes)
                .map_err(|_| Error::Limit("source tensor byte size"))?;
            if input_bytes > limits.max_tensor_input_bytes {
                return Err(Error::Limit("source tensor bytes"));
            }
            let mut shape = layout.shape;
            let transform = crate::gpt::Transform::plan(config, &raw_name, &mut shape)?;
            let component = Component {
                file: value.file,
                tensor: value.tensor,
                source: layout.dtype,
                input_bytes,
                shape,
                heads,
                transform,
            };
            if options.fuse_qkv
                && let Some((layer, slot, bias)) = qkv_slot(&mapped.name)
            {
                let entry = &mut fusion[layer];
                if entry.weight_mask == 7 && (entry.bias_mask == 0 || entry.bias_mask == 7) {
                    let slots = if bias {
                        &mut entry.bias
                    } else {
                        &mut entry.weight
                    };
                    if slots[slot].replace(component).is_some() {
                        return Err(Error::Invalid("duplicate canonical QKV component"));
                    }
                    if slots.iter().all(Option::is_some) {
                        let mut parts = Vec::new();
                        reserve(&mut parts, 3, "fused component descriptors")?;
                        for slot in slots {
                            parts.push(slot.take().expect("three present components"));
                        }
                        let mut name = string("blk.")?;
                        name.push_str(&layer.to_string());
                        name.push_str(if bias {
                            ".attn_qkv.bias"
                        } else {
                            ".attn_qkv.weight"
                        });
                        tensors.push(make_tensor(name, "ATTN_QKV", parts, output_type, limits)?);
                    }
                    continue;
                }
            }
            let mut parts = Vec::new();
            reserve(&mut parts, 1, "component descriptor")?;
            parts.push(component);
            tensors.push(make_tensor(
                mapped.name,
                mapped.kind,
                parts,
                output_type,
                limits,
            )?);
        }
        if fusion
            .iter()
            .any(|v| v.weight.iter().any(Option::is_some) || v.bias.iter().any(Option::is_some))
        {
            return Err(Error::Invalid("unconsumed QKV fusion component"));
        }
        let mut names = Vec::new();
        reserve(&mut names, tensors.len(), "output-name index")?;
        for tensor in &tensors {
            names.push(tensor.name.as_str());
        }
        names.sort_unstable();
        if names.windows(2).any(|p| p[0] == p[1]) {
            return Err(Error::Invalid("duplicate canonical GGUF tensor name"));
        }
        let mut parameters = 0u64;
        for tensor in &tensors {
            let count = tensor
                .shape
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d))
                .ok_or(Error::Limit("parameter product"))?;
            parameters = parameters
                .checked_add(count)
                .ok_or(Error::Limit("parameter total"))?;
        }
        Ok(Self {
            tensors,
            output_type,
            parameters,
        })
    }
}
fn layout(checkpoint: &Checkpoint, value: &Contribution) -> Result<TypedLayout> {
    let (_, file) = checkpoint
        .file(value.file)
        .ok_or(Error::Invalid("source file index"))?;
    Ok(file
        .tensors()
        .get(value.tensor)
        .ok_or(Error::Invalid("source tensor index"))?
        .typed_layout()?)
}
fn contributions(
    checkpoint: &Checkpoint,
    family: Architecture,
    limits: &Limits,
) -> Result<Vec<Contribution>> {
    let mut all = Vec::new();
    let mut order = 0usize;
    let mut text_bytes = 0usize;
    for file_index in 0..checkpoint.file_count() {
        let (_, file) = checkpoint
            .file(file_index)
            .ok_or(Error::Invalid("checkpoint owner index"))?;
        for (tensor_index, tensor) in file.tensors().iter().enumerate() {
            let Some(name) = filter_name(
                tensor.name().as_wtf8(),
                family,
                limits.max_tensor_name_bytes,
            )?
            else {
                continue;
            };
            text_bytes = text_bytes
                .checked_add(name.len())
                .filter(|n| *n <= limits.checkpoint.max_text_bytes)
                .ok_or(Error::Limit("filtered tensor-name bytes"))?;
            if all.len() >= limits.checkpoint.max_tensors {
                return Err(Error::Limit("source contributions"));
            }
            reserve(&mut all, 1, "source contribution descriptors")?;
            all.push(Contribution {
                name,
                file: file_index,
                tensor: tensor_index,
                order,
            });
            order = order.checked_add(1).ok_or(Error::Limit("source order"))?;
        }
    }
    // Python dict overwrites the value while keeping the first key's insertion order.
    all.sort_unstable_by(|a, b| a.name.cmp(&b.name).then(a.order.cmp(&b.order)));
    let mut unique: Vec<Contribution> = Vec::new();
    reserve(&mut unique, all.len(), "unique contributions")?;
    for value in all {
        if let Some(last) = unique.last_mut().filter(|last| last.name == value.name) {
            last.file = value.file;
            last.tensor = value.tensor;
        } else {
            unique.push(value);
        }
    }
    unique.sort_unstable_by_key(|v| v.order);
    Ok(unique)
}
fn replace_bytes(input: &[u8], from: &[u8], to: &[u8]) -> Result<Vec<u8>> {
    let count = input.windows(from.len()).filter(|v| *v == from).count();
    let size = input
        .len()
        .checked_sub(count * from.len())
        .and_then(|n| n.checked_add(count.checked_mul(to.len())?))
        .ok_or(Error::Limit("renamed tensor length"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| Error::Limit("tensor name allocation"))?;
    let mut i = 0;
    while i < input.len() {
        if input[i..].starts_with(from) {
            output.extend_from_slice(to);
            i += from.len();
        } else {
            output.push(input[i]);
            i += 1;
        }
    }
    Ok(output)
}
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|v| v == needle)
}
fn filter_name(raw: &[u8], family: Architecture, max: usize) -> Result<Option<String>> {
    if raw.len() > max {
        return Err(Error::Limit("source tensor name"));
    }
    // Only the frozen Llama subclass strips this wrapper, before TextModel's
    // multimodal filter. Qwen, GPT and BLOOM inherit the base name unchanged.
    let first: Cow<'_, [u8]> = if family == Architecture::Llama {
        Cow::Owned(replace_bytes(raw, b"text_model.", b"")?)
    } else {
        Cow::Borrowed(raw)
    };
    let check = first.strip_prefix(b"model.").unwrap_or(&first);
    if [
        b"mlp".as_slice(),
        b"vit.",
        b"vpm.",
        b"siglip2.",
        b"conformer.",
        b"connector.",
        b"merger.",
        b"resampler.",
        b"sound_encoder.",
        b"sound_projection.",
        b"speech_embeddings.",
    ]
    .iter()
    .any(|prefix| check.starts_with(prefix))
        || [
            b"visual.".as_slice(),
            b"vision.",
            b"audio.",
            b"talker.",
            b"vision_",
            b"audio_",
            b"token2wav.",
            b"code2wav.",
            b"projector.",
            b"pre_mm_projector_norm",
            b"image_newline",
            b"view_seperator",
            b"patch_embed",
            b"patch_embedding",
            b"patch_merger.",
            b"patch_merge_mlp.",
        ]
        .iter()
        .any(|part| contains(check, part))
    {
        return Ok(None);
    }
    // ModelBase tests this suffix before it removes the language_model wrapper.
    let second = if first.ends_with(b"e_score_correction_bias") {
        Cow::Owned(replace_bytes(
            &first,
            b"e_score_correction_bias",
            b"e_score_correction.bias",
        )?)
    } else {
        first
    };
    let third = replace_bytes(&second, b"language_model.", b"")?;
    Ok(Some(String::from_utf8(third).map_err(|_| {
        Error::Invalid("admitted tensor name is not Unicode scalar text")
    })?))
}
struct Mapped {
    name: String,
    kind: &'static str,
}
fn map_name(name: &str, layers: usize, table: &'static [TensorName]) -> Result<Option<Mapped>> {
    for suffix in ["", ".weight", ".bias"] {
        let Some(base) = (if suffix.is_empty() {
            Some(name)
        } else {
            name.strip_suffix(suffix)
        }) else {
            continue;
        };
        // The original map is assigned in declaration order; a later alias wins.
        for entry in table.iter().rev() {
            let number = if entry.block {
                if layers == 0 {
                    continue;
                }
                if let Some((before, after)) = entry.key.split_once("{bid}") {
                    let Some(value) = base
                        .strip_prefix(before)
                        .and_then(|s| s.strip_suffix(after))
                    else {
                        continue;
                    };
                    if value.is_empty()
                        || !value.bytes().all(|b| b.is_ascii_digit())
                        || (value.len() > 1 && value.starts_with('0'))
                    {
                        continue;
                    }
                    let Ok(number) = value.parse::<usize>() else {
                        continue;
                    };
                    if number >= layers {
                        continue;
                    }
                    Some(number)
                } else if base == entry.key {
                    Some(layers - 1)
                } else {
                    continue;
                }
            } else if base == entry.key {
                None
            } else {
                continue;
            };
            let mut result = if let Some((before, after)) = entry.canonical.split_once("{bid}") {
                let mut result = string(before)?;
                result.push_str(&number.expect("block mapping").to_string());
                result.push_str(after);
                result
            } else {
                string(entry.canonical)?
            };
            result.push_str(suffix);
            return Ok(Some(Mapped {
                name: result,
                kind: entry.kind,
            }));
        }
    }
    Ok(None)
}
fn qkv_slot(name: &str) -> Option<(usize, usize, bool)> {
    let rest = name.strip_prefix("blk.")?;
    let (index, rest) = rest.split_once('.')?;
    let index = index.parse().ok()?;
    let (projection, suffix) = rest.rsplit_once('.')?;
    let slot = match projection {
        "attn_q" => 0,
        "attn_k" => 1,
        "attn_v" => 2,
        _ => return None,
    };
    let bias = match suffix {
        "weight" => false,
        "bias" => true,
        _ => return None,
    };
    Some((index, slot, bias))
}
#[derive(Default)]
struct Fusion {
    weight_mask: u8,
    bias_mask: u8,
    weight: [Option<Component>; 3],
    bias: [Option<Component>; 3],
}
fn make_tensor(
    name: String,
    kind: &str,
    parts: Vec<Component>,
    output: OutputType,
    limits: &Limits,
) -> Result<Tensor> {
    let first = parts.first().ok_or(Error::Invalid("empty tensor plan"))?;
    let mut shape = Vec::new();
    reserve(&mut shape, first.shape.len(), "output shape")?;
    shape.extend_from_slice(&first.shape);
    if parts.len() > 1 {
        if shape.is_empty() {
            return Err(Error::Invalid("cannot concatenate scalar QKV"));
        }
        for part in &parts[1..] {
            if part.shape.len() != shape.len() || part.shape[1..] != shape[1..] {
                return Err(Error::Invalid(
                    "QKV shapes differ outside concatenation axis",
                ));
            }
            shape[0] = shape[0]
                .checked_add(part.shape[0])
                .ok_or(Error::Limit("fused dimension"))?;
        }
    }
    let work = if parts.iter().all(|p| p.source == SourceType::F16) {
        WorkingType::F16
    } else {
        WorkingType::F32
    };
    let count = shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or(Error::Limit("output element count"))?;
    let input_len = usize::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(work.bytes()))
        .ok_or(Error::Limit("working byte size"))?;
    let mut dtype = match output {
        OutputType::F32 => DType::F32,
        OutputType::F16 => DType::F16,
        OutputType::Bf16 => DType::BF16,
        OutputType::Q8_0 => DType::Q8_0,
        OutputType::Tq1_0 => DType::TQ1_0,
        OutputType::Tq2_0 => DType::TQ2_0,
        OutputType::Auto => return Err(Error::Invalid("unresolved output type")),
    };
    if shape.len() <= 1
        || name.ends_with("_norm.weight")
        || !name.ends_with(".weight")
        || matches!(
            kind,
            "FFN_GATE_INP" | "FFN_GATE_INP_SHEXP" | "POS_EMBD" | "TOKEN_TYPES"
        )
    {
        dtype = DType::F32;
    } else if matches!(dtype, DType::TQ1_0 | DType::TQ2_0)
        && matches!(kind, "TOKEN_EMBD" | "OUTPUT")
    {
        dtype = DType::F16;
    }
    let width = usize::try_from(shape.last().copied().unwrap_or(1))
        .map_err(|_| Error::Limit("row width"))?;
    let encoded_bytes = match codec::encoded_len(work, input_len, width, dtype) {
        Ok(n) => n,
        Err(codec::CodecError::RowWidth { .. }) => {
            dtype = DType::F16;
            codec::encoded_len(work, input_len, width, dtype)?
        }
        Err(e) => return Err(e.into()),
    };
    if encoded_bytes as u64 > limits.max_output_bytes {
        return Err(Error::Limit("tensor output bytes"));
    }
    Ok(Tensor {
        name,
        shape,
        dtype,
        encoded_bytes,
        work,
        parts,
    })
}
struct WorkBudget {
    allocated: usize,
    max: usize,
}
impl WorkBudget {
    fn buffer(&mut self, length: usize) -> Result<Vec<u8>> {
        self.allocated
            .checked_add(length)
            .filter(|n| *n <= self.max)
            .ok_or(Error::Limit("per-output tensor working allocations"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(length)
            .map_err(|_| Error::Limit("tensor working allocation"))?;
        self.allocated = self
            .allocated
            .checked_add(data.capacity())
            .filter(|n| *n <= self.max)
            .ok_or(Error::Limit("actual tensor working allocation capacity"))?;
        data.resize(length, 0);
        Ok(data)
    }
}
impl Tensor {
    pub fn encode(
        &self,
        checkpoint: &Checkpoint,
        profile: SourceProfile,
        limits: &Limits,
    ) -> Result<Vec<u8>> {
        let mut budget = WorkBudget {
            allocated: 0,
            max: limits.max_working_bytes,
        };
        let count = self
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or(Error::Limit("tensor elements"))?;
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(self.work.bytes()))
            .ok_or(Error::Limit("working bytes"))?;
        let mut joined = budget.buffer(bytes)?;
        let mut cursor = 0;
        for part in &self.parts {
            let (_, file) = checkpoint
                .file(part.file)
                .ok_or(Error::Invalid("file owner"))?;
            let mut reader = file.reader(part.tensor)?;
            if reader.len() != part.input_bytes as u64 {
                return Err(Error::Invalid("typed source range changed"));
            }
            let mut raw = budget.buffer(part.input_bytes)?;
            reader.read_exact(&mut raw)?;
            let (kind, prepared_bytes) = codec::prepared_len(part.source, raw.len(), false)?;
            let mut prepared = budget.buffer(prepared_bytes)?;
            codec::prepare_into_with_profile(part.source, &raw, false, profile, &mut prepared)?;
            drop(raw);
            if !part.transform.is_identity() {
                let mut transformed = budget.buffer(prepared.len())?;
                part.transform
                    .apply(&prepared, &mut transformed, kind.bytes())?;
                prepared = transformed;
            }
            if let Some(heads) = part.heads {
                let rows =
                    usize::try_from(part.shape[0]).map_err(|_| Error::Limit("permutation rows"))?;
                let mut permuted = budget.buffer(prepared.len())?;
                if let Some(row_bytes) = prepared.len().checked_div(rows) {
                    let width = rows / heads;
                    for head in 0..heads {
                        for j in 0..width / 2 {
                            for half in 0..2 {
                                let from = (head * width + half * (width / 2) + j) * row_bytes;
                                let to = (head * width + 2 * j + half) * row_bytes;
                                permuted[to..to + row_bytes]
                                    .copy_from_slice(&prepared[from..from + row_bytes]);
                            }
                        }
                    }
                }
                prepared = permuted;
            }
            let remaining = &mut joined[cursor..];
            if kind == self.work {
                remaining[..prepared.len()].copy_from_slice(&prepared);
                cursor += prepared.len();
            } else {
                let (_, promoted_len) = codec::prepared_len(SourceType::F16, prepared.len(), true)?;
                codec::prepare_into_with_profile(
                    SourceType::F16,
                    &prepared,
                    true,
                    profile,
                    &mut remaining[..promoted_len],
                )?;
                cursor += promoted_len;
            }
        }
        if cursor != joined.len() {
            return Err(Error::Invalid("planned concatenation length"));
        }
        let mut encoded = budget.buffer(self.encoded_bytes)?;
        let width = usize::try_from(self.shape.last().copied().unwrap_or(1))
            .map_err(|_| Error::Limit("row width"))?;
        codec::encode_into_with_profile(
            self.work,
            &joined,
            width,
            self.dtype,
            profile,
            &mut encoded,
        )?;
        Ok(encoded)
    }
}

#[cfg(test)]
mod bloom_alias_reference {
    use super::*;
    #[test]
    fn frozen_bloom_alias_lookup() {
        let raw = llama_checkpoint::json::parse(
            include_bytes!("../tests/fixtures/bloom-conversion/aliases.json"),
            llama_checkpoint::Limits::default(),
        )
        .unwrap();
        let object = raw.as_object().unwrap();
        let cases = object.get("cases").unwrap().as_array().unwrap();
        assert_eq!(cases.len(), 576);
        for case in cases {
            let row = case.as_object().unwrap();
            let name = row
                .get("name")
                .unwrap()
                .as_text()
                .unwrap()
                .as_str()
                .unwrap();
            let expected = row.get("result").unwrap();
            let result = map_name(name, 2, crate::bloom_tables::BLOOM_TENSOR_NAMES).unwrap();
            let expected = if matches!(expected, llama_checkpoint::Value::Null) {
                None
            } else {
                Some(expected.as_text().unwrap().as_str().unwrap())
            };
            assert_eq!(result.as_ref().map(|v| v.name.as_str()), expected, "{name}");
        }
    }
}

#[cfg(test)]
mod family_filter_reference {
    use super::*;
    use crate::util::{integer, object, text};
    use llama_checkpoint::Value as Json;

    #[test]
    fn native_bloom_related_family_filters() {
        let root = llama_checkpoint::json::parse(
            include_bytes!("../tests/fixtures/bloom-conversion/filter-reference.json"),
            llama_checkpoint::Limits::default(),
        )
        .unwrap();
        let root = object(&root).unwrap();
        let host = if cfg!(windows) { "windows" } else { "linux" };
        let profile = object(root.get("profiles").unwrap())
            .unwrap()
            .get(host)
            .unwrap();
        let families = profile.as_array().unwrap();
        assert_eq!(families.len(), 6);
        let mut decisions = 0;
        for family in families {
            let family = object(family).unwrap();
            let name = text(family.get("family").unwrap()).unwrap();
            let kind = match name {
                "llama" => Architecture::Llama,
                "qwen2" => Architecture::Qwen2,
                "qwen3" => Architecture::Qwen3,
                "gpt2" => Architecture::Gpt2,
                "gptneox" => Architecture::GptNeoX,
                "bloom" => Architecture::Bloom,
                _ => panic!("unrecognized native family {name}"),
            };
            for case in family.get("cases").unwrap().as_array().unwrap() {
                let case = object(case).unwrap();
                let input = text(case.get("name").unwrap()).unwrap();
                let expected = case.get("result").unwrap();
                let expected = match expected {
                    Json::Null => None,
                    value => Some(text(value).unwrap()),
                };
                assert_eq!(
                    filter_name(
                        input.as_bytes(),
                        kind,
                        Limits::default().max_tensor_name_bytes
                    )
                    .unwrap()
                    .as_deref(),
                    expected,
                    "{name}: {input:?}"
                );
                decisions += 1;
            }
        }
        assert_eq!(
            decisions,
            integer(root.get("decisions_per_host").unwrap()).unwrap()
        );
        assert!(decisions > 1000);
    }
}

#[cfg(test)]
mod falcon_alias_reference {
    use super::map_name;
    use crate::util::{JsonBudget, integer, object, text};
    use llama_checkpoint::Value as Json;

    #[test]
    fn frozen_falcon_alias_lookup() {
        let root = JsonBudget::new(llama_checkpoint::Limits::default())
            .parse(include_bytes!(
                "../tests/fixtures/falcon-conversion/aliases.json"
            ))
            .unwrap();
        let root = object(&root).unwrap();
        let layers = integer(root.get("layers").unwrap()).unwrap() as usize;
        let cases = root.get("cases").unwrap().as_array().unwrap();
        for case in cases {
            let row = object(case).unwrap();
            let name = text(row.get("name").unwrap()).unwrap();
            let actual = map_name(name, layers, crate::falcon_tables::FALCON_TENSOR_NAMES).unwrap();
            match row.get("result").unwrap() {
                Json::Null => assert!(actual.is_none(), "{name}"),
                expected => assert_eq!(actual.unwrap().name, text(expected).unwrap(), "{name}"),
            }
        }
        assert_eq!(cases.len(), 510);
    }
}

#[cfg(test)]
mod starcoder_alias_reference {
    use super::map_name;
    use crate::util::{JsonBudget, integer, object, text};
    use llama_checkpoint::Value as Json;

    #[test]
    fn frozen_starcoder_alias_lookup() {
        let root = JsonBudget::new(llama_checkpoint::Limits::default())
            .parse(include_bytes!(
                "../tests/fixtures/starcoder-conversion/aliases.json"
            ))
            .unwrap();
        let root = object(&root).unwrap();
        let mut total = 0;
        for family in root.get("families").unwrap().as_array().unwrap() {
            let family = object(family).unwrap();
            let tables = match text(family.get("architecture").unwrap()).unwrap() {
                "starcoder" => crate::starcoder_tables::STARCODER_TENSOR_NAMES,
                "starcoder2" => crate::starcoder_tables::STARCODER2_TENSOR_NAMES,
                _ => panic!("unknown independent fixture architecture"),
            };
            let layers = integer(family.get("layers").unwrap()).unwrap() as usize;
            for case in family.get("cases").unwrap().as_array().unwrap() {
                let row = object(case).unwrap();
                let name = text(row.get("name").unwrap()).unwrap();
                let actual = map_name(name, layers, tables).unwrap();
                match row.get("result").unwrap() {
                    Json::Null => assert!(actual.is_none(), "{name}"),
                    expected => assert_eq!(actual.unwrap().name, text(expected).unwrap(), "{name}"),
                }
                total += 1;
            }
        }
        assert_eq!(total, 1_248);
    }
}
