//! Reproduce two pinned pretrained test inputs from official SafeTensors.
//!
//! This is fixture preparation, not a general HF-to-GGUF converter. Metadata
//! and tensor descriptors are independent original-converter capture bytes.
//! Every input, transformed tensor and complete staged output is hash-checked.
use crate::Result;
use llama_checkpoint::{DType, Limits, SafetensorsFile, Value};
use llama_conformance::digest::{self, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_RECIPE: usize = 256 * 1024;
const MAX_HEADER: usize = 2 * 1024 * 1024;
const MAX_SOURCE: u64 = 600 * 1024 * 1024;
const MAX_OUTPUT: u64 = 300 * 1024 * 1024;
const MAX_RANDOM_TENSOR: usize = 16 * 1024 * 1024;
const MAX_TENSORS: usize = 256;
const CHUNK_ELEMENTS: usize = 8192;
const BUFFER_BYTES: usize = CHUNK_ELEMENTS * 4;

struct Embedded {
    name: &'static str,
    recipe: &'static [u8],
    recipe_sha256: &'static str,
    header: &'static [u8],
    ignored: usize,
}
const GPT2: Embedded = Embedded {
    name: "gpt2-124m-f16",
    recipe: include_bytes!("../../fixtures/pretrained-preparation/gpt2-124m-f16-recipe.json"),
    recipe_sha256: "0ef173d3ac511f2a105c189ba64097a7b752dc51801fcad6c23fcedcca6288fc",
    header: include_bytes!("../../fixtures/pretrained-preparation/gpt2-124m-f16-header.bin"),
    ignored: 12,
};
const PYTHIA: Embedded = Embedded {
    name: "pythia-14m-f16",
    recipe: include_bytes!("../../fixtures/pretrained-preparation/pythia-14m-f16-recipe.json"),
    recipe_sha256: "c26683ae1e5c0e73f4c043177a1820a232d28b1ac735ca5f62fae73432e609a1",
    header: include_bytes!("../../fixtures/pretrained-preparation/pythia-14m-f16-header.bin"),
    ignored: 0,
};
fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .as_object()
        .and_then(|v| v.get(key))
        .ok_or_else(|| format!("fixture field {key} is absent").into())
}
fn text(value: &Value) -> Result<&str> {
    value
        .as_text()
        .and_then(|v| v.as_str())
        .ok_or_else(|| "fixture string is not Unicode".into())
}
fn integer(value: &Value) -> Result<u64> {
    match value {
        Value::Number(value) => value
            .as_integer()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "fixture integer is out of range".into()),
        _ => Err("fixture integer is not an integer".into()),
    }
}
fn size(value: &Value) -> Result<usize> {
    Ok(usize::try_from(integer(value)?)?)
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .ok_or_else(|| "fixture array is not an array".into())
}
fn sha(value: &Value) -> Result<&str> {
    let value = text(value)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("fixture SHA-256 is malformed".into());
    }
    Ok(value)
}
fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<()> {
    values
        .try_reserve_exact(count)
        .map_err(|_| "fixture allocation failed".into())
}
fn dimensions(value: &Value) -> Result<Vec<u64>> {
    let values = array(value)?;
    if values.is_empty() || values.len() > 4 {
        return Err("fixture tensor rank is outside 1..=4".into());
    }
    let mut result = Vec::new();
    reserve(&mut result, values.len())?;
    for value in values {
        let value = integer(value)?;
        if value == 0 || value > MAX_SOURCE {
            return Err("fixture tensor dimension is out of bounds".into());
        }
        result.push(value);
    }
    Ok(result)
}
fn product(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1u64, |n, v| {
        n.checked_mul(*v)
            .ok_or_else(|| "fixture shape overflow".into())
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    F16,
    F32,
}
impl Scalar {
    fn parse(value: &Value) -> Result<Self> {
        match text(value)? {
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            _ => Err("fixture dtype must be F16 or F32".into()),
        }
    }
    fn bytes(self) -> usize {
        if self == Self::F32 { 4 } else { 2 }
    }
    fn checkpoint(self) -> DType {
        if self == Self::F32 {
            DType::F32
        } else {
            DType::F16
        }
    }
}
#[derive(Clone, Copy)]
struct Expected<'a> {
    bytes: u64,
    sha256: &'a str,
}
impl<'a> Expected<'a> {
    fn parse(value: &'a Value, maximum: u64) -> Result<Self> {
        let result = Self {
            bytes: integer(field(value, "bytes")?)?,
            sha256: sha(field(value, "sha256")?)?,
        };
        if result.bytes == 0 || result.bytes > maximum {
            return Err("fixture file size is out of bounds".into());
        }
        Ok(result)
    }
}
struct SourceTensor<'a> {
    name: &'a str,
    shape: Vec<u64>,
    dtype: Scalar,
    start: u64,
    end: u64,
    sha256: &'a str,
}
impl<'a> SourceTensor<'a> {
    fn parse(value: &'a Value) -> Result<Self> {
        let offsets = array(field(value, "source_offsets")?)?;
        if offsets.len() != 2 {
            return Err("fixture source range needs two offsets".into());
        }
        let result = Self {
            name: text(field(value, "source_name")?)?,
            shape: dimensions(field(value, "source_shape")?)?,
            dtype: Scalar::parse(field(value, "source_dtype")?)?,
            start: integer(&offsets[0])?,
            end: integer(&offsets[1])?,
            sha256: sha(field(value, "source_sha256")?)?,
        };
        if result.name.is_empty() || result.name.len() > 256 {
            return Err("fixture source tensor name is out of bounds".into());
        }
        let bytes = product(&result.shape)?
            .checked_mul(result.dtype.bytes() as u64)
            .ok_or("fixture source byte count overflow")?;
        if result.end.checked_sub(result.start) != Some(bytes) || result.end > MAX_SOURCE {
            return Err("fixture source tensor range disagrees with shape".into());
        }
        Ok(result)
    }
    fn bind(&self, checkpoint: &SafetensorsFile) -> Result<usize> {
        let index = checkpoint
            .find_str(self.name)
            .ok_or_else(|| format!("missing source tensor {}", self.name))?;
        let info = &checkpoint.tensors()[index];
        let typed = info.typed_layout()?;
        let offset = info.absolute_offset().as_integer().and_then(|v| v.as_u64());
        if typed.shape != self.shape
            || typed.dtype != self.dtype.checkpoint()
            || typed.bytes != self.end - self.start
            || offset != checkpoint.data_offset().checked_add(self.start)
        {
            return Err(format!("source tensor descriptor differs: {}", self.name).into());
        }
        // Validate the complete recorded range before any output allocation.
        checkpoint.reader(index)?;
        Ok(index)
    }
}
#[derive(Clone, Copy)]
enum Transform {
    Identity,
    Transpose,
    NeoX {
        heads: usize,
        head_dim: usize,
        hidden: usize,
    },
}
struct Tensor<'a> {
    source: SourceTensor<'a>,
    name: &'a str,
    shape: Vec<u64>,
    dtype: Scalar,
    offset: u64,
    bytes: u64,
    sha256: &'a str,
    transform: Transform,
}
impl<'a> Tensor<'a> {
    fn parse(value: &'a Value) -> Result<Self> {
        let parameters = field(value, "parameters")?;
        let transform = match text(field(value, "transform")?)? {
            "identity" => Transform::Identity,
            "transpose_2d" => Transform::Transpose,
            "neox_qkv_head_major_to_qkv_major" => Transform::NeoX {
                heads: size(field(parameters, "n_head")?)?,
                head_dim: size(field(parameters, "head_dim")?)?,
                hidden: size(field(parameters, "hidden_size")?)?,
            },
            _ => return Err("unsupported pinned fixture transform".into()),
        };
        let result = Self {
            source: SourceTensor::parse(value)?,
            name: text(field(value, "target_name")?)?,
            shape: dimensions(field(value, "target_shape")?)?,
            dtype: Scalar::parse(field(value, "target_dtype")?)?,
            offset: integer(field(value, "target_offset")?)?,
            bytes: integer(field(value, "target_bytes")?)?,
            sha256: sha(field(value, "target_sha256")?)?,
            transform,
        };
        if result.name.is_empty() || result.name.len() > 256 || result.shape.len() > 2 {
            return Err("fixture output descriptor is out of bounds".into());
        }
        let elements = product(&result.source.shape)?;
        if elements != product(&result.shape)?
            || elements.checked_mul(result.dtype.bytes() as u64) != Some(result.bytes)
        {
            return Err("fixture output shape/size differs".into());
        }
        match transform {
            Transform::Identity => {
                if !result.source.shape.iter().rev().eq(result.shape.iter()) {
                    return Err("identity fixture shape differs".into());
                }
            }
            Transform::Transpose => {
                if result.source.shape.len() != 2 || result.shape != result.source.shape {
                    return Err("transpose fixture shape differs".into());
                }
            }
            Transform::NeoX {
                heads,
                head_dim,
                hidden,
            } => {
                if heads == 0
                    || head_dim == 0
                    || heads.checked_mul(head_dim) != Some(hidden)
                    || result.source.shape.len() > 2
                    || result.source.shape[0]
                        != (hidden as u64).checked_mul(3).ok_or("QKV rows overflow")?
                    || (result.source.shape.len() == 2 && result.source.shape[1] != hidden as u64)
                    || !result.source.shape.iter().rev().eq(result.shape.iter())
                {
                    return Err("NeoX fixture head geometry differs".into());
                }
            }
        }
        if !matches!(transform, Transform::Identity)
            && result.source.end - result.source.start > MAX_RANDOM_TENSOR as u64
        {
            return Err("fixture random-access tensor exceeds memory bound".into());
        }
        Ok(result)
    }
    fn source_index(&self, index: usize) -> Result<usize> {
        match self.transform {
            Transform::Identity => Ok(index),
            Transform::Transpose => {
                let rows = usize::try_from(self.source.shape[0])?;
                let cols = usize::try_from(self.source.shape[1])?;
                (index % rows)
                    .checked_mul(cols)
                    .and_then(|v| v.checked_add(index / rows))
                    .ok_or_else(|| "transpose index overflow".into())
            }
            Transform::NeoX {
                heads,
                head_dim,
                hidden,
            } => {
                let columns = if self.source.shape.len() == 2 {
                    hidden
                } else {
                    1
                };
                let row = index / columns;
                let part = row / hidden;
                let head = (row % hidden) / head_dim;
                let position = row % head_dim;
                if part >= 3 || head >= heads {
                    return Err("QKV source index outside heads".into());
                }
                head.checked_mul(3)
                    .and_then(|v| v.checked_add(part))
                    .and_then(|v| v.checked_mul(head_dim))
                    .and_then(|v| v.checked_add(position))
                    .and_then(|v| v.checked_mul(columns))
                    .and_then(|v| v.checked_add(index % columns))
                    .ok_or_else(|| "QKV index overflow".into())
            }
        }
    }
}
struct Plan<'a> {
    source: Expected<'a>,
    source_data: u64,
    source_count: usize,
    output: Expected<'a>,
    header: &'static [u8],
    tensors: Vec<Tensor<'a>>,
    ignored: Vec<SourceTensor<'a>>,
}
impl<'a> Plan<'a> {
    fn parse(value: &'a Value, embedded: &Embedded) -> Result<Self> {
        if integer(field(value, "schema")?)? != 1
            || text(field(value, "fixture")?)? != embedded.name
        {
            return Err("pinned fixture recipe identity differs".into());
        }
        let source = field(value, "source")?;
        let output = field(value, "output")?;
        let header = Expected::parse(field(value, "header")?, MAX_HEADER as u64)?;
        if embedded.header.len() < 24
            || header.bytes != embedded.header.len() as u64
            || digest::sha256(embedded.header) != header.sha256
            || &embedded.header[..4] != b"GGUF"
            || u32::from_le_bytes(embedded.header[4..8].try_into()?) != 3
            || integer(field(output, "alignment")?)? != 32
            || header.bytes % 32 != 0
        {
            return Err("captured fixture GGUF header differs".into());
        }
        let rows = array(field(value, "tensors")?)?;
        let ignored = array(field(value, "ignored")?)?;
        let source_count = size(field(source, "tensor_count")?)?;
        if rows.is_empty()
            || rows.len() + ignored.len() != source_count
            || source_count > MAX_TENSORS
            || ignored.len() != embedded.ignored
            || size(field(output, "tensor_count")?)? != rows.len()
            || u64::from_le_bytes(embedded.header[8..16].try_into()?) != rows.len() as u64
        {
            return Err("fixture tensor inventory differs".into());
        }
        let mut tensors = Vec::new();
        reserve(&mut tensors, rows.len())?;
        let mut position = 0u64;
        for row in rows {
            let tensor = Tensor::parse(row)?;
            let padded = position
                .checked_add(31)
                .ok_or("fixture alignment overflow")?
                & !31;
            if tensor.offset != padded || tensors.iter().any(|v: &Tensor<'_>| v.name == tensor.name)
            {
                return Err("fixture tensor layout is not contiguous/aligned/unique".into());
            }
            position = tensor
                .offset
                .checked_add(tensor.bytes)
                .ok_or("fixture payload size overflow")?;
            tensors.push(tensor);
        }
        let mut skipped = Vec::new();
        reserve(&mut skipped, ignored.len())?;
        for row in ignored {
            let source = SourceTensor::parse(row)?;
            let layer = source
                .name
                .strip_prefix("h.")
                .and_then(|v| v.strip_suffix(".attn.bias"))
                .and_then(|v| v.parse::<usize>().ok());
            if embedded.name != GPT2.name
                || layer.is_none_or(|n| n >= 12)
                || source.shape != [1, 1, 1024, 1024]
                || source.dtype != Scalar::F32
                || source.sha256
                    != "e16420835229addd98c66a061dd8a9bea9b405eff3738f4bb00ff603f98400d1"
            {
                return Err(
                    "only the twelve qualified GPT2 static causal masks may be omitted".into(),
                );
            }
            skipped.push(source);
        }
        let output = Expected::parse(output, MAX_OUTPUT)?;
        if header.bytes.checked_add(position) != Some(output.bytes) {
            return Err("fixture final file length differs from header and tensors".into());
        }
        Ok(Self {
            source: Expected::parse(source, MAX_SOURCE)?,
            source_data: integer(field(source, "data_start")?)?,
            source_count,
            output,
            header: embedded.header,
            tensors,
            ignored: skipped,
        })
    }
    fn bind(&self, checkpoint: &SafetensorsFile) -> Result<Vec<usize>> {
        if checkpoint.file_len() != self.source.bytes
            || checkpoint.data_offset() != self.source_data
            || checkpoint.tensors().len() != self.source_count
        {
            return Err("official SafeTensors header inventory differs".into());
        }
        let mut seen = Vec::new();
        reserve(&mut seen, self.source_count)?;
        seen.resize(self.source_count, false);
        let mut indices = Vec::new();
        reserve(&mut indices, self.tensors.len())?;
        for tensor in &self.tensors {
            let index = tensor.source.bind(checkpoint)?;
            if seen[index] {
                return Err("fixture source tensor is reused".into());
            }
            seen[index] = true;
            indices.push(index);
        }
        for source in &self.ignored {
            let index = source.bind(checkpoint)?;
            if seen[index] {
                return Err("omitted fixture source tensor is reused".into());
            }
            seen[index] = true;
            let (bytes, hash) = digest::hash_reader(checkpoint.reader(index)?)?;
            if bytes != source.end - source.start || hash != source.sha256 {
                return Err("qualified static causal-mask payload differs".into());
            }
        }
        if seen.contains(&false) {
            return Err("unaccounted official checkpoint tensor".into());
        }
        Ok(indices)
    }
}
fn parse_recipe(embedded: &Embedded) -> Result<Value> {
    if embedded.recipe.len() > MAX_RECIPE
        || embedded.header.len() < 24
        || embedded.header.len() > MAX_HEADER
        || digest::sha256(embedded.recipe) != embedded.recipe_sha256
    {
        return Err("embedded fixture data identity differs".into());
    }
    Ok(llama_checkpoint::json::parse(
        embedded.recipe,
        Limits {
            max_header_bytes: MAX_RECIPE as u64,
            max_total_header_bytes: MAX_RECIPE as u64,
            max_json_nodes: 30_000,
            max_text_bytes: 128 * 1024,
            max_string_bytes: 1024,
            max_json_depth: 12,
            ..Limits::default()
        },
    )?)
}
fn verify_owner(owner: &File, expected: Expected<'_>, label: &str) -> Result<()> {
    let metadata = owner.metadata()?;
    if !metadata.is_file() || metadata.len() != expected.bytes {
        return Err(format!("{label} is not the expected regular file/length").into());
    }
    let mut reader = owner.try_clone()?;
    reader.seek(SeekFrom::Start(0))?;
    let (bytes, sha256) = digest::hash_reader(
        reader.take(
            expected
                .bytes
                .checked_add(1)
                .ok_or("fixture hash bound overflow")?,
        ),
    )?;
    if bytes != expected.bytes
        || sha256 != expected.sha256
        || owner.metadata()?.len() != expected.bytes
    {
        return Err(format!("{label} SHA-256 differs; no output was published").into());
    }
    Ok(())
}
struct Staged {
    path: PathBuf,
    owner: Option<File>,
}
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
impl Staged {
    fn create(target: &Path) -> Result<Self> {
        if target.file_name().is_none() {
            return Err("fixture target needs a filename".into());
        }
        let parent = target
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        for _ in 0..16 {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".rust-llama-fixture-{}-{serial}.tmp",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(owner) => {
                    return Ok(Self {
                        path,
                        owner: Some(owner),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err("fixture temporary name collision budget exceeded".into())
    }
    fn publish(mut self, target: &Path) -> Result<()> {
        // A same-directory hard link is an atomic no-clobber publication on
        // the supported host filesystems. Unsupported filesystems fail closed.
        drop(self.owner.take());
        fs::hard_link(&self.path, target)?;
        Ok(())
    }
}
impl Drop for Staged {
    fn drop(&mut self) {
        drop(self.owner.take());
        let _ = fs::remove_file(&self.path);
    }
}
struct Output<'a> {
    owner: &'a mut File,
    position: u64,
    limit: u64,
}
impl Output<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let next = self
            .position
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= self.limit)
            .ok_or("fixture output budget exceeded")?;
        self.owner.write_all(bytes)?;
        self.position = next;
        Ok(())
    }
    fn pad_to(&mut self, target: u64) -> Result<()> {
        let gap = target
            .checked_sub(self.position)
            .filter(|n| *n < 32)
            .ok_or("fixture padding gap differs")?;
        self.write(&[0u8; 32][..usize::try_from(gap)?])
    }
}
struct TensorOutput<'a, 'b> {
    output: &'a mut Output<'b>,
    hash: Sha256,
    bytes: u64,
}
impl TensorOutput<'_, '_> {
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.output.write(bytes)?;
        self.hash.update(bytes);
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or("tensor output size overflow")?;
        Ok(())
    }
}
fn scalar(input: &[u8], source: Scalar, target: Scalar, output: &mut [u8]) -> Result<()> {
    if source == target {
        output.copy_from_slice(input);
    } else if source == Scalar::F32 {
        output.copy_from_slice(
            &llama_quant::f32_to_f16(f32::from_le_bytes(input.try_into()?)).to_le_bytes(),
        );
    } else {
        output.copy_from_slice(
            &llama_quant::f16_to_f32(u16::from_le_bytes(input.try_into()?)).to_le_bytes(),
        );
    }
    Ok(())
}
fn write_tensor(
    checkpoint: &SafetensorsFile,
    index: usize,
    tensor: &Tensor<'_>,
    output: &mut Output<'_>,
) -> Result<()> {
    let mut sink = TensorOutput {
        output,
        hash: Sha256::new(),
        bytes: 0,
    };
    let mut source_hash = Sha256::new();
    let elements = usize::try_from(product(&tensor.source.shape)?)?;
    let source_width = tensor.source.dtype.bytes();
    let target_width = tensor.dtype.bytes();
    let mut converted = [0u8; BUFFER_BYTES];
    if matches!(tensor.transform, Transform::Identity) {
        let mut reader = checkpoint.reader(index)?;
        let mut input = [0u8; BUFFER_BYTES];
        let mut remaining = elements;
        while remaining > 0 {
            let count = remaining.min(CHUNK_ELEMENTS);
            let raw = &mut input[..count * source_width];
            reader.read_exact(raw)?;
            source_hash.update(raw);
            if tensor.source.dtype == tensor.dtype {
                sink.write(raw)?;
            } else {
                let result = &mut converted[..count * target_width];
                for (input, output) in raw
                    .chunks_exact(source_width)
                    .zip(result.chunks_exact_mut(target_width))
                {
                    scalar(input, tensor.source.dtype, tensor.dtype, output)?;
                }
                sink.write(result)?;
            }
            remaining -= count;
        }
    } else {
        let input = checkpoint.read_tensor(index)?;
        source_hash.update(&input);
        let mut at = 0;
        while at < elements {
            let count = (elements - at).min(CHUNK_ELEMENTS);
            let result = &mut converted[..count * target_width];
            for (offset, value) in result.chunks_exact_mut(target_width).enumerate() {
                let position = tensor
                    .source_index(at + offset)?
                    .checked_mul(source_width)
                    .ok_or("fixture source byte index overflow")?;
                let end = position
                    .checked_add(source_width)
                    .ok_or("fixture scalar range overflow")?;
                let raw = input
                    .get(position..end)
                    .ok_or("fixture source scalar is out of range")?;
                scalar(raw, tensor.source.dtype, tensor.dtype, value)?;
            }
            sink.write(result)?;
            at += count;
        }
    }
    if digest::hex(&source_hash.finish()) != tensor.source.sha256 {
        return Err(format!("source payload hash differs: {}", tensor.source.name).into());
    }
    if sink.bytes != tensor.bytes || digest::hex(&sink.hash.finish()) != tensor.sha256 {
        return Err(format!("transformed tensor differs from original: {}", tensor.name).into());
    }
    Ok(())
}
fn prepare(plan: &Plan<'_>, input: &Path, target: &Path) -> Result<()> {
    match fs::symlink_metadata(target) {
        Ok(_) => return Err("fixture destination already exists; refusing overwrite".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    // Explicit local source path, retained once. Its exact official whole-file
    // identity is checked before parsing and again before publication.
    if !fs::metadata(input)?.is_file() {
        return Err("fixture input is not a regular file".into());
    }
    let owner = File::open(input)?;
    verify_owner(&owner, plan.source, "official source")?;
    let guard = owner.try_clone()?;
    let checkpoint = SafetensorsFile::from_file(
        owner,
        Limits {
            max_files: 1,
            max_header_bytes: 128 * 1024,
            max_total_header_bytes: 128 * 1024,
            max_json_nodes: 15_000,
            max_text_bytes: 128 * 1024,
            max_string_bytes: 1024,
            max_json_depth: 12,
            max_tensors: MAX_TENSORS,
            max_dimensions: 1024,
            max_read_bytes: MAX_RANDOM_TENSOR,
            ..Limits::default()
        },
    )?;
    let indices = plan.bind(&checkpoint)?;
    let mut staged = Staged::create(target)?;
    {
        let mut output = Output {
            owner: staged.owner.as_mut().ok_or("staging owner is absent")?,
            position: 0,
            limit: plan.output.bytes,
        };
        output.write(plan.header)?;
        for (tensor, index) in plan.tensors.iter().zip(indices) {
            output.pad_to(
                (plan.header.len() as u64)
                    .checked_add(tensor.offset)
                    .ok_or("fixture output offset overflow")?,
            )?;
            write_tensor(&checkpoint, index, tensor, &mut output)?;
        }
        if output.position != plan.output.bytes {
            return Err("fixture output is incomplete".into());
        }
        output.owner.flush()?;
        output.owner.sync_all()?;
    }
    verify_owner(&guard, plan.source, "official source after preparation")?;
    verify_owner(
        staged.owner.as_ref().ok_or("staging owner is absent")?,
        plan.output,
        "complete staged GGUF",
    )?;
    staged.publish(target)?;
    Ok(())
}
pub(crate) fn run(args: &[String]) -> Result<()> {
    crate::require_len(args, 3, 3)?;
    let embedded = match args[0].as_str() {
        "gpt2-124m-f16" => &GPT2,
        "pythia-14m-f16" => &PYTHIA,
        _ => return Err("fixture must be gpt2-124m-f16 or pythia-14m-f16; this command is not a general converter".into()),
    };
    let value = parse_recipe(embedded)?;
    let plan = Plan::parse(&value, embedded)?;
    prepare(&plan, Path::new(&args[1]), Path::new(&args[2]))?;
    println!(
        "Prepared {}: {} bytes, SHA-256 {}; {} learned tensors, {} qualified static masks omitted.",
        embedded.name,
        plan.output.bytes,
        plan.output.sha256,
        plan.tensors.len(),
        plan.ignored.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "rust-llama-fixture-test-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn independent_embedded_recipes_and_headers_have_exact_layouts() {
        for (fixture, learned, skipped, bytes) in
            [(&GPT2, 148, 12, 252_470_528), (&PYTHIA, 76, 0, 29_929_440)]
        {
            let value = parse_recipe(fixture).unwrap();
            let plan = Plan::parse(&value, fixture).unwrap();
            assert_eq!(
                (plan.tensors.len(), plan.ignored.len(), plan.output.bytes),
                (learned, skipped, bytes)
            );
        }
    }
    #[test]
    fn corrupt_source_is_rejected_before_parsing_or_publication() {
        let value = parse_recipe(&PYTHIA).unwrap();
        let mut plan = Plan::parse(&value, &PYTHIA).unwrap();
        let good = digest::sha256(b"good");
        plan.source = Expected {
            bytes: 4,
            sha256: &good,
        };
        let directory = Directory::new();
        let source = directory.0.join("input");
        let target = directory.0.join("output");
        fs::write(&source, b"evil").unwrap();
        let error = prepare(&plan, &source, &target).unwrap_err().to_string();
        assert!(error.contains("official source SHA-256"), "{error}");
        assert!(!target.exists());
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
        fs::write(&target, b"owned").unwrap();
        assert!(prepare(&plan, &source, &target).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"owned");
    }
    #[test]
    fn publication_race_does_not_clobber_destination() {
        let directory = Directory::new();
        let target = directory.0.join("output");
        let mut staged = Staged::create(&target).unwrap();
        staged
            .owner
            .as_mut()
            .unwrap()
            .write_all(b"verified")
            .unwrap();
        fs::write(&target, b"another owner").unwrap();
        assert!(staged.publish(&target).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"another owner");
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }
    #[test]
    fn retained_source_handle_is_rechecked_and_output_budget_is_checked() {
        let directory = Directory::new();
        let source = directory.0.join("input");
        fs::write(&source, b"good").unwrap();
        let owner = File::open(&source).unwrap();
        let good = digest::sha256(b"good");
        let expected = Expected {
            bytes: 4,
            sha256: &good,
        };
        verify_owner(&owner, expected, "source").unwrap();
        fs::write(&source, b"evil").unwrap();
        assert!(verify_owner(&owner, expected, "source").is_err());
        let mut output = Staged::create(&directory.0.join("out")).unwrap();
        let mut bounded = Output {
            owner: output.owner.as_mut().unwrap(),
            position: 0,
            limit: 3,
        };
        assert!(bounded.write(b"four").is_err());
        assert_eq!(bounded.position, 0);
        assert_eq!(bounded.owner.metadata().unwrap().len(), 0);
    }
}
