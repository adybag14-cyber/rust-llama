use crate::metadata::{metadata_u32, model_descriptor};
use crate::{
    DType, Error, MetadataOverride, OverrideValue, QuantizeLimits, QuantizeParams, Result,
    TensorSpec, assign_types, invalid, is_quantized, requires_imatrix,
};
use llama_gguf::{Endian, GgufSet, RawTensorDescriptor, RawValue, WriteLayout};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Debug)]
pub struct TensorReport {
    pub original_name: String,
    pub name: String,
    pub dimensions: Vec<u64>,
    pub source_type: DType,
    pub target_type: DType,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub shard: usize,
    pub requires_imatrix: bool,
}
#[derive(Clone, Debug)]
pub struct QuantizationReport {
    pub outputs: Vec<PathBuf>,
    pub tensors: Vec<TensorReport>,
    pub input_tensor_bytes: u64,
    pub output_tensor_bytes: u64,
    pub output_file_bytes: u64,
    pub requires_imatrix: bool,
    pub written: bool,
}
fn c_prefix(bytes: &[u8]) -> &[u8] {
    &bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]
}
fn put_metadata(
    metadata: &mut Vec<(Vec<u8>, RawValue)>,
    key: Vec<u8>,
    value: RawValue,
) -> Result<()> {
    let key = c_prefix(&key).to_vec();
    if key.is_empty() {
        return Err(invalid("empty metadata key"));
    }
    if let Some(i) = metadata.iter().position(|(k, _)| *k == key) {
        metadata.remove(i);
    }
    metadata.push((key, value));
    Ok(())
}
fn override_for<'a>(p: &'a QuantizeParams, key: &str) -> Option<&'a OverrideValue> {
    p.kv_overrides
        .iter()
        .find(|v| c_prefix(&v.key) == key.as_bytes())
        .map(|v| &v.value)
}
fn guessed_ftype(set: &GgufSet, indices: &[usize], p: &QuantizeParams) -> Result<u32> {
    if set.metadata().contains_key("general.file_type")
        || matches!(
            override_for(p, "general.file_type"),
            Some(OverrideValue::Integer(_))
        )
    {
        return metadata_u32(set, p, "general.file_type", None);
    }
    let mut counts = BTreeMap::new();
    let mut maximum = 0;
    let mut dominant = DType::F32;
    for &i in indices {
        let dtype = set.tensors()[i].dtype;
        let count = counts.entry(dtype.id()).or_insert(0usize);
        *count += 1;
        if *count > maximum {
            maximum = *count;
            dominant = dtype;
        }
    }
    use DType::*;
    let id = match dominant {
        F32 => 0,
        F16 => 1,
        BF16 => 32,
        Q4_0 => 2,
        Q4_1 => 3,
        Q5_0 => 8,
        Q5_1 => 9,
        Q8_0 => 7,
        Q2_K => 10,
        Q3_K => 12,
        Q4_K => 15,
        Q5_K => 17,
        Q6_K => 18,
        TQ1_0 => 36,
        TQ2_0 => 37,
        IQ2_XXS => 19,
        IQ2_XS => 20,
        IQ2_S => 28,
        IQ3_XXS => 23,
        IQ1_S => 24,
        IQ1_M => 31,
        IQ4_NL => 25,
        IQ4_XS => 30,
        IQ3_S => 26,
        NVFP4 => 39,
        Q1_0 => 40,
        Q2_0 => 41,
        F8_E4M3 => 42,
        _ => 0,
    };
    Ok(id | 1024)
}
fn layer_span(name: &str) -> Result<Option<(std::ops::Range<usize>, i32)>> {
    let bytes = name.as_bytes();
    let mut start = 0;
    while let Some(pos) = name[start..].find("blk.") {
        let begin = start + pos + 4;
        let mut end = begin;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1
        }
        if end > begin && bytes.get(end) == Some(&b'.') {
            let id = name[begin..end]
                .parse()
                .map_err(|_| invalid("layer number exceeds INT32"))?;
            return Ok(Some((begin..end, id)));
        }
        start = begin;
    }
    Ok(None)
}
fn remap(
    name: &str,
    prune: &[i32],
    mapped: &mut BTreeMap<i32, String>,
    next: &mut i32,
) -> Result<Option<String>> {
    if prune.is_empty() {
        return Ok(Some(name.into()));
    }
    let Some((span, id)) = layer_span(name)? else {
        return Ok(Some(name.into()));
    };
    if let std::collections::btree_map::Entry::Vacant(e) = mapped.entry(id) {
        let value = if prune.contains(&id) {
            String::new()
        } else if id < prune[0] {
            *next = id
                .checked_add(1)
                .ok_or_else(|| invalid("layer index overflow"))?;
            id.to_string()
        } else {
            let s = next.to_string();
            *next = next
                .checked_add(1)
                .ok_or_else(|| invalid("layer index overflow"))?;
            s
        };
        e.insert(value);
    }
    let value = &mapped[&id];
    if value.is_empty() {
        return Ok(None);
    }
    let mut result = name.to_owned();
    result.replace_range(span, value);
    Ok(Some(result))
}
fn input_indices(set: &GgufSet) -> Vec<usize> {
    let mut indices: Vec<_> = (0..set.tensors().len()).collect();
    indices.sort_by(|&a, &b| {
        let x = &set.tensors()[a].name;
        let y = &set.tensors()[b].name;
        (crate::planner::native_layer_index(x).unwrap_or(-1), x)
            .cmp(&(crate::planner::native_layer_index(y).unwrap_or(-1), y))
    });
    indices
}
fn validate_params(p: &QuantizeParams, l: &QuantizeLimits) -> Result<usize> {
    if l.max_threads == 0 || l.max_work_bytes == 0 {
        return Err(Error::Limit(
            "zero quantization worker/buffer budget".into(),
        ));
    }
    if p.tt_overrides.len() > l.max_overrides
        || p.kv_overrides.len() > l.max_overrides
        || p.prune_layers.len() > l.max_overrides
    {
        return Err(Error::Limit("quantization override count".into()));
    }
    let n = if p.nthread <= 0 {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(l.max_threads)
    } else {
        p.nthread as usize
    };
    if n > l.max_threads {
        return Err(Error::Limit(format!(
            "requested {n} threads exceeds {}",
            l.max_threads
        )));
    }
    let mut count = 0usize;
    if let Some(matrix) = &p.imatrix {
        for values in matrix.values() {
            count = count
                .checked_add(values.len())
                .ok_or_else(|| Error::Limit("importance value count overflow".into()))?;
            if count > l.max_importance_values {
                return Err(Error::Limit("importance matrix value budget".into()));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("importance matrix contains non-finite values"));
            }
        }
    }
    Ok(n)
}
fn reject_input_output_alias(set: &GgufSet, path: &Path) -> Result<()> {
    if path.exists() {
        let target = fs::canonicalize(path)?;
        for i in 0..set.shard_count() {
            let input = &set
                .shard_info(i)
                .ok_or_else(|| invalid("missing shard"))?
                .path;
            if fs::canonicalize(input)? == target {
                return Err(invalid("output aliases an input file"));
            }
            if llama_platform_loader::file_identity::same_file(input, path)? {
                return Err(invalid("output is an alias of an input file"));
            }
        }
    }
    Ok(())
}
fn header(
    out: &mut impl Write,
    metadata: &[(Vec<u8>, RawValue)],
    tensors: &[&TensorReport],
    limits: &llama_gguf::Limits,
) -> Result<WriteLayout> {
    let entries: Vec<_> = metadata.iter().map(|(k, v)| (k.as_slice(), v)).collect();
    let descriptions: Vec<_> = tensors
        .iter()
        .map(|t| RawTensorDescriptor {
            name: t.name.as_bytes(),
            dimensions: &t.dimensions,
            dtype: t.target_type,
        })
        .collect();
    Ok(llama_gguf::write_tool_header(
        out,
        &entries,
        &descriptions,
        limits,
    )?)
}
/// Quantize a complete model using bounded defaults. Output payloads are staged
/// beside their destinations before publication; dry-run creates no output.
pub fn quantize_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    params: &QuantizeParams,
) -> Result<QuantizationReport> {
    quantize_file_with_limits(input, output, params, &QuantizeLimits::default())
}

pub fn quantize_file_with_limits(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    params: &QuantizeParams,
    limits: &QuantizeLimits,
) -> Result<QuantizationReport> {
    let threads = validate_params(params, limits)?;
    let mut set = GgufSet::open_with_limits(input, limits.input.clone())?;
    if (0..set.shard_count()).any(|i| set.shard_info(i).is_none_or(|s| s.endian != Endian::Little))
    {
        return Err(Error::Unsupported(
            "frozen x86 model quantizer requires little-endian GGUF".into(),
        ));
    }
    let model = model_descriptor(&set, params)?;
    let indices = input_indices(&set);
    let output_ftype = if params.only_copy {
        guessed_ftype(&set, &indices, params)?
    } else {
        params.ftype.id()
    };
    let mut metadata = Vec::new();
    for key in set.raw_metadata_order() {
        put_metadata(&mut metadata, key.to_vec(), set.raw_metadata()[key].clone())?;
    }
    put_metadata(
        &mut metadata,
        b"general.quantization_version".to_vec(),
        RawValue::U32(2),
    )?;
    put_metadata(
        &mut metadata,
        b"general.file_type".to_vec(),
        RawValue::U32(output_ftype),
    )?;
    metadata.retain(|(key, _)| {
        !matches!(
            key.as_slice(),
            b"split.no" | b"split.count" | b"split.tensors.count"
        )
    });
    for MetadataOverride { key, value: _ } in &params.kv_overrides {
        if c_prefix(key).is_empty() {
            return Err(invalid("metadata override key is empty"));
        }
    }
    for item in &params.kv_overrides {
        put_metadata(&mut metadata, item.key.clone(), item.output_value()?)?;
    }
    let mut mapped = BTreeMap::new();
    let mut next = 0;
    let mut kept = Vec::new();
    for index in indices {
        let tensor = &set.tensors()[index];
        if let Some(name) = remap(&tensor.name, &params.prune_layers, &mut mapped, &mut next)? {
            kept.push((index, name))
        }
    }
    if !params.prune_layers.is_empty() {
        put_metadata(
            &mut metadata,
            format!("{}.block_count", model.architecture).into_bytes(),
            RawValue::U32(next as u32),
        )?;
    }
    if params.keep_split {
        kept.sort_by_key(|(i, _)| {
            let t = &set.tensors()[*i];
            (set.tensor_shard(&t.name).unwrap_or(0), t.offset)
        });
    }
    let mut names = BTreeSet::new();
    for (_, name) in &kept {
        if !names.insert(name.clone()) {
            return Err(invalid(format!(
                "layer pruning creates duplicate tensor {name}"
            )));
        }
    }
    let specs: Vec<_> = kept
        .iter()
        .map(|(i, name)| {
            let t = &set.tensors()[*i];
            TensorSpec {
                name: name.clone(),
                dimensions: t.dimensions.clone(),
                dtype: t.dtype,
            }
        })
        .collect();
    let types = assign_types(&model, params, &specs, params.imatrix.is_some())?;
    let mut tensors = Vec::new();
    let mut importance_names = Vec::new();
    let mut total_old = 0u64;
    let mut total_new = 0u64;
    for (((index, name), spec), target) in kept.iter().zip(&specs).zip(types) {
        let source = &set.tensors()[*index];
        let width =
            usize::try_from(spec.dimensions[0]).map_err(|_| invalid("row width exceeds usize"))?;
        let output_bytes =
            spec.dimensions
                .iter()
                .skip(1)
                .try_fold(target.row_bytes(width)? as u64, |a, &b| {
                    a.checked_mul(b)
                        .ok_or_else(|| invalid("tensor size overflow"))
                })?;
        let requires = requires_imatrix(name, target, params.ftype)
            && crate::allows_quantization(params, &model.architecture, spec);
        if !params.dry_run && params.imatrix.is_none() && requires {
            return Err(invalid(format!(
                "{name}: quantization requires an importance matrix"
            )));
        }
        let mut importance_name = name.clone();
        if params.imatrix.is_some()
            && !mapped.is_empty()
            && let Some((span, _)) = layer_span(name)?
        {
            let block = &name[span.clone()];
            let original = mapped
                .iter()
                .find(|(_, value)| value.as_str() == block)
                .ok_or_else(|| invalid("importance layer mapping has no original"))?
                .0;
            importance_name.replace_range(span, &original.to_string());
        }
        importance_names.push(importance_name);
        total_old = total_old
            .checked_add(source.byte_len)
            .ok_or_else(|| invalid("source byte count overflow"))?;
        total_new = total_new
            .checked_add(output_bytes)
            .ok_or_else(|| invalid("output byte count overflow"))?;
        tensors.push(TensorReport {
            original_name: source.name.clone(),
            name: name.clone(),
            dimensions: spec.dimensions.clone(),
            source_type: source.dtype,
            target_type: target,
            source_bytes: source.byte_len,
            output_bytes,
            shard: if params.keep_split {
                set.tensor_shard(&source.name)
                    .ok_or_else(|| invalid("tensor shard missing"))?
            } else {
                0
            },
            requires_imatrix: requires,
        });
    }
    let count = tensors.iter().map(|t| t.shard + 1).max().unwrap_or(1);
    if count > limits.input.max_shards || count > u16::MAX as usize {
        return Err(Error::Limit("output shard count".into()));
    }
    let mut outputs = Vec::new();
    let mut metadata_sets = Vec::new();
    let mut layouts = Vec::new();
    let mut total_files = 0u64;
    for i in 0..count {
        let selected: Vec<_> = tensors.iter().filter(|t| t.shard == i).collect();
        if i > 0 && selected.is_empty() {
            return Err(invalid(
                "pruning leaves an empty intermediate native output shard",
            ));
        }
        let mut meta = if i == 0 { metadata.clone() } else { Vec::new() };
        if count > 1 {
            put_metadata(&mut meta, b"split.no".to_vec(), RawValue::U16(i as u16))?;
            put_metadata(
                &mut meta,
                b"split.count".to_vec(),
                RawValue::U16(count as u16),
            )?;
            put_metadata(
                &mut meta,
                b"split.tensors.count".to_vec(),
                RawValue::I32(
                    i32::try_from(tensors.len())
                        .map_err(|_| invalid("tensor count exceeds INT32"))?,
                ),
            )?;
        }
        let layout = header(&mut io::sink(), &meta, &selected, &limits.input.gguf)?;
        let path = if params.keep_split {
            llama_gguf::split::split_path(output.as_ref(), i, count)?
        } else {
            output.as_ref().to_path_buf()
        };
        if !params.dry_run {
            reject_input_output_alias(&set, &path)?;
        }
        if !params.dry_run && path.is_dir() {
            return Err(invalid("output path is a directory"));
        }
        total_files = total_files
            .checked_add(layout.file_bytes)
            .ok_or_else(|| invalid("output file size overflow"))?;
        outputs.push(path);
        metadata_sets.push(meta);
        layouts.push(layout);
    }
    let mut report = QuantizationReport {
        outputs,
        tensors,
        input_tensor_bytes: total_old,
        output_tensor_bytes: total_new,
        output_file_bytes: total_files,
        requires_imatrix: false,
        written: false,
    };
    report.requires_imatrix = report.tensors.iter().any(|t| t.requires_imatrix);
    if params.dry_run {
        return Ok(report);
    }
    let mut staged = Vec::new();
    for i in 0..count {
        let mut stage = Staged::new(&report.outputs[i])?;
        let file = stage
            .file
            .take()
            .ok_or_else(|| invalid("staging file is missing"))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        let selected: Vec<_> = report.tensors.iter().filter(|t| t.shard == i).collect();
        let actual = header(
            &mut writer,
            &metadata_sets[i],
            &selected,
            &limits.input.gguf,
        )?;
        if actual != layouts[i] {
            return Err(invalid("output header plan changed"));
        }
        for (t, name) in report
            .tensors
            .iter()
            .zip(&importance_names)
            .filter(|(t, _)| t.shard == i)
        {
            write_tensor(&mut set, &mut writer, t, name, params, limits, threads)?;
            zeros(&mut writer, (32 - t.output_bytes % 32) % 32)?;
        }
        writer.flush()?;
        let file = writer.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        file.sync_all()?;
        if file.metadata()?.len() != layouts[i].file_bytes {
            return Err(invalid("staged output length differs from plan"));
        }
        drop(file);
        staged.push(stage);
    }
    // All conversion and input validation completes before publication. Each
    // rename is atomic on the supported local filesystems; several shard renames
    // are not a filesystem-wide transaction if a later rename itself fails.
    for (stage, destination) in staged.iter_mut().zip(&report.outputs) {
        reject_input_output_alias(&set, destination)?;
        fs::rename(&stage.path, destination)?;
        stage.published = true;
    }
    report.written = true;
    Ok(report)
}
static TEMP_ID: AtomicU64 = AtomicU64::new(0);
struct Staged {
    path: PathBuf,
    file: Option<File>,
    published: bool,
}
impl Staged {
    fn new(destination: &Path) -> Result<Self> {
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        for _ in 0..100 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".rust-llama-quantize-{}-{id}.tmp",
                std::process::id()
            ));
            match OpenOptions::new().create_new(true).write(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                        published: false,
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err(Error::Limit("temporary file name collision budget".into()))
    }
}
impl Drop for Staged {
    fn drop(&mut self) {
        if !self.published {
            drop(self.file.take());
            let _ = fs::remove_file(&self.path);
        }
    }
}
fn zeros(out: &mut impl Write, mut count: u64) -> Result<()> {
    let zero = [0; 32];
    while count > 0 {
        let n = count.min(32) as usize;
        out.write_all(&zero[..n])?;
        count -= n as u64;
    }
    Ok(())
}
fn bytes(n: usize) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("payload allocation".into()))?;
    v.resize(n, 0);
    Ok(v)
}
fn floats(n: usize) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("decoded payload allocation".into()))?;
    v.resize(n, 0.0);
    Ok(v)
}
fn validate_copied(dtype: DType, input: &[u8], width: usize, decoded: &mut [f32]) -> Result<()> {
    if matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
        return Ok(());
    }
    if dtype == DType::F64 {
        if input
            .as_chunks::<8>()
            .0
            .iter()
            .any(|b| !f64::from_le_bytes(*b).is_finite())
        {
            return Err(invalid("copied F64 tensor contains non-finite data"));
        }
        return Ok(());
    }
    let row = dtype.row_bytes(width)?;
    if decoded.len() != width {
        return Err(invalid("validation scratch width mismatch"));
    }
    for source in input.chunks_exact(row) {
        llama_quant::dequantize_into(dtype, source, decoded)?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err(invalid("copied tensor contains non-finite data"));
        }
        if dtype == DType::Q8_1
            && source
                .as_chunks::<36>()
                .0
                .iter()
                .any(|b| !llama_quant::f16_to_f32(u16::from_le_bytes([b[2], b[3]])).is_finite())
        {
            return Err(invalid("copied Q8_1 activation sum is non-finite"));
        }
    }
    Ok(())
}
fn write_tensor(
    set: &mut GgufSet,
    out: &mut impl Write,
    t: &TensorReport,
    importance_name: &str,
    p: &QuantizeParams,
    l: &QuantizeLimits,
    threads: usize,
) -> Result<()> {
    let width = usize::try_from(t.dimensions[0]).map_err(|_| invalid("row width exceeds usize"))?;
    let source_row = t.source_type.row_bytes(width)?;
    let target_row = t.target_type.row_bytes(width)?;
    let rows = usize::try_from(t.source_bytes / source_row as u64)
        .map_err(|_| invalid("row count exceeds usize"))?;
    let slab_target = if p.max_buf_size == 0 {
        8 * 1024 * 1024 * 1024
    } else {
        p.max_buf_size
    };
    if t.source_type == t.target_type {
        let validate_bytes = width
            .checked_mul(4)
            .ok_or_else(|| invalid("validation row overflow"))?;
        if source_row
            .checked_add(validate_bytes)
            .is_none_or(|n| n > l.max_work_bytes)
        {
            return Err(Error::Limit(
                "one source/validation row exceeds working buffer budget".into(),
            ));
        }
        let available = l.max_work_bytes - validate_bytes;
        let slab_rows = (slab_target
            .min(available)
            .min(l.input.gguf.max_tensor_read_bytes)
            / source_row)
            .max(1)
            .min(rows.max(1));
        let mut buffer = bytes(
            slab_rows
                .checked_mul(source_row)
                .ok_or_else(|| invalid("copy slab overflow"))?,
        )?;
        let mut validation = floats(width)?;
        let mut start = 0usize;
        while start < rows {
            let count = slab_rows.min(rows - start);
            let view = &mut buffer[..count * source_row];
            set.read_tensor_range(&t.original_name, (start * source_row) as u64, view)?;
            validate_copied(t.source_type, view, width, &mut validation)?;
            out.write_all(view)?;
            start += count;
        }
        return Ok(());
    }
    if is_quantized(t.source_type) && !p.allow_requantize {
        return Err(invalid(format!(
            "requantizing {:?} is disabled",
            t.source_type
        )));
    }
    if !is_quantized(t.source_type)
        && !matches!(
            t.source_type,
            DType::F32 | DType::F16 | DType::BF16 | DType::F8_E4M3
        )
    {
        return Err(Error::Unsupported(format!(
            "cannot dequantize/convert source tensor {:?}",
            t.source_type
        )));
    }
    let experts = t.dimensions.iter().skip(2).try_fold(1usize, |a, &b| {
        a.checked_mul(usize::try_from(b).map_err(|_| invalid("expert dimension overflow"))?)
            .ok_or_else(|| invalid("expert count overflow"))
    })?;
    let expected = width
        .checked_mul(experts)
        .ok_or_else(|| invalid("importance length overflow"))?;
    let importance = p.imatrix.as_ref().and_then(|m| m.get(importance_name));
    let importance = match importance {
        Some(v) if v.len() == expected => Some(v.as_slice()),
        Some(_)
            if matches!(
                t.name.as_str(),
                "token_embd.weight" | "per_layer_token_embd.weight"
            ) =>
        {
            None
        }
        Some(_) => return Err(invalid(format!("importance length differs for {}", t.name))),
        None => None,
    };
    if importance.is_none() && t.requires_imatrix {
        return Err(invalid(format!("missing importance matrix for {}", t.name)));
    }
    let decoded_row = width
        .checked_mul(4)
        .ok_or_else(|| invalid("decoded row byte overflow"))?;
    let per_row = source_row
        .checked_add(target_row)
        .and_then(|v| v.checked_add(decoded_row))
        .ok_or_else(|| invalid("row buffer size overflow"))?;
    // Each active worker temporarily owns one packed row returned by the
    // scalar codec in addition to its disjoint destination in the output slab.
    // With fewer rows than workers all rows need that additional row buffer;
    // larger slabs need only one such buffer per configured worker.
    let single_row = per_row
        .checked_add(target_row)
        .ok_or_else(|| invalid("worker row buffer size overflow"))?;
    let small_slab_rows = l.max_work_bytes / single_row;
    let budget_rows = if small_slab_rows < threads {
        small_slab_rows
    } else {
        let worker_bytes = target_row
            .checked_mul(threads)
            .ok_or_else(|| invalid("worker buffer size overflow"))?;
        (l.max_work_bytes - worker_bytes) / per_row
    };
    if budget_rows == 0 {
        return Err(Error::Limit(
            "one conversion/worker row exceeds working buffer budget".into(),
        ));
    }
    let slab_rows = (slab_target / per_row)
        .max(1)
        .min(budget_rows)
        .min((l.input.gguf.max_tensor_read_bytes / source_row).max(1))
        .min(rows.max(1));
    let mut source = bytes(
        slab_rows
            .checked_mul(source_row)
            .ok_or_else(|| invalid("source slab size overflow"))?,
    )?;
    let mut decoded = floats(
        slab_rows
            .checked_mul(width)
            .ok_or_else(|| invalid("decoded slab size overflow"))?,
    )?;
    let mut target = bytes(
        slab_rows
            .checked_mul(target_row)
            .ok_or_else(|| invalid("output slab size overflow"))?,
    )?;
    let rows_per_expert = usize::try_from(*t.dimensions.get(1).unwrap_or(&1))
        .map_err(|_| invalid("expert row count overflow"))?;
    let mut first = 0;
    while first < rows {
        let count = slab_rows.min(rows - first);
        let input = &mut source[..count * source_row];
        let values = &mut decoded[..count * width];
        let output = &mut target[..count * target_row];
        set.read_tensor_range(&t.original_name, (first * source_row) as u64, input)?;
        llama_quant::dequantize_into(t.source_type, input, values)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("source tensor contains non-finite values"));
        }
        encode_rows(
            t.target_type,
            values,
            output,
            width,
            target_row,
            first,
            rows_per_expert,
            importance,
            threads,
        )?;
        validate_copied(t.target_type, output, width, &mut decoded[..width])?;
        out.write_all(output)?;
        first += count;
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn encode_rows(
    dtype: DType,
    values: &[f32],
    out: &mut [u8],
    width: usize,
    row_bytes: usize,
    first: usize,
    rows_per_expert: usize,
    importance: Option<&[f32]>,
    threads: usize,
) -> Result<()> {
    let count = values.len() / width;
    let workers = threads.min(count).max(1);
    let chunk = count.div_ceil(workers);
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for (i, (input, output)) in values
            .chunks(chunk * width)
            .zip(out.chunks_mut(chunk * row_bytes))
            .enumerate()
        {
            let mut action = move || -> Result<()> {
                for (row, (v, dst)) in input
                    .chunks_exact(width)
                    .zip(output.chunks_exact_mut(row_bytes))
                    .enumerate()
                {
                    let encoded = if let Some(weights) = importance {
                        let expert = (first + i * chunk + row) / rows_per_expert;
                        let start = expert
                            .checked_mul(width)
                            .ok_or_else(|| invalid("importance expert offset overflow"))?;
                        let slice = weights
                            .get(start..start + width)
                            .ok_or_else(|| invalid("importance expert row missing"))?;
                        llama_quant::quantize_with_weights(dtype, v, slice)?
                    } else {
                        llama_quant::quantize(dtype, v)?
                    };
                    dst.copy_from_slice(&encoded);
                }
                Ok(())
            };
            if workers == 1 {
                action()?
            } else {
                handles.push(std::thread::Builder::new().spawn_scoped(scope, action)?);
            }
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| invalid("quantization worker panicked"))??;
        }
        Ok(())
    })
}
