//! Owned dense Llama conversion planning and bounded streaming publication.
use crate::authorship::Naming;
use crate::codec::SourceProfile;
use crate::config::Config;
use crate::util::{JsonBudget, Metadata, integer, real, text};
use crate::{Error, Limits, Options, OutputType, Result, reserve};
use llama_checkpoint::{Checkpoint, CheckpointOptions, NamePolicy, SourceDirectory};
use llama_gguf::{RawTensorDescriptor, RawValue, WriteLayout};
use llama_quant::DType;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A resolved conversion. These counts describe the written dense checkpoint,
/// including generated RoPE factors, rather than inferred runtime support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub output_type: OutputType,
    pub source_profile: SourceProfile,
    pub tensors: usize,
    pub parameters: u64,
    pub bytes: u64,
    pub vocab_only: bool,
}
struct ExtraRope {
    parameters: crate::rope::Parameters,
    bytes: usize,
    shape: [u64; 1],
}
/// Retains the validated checkpoint/sidecar owners for the entire conversion.
/// Input tensor contents must remain stable on Unix. No Python/reference runtime
/// is required. Planning validates metadata/layout and does not publish a file.
pub struct Conversion {
    checkpoint: Checkpoint,
    source: SourceDirectory,
    input_root: PathBuf,
    plan: crate::tensors::Plan,
    extra: Option<ExtraRope>,
    metadata: Metadata,
    dimensions: Vec<Vec<u64>>,
    layout: WriteLayout,
    naming: Naming,
    limits: Limits,
    summary: Summary,
}
impl Conversion {
    pub fn prepare(input: impl AsRef<Path>, options: Options, limits: Limits) -> Result<Self> {
        let input = input.as_ref();
        let path_text = input
            .to_str()
            .ok_or(Error::Invalid("input directory is not Unicode scalar text"))?;
        let folder = input
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or(Error::Invalid("input directory has no Unicode name"))?;
        let input_root = input.canonicalize()?;
        let profile = options
            .source_profile
            .unwrap_or_else(SourceProfile::current);
        let mut source = SourceDirectory::open(input, limits.sidecars.clone())?;
        let mut budget = JsonBudget::new(limits.json.clone());
        let raw = budget.parse(source.read_required("config.json")?)?;
        let config = Config::new(raw, path_text, &mut budget, &limits)?;
        if !options.vocab_only && source.read_optional("hf_quant_config.json")?.is_some() {
            return Err(Error::Unsupported(
                "HF quantization sidecar requires the separate prequantized conversion contract",
            ));
        }
        let checkpoint = Checkpoint::open_source(
            &source,
            CheckpointOptions {
                names: NamePolicy::Unchanged,
                ..CheckpointOptions::default()
            },
            limits.checkpoint.clone(),
        )?;
        let plan = crate::tensors::Plan::new(&checkpoint, &config, &options, &limits)?;
        let extra = if options.vocab_only {
            None
        } else {
            extra_rope(&config, profile, &limits)?
        };
        let count = plan
            .tensors
            .len()
            .checked_add(usize::from(extra.is_some()))
            .ok_or(Error::Limit("output tensor count"))?;
        if count > limits.gguf.max_tensors {
            return Err(Error::Limit("output tensor count"));
        }
        let parameters = plan
            .parameters
            .checked_add(extra.as_ref().map_or(0, |e| e.shape[0]))
            .ok_or(Error::Limit("parameter total"))?;
        let mut metadata = Metadata::default();
        metadata.string("general.architecture", "llama")?;
        metadata.string("general.type", "model")?;
        let naming = crate::authorship::write(
            &mut source,
            config.root(),
            folder,
            options.model_name.as_deref(),
            options.metadata_override.as_deref(),
            parameters,
            &mut budget,
            &limits,
            profile,
            &mut metadata,
        )?;
        config.metadata(&mut metadata, plan.output_type)?;
        metadata.uint("general.quantization_version", 2)?;
        crate::vocab::write(
            &mut source,
            &config,
            &mut budget,
            &mut metadata,
            &limits,
            profile,
        )?;
        let mut dimensions = Vec::new();
        reserve(
            &mut dimensions,
            plan.tensors.len(),
            "reversed tensor shapes",
        )?;
        for tensor in &plan.tensors {
            let mut shape = Vec::new();
            reserve(&mut shape, tensor.shape.len(), "GGUF tensor shape")?;
            shape.extend(tensor.shape.iter().rev().copied());
            dimensions.push(shape);
        }
        let descriptors = descriptors(&plan, &dimensions, extra.as_ref())?;
        let metadata_refs = metadata_refs(&metadata)?;
        // The same checked shared writer calculates encoded_header_bytes and
        // aligned layout. No duplicated wire-size calculation or byte patching.
        let layout = llama_gguf::write_raw_header(
            &mut io::sink(),
            &metadata_refs,
            &descriptors,
            &limits.gguf,
        )?;
        let expected_bytes = if options.vocab_only {
            layout.encoded_header_bytes
        } else {
            layout.file_bytes
        };
        if expected_bytes > limits.max_output_bytes {
            return Err(Error::Limit("total output bytes"));
        }
        let offset = usize::from(extra.is_some());
        if let Some(extra) = &extra
            && layout.tensor_bytes[0] != extra.bytes as u64
        {
            return Err(Error::Invalid("RoPE planned byte count"));
        }
        for (tensor, &bytes) in plan.tensors.iter().zip(&layout.tensor_bytes[offset..]) {
            if bytes != tensor.encoded_bytes as u64 {
                return Err(Error::Invalid("tensor planned byte count"));
            }
        }
        let summary = Summary {
            output_type: plan.output_type,
            source_profile: profile,
            tensors: count,
            parameters,
            bytes: expected_bytes,
            vocab_only: options.vocab_only,
        };
        drop(metadata_refs);
        drop(descriptors);
        Ok(Self {
            checkpoint,
            source,
            input_root,
            plan,
            extra,
            metadata,
            dimensions,
            layout,
            naming,
            limits,
            summary,
        })
    }
    pub fn summary(&self) -> &Summary {
        &self.summary
    }
    /// Stream a complete prepared GGUF to the caller's writer. As with Write,
    /// an I/O/conversion failure may leave partial bytes in that caller-owned
    /// sink. write_file uses a separate completed temporary file for publication.
    pub fn write_to<W: Write>(&self, output: &mut W) -> Result<()> {
        let metadata = metadata_refs(&self.metadata)?;
        let tensors = descriptors(&self.plan, &self.dimensions, self.extra.as_ref())?;
        if self.summary.vocab_only {
            let mut bounded = HeaderPrefix {
                output,
                remaining: self.layout.encoded_header_bytes,
            };
            let layout =
                llama_gguf::write_raw_header(&mut bounded, &metadata, &tensors, &self.limits.gguf)?;
            if bounded.remaining != 0 || layout != self.layout {
                return Err(Error::Invalid("vocabulary header plan changed"));
            }
            return Ok(());
        }
        let layout = llama_gguf::write_raw_header(output, &metadata, &tensors, &self.limits.gguf)?;
        if layout != self.layout {
            return Err(Error::Invalid("prepared header plan changed"));
        }
        let mut total = layout.data_offset;
        if let Some(extra) = &self.extra {
            let mut data = Vec::new();
            data.try_reserve_exact(extra.bytes)
                .map_err(|_| Error::Limit("RoPE factor output"))?;
            data.resize(extra.bytes, 0);
            crate::rope::generate_into_with_profile(
                &extra.parameters,
                self.summary.source_profile,
                &mut data,
            )?;
            output.write_all(&data)?;
            total = write_padding(output, total, data.len() as u64, layout.alignment)?;
        }
        for tensor in &self.plan.tensors {
            let data =
                tensor.encode(&self.checkpoint, self.summary.source_profile, &self.limits)?;
            output.write_all(&data)?;
            total = write_padding(output, total, data.len() as u64, layout.alignment)?;
        }
        if total != self.summary.bytes {
            return Err(Error::Invalid("written byte total differs from plan"));
        }
        Ok(())
    }
    /// Choose the frozen conventional name for an existing directory, or fill
    /// supported ftype fields in an explicit output filename. Non-UTF8 model-card
    /// metadata remains valid in GGUF; automatic filenames require Unicode.
    pub fn output_path(&self, requested: impl AsRef<Path>) -> Result<PathBuf> {
        let requested = requested.as_ref();
        if requested.is_dir() {
            let raw = self.naming.basename.as_deref().unwrap_or(&self.naming.name);
            let name = std::str::from_utf8(raw).map_err(|_| {
                Error::Invalid("binary model name cannot form an automatic filename")
            })?;
            let mut name = crate::python_unicode::strip(name, self.summary.source_profile)?
                .replace([' ', '/'], "-");
            let mut append = |value: Option<&[u8]>, strip: bool| -> Result<()> {
                if let Some(value) = value {
                    let value = std::str::from_utf8(value).map_err(|_| {
                        Error::Invalid(
                            "binary model name component cannot form an automatic filename",
                        )
                    })?;
                    let value = if strip {
                        crate::python_unicode::strip(value, self.summary.source_profile)?
                    } else {
                        value
                    };
                    let part = value.replace(' ', "-");
                    name.try_reserve(part.len() + 1)
                        .map_err(|_| Error::Limit("output name"))?;
                    name.push('-');
                    name.push_str(&part);
                }
                Ok(())
            };
            if !self.summary.vocab_only {
                append(self.naming.size_label.as_deref(), false)?;
            }
            append(self.naming.finetune.as_deref(), true)?;
            append(self.naming.version.as_deref(), true)?;
            if self.summary.vocab_only {
                append(Some(b"vocab"), false)?;
            } else {
                append(
                    Some(
                        self.summary
                            .output_type
                            .name()
                            .to_ascii_uppercase()
                            .as_bytes(),
                    ),
                    false,
                )?;
            }
            name.try_reserve(5)
                .map_err(|_| Error::Limit("output name"))?;
            name.push_str(".gguf");
            checked_filename(&name)?;
            Ok(requested.join(name))
        } else {
            let filename = requested
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or(Error::Invalid("output filename is not Unicode"))?;
            let filename = fill_filename(filename, self.summary.output_type)?;
            checked_filename(&filename)?;
            Ok(requested.with_file_name(filename))
        }
    }
    /// Write, flush and sync a sibling temporary file, then atomically rename it
    /// into place. Existing output bytes survive every pre-publication failure.
    /// All input paths are protected from direct replacement. The destination
    /// parent and input path namespace must remain exclusively controlled/stable
    /// through publication; this is not an atomic filesystem snapshot.
    pub fn write_file(&self, requested: impl AsRef<Path>) -> Result<PathBuf> {
        let target = self.output_path(requested)?;
        let parent = target
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()?;
        let filename = target
            .file_name()
            .ok_or(Error::Invalid("output filename is absent"))?;
        let target = parent.join(filename);
        self.reject_input_path(&target)?;
        let mut temporary = None;
        for _ in 0..128 {
            let number = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".rust-llama-convert-{}-{number}.tmp",
                std::process::id()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(owner) => {
                    temporary = Some(Temporary {
                        path: Some(path),
                        owner,
                    });
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        let mut temporary = temporary.ok_or(Error::Limit("temporary output-name attempts"))?;
        {
            let mut buffered = Buffer::new(&mut temporary.owner)?;
            self.write_to(&mut buffered)?;
            buffered.flush()?;
        }
        if temporary.owner.metadata()?.len() != self.summary.bytes {
            return Err(Error::Invalid("completed output size differs from plan"));
        }
        temporary.owner.sync_all()?;
        self.reject_input_path(&target)?;
        fs::rename(temporary.path.as_ref().expect("temporary path"), &target)?;
        temporary.path = None;
        Ok(target)
    }
    fn reject_input_path(&self, target: &Path) -> Result<()> {
        let selected = match target.canonicalize() {
            Ok(path) => path,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        for name in self.source.cached_names() {
            let name = name
                .as_str()
                .ok_or(Error::Invalid("cached sidecar name is not Unicode"))?;
            if self.input_root.join(name).canonicalize()? == selected {
                return Err(Error::Invalid("output path names a retained sidecar input"));
            }
        }
        for index in 0..self.checkpoint.file_count() {
            let (name, _) = self
                .checkpoint
                .file(index)
                .ok_or(Error::Invalid("checkpoint owner"))?;
            let name = name
                .as_str()
                .ok_or(Error::Invalid("selected input path is not Unicode"))?;
            if self.input_root.join(name).canonicalize()? == selected {
                return Err(Error::Invalid("output path names a retained tensor input"));
            }
        }
        let index = self.input_root.join("model.safetensors.index.json");
        match index.canonicalize() {
            Ok(index) if index == selected => Err(Error::Invalid(
                "output path names the selected checkpoint index",
            )),
            Ok(_) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}
fn extra_rope(
    config: &Config,
    profile: SourceProfile,
    limits: &Limits,
) -> Result<Option<ExtraRope>> {
    let fields = config.rope()?;
    let Some(kind) = fields.full("rope_type")? else {
        return Ok(None);
    };
    if !text(kind)?.eq_ignore_ascii_case("llama3") {
        return Ok(None);
    }
    let dim = if let Some(dim) = config
        .get("head_dim")?
        .filter(|v| !matches!(v, llama_checkpoint::Value::Null))
    {
        integer(dim)?
    } else {
        crate::config::floor_div(
            integer(config.required(&["hidden_size"])?)?,
            integer(config.required(&["num_attention_heads"])?)?,
        )?
    };
    let mut parameters = crate::rope::Parameters::new(dim);
    for (key, target) in [
        ("rope_theta", &mut parameters.theta),
        ("factor", &mut parameters.factor),
        ("low_freq_factor", &mut parameters.low_frequency_factor),
        ("high_freq_factor", &mut parameters.high_frequency_factor),
        (
            "original_max_position_embeddings",
            &mut parameters.original_context_length,
        ),
    ] {
        if let Some(value) = fields.full(key)? {
            *target = real(value)?;
        }
    }
    let bytes = crate::rope::encoded_len_with_profile(&parameters, profile)?;
    if bytes > limits.max_working_bytes || bytes as u64 > limits.max_output_bytes {
        return Err(Error::Limit("generated RoPE tensor bytes"));
    }
    Ok(Some(ExtraRope {
        parameters,
        bytes,
        shape: [(bytes / 4) as u64],
    }))
}
fn metadata_refs(metadata: &Metadata) -> Result<Vec<(&[u8], &RawValue)>> {
    let mut result = Vec::new();
    reserve(&mut result, metadata.0.len(), "ordered metadata references")?;
    result.extend(
        metadata
            .0
            .iter()
            .map(|(key, value)| (key.as_bytes(), value)),
    );
    Ok(result)
}
fn descriptors<'a>(
    plan: &'a crate::tensors::Plan,
    dimensions: &'a [Vec<u64>],
    extra: Option<&'a ExtraRope>,
) -> Result<Vec<RawTensorDescriptor<'a>>> {
    let mut result = Vec::new();
    reserve(
        &mut result,
        plan.tensors.len() + usize::from(extra.is_some()),
        "output tensor references",
    )?;
    if let Some(extra) = extra {
        result.push(RawTensorDescriptor {
            name: b"rope_freqs.weight",
            dimensions: &extra.shape,
            dtype: DType::F32,
        });
    }
    for (tensor, shape) in plan.tensors.iter().zip(dimensions) {
        result.push(RawTensorDescriptor {
            name: tensor.name.as_bytes(),
            dimensions: shape,
            dtype: tensor.dtype,
        });
    }
    Ok(result)
}
fn write_padding<W: Write>(output: &mut W, total: u64, length: u64, alignment: u32) -> Result<u64> {
    let mask = u64::from(alignment) - 1;
    let padded = length
        .checked_add(mask)
        .map(|n| n & !mask)
        .ok_or(Error::Limit("tensor padding"))?;
    let mut count = padded - length;
    let zeros = [0u8; 4096];
    while count > 0 {
        let take = count.min(zeros.len() as u64) as usize;
        output.write_all(&zeros[..take])?;
        count -= take as u64;
    }
    total
        .checked_add(padded)
        .ok_or(Error::Limit("output byte count"))
}
struct HeaderPrefix<'a, W> {
    output: &'a mut W,
    remaining: u64,
}
impl<W: Write> Write for HeaderPrefix<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = self.remaining.min(bytes.len() as u64) as usize;
        self.output.write_all(&bytes[..count])?;
        self.remaining -= count as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}
struct Buffer<'a, W> {
    output: &'a mut W,
    data: Vec<u8>,
    used: usize,
}
impl<'a, W: Write> Buffer<'a, W> {
    fn new(output: &'a mut W) -> Result<Self> {
        let mut data = Vec::new();
        data.try_reserve_exact(64 * 1024)
            .map_err(|_| Error::Limit("output buffer"))?;
        data.resize(64 * 1024, 0);
        Ok(Self {
            output,
            data,
            used: 0,
        })
    }
    fn drain(&mut self) -> io::Result<()> {
        self.output.write_all(&self.data[..self.used])?;
        self.used = 0;
        Ok(())
    }
}
impl<W: Write> Write for Buffer<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.data.len() - self.used {
            self.drain()?;
        }
        if bytes.len() >= self.data.len() {
            self.output.write_all(bytes)?;
        } else {
            self.data[self.used..self.used + bytes.len()].copy_from_slice(bytes);
            self.used += bytes.len();
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.drain()?;
        self.output.flush()
    }
}
struct Temporary {
    path: Option<PathBuf>,
    owner: File,
}
impl Drop for Temporary {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}
fn checked_filename(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." || name.contains(['\0', '/', '\\']) {
        return Err(Error::Invalid("generated name is not one filename"));
    }
    Ok(())
}
fn fill_filename(input: &str, kind: OutputType) -> Result<String> {
    let lower = kind.name();
    let upper = lower.to_ascii_uppercase();
    let mut output = String::new();
    output
        .try_reserve(input.len())
        .map_err(|_| Error::Limit("output filename"))?;
    let mut at = 0;
    let mut automatic = false;
    let mut numeric = false;
    while at < input.len() {
        let rest = &input[at..];
        if rest.starts_with("{{") {
            output.push('{');
            at += 2;
            continue;
        }
        if rest.starts_with("}}") {
            output.push('}');
            at += 2;
            continue;
        }
        if rest.starts_with('{') {
            let end = rest
                .find('}')
                .ok_or(Error::Invalid("unclosed filename format field"))?;
            let field = &rest[1..end];
            let value = match field {
                "" => {
                    if automatic || numeric {
                        return Err(Error::Invalid("filename positional format index"));
                    }
                    automatic = true;
                    lower
                }
                "0" => {
                    if automatic {
                        return Err(Error::Invalid("mixed filename field numbering"));
                    }
                    numeric = true;
                    lower
                }
                "ftype" | "outtype" => lower,
                "FTYPE" | "OUTTYPE" => upper.as_str(),
                _ => {
                    return Err(Error::Unsupported(
                        "filename format field/conversion outside captured ftype substitutions",
                    ));
                }
            };
            output
                .try_reserve(value.len())
                .map_err(|_| Error::Limit("formatted output filename"))?;
            output.push_str(value);
            at += end + 1;
        } else {
            let ch = rest.chars().next().expect("nonempty filename remainder");
            if ch == '}' {
                return Err(Error::Invalid("unmatched filename closing brace"));
            }
            output
                .try_reserve(ch.len_utf8())
                .map_err(|_| Error::Limit("formatted filename"))?;
            output.push(ch);
            at += ch.len_utf8();
        }
    }
    Ok(output)
}
