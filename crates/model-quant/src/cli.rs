//! Frozen llama-quantize argument flow and bounded file preparation.
//! Copyright (c) 2023-2026 The ggml authors. MIT license.
use crate::cli_scalar::{FILE_TYPES, MAX_SCALAR_BYTES, c_prefix, c_space, integer_prefix};
pub use crate::cli_scalar::{
    parse_dtype, parse_file_type, parse_metadata_override, parse_prune_layers, parse_tensor_type,
};
use crate::imatrix::{ImatrixLimits, prepare_imatrix};
use crate::{
    Error, MetadataOverride, OverrideValue, QuantizationReport, QuantizeLimits, QuantizeParams,
    Result, invalid, quantize_file_with_limits,
};
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct CliLimits {
    pub max_arguments: usize,
    pub max_argument_bytes: usize,
    pub max_tensor_type_file_bytes: usize,
    pub max_overrides: usize,
    pub imatrix: ImatrixLimits,
    pub quantization: QuantizeLimits,
}
impl Default for CliLimits {
    fn default() -> Self {
        Self {
            max_arguments: 4096,
            max_argument_bytes: 1024 * 1024,
            max_tensor_type_file_bytes: 16 * 1024 * 1024,
            max_overrides: 4096,
            imatrix: ImatrixLimits::default(),
            quantization: QuantizeLimits::default(),
        }
    }
}
#[derive(Debug)]
pub struct CliError {
    pub error: Error,
    /// The frozen executable prints usage for option errors, but not for
    /// failed model conversion, invalid positional types or invalid MiB values.
    pub usage: bool,
}
impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for CliError {}
fn failure(error: Error) -> CliError {
    CliError {
        error,
        usage: false,
    }
}
fn usage(error: Error) -> CliError {
    CliError { error, usage: true }
}
type CliResult<T> = std::result::Result<T, CliError>;

#[derive(Clone, Debug)]
pub struct Command {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Original alias spelling from the CLI table; also controls default names.
    pub file_type_name: &'static str,
    pub params: QuantizeParams,
}
fn check_overrides(params: &QuantizeParams, bound: usize) -> Result<()> {
    if [
        params.kv_overrides.len(),
        params.tt_overrides.len(),
        params.prune_layers.len(),
    ]
    .into_iter()
    .any(|n| n > bound)
    {
        return Err(Error::Limit("CLI override count".into()));
    }
    Ok(())
}
fn tensor_type_file(path: &str, params: &mut QuantizeParams, limits: &CliLimits) -> Result<()> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len > limits.max_tensor_type_file_bytes as u64 {
        return Err(Error::Limit("tensor type file bytes".into()));
    }
    let bound = limits
        .max_tensor_type_file_bytes
        .checked_add(1)
        .ok_or_else(|| Error::Limit("tensor type file byte bound".into()))?;
    let mut bytes = Vec::new();
    let initial = usize::try_from(file_len)
        .map_err(|_| Error::Limit("tensor type file length exceeds usize".into()))?;
    bytes
        .try_reserve_exact(initial)
        .map_err(|_| Error::Limit("tensor type file allocation".into()))?;
    file.take(bound as u64).read_to_end(&mut bytes)?;
    if bytes.len() >= bound {
        return Err(Error::Limit(
            "tensor type file changed beyond its byte bound".into(),
        ));
    }
    for token in bytes.split(|b| c_space(*b)).filter(|t| !t.is_empty()) {
        if params.tt_overrides.len() >= limits.max_overrides {
            return Err(Error::Limit("tensor type file override count".into()));
        }
        let token = std::str::from_utf8(token)
            .map_err(|_| Error::Unsupported("tensor type file must be UTF-8".into()))?;
        params.tt_overrides.push(parse_tensor_type(token)?);
    }
    Ok(())
}
fn option_value<'a>(args: &'a [String], index: &mut usize) -> CliResult<&'a str> {
    *index += 1;
    args.get(*index)
        .map(|s| c_prefix(s))
        .ok_or_else(|| usage(invalid("missing option value")))
}
fn truncate_metadata(bytes: &[u8]) -> Vec<u8> {
    bytes[..bytes
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(bytes.len())
        .min(127)]
        .to_vec()
}
/// Parse arguments excluding argv[0]. Importance data and override files are
/// read here, in the frozen order, before positional file-type interpretation.
pub fn parse_command(args: &[String], limits: &CliLimits) -> CliResult<Command> {
    if args.len() > limits.max_arguments {
        return Err(failure(Error::Limit("CLI argument count".into())));
    }
    let mut total = 0usize;
    for arg in args {
        total = total
            .checked_add(arg.len())
            .filter(|n| *n <= limits.max_argument_bytes)
            .ok_or_else(|| failure(Error::Limit("CLI argument bytes".into())))?;
        if arg.len() > MAX_SCALAR_BYTES {
            return Err(failure(Error::Limit(
                "CLI argument exceeds 16384 bytes".into(),
            )));
        }
    }
    if args.len() < 2 {
        return Err(usage(invalid("not enough arguments")));
    }
    let mut params = QuantizeParams::default();
    let mut matrix_file = "";
    let mut included = Vec::new();
    let mut excluded = Vec::new();
    let mut index = 0;
    while index < args.len() && c_prefix(&args[index]).starts_with("--") {
        match c_prefix(&args[index]) {
            "--leave-output-tensor" => params.quantize_output_tensor = false,
            "--output-tensor-type" => {
                params.output_tensor_type =
                    Some(parse_dtype(option_value(args, &mut index)?).map_err(usage)?);
            }
            "--token-embedding-type" => {
                params.token_embedding_type =
                    Some(parse_dtype(option_value(args, &mut index)?).map_err(usage)?);
            }
            "--tensor-type" => {
                params
                    .tt_overrides
                    .push(parse_tensor_type(option_value(args, &mut index)?).map_err(usage)?);
            }
            "--tensor-type-file" => {
                tensor_type_file(option_value(args, &mut index)?, &mut params, limits)
                    .map_err(usage)?;
            }
            "--prune-layers" => {
                let value = parse_prune_layers(option_value(args, &mut index)?).map_err(usage)?;
                if params
                    .prune_layers
                    .len()
                    .checked_add(value.len())
                    .is_none_or(|n| n > limits.max_overrides)
                {
                    return Err(usage(Error::Limit("CLI pruned layer count".into())));
                }
                params.prune_layers.extend(value);
                params.prune_layers.sort_unstable();
                params.prune_layers.dedup();
            }
            "--override-kv" => {
                params
                    .kv_overrides
                    .push(parse_metadata_override(option_value(args, &mut index)?).map_err(usage)?);
            }
            "--dry-run" => params.dry_run = true,
            "--allow-requantize" => params.allow_requantize = true,
            "--pure" => params.pure = true,
            "--imatrix" => matrix_file = option_value(args, &mut index)?,
            "--include-weights" => included.push(option_value(args, &mut index)?.to_owned()),
            "--exclude-weights" => excluded.push(option_value(args, &mut index)?.to_owned()),
            "--keep-split" => params.keep_split = true,
            "--max-buffer-size" => {
                let value = option_value(args, &mut index)?;
                let mib = integer_prefix(value, 32)
                    .map_err(failure)?
                    .filter(|n| *n > 0)
                    .ok_or_else(|| failure(invalid("invalid --max-buffer-size")))?;
                params.max_buf_size = (mib as usize)
                    .checked_mul(1024 * 1024)
                    .ok_or_else(|| failure(Error::Limit("CLI buffer bytes".into())))?;
            }
            _ => return Err(usage(invalid("unknown option"))),
        }
        check_overrides(&params, limits.max_overrides).map_err(usage)?;
        index += 1;
    }
    if args.len() - index < 2 {
        return Err(usage(invalid("bad positional arguments")));
    }
    if !included.is_empty() && !excluded.is_empty() {
        return Err(usage(invalid(
            "--include-weights and --exclude-weights cannot be used together",
        )));
    }
    let matrix =
        prepare_imatrix(matrix_file, &included, &excluded, &limits.imatrix).map_err(failure)?;
    if !matrix.entries.is_empty() {
        let add =
            2 + usize::from(!matrix.datasets.is_empty()) + usize::from(matrix.chunk_count > 0);
        if params
            .kv_overrides
            .len()
            .checked_add(add)
            .is_none_or(|n| n > limits.max_overrides)
        {
            return Err(failure(Error::Limit("CLI generated override count".into())));
        }
        let mut put = |key: &str, value| {
            params.kv_overrides.push(MetadataOverride {
                key: key.as_bytes().to_vec(),
                value,
            })
        };
        put(
            "quantize.imatrix.file",
            OverrideValue::String(truncate_metadata(matrix_file.as_bytes())),
        );
        if let Some(dataset) = matrix.datasets.first() {
            put(
                "quantize.imatrix.dataset",
                OverrideValue::String(truncate_metadata(dataset)),
            );
        }
        put(
            "quantize.imatrix.entries_count",
            OverrideValue::Integer(matrix.entries.len() as i64),
        );
        if matrix.chunk_count > 0 {
            put(
                "quantize.imatrix.chunks_count",
                OverrideValue::Integer(i64::from(matrix.chunk_count)),
            );
        }
        params.imatrix = Some(matrix.entries);
    }
    // Empty keys are accepted by the source scalar parser and terminate the
    // C override array, including any generated importance metadata after them.
    if let Some(end) = params.kv_overrides.iter().position(|v| v.key.is_empty()) {
        params.kv_overrides.truncate(end);
    }
    let input = c_prefix(&args[index]);
    index += 1;
    let (output, selected) = if let Ok(selected) = parse_file_type(c_prefix(&args[index])) {
        let output = if params.dry_run {
            String::new()
        } else {
            let prefix = input.rfind(['/', '\\']).map_or("", |pos| &input[..=pos]);
            format!(
                "{prefix}ggml-model-{}{}",
                selected.1,
                if params.keep_split { "" } else { ".gguf" }
            )
        };
        index += 1;
        (output, selected)
    } else {
        let mut output = c_prefix(&args[index]).to_owned();
        if params.keep_split && output.contains(".gguf") {
            let new_len = output.len() - 5;
            // The source removes five bytes, even when ".gguf" is not a suffix.
            // Paths ending in an incomplete UTF-8 sequence have no portable
            // Unicode representation and are outside the admitted path domain.
            if !output.is_char_boundary(new_len) {
                return Err(failure(Error::Unsupported(
                    "split output truncates UTF-8 path bytes".into(),
                )));
            }
            output.truncate(new_len);
        }
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| failure(invalid("missing ftype")))?;
        let selected = parse_file_type(value).map_err(failure)?;
        index += 1;
        (output, selected)
    };
    params.ftype = selected.0;
    params.only_copy = selected.1 == "COPY";
    if let Some(value) = args.get(index) {
        params.nthread = integer_prefix(value, 32)
            .map_err(failure)?
            .ok_or_else(|| failure(invalid("invalid nthread")))? as i32;
    }
    // Frozen filesystem::equivalent uses an error_code and only rejects true;
    // missing/inaccessible paths are left to the file quantizer's own errors.
    if !params.dry_run
        && llama_platform_loader::file_identity::same_file(input, &output).unwrap_or(false)
    {
        return Err(failure(invalid("input and output files are the same")));
    }
    Ok(Command {
        input: PathBuf::from(input),
        output: PathBuf::from(output),
        file_type_name: selected.1,
        params,
    })
}
impl Command {
    pub fn execute(&self, limits: &QuantizeLimits) -> Result<QuantizationReport> {
        // Native dry-run never examines the destination. An empty path keeps
        // the core's publication checks independent of the user's unused name.
        let output = if self.params.dry_run {
            Path::new("")
        } else {
            self.output.as_path()
        };
        quantize_file_with_limits(&self.input, output, &self.params, limits)
    }
}
pub fn write_usage(mut out: impl Write) -> std::io::Result<()> {
    writeln!(
        out,
        "usage: llama-quantize [options] model-f32.gguf [model-quant.gguf] type [nthreads]\n"
    )?;
    writeln!(
        out,
        "Options must precede the input model. --help prints this usage and exits 1."
    )?;
    for (flag, meaning) in [
        (
            "--allow-requantize",
            "allow requantization of quantized tensors",
        ),
        ("--leave-output-tensor", "leave output.weight unchanged"),
        ("--pure", "disable mixed tensor-type presets"),
        ("--imatrix FILE", "read a legacy or GGUF importance matrix"),
        (
            "--include-weights TEXT",
            "include importance rows whose names contain TEXT; repeatable",
        ),
        (
            "--exclude-weights TEXT",
            "exclude importance rows whose names contain TEXT; repeatable",
        ),
        (
            "--output-tensor-type TYPE",
            "set the output.weight ggml tensor type",
        ),
        (
            "--token-embedding-type TYPE",
            "set the token-embedding ggml tensor type",
        ),
        (
            "--tensor-type PATTERN=TYPE",
            "set a tensor-name regex override; repeatable",
        ),
        (
            "--tensor-type-file FILE",
            "read whitespace-separated PATTERN=TYPE overrides",
        ),
        (
            "--prune-layers L0,L1,...",
            "remove zero-based layers and remap surviving names",
        ),
        ("--keep-split", "retain the input shard layout"),
        (
            "--override-kv KEY=TYPE:VALUE",
            "append int, float, bool or str metadata overrides",
        ),
        ("--dry-run", "calculate output sizes without writing files"),
        (
            "--max-buffer-size MiB",
            "set the row slab target; default 8192 MiB",
        ),
    ] {
        writeln!(out, "  {flag:<33} {meaning}")?;
    }
    writeln!(
        out,
        "\nInclude and exclude filters cannot be combined. Types (names or numeric IDs):"
    )?;
    for (name, id) in FILE_TYPES {
        writeln!(out, "  {name:<12} {id}")?;
    }
    writeln!(
        out,
        "\nCOPY preserves tensor types. Requantization and layer pruning can reduce model quality."
    )?;
    writeln!(
        out,
        "Files are validated and staged before publication. Scalar CPU execution has bounded working memory."
    )?;
    Ok(())
}
/// Run the native Rust CLI. Timing and build banners are intentionally not a
/// byte-stable interface; exit status, usage class and GGUF bytes are tested.
pub fn run(
    args: &[String],
    limits: &CliLimits,
    mut stdout: impl Write,
    mut stderr: impl Write,
) -> i32 {
    let command = match parse_command(args, limits) {
        Ok(value) => value,
        Err(error) => {
            let _ = writeln!(stderr, "llama-quantize: {error}");
            if error.usage {
                let _ = write_usage(&mut stdout);
            }
            return 1;
        }
    };
    let prefix = if command.params.dry_run {
        "calculating quantization size for"
    } else {
        "quantizing"
    };
    let _ = writeln!(
        stderr,
        "llama_quantize: {prefix} '{}' as {}",
        command.input.display(),
        command.file_type_name
    );
    match command.execute(&limits.quantization) {
        Ok(report) => {
            if command.params.dry_run && command.params.imatrix.is_none() && report.requires_imatrix
            {
                let _ = writeln!(
                    stderr,
                    "llama_model_quantize_impl: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!"
                );
            }
            let _ = writeln!(
                stdout,
                "{} tensors: {} input tensor bytes, {} output tensor bytes, {} output file bytes{}",
                report.tensors.len(),
                report.input_tensor_bytes,
                report.output_tensor_bytes,
                report.output_file_bytes,
                if report.written {
                    " written"
                } else {
                    " (dry run)"
                }
            );
            0
        }
        Err(error) => {
            if matches!(&error, Error::Invalid(text) if text.contains("requires an importance matrix"))
            {
                let _ = writeln!(
                    stderr,
                    "llama_model_quantize: failed to quantize: this quantization requires an imatrix!"
                );
            }
            let _ = writeln!(stderr, "llama-quantize: {error}");
            1
        }
    }
}
