use llama_convert::{Conversion, Limits, Options, OutputType};
use std::ffi::OsString;
use std::path::PathBuf;

const HELP: &str =
    "Convert a supported dense Hugging Face Llama, Qwen, GPT-2, GPT-NeoX or BLOOM Safetensors directory to GGUF.

Usage: convert-hf-to-gguf [OPTIONS] MODEL_DIRECTORY

Options:
  --outfile PATH          Output file or existing directory (default: model directory)
  --outtype TYPE          auto, f32, f16, bf16, q8_0, tq1_0 or tq2_0 (default: auto)
  --vocab-only            Emit the original unpadded vocabulary-only GGUF header
  --fuse-qkv              Fuse complete compatible Q/K/V projection groups
  --model-name NAME       Override the generated model name
  --metadata RELATIVE     Authorship override relative to the retained input directory
  --dry-run              Validate and describe the conversion without creating output
  --max-tensor-input-mib N  Bound one input tensor (default: 1024)
  --max-working-mib N     Bound charged allocations for one output tensor (default: 2048)
  --max-output-gib N      Bound complete output size (default: 512)
  -h, --help             Show this help
  --version              Show this program's version

The admitted architectures are LlamaForCausalLM, LLaMAForCausalLM, LlamaModel,
Qwen2ForCausalLM, Qwen2Model, Qwen3ForCausalLM, Qwen3Model, GPT2LMHeadModel
GPTNeoXForCausalLM, BloomForCausalLM and BloomModel.
See crates/convert/README.md for tokenizer, dtype and conversion boundaries.
Other upstream converters and options remain explicit unsupported work.
";

struct Arguments {
    input: PathBuf,
    output: Option<PathBuf>,
    options: Options,
    limits: Limits,
    dry_run: bool,
}

fn value(
    inline: Option<&str>,
    args: &mut impl Iterator<Item = OsString>,
    flag: &str,
) -> Result<OsString, String> {
    inline
        .map(OsString::from)
        .or_else(|| args.next())
        .ok_or_else(|| format!("{flag} needs a value"))
}
fn string(value: OsString, flag: &str) -> Result<String, String> {
    value
        .into_string()
        .map_err(|_| format!("{flag} requires Unicode text"))
}
fn size(value: OsString, flag: &str, shift: u32) -> Result<u64, String> {
    let text = string(value, flag)?;
    let count = text
        .parse::<u64>()
        .map_err(|_| format!("{flag} requires a nonnegative integer"))?;
    count
        .checked_mul(1u64 << shift)
        .ok_or_else(|| format!("{flag} is too large"))
}
fn arguments() -> Result<Option<Arguments>, String> {
    let mut args = std::env::args_os().skip(1);
    let (mut input, mut output, mut dry_run) = (None, None, false);
    let mut options = Options::default();
    let mut limits = Limits::default();
    let mut positional = false;
    while let Some(arg) = args.next() {
        if !positional && arg == "--" {
            positional = true;
            continue;
        }
        if !positional && arg.to_str().is_some_and(|v| v.starts_with('-')) {
            let text = arg.to_str().expect("checked Unicode option");
            let (flag, inline) = text
                .split_once('=')
                .map_or((text, None), |(flag, value)| (flag, Some(value)));
            match flag {
                "-h" | "--help" if inline.is_none() => {
                    print!("{HELP}");
                    return Ok(None);
                }
                "--version" if inline.is_none() => {
                    println!("convert-hf-to-gguf {}", env!("CARGO_PKG_VERSION"));
                    return Ok(None);
                }
                "--outfile" => output = Some(PathBuf::from(value(inline, &mut args, flag)?)),
                "--outtype" => {
                    options.output_type =
                        OutputType::parse(&string(value(inline, &mut args, flag)?, flag)?)
                            .map_err(|e| e.to_string())?;
                }
                "--vocab-only" if inline.is_none() => options.vocab_only = true,
                "--fuse-qkv" if inline.is_none() => options.fuse_qkv = true,
                "--dry-run" if inline.is_none() => dry_run = true,
                "--model-name" => {
                    options.model_name = Some(string(value(inline, &mut args, flag)?, flag)?)
                }
                "--metadata" => {
                    options.metadata_override = Some(string(value(inline, &mut args, flag)?, flag)?)
                }
                "--max-tensor-input-mib" => {
                    limits.max_tensor_input_bytes =
                        usize::try_from(size(value(inline, &mut args, flag)?, flag, 20)?)
                            .map_err(|_| format!("{flag} exceeds the host address range"))?;
                }
                "--max-working-mib" => {
                    limits.max_working_bytes =
                        usize::try_from(size(value(inline, &mut args, flag)?, flag, 20)?)
                            .map_err(|_| format!("{flag} exceeds the host address range"))?;
                }
                "--max-output-gib" => {
                    limits.max_output_bytes = size(value(inline, &mut args, flag)?, flag, 30)?
                }
                _ => {
                    return Err(format!(
                        "unsupported option {text}; use --help for admitted options"
                    ));
                }
            }
        } else if input.replace(PathBuf::from(arg)).is_some() {
            return Err("exactly one model directory is required".into());
        }
    }
    Ok(Some(Arguments {
        input: input.ok_or("a model directory is required; use --help")?,
        output,
        options,
        limits,
        dry_run,
    }))
}
fn run(args: Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let conversion = Conversion::prepare(&args.input, args.options, args.limits)?;
    let destination = args.output.as_deref().unwrap_or(&args.input);
    let summary = conversion.summary();
    let path = conversion.output_path(destination)?;
    if args.dry_run {
        println!(
            "{}: {} tensors, {} parameters, {} bytes, {}",
            path.display(),
            summary.tensors,
            summary.parameters,
            summary.bytes,
            summary.output_type.name()
        );
        return Ok(());
    }
    let output = conversion.write_file(destination)?;
    eprintln!(
        "Wrote {} tensors, {} parameters and {} bytes as {}.",
        summary.tensors,
        summary.parameters,
        summary.bytes,
        summary.output_type.name()
    );
    println!("{}", output.display());
    Ok(())
}
fn main() {
    match arguments() {
        Ok(None) => {}
        Ok(Some(args)) => {
            if let Err(error) = run(args) {
                eprintln!("convert-hf-to-gguf: {error}");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("convert-hf-to-gguf: {error}");
            std::process::exit(2);
        }
    }
}
