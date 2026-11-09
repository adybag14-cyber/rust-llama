//! Command-line port of the frozen tools/gguf-split utility.
use llama_gguf::ShardLimits;
use llama_gguf::split::{self, MergeOptions, SplitLimit, SplitOptions};
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

fn usage(out: &mut impl Write, executable: &OsStr) -> io::Result<()> {
    writeln!(out)?;
    writeln!(
        out,
        "usage: {} [options] GGUF_IN GGUF_OUT",
        executable.to_string_lossy()
    )?;
    writeln!(out)?;
    writeln!(out, "Apply a GGUF operation on IN to OUT.")?;
    writeln!(out, "options:")?;
    writeln!(
        out,
        "  -h, --help              show this help message and exit"
    )?;
    writeln!(out, "  --version               show version and build info")?;
    writeln!(
        out,
        "  --split                 split GGUF to multiple GGUF (enabled by default)"
    )?;
    writeln!(
        out,
        "  --merge                 merge multiple GGUF to a single GGUF"
    )?;
    writeln!(
        out,
        "  --split-max-tensors     max tensors in each split (default: 128)"
    )?;
    writeln!(out, "  --split-max-size N(M|G) max size per split")?;
    writeln!(
        out,
        "  --no-tensor-first-split do not add tensors to the first split (disabled by default)"
    )?;
    writeln!(
        out,
        "  --dry-run               only print out a split plan and exit, without writing any new files"
    )?;
    writeln!(
        out,
        "  --delete-splits         delete split files only after the complete merge is flushed and synced"
    )?;
    writeln!(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Split,
    Merge,
}
enum Action {
    Help,
    Version,
    Run {
        operation: Operation,
        input: PathBuf,
        output: PathBuf,
        split: SplitOptions,
        merge: MergeOptions,
    },
}

// Match the defined decimal-prefix behavior of atoi/sscanf("%d"). Overflow and
// a missing sscanf value are checked instead of reproducing C undefined behavior.
fn decimal_prefix(text: &str) -> Result<Option<i32>, String> {
    let bytes = text.as_bytes();
    let mut index = 0;
    while bytes
        .get(index)
        .is_some_and(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c'))
    {
        index += 1;
    }
    let negative = bytes.get(index) == Some(&b'-');
    if negative || bytes.get(index) == Some(&b'+') {
        index += 1;
    }
    let start = index;
    let mut value = 0u64;
    while let Some(&digit) = bytes.get(index).filter(|c| c.is_ascii_digit()) {
        value = value
            .checked_mul(10)
            .and_then(|n| n.checked_add(u64::from(digit - b'0')))
            .ok_or_else(|| {
                "error: decimal argument overflows the source signed integer".to_string()
            })?;
        if value > i32::MAX as u64 + u64::from(negative) {
            return Err("error: decimal argument overflows the source signed integer".into());
        }
        index += 1;
    }
    if index == start {
        return Ok(None);
    }
    let value = if negative {
        -(value as i64)
    } else {
        value as i64
    };
    Ok(Some(value as i32))
}
fn size(text: &str) -> Result<u64, String> {
    let unit = match text.as_bytes().last() {
        Some(b'M') => 1_000_000u64,
        Some(b'G') => 1_000_000_000u64,
        Some(c) => {
            return Err(format!(
                "error: supported units are M (megabytes) or G (gigabytes), but got: {}",
                char::from(*c)
            ));
        }
        None => return Err("error: size argument is empty".into()),
    };
    let n = decimal_prefix(text)?
        .ok_or_else(|| "error: size argument has no decimal value".to_string())?;
    if n <= 0 {
        return Err("error: size must be a positive value".into());
    }
    (n as u64)
        .checked_mul(unit)
        .ok_or_else(|| "error: size argument overflows the byte count".into())
}
fn parse(args: &[OsString]) -> Result<Action, String> {
    let mut operation = None;
    let mut mode = None;
    let mut split = SplitOptions::default();
    let mut merge = MergeOptions::default();
    let mut index = 0;
    while let Some(raw) = args
        .get(index)
        .and_then(|x| x.to_str())
        .filter(|x| x.starts_with("--"))
    {
        let arg = raw.replace('_', "-");
        match arg.as_str() {
            "--help" => return Ok(Action::Help),
            "--version" => return Ok(Action::Version),
            "--dry-run" => {
                split.dry_run = true;
                merge.dry_run = true;
            }
            "--no-tensor-first-split" => split.no_tensor_first_split = true,
            "--delete-splits" => merge.delete_splits = true,
            "--split" | "--merge" => {
                let next = if arg == "--split" {
                    Operation::Split
                } else {
                    Operation::Merge
                };
                if operation.is_some_and(|old| old != next) {
                    return Err(
                        "error: either --split or --merge can be specified, but not both".into(),
                    );
                }
                operation = Some(next);
            }
            "--split-max-tensors" | "--split-max-size" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("error: invalid parameter for argument: {arg}"))?;
                let value = value
                    .to_str()
                    .ok_or_else(|| format!("error: non-text parameter for argument: {arg}"))?;
                let is_size = arg == "--split-max-size";
                if mode.is_some_and(|old| old != is_size) {
                    return Err("error: either --split-max-tensors or --split-max-size can be specified, but not both".into());
                }
                mode = Some(is_size);
                split.limit = if is_size {
                    SplitLimit::Bytes(size(value)?)
                } else {
                    SplitLimit::Tensors(decimal_prefix(value)?.unwrap_or(0))
                };
            }
            _ => return Err(format!("error: unknown argument: {arg}")),
        }
        index += 1;
    }
    if args.len() - index != 2 {
        return Err("error: bad arguments".into());
    }
    Ok(Action::Run {
        operation: operation.unwrap_or(Operation::Split),
        input: PathBuf::from(&args[index]),
        output: PathBuf::from(&args[index + 1]),
        split,
        merge,
    })
}
fn main() -> ExitCode {
    let mut args = std::env::args_os();
    let executable = args
        .next()
        .unwrap_or_else(|| OsString::from("llama-gguf-split"));
    let args: Vec<_> = args.collect();
    let mut out = io::stdout().lock();
    let mut err = io::stderr().lock();
    let action = match parse(&args) {
        Ok(action) => action,
        Err(message) => {
            let _ = writeln!(err, "{message}");
            let _ = usage(&mut out, &executable);
            return ExitCode::FAILURE;
        }
    };
    let result: Result<(), Box<dyn std::error::Error>> = match action {
        Action::Help => usage(&mut out, &executable).map_err(Into::into),
        Action::Version => writeln!(
            err,
            "version: rust-llama {}\nnative Rust GGUF utility for {}-{}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::ARCH,
            std::env::consts::OS
        )
        .map_err(Into::into),
        Action::Run {
            operation,
            input,
            output,
            split,
            merge,
        } => {
            let limits = ShardLimits::default();
            match operation {
                Operation::Split => {
                    split::split(&input, &output, split, &limits, &mut out, &mut err)
                }
                Operation::Merge => split::merge(&input, &output, merge, &limits, &mut err),
            }
            .map(|_| ())
            .map_err(Into::into)
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(err, "error: {error}");
            ExitCode::FAILURE
        }
    }
}
