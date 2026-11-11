//! Native Rust entry point for the frozen quantize CLI contract.
#![forbid(unsafe_code)]
use llama_model_quant::cli::{CliLimits, run};
fn main() {
    let limits = CliLimits::default();
    let mut args = Vec::new();
    let mut bytes = 0usize;
    for arg in std::env::args_os().skip(1) {
        bytes = match bytes.checked_add(arg.len()) {
            Some(n) if n <= limits.max_argument_bytes => n,
            _ => {
                eprintln!("llama-quantize: argument byte limit");
                std::process::exit(1);
            }
        };
        if args.len() >= limits.max_arguments {
            eprintln!("llama-quantize: argument count limit");
            std::process::exit(1);
        }
        match arg.into_string() {
            Ok(arg) => args.push(arg),
            Err(_) => {
                eprintln!("llama-quantize: arguments must be Unicode");
                std::process::exit(1);
            }
        }
    }
    std::process::exit(run(
        &args,
        &limits,
        std::io::stdout().lock(),
        std::io::stderr().lock(),
    ));
}
