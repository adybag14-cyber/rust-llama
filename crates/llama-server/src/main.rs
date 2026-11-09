use llama_server::{
    InferenceProvider, ProviderStatus, RuntimeProvider, Server, ServerConfig, UnavailableProvider,
};
use std::ffi::OsString;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    if let Err(error) = run() {
        eprintln!("llama-server: {error}");
        std::process::exit(1);
    }
}

fn next_value(
    args: &mut impl Iterator<Item = OsString>,
    flag: &str,
) -> Result<OsString, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} needs a value").into())
}

fn text_value(
    args: &mut impl Iterator<Item = OsString>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    next_value(args, flag)?
        .into_string()
        .map_err(|_| format!("{flag} requires valid Unicode text").into())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ServerConfig::default();
    let mut model_path: Option<PathBuf> = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let arg = arg
            .to_str()
            .ok_or("Options must be Unicode; the --model path may use native OS encoding")?;
        match arg {
            "--help" | "-h" => {
                println!(
                    "Rust llama-server\n\nUsage: llama-server [--model PATH.gguf] [--bind 127.0.0.1:8080]\n                    [--api-key KEY] [--allow-origin ORIGIN]\n                    [--max-connections N] [--max-generations N]\n\n--model / -m loads an explicitly supported GGUF with the Rust CPU runtime.\nLoad failures stop startup; unsupported model features are reported explicitly.\nWithout --model, health and generation report HTTP 503.\nGeneration supports temperature, explicit u32 seed, token limit and stop strings.\nAn omitted seed is 0; the ambient-random seed sentinel is not implemented.\nThe server stays running until stdin receives 'quit' or the process exits.\nTLS and the full original protocol/CLI surface are not yet implemented."
                );
                return Ok(());
            }
            "--model" | "-m" => {
                if model_path.is_some() {
                    return Err("--model may be supplied only once".into());
                }
                model_path = Some(PathBuf::from(next_value(&mut args, arg)?));
            }
            "--bind" => config.bind = text_value(&mut args, arg)?.parse()?,
            "--api-key" => config.api_keys.push(text_value(&mut args, arg)?),
            "--allow-origin" => config.allowed_origins.push(text_value(&mut args, arg)?),
            "--max-connections" => {
                config.max_connections = text_value(&mut args, arg)?.parse()?;
            }
            "--max-generations" => {
                config.max_generations = text_value(&mut args, arg)?.parse()?;
            }
            _ => return Err(format!("Unknown argument {arg}; use --help").into()),
        }
    }
    config.validate()?;
    let provider: Arc<dyn InferenceProvider> = match model_path {
        Some(path) => {
            let provider = RuntimeProvider::load(&path)?;
            eprintln!("Loaded Rust model: {}", provider.model_id());
            Arc::new(provider)
        }
        None => Arc::new(UnavailableProvider),
    };
    let status = match provider.status() {
        ProviderStatus::Ready => "Rust model runtime ready",
        ProviderStatus::Unavailable(_) => "no model runtime connected",
    };
    let server = Server::new(config, provider)?.spawn()?;
    println!("Listening on http://{}; {status}", server.local_addr());
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            match line {
                Ok(line) if line.trim() == "quit" => {
                    let _ = sender.send(());
                    return;
                }
                Err(_) => return,
                _ => {}
            }
        }
        // EOF is normal when launched as a daemon. Do not terminate the listener.
        std::mem::forget(sender);
    });
    let _ = receiver.recv();
    server.shutdown()?;
    Ok(())
}
