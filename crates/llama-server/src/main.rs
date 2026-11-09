use llama_inference::LoadOptions;
use llama_server::{
    EmbeddingOptions, InferenceProvider, Pooling, ProviderStatus, RuntimeProvider, Server,
    ServerConfig, UnavailableProvider,
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
    let mut embeddings = false;
    let mut embedding_options = EmbeddingOptions::default();
    let mut load_options = LoadOptions::default();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let arg = arg
            .to_str()
            .ok_or("Options must be Unicode; the --model path may use native OS encoding")?;
        match arg {
            "--help" | "-h" => {
                println!(
                    "Rust llama-server\n\nUsage: llama-server [--model PATH.gguf] [--bind 127.0.0.1:8080]\n                    [--api-key KEY] [--allow-origin ORIGIN]\n                    [--max-connections N] [--max-generations N]\n                    [--ctx-size N] [--max-weight-mib N]\n                    [--embeddings] [--pooling model|none|mean|cls|last]\n                    [--embd-normalize N] [--max-embedding-inputs N]\n\n--model / -m loads an explicitly supported GGUF with the Rust CPU runtime.\nLoad failures stop startup; unsupported model features are reported explicitly.\n--ctx-size / -c bounds the context (default 1024, capped at the training context).\n--max-weight-mib bounds the decoded F32 weight cache (default 2048 MiB).\nPacked weights and runtime state are additional allocations, outside that weight budget.\nWithout --model, health and generation report HTTP 503.\nGeneration supports temperature, explicit u32 seed, token limit and stop strings.\nAn omitted seed is 0; the ambient-random seed sentinel is not implemented.\n--embeddings enables embedding routes on supported models.\nPooled embeddings use normalization 2 (Euclidean) by default; unpooled vectors remain raw.\nNative routes preserve token-level vectors; /v1/embeddings requires pooling.\nThe server stays running until stdin receives 'quit' or the process exits.\nTLS and the full original protocol/CLI surface are not yet implemented."
                );
                return Ok(());
            }
            "--model" | "-m" => {
                if model_path.is_some() {
                    return Err("--model may be supplied only once".into());
                }
                model_path = Some(PathBuf::from(next_value(&mut args, arg)?));
            }
            "--ctx-size" | "-c" => {
                load_options.context_length = text_value(&mut args, arg)?.parse()?;
            }
            "--max-weight-mib" => {
                let mib: usize = text_value(&mut args, arg)?.parse()?;
                load_options.max_decoded_weight_bytes = mib
                    .checked_mul(1024 * 1024)
                    .ok_or("--max-weight-mib exceeds the addressable byte range")?;
            }
            "--embeddings" | "--embedding" => embeddings = true,
            "--pooling" => {
                let value = text_value(&mut args, arg)?;
                embedding_options.pooling = match value.as_str() {
                    "model" => None,
                    "none" => Some(Pooling::None),
                    "mean" => Some(Pooling::Mean),
                    "cls" => Some(Pooling::First),
                    "last" => Some(Pooling::Last),
                    _ => return Err("--pooling must be model, none, mean, cls, or last".into()),
                };
            }
            "--embd-normalize" => {
                embedding_options.normalization = text_value(&mut args, arg)?.parse()?;
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
            "--max-embedding-inputs" => {
                config.max_embedding_inputs = text_value(&mut args, arg)?.parse()?;
            }
            _ => return Err(format!("Unknown argument {arg}; use --help").into()),
        }
    }
    config.validate()?;
    if embeddings && model_path.is_none() {
        return Err("--embeddings requires --model".into());
    }
    let provider: Arc<dyn InferenceProvider> = match model_path {
        Some(path) => {
            let mut provider = RuntimeProvider::load_with_options(&path, &load_options)?;
            if embeddings {
                provider = provider.with_embeddings(embedding_options)?;
            }
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
