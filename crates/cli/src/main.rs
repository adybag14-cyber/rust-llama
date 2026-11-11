//! Native command line for the integrated Rust runtime.
#![forbid(unsafe_code)]
use llama_behavior::json::{self, Value};
use llama_gguf::Gguf;
use llama_inference::{GenerationOptions, LoadOptions, Model, Pooling};
use llama_tokenizer::Tokenizer;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::File,
    io::{self, Read, Write},
    path::PathBuf,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn bad(s: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, s.into()).into()
}
const HELP: &str = "rust-llama: native Rust GGUF inference\n\nCommands:\n  generate    Load a supported model and stream generated text (default)\n  inspect     Validate GGUF and print metadata/tensor descriptors as JSON\n  tokenize    Print token IDs and byte-exact pieces (supports vocabulary-only GGUF)\n  detokenize  Decode --tokens 1,2,3 as UTF-8 text\n  logits      Print full per-token logits for conformance\n  embed       Encode a dense BERT sequence and print raw embeddings as JSON\n\nOptions:\n  -m, --model PATH          GGUF model or vocabulary file (required)\n  -p, --prompt TEXT         Prompt text\n      --prompt-file PATH   Read a UTF-8 prompt, at most 4 MiB\n  -n, --n-predict N         Maximum generated tokens (default 128)\n      --temperature T      Zero is greedy (default 0)\n      --seed N             Explicit MT19937 u32 seed (default 0)\n      --stop TEXT          Stop string; repeatable\n  -c, --ctx-size N          Context bound (default 1024, capped to training limit)\n      --max-weight-mib N   Decoded F32 weight budget (default 2048)\n      --no-special         Do not insert special tokens in tokenize/logits/embed\n      --literal-special    Do not parse control strings in tokenize/logits/embed\n      --render-special     Render control tokens in detokenize\n      --tokens IDS         Comma-separated token IDs for detokenize/logits/embed\n      --pooling MODE       embed: model (default), none, mean, first, last\n  -h, --help               Show this help\n\nModel support: dense Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX and BLOOM plus Mamba-1 generation; dense BERT/WordPiece embeddings.\nPacked matrix arithmetic preserves source activation conversion; decoded F32 row caches are retained.\nOther model modes fail explicitly.\nGeneration text goes to stdout; timing/token statistics go to stderr.\n";
#[derive(Default)]
struct Args {
    command: String,
    values: BTreeMap<String, String>,
    flags: BTreeSet<String>,
    stops: Vec<String>,
}
impl Args {
    fn parse() -> Result<Self> {
        let mut raw = std::env::args().skip(1).peekable();
        let mut a = Args {
            command: "generate".into(),
            ..Self::default()
        };
        if raw.peek().is_some_and(|x| !x.starts_with('-')) {
            a.command = raw.next().unwrap();
        }
        while let Some(key) = raw.next() {
            let key = match key.as_str() {
                "-m" => "--model",
                "-p" => "--prompt",
                "-n" => "--n-predict",
                "-c" => "--ctx-size",
                "-h" => "--help",
                x => x,
            };
            if matches!(
                key,
                "--help" | "--no-special" | "--literal-special" | "--render-special"
            ) {
                if !a.flags.insert(key.into()) {
                    return Err(bad(format!("duplicate flag {key}")));
                }
                continue;
            }
            if !matches!(
                key,
                "--model"
                    | "--prompt"
                    | "--prompt-file"
                    | "--n-predict"
                    | "--temperature"
                    | "--seed"
                    | "--stop"
                    | "--ctx-size"
                    | "--max-weight-mib"
                    | "--tokens"
                    | "--pooling"
            ) {
                return Err(bad(format!("unknown option {key}")));
            }
            let value = raw
                .next()
                .ok_or_else(|| bad(format!("missing value for {key}")))?;
            if key == "--stop" {
                a.stops.push(value)
            } else if a.values.insert(key.into(), value).is_some() {
                return Err(bad(format!("duplicate option {key}")));
            }
        }
        Ok(a)
    }
    fn value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
    fn number<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        self.value(key).map_or(Ok(default), |s| {
            s.parse().map_err(|_| bad(format!("invalid {key}: {s}")))
        })
    }
    fn prompt(&self) -> Result<String> {
        if self.value("--prompt").is_some() && self.value("--prompt-file").is_some() {
            return Err(bad("choose --prompt or --prompt-file"));
        }
        if let Some(path) = self.value("--prompt-file") {
            let mut s = String::new();
            File::open(path)?
                .take(4 * 1024 * 1024 + 1)
                .read_to_string(&mut s)?;
            if s.len() > 4 * 1024 * 1024 {
                return Err(bad("prompt file exceeds 4 MiB"));
            }
            Ok(s)
        } else {
            Ok(self.value("--prompt").unwrap_or("").into())
        }
    }
    fn tokens(&self, t: &Tokenizer) -> Result<Vec<i32>> {
        if let Some(ids) = self.value("--tokens") {
            if self.value("--prompt").is_some() || self.value("--prompt-file").is_some() {
                return Err(bad("choose --tokens or a prompt"));
            }
            if ids.is_empty() {
                return Ok(vec![]);
            }
            ids.split(',')
                .map(|id| {
                    id.trim()
                        .parse()
                        .map_err(|_| bad(format!("invalid token ID {id}")))
                })
                .collect()
        } else {
            Ok(t.tokenize(
                &self.prompt()?,
                !self.flags.contains("--no-special"),
                !self.flags.contains("--literal-special"),
            )?)
        }
    }
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| {
            [
                DIGITS[(b >> 4) as usize] as char,
                DIGITS[(b & 15) as usize] as char,
            ]
        })
        .collect()
}
fn print_json(value: &Value) -> Result<()> {
    println!("{}", json::stringify(value)?);
    Ok(())
}
fn metadata(v: &llama_gguf::Value) -> Value {
    use llama_gguf::Value as G;
    match v {
        G::String(x) => Value::String(x.clone()),
        G::ByteString(x) => Value::object([
            ("gguf_type", Value::String("STRING".into())),
            ("bytes_hex", Value::String(hex(x))),
        ]),
        G::Bool(x) => Value::Bool(*x),
        G::Array {
            element_type,
            values,
        } => Value::object([
            ("element_type", Value::from(*element_type)),
            ("count", Value::from(values.len() as u64)),
        ]),
        G::F32(x) => {
            if x.is_finite() {
                Value::Number(*x as f64)
            } else {
                Value::String(x.to_string())
            }
        }
        G::F64(x) => {
            if x.is_finite() {
                Value::Number(*x)
            } else {
                Value::String(x.to_string())
            }
        }
        G::U8(_) | G::U16(_) | G::U32(_) | G::U64(_) => Value::from(v.as_u64().unwrap()),
        _ => Value::from(v.as_i64().unwrap()),
    }
}
fn run() -> Result<()> {
    let a = Args::parse()?;
    if a.flags.contains("--help") {
        print!("{HELP}");
        return Ok(());
    }
    if a.command != "embed" && a.value("--pooling").is_some() {
        return Err(bad("--pooling is only supported by embed"));
    }
    if a.command == "embed" {
        if a.flags.contains("--render-special") {
            return Err(bad("--render-special is only supported by detokenize"));
        }
        if ["--n-predict", "--temperature", "--seed"]
            .iter()
            .any(|key| a.value(key).is_some())
            || !a.stops.is_empty()
        {
            return Err(bad("generation options are not supported by embed"));
        }
    }
    let path = PathBuf::from(
        a.value("--model")
            .ok_or_else(|| bad("--model is required; use --help"))?,
    );
    match a.command.as_str() {
        "inspect" => {
            let g = Gguf::open(path)?;
            let tensors = g
                .tensors
                .iter()
                .map(|t| {
                    Value::object([
                        ("name", Value::from(t.name.clone())),
                        ("dtype_id", Value::from(t.dtype.id())),
                        ("dtype", Value::from(format!("{:?}", t.dtype))),
                        (
                            "shape",
                            Value::Array(t.dimensions.iter().map(|&n| Value::from(n)).collect()),
                        ),
                        ("offset", Value::from(t.offset)),
                        ("bytes", Value::from(t.byte_len)),
                    ])
                })
                .collect();
            print_json(&Value::object([
                ("version", Value::from(g.version)),
                ("endian", Value::from(format!("{:?}", g.endian))),
                ("alignment", Value::from(g.alignment)),
                (
                    "metadata",
                    Value::Object(
                        g.metadata
                            .iter()
                            .map(|(k, v)| (k.clone(), metadata(v)))
                            .collect(),
                    ),
                ),
                ("tensors", Value::Array(tensors)),
            ]))
        }
        "tokenize" | "detokenize" => {
            let g = Gguf::open(path)?;
            let t = Tokenizer::from_metadata(&g.metadata)?;
            let ids = a.tokens(&t)?;
            if a.command == "detokenize" {
                if a.value("--tokens").is_none() {
                    return Err(bad("detokenize requires --tokens"));
                }
                let bytes = t.detokenize_bytes(&ids, true, a.flags.contains("--render-special"))?;
                io::stdout().write_all(&bytes)?;
                Ok(())
            } else {
                let pieces: Vec<_> = ids
                    .iter()
                    .map(|&id| t.token_piece(id, false).map(|v| Value::from(hex(&v))))
                    .collect::<std::result::Result<_, _>>()?;
                print_json(&Value::object([
                    (
                        "tokens",
                        Value::Array(ids.iter().map(|&x| Value::from(x)).collect()),
                    ),
                    ("pieces_hex", Value::Array(pieces)),
                    (
                        "detokenized_hex",
                        Value::from(hex(&t.detokenize_bytes(&ids, true, false)?)),
                    ),
                ]))
            }
        }
        "generate" | "logits" | "embed" => {
            let budget: usize = a.number("--max-weight-mib", 2048usize)?;
            let options = LoadOptions {
                max_decoded_weight_bytes: budget
                    .checked_mul(1024 * 1024)
                    .ok_or_else(|| bad("weight budget overflow"))?,
                context_length: a.number("--ctx-size", 1024)?,
            };
            let start = Instant::now();
            let model = Model::load_with_options(path, &options)?;
            let load_seconds = start.elapsed().as_secs_f64();
            if a.command == "embed" {
                let pooling = match a.value("--pooling").unwrap_or("model") {
                    "model" => model.default_embedding_pooling().unwrap_or(Pooling::None),
                    "none" => Pooling::None,
                    "mean" => Pooling::Mean,
                    "first" => Pooling::First,
                    "last" => Pooling::Last,
                    other => return Err(bad(format!("unknown pooling mode {other}"))),
                };
                let tokens = a.tokens(model.tokenizer())?;
                let encoded = model.embed_tokens(&tokens, pooling, &|| false)?;
                let floats = |values: Vec<f32>| {
                    Value::Array(
                        values
                            .into_iter()
                            .map(|v| Value::Number(v as f64))
                            .collect(),
                    )
                };
                return print_json(&Value::object([
                    ("model", Value::from(model.model_id())),
                    (
                        "tokens",
                        Value::Array(tokens.into_iter().map(Value::from).collect()),
                    ),
                    (
                        "dimensions",
                        Value::from(model.embedding_dimensions() as u64),
                    ),
                    (
                        "pooling",
                        Value::from(format!("{pooling:?}").to_lowercase()),
                    ),
                    ("normalized", Value::Bool(false)),
                    (
                        "embeddings",
                        Value::Array(encoded.tokens.into_iter().map(floats).collect()),
                    ),
                    ("pooled", encoded.pooled.map(floats).unwrap_or(Value::Null)),
                ]));
            }
            if a.command == "logits" {
                let tokens = a.tokens(model.tokenizer())?;
                let logits = model.logits_for_tokens(&tokens)?;
                return print_json(&Value::object([
                    (
                        "tokens",
                        Value::Array(tokens.into_iter().map(Value::from).collect()),
                    ),
                    (
                        "logits",
                        Value::Array(
                            logits
                                .into_iter()
                                .map(|v| {
                                    Value::Array(
                                        v.into_iter().map(|x| Value::Number(x as f64)).collect(),
                                    )
                                })
                                .collect(),
                        ),
                    ),
                ]));
            }
            if a.value("--tokens").is_some()
                || a.flags.contains("--no-special")
                || a.flags.contains("--literal-special")
            {
                return Err(bad(
                    "generation currently uses model BOS/EOS rules and parses special strings",
                ));
            }
            let options = GenerationOptions {
                max_tokens: a.number("--n-predict", 128)?,
                temperature: a.number("--temperature", 0.)?,
                seed: a.number("--seed", 0)?,
                stop: a.stops.clone(),
            };
            let start = Instant::now();
            let mut output = io::stdout().lock();
            let mut output_error = None;
            let stats = model.generate(&a.prompt()?, &options, &|| false, &mut |text| match output
                .write_all(text.as_bytes())
                .and_then(|_| output.flush())
            {
                Ok(()) => true,
                Err(e) => {
                    output_error = Some(e);
                    false
                }
            });
            if let Some(e) = output_error {
                return Err(e.into());
            }
            let stats = stats?;
            eprintln!(
                "{}",
                json::stringify(&Value::object([
                    ("model", Value::from(model.model_id())),
                    ("prompt_tokens", Value::from(stats.prompt_tokens as u64)),
                    (
                        "completion_tokens",
                        Value::from(stats.completion_tokens as u64)
                    ),
                    (
                        "finish_reason",
                        Value::from(format!("{:?}", stats.finish_reason))
                    ),
                    ("load_seconds", Value::Number(load_seconds)),
                    (
                        "generation_seconds",
                        Value::Number(start.elapsed().as_secs_f64())
                    ),
                    (
                        "decoded_weight_bytes",
                        Value::from(model.decoded_weight_bytes() as u64)
                    )
                ]))?
            );
            Ok(())
        }
        _ => Err(bad(format!("unknown command {}; use --help", a.command))),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("rust-llama: {e}");
        std::process::exit(1)
    }
}
