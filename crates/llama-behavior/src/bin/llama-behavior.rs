//! Native CLI for the behavior capabilities implemented in this checkpoint.
use llama_behavior::{
    Error, Result, chat,
    grammar::Grammar,
    json::{self, Value},
    sampling::{Candidates, Chain, Config},
    speculative,
};
use std::io::{self, Read, Write};
use std::path::Path;

const HELP: &str = "llama-behavior 0.1.0 (T10 progress checkpoint)\n\nUsage:\n  llama-behavior templates\n  llama-behavior chat TEMPLATE [--no-assistant] [--input FILE]\n  llama-behavior sample [FILE|-]\n  llama-behavior grammar GRAMMAR_FILE [INPUT_FILE|-] [--prefix]\n  llama-behavior json [FILE|-]\n  llama-behavior speculative [FILE|-]\n\nchat reads a JSON array of {role,content}. sample reads {logits,samplers,history?}.\nspeculative reads {draft,target}; target includes one bonus prediction.\nWithout FILE, input comes from stdin. Inputs are bounded to 16 MiB.\nExit: 0 success, 1 grammar mismatch, 2 invalid input/unsupported operation.\nThis checkpoint does not load models or perform model generation.\n";

fn read(path: Option<&str>) -> Result<String> {
    let limit = 16 * 1024 * 1024;
    let mut bytes = Vec::new();
    let mut reader: Box<dyn Read> = match path {
        Some(path) if path != "-" => Box::new(
            std::fs::File::open(Path::new(path))
                .map_err(|e| Error::new(format!("cannot open {path}: {e}")))?,
        ),
        _ => Box::new(io::stdin()),
    };
    reader
        .by_ref()
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::new(format!("input read failed: {e}")))?;
    if bytes.len() > limit {
        return Err(Error::new("input exceeds 16 MiB limit"));
    }
    String::from_utf8(bytes).map_err(|_| Error::new("input is not valid UTF-8"))
}
fn write(text: &str) -> Result<()> {
    io::stdout()
        .write_all(text.as_bytes())
        .map_err(|e| Error::new(format!("output write failed: {e}")))
}
fn write_json(value: Value) -> Result<()> {
    write(&(json::stringify(&value)? + "\n"))
}
fn array<'a>(value: &'a Value, key: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new(format!("{key} must be an array")))
}
fn tokens(value: &Value, key: &str) -> Result<Vec<i32>> {
    array(value, key)?
        .iter()
        .map(|x| {
            x.as_i64()
                .and_then(|x| i32::try_from(x).ok())
                .filter(|x| *x >= 0)
                .ok_or_else(|| Error::new("token IDs must be non-negative i32 integers"))
        })
        .collect()
}
fn single_path(args: &[String]) -> Result<Option<&str>> {
    if args.len() > 1 {
        return Err(Error::new("unexpected additional argument"));
    }
    Ok(args.first().map(String::as_str))
}

fn run(args: &[String]) -> Result<u8> {
    let Some(command) = args.first().map(String::as_str) else {
        write(HELP)?;
        return Ok(0);
    };
    match command {
        "--help" | "-h" | "help" => {
            write(HELP)?;
        }
        "--version" => {
            write("llama-behavior 0.1.0\n")?;
        }
        "templates" => {
            if args.len() != 1 {
                return Err(Error::new("templates takes no additional arguments"));
            }
            let mut names = chat::BUILTIN_TEMPLATES.to_vec();
            names.sort_unstable();
            for name in names {
                write(name)?;
                write("\n")?;
            }
        }
        "chat" => {
            let template = args
                .get(1)
                .ok_or_else(|| Error::new("chat requires a template name"))?;
            let mut path = None;
            let mut add = true;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--no-assistant" => {
                        if !add {
                            return Err(Error::new("duplicate --no-assistant"));
                        }
                        add = false;
                    }
                    "--input" => {
                        if path.is_some() {
                            return Err(Error::new("duplicate --input"));
                        }
                        i += 1;
                        path = Some(
                            args.get(i)
                                .ok_or_else(|| Error::new("--input requires a path"))?
                                .as_str(),
                        );
                    }
                    unknown => return Err(Error::new(format!("unknown chat argument {unknown}"))),
                }
                i += 1;
            }
            let input = json::parse(&read(path)?)?;
            let rows = input
                .as_array()
                .ok_or_else(|| Error::new("chat input must be a message array"))?;
            let mut messages = Vec::new();
            for row in rows {
                let role = row
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::new("each message requires a string role"))?;
                let content = row
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::new("each message requires string content"))?;
                messages.push(chat::Message::new(role, content));
            }
            write(&chat::render(template, &messages, add)?)?;
        }
        "sample" => {
            let value = json::parse(&read(single_path(&args[1..])?)?)?;
            let configs = array(&value, "samplers")?
                .iter()
                .map(Config::from_json)
                .collect::<Result<Vec<_>>>()?;
            let logits = array(&value, "logits")?
                .iter()
                .map(|x| {
                    if x.as_str() == Some("-inf") {
                        return Ok(f32::NEG_INFINITY);
                    }
                    let v = x
                        .as_f64()
                        .ok_or_else(|| Error::new("logit must be numeric or the string -inf"))?;
                    if !v.is_finite() || v.abs() > f32::MAX as f64 {
                        return Err(Error::new("logit outside finite f32 range"));
                    }
                    Ok(v as f32)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut chain = Chain::new(configs)?;
            if value.get("history").is_some() {
                for token in tokens(&value, "history")? {
                    chain.accept(token)?;
                }
            }
            let mut candidates = Candidates::from_logits(&logits)?;
            chain.apply(&mut candidates)?;
            let values = candidates
                .data
                .iter()
                .map(|t| {
                    Value::object([
                        ("id", Value::from(t.id)),
                        (
                            "logit",
                            if t.logit == f32::NEG_INFINITY {
                                Value::from("-inf")
                            } else {
                                Value::Number(t.logit as f64)
                            },
                        ),
                        ("p", Value::Number(t.p as f64)),
                    ])
                })
                .collect();
            write_json(Value::object([
                (
                    "token",
                    candidates
                        .selected_token()
                        .map(Value::from)
                        .unwrap_or(Value::Null),
                ),
                (
                    "selected",
                    candidates
                        .selected
                        .map(|i| Value::from(i as u64))
                        .unwrap_or(Value::Null),
                ),
                ("sorted", Value::Bool(candidates.sorted)),
                ("candidates", Value::Array(values)),
            ]))?;
        }
        "grammar" => {
            let grammar_path = args
                .get(1)
                .ok_or_else(|| Error::new("grammar requires a GBNF file"))?;
            let mut path = None;
            let mut prefix = false;
            for arg in &args[2..] {
                if arg == "--prefix" {
                    if prefix {
                        return Err(Error::new("duplicate --prefix"));
                    }
                    prefix = true;
                } else if path.is_none() {
                    path = Some(arg.as_str());
                } else {
                    return Err(Error::new("unexpected grammar argument"));
                }
            }
            let grammar = Grammar::parse(&read(Some(grammar_path))?)?;
            let text = read(path)?;
            let mut r = grammar.recognizer()?;
            let allowed = r.accept_bytes(text.as_bytes())?;
            let complete = allowed && r.can_end();
            let accepted = if prefix { allowed } else { complete };
            write_json(Value::object([
                ("accepted", Value::Bool(accepted)),
                ("complete", Value::Bool(complete)),
                ("prefix_allowed", Value::Bool(allowed)),
            ]))?;
            return Ok(if accepted { 0 } else { 1 });
        }
        "json" => {
            write_json(json::parse(&read(single_path(&args[1..])?)?)?)?;
        }
        "speculative" => {
            let value = json::parse(&read(single_path(&args[1..])?)?)?;
            let accepted =
                speculative::verify_greedy(&tokens(&value, "draft")?, &tokens(&value, "target")?)?;
            write_json(Value::object([
                (
                    "accepted_draft",
                    Value::from(accepted.accepted_draft as u64),
                ),
                (
                    "rejected_draft",
                    Value::from(accepted.rejected_draft as u64),
                ),
                (
                    "emitted",
                    Value::Array(accepted.emitted.into_iter().map(Value::from).collect()),
                ),
            ]))?;
        }
        other => {
            return Err(Error::new(format!(
                "unsupported command {other}; see --help"
            )));
        }
    }
    Ok(0)
}
fn main() -> std::process::ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run(&args) {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            eprintln!("llama-behavior: {error}");
            std::process::ExitCode::from(2)
        }
    }
}
