//! The frozen `src/llama-chat.cpp` native string formatter and its template
//! detection heuristics. This is deliberately separate from the still-pending
//! full Jinja interpreter in `common/jinja`.

use crate::{Error, Result};

pub const BUILTIN_TEMPLATES: &[&str] = &[
    "chatml",
    "llama2",
    "llama2-sys",
    "llama2-sys-bos",
    "llama2-sys-strip",
    "mistral-v1",
    "mistral-v3",
    "mistral-v3-tekken",
    "mistral-v7",
    "mistral-v7-tekken",
    "phi3",
    "phi4",
    "falcon3",
    "zephyr",
    "monarch",
    "gemma",
    "orion",
    "openchat",
    "vicuna",
    "vicuna-orca",
    "deepseek",
    "deepseek2",
    "deepseek3",
    "deepseek-ocr",
    "command-r",
    "llama3",
    "chatglm3",
    "chatglm4",
    "glmedge",
    "minicpm",
    "exaone3",
    "exaone4",
    "exaone-moe",
    "rwkv-world",
    "granite",
    "granite-4.0",
    "granite-4.1",
    "gigachat",
    "megrez",
    "yandex",
    "bailing",
    "bailing-think",
    "bailing2",
    "llama4",
    "smolvlm",
    "hunyuan-moe",
    "gpt-oss",
    "hunyuan-dense",
    "hunyuan-vl",
    "kimi-k2",
    "seed_oss",
    "grok-2",
    "pangu-embedded",
    "solar-open",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Borrowed byte strings for native chat formatting. Embedded NULs are rejected;
/// C callers pass only the bytes preceding their first terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteMessage<'a> {
    pub role: &'a [u8],
    pub content: &'a [u8],
}

/// UTF-8 convenience entry point for the same byte-oriented detection rules.
pub fn detect_template(template: &str) -> Option<&'static str> {
    detect_template_bytes(template.as_bytes())
}

/// Same ordered heuristics as the original native formatter. The returned
/// `dots1` tag is an internal dispatch tag, not a newly registered alias.
pub fn detect_template_bytes(template: &[u8]) -> Option<&'static str> {
    if let Some(&name) = BUILTIN_TEMPLATES
        .iter()
        .find(|&&name| name.as_bytes() == template)
    {
        return Some(name);
    }
    let has = |s: &str| template.windows(s.len()).any(|part| part == s.as_bytes());
    Some(if has("<|im_start|>") {
        if has("<|im_sep|>") {
            "phi4"
        } else if has("<end_of_utterance>") {
            "smolvlm"
        } else {
            "chatml"
        }
    } else if template.starts_with(b"mistral") || has("[INST]") {
        if has("[SYSTEM_PROMPT]") {
            "mistral-v7"
        } else if has("' [INST] ' + system_message") || has("[AVAILABLE_TOOLS]") {
            if has(" [INST]") {
                "mistral-v1"
            } else if has("\"[INST]\"") {
                "mistral-v3-tekken"
            } else {
                "mistral-v3"
            }
        } else if has("content.strip()") {
            "llama2-sys-strip"
        } else if has("bos_token + '[INST]") {
            "llama2-sys-bos"
        } else if has("<<SYS>>") {
            "llama2-sys"
        } else {
            "llama2"
        }
    } else if has("<|assistant|>") && has("<|end|>") {
        "phi3"
    } else if has("[gMASK]<sop>") {
        "chatglm4"
    } else if has("<|assistant|>") && has("<|user|>") {
        if has("<|tool_declare|>") {
            "exaone-moe"
        } else if has("</s>") {
            "falcon3"
        } else {
            "glmedge"
        }
    } else if has("<|{{ item['role'] }}|>") && has("<|begin_of_image|>") {
        "glmedge"
    } else if has("<|user|>") && has("<|endoftext|>") {
        "zephyr"
    } else if has("bos_token + message['role']") {
        "monarch"
    } else if has("<start_of_turn>") {
        "gemma"
    } else if has("'\\n\\nAssistant: ' + eos_token") {
        "orion"
    } else if has("GPT4 Correct ") {
        "openchat"
    } else if has("USER: ") && has("ASSISTANT: ") {
        if has("SYSTEM: ") {
            "vicuna-orca"
        } else {
            "vicuna"
        }
    } else if has("### Instruction:") && has("<|EOT|>") {
        "deepseek"
    } else if has("<|START_OF_TURN_TOKEN|>") && has("<|USER_TOKEN|>") {
        "command-r"
    } else if has("<|start_header_id|>") && has("<|end_header_id|>") {
        "llama3"
    } else if has("[gMASK]sop") {
        "chatglm3"
    } else if has("<用户>") {
        "minicpm"
    } else if has("'Assistant: ' + message['content'] + eos_token") {
        "deepseek2"
    } else if has("<｜Assistant｜>") && has("<｜User｜>") && has("<｜end▁of▁sentence｜>")
    {
        "deepseek3"
    } else if has("[|system|]") && has("[|assistant|]") && has("[|endofturn|]") {
        if has("[|tool|]") {
            "exaone4"
        } else {
            "exaone3"
        }
    } else if has("rwkv-world") || has("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}") {
        "rwkv-world"
    } else if has("<|start_of_role|>") {
        if has("<tool_call>") || has("<tools>") {
            if has("g4_default_system_message") {
                "granite-4.0"
            } else {
                "granite-4.1"
            }
        } else {
            "granite"
        }
    } else if has(
        "message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]",
    ) {
        "gigachat"
    } else if has("<|role_start|>") {
        "megrez"
    } else if has(" Ассистент:") {
        "yandex"
    } else if has("<role>ASSISTANT</role>") && has("'HUMAN'") {
        "bailing"
    } else if has("<role>ASSISTANT</role>") && has("\"HUMAN\"") && has("<think>") {
        "bailing-think"
    } else if has("<role>ASSISTANT</role>") && has("<role>HUMAN</role>") && has("<|role_end|>") {
        "bailing2"
    } else if has("<|header_start|>") && has("<|header_end|>") {
        "llama4"
    } else if has("<|endofuserprompt|>") {
        "dots1"
    } else if has("<|extra_0|>") && has("<|extra_4|>") {
        "hunyuan-moe"
    } else if has("<|start|>") && has("<|channel|>") {
        "gpt-oss"
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_begin▁of▁sentence｜>") {
        "hunyuan-vl"
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_place▁holder▁no▁3｜>") {
        "hunyuan-dense"
    } else if has("<|im_assistant|>assistant<|im_middle|>") {
        "kimi-k2"
    } else if has("<seed:bos>") {
        "seed_oss"
    } else if has("'Assistant: '  + message['content'] + '<|separator|>") {
        "grok-2"
    } else if has("[unused9]系统：[unused10]") {
        "pangu-embedded"
    } else if has("<|begin|>") && has("<|end|>") && has("<|content|>") {
        "solar-open"
    } else {
        return None;
    })
}

fn trim(value: &[u8]) -> &[u8] {
    let whitespace = |b: &u8| matches!(*b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c);
    let start = value
        .iter()
        .position(|b| !whitespace(b))
        .unwrap_or(value.len());
    let end = value
        .iter()
        .rposition(|b| !whitespace(b))
        .map_or(start, |i| i + 1);
    &value[start..end]
}

fn copy_bytes(value: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    append_bytes(&mut out, value)?;
    Ok(out)
}
fn append_bytes(target: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    target
        .try_reserve(value.len())
        .map_err(|_| Error::new("chat byte allocation failed"))?;
    target.extend_from_slice(value);
    Ok(())
}
fn uppercase(value: &[u8]) -> Result<Vec<u8>> {
    let mut out = copy_bytes(value)?;
    out.make_ascii_uppercase();
    Ok(out)
}

struct Output {
    value: Vec<u8>,
    limit: usize,
}
impl Output {
    fn add(&mut self, parts: &[&[u8]]) -> Result<()> {
        let total = parts
            .iter()
            .try_fold(self.value.len(), |total, p| total.checked_add(p.len()))
            .ok_or_else(|| Error::new("chat output size overflow"))?;
        if total > self.limit {
            return Err(Error::new("chat output exceeds byte limit"));
        }
        self.value
            .try_reserve(total - self.value.len())
            .map_err(|_| Error::new("chat output allocation failed"))?;
        for part in parts {
            self.value.extend_from_slice(part);
        }
        Ok(())
    }
}

pub fn render(template: &str, messages: &[Message], add_assistant: bool) -> Result<String> {
    render_with_limit(template, messages, add_assistant, 16 * 1024 * 1024)
}

/// Preserve the safe UTF-8 API's empty-role/NUL validation and ASCII case policy.
pub fn render_with_limit(
    template: &str,
    messages: &[Message],
    add_assistant: bool,
    max_bytes: usize,
) -> Result<String> {
    if messages.len() > 65_536 {
        return Err(Error::new("chat message count exceeds limit"));
    }
    let mut borrowed = Vec::new();
    borrowed
        .try_reserve_exact(messages.len())
        .map_err(|_| Error::new("chat message allocation failed"))?;
    for message in messages {
        if message.role.is_empty() {
            return Err(Error::new("empty chat role is invalid"));
        }
        borrowed.push(ByteMessage {
            role: message.role.as_bytes(),
            content: message.content.as_bytes(),
        });
    }
    String::from_utf8(render_bytes_with_limit(
        template.as_bytes(),
        &borrowed,
        add_assistant,
        max_bytes,
    )?)
    .map_err(|_| Error::new("UTF-8 chat formatter produced invalid text"))
}

/// Byte-preserving native formatter under the verified C locale. Empty roles
/// are retained. Case conversion is ASCII-only; the C ABI separately rejects
/// source-undefined signed-negative arguments to toupper.
pub fn render_bytes(
    template: &[u8],
    messages: &[ByteMessage<'_>],
    add_assistant: bool,
) -> Result<Vec<u8>> {
    render_bytes_with_limit(template, messages, add_assistant, 16 * 1024 * 1024)
}

/// Format byte strings with a checked output and per-message byte ceiling.
/// No source C/C++ code or foreign formatter is linked by this implementation.
pub fn render_bytes_with_limit(
    template: &[u8],
    messages: &[ByteMessage<'_>],
    add_assistant: bool,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    if template.len() > 1024 * 1024 {
        return Err(Error::new("template exceeds 1 MiB limit"));
    }
    let name = detect_template_bytes(template).ok_or_else(|| {
        Error::new("unrecognized native chat template; full Jinja evaluation is not implemented")
    })?;
    if messages.len() > 65_536 {
        return Err(Error::new("chat message count exceeds limit"));
    }
    for message in messages {
        if message.role.contains(&0) || message.content.contains(&0) {
            return Err(Error::new(
                "NUL in chat message cannot be represented by the source C-string contract",
            ));
        }
        if message.role.len() > max_bytes || message.content.len() > max_bytes {
            return Err(Error::new("chat message exceeds byte limit"));
        }
    }
    let mut out = Output {
        value: Vec::new(),
        limit: max_bytes,
    };
    match name {
        "chatml" | "phi4" => {
            let separator = if name == "phi4" {
                "<|im_sep|>".as_bytes()
            } else {
                "\n".as_bytes()
            };
            let suffix = if name == "phi4" {
                "<|im_end|>".as_bytes()
            } else {
                "<|im_end|>\n".as_bytes()
            };
            for m in messages {
                out.add(&[
                    "<|im_start|>".as_bytes(),
                    m.role,
                    separator,
                    m.content,
                    suffix,
                ])?;
            }
            if add_assistant {
                out.add(&["<|im_start|>assistant".as_bytes(), separator])?;
            }
        }
        "mistral-v7" | "mistral-v7-tekken" => {
            let space = if name == "mistral-v7" {
                " ".as_bytes()
            } else {
                "".as_bytes()
            };
            for m in messages {
                match m.role {
                    b"system" => out.add(&[
                        "[SYSTEM_PROMPT]".as_bytes(),
                        space,
                        m.content,
                        "[/SYSTEM_PROMPT]".as_bytes(),
                    ])?,
                    b"user" => {
                        out.add(&["[INST]".as_bytes(), space, m.content, "[/INST]".as_bytes()])?
                    }
                    _ => out.add(&[space, m.content, "</s>".as_bytes()])?,
                }
            }
        }
        "mistral-v1" | "mistral-v3" | "mistral-v3-tekken" => {
            let leading = if name == "mistral-v1" {
                " ".as_bytes()
            } else {
                "".as_bytes()
            };
            let trailing = if name == "mistral-v3-tekken" {
                "".as_bytes()
            } else {
                " ".as_bytes()
            };
            let mut inside = false;
            for m in messages {
                if !inside {
                    out.add(&[leading, "[INST]".as_bytes(), trailing])?;
                    inside = true;
                }
                match m.role {
                    b"system" => out.add(&[m.content, "\n\n".as_bytes()])?,
                    b"user" => out.add(&[m.content, leading, "[/INST]".as_bytes()])?,
                    _ => {
                        out.add(&[
                            trailing,
                            if name == "mistral-v3" {
                                trim(m.content)
                            } else {
                                m.content
                            },
                            "</s>".as_bytes(),
                        ])?;
                        inside = false;
                    }
                }
            }
        }
        "llama2" | "llama2-sys" | "llama2-sys-bos" | "llama2-sys-strip" => {
            let mut inside = true;
            out.add(&["[INST] ".as_bytes()])?;
            for m in messages {
                let content = if name == "llama2-sys-strip" {
                    trim(m.content)
                } else {
                    m.content
                };
                if !inside {
                    out.add(&[if name == "llama2-sys-bos" {
                        "<s>[INST] ".as_bytes()
                    } else {
                        "[INST] ".as_bytes()
                    }])?;
                    inside = true;
                }
                match m.role {
                    b"system" => {
                        if name != "llama2" {
                            out.add(&[
                                "<<SYS>>\n".as_bytes(),
                                content,
                                "\n<</SYS>>\n\n".as_bytes(),
                            ])?;
                        } else {
                            out.add(&[content, "\n".as_bytes()])?;
                        }
                    }
                    b"user" => out.add(&[content, " [/INST]".as_bytes()])?,
                    _ => {
                        out.add(&[content, "</s>".as_bytes()])?;
                        inside = false;
                    }
                }
            }
        }
        "phi3" | "falcon3" | "zephyr" | "glmedge" | "chatglm3" | "chatglm4" => {
            if name == "chatglm3" {
                out.add(&["[gMASK]sop".as_bytes()])?;
            }
            if name == "chatglm4" {
                out.add(&["[gMASK]<sop>".as_bytes()])?;
            }
            let (start, end, prompt) = match name {
                "phi3" => (
                    "\n".as_bytes(),
                    "<|end|>\n".as_bytes(),
                    "<|assistant|>\n".as_bytes(),
                ),
                "falcon3" => (
                    "\n".as_bytes(),
                    "\n".as_bytes(),
                    "<|assistant|>\n".as_bytes(),
                ),
                "zephyr" => (
                    "\n".as_bytes(),
                    "<|endoftext|>\n".as_bytes(),
                    "<|assistant|>\n".as_bytes(),
                ),
                "chatglm3" => ("\n ".as_bytes(), "".as_bytes(), "<|assistant|>".as_bytes()),
                "chatglm4" => ("\n".as_bytes(), "".as_bytes(), "<|assistant|>\n".as_bytes()),
                _ => ("\n".as_bytes(), "".as_bytes(), "<|assistant|>".as_bytes()),
            };
            for m in messages {
                out.add(&[
                    "<|".as_bytes(),
                    m.role,
                    "|>".as_bytes(),
                    start,
                    m.content,
                    end,
                ])?;
            }
            if add_assistant {
                out.add(&[prompt])?;
            }
        }
        "monarch" => {
            for (i, m) in messages.iter().enumerate() {
                out.add(&[
                    if i == 0 {
                        "".as_bytes()
                    } else {
                        "<s>".as_bytes()
                    },
                    m.role,
                    "\n".as_bytes(),
                    m.content,
                    "</s>\n".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["<s>assistant\n".as_bytes()])?;
            }
        }
        "gemma" => {
            let mut system = Vec::new();
            for m in messages {
                if m.role == b"system" {
                    if system
                        .len()
                        .checked_add(trim(m.content).len())
                        .is_none_or(|n| n > max_bytes)
                    {
                        return Err(Error::new("system prompt exceeds limit"));
                    }
                    append_bytes(&mut system, trim(m.content))?;
                    continue;
                }
                let role = if m.role == b"assistant" {
                    b"model"
                } else {
                    m.role
                };
                out.add(&["<start_of_turn>".as_bytes(), role, "\n".as_bytes()])?;
                if !system.is_empty() && role != b"model" {
                    out.add(&[&system, "\n\n".as_bytes()])?;
                    system.clear();
                }
                out.add(&[trim(m.content), "<end_of_turn>\n".as_bytes()])?;
            }
            if add_assistant {
                out.add(&["<start_of_turn>model\n".as_bytes()])?;
            }
        }
        "orion" => {
            let mut system = Vec::new();
            for m in messages {
                match m.role {
                    b"system" => {
                        if system
                            .len()
                            .checked_add(m.content.len())
                            .is_none_or(|n| n > max_bytes)
                        {
                            return Err(Error::new("system prompt exceeds limit"));
                        }
                        append_bytes(&mut system, m.content)?;
                    }
                    b"user" => {
                        out.add(&["Human: ".as_bytes()])?;
                        if !system.is_empty() {
                            out.add(&[&system, "\n\n".as_bytes()])?;
                            system.clear();
                        }
                        out.add(&[m.content, "\n\nAssistant: </s>".as_bytes()])?;
                    }
                    _ => out.add(&[m.content, "</s>".as_bytes()])?,
                }
            }
        }
        "openchat" => {
            for m in messages {
                if m.role == b"system" {
                    out.add(&[m.content, "<|end_of_turn|>".as_bytes()])?;
                } else {
                    let mut role = copy_bytes(m.role)?;
                    if let Some(first) = role.first_mut() {
                        first.make_ascii_uppercase();
                    }
                    out.add(&[
                        "GPT4 Correct ".as_bytes(),
                        &role,
                        ": ".as_bytes(),
                        m.content,
                        "<|end_of_turn|>".as_bytes(),
                    ])?;
                }
            }
            if add_assistant {
                out.add(&["GPT4 Correct Assistant:".as_bytes()])?;
            }
        }
        "vicuna" | "vicuna-orca" => {
            for m in messages {
                match m.role {
                    b"system" => {
                        if name == "vicuna-orca" {
                            out.add(&["SYSTEM: ".as_bytes(), m.content, "\n".as_bytes()])?;
                        } else {
                            out.add(&[m.content, "\n\n".as_bytes()])?;
                        }
                    }
                    b"user" => out.add(&["USER: ".as_bytes(), m.content, "\n".as_bytes()])?,
                    b"assistant" => {
                        out.add(&["ASSISTANT: ".as_bytes(), m.content, "</s>\n".as_bytes()])?
                    }
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&["ASSISTANT:".as_bytes()])?;
            }
        }
        "deepseek" => {
            for m in messages {
                match m.role {
                    b"system" => out.add(&[m.content])?,
                    b"user" => {
                        out.add(&["### Instruction:\n".as_bytes(), m.content, "\n".as_bytes()])?
                    }
                    b"assistant" => out.add(&[
                        "### Response:\n".as_bytes(),
                        m.content,
                        "\n<|EOT|>\n".as_bytes(),
                    ])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&["### Response:\n".as_bytes()])?;
            }
        }
        "command-r" => {
            for m in messages {
                let role = match m.role {
                    b"system" => "<|SYSTEM_TOKEN|>".as_bytes(),
                    b"user" => "<|USER_TOKEN|>".as_bytes(),
                    b"assistant" => "<|CHATBOT_TOKEN|>".as_bytes(),
                    _ => continue,
                };
                out.add(&[
                    "<|START_OF_TURN_TOKEN|>".as_bytes(),
                    role,
                    trim(m.content),
                    "<|END_OF_TURN_TOKEN|>".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>".as_bytes()])?;
            }
        }
        "llama3" | "llama4" => {
            let (start, end, eot) = if name == "llama3" {
                (
                    "<|start_header_id|>".as_bytes(),
                    "<|end_header_id|>".as_bytes(),
                    "<|eot_id|>".as_bytes(),
                )
            } else {
                (
                    "<|header_start|>".as_bytes(),
                    "<|header_end|>".as_bytes(),
                    "<|eot|>".as_bytes(),
                )
            };
            for m in messages {
                out.add(&[start, m.role, end, "\n\n".as_bytes(), trim(m.content), eot])?;
            }
            if add_assistant {
                out.add(&[start, b"assistant", end, "\n\n".as_bytes()])?;
            }
        }
        "minicpm" => {
            for m in messages {
                if m.role == b"user" {
                    out.add(&["<用户>".as_bytes(), trim(m.content), "<AI>".as_bytes()])?;
                } else {
                    out.add(&[trim(m.content)])?;
                }
            }
        }
        "deepseek2" | "deepseek3" => {
            for m in messages {
                match m.role {
                    b"system" => out.add(&[m.content, "\n\n".as_bytes()])?,
                    b"user" => {
                        if name == "deepseek2" {
                            out.add(&["User: ".as_bytes(), m.content, "\n\n".as_bytes()])?;
                        } else {
                            out.add(&["<｜User｜>".as_bytes(), m.content])?;
                        }
                    }
                    b"assistant" => out.add(&[
                        if name == "deepseek2" {
                            "Assistant: ".as_bytes()
                        } else {
                            "<｜Assistant｜>".as_bytes()
                        },
                        m.content,
                        "<｜end▁of▁sentence｜>".as_bytes(),
                    ])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&[if name == "deepseek2" {
                    "Assistant:".as_bytes()
                } else {
                    "<｜Assistant｜>".as_bytes()
                }])?;
            }
        }
        "deepseek-ocr" => {
            for m in messages {
                out.add(&[m.content])?;
            }
        }
        "exaone3" | "exaone4" => {
            for m in messages {
                if !matches!(m.role, b"system" | b"user" | b"assistant")
                    && !(m.role == b"tool" && name == "exaone4")
                {
                    continue;
                }
                out.add(&[
                    "[|".as_bytes(),
                    m.role,
                    "|]".as_bytes(),
                    trim(m.content),
                    if m.role == b"user" {
                        "\n".as_bytes()
                    } else {
                        "[|endofturn|]\n".as_bytes()
                    },
                ])?;
            }
            if add_assistant {
                out.add(&["[|assistant|]".as_bytes()])?;
            }
        }
        "exaone-moe" => {
            for m in messages {
                if matches!(m.role, b"system" | b"user" | b"assistant" | b"tool") {
                    out.add(&[
                        "<|".as_bytes(),
                        m.role,
                        "|>\n".as_bytes(),
                        trim(m.content),
                        "<|endofturn|>\n".as_bytes(),
                    ])?;
                }
            }
            if add_assistant {
                out.add(&["<|assistant|>\n".as_bytes()])?;
            }
        }
        "rwkv-world" => {
            for (i, m) in messages.iter().enumerate() {
                let role = match m.role {
                    b"system" => "System".as_bytes(),
                    b"user" => "User".as_bytes(),
                    b"assistant" => "Assistant".as_bytes(),
                    _ => continue,
                };
                out.add(&[role, ": ".as_bytes(), trim(m.content), "\n\n".as_bytes()])?;
                if m.role == b"user" && i + 1 == messages.len() {
                    out.add(&["Assistant:".as_bytes()])?;
                }
            }
        }
        "granite" | "granite-4.0" | "granite-4.1" => {
            for m in messages {
                let role = if name != "granite" && m.role == b"assistant_tool_call" {
                    b"assistant"
                } else {
                    m.role
                };
                out.add(&[
                    "<|start_of_role|>".as_bytes(),
                    role,
                    "<|end_of_role|>".as_bytes(),
                ])?;
                if m.role == b"assistant_tool_call" {
                    out.add(&["<|tool_call|>".as_bytes()])?;
                }
                out.add(&[m.content, "<|end_of_text|>\n".as_bytes()])?;
            }
            if add_assistant {
                out.add(&["<|start_of_role|>assistant<|end_of_role|>".as_bytes()])?;
            }
        }
        "gigachat" => {
            let system = messages.first().is_some_and(|m| m.role == b"system");
            out.add(&["<s>".as_bytes()])?;
            if system {
                out.add(&[messages[0].content, "<|message_sep|>".as_bytes()])?;
            }
            for m in &messages[usize::from(system)..] {
                match m.role {
                    b"user" => out.add(&[
                        "user<|role_sep|>".as_bytes(),
                        m.content,
                        "<|message_sep|>available functions<|role_sep|>[]<|message_sep|>"
                            .as_bytes(),
                    ])?,
                    b"assistant" => out.add(&[
                        "assistant<|role_sep|>".as_bytes(),
                        m.content,
                        "<|message_sep|>".as_bytes(),
                    ])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&["assistant<|role_sep|>".as_bytes()])?;
            }
        }
        "megrez" => {
            for m in messages {
                out.add(&[
                    "<|role_start|>".as_bytes(),
                    m.role,
                    "<|role_end|>".as_bytes(),
                    m.content,
                    "<|turn_end|>".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["<|role_start|>assistant<|role_end|>".as_bytes()])?;
            }
        }
        "yandex" => {
            for m in messages {
                match m.role {
                    b"user" => {
                        out.add(&[" Пользователь: ".as_bytes(), m.content, "\n\n".as_bytes()])?
                    }
                    b"assistant" => {
                        out.add(&[" Ассистент: ".as_bytes(), m.content, "\n\n".as_bytes()])?
                    }
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&[" Ассистент:[SEP]".as_bytes()])?;
            }
        }
        "bailing" | "bailing-think" | "bailing2" => {
            if name == "bailing2" && !messages.first().is_some_and(|m| m.role == b"system") {
                out.add(&["<role>SYSTEM</role>detailed thinking off<|role_end|>".as_bytes()])?;
            }
            for m in messages {
                let role = if m.role == b"user" {
                    copy_bytes(b"HUMAN")?
                } else {
                    uppercase(m.role)?
                };
                out.add(&[
                    "<role>".as_bytes(),
                    &role,
                    "</role>".as_bytes(),
                    m.content,
                    if name == "bailing2" {
                        "<|role_end|>".as_bytes()
                    } else {
                        "".as_bytes()
                    },
                ])?;
            }
            if add_assistant {
                out.add(&[
                    "<role>ASSISTANT</role>".as_bytes(),
                    if name == "bailing-think" {
                        "<think>".as_bytes()
                    } else {
                        "".as_bytes()
                    },
                ])?;
            }
        }
        "smolvlm" => {
            out.add(&["<|im_start|>".as_bytes()])?;
            for m in messages {
                if m.role == b"system" {
                    out.add(&[m.content, "\n\n".as_bytes()])?;
                } else {
                    out.add(&[
                        if m.role == b"user" {
                            "User: ".as_bytes()
                        } else {
                            "Assistant: ".as_bytes()
                        },
                        m.content,
                        "<end_of_utterance>\n".as_bytes(),
                    ])?;
                }
            }
            if add_assistant {
                out.add(&["Assistant:".as_bytes()])?;
            }
        }
        "dots1" => {
            for m in messages {
                let (start, end) = match m.role {
                    b"system" => ("<|system|>".as_bytes(), "<|endofsystem|>".as_bytes()),
                    b"user" => (
                        "<|userprompt|>".as_bytes(),
                        "<|endofuserprompt|>".as_bytes(),
                    ),
                    _ => ("<|response|>".as_bytes(), "<|endofresponse|>".as_bytes()),
                };
                out.add(&[start, m.content, end])?;
            }
            if add_assistant {
                out.add(&["<|response|>".as_bytes()])?;
            }
        }
        "hunyuan-moe" => {
            for m in messages {
                match m.role {
                    b"system" => out.add(&[
                        "<|startoftext|>".as_bytes(),
                        m.content,
                        "<|extra_4|>".as_bytes(),
                    ])?,
                    b"assistant" => out.add(&[m.content, "<|eos|>".as_bytes()])?,
                    _ => out.add(&[
                        "<|startoftext|>".as_bytes(),
                        m.content,
                        "<|extra_0|>".as_bytes(),
                    ])?,
                }
            }
        }
        "gpt-oss" => {
            for m in messages {
                out.add(&[
                    "<|start|>".as_bytes(),
                    m.role,
                    "<|message|>".as_bytes(),
                    m.content,
                    if m.role == b"assistant" {
                        "<|return|>".as_bytes()
                    } else {
                        "<|end|>".as_bytes()
                    },
                ])?;
            }
            if add_assistant {
                out.add(&["<|start|>assistant".as_bytes()])?;
            }
        }
        "hunyuan-dense" => {
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role == b"system" {
                    out.add(&[m.content, "<｜hy_place▁holder▁no▁3｜>".as_bytes()])?;
                }
                if m.role == b"assistant" {
                    out.add(&[
                        "<｜hy_Assistant｜>".as_bytes(),
                        m.content,
                        "<｜hy_place▁holder▁no▁2｜>".as_bytes(),
                    ])?;
                } else if m.role == b"user" {
                    out.add(&[
                        "<｜hy_User｜>".as_bytes(),
                        m.content,
                        "<｜hy_Assistant｜>".as_bytes(),
                    ])?;
                }
            }
        }
        "hunyuan-vl" => {
            out.add(&["<｜hy_begin▁of▁sentence｜>".as_bytes()])?;
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role == b"system" {
                    out.add(&[m.content, "<｜hy_place▁holder▁no▁3｜>".as_bytes()])?;
                    continue;
                }
                if m.role == b"user" {
                    out.add(&[m.content, "<｜hy_User｜>".as_bytes()])?;
                } else if m.role == b"assistant" {
                    out.add(&[m.content, "<｜hy_Assistant｜>".as_bytes()])?;
                }
            }
        }
        "kimi-k2" => {
            for m in messages {
                let prefix = match m.role {
                    b"system" => "<|im_system|>system<|im_middle|>".as_bytes(),
                    b"user" => "<|im_user|>user<|im_middle|>".as_bytes(),
                    b"assistant" => "<|im_assistant|>assistant<|im_middle|>".as_bytes(),
                    b"tool" => "<|im_system|>tool<|im_middle|>".as_bytes(),
                    _ => "".as_bytes(),
                };
                out.add(&[prefix, m.content, "<|im_end|>".as_bytes()])?;
            }
            if add_assistant {
                out.add(&["<|im_assistant|>assistant<|im_middle|>".as_bytes()])?;
            }
        }
        "seed_oss" => {
            for m in messages {
                out.add(&[
                    "<seed:bos>".as_bytes(),
                    m.role,
                    "\n".as_bytes(),
                    if m.role == b"assistant" {
                        trim(m.content)
                    } else {
                        m.content
                    },
                    "<seed:eos>".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["<seed:bos>assistant\n".as_bytes()])?;
            }
        }
        "grok-2" => {
            for m in messages {
                let prefix = match m.role {
                    b"system" => "System: ".as_bytes(),
                    b"user" => "Human: ".as_bytes(),
                    b"assistant" => "Assistant: ".as_bytes(),
                    _ => continue,
                };
                out.add(&[
                    prefix,
                    if m.role == b"assistant" {
                        m.content
                    } else {
                        trim(m.content)
                    },
                    "<|separator|>\n\n".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["Assistant:".as_bytes()])?;
            }
        }
        "pangu-embedded" => {
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role != b"system" {
                    out.add(&["[unused9]系统：[unused10]".as_bytes()])?;
                }
                let role = match m.role {
                    b"system" => "系统".as_bytes(),
                    b"user" => "用户".as_bytes(),
                    b"assistant" => "助手".as_bytes(),
                    b"tool" => "工具".as_bytes(),
                    b"function" => "方法".as_bytes(),
                    _ => continue,
                };
                out.add(&[
                    "[unused9]".as_bytes(),
                    role,
                    "：".as_bytes(),
                    m.content,
                    "[unused10]".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["[unused9]助手：".as_bytes()])?;
            }
        }
        "solar-open" => {
            for m in messages {
                out.add(&[
                    "<|begin|>".as_bytes(),
                    m.role,
                    "<|content|>".as_bytes(),
                    m.content,
                    "<|end|>".as_bytes(),
                ])?;
            }
            if add_assistant {
                out.add(&["<|begin|>assistant".as_bytes()])?;
            }
        }
        _ => return Err(Error::new("native template dispatch is not implemented")),
    }
    Ok(out.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chatml_exact_bytes() {
        let m = [
            Message::new("system", "Be terse."),
            Message::new("user", "Hi 世界"),
        ];
        assert_eq!(
            render("chatml", &m, true).unwrap(),
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi 世界<|im_end|>\n<|im_start|>assistant\n"
        );
    }
    #[test]
    fn gemma_merges_system_only_into_non_model_turn() {
        let m = [
            Message::new("system", " S "),
            Message::new("assistant", " A "),
            Message::new("user", " U "),
        ];
        assert_eq!(
            render("gemma", &m, true).unwrap(),
            "<start_of_turn>model\nA<end_of_turn>\n<start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\n"
        );
    }
    #[test]
    fn llama2_history_and_ascii_whitespace_contract() {
        let m = [
            Message::new("system", " S "),
            Message::new("user", " U "),
            Message::new("assistant", " A "),
            Message::new("user", " V "),
        ];
        assert_eq!(
            render("llama2-sys-bos", &m, true).unwrap(),
            "[INST] <<SYS>>\n S \n<</SYS>>\n\n U  [/INST] A </s><s>[INST]  V  [/INST]"
        );
        let m = [Message::new("user", "\u{a0}x\u{a0}")];
        assert!(
            render("llama3", &m, false)
                .unwrap()
                .contains("\u{a0}x\u{a0}")
        );
    }
    #[test]
    fn every_registered_native_name_has_implementation() {
        assert_eq!(BUILTIN_TEMPLATES.len(), 54);
        let m = [
            Message::new("system", "system"),
            Message::new("user", "text"),
            Message::new("assistant", "reply"),
            Message::new("tool", "data"),
        ];
        for &name in BUILTIN_TEMPLATES {
            for flag in [true, false] {
                render(name, &m, flag).unwrap();
                render(name, &[], flag).unwrap();
            }
        }
    }
    #[test]
    fn native_detection_order_and_unknown_jinja_are_explicit() {
        assert_eq!(detect_template("<|im_start|><|im_sep|>"), Some("phi4"));
        assert_eq!(
            detect_template("[INST] [SYSTEM_PROMPT]"),
            Some("mistral-v7")
        );
        assert_eq!(detect_template("<|endofuserprompt|>"), Some("dots1"));
        assert!(render("{{ unknown_function() }}", &[], false).is_err());
    }
    #[test]
    fn malformed_roles_nuls_and_output_limits_are_rejected() {
        assert!(render("openchat", &[Message::new("", "x")], true).is_err());
        assert!(render("chatml", &[Message::new("user", "x\0y")], true).is_err());
        assert!(render_with_limit("chatml", &[Message::new("user", "x")], true, 8).is_err());
    }
}
