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

/// Same ordered heuristics as the original native formatter. The returned
/// `dots1` tag is an internal dispatch tag, not a newly registered alias.
pub fn detect_template(template: &str) -> Option<&'static str> {
    if let Some(&name) = BUILTIN_TEMPLATES.iter().find(|&&name| name == template) {
        return Some(name);
    }
    let has = |s: &str| template.contains(s);
    Some(if has("<|im_start|>") {
        if has("<|im_sep|>") {
            "phi4"
        } else if has("<end_of_utterance>") {
            "smolvlm"
        } else {
            "chatml"
        }
    } else if template.starts_with("mistral") || has("[INST]") {
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

fn trim(value: &str) -> &str {
    value.trim_matches([' ', '\t', '\n', '\r', '\u{b}', '\u{c}'])
}

struct Output {
    value: String,
    limit: usize,
}
impl Output {
    fn add(&mut self, parts: &[&str]) -> Result<()> {
        let total = parts
            .iter()
            .try_fold(self.value.len(), |total, p| total.checked_add(p.len()))
            .ok_or_else(|| Error::new("chat output size overflow"))?;
        if total > self.limit {
            return Err(Error::new("chat output exceeds byte limit"));
        }
        for part in parts {
            self.value.push_str(part);
        }
        Ok(())
    }
}

pub fn render(template: &str, messages: &[Message], add_assistant: bool) -> Result<String> {
    render_with_limit(template, messages, add_assistant, 16 * 1024 * 1024)
}

pub fn render_with_limit(
    template: &str,
    messages: &[Message],
    add_assistant: bool,
    max_bytes: usize,
) -> Result<String> {
    if template.len() > 1024 * 1024 {
        return Err(Error::new("template exceeds 1 MiB limit"));
    }
    let name = detect_template(template).ok_or_else(|| {
        Error::new("unrecognized native chat template; full Jinja evaluation is not implemented")
    })?;
    if messages.len() > 65_536 {
        return Err(Error::new("chat message count exceeds limit"));
    }
    for message in messages {
        if message.role.is_empty() {
            return Err(Error::new("empty chat role is invalid"));
        }
        if message.role.contains('\0') || message.content.contains('\0') {
            return Err(Error::new(
                "NUL in chat message cannot be represented by the source C-string contract",
            ));
        }
        if message.role.len() > max_bytes || message.content.len() > max_bytes {
            return Err(Error::new("chat message exceeds byte limit"));
        }
    }
    let mut out = Output {
        value: String::new(),
        limit: max_bytes,
    };
    match name {
        "chatml" | "phi4" => {
            let separator = if name == "phi4" { "<|im_sep|>" } else { "\n" };
            let suffix = if name == "phi4" {
                "<|im_end|>"
            } else {
                "<|im_end|>\n"
            };
            for m in messages {
                out.add(&["<|im_start|>", &m.role, separator, &m.content, suffix])?;
            }
            if add_assistant {
                out.add(&["<|im_start|>assistant", separator])?;
            }
        }
        "mistral-v7" | "mistral-v7-tekken" => {
            let space = if name == "mistral-v7" { " " } else { "" };
            for m in messages {
                match m.role.as_str() {
                    "system" => {
                        out.add(&["[SYSTEM_PROMPT]", space, &m.content, "[/SYSTEM_PROMPT]"])?
                    }
                    "user" => out.add(&["[INST]", space, &m.content, "[/INST]"])?,
                    _ => out.add(&[space, &m.content, "</s>"])?,
                }
            }
        }
        "mistral-v1" | "mistral-v3" | "mistral-v3-tekken" => {
            let leading = if name == "mistral-v1" { " " } else { "" };
            let trailing = if name == "mistral-v3-tekken" { "" } else { " " };
            let mut inside = false;
            for m in messages {
                if !inside {
                    out.add(&[leading, "[INST]", trailing])?;
                    inside = true;
                }
                match m.role.as_str() {
                    "system" => out.add(&[&m.content, "\n\n"])?,
                    "user" => out.add(&[&m.content, leading, "[/INST]"])?,
                    _ => {
                        out.add(&[
                            trailing,
                            if name == "mistral-v3" {
                                trim(&m.content)
                            } else {
                                &m.content
                            },
                            "</s>",
                        ])?;
                        inside = false;
                    }
                }
            }
        }
        "llama2" | "llama2-sys" | "llama2-sys-bos" | "llama2-sys-strip" => {
            let mut inside = true;
            out.add(&["[INST] "])?;
            for m in messages {
                let content = if name == "llama2-sys-strip" {
                    trim(&m.content)
                } else {
                    &m.content
                };
                if !inside {
                    out.add(&[if name == "llama2-sys-bos" {
                        "<s>[INST] "
                    } else {
                        "[INST] "
                    }])?;
                    inside = true;
                }
                match m.role.as_str() {
                    "system" => {
                        if name != "llama2" {
                            out.add(&["<<SYS>>\n", content, "\n<</SYS>>\n\n"])?;
                        } else {
                            out.add(&[content, "\n"])?;
                        }
                    }
                    "user" => out.add(&[content, " [/INST]"])?,
                    _ => {
                        out.add(&[content, "</s>"])?;
                        inside = false;
                    }
                }
            }
        }
        "phi3" | "falcon3" | "zephyr" | "glmedge" | "chatglm3" | "chatglm4" => {
            if name == "chatglm3" {
                out.add(&["[gMASK]sop"])?;
            }
            if name == "chatglm4" {
                out.add(&["[gMASK]<sop>"])?;
            }
            let (start, end, prompt) = match name {
                "phi3" => ("\n", "<|end|>\n", "<|assistant|>\n"),
                "falcon3" => ("\n", "\n", "<|assistant|>\n"),
                "zephyr" => ("\n", "<|endoftext|>\n", "<|assistant|>\n"),
                "chatglm3" => ("\n ", "", "<|assistant|>"),
                "chatglm4" => ("\n", "", "<|assistant|>\n"),
                _ => ("\n", "", "<|assistant|>"),
            };
            for m in messages {
                out.add(&["<|", &m.role, "|>", start, &m.content, end])?;
            }
            if add_assistant {
                out.add(&[prompt])?;
            }
        }
        "monarch" => {
            for (i, m) in messages.iter().enumerate() {
                out.add(&[
                    if i == 0 { "" } else { "<s>" },
                    &m.role,
                    "\n",
                    &m.content,
                    "</s>\n",
                ])?;
            }
            if add_assistant {
                out.add(&["<s>assistant\n"])?;
            }
        }
        "gemma" => {
            let mut system = String::new();
            for m in messages {
                if m.role == "system" {
                    if system
                        .len()
                        .checked_add(trim(&m.content).len())
                        .is_none_or(|n| n > max_bytes)
                    {
                        return Err(Error::new("system prompt exceeds limit"));
                    }
                    system.push_str(trim(&m.content));
                    continue;
                }
                let role = if m.role == "assistant" {
                    "model"
                } else {
                    &m.role
                };
                out.add(&["<start_of_turn>", role, "\n"])?;
                if !system.is_empty() && role != "model" {
                    out.add(&[&system, "\n\n"])?;
                    system.clear();
                }
                out.add(&[trim(&m.content), "<end_of_turn>\n"])?;
            }
            if add_assistant {
                out.add(&["<start_of_turn>model\n"])?;
            }
        }
        "orion" => {
            let mut system = String::new();
            for m in messages {
                match m.role.as_str() {
                    "system" => {
                        if system
                            .len()
                            .checked_add(m.content.len())
                            .is_none_or(|n| n > max_bytes)
                        {
                            return Err(Error::new("system prompt exceeds limit"));
                        }
                        system.push_str(&m.content);
                    }
                    "user" => {
                        out.add(&["Human: "])?;
                        if !system.is_empty() {
                            out.add(&[&system, "\n\n"])?;
                            system.clear();
                        }
                        out.add(&[&m.content, "\n\nAssistant: </s>"])?;
                    }
                    _ => out.add(&[&m.content, "</s>"])?,
                }
            }
        }
        "openchat" => {
            for m in messages {
                if m.role == "system" {
                    out.add(&[&m.content, "<|end_of_turn|>"])?;
                } else {
                    let mut role = m.role.clone();
                    if role.as_bytes()[0].is_ascii() {
                        role[0..1].make_ascii_uppercase();
                    }
                    out.add(&["GPT4 Correct ", &role, ": ", &m.content, "<|end_of_turn|>"])?;
                }
            }
            if add_assistant {
                out.add(&["GPT4 Correct Assistant:"])?;
            }
        }
        "vicuna" | "vicuna-orca" => {
            for m in messages {
                match m.role.as_str() {
                    "system" => {
                        if name == "vicuna-orca" {
                            out.add(&["SYSTEM: ", &m.content, "\n"])?;
                        } else {
                            out.add(&[&m.content, "\n\n"])?;
                        }
                    }
                    "user" => out.add(&["USER: ", &m.content, "\n"])?,
                    "assistant" => out.add(&["ASSISTANT: ", &m.content, "</s>\n"])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&["ASSISTANT:"])?;
            }
        }
        "deepseek" => {
            for m in messages {
                match m.role.as_str() {
                    "system" => out.add(&[&m.content])?,
                    "user" => out.add(&["### Instruction:\n", &m.content, "\n"])?,
                    "assistant" => out.add(&["### Response:\n", &m.content, "\n<|EOT|>\n"])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&["### Response:\n"])?;
            }
        }
        "command-r" => {
            for m in messages {
                let role = match m.role.as_str() {
                    "system" => "<|SYSTEM_TOKEN|>",
                    "user" => "<|USER_TOKEN|>",
                    "assistant" => "<|CHATBOT_TOKEN|>",
                    _ => continue,
                };
                out.add(&[
                    "<|START_OF_TURN_TOKEN|>",
                    role,
                    trim(&m.content),
                    "<|END_OF_TURN_TOKEN|>",
                ])?;
            }
            if add_assistant {
                out.add(&["<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"])?;
            }
        }
        "llama3" | "llama4" => {
            let (start, end, eot) = if name == "llama3" {
                ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
            } else {
                ("<|header_start|>", "<|header_end|>", "<|eot|>")
            };
            for m in messages {
                out.add(&[start, &m.role, end, "\n\n", trim(&m.content), eot])?;
            }
            if add_assistant {
                out.add(&[start, "assistant", end, "\n\n"])?;
            }
        }
        "minicpm" => {
            for m in messages {
                if m.role == "user" {
                    out.add(&["<用户>", trim(&m.content), "<AI>"])?;
                } else {
                    out.add(&[trim(&m.content)])?;
                }
            }
        }
        "deepseek2" | "deepseek3" => {
            for m in messages {
                match m.role.as_str() {
                    "system" => out.add(&[&m.content, "\n\n"])?,
                    "user" => {
                        if name == "deepseek2" {
                            out.add(&["User: ", &m.content, "\n\n"])?;
                        } else {
                            out.add(&["<｜User｜>", &m.content])?;
                        }
                    }
                    "assistant" => out.add(&[
                        if name == "deepseek2" {
                            "Assistant: "
                        } else {
                            "<｜Assistant｜>"
                        },
                        &m.content,
                        "<｜end▁of▁sentence｜>",
                    ])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&[if name == "deepseek2" {
                    "Assistant:"
                } else {
                    "<｜Assistant｜>"
                }])?;
            }
        }
        "deepseek-ocr" => {
            for m in messages {
                out.add(&[&m.content])?;
            }
        }
        "exaone3" | "exaone4" => {
            for m in messages {
                if !matches!(m.role.as_str(), "system" | "user" | "assistant")
                    && !(m.role == "tool" && name == "exaone4")
                {
                    continue;
                }
                out.add(&[
                    "[|",
                    &m.role,
                    "|]",
                    trim(&m.content),
                    if m.role == "user" {
                        "\n"
                    } else {
                        "[|endofturn|]\n"
                    },
                ])?;
            }
            if add_assistant {
                out.add(&["[|assistant|]"])?;
            }
        }
        "exaone-moe" => {
            for m in messages {
                if matches!(m.role.as_str(), "system" | "user" | "assistant" | "tool") {
                    out.add(&["<|", &m.role, "|>\n", trim(&m.content), "<|endofturn|>\n"])?;
                }
            }
            if add_assistant {
                out.add(&["<|assistant|>\n"])?;
            }
        }
        "rwkv-world" => {
            for (i, m) in messages.iter().enumerate() {
                let role = match m.role.as_str() {
                    "system" => "System",
                    "user" => "User",
                    "assistant" => "Assistant",
                    _ => continue,
                };
                out.add(&[role, ": ", trim(&m.content), "\n\n"])?;
                if m.role == "user" && i + 1 == messages.len() {
                    out.add(&["Assistant:"])?;
                }
            }
        }
        "granite" | "granite-4.0" | "granite-4.1" => {
            for m in messages {
                let role = if name != "granite" && m.role == "assistant_tool_call" {
                    "assistant"
                } else {
                    &m.role
                };
                out.add(&["<|start_of_role|>", role, "<|end_of_role|>"])?;
                if m.role == "assistant_tool_call" {
                    out.add(&["<|tool_call|>"])?;
                }
                out.add(&[&m.content, "<|end_of_text|>\n"])?;
            }
            if add_assistant {
                out.add(&["<|start_of_role|>assistant<|end_of_role|>"])?;
            }
        }
        "gigachat" => {
            let system = messages.first().is_some_and(|m| m.role == "system");
            out.add(&["<s>"])?;
            if system {
                out.add(&[&messages[0].content, "<|message_sep|>"])?;
            }
            for m in &messages[usize::from(system)..] {
                match m.role.as_str() {
                    "user" => out.add(&[
                        "user<|role_sep|>",
                        &m.content,
                        "<|message_sep|>available functions<|role_sep|>[]<|message_sep|>",
                    ])?,
                    "assistant" => {
                        out.add(&["assistant<|role_sep|>", &m.content, "<|message_sep|>"])?
                    }
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&["assistant<|role_sep|>"])?;
            }
        }
        "megrez" => {
            for m in messages {
                out.add(&[
                    "<|role_start|>",
                    &m.role,
                    "<|role_end|>",
                    &m.content,
                    "<|turn_end|>",
                ])?;
            }
            if add_assistant {
                out.add(&["<|role_start|>assistant<|role_end|>"])?;
            }
        }
        "yandex" => {
            for m in messages {
                match m.role.as_str() {
                    "user" => out.add(&[" Пользователь: ", &m.content, "\n\n"])?,
                    "assistant" => out.add(&[" Ассистент: ", &m.content, "\n\n"])?,
                    _ => {}
                }
            }
            if add_assistant {
                out.add(&[" Ассистент:[SEP]"])?;
            }
        }
        "bailing" | "bailing-think" | "bailing2" => {
            if name == "bailing2" && !messages.first().is_some_and(|m| m.role == "system") {
                out.add(&["<role>SYSTEM</role>detailed thinking off<|role_end|>"])?;
            }
            for m in messages {
                let role = if m.role == "user" {
                    "HUMAN".to_owned()
                } else {
                    m.role.to_ascii_uppercase()
                };
                out.add(&[
                    "<role>",
                    &role,
                    "</role>",
                    &m.content,
                    if name == "bailing2" {
                        "<|role_end|>"
                    } else {
                        ""
                    },
                ])?;
            }
            if add_assistant {
                out.add(&[
                    "<role>ASSISTANT</role>",
                    if name == "bailing-think" {
                        "<think>"
                    } else {
                        ""
                    },
                ])?;
            }
        }
        "smolvlm" => {
            out.add(&["<|im_start|>"])?;
            for m in messages {
                if m.role == "system" {
                    out.add(&[&m.content, "\n\n"])?;
                } else {
                    out.add(&[
                        if m.role == "user" {
                            "User: "
                        } else {
                            "Assistant: "
                        },
                        &m.content,
                        "<end_of_utterance>\n",
                    ])?;
                }
            }
            if add_assistant {
                out.add(&["Assistant:"])?;
            }
        }
        "dots1" => {
            for m in messages {
                let (start, end) = match m.role.as_str() {
                    "system" => ("<|system|>", "<|endofsystem|>"),
                    "user" => ("<|userprompt|>", "<|endofuserprompt|>"),
                    _ => ("<|response|>", "<|endofresponse|>"),
                };
                out.add(&[start, &m.content, end])?;
            }
            if add_assistant {
                out.add(&["<|response|>"])?;
            }
        }
        "hunyuan-moe" => {
            for m in messages {
                match m.role.as_str() {
                    "system" => out.add(&["<|startoftext|>", &m.content, "<|extra_4|>"])?,
                    "assistant" => out.add(&[&m.content, "<|eos|>"])?,
                    _ => out.add(&["<|startoftext|>", &m.content, "<|extra_0|>"])?,
                }
            }
        }
        "gpt-oss" => {
            for m in messages {
                out.add(&[
                    "<|start|>",
                    &m.role,
                    "<|message|>",
                    &m.content,
                    if m.role == "assistant" {
                        "<|return|>"
                    } else {
                        "<|end|>"
                    },
                ])?;
            }
            if add_assistant {
                out.add(&["<|start|>assistant"])?;
            }
        }
        "hunyuan-dense" => {
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role == "system" {
                    out.add(&[&m.content, "<｜hy_place▁holder▁no▁3｜>"])?;
                }
                if m.role == "assistant" {
                    out.add(&[
                        "<｜hy_Assistant｜>",
                        &m.content,
                        "<｜hy_place▁holder▁no▁2｜>",
                    ])?;
                } else if m.role == "user" {
                    out.add(&["<｜hy_User｜>", &m.content, "<｜hy_Assistant｜>"])?;
                }
            }
        }
        "hunyuan-vl" => {
            out.add(&["<｜hy_begin▁of▁sentence｜>"])?;
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role == "system" {
                    out.add(&[&m.content, "<｜hy_place▁holder▁no▁3｜>"])?;
                    continue;
                }
                if m.role == "user" {
                    out.add(&[&m.content, "<｜hy_User｜>"])?;
                } else if m.role == "assistant" {
                    out.add(&[&m.content, "<｜hy_Assistant｜>"])?;
                }
            }
        }
        "kimi-k2" => {
            for m in messages {
                let prefix = match m.role.as_str() {
                    "system" => "<|im_system|>system<|im_middle|>",
                    "user" => "<|im_user|>user<|im_middle|>",
                    "assistant" => "<|im_assistant|>assistant<|im_middle|>",
                    "tool" => "<|im_system|>tool<|im_middle|>",
                    _ => "",
                };
                out.add(&[prefix, &m.content, "<|im_end|>"])?;
            }
            if add_assistant {
                out.add(&["<|im_assistant|>assistant<|im_middle|>"])?;
            }
        }
        "seed_oss" => {
            for m in messages {
                out.add(&[
                    "<seed:bos>",
                    &m.role,
                    "\n",
                    if m.role == "assistant" {
                        trim(&m.content)
                    } else {
                        &m.content
                    },
                    "<seed:eos>",
                ])?;
            }
            if add_assistant {
                out.add(&["<seed:bos>assistant\n"])?;
            }
        }
        "grok-2" => {
            for m in messages {
                let prefix = match m.role.as_str() {
                    "system" => "System: ",
                    "user" => "Human: ",
                    "assistant" => "Assistant: ",
                    _ => continue,
                };
                out.add(&[
                    prefix,
                    if m.role == "assistant" {
                        &m.content
                    } else {
                        trim(&m.content)
                    },
                    "<|separator|>\n\n",
                ])?;
            }
            if add_assistant {
                out.add(&["Assistant:"])?;
            }
        }
        "pangu-embedded" => {
            for (i, m) in messages.iter().enumerate() {
                if i == 0 && m.role != "system" {
                    out.add(&["[unused9]系统：[unused10]"])?;
                }
                let role = match m.role.as_str() {
                    "system" => "系统",
                    "user" => "用户",
                    "assistant" => "助手",
                    "tool" => "工具",
                    "function" => "方法",
                    _ => continue,
                };
                out.add(&["[unused9]", role, "：", &m.content, "[unused10]"])?;
            }
            if add_assistant {
                out.add(&["[unused9]助手："])?;
            }
        }
        "solar-open" => {
            for m in messages {
                out.add(&["<|begin|>", &m.role, "<|content|>", &m.content, "<|end|>"])?;
            }
            if add_assistant {
                out.add(&["<|begin|>assistant"])?;
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
