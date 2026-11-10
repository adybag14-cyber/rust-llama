//! Frozen native chat C entry points. The shared formatter owns byte semantics;
//! this module supplies foreign-string lifetimes, checked limits and exact C
//! output-buffer rules.
use crate::*;
use llama_behavior::chat::{ByteMessage, detect_template_bytes, render_bytes_with_limit};

const MAX_MESSAGES: usize = 65_536;
const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TEMPLATE_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

// std::map iteration in the frozen implementation is lexically ordered. CStr
// literals have permanent storage; returned pointers need no allocation/free.
static BUILTINS: &[&CStr] = &[
    c"bailing",
    c"bailing-think",
    c"bailing2",
    c"chatglm3",
    c"chatglm4",
    c"chatml",
    c"command-r",
    c"deepseek",
    c"deepseek-ocr",
    c"deepseek2",
    c"deepseek3",
    c"exaone-moe",
    c"exaone3",
    c"exaone4",
    c"falcon3",
    c"gemma",
    c"gigachat",
    c"glmedge",
    c"gpt-oss",
    c"granite",
    c"granite-4.0",
    c"granite-4.1",
    c"grok-2",
    c"hunyuan-dense",
    c"hunyuan-moe",
    c"hunyuan-vl",
    c"kimi-k2",
    c"llama2",
    c"llama2-sys",
    c"llama2-sys-bos",
    c"llama2-sys-strip",
    c"llama3",
    c"llama4",
    c"megrez",
    c"minicpm",
    c"mistral-v1",
    c"mistral-v3",
    c"mistral-v3-tekken",
    c"mistral-v7",
    c"mistral-v7-tekken",
    c"monarch",
    c"openchat",
    c"orion",
    c"pangu-embedded",
    c"phi3",
    c"phi4",
    c"rwkv-world",
    c"seed_oss",
    c"smolvlm",
    c"solar-open",
    c"vicuna",
    c"vicuna-orca",
    c"yandex",
    c"zephyr",
];

unsafe fn bytes<'a>(value: *const c_char) -> Result<&'a [u8]> {
    if value.is_null() {
        return Err("null chat role/content string".into());
    }
    // SAFETY: foreign callers supply a live NUL-terminated string. The borrowed
    // bytes remain live and unchanged through this call, including rendering.
    Ok(unsafe { CStr::from_ptr(value) }.to_bytes())
}

fn check_source_case_domain(name: &str, role: &[u8]) -> Result<()> {
    // The original calls ::toupper(char) without an unsigned-char conversion.
    // On signed-char targets 0x80..=0xfe are outside the C function's domain;
    // 0xff maps to the defined EOF argument. Do not invent an oracle for UB.
    if c_char::MIN < 0 {
        let checked = match name {
            "openchat" => role.get(..1).unwrap_or(&[]),
            "bailing" | "bailing-think" | "bailing2" => role,
            _ => &[],
        };
        if checked.iter().any(|b| (0x80..=0xfe).contains(b)) {
            return Err("chat role is outside the source's defined toupper argument domain".into());
        }
    }
    Ok(())
}

ffi!(
    llama_chat_builtin_templates(output: *mut *const c_char, len: usize) -> i32,
    -1,
    {
        let written = len.min(BUILTINS.len());
        if written != 0 {
            // SAFETY: caller owns at least min(len, count) pointer elements.
            unsafe {
                reference_mut(output)?;
            }
            for (i, name) in BUILTINS.iter().take(written).enumerate() {
                unsafe {
                    output.add(i).write(name.as_ptr());
                }
            }
        }
        Ok(BUILTINS.len() as i32)
    }
);

ffi!(
    llama_chat_apply_template(
        template: *const c_char,
        chat: *const LlamaChatMessage,
        n_msg: usize,
        add_ass: u8,
        buffer: *mut c_char,
        length: i32,
    ) -> i32,
    -1,
    {
        let add_assistant = boolean(add_ass)?;
        let template = if template.is_null() {
            b"chatml".as_slice()
        } else {
            unsafe { bytes(template)? }
        };
        if template.len() > MAX_TEMPLATE_BYTES {
            return Err("chat template exceeds 1 MiB limit".into());
        }
        let name = detect_template_bytes(template)
            .ok_or("unrecognized native chat template; full Jinja evaluation is unavailable")?;
        if n_msg > MAX_MESSAGES {
            return Err("chat message count exceeds 65536 limit".into());
        }
        // SAFETY: caller owns n_msg live message elements; NULL is valid for zero.
        let supplied = unsafe { input_slice(chat, n_msg)? };
        let mut messages = Vec::new();
        messages
            .try_reserve_exact(n_msg)
            .map_err(|_| "chat message allocation failed")?;
        let mut input_bytes = 0usize;
        for message in supplied {
            let role = unsafe { bytes(message.role)? };
            let content = unsafe { bytes(message.content)? };
            input_bytes = input_bytes
                .checked_add(role.len())
                .and_then(|n| n.checked_add(content.len()))
                .ok_or("chat input byte count overflow")?;
            if input_bytes > MAX_INPUT_BYTES {
                return Err("chat input exceeds 16 MiB cumulative byte limit".into());
            }
            check_source_case_domain(name, role)?;
            messages.push(ByteMessage { role, content });
        }
        let output = render_bytes_with_limit(template, &messages, add_assistant, MAX_OUTPUT_BYTES)
            .map_err(|e| e.to_string())?;
        let required = i32::try_from(output.len()).map_err(|_| "chat output exceeds INT32_MAX")?;
        // Frozen llama.cpp uses strncpy, including zero-filling *all* remaining
        // capacity. A full/truncated buffer receives no extra terminator.
        // NULL and nonpositive lengths are count-only calls in the source.
        if !buffer.is_null() && length > 0 {
            let capacity = length as usize;
            let copied = output.len().min(capacity);
            // SAFETY: caller owns length exclusive bytes disjoint from inputs.
            unsafe {
                reference_mut(buffer)?;
                ptr::copy_nonoverlapping(output.as_ptr(), buffer.cast::<u8>(), copied);
                ptr::write_bytes(buffer.add(copied), 0, capacity - copied);
            }
        }
        Ok(required)
    }
);

#[cfg(test)]
mod tests;
