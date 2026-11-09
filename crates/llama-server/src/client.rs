//! Rust-authored diagnostics rendering. No browser script or legacy UI is embedded.
pub fn escape_html(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

pub fn diagnostics(status: &str, model_count: usize) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Rust llama server</title><style>body{{font:18px system-ui;max-width:52rem;margin:3rem auto;padding:0 1rem;background:#f6f5ef;color:#19271e}}main{{background:white;padding:2rem;border-radius:1rem}}code{{overflow-wrap:anywhere}}a{{color:#216044}}</style><main><h1>Rust llama server</h1><p role=\"status\">{}</p><p>Available models: {model_count}</p><h2>Transport checkpoint</h2><p>This Rust-authored diagnostics page exposes server status. The Rust inference runtime is an explicit provider integration boundary.</p><ul><li><a href=\"/health\">Health</a></li><li><a href=\"/v1/models\">Models</a></li><li><a href=\"/props\">Properties</a></li></ul><p>Use POST <code>/v1/chat/completions</code> with JSON messages for a connected provider. This checkpoint does not ship a WASM chat client or a model.</p></main></html>",
        escape_html(status)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn html_text_is_escaped() {
        assert_eq!(
            escape_html("<script>\"'&"),
            "&lt;script&gt;&quot;&#39;&amp;"
        );
    }
}
