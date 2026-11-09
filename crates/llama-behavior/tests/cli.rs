use std::io::Write;
use std::process::{Command, Stdio};

fn invoke(args: &[&str], input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llama-behavior"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn chat_cli_preserves_exact_template_bytes() {
    let out = invoke(
        &["chat", "chatml", "--no-assistant"],
        r#"[{"role":"user","content":"Hello 世界"}]"#,
    );
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "<|im_start|>user\nHello 世界<|im_end|>\n"
    );
}

#[test]
fn sampler_cli_uses_real_logits_and_rejects_invalid_config() {
    let out = invoke(
        &["sample"],
        r#"{"logits":[1,3,2],"samplers":[{"kind":"top-k","k":2},{"kind":"greedy"}]}"#,
    );
    assert!(out.status.success());
    let value = llama_behavior::json::parse(std::str::from_utf8(&out.stdout).unwrap()).unwrap();
    assert_eq!(value.get("token").unwrap().as_i64(), Some(1));
    let bad = invoke(
        &["sample"],
        r#"{"logits":[1],"samplers":[{"kind":"missing"}]}"#,
    );
    assert_eq!(bad.status.code(), Some(2));
}

#[test]
fn grammar_cli_distinguishes_match_mismatch_and_bad_input() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/t10/grammars/json.gbnf"
    );
    assert!(invoke(&["grammar", path], "{}").status.success());
    assert_eq!(invoke(&["grammar", path], "[]").status.code(), Some(1));
    assert_eq!(invoke(&["unknown"], "").status.code(), Some(2));
    assert!(
        invoke(&["grammar", path, "--prefix"], "{\"a\":")
            .status
            .success()
    );
}

#[test]
fn json_cli_preserves_large_integer_and_speculative_cli_bonus() {
    let out = invoke(&["json"], r#"{"id":9007199254740993}"#);
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "{\"id\":9007199254740993}\n"
    );
    let out = invoke(&["speculative"], r#"{"draft":[1,2,3],"target":[1,9,8,7]}"#);
    assert!(out.status.success());
    let value = llama_behavior::json::parse(std::str::from_utf8(&out.stdout).unwrap()).unwrap();
    assert_eq!(value.get("accepted_draft").unwrap().as_u64(), Some(1));
}
