//! Regression for shared generation's validation order and lazy CPU execution.
use llama_inference::{Error, FinishReason, GenerationOptions, Model};
use std::{cell::Cell, path::PathBuf};
fn model(name: &str) -> Model {
    Model::load(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/inference")
            .join(name),
    )
    .unwrap()
}
#[test]
fn cancellation_precedes_bert_decoder_rejection_and_option_validation() {
    let model = model("tiny-bert-missing-pooling.gguf");
    let calls = Cell::new(0);
    let mut emit = |_: &str| -> bool { panic!("rejected generation must not emit") };
    let options = GenerationOptions {
        temperature: f32::NAN,
        ..GenerationOptions::default()
    };
    let result = model.generate(
        "hello",
        &options,
        &|| {
            calls.set(calls.get() + 1);
            true
        },
        &mut emit,
    );
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(calls.get(), 1);
    // With cancellation cleared, the original architecture error precedes
    // invalid sampler options and retains the generation API's message.
    let result = model.generate("hello", &options, &|| false, &mut emit);
    assert!(matches!(result, Err(Error::Unsupported(s))
        if s == "BERT is an embedding model; generation is unavailable"));
}
#[test]
fn zero_generation_limit_preserves_prompt_usage_and_emits_nothing() {
    let model = model("tiny-qwen2-separate.gguf");
    let calls = Cell::new(0);
    let stats = model
        .generate(
            "hello",
            &GenerationOptions {
                max_tokens: 0,
                ..GenerationOptions::default()
            },
            &|| {
                calls.set(calls.get() + 1);
                false
            },
            &mut |_| panic!("zero generation limit must not emit"),
        )
        .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(stats.prompt_tokens, 4);
    assert_eq!(stats.completion_tokens, 0);
    assert_eq!(stats.finish_reason, FinishReason::Length);
}
#[test]
fn lazy_cpu_session_preserves_native_generation_and_eos_behavior() {
    let model = model("tiny-qwen2-separate.gguf");
    // These complete strings and usage counts come from the independently
    // captured tiny-qwen2-separate-metadata.json C API fixture.
    for (prompt, expected, prompt_tokens, completion_tokens, reason) in [
        ("", "\n\n\n\n", 1, 4, FinishReason::Length),
        ("a cat", "", 4, 1, FinishReason::Eos),
    ] {
        let mut text = String::new();
        let stats = model
            .generate(
                prompt,
                &GenerationOptions {
                    max_tokens: 4,
                    ..GenerationOptions::default()
                },
                &|| false,
                &mut |piece| {
                    text.push_str(piece);
                    true
                },
            )
            .unwrap();
        assert_eq!(text, expected);
        assert_eq!(stats.prompt_tokens, prompt_tokens);
        assert_eq!(stats.completion_tokens, completion_tokens);
        assert_eq!(stats.finish_reason, reason);
    }
}
