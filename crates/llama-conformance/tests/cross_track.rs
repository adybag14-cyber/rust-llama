//! Cross-track contract checks use real crate APIs, never a fake model runtime.
use llama_behavior::{
    chat,
    grammar::Grammar,
    json::{self, Value},
    sampling::{Chain, Config},
    speculative::Transaction,
};
use llama_media::{DecodeLimits, audio, image};

#[test]
fn unicode_chat_survives_json_and_incremental_grammar() {
    let original = "A£🦀界\n";
    let data = Value::object([
        ("content", Value::from(original)),
        ("request_id", Value::from(9_007_199_254_740_993u64)),
    ]);
    let saved = json::stringify(&data).unwrap();
    let parsed = json::parse(&saved).unwrap();
    assert_eq!(
        parsed.get("request_id").unwrap().as_u64(),
        Some(9_007_199_254_740_993)
    );
    let content = parsed.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, original);
    let rendered = chat::render("chatml", &[chat::Message::new("user", content)], true).unwrap();
    assert_eq!(
        rendered,
        format!("<|im_start|>user\n{original}<|im_end|>\n<|im_start|>assistant\n")
    );
    let grammar = Grammar::parse("root ::= \"A£🦀界\\n\"").unwrap();
    let mut recognizer = grammar.recognizer().unwrap();
    for byte in original.as_bytes() {
        assert!(recognizer.accept_bytes(&[*byte]).unwrap());
    }
    assert!(recognizer.can_end());
    assert!(!recognizer.would_accept(b"x").unwrap());
    assert!(recognizer.can_end());
}

#[test]
fn accepted_speculative_prefix_preserves_future_sampling() {
    let setup = [Config::TopK { k: 3 }, Config::Dist { seed: 71 }];
    let mut reference = Chain::new(setup.clone()).unwrap();
    let mut draft = Chain::new(setup).unwrap();
    let logits = [-3.0, 0.5, 1.5, 2.0, 0.0];
    let first = reference.sample_and_accept(&logits).unwrap();
    let mut transaction = Transaction::new(&mut draft, 4).unwrap();
    for _ in 0..4 {
        transaction.propose(&logits).unwrap();
    }
    assert_eq!(transaction.commit_prefix(1).unwrap(), [first]);
    for _ in 0..20 {
        assert_eq!(
            draft.sample_and_accept(&logits).unwrap(),
            reference.sample_and_accept(&logits).unwrap()
        );
    }
}

#[test]
fn bounded_malformed_corpus_never_panics_across_media_and_json() {
    let limits = DecodeLimits {
        max_input_bytes: 1024,
        max_image_pixels: 64,
        max_audio_samples: 128,
    };
    let mut seed = 0x4d59_5df4_d0f3_3173u64;
    for i in 0..1024 {
        let mut input = vec![0u8; i % 257];
        for byte in &mut input {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *byte = seed as u8;
        }
        let _ = image::decode(&input, limits);
        let _ = audio::decode_wav(&input, limits);
        let text = String::from_utf8_lossy(&input);
        let _ = json::parse_with_limits(
            &text,
            json::Limits {
                max_bytes: 1024,
                max_depth: 16,
                max_nodes: 128,
            },
        );
    }
}
