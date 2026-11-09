//! Captured from the opt-in Rust reference_oracle example using unmodified
//! frozen-source GGML operators. Ordinary tests remain native-library-free.
use llama_model_runtime::*;
use std::{collections::BTreeMap, sync::Arc};
fn expected() -> BTreeMap<&'static str, Vec<f32>> {
    include_str!("data/ggml_reference.tsv")
        .lines()
        .map(|line| {
            let mut columns = line.split('\t');
            let name = columns.next().unwrap();
            let values = columns.map(|x| x.parse().unwrap()).collect();
            (name, values)
        })
        .collect()
}
fn close(name: &str, actual: &[f32], reference: &[f32]) {
    assert_eq!(actual.len(), reference.len());
    for (i, (a, b)) in actual.iter().zip(reference).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 3e-6 + 3e-6 * b.abs(),
            "{name}[{i}] expected {b}, got {a}"
        );
    }
}
fn compare_decoder(name: &str, model: Model) {
    let expected = expected();
    let tokens: Vec<_> = [1, 2, 3, 4]
        .iter()
        .enumerate()
        .map(|(i, t)| BatchToken::token(0, i, *t))
        .collect();
    let mut context = Context::new(Arc::new(model), ContextOptions::default()).unwrap();
    for (i, output) in context.decode_batch(&tokens).unwrap().iter().enumerate() {
        for (kind, values) in [
            ("embedding", output.embedding.as_ref().unwrap()),
            ("logits", output.logits.as_ref().unwrap()),
        ] {
            let key = format!("{name}.{kind}.{i}");
            close(&key, values, &expected[&*key]);
        }
    }
}
#[test]
fn llama_outputs_match_frozen_ggml_graph() {
    compare_decoder("llama", Model::Llama(fixtures::tiny_llama().unwrap()));
}
#[test]
fn mamba_outputs_match_frozen_ggml_scan_and_convolution() {
    compare_decoder("mamba", Model::Mamba(fixtures::tiny_mamba().unwrap()));
}
#[test]
fn bert_outputs_match_frozen_ggml_including_half_precision_gelu_table() {
    let expected = expected();
    let output = fixtures::tiny_bert()
        .unwrap()
        .encode(&[1, 2, 3, 4], Pooling::None)
        .unwrap();
    for (i, values) in output.tokens.iter().enumerate() {
        let key = format!("bert.embedding.{i}");
        close(&key, values, &expected[&*key]);
    }
    assert_eq!(expected.len(), 20);
    assert_eq!(expected.values().map(Vec::len).sum::<usize>(), 104);
}
