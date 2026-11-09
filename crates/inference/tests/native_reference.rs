//! Expected values were captured from the frozen original C API before the
//! Rust model was evaluated. See fixtures/inference/oracle-provenance.json.
//! The pretrained model test is explicit and opt-in because model weights are
//! external test inputs, not source dependencies or bundled product data.
#[path = "support/native_profile.rs"]
mod native_profile;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::{hash_reader, sha256};
use llama_gguf::{Gguf, Limits};
use llama_inference::{FinishReason, GenerationOptions, Model};
use llama_tokenizer::Tokenizer;
use native_profile::source_profile;
use std::{fs::File, io::Cursor};

const STORIES_SHA256: &str = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d";
const VOCAB_SHA256: &str = "16c3724582d59aa8bf84711894e833f916ee46a31d80e21312759c48bf8d0e69";

fn bytes_from_hex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn tokens(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| i32::try_from(value.as_i64().unwrap()).unwrap())
        .collect()
}

#[test]
fn original_spm_68_unicode_special_and_empty_cases() {
    let vocabulary = include_bytes!("../../../fixtures/inference/ggml-vocab-llama-spm.gguf");
    assert_eq!(sha256(vocabulary), VOCAB_SHA256);
    let gguf = Gguf::read(Cursor::new(vocabulary), Limits::default()).unwrap();
    let tokenizer = Tokenizer::from_metadata(&gguf.metadata).unwrap();
    assert_eq!(tokenizer.vocabulary_size(), 32000);
    let reference = json::parse(include_str!(
        "../../../fixtures/inference/llama-spm-reference.json"
    ))
    .unwrap();
    let cases = reference.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 68);
    let mut checked_pieces = 0;
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let expected = tokens(case.get("tokens").unwrap());
        let actual = tokenizer
            .tokenize(
                case.get("prompt").unwrap().as_str().unwrap(),
                case.get("add_special").unwrap().as_bool().unwrap(),
                case.get("parse_special").unwrap().as_bool().unwrap(),
            )
            .unwrap();
        assert_eq!(actual, expected, "{name} token IDs");
        let pieces = case.get("pieces_hex").unwrap().as_array().unwrap();
        let special = case.get("pieces_special_hex").unwrap().as_array().unwrap();
        assert_eq!(pieces.len(), actual.len());
        assert_eq!(special.len(), actual.len());
        for (index, &id) in actual.iter().enumerate() {
            assert_eq!(
                tokenizer.token_piece(id, false).unwrap(),
                bytes_from_hex(pieces[index].as_str().unwrap()),
                "{name} piece {index}"
            );
            assert_eq!(
                tokenizer.token_piece(id, true).unwrap(),
                bytes_from_hex(special[index].as_str().unwrap()),
                "{name} special piece {index}"
            );
            checked_pieces += 1;
        }
        assert_eq!(
            tokenizer.detokenize_bytes(&actual, true, false).unwrap(),
            bytes_from_hex(case.get("detokenized_hex").unwrap().as_str().unwrap()),
            "{name} detokenized bytes"
        );
    }
    eprintln!(
        "native SPM: 68 cases, {checked_pieces} pieces in both special modes, exact IDs and bytes"
    );
}

/// This tolerance was fixed before inspecting Rust results. It allows bounded
/// f32 accumulation-order differences between the original vectorized CPU path
/// and the Rust scalar path for this 64-wide five-layer model. Token IDs and
/// emitted bytes remain exact; the numerical threshold is not adjusted on failure.
/// The Q4_0 regression reuses this same budget without widening it.
const LOGIT_ABSOLUTE: f64 = 2e-4;
const LOGIT_RELATIVE: f64 = 2e-5;

fn compare_row(
    actual: &[f32],
    expected: &Value,
    name: &str,
    row: usize,
    maxima: &mut (f64, f64, usize),
) {
    compare_values(
        actual,
        expected
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_f64().unwrap()),
        name,
        row,
        maxima,
    );
}

fn compare_values(
    actual: &[f32],
    expected: impl ExactSizeIterator<Item = f64>,
    name: &str,
    row: usize,
    maxima: &mut (f64, f64, usize),
) {
    assert_eq!(actual.len(), expected.len(), "{name} row {row} shape");
    for (column, (&a, e)) in actual.iter().zip(expected).enumerate() {
        let difference = (f64::from(a) - e).abs();
        let allowed = LOGIT_ABSOLUTE + LOGIT_RELATIVE * e.abs();
        maxima.0 = maxima.0.max(difference);
        maxima.1 = maxima.1.max(difference / allowed);
        maxima.2 += 1;
        assert!(
            a.is_finite() && e.is_finite() && difference <= allowed,
            "{name} row {row} logit {column}: actual={a:.9e} reference={e:.9e} abs={difference:.9e} allowed={allowed:.9e}"
        );
    }
}

fn argmax_first(values: &[f32]) -> i32 {
    assert!(!values.is_empty());
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    i32::try_from(best).unwrap()
}

#[test]
#[ignore = "requires hash-pinned external stories260K.gguf; set LLAMA_TEST_MODEL and run --ignored --nocapture"]
fn trained_stories260k_full_logits_greedy_ids_and_text() {
    let path = std::env::var_os("LLAMA_TEST_MODEL")
        .expect("LLAMA_TEST_MODEL must point to the external stories260K.gguf");
    let (length, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(length, 1_185_376);
    assert_eq!(hash, STORIES_SHA256, "wrong external model fixture");
    let model = Model::load(&path).unwrap();
    let reference = json::parse(source_profile(
        include_str!("../../../fixtures/inference/stories260k-reference.json"),
        include_str!("../../../fixtures/inference/stories260k-gnu-reference.json"),
    ))
    .unwrap();
    assert_eq!(model.vocabulary_size(), 512);
    let cases = reference.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 6);
    let maximum = reference.get("max_tokens").unwrap().as_u64().unwrap() as usize;
    let mut maxima = (0.0f64, 0.0f64, 0usize);
    let mut greedy_count = 0;
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let prompt = case.get("prompt").unwrap().as_str().unwrap();
        let prompt_tokens = tokens(case.get("tokens").unwrap());
        assert_eq!(
            model.tokenize(prompt, true).unwrap(),
            prompt_tokens,
            "{name}"
        );
        let expected_greedy = tokens(case.get("greedy_tokens").unwrap());
        assert!(!expected_greedy.is_empty());
        let mut sequence = prompt_tokens.clone();
        sequence.extend_from_slice(&expected_greedy[..expected_greedy.len() - 1]);
        let rows = model.logits_for_tokens(&sequence).unwrap();
        assert_eq!(rows.len(), sequence.len());
        let prompt_rows = case.get("prompt_logits").unwrap().as_array().unwrap();
        assert_eq!(prompt_rows.len(), prompt_tokens.len());
        for (row, expected) in prompt_rows.iter().enumerate() {
            compare_row(&rows[row], expected, name, row, &mut maxima);
        }
        let greedy_rows = case.get("greedy_logits").unwrap().as_array().unwrap();
        assert_eq!(greedy_rows.len(), expected_greedy.len());
        for (step, (&token, expected)) in expected_greedy.iter().zip(greedy_rows).enumerate() {
            let row = prompt_tokens.len() - 1 + step;
            compare_row(&rows[row], expected, name, row, &mut maxima);
            assert_eq!(argmax_first(&rows[row]), token, "{name} greedy step {step}");
            greedy_count += 1;
        }
        let mut generated = String::new();
        let stats = model
            .generate(
                prompt,
                &GenerationOptions {
                    max_tokens: maximum,
                    temperature: 0.0,
                    seed: 0,
                    stop: Vec::new(),
                },
                &|| false,
                &mut |piece| {
                    generated.push_str(piece);
                    true
                },
            )
            .unwrap();
        assert_eq!(
            generated.as_bytes(),
            bytes_from_hex(case.get("generated_hex").unwrap().as_str().unwrap()),
            "{name} actual autoregressive emitted bytes"
        );
        assert_eq!(
            stats.prompt_tokens,
            prompt_tokens.len(),
            "{name} prompt usage"
        );
        assert_eq!(
            stats.completion_tokens,
            expected_greedy.len(),
            "{name} completion usage"
        );
        assert_eq!(
            stats.finish_reason,
            if case.get("finished_eog").unwrap().as_bool().unwrap() {
                FinishReason::Eos
            } else {
                FinishReason::Length
            },
            "{name} finish reason"
        );
    }
    assert_eq!(greedy_count, 96);
    eprintln!(
        "native trained-model conformance: cases=6 greedy_ids={greedy_count} compared_logits={} max_abs={:.9e} max_fraction_of_tolerance={:.9e}",
        maxima.2, maxima.0, maxima.1
    );
}

#[test]
#[ignore = "requires hash-pinned external stories15M-q4_0.gguf; set LLAMA_TEST_Q4_MODEL and run --release --ignored --nocapture"]
fn trained_stories15m_q4_full_logits_greedy_ids_and_text() {
    let path = std::env::var_os("LLAMA_TEST_Q4_MODEL")
        .expect("LLAMA_TEST_Q4_MODEL must point to the external stories15M-q4_0.gguf");
    let (length, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(length, 19_077_344);
    assert_eq!(
        hash, "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
        "wrong external Q4_0 model fixture"
    );
    let expected = source_profile(
        include_bytes!("../../../fixtures/inference/stories15m-q4-logits.f32le.bin") as &[u8],
        include_bytes!("../../../fixtures/inference/stories15m-q4-gnu-logits.f32le.bin") as &[u8],
    );
    assert_eq!(expected.len(), 2_560_000);
    assert_eq!(
        sha256(expected),
        source_profile(
            "579a521b36d682c9b26cac4d533b00beb4803e3bf5bfe2f86a29a07a4959fd21",
            "579a521b36d682c9b26cac4d533b00beb4803e3bf5bfe2f86a29a07a4959fd21"
        ),
        "native Q4_0 expected logits drifted"
    );
    let reference = json::parse(source_profile(
        include_str!("../../../fixtures/inference/stories15m-q4-metadata.json"),
        include_str!("../../../fixtures/inference/stories15m-q4-gnu-metadata.json"),
    ))
    .unwrap();
    assert_eq!(
        reference.get("kv_type").and_then(Value::as_str),
        Some("F32")
    );
    assert_eq!(
        reference.get("flash_attention").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(reference.get("threads").and_then(Value::as_u64), Some(1));
    assert_eq!(
        reference.get("n_vocab").and_then(Value::as_u64),
        Some(32_000)
    );
    assert_eq!(reference.get("row_count").and_then(Value::as_u64), Some(20));
    let declared = reference.get("comparison_tolerance").unwrap();
    assert_eq!(
        declared.get("absolute").unwrap().as_f64(),
        Some(LOGIT_ABSOLUTE)
    );
    assert_eq!(
        declared.get("relative").unwrap().as_f64(),
        Some(LOGIT_RELATIVE)
    );
    let model = Model::load(&path).unwrap();
    assert_eq!(model.vocabulary_size(), 32_000);
    let cases = reference.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 1);
    let mut maxima = (0.0f64, 0.0f64, 0usize);
    let mut bitwise_equal = 0;
    let mut next_row = 0;
    let mut greedy_count = 0;
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let prompt = case.get("prompt").unwrap().as_str().unwrap();
        let prompt_tokens = tokens(case.get("tokens").unwrap());
        assert_eq!(
            model.tokenize(prompt, true).unwrap(),
            prompt_tokens,
            "{name} token IDs"
        );
        let greedy = tokens(case.get("greedy_tokens").unwrap());
        assert_eq!(greedy.len(), 16);
        let mut sequence = prompt_tokens.clone();
        sequence.extend_from_slice(&greedy[..greedy.len() - 1]);
        let rows = model.logits_for_tokens(&sequence).unwrap();
        assert_eq!(rows.len(), sequence.len());
        assert_eq!(case.get("first_row").unwrap().as_u64(), Some(next_row));
        assert_eq!(
            case.get("row_count").unwrap().as_u64(),
            Some(rows.len() as u64)
        );
        for (row, actual) in rows.iter().enumerate() {
            let offset = (next_row as usize + row) * 32_000 * 4;
            let raw_row = &expected[offset..offset + 32_000 * 4];
            let values: Vec<f32> = raw_row
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            compare_values(
                actual,
                values.iter().copied().map(f64::from),
                name,
                row,
                &mut maxima,
            );
            bitwise_equal += actual
                .iter()
                .zip(&values)
                .filter(|(a, e)| a.to_bits() == e.to_bits())
                .count();
        }
        for (step, &token) in greedy.iter().enumerate() {
            assert_eq!(
                argmax_first(&rows[prompt_tokens.len() - 1 + step]),
                token,
                "{name} greedy step {step}"
            );
            greedy_count += 1;
        }
        next_row += rows.len() as u64;
        let mut generated = String::new();
        let stats = model
            .generate(
                prompt,
                &GenerationOptions {
                    max_tokens: 16,
                    temperature: 0.0,
                    seed: 0,
                    stop: Vec::new(),
                },
                &|| false,
                &mut |piece| {
                    generated.push_str(piece);
                    true
                },
            )
            .unwrap();
        assert_eq!(
            generated.as_bytes(),
            bytes_from_hex(case.get("generated_hex").unwrap().as_str().unwrap()),
            "{name} actual autoregressive emitted bytes"
        );
        assert_eq!(stats.prompt_tokens, prompt_tokens.len());
        assert_eq!(stats.completion_tokens, greedy.len());
        assert_eq!(stats.finish_reason, FinishReason::Length);
    }
    assert_eq!(next_row, 20);
    assert_eq!(greedy_count, 16);
    assert_eq!(maxima.2, 640_000);
    eprintln!(
        "native trained Q4_0 conformance: cases=1 greedy_ids={greedy_count} compared_logits={} max_abs={:.9e} max_fraction_of_tolerance={:.9e} bitwise_equal={bitwise_equal} (informational; portable gate uses declared tolerance)",
        maxima.2, maxima.0, maxima.1
    );
}

#[test]
#[ignore = "requires hash-pinned external SmolLM2-135M-Q4_K_M.gguf; set LLAMA_TEST_BPE_MODEL and run --release --ignored --nocapture"]
fn trained_smollm2_bpe_q4_k_full_logits_greedy_ids_and_text() {
    let path = std::env::var_os("LLAMA_TEST_BPE_MODEL")
        .expect("LLAMA_TEST_BPE_MODEL must point to the external SmolLM2-135M-Q4_K_M.gguf");
    let (length, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(length, 101_016_128);
    assert_eq!(
        hash, "e3131339bf4e8065265593d4fd8f7bb7ff2d3abff1edb5618aa1197b89cad9f5",
        "wrong external BPE model fixture"
    );
    let expected = source_profile(
        include_bytes!("../../../fixtures/inference/smollm2-q4-k-m-logits.f32le.bin") as &[u8],
        include_bytes!("../../../fixtures/inference/smollm2-q4-k-m-gnu-logits.f32le.bin")
            as &[u8],
    );
    assert_eq!(expected.len(), 6_684_672);
    assert_eq!(
        sha256(expected),
        source_profile(
            "62e6457578e3dd3c5f83f942ea24a4e47e66a82fd7336255393fa4d40f3104bc",
            "34fab2f5e1b5b7359c66e225f4adf0a583da043e2ae20738f395d6695bead980"
        )
    );
    let metadata = source_profile(
        include_str!("../../../fixtures/inference/smollm2-q4-k-m-metadata.json"),
        include_str!("../../../fixtures/inference/smollm2-q4-k-m-gnu-metadata.json"),
    );
    assert_eq!(
        sha256(metadata.as_bytes()),
        source_profile(
            "cbab26cf03d54648ff49621b1a19dcb2d91d13ebcbcf7ebc54cd32116ffce1f6",
            "3af505411ccd98b2c9623517e98c94eec96f95f8f693a760d78599ef0dc3050c"
        )
    );
    let reference = json::parse(metadata).unwrap();
    assert_eq!(
        reference.get("kv_type").and_then(Value::as_str),
        Some("F32")
    );
    assert_eq!(
        reference.get("flash_attention").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(reference.get("threads").and_then(Value::as_u64), Some(1));
    assert_eq!(
        reference.get("n_vocab").and_then(Value::as_u64),
        Some(49_152)
    );
    let declared = reference.get("comparison_tolerance").unwrap();
    assert_eq!(
        declared.get("absolute").unwrap().as_f64(),
        Some(LOGIT_ABSOLUTE)
    );
    assert_eq!(
        declared.get("relative").unwrap().as_f64(),
        Some(LOGIT_RELATIVE)
    );
    let model = Model::load_with_options(
        &path,
        &llama_inference::LoadOptions {
            context_length: 128,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.vocabulary_size(), 49_152);
    assert_eq!(
        compare_packed_decoder(&model, &reference, expected, LOGIT_ABSOLUTE, LOGIT_RELATIVE),
        (34, 24, 1_671_168)
    );
}

/// Shared evaluator for independently captured compact rows. Each caller pins
/// every fixture and supplies the numerical budget fixed before comparison.
fn compare_packed_decoder(
    model: &Model,
    reference: &Value,
    expected: &[u8],
    absolute: f64,
    relative: f64,
) -> (usize, usize, usize) {
    assert_eq!(
        reference.get("kv_type").and_then(Value::as_str),
        Some("F32")
    );
    assert_eq!(reference.get("threads").and_then(Value::as_u64), Some(1));
    assert_eq!(
        reference.get("flash_attention").and_then(Value::as_bool),
        Some(false)
    );
    let declared = reference.get("comparison_tolerance").unwrap();
    assert_eq!(declared.get("absolute").unwrap().as_f64(), Some(absolute));
    assert_eq!(declared.get("relative").unwrap().as_f64(), Some(relative));
    let vocabulary = model.vocabulary_size();
    assert_eq!(
        reference.get("n_vocab").and_then(Value::as_u64),
        Some(vocabulary as u64)
    );
    let expected_rows = reference.get("row_count").unwrap().as_u64().unwrap() as usize;
    assert_eq!(expected.len(), expected_rows * vocabulary * 4);
    let maximum = reference.get("max_tokens").unwrap().as_u64().unwrap() as usize;
    let cases = reference.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let mut maximum_absolute = 0.0_f64;
    let mut maximum_fraction = 0.0_f64;
    let mut compared = 0;
    let mut bitwise_equal = 0;
    let mut next_row = 0;
    let mut greedy_count = 0;
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let prompt = case.get("prompt").unwrap().as_str().unwrap();
        let prompt_tokens = tokens(case.get("tokens").unwrap());
        assert_eq!(
            model.tokenize(prompt, true).unwrap(),
            prompt_tokens,
            "{name} token IDs"
        );
        assert!(!prompt_tokens.is_empty());
        for (index, &token) in prompt_tokens.iter().enumerate() {
            for (special, field) in [(false, "pieces_hex"), (true, "pieces_special_hex")] {
                assert_eq!(
                    model.tokenizer().token_piece(token, special).unwrap(),
                    bytes_from_hex(
                        case.get(field).unwrap().as_array().unwrap()[index]
                            .as_str()
                            .unwrap()
                    ),
                    "{name} prompt piece {index}, special={special}"
                );
            }
        }
        assert_eq!(
            model
                .tokenizer()
                .detokenize_bytes(&prompt_tokens, true, false)
                .unwrap(),
            bytes_from_hex(case.get("detokenized_hex").unwrap().as_str().unwrap()),
            "{name} detokenized prompt"
        );
        let greedy = tokens(case.get("greedy_tokens").unwrap());
        assert!(!greedy.is_empty());
        assert!(greedy.len() <= maximum);
        let mut sequence = prompt_tokens.clone();
        sequence.extend_from_slice(&greedy[..greedy.len() - 1]);
        let rows = model.logits_for_tokens(&sequence).unwrap();
        assert_eq!(rows.len(), sequence.len());
        assert_eq!(
            case.get("first_row").unwrap().as_u64(),
            Some(next_row as u64)
        );
        assert_eq!(
            case.get("row_count").unwrap().as_u64(),
            Some(rows.len() as u64)
        );
        for (row, actual) in rows.iter().enumerate() {
            assert_eq!(actual.len(), vocabulary);
            let offset = (next_row + row) * vocabulary * 4;
            let raw_row = &expected[offset..offset + vocabulary * 4];
            for (column, (&actual, bytes)) in
                actual.iter().zip(raw_row.as_chunks::<4>().0).enumerate()
            {
                let native = f32::from_le_bytes(*bytes);
                let difference = (f64::from(actual) - f64::from(native)).abs();
                let allowed = absolute + relative * f64::from(native).abs();
                assert!(
                    actual.is_finite() && native.is_finite() && difference <= allowed,
                    "{name} row {row} logit {column}: actual={actual:.9e} reference={native:.9e} abs={difference:.9e} allowed={allowed:.9e}"
                );
                maximum_absolute = maximum_absolute.max(difference);
                maximum_fraction = maximum_fraction.max(difference / allowed);
                bitwise_equal += usize::from(actual.to_bits() == native.to_bits());
                compared += 1;
            }
        }
        for (step, &token) in greedy.iter().enumerate() {
            assert_eq!(
                argmax_first(&rows[prompt_tokens.len() - 1 + step]),
                token,
                "{name} greedy step {step}"
            );
            greedy_count += 1;
        }
        next_row += rows.len();
        let mut generated = String::new();
        let stats = model
            .generate(
                prompt,
                &GenerationOptions {
                    max_tokens: maximum,
                    temperature: 0.0,
                    seed: 0,
                    stop: Vec::new(),
                },
                &|| false,
                &mut |piece| {
                    generated.push_str(piece);
                    true
                },
            )
            .unwrap();
        assert_eq!(
            generated.as_bytes(),
            bytes_from_hex(case.get("generated_hex").unwrap().as_str().unwrap()),
            "{name} actual autoregressive emitted bytes"
        );
        assert_eq!(stats.prompt_tokens, prompt_tokens.len());
        assert_eq!(stats.completion_tokens, greedy.len());
        assert_eq!(
            stats.finish_reason,
            if case.get("finished_eog").unwrap().as_bool().unwrap() {
                FinishReason::Eos
            } else {
                FinishReason::Length
            }
        );
    }
    assert_eq!(next_row, expected_rows);
    eprintln!(
        "native {} conformance: cases={} rows={next_row} greedy_ids={greedy_count} compared_logits={compared} max_abs={maximum_absolute:.9e} max_fraction_of_tolerance={maximum_fraction:.9e} bitwise_equal={bitwise_equal} (informational; gate uses unchanged declared tolerance)",
        model.model_id(),
        cases.len()
    );
    (next_row, greedy_count, compared)
}

#[test]
#[ignore = "requires hash-pinned external mamba-130m-hf.f16.gguf; set LLAMA_TEST_MAMBA_MODEL and run --release --ignored --nocapture"]
fn trained_mamba130m_full_logits_greedy_ids_and_text() {
    let path = std::env::var_os("LLAMA_TEST_MAMBA_MODEL")
        .expect("LLAMA_TEST_MAMBA_MODEL must point to the external mamba-130m-hf.f16.gguf");
    let (length, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(length, 261_787_232);
    assert_eq!(
        hash,
        "cde77cf92d0fbee2346f342faf7f4992622b28d163277a7e52eefa0331be7e32"
    );
    let expected = source_profile(
        include_bytes!("../../../fixtures/inference/mamba-130m-f16-logits.f32le.bin") as &[u8],
        include_bytes!("../../../fixtures/inference/mamba-130m-f16-gnu-logits.f32le.bin")
            as &[u8],
    );
    assert_eq!(expected.len(), 6_838_080);
    assert_eq!(
        sha256(expected),
        source_profile(
            "464194ea61d22ed122a49f2dc193b5abb52cd3fbf8cb6b9da36ed2b182071b0b",
            "a6b70e05d1f71276bb499c0d532edd98106b23f31759cdc46719e863f8fdd9f7"
        )
    );
    let metadata = source_profile(
        include_str!("../../../fixtures/inference/mamba-130m-f16-metadata.json"),
        include_str!("../../../fixtures/inference/mamba-130m-f16-gnu-metadata.json"),
    );
    assert_eq!(
        sha256(metadata.as_bytes()),
        source_profile(
            "ab0aa643da3d8098b3f36733e76b3fdeb840c8ed7a191f258500e9dc6b779254",
            "15ed0b80b811055f9bc10f881147dab183addc2906cf79f2095b4c3f9a25bd31"
        )
    );
    let reference = json::parse(metadata).unwrap();
    let model = Model::load_with_options(
        &path,
        &llama_inference::LoadOptions {
            context_length: 128,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.vocabulary_size(), 50_280);
    assert_eq!(model.embedding_dimensions(), 768);
    assert!(model.supports_generation());
    assert_eq!(
        compare_packed_decoder(&model, &reference, expected, LOGIT_ABSOLUTE, LOGIT_RELATIVE),
        (34, 24, 1_709_520)
    );
}

#[test]
fn tiny_mamba_native_default_rms_and_untied_output() {
    struct Fixture {
        name: &'static str,
        model: &'static [u8],
        model_hash: &'static str,
        logits: &'static [u8],
        logits_hash: &'static str,
        metadata: &'static str,
        metadata_hash: &'static str,
    }
    let fixtures = [
        Fixture {
            name: "tiny-mamba-default",
            model: include_bytes!("../../../fixtures/inference/tiny-mamba-default.gguf"),
            model_hash: "50ec6a3787b9311f9a25b801180003b122009d442e72921d86a78532666aa003",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-mamba-default-logits.f32le.bin")
                    as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-mamba-default-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "0f34c8bf70fa0bc8328a2364d8deb5067e8e0808c6bc438518e6539d26b588cb",
                "0f34c8bf70fa0bc8328a2364d8deb5067e8e0808c6bc438518e6539d26b588cb",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-mamba-default-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-mamba-default-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "fc9a21785c2c443b67713bb933392e421414578c2955a37139d33e4583cdc520",
                "c42663f9eea3221a9f8c7772300e5f2b8697d517d76a5834fba3a0c79642c5d7",
            ),
        },
        Fixture {
            name: "tiny-mamba-rms",
            model: include_bytes!("../../../fixtures/inference/tiny-mamba-rms.gguf"),
            model_hash: "29bb3f8ed8e05c80676bb64a5d09d19a7ed8592240c494e3126cdaa3aacdfb85",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-mamba-rms-logits.f32le.bin")
                    as &[u8],
                include_bytes!("../../../fixtures/inference/tiny-mamba-rms-gnu-logits.f32le.bin")
                    as &[u8],
            ),
            logits_hash: source_profile(
                "6602807c99077508c545c185924694e6a045f50e15043473af0a945e0be209e5",
                "37d640fdf268c9ecdae5e77e699f7f467f394da5d5f936fd10a9d2ccd9918ef9",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-mamba-rms-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-mamba-rms-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "fc9a21785c2c443b67713bb933392e421414578c2955a37139d33e4583cdc520",
                "ec9199623d56503f278dd3c6cfcac9307f0521cb5b7f899214695c56a09683b4",
            ),
        },
        Fixture {
            name: "tiny-mamba-untied",
            model: include_bytes!("../../../fixtures/inference/tiny-mamba-untied.gguf"),
            model_hash: "e0dabebc8a3e44e5bda8fcc4e204034a0d9c302f7fd02a95ecf4fab0dce5b24f",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-mamba-untied-logits.f32le.bin")
                    as &[u8],
                include_bytes!("../../../fixtures/inference/tiny-mamba-untied-gnu-logits.f32le.bin")
                    as &[u8],
            ),
            logits_hash: source_profile(
                "fa3de6cd4943ede26829edbb8aa87831b25519c36cd80be4480c8ec97d29c4b1",
                "c01a67348101f46c723bf21e16e144a8b5bcc042cd2c7020d44f2348aa6c8445",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-mamba-untied-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-mamba-untied-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "53a89314effd547420dfa4d0e1fd88bca3812d409f129bb1e1027c9aa83da472",
                "49d4523c325424d5cf9105108dc3c9c30d14a7b1115d7a35bbc8068695ced007",
            ),
        },
    ];
    for fixture in fixtures {
        assert_eq!(sha256(fixture.model), fixture.model_hash);
        assert_eq!(sha256(fixture.logits), fixture.logits_hash);
        assert_eq!(sha256(fixture.metadata.as_bytes()), fixture.metadata_hash);
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/inference")
            .join(format!("{}.gguf", fixture.name));
        let model = Model::load(path).unwrap();
        assert_eq!(model.model_id(), fixture.name);
        assert_eq!(model.vocabulary_size(), 512);
        assert_eq!(model.context_length(), 32);
        assert_eq!(model.embedding_dimensions(), 4);
        assert!(model.supports_generation());
        assert!(!model.supports_embeddings());
        let reference = json::parse(fixture.metadata).unwrap();
        assert_eq!(
            compare_packed_decoder(&model, &reference, fixture.logits, 2e-6, 2e-6),
            (18, 12, 9_216)
        );
    }
}

#[test]
#[ignore = "requires hash-pinned external Qwen2.5-0.5B Q4_K_M; set LLAMA_TEST_QWEN2_MODEL and run --release --ignored --nocapture"]
fn trained_qwen2_05b_q4_k_full_logits_greedy_ids_and_text() {
    let path = std::env::var_os("LLAMA_TEST_QWEN2_MODEL")
        .expect("LLAMA_TEST_QWEN2_MODEL must point to qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let (length, hash) = hash_reader(File::open(&path).unwrap()).unwrap();
    assert_eq!(length, 491_400_032);
    assert_eq!(
        hash,
        "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
    );
    let expected = source_profile(
        include_bytes!("../../../fixtures/inference/qwen2-05b-q4-k-m-logits.f32le.bin") as &[u8],
        include_bytes!("../../../fixtures/inference/qwen2-05b-q4-k-m-gnu-logits.f32le.bin")
            as &[u8],
    );
    assert_eq!(expected.len(), 20663296);
    assert_eq!(
        sha256(expected),
        source_profile(
            "7d72a020440c058295accce093e220d6988d8d864ade9be89721f57e65110e9b",
            "41911148135e2fd5b04a2ef08645bed2db47d2aca93657137020b8fb14ea2e3d"
        )
    );
    let metadata = source_profile(
        include_str!("../../../fixtures/inference/qwen2-05b-q4-k-m-metadata.json"),
        include_str!("../../../fixtures/inference/qwen2-05b-q4-k-m-gnu-metadata.json"),
    );
    assert_eq!(
        sha256(metadata.as_bytes()),
        source_profile(
            "251a33d1f6a51faf83856b69d4bd14832c086d15d83b42fc17e9e8eded36faf0",
            "bdb3847e8c6f7cc3d5d81e3fb26cde6bc0c23f0a261fe081ba8188dab7e9c871"
        )
    );
    let reference = json::parse(metadata).unwrap();
    let model = Model::load_with_options(
        &path,
        &llama_inference::LoadOptions {
            context_length: 128,
            max_decoded_weight_bytes: 3 * 1024 * 1024 * 1024,
        },
    )
    .unwrap();
    assert_eq!(model.vocabulary_size(), 151_936);
    assert_eq!(model.embedding_dimensions(), 896);
    assert!(model.supports_generation());
    assert_eq!(
        compare_packed_decoder(&model, &reference, expected, LOGIT_ABSOLUTE, LOGIT_RELATIVE),
        (34, 24, 5_165_824)
    );
}

#[test]
fn tiny_qwen2_separate_fused_partial_bias_and_output_graphs() {
    struct Fixture {
        name: &'static str,
        model: &'static [u8],
        model_hash: &'static str,
        logits: &'static [u8],
        logits_hash: &'static str,
        metadata: &'static str,
        metadata_hash: &'static str,
        expected_counts: (usize, usize, usize),
        width: usize,
    }
    let fixtures = [
        Fixture {
            name: "tiny-qwen2-fused-bias",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-bias.gguf"),
            model_hash: "b9ee4bfdd5975a4acdeea484e9313a76ee51c1b673dd91c06d744c29ae83d572",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-bias-logits.f32le.bin")
                    as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-bias-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2a99a0237a52c031ce7d9988509680c0d482ec92764ebd45a8ca430b56ecb6f9",
                "d84d66700c57b0a72e839a0c9fbcff8e9054080444cb662ee4dfbb064f484015",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-fused-bias-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-qwen2-fused-bias-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "092d295a68a762ad4754b9d09170709b728945d3b1d68bfbab0d4a41e7baf2a0",
                "0f8730fd0f0ede13ec0d47334a642108b50dfd5312f581b535820ab27d33b136",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-no-bias",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-no-bias.gguf"),
            model_hash: "1f0c389081ab453dfc3029898957a8d8a6574bb3374be1990a47485e0cd0c212",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-no-bias-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-no-bias-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-fused-no-bias-metadata.json"),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-no-bias-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-partial-k",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-partial-k.gguf"),
            model_hash: "d2114ca3f712e713135632d3c64e8250b4b5f2e79f5413142667f0299c178c4e",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-k-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-k-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-k-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-k-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-partial-kv",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-partial-kv.gguf"),
            model_hash: "67b238898000d1529f281f46a5ca191f58fd55a5ea713ba545c0706fa2c91ee0",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-kv-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-kv-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-kv-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-kv-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-partial-q",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-partial-q.gguf"),
            model_hash: "58dbcbd4edb6c27cbdda671b6b5e7fb0433bccfa01e1690145b348cb9895c464",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-q-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-q-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-q-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-q-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-partial-qk",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-partial-qk.gguf"),
            model_hash: "e4d194774cf86d15466dc6bb8f868cc0fdbcad06b9dd2a719ce84b51976b23a4",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qk-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qk-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qk-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qk-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-partial-qv",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-partial-qv.gguf"),
            model_hash: "4b1e812c0c19fbcde674ef17ab4f9cb49a740475189693a2f9f61f9b46e10008",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qv-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qv-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qv-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-qv-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-partial-v",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-partial-v.gguf"),
            model_hash: "45fa8f9701971f670a888d85559ad501257c576ca8096d4d79b263d7fd978c3f",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-v-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-v-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-v-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-partial-v-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-fused-q4",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-q4.gguf"),
            model_hash: "45cdda6eaad3427e2a34f13f77319dae9585347a70f8e93fd8b54c4f217a76bc",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-qwen2-fused-q4-logits.f32le.bin")
                    as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-q4-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "682b59f2892ca1e93c1912e7ed8e60333b25cdab352bb8b3fc88b290e2b1074b",
                "682b59f2892ca1e93c1912e7ed8e60333b25cdab352bb8b3fc88b290e2b1074b",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-fused-q4-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-qwen2-fused-q4-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "379272934eae4109190464262d5a83dc2769d446cf8d168794547366b0724070",
                "375d06032612833e8808e18fa88a64a03081f458eb373f6dcd176e8f5acad552",
            ),
            expected_counts: (18, 12, 9216),
            width: 32,
        },
        Fixture {
            name: "tiny-qwen2-fused-separate-bias",
            model: include_bytes!(
                "../../../fixtures/inference/tiny-qwen2-fused-separate-bias.gguf"
            ),
            model_hash: "e983af0b7576ea6fd55a7c84c2a796d3099c0fb17d3bf7524b884eb9086167f8",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-separate-bias-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-fused-separate-bias-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2a99a0237a52c031ce7d9988509680c0d482ec92764ebd45a8ca430b56ecb6f9",
                "d84d66700c57b0a72e839a0c9fbcff8e9054080444cb662ee4dfbb064f484015",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-separate-bias-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-fused-separate-bias-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "092d295a68a762ad4754b9d09170709b728945d3b1d68bfbab0d4a41e7baf2a0",
                "0f8730fd0f0ede13ec0d47334a642108b50dfd5312f581b535820ab27d33b136",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-no-bias",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-no-bias.gguf"),
            model_hash: "95bcd31def9db2542498c500253690b8bea09e69c89328a92be419de7ffc9d45",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-qwen2-no-bias-logits.f32le.bin")
                    as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-no-bias-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2af0b0a115e92e476d3b847e63e5ba6c7eab2ccf1b09fed80e5f66fca02660f7",
                "2a32c9c4aee0feb35e1f7c2387ad51bd1cd844072603feeece11d4d66ba9010a",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-no-bias-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-qwen2-no-bias-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "9eb2dbf91802b8393184f876b860367b2719c98e71fc2c8f1a0617d72e0c379c",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-output-bias",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-output-bias.gguf"),
            model_hash: "f473595fc9dcd8c7c3d2a89f14ff0e1810f441429cf3880fa54d7738274ccda7",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-output-bias-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-output-bias-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "a97a9d4db74823cd25045ea4ad705d0ffde1002f64d3c106b473693623886967",
                "4bba80f3a2f6668b215b43bfa31df0bd9d1503213eea9131ce758e6e5df55844",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-output-bias-metadata.json"),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-output-bias-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "82221821af00746ea6d7026b73ccd9f4b64577ff0adb0a3c8eaabc46085d28ba",
                "23a9d75aaa0f488ba33e792f83e59fe611915c2d64eea5e4a3b090b5c524e862",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate-partial-k",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-separate-partial-k.gguf"),
            model_hash: "c25f0ecb72e01b9530ed9f18d69df5299c228f3cd014084c360f406a001ce191",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-k-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-k-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "0628efbbbc016cd9dd13859d1bc4eb94b2a89823cf7bd0facce434f94f045598",
                "9a14a2597ca4a1563dc5aa9650ef01d5986e0d23f78cb40389f147c332bd32d0",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-k-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-k-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "d87edaf9423dc7cf1dcdbc8ea5b396ffff80cbd6df0b1cf4a5561764439665ea",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate-partial-kv",
            model: include_bytes!(
                "../../../fixtures/inference/tiny-qwen2-separate-partial-kv.gguf"
            ),
            model_hash: "45369b1631d9dc2c3e5c9d1409e287f05a80dbfedee492907aa2e7e56ba6be5b",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-kv-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-kv-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "481ae86553f9ec9b572c754bf71a91e960c40abbe1f8a922516ff3bdb358b2d7",
                "7ae4cd820f507698609139b894a347a1a25a6d53b1f26a56dc42b8ff0a6c716d",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-kv-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-kv-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "247522218bcc291e21e134e9204819dca31c6c1992bd4cf37411e8b9f1405d33",
                "1abc4a81dab409adbeda51f0e1f8b15cfed61fc40e9eec03fe092c06fd7b2a7b",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate-partial-q",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-separate-partial-q.gguf"),
            model_hash: "a802cef6ad95ce5c29a6650ecbef45135be2bc0b54625292bc407990d7765ee5",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-q-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-q-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "5821ddc450ce416834eb5ef2af4c41ba97972fa0eebc9d84f9fde3c45d303b48",
                "4a36be6ddbbb6b170e4b87443f9553f8c5bd95a9bd18ec0d865027ebaeae4ac8",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-q-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-q-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "6f02ae8449232c0bf5fd54ec33c09e6197ed186bf355fa1fa1d5261e37e8d1cb",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate-partial-qk",
            model: include_bytes!(
                "../../../fixtures/inference/tiny-qwen2-separate-partial-qk.gguf"
            ),
            model_hash: "61ba0279c0924d380baa26dc8abf707a8e8eb7f73e51158bfc4ae6c2ac3eb538",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qk-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qk-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "c4f15c3b336282de7ae3fa260238ed4268c0c29a30af9af7cd9b9d938d6e73bc",
                "0e0c00ee05871938592f4b003bcd67312c5c759ccb23659f645ef21682d1e1c8",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qk-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qk-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "c2bb0d7276c8b5e63437b7dae31aba2ab0e61e97b49d7c8cdcfe398601abbb8a",
                "1c7e6dfd3835ad1d90f824f0c54cf95b11134fb8295d2719dff7a501f5005840",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate-partial-qv",
            model: include_bytes!(
                "../../../fixtures/inference/tiny-qwen2-separate-partial-qv.gguf"
            ),
            model_hash: "783bdef7b6630b02abe5191434c92fce502389e47502ad9c66ae086bd49c1205",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qv-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qv-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "5a244e40577e22d2a073958f42af585316b156cbd39de82e4d1053f58b8b6a71",
                "02a8364ade77b3ad515d46037f54c38057665fad4c950d26a6852c4d601d4ee0",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qv-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-qv-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "092d295a68a762ad4754b9d09170709b728945d3b1d68bfbab0d4a41e7baf2a0",
                "b59cbe82bc820652d7e3cf5a91ef30d80c956475f2e537c912078a549dbd3a28",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate-partial-v",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-separate-partial-v.gguf"),
            model_hash: "c01a7178aa883ede5c016f50f2a30a53c3e9e456f5edc28166948f9d682e8700",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-v-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-v-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "ef1b00a222c655ea5846916841feb9a2dfc027d422b3469a4f54d8a8427da00d",
                "fb527f99fb999eb029abec6d0b47ee91353ece175af6d06ed3e92b3afe5c17e5",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-v-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-separate-partial-v-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "247522218bcc291e21e134e9204819dca31c6c1992bd4cf37411e8b9f1405d33",
                "ee3e5875ff5023b434c5968d7b1623610fa07613aec555b3c63954b8b8e33524",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-separate",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-separate.gguf"),
            model_hash: "8ab2b140510fdb657b1c166b74148f153c6ebc943c52766ab82903104057d28f",
            logits: source_profile(
                include_bytes!("../../../fixtures/inference/tiny-qwen2-separate-logits.f32le.bin")
                    as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-separate-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "2a99a0237a52c031ce7d9988509680c0d482ec92764ebd45a8ca430b56ecb6f9",
                "d84d66700c57b0a72e839a0c9fbcff8e9054080444cb662ee4dfbb064f484015",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-separate-metadata.json"),
                include_str!("../../../fixtures/inference/tiny-qwen2-separate-gnu-metadata.json"),
            ),
            metadata_hash: source_profile(
                "092d295a68a762ad4754b9d09170709b728945d3b1d68bfbab0d4a41e7baf2a0",
                "0f8730fd0f0ede13ec0d47334a642108b50dfd5312f581b535820ab27d33b136",
            ),
            expected_counts: (15, 9, 7680),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-untied-output-bias",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-untied-output-bias.gguf"),
            model_hash: "27536ee416bac393c9be51cdc90cecc7dc381a2f4453c1bcce22ef057963e92e",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-bias-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-bias-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "88bb25234162f7c64b46ac8e1f76e947c2fc3042494b01d6d46432dab404f366",
                "13574e9b8dd22b292f28ef55944354cca2eac16f65b7947b738c0e8ffe06ee0d",
            ),
            metadata: source_profile(
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-bias-metadata.json"
                ),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-bias-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "3814361ca97cf3a36b6a51b236f43303386c14bb8ebacd32b5fcfc7e67c15d9b",
                "a31ad82c860a74379a9a1e58673558f08ee75b9d96b484e02c70c7adc74a9f6c",
            ),
            expected_counts: (18, 12, 9216),
            width: 8,
        },
        Fixture {
            name: "tiny-qwen2-untied-output",
            model: include_bytes!("../../../fixtures/inference/tiny-qwen2-untied-output.gguf"),
            model_hash: "d6603fcc029d40af24e0be477e94d3e60a96e5fa3da86ba8a78376fb5e342f28",
            logits: source_profile(
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-logits.f32le.bin"
                ) as &[u8],
                include_bytes!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-gnu-logits.f32le.bin"
                ) as &[u8],
            ),
            logits_hash: source_profile(
                "f4ff293121ab5c1c3b927e3b95e820fc9d1386511364efab0b9e6dd05e59b6ea",
                "b96446496c1c1d24ef33438e0fdccea27726ac3d7584770a445648a17fdd63d9",
            ),
            metadata: source_profile(
                include_str!("../../../fixtures/inference/tiny-qwen2-untied-output-metadata.json"),
                include_str!(
                    "../../../fixtures/inference/tiny-qwen2-untied-output-gnu-metadata.json"
                ),
            ),
            metadata_hash: source_profile(
                "20ee0a4487f672d869800cbf562a8a0d20184e5812ee148b5e9b7379a031dad1",
                "838b2e788ca6541383db5a4624d741114308c50b9511b6f869c904731f1885fd",
            ),
            expected_counts: (18, 12, 9216),
            width: 8,
        },
    ];
    for fixture in fixtures {
        assert_eq!(sha256(fixture.model), fixture.model_hash);
        assert_eq!(sha256(fixture.logits), fixture.logits_hash);
        assert_eq!(sha256(fixture.metadata.as_bytes()), fixture.metadata_hash);
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/inference")
            .join(format!("{}.gguf", fixture.name));
        let model = Model::load(path).unwrap();
        assert_eq!(model.model_id(), fixture.name);
        assert_eq!(model.vocabulary_size(), 512);
        assert_eq!(model.context_length(), 32);
        assert_eq!(model.embedding_dimensions(), fixture.width);
        assert!(model.supports_generation());
        assert!(!model.supports_embeddings());
        let reference = json::parse(fixture.metadata).unwrap();
        assert_eq!(
            compare_packed_decoder(&model, &reference, fixture.logits, 2e-6, 2e-6),
            fixture.expected_counts
        );
    }
}
