//! Expected values were captured from the frozen original C API before the
//! Rust model was evaluated. See fixtures/inference/oracle-provenance.json.
//! The pretrained model test is explicit and opt-in because model weights are
//! external test inputs, not source dependencies or bundled product data.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::{hash_reader, sha256};
use llama_gguf::{Gguf, Limits};
use llama_inference::{FinishReason, GenerationOptions, Model};
use llama_tokenizer::Tokenizer;
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
    let reference = json::parse(include_str!(
        "../../../fixtures/inference/stories260k-reference.json"
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
    let expected = include_bytes!("../../../fixtures/inference/stories15m-q4-logits.f32le.bin");
    assert_eq!(expected.len(), 2_560_000);
    assert_eq!(
        sha256(expected),
        "579a521b36d682c9b26cac4d533b00beb4803e3bf5bfe2f86a29a07a4959fd21",
        "native Q4_0 expected logits drifted"
    );
    let reference = json::parse(include_str!(
        "../../../fixtures/inference/stories15m-q4-metadata.json"
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
    let expected = include_bytes!("../../../fixtures/inference/smollm2-q4-k-m-logits.f32le.bin");
    assert_eq!(expected.len(), 6_684_672);
    assert_eq!(
        sha256(expected),
        "62e6457578e3dd3c5f83f942ea24a4e47e66a82fd7336255393fa4d40f3104bc"
    );
    let metadata = include_str!("../../../fixtures/inference/smollm2-q4-k-m-metadata.json");
    assert_eq!(
        sha256(metadata.as_bytes()),
        "cbab26cf03d54648ff49621b1a19dcb2d91d13ebcbcf7ebc54cd32116ffce1f6"
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
    let expected = include_bytes!("../../../fixtures/inference/mamba-130m-f16-logits.f32le.bin");
    assert_eq!(expected.len(), 6_838_080);
    assert_eq!(
        sha256(expected),
        "464194ea61d22ed122a49f2dc193b5abb52cd3fbf8cb6b9da36ed2b182071b0b"
    );
    let metadata = include_str!("../../../fixtures/inference/mamba-130m-f16-metadata.json");
    assert_eq!(
        sha256(metadata.as_bytes()),
        "ab0aa643da3d8098b3f36733e76b3fdeb840c8ed7a191f258500e9dc6b779254"
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
            logits: include_bytes!(
                "../../../fixtures/inference/tiny-mamba-default-logits.f32le.bin"
            ),
            logits_hash: "0f34c8bf70fa0bc8328a2364d8deb5067e8e0808c6bc438518e6539d26b588cb",
            metadata: include_str!("../../../fixtures/inference/tiny-mamba-default-metadata.json"),
            metadata_hash: "fc9a21785c2c443b67713bb933392e421414578c2955a37139d33e4583cdc520",
        },
        Fixture {
            name: "tiny-mamba-rms",
            model: include_bytes!("../../../fixtures/inference/tiny-mamba-rms.gguf"),
            model_hash: "29bb3f8ed8e05c80676bb64a5d09d19a7ed8592240c494e3126cdaa3aacdfb85",
            logits: include_bytes!("../../../fixtures/inference/tiny-mamba-rms-logits.f32le.bin"),
            logits_hash: "6602807c99077508c545c185924694e6a045f50e15043473af0a945e0be209e5",
            metadata: include_str!("../../../fixtures/inference/tiny-mamba-rms-metadata.json"),
            metadata_hash: "fc9a21785c2c443b67713bb933392e421414578c2955a37139d33e4583cdc520",
        },
        Fixture {
            name: "tiny-mamba-untied",
            model: include_bytes!("../../../fixtures/inference/tiny-mamba-untied.gguf"),
            model_hash: "e0dabebc8a3e44e5bda8fcc4e204034a0d9c302f7fd02a95ecf4fab0dce5b24f",
            logits: include_bytes!(
                "../../../fixtures/inference/tiny-mamba-untied-logits.f32le.bin"
            ),
            logits_hash: "fa3de6cd4943ede26829edbb8aa87831b25519c36cd80be4480c8ec97d29c4b1",
            metadata: include_str!("../../../fixtures/inference/tiny-mamba-untied-metadata.json"),
            metadata_hash: "53a89314effd547420dfa4d0e1fd88bca3812d409f129bb1e1027c9aa83da472",
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
