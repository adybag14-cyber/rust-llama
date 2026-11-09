#![cfg(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    )
))]
use llama_quant::cpu_math::{MathPolicy, gelu_quick_f32_with_policy};

#[test]
fn scalar_quick_gelu_matches_original_inline_function_on_this_host() {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("fixtures/quick-scalar-msvc-reference.bin").as_slice();
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("fixtures/quick-scalar-gnu-reference.bin").as_slice();
    assert_eq!(&bytes[..8], b"QSCL0001");
    let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    assert_eq!(count, 502_058);
    assert_eq!(bytes.len(), 16 + count * 8);
    let source = &bytes[16..16 + count * 4];
    let expected = &bytes[16 + count * 4..];
    let mut values = source
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect::<Vec<_>>();
    gelu_quick_f32_with_policy(&mut values, MathPolicy::Scalar);
    let mut different = 0;
    let mut examples = Vec::new();
    for (index, (actual, expected)) in values.iter().zip(expected.as_chunks::<4>().0).enumerate() {
        let expected = u32::from_le_bytes(*expected);
        if actual.to_bits() != expected {
            different += 1;
            if examples.len() < 16 {
                examples.push((
                    index,
                    u32::from_le_bytes(source[index * 4..index * 4 + 4].try_into().unwrap()),
                    actual.to_bits(),
                    expected,
                ));
            }
        }
    }
    assert_eq!(
        different, 0,
        "scalar quick GELU mismatches(index,input,actual,expected):{examples:08x?}"
    );
    println!("Original scalar quick GELU: {count} raw float output words exact");
}
