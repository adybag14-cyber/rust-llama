//! The expected words were produced by an independently compiled native C++
//! call to the operating system erff before this Rust boundary was implemented.
#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    ),
    not(target_feature = "crt-static")
))]
use llama_platform_loader::math::{ErfProvider, erf_f32, erf_provider};

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    ),
    not(target_feature = "crt-static")
))]
#[test]
fn native_system_erf_matches_all_captured_raw_float_words() {
    let (expected_provider, bytes): (ErfProvider, &[u8]) = {
        #[cfg(all(
            target_arch = "x86_64",
            target_os = "windows",
            target_env = "msvc",
            not(target_feature = "crt-static")
        ))]
        {
            (
                ErfProvider::WindowsUcrt,
                include_bytes!("fixtures/system-erf-msvc-reference.bin"),
            )
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_os = "linux",
            target_env = "gnu",
            not(target_feature = "crt-static")
        ))]
        {
            (
                ErfProvider::GnuLibm,
                include_bytes!("fixtures/system-erf-gnu-reference.bin"),
            )
        }
    };
    assert_eq!(erf_provider(), Some(expected_provider));
    assert_eq!(&bytes[..8], b"ERFSYS01");
    let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    assert_eq!(count, 502_058);
    assert_eq!(bytes.len(), 16 + count * 8);
    let inputs = &bytes[16..16 + count * 4];
    let expected = &bytes[16 + count * 4..];
    let mut mismatches = 0;
    let mut examples = Vec::new();
    for (index, (input, expected)) in inputs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<4>().0)
        .enumerate()
    {
        let input = u32::from_le_bytes(*input);
        let expected = u32::from_le_bytes(*expected);
        let actual = erf_f32(f32::from_bits(input)).unwrap().to_bits();
        if actual != expected {
            mismatches += 1;
            if examples.len() < 16 {
                examples.push((index, input, expected, actual));
            }
        }
    }
    assert_eq!(mismatches, 0, "raw-word mismatches: {examples:08x?}");
    eprintln!("system erff provider {expected_provider:?}: {count} raw-word comparisons exact");
}

#[cfg(not(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    ),
    not(target_feature = "crt-static")
)))]
#[test]
fn unverified_or_static_crt_targets_decline_the_native_provider() {
    assert_eq!(llama_platform_loader::math::erf_provider(), None);
    assert_eq!(llama_platform_loader::math::erf_f32(0.0), None);
}
