use llama_media::{
    DecodeLimits, MediaError,
    image::{ImageSize, decode},
};
use std::path::Path;

#[test]
fn png_color_depths_filters_and_original_fixture_match_stb_bytes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/t09/png");
    let mut checked = 0usize;
    let mut bytes = 0usize;
    for row in include_str!("../../../fixtures/t09/png/corpus.tsv")
        .lines()
        .skip(1)
    {
        let columns: Vec<&str> = row.split('\t').collect();
        let input = std::fs::read(root.join(columns[0])).unwrap();
        let expected = std::fs::read(root.join(format!("{}.rgb", columns[0]))).unwrap();
        let image = decode(&input, DecodeLimits::default())
            .unwrap_or_else(|e| panic!("{}: {e}", columns[0]));
        assert_eq!(
            image.size(),
            ImageSize::new(columns[1].parse().unwrap(), columns[2].parse().unwrap()).unwrap()
        );
        assert_eq!(image.data(), expected, "{}", columns[0]);
        checked += 1;
        bytes += expected.len();
    }
    eprintln!("oracle PNG: {checked} input fixtures, {bytes} exact RGB bytes");
    assert_eq!(checked, 16);
}

#[test]
fn png_truncation_crc_and_limits_are_enforced() {
    let input = include_bytes!("../../../fixtures/t09/png/color6-depth16.png");
    for end in 0..input.len() {
        assert!(
            decode(&input[..end], DecodeLimits::default()).is_err(),
            "prefix {end}"
        );
    }
    for offset in 8..input.len() {
        let mut bad = input.to_vec();
        bad[offset] ^= 0x40;
        assert!(
            decode(&bad, DecodeLimits::default()).is_err(),
            "corruption at {offset}"
        );
    }
    assert!(matches!(
        decode(
            input,
            DecodeLimits {
                max_image_pixels: 44,
                ..Default::default()
            }
        ),
        Err(MediaError::Limit { .. })
    ));
    let large = include_bytes!("../../../fixtures/t09/png/original-test-1-positive.png");
    for end in [8, 29, 33, 1000, large.len() / 2, large.len() - 1] {
        assert!(decode(&large[..end], DecodeLimits::default()).is_err());
    }
}
