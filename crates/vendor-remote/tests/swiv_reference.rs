//! Frozen-original generated ELF fixtures; metadata/hashes in fixtures/manifest.json.
use llama_vendor_remote::dsp::{ElfImage, append_swiv};
#[test]
fn elf32_swiv_matches_original_bytes_including_two_bss_tails() {
    let input = include_bytes!("fixtures/hexagon32-input.elf");
    let expected = include_bytes!("fixtures/hexagon32-reference-swiv.elf");
    let actual = append_swiv(input).unwrap();
    assert_eq!(actual.as_slice(), expected);
    assert!(ElfImage::parse(&actual).unwrap().verify_swiv().unwrap());
}
#[test]
fn elf64_swiv_matches_original_bytes_when_header_is_in_load_segment() {
    let input = include_bytes!("fixtures/hexagon64-input.elf");
    let expected = include_bytes!("fixtures/hexagon64-reference-swiv.elf");
    let actual = append_swiv(input).unwrap();
    assert_eq!(actual.as_slice(), expected);
    assert!(ElfImage::parse(&actual).unwrap().verify_swiv().unwrap());
}
