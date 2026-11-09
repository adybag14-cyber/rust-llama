//! Select independently captured output for the frozen compiler/target profile.
//! GNU x86_64 defines __SSE2__ for the baseline build; the reference MSVC
//! baseline takes scalar row branches. Keep this selector independent of the
//! implementation under test so a runtime dispatch regression cannot select
//! a different expected corpus and conceal itself.
pub fn source_profile<T>(msvc: T, gnu: T) -> T {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        gnu
    } else {
        msvc
    }
}
