//! Narrow bindings to standard operating-system scalar math.
//!
//! The original CPU GELU_ERF calls the OS C-runtime erff. These declarations
//! preserve that same external boundary; no llama/ggml or vendor math library is
//! loaded. Only the two independently exercised dynamic-CRT target profiles
//! select this provider. Other targets, including static-CRT configurations,
//! return None so callers can select their documented pure Rust fallback.
//!
//! Numerical evidence assumes nearest-even rounding and gradual underflow.
//! This API does not reproduce or reset floating exception status flags.

/// The standard OS math provider selected at build time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErfProvider {
    /// Windows Universal C Runtime, linked through its ucrt import library.
    WindowsUcrt,
    /// GNU/Linux system libm, linked with -lm.
    GnuLibm,
}

pub const fn erf_provider() -> Option<ErfProvider> {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc",
        not(target_feature = "crt-static")
    )) {
        Some(ErfProvider::WindowsUcrt)
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu",
        not(target_feature = "crt-static")
    )) {
        Some(ErfProvider::GnuLibm)
    } else {
        None
    }
}

/// Call the platform's C float erff(float) when its ABI/linkage profile is
/// exercised. Every F32 bit pattern is a valid argument, including infinities,
/// signed zero and NaNs. This call owns no memory or external handle.
pub fn erf_f32(value: f32) -> Option<f32> {
    #[cfg(all(
        target_arch = "x86_64",
        any(
            all(target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", target_env = "gnu")
        ),
        not(target_feature = "crt-static")
    ))]
    {
        // SAFETY: erff has this exact C ABI on the two selected targets and
        // accepts every C float value. The system import remains valid for the
        // process lifetime. No pointer, alias, allocation or callback crosses
        // the boundary; the result is an ordinary F32 value.
        Some(unsafe { system::c_erff(value) })
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        any(
            all(target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", target_env = "gnu")
        ),
        not(target_feature = "crt-static")
    )))]
    {
        let _ = value;
        None
    }
}

#[cfg(all(
    target_arch = "x86_64",
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu")
    ),
    not(target_feature = "crt-static")
))]
mod system {
    #[cfg(target_os = "windows")]
    #[link(name = "ucrt")]
    unsafe extern "C" {
        #[link_name = "erff"]
        pub(super) fn c_erff(value: f32) -> f32;
    }

    #[cfg(target_os = "linux")]
    #[link(name = "m")]
    unsafe extern "C" {
        #[link_name = "erff"]
        pub(super) fn c_erff(value: f32) -> f32;
    }
}
