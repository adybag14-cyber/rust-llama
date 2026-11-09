//! Partial, native Rust implementation of the frozen llama.cpp public C ABI.
//!
//! This crate never links a llama/ggml library. See COVERAGE.md for exported and
//! unavailable operations and the accepted execution configuration.
//!
//! # Foreign pointer contract
//! All non-null pointers must reference live, correctly aligned allocations of
//! the documented size. Writable outputs must be exclusive for the call and must
//! not overlap inputs. Strings marked NUL-terminated must contain a terminator.
//! Handles must originate from this library; owning handles are freed exactly
//! once, and borrowed model/vocabulary/output pointers must not be freed.
//! Context operations require exclusive access (including getters while decoding).
//! Callbacks must not unwind, free/reenter their active handle, or retain transient
//! pointers. The library catches Rust panics, but cannot validate arbitrary foreign
//! allocations or recover from allocator aborts and access violations.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod abi;
mod batch;
mod context;
mod model;
mod sampler;
mod sampler_grammar;
mod sampler_vocab;
mod vocab;
pub use abi::*;
pub use context::ContextHandle;
pub use model::{ModelHandle, VocabHandle};

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

type Result<T> = std::result::Result<T, String>;
const MAX_FOREIGN_ELEMENTS: usize = 16 * 1024 * 1024;
thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}
fn set_error(message: &str) {
    LAST_ERROR.with(|slot| {
        // Replacing interior NULs keeps diagnostics valid C strings.
        *slot.borrow_mut() = CString::new(message.replace('\0', "\\0")).unwrap_or_default();
    });
}
fn boundary<T>(fallback: T, action: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            set_error(&message);
            fallback
        }
        Err(_) => {
            set_error("Rust panic caught at llama C ABI boundary");
            fallback
        }
    }
}

/// Every exported pointer-taking function uses the crate-level foreign contract.
macro_rules! ffi {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty, $fallback:expr, $body:block) => {
        #[doc = "# Safety\nThe foreign pointer and concurrency contract in the crate documentation applies."]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($($arg:$ty),*) -> $ret {
            crate::boundary($fallback, || $body)
        }
    };
}
pub(crate) use ffi;

unsafe fn reference<'a, T>(value: *const T) -> Result<&'a T> {
    if value.is_null() || !(value as usize).is_multiple_of(align_of::<T>()) {
        return Err("null or misaligned foreign pointer".into());
    }
    // SAFETY: remaining validity and lifetime requirements belong to the caller.
    Ok(unsafe { &*value })
}
unsafe fn reference_mut<'a, T>(value: *mut T) -> Result<&'a mut T> {
    if value.is_null() || !(value as usize).is_multiple_of(align_of::<T>()) {
        return Err("null or misaligned foreign pointer".into());
    }
    // SAFETY: caller grants exclusive access to this live object for the call.
    Ok(unsafe { &mut *value })
}
fn count(value: i32) -> Result<usize> {
    let n = usize::try_from(value).map_err(|_| "negative foreign element count")?;
    if n > MAX_FOREIGN_ELEMENTS {
        return Err("foreign element count exceeds safety limit".into());
    }
    Ok(n)
}
fn boolean(value: u8) -> Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err("C bool must be 0 or 1".into()),
    }
}
unsafe fn input_slice<'a, T>(value: *const T, n: usize) -> Result<&'a [T]> {
    if n == 0 {
        return Ok(&[]);
    }
    if n > MAX_FOREIGN_ELEMENTS || n > isize::MAX as usize / size_of::<T>().max(1) {
        return Err("foreign slice exceeds supported limit".into());
    }
    // SAFETY: validates alignment/non-null; caller supplies the whole allocation.
    unsafe {
        reference(value)?;
    }
    Ok(unsafe { slice::from_raw_parts(value, n) })
}
unsafe fn c_string<'a>(value: *const c_char) -> Result<&'a str> {
    if value.is_null() {
        return Err("null C string".into());
    }
    // SAFETY: caller promises a live NUL-terminated string.
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|_| "C string is not UTF-8".into())
}
unsafe fn copy_buffer<T: Copy>(values: &[T], target: *mut T, capacity: i32) -> Result<i32> {
    let cap = count(capacity)?;
    let required = i32::try_from(values.len()).map_err(|_| "result exceeds INT32_MAX")?;
    if values.len() > cap {
        return Ok(-required);
    }
    if !values.is_empty() {
        // SAFETY: caller provides exclusive capacity elements, disjoint from values.
        unsafe {
            reference_mut(target)?;
            ptr::copy_nonoverlapping(values.as_ptr(), target, values.len());
        }
    }
    Ok(required)
}
unsafe fn copy_string(value: &str, target: *mut c_char, capacity: usize) -> Result<i32> {
    if capacity > 0 {
        unsafe {
            reference_mut(target)?;
        }
        let n = value.len().min(capacity - 1);
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), target.cast(), n);
            *target.add(n) = 0;
        }
    }
    i32::try_from(value.len()).map_err(|_| "string length exceeds INT32_MAX".into())
}

ffi!(llama_rust_last_error() -> *const c_char, ptr::null(), {
    Ok(LAST_ERROR.with(|slot| slot.borrow().as_ptr()))
});
ffi!(llama_rust_clear_error() -> (), (), { set_error(""); Ok(()) });
ffi!(llama_rust_abi_layout(index: usize) -> usize, usize::MAX, {
    Ok(abi::layout(index))
});
ffi!(llama_version() -> *const c_char, ptr::null(), {
    Ok(c"rust-llama 0.1.0; partial frozen llama.cpp C ABI".as_ptr())
});
ffi!(llama_backend_init() -> (), (), { Ok(()) });
ffi!(llama_backend_free() -> (), (), { Ok(()) });
ffi!(llama_model_default_params() -> LlamaModelParams, LlamaModelParams::default(), {
    Ok(LlamaModelParams::default())
});
ffi!(llama_context_default_params() -> LlamaContextParams, LlamaContextParams::default(), {
    Ok(LlamaContextParams::default())
});
ffi!(llama_time_us() -> i64, 0, {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    Ok(EPOCH.get_or_init(Instant::now).elapsed().as_micros().min(i64::MAX as u128) as i64)
});
ffi!(llama_max_devices() -> usize, 0, { Ok(16) });
ffi!(llama_max_parallel_sequences() -> usize, 0, { Ok(256) });
ffi!(llama_max_tensor_buft_overrides() -> usize, 0, { Ok(4096) });
ffi!(llama_supports_mmap() -> bool, false, { Ok(false) });
ffi!(llama_supports_mlock() -> bool, false, { Ok(false) });
ffi!(llama_supports_gpu_offload() -> bool, false, { Ok(false) });
ffi!(llama_supports_rpc() -> bool, false, { Ok(false) });

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn panic_is_contained_and_diagnostic_is_thread_local() {
        assert_eq!(
            boundary(-7, || -> Result<i32> { panic!("contained test") }),
            -7
        );
        let message = unsafe { CStr::from_ptr(llama_rust_last_error()) }
            .to_str()
            .unwrap();
        assert!(message.contains("panic caught"));
        std::thread::spawn(|| {
            assert_eq!(
                unsafe { CStr::from_ptr(llama_rust_last_error()) }.to_bytes(),
                b""
            );
            set_error("worker only");
        })
        .join()
        .unwrap();
        assert!(
            unsafe { CStr::from_ptr(llama_rust_last_error()) }
                .to_str()
                .unwrap()
                .contains("panic caught")
        );
    }
    #[test]
    fn count_only_and_truncated_buffers_follow_c_conventions() {
        unsafe {
            assert_eq!(copy_buffer(&[1i32, 2, 3], ptr::null_mut(), 0).unwrap(), -3);
            let mut out = [0; 2];
            assert_eq!(copy_buffer(&[4i32, 5, 6], out.as_mut_ptr(), 2).unwrap(), -3);
            assert_eq!(out, [0, 0]);
            let mut chars = [99i8; 3];
            assert_eq!(copy_string("abcd", chars.as_mut_ptr(), 3).unwrap(), 4);
            assert_eq!(chars, [97, 98, 0]);
            assert!(copy_buffer::<u8>(&[], ptr::null_mut(), -1).is_err());
        }
    }
}

#[cfg(test)]
mod sampler_reference_tests;
