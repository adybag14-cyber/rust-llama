//! Small external-driver loader shared by this scoped progress workspace.
//!
//! This is an integration seam pending T01 platform integration. It never loads
//! llama/ggml and is intentionally not Send or Sync. Symbols must not outlive
//! the library that owns them.
pub mod math;

use std::ffi::{CStr, CString, c_void};
use std::fmt;
use std::ptr::NonNull;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError(pub String);
impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for LoadError {}

#[derive(Debug)]
pub struct DynamicLibrary {
    handle: NonNull<c_void>,
    name: String,
}

impl DynamicLibrary {
    /// Try an explicit list of system library names. No search path mutation.
    pub fn open(candidates: &[&str]) -> Result<Self, LoadError> {
        let mut failures = Vec::new();
        for name in candidates {
            let c_name =
                CString::new(*name).map_err(|_| LoadError("NUL in library name".into()))?;
            match native::open(&c_name) {
                Ok(handle) => {
                    return Ok(Self {
                        handle,
                        name: (*name).into(),
                    });
                }
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        Err(LoadError(if failures.is_empty() {
            "no library candidates supplied".into()
        } else {
            failures.join("; ")
        }))
    }
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Resolve a vendor function or data pointer.
    ///
    /// # Safety
    /// `T` must be the exact ABI-compatible pointer type for `name`. It must not
    /// be used after this library is dropped. Calling a function also requires
    /// obeying the vendor's argument, thread and lifetime contract.
    pub unsafe fn symbol<T: Copy>(&self, name: &CStr) -> Result<T, LoadError> {
        if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
            return Err(LoadError("symbol type must have pointer size".into()));
        }
        let pointer = native::symbol(self.handle, name)?;
        // SAFETY: the caller promises the exact ABI pointer type, and the size
        // check above guarantees that transmute_copy reads no extra bytes.
        Ok(unsafe { std::mem::transmute_copy(&pointer) })
    }
}
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        native::close(self.handle);
    }
}

#[cfg(unix)]
mod native {
    use super::*;
    use std::ffi::{c_char, c_int};
    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }
    fn last_error() -> String {
        // SAFETY: dlerror returns a thread-local C string valid until the next
        // loader call on this thread; we copy it immediately.
        unsafe {
            let error = dlerror();
            if error.is_null() {
                "dynamic loader returned a null handle".into()
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        }
    }
    pub fn open(name: &CStr) -> Result<NonNull<c_void>, LoadError> {
        // SAFETY: NUL-terminated name, RTLD_NOW | RTLD_LOCAL, owned result.
        NonNull::new(unsafe { dlopen(name.as_ptr(), 2) }).ok_or_else(|| LoadError(last_error()))
    }
    pub fn symbol(handle: NonNull<c_void>, name: &CStr) -> Result<*mut c_void, LoadError> {
        // SAFETY: handle is kept alive by DynamicLibrary and name is CStr.
        let pointer = unsafe {
            dlerror();
            dlsym(handle.as_ptr(), name.as_ptr())
        };
        if pointer.is_null() {
            Err(LoadError(last_error()))
        } else {
            Ok(pointer)
        }
    }
    pub fn close(handle: NonNull<c_void>) {
        // SAFETY: this is the single close of a successfully opened handle.
        unsafe {
            dlclose(handle.as_ptr());
        }
    }
}

#[cfg(windows)]
mod native {
    use super::*;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(handle: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(handle: *mut c_void) -> i32;
    }
    pub fn open(name: &CStr) -> Result<NonNull<c_void>, LoadError> {
        let wide: Vec<u16> = name
            .to_string_lossy()
            .encode_utf16()
            .chain(Some(0))
            .collect();
        // SAFETY: wide is terminated and retained for the duration of the call.
        NonNull::new(unsafe { LoadLibraryW(wide.as_ptr()) })
            .ok_or_else(|| LoadError(std::io::Error::last_os_error().to_string()))
    }
    pub fn symbol(handle: NonNull<c_void>, name: &CStr) -> Result<*mut c_void, LoadError> {
        // SAFETY: handle remains alive and name is a terminated C string.
        let pointer = unsafe { GetProcAddress(handle.as_ptr(), name.as_ptr().cast()) };
        if pointer.is_null() {
            Err(LoadError(std::io::Error::last_os_error().to_string()))
        } else {
            Ok(pointer)
        }
    }
    pub fn close(handle: NonNull<c_void>) {
        // SAFETY: one close of the owned module handle.
        unsafe {
            FreeLibrary(handle.as_ptr());
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod native {
    use super::*;
    pub fn open(_: &CStr) -> Result<NonNull<c_void>, LoadError> {
        Err(LoadError(
            "dynamic loading unavailable on this target".into(),
        ))
    }
    pub fn symbol(_: NonNull<c_void>, _: &CStr) -> Result<*mut c_void, LoadError> {
        Err(LoadError(
            "dynamic loading unavailable on this target".into(),
        ))
    }
    pub fn close(_: NonNull<c_void>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_library_is_an_error() {
        assert!(
            DynamicLibrary::open(&["llama-rewrite-intentionally-missing-20261108.so"]).is_err()
        );
        assert!(DynamicLibrary::open(&[]).is_err());
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn loads_a_real_system_symbol_and_rejects_missing() {
        let library = DynamicLibrary::open(&["libc.so.6"]).unwrap();
        // SAFETY: strlen has precisely this C ABI, and the library stays alive.
        let strlen: unsafe extern "C" fn(*const std::ffi::c_char) -> usize =
            unsafe { library.symbol(c"strlen") }.unwrap();
        // SAFETY: static, terminated bytes with a lifetime beyond the call.
        assert_eq!(unsafe { strlen(c"rust".as_ptr()) }, 4);
        // SAFETY: this symbol is intentionally absent, so no pointer is used.
        assert!(unsafe { library.symbol::<*mut c_void>(c"missing_rewrite_symbol") }.is_err());
        // SAFETY: rejected before lookup because u8 is not pointer sized.
        assert!(unsafe { library.symbol::<u8>(c"strlen") }.is_err());
    }

    #[cfg(windows)]
    #[test]
    fn loads_real_windows_symbol_and_rejects_missing() {
        let library = DynamicLibrary::open(&["kernel32.dll"]).unwrap();
        // SAFETY: the documented Win32 function takes no arguments and returns
        // DWORD. The library remains owned until every symbol use has ended.
        let process_id: unsafe extern "system" fn() -> u32 =
            unsafe { library.symbol(c"GetCurrentProcessId") }.unwrap();
        // SAFETY: GetCurrentProcessId has no preconditions or output pointers.
        assert_eq!(unsafe { process_id() }, std::process::id());
        // SAFETY: this deliberately absent symbol is never invoked.
        assert!(unsafe { library.symbol::<*mut c_void>(c"missing_rewrite_symbol") }.is_err());
        // SAFETY: rejected before lookup because the value has wrong ABI width.
        assert!(unsafe { library.symbol::<u8>(c"GetCurrentProcessId") }.is_err());
    }
}
