//! Global native logger state. Foreign callbacks execute without the state lock.
use crate::*;
use std::ffi::c_void;
use std::sync::Mutex;

#[cfg(test)]
pub(crate) static TEST_LOGGER_LOCK: Mutex<()> = Mutex::new(());

type Callback = unsafe extern "C" fn(i32, *const c_char, *mut c_void);

#[derive(Clone, Copy)]
struct Logger {
    callback: Callback,
    // This is opaque caller data. Integer transport avoids a Send promise for its pointee.
    user_data: usize,
}
static LOGGER: Mutex<Logger> = Mutex::new(Logger {
    callback: default_callback,
    user_data: 0,
});

fn current() -> Result<Logger> {
    Ok(*LOGGER.lock().map_err(|_| "logger state poisoned")?)
}

pub(crate) fn emit(level: i32, message: &CStr) -> Result<()> {
    let logger = current()?;
    // SAFETY: the caller registered this callback and keeps its opaque data live.
    // The message is retained for the callback's duration; no state lock is held.
    unsafe { (logger.callback)(level, message.as_ptr(), logger.user_data as *mut c_void) };
    Ok(())
}

// Preserve the system C runtime's stderr stream, text mode and redirection on the
// independently verified targets. These declarations link the system runtime,
// never a llama or ggml library; all logger and performance logic is Rust.
#[cfg(any(target_os = "linux", all(target_os = "windows", target_env = "msvc")))]
unsafe extern "C" {
    fn fputs(text: *const c_char, stream: *mut c_void) -> i32;
    fn fflush(stream: *mut c_void) -> i32;
}
#[cfg(target_os = "linux")]
unsafe extern "C" {
    static mut stderr: *mut c_void;
}
#[cfg(all(target_os = "windows", target_env = "msvc"))]
unsafe extern "C" {
    fn __acrt_iob_func(index: u32) -> *mut c_void;
}

unsafe extern "C" fn default_callback(_level: i32, text: *const c_char, _data: *mut c_void) {
    #[cfg(target_os = "linux")]
    unsafe {
        let stream = stderr;
        fputs(text, stream);
        fflush(stream);
    }
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    unsafe {
        let stream = __acrt_iob_func(2);
        fputs(text, stream);
        fflush(stream);
    }
    #[cfg(not(any(target_os = "linux", all(target_os = "windows", target_env = "msvc"))))]
    {
        use std::io::Write;
        let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
        let mut stderr = std::io::stderr().lock();
        let _ = stderr.write_all(bytes);
        let _ = stderr.flush();
    }
}

ffi!(
    llama_log_set(callback: LogCallback, user_data: *mut c_void) -> (),
    (),
    {
        *LOGGER.lock().map_err(|_| "logger state poisoned")? = Logger {
            callback: callback.unwrap_or(default_callback),
            user_data: user_data as usize,
        };
        Ok(())
    }
);
ffi!(
    llama_log_get(callback: *mut LogCallback, user_data: *mut *mut c_void) -> (),
    (),
    {
        // Validate both outputs before writing either; foreign allocations must be disjoint.
        let callback = unsafe { reference_mut(callback)? };
        let user_data = unsafe { reference_mut(user_data)? };
        let logger = current()?;
        *callback = Some(logger.callback);
        *user_data = logger.user_data as *mut c_void;
        Ok(())
    }
);
