//! Opaque C FILE* borrowing through documented platform CRT interfaces.
//! No FILE layout is declared or inspected. The caller keeps its CRT stream;
//! tensor reads use an owned OS-handle duplicate of the same open file.
use crate::Result;
use std::ffi::c_void;
use std::fs::File;

pub(crate) struct Source {
    stream: *mut c_void,
    origin: i64,
    file: Option<File>,
    active: bool,
}
impl Source {
    /// The caller supplies a live, seekable binary FILE* from the platform CRT,
    /// exclusively for this call. Concurrent cursor/close operations are invalid.
    pub(crate) unsafe fn new(stream: *mut c_void) -> Result<Self> {
        if stream.is_null() {
            return Err("null FILE pointer".into());
        }
        let origin = unsafe { platform::position(stream)? };
        // Establish a coherent CRT cursor before touching the shared OS handle.
        unsafe { platform::restore(stream, origin)? };
        let file = unsafe { platform::duplicate(stream)? };
        Ok(Self {
            stream,
            origin,
            file: Some(file),
            active: true,
        })
    }
    pub(crate) fn origin(&self) -> u64 {
        self.origin as u64
    }
    pub(crate) fn take_file(&mut self) -> Result<File> {
        self.file
            .take()
            .ok_or_else(|| "FILE handle already consumed".into())
    }
    pub(crate) fn finish(mut self) -> Result<()> {
        let result = unsafe { platform::restore(self.stream, self.origin) };
        self.active = false;
        result
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        if self.active {
            // Also restore on ordinary parsing errors and unwinding. A caller
            // that closes/reenters its FILE concurrently violates the contract.
            let _ = unsafe { platform::restore(self.stream, self.origin) };
        }
    }
}

#[cfg(all(target_os = "windows", target_env = "msvc"))]
mod platform {
    use super::*;
    use std::os::windows::io::{BorrowedHandle, RawHandle};

    unsafe extern "C" {
        fn _fileno(stream: *mut c_void) -> i32;
        fn _get_osfhandle(fd: i32) -> isize;
        fn _ftelli64(stream: *mut c_void) -> i64;
        fn _fseeki64(stream: *mut c_void, offset: i64, origin: i32) -> i32;
    }
    pub(super) unsafe fn position(stream: *mut c_void) -> Result<i64> {
        let offset = unsafe { _ftelli64(stream) };
        if offset < 0 {
            Err("FILE is not a seekable binary input stream".into())
        } else {
            Ok(offset)
        }
    }
    pub(super) unsafe fn restore(stream: *mut c_void, offset: i64) -> Result<()> {
        // Seek through the end to discard any previous CRT read-ahead, then
        // restore the public logical position. The original Windows library
        // mixes fread and raw HANDLE reads and leaves a stale CRT buffer;
        // this implementation deliberately keeps the caller stream coherent.
        if unsafe { _fseeki64(stream, 0, 2) } != 0 || unsafe { _fseeki64(stream, offset, 0) } != 0 {
            Err("unable to restore FILE position".into())
        } else {
            Ok(())
        }
    }
    pub(super) unsafe fn duplicate(stream: *mut c_void) -> Result<File> {
        let fd = unsafe { _fileno(stream) };
        if fd < 0 {
            return Err("FILE has no readable OS descriptor".into());
        }
        let handle = unsafe { _get_osfhandle(fd) };
        if handle == 0 || handle == -1 || handle == -2 {
            return Err("FILE has no disk handle".into());
        }
        // SAFETY: the live CRT stream retains this handle while the standard
        // library duplicates it; only the returned duplicate becomes owned.
        let borrowed = unsafe { BorrowedHandle::borrow_raw(handle as RawHandle) };
        borrowed
            .try_clone_to_owned()
            .map(File::from)
            .map_err(|e| e.to_string())
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu", target_pointer_width = "64"))]
mod platform {
    use super::*;
    use std::os::fd::BorrowedFd;

    unsafe extern "C" {
        fn fileno(stream: *mut c_void) -> i32;
        fn ftello(stream: *mut c_void) -> i64;
        fn fseeko(stream: *mut c_void, offset: i64, origin: i32) -> i32;
    }
    pub(super) unsafe fn position(stream: *mut c_void) -> Result<i64> {
        let offset = unsafe { ftello(stream) };
        if offset < 0 {
            Err("FILE is not a seekable binary input stream".into())
        } else {
            Ok(offset)
        }
    }
    pub(super) unsafe fn restore(stream: *mut c_void, offset: i64) -> Result<()> {
        if unsafe { fseeko(stream, 0, 2) } != 0 || unsafe { fseeko(stream, offset, 0) } != 0 {
            Err("unable to restore FILE position".into())
        } else {
            Ok(())
        }
    }
    pub(super) unsafe fn duplicate(stream: *mut c_void) -> Result<File> {
        let fd = unsafe { fileno(stream) };
        if fd < 0 {
            return Err("FILE has no readable OS descriptor".into());
        }
        // SAFETY: the caller's live stream retains fd during duplication.
        let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
        borrowed
            .try_clone_to_owned()
            .map(File::from)
            .map_err(|e| e.to_string())
    }
}

#[cfg(not(any(
    all(target_os = "windows", target_env = "msvc"),
    all(target_os = "linux", target_env = "gnu", target_pointer_width = "64")
)))]
mod platform {
    use super::*;
    pub(super) unsafe fn position(_: *mut c_void) -> Result<i64> {
        Err("FILE loading requires the measured MSVC or 64-bit GNU CRT profile".into())
    }
    pub(super) unsafe fn restore(_: *mut c_void, _: i64) -> Result<()> {
        Err("FILE loading requires the measured MSVC or 64-bit GNU CRT profile".into())
    }
    pub(super) unsafe fn duplicate(_: *mut c_void) -> Result<File> {
        Err("FILE loading requires the measured MSVC or 64-bit GNU CRT profile".into())
    }
}

#[cfg(all(
    test,
    any(
        all(target_os = "windows", target_env = "msvc"),
        all(target_os = "linux", target_env = "gnu", target_pointer_width = "64")
    )
))]
mod tests;
