//! Private file-capability boundary. No tensor logic lives in this module.
//! Linux: component-wise openat(O_PATH|O_NOFOLLOW), retained parents, then
//! /proc/self/fd reopens only a verified regular-file handle for reading.
//! Windows: NtCreateFile relative to a retained directory with OBJ_DONT_REPARSE.
//! All imported handles are adopted exactly once and closed by File on every path.
use crate::{Budget, Error, Result, Text, reserve};
use std::fs::File;
use std::io;
use std::path::Path;

pub(crate) fn open_trusted(path: &Path) -> Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Read sharing only: deny concurrent write/delete while an owner is alive.
        Ok(std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(path)?)
    }
    #[cfg(not(windows))]
    {
        Ok(File::open(path)?)
    }
}
pub(crate) fn read_at(file: &File, output: &mut [u8], offset: u64) -> io::Result<usize> {
    loop {
        #[cfg(unix)]
        let result = {
            use std::os::unix::fs::FileExt;
            file.read_at(output, offset)
        };
        #[cfg(windows)]
        let result = {
            use std::os::windows::fs::FileExt;
            file.seek_read(output, offset)
        };
        #[cfg(not(any(unix, windows)))]
        let result: io::Result<usize> = Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positioned file reads",
        ));
        match result {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            other => return other,
        }
    }
}
pub(crate) fn read_exact_at(file: &File, mut output: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !output.is_empty() {
        let count = read_at(file, output, offset)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "retained file was truncated",
            ));
        }
        offset = offset
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file offset overflow"))?;
        output = &mut output[count..];
    }
    Ok(())
}
pub(crate) struct Root {
    owner: File,
}
impl Root {
    pub fn open(path: &Path) -> Result<Self> {
        #[cfg(windows)]
        let owner = {
            use std::os::windows::fs::OpenOptionsExt;
            // The explicitly supplied root is trusted. Sharing all root access
            // avoids locking unrelated parent directories; relative NT opens
            // retain its identity even if the pathname is moved.
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(1 | 2 | 4)
                .custom_flags(0x0200_0000)
                .open(path)?
        };
        #[cfg(not(windows))]
        let owner = File::open(path)?;
        if !owner.metadata()?.is_dir() {
            return Err(Error::Path("checkpoint root is not a directory"));
        }
        Ok(Self { owner })
    }
    pub fn entries(&self, budget: &mut Budget) -> Result<Vec<Text>> {
        host::entries(&self.owner, budget)
    }
    pub fn open_relative(&self, name: &Text, budget: &mut Budget) -> Result<File> {
        let components = components(name, budget)?;
        host::relative(&self.owner, &components, budget)
    }
    pub fn optional_file(&self, name: &Text, budget: &mut Budget) -> Result<Option<File>> {
        match self.open_relative(name, budget) {
            Ok(file) => Ok(Some(file)),
            Err(Error::Io(e))
                if matches!(
                    e.kind(),
                    io::ErrorKind::NotFound
                        | io::ErrorKind::NotADirectory
                        | io::ErrorKind::IsADirectory
                ) =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}
fn components(name: &Text, budget: &mut Budget) -> Result<Vec<Text>> {
    if name.as_wtf8().len() > budget.limits.max_path_bytes {
        return Err(Error::Limit("path bytes"));
    }
    let bytes = name.as_wtf8();
    if bytes.is_empty() || bytes[0] == b'/' || (cfg!(windows) && bytes[0] == b'\\') {
        return Err(Error::Path("empty or absolute indexed path"));
    }
    if bytes.contains(&0) || (cfg!(windows) && bytes.contains(&b':')) {
        return Err(Error::Path("NUL, drive or alternate-stream indexed path"));
    }
    let mut parts = Vec::new();
    for part in bytes.split(|b| *b == b'/' || (cfg!(windows) && *b == b'\\')) {
        if part.is_empty() || part == b"." {
            continue;
        }
        if part == b".." {
            return Err(Error::Path("parent traversal in indexed path"));
        }
        if parts.len() >= budget.limits.max_path_components {
            return Err(Error::Limit("path components"));
        }
        reserve(&mut parts, 1, "path components allocation")?;
        parts.push(Text::from_bytes(part, budget)?);
    }
    if parts.is_empty() {
        return Err(Error::Path("indexed path has no filename"));
    }
    Ok(parts)
}

#[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
type OpenHook = Option<Box<dyn FnMut(usize)>>;
#[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
thread_local! {
    static OPEN_HOOK: std::cell::RefCell<OpenHook> =
        std::cell::RefCell::new(None);
}
#[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
pub(crate) fn set_open_hook(callback: OpenHook) {
    OPEN_HOOK.with(|hook| *hook.borrow_mut() = callback);
}
#[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
fn hook(stage: usize) {
    OPEN_HOOK.with(|hook| {
        if let Some(callback) = hook.borrow_mut().as_mut() {
            callback(stage);
        }
    });
}
#[cfg(all(
    not(test),
    any(windows, all(target_os = "linux", target_arch = "x86_64"))
))]
fn hook(_: usize) {}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod host {
    use super::*;
    use std::ffi::OsStr;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::raw::{c_char, c_int};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    // Linux UAPI values, independently checked against the pinned GNU host's
    // bits/fcntl-linux.h. No O_CREAT: openat has no mode argument here.
    const O_DIRECTORY: c_int = 0o200000;
    const O_NOFOLLOW: c_int = 0o400000;
    const O_CLOEXEC: c_int = 0o2000000;
    const O_PATH: c_int = 0o10000000;
    unsafe extern "C" {
        fn openat(dirfd: c_int, pathname: *const c_char, flags: c_int, ...) -> c_int;
    }
    fn fd_path<T>(file: &File, operation: impl FnOnce(&Path) -> T) -> T {
        let mut bytes = [0u8; 64];
        let prefix = b"/proc/self/fd/";
        bytes[..prefix.len()].copy_from_slice(prefix);
        let mut digits = [0u8; 20];
        let mut at = digits.len();
        let mut number = file.as_raw_fd() as u32;
        loop {
            at -= 1;
            digits[at] = b'0' + (number % 10) as u8;
            number /= 10;
            if number == 0 {
                break;
            }
        }
        let length = prefix.len() + digits.len() - at;
        bytes[prefix.len()..length].copy_from_slice(&digits[at..]);
        operation(Path::new(OsStr::from_bytes(&bytes[..length])))
    }
    fn name_bytes(text: &Text, budget: &mut Budget) -> Result<Vec<u8>> {
        let mut length = 0usize;
        for code in text.codepoints() {
            let width = if (0xdc80..=0xdcff).contains(&code) {
                1
            } else if (0xd800..=0xdfff).contains(&code) {
                return Err(Error::Path(
                    "indexed path cannot be encoded by source filesystem rules",
                ));
            } else if code < 0x80 {
                1
            } else if code < 0x800 {
                2
            } else if code < 0x10000 {
                3
            } else {
                4
            };
            length = length
                .checked_add(width)
                .ok_or(Error::Limit("native path length"))?;
        }
        budget.text(length)?;
        let capacity = length
            .checked_add(1)
            .ok_or(Error::Limit("native path terminator"))?;
        let mut bytes = Vec::new();
        reserve(&mut bytes, capacity, "native path allocation")?;
        for code in text.codepoints() {
            if (0xdc80..=0xdcff).contains(&code) {
                bytes.push((code - 0xdc00) as u8);
            } else {
                Text::push_codepoint(&mut bytes, code)?;
            }
        }
        bytes.push(0);
        Ok(bytes)
    }
    fn from_os(bytes: &[u8], budget: &mut Budget) -> Result<Text> {
        let mut result = Vec::new();
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let (valid, invalid) = match std::str::from_utf8(remaining) {
                Ok(_) => (remaining.len(), 0),
                Err(error) => (
                    error.valid_up_to(),
                    error
                        .error_len()
                        .unwrap_or(remaining.len() - error.valid_up_to()),
                ),
            };
            let add = valid
                .checked_add(
                    invalid
                        .checked_mul(3)
                        .ok_or(Error::Limit("filesystem text size"))?,
                )
                .ok_or(Error::Limit("filesystem text size"))?;
            if result
                .len()
                .checked_add(add)
                .is_none_or(|n| n > budget.limits.max_string_bytes)
            {
                return Err(Error::Limit("filesystem string bytes"));
            }
            budget.text(add)?;
            reserve(&mut result, add, "filesystem text allocation")?;
            result.extend_from_slice(&remaining[..valid]);
            for &byte in &remaining[valid..valid + invalid] {
                Text::push_codepoint(&mut result, 0xdc00 + u32::from(byte))?;
            }
            remaining = &remaining[valid + invalid..];
        }
        Ok(Text(result))
    }
    pub(super) fn entries(root: &File, budget: &mut Budget) -> Result<Vec<Text>> {
        let mut result = Vec::new();
        // A procfs FD link resolves the retained directory, not its former name.
        for entry in fd_path(root, |path| std::fs::read_dir(path))? {
            budget.entry()?;
            let name = entry?.file_name();
            let name = from_os(name.as_bytes(), budget)?;
            reserve(&mut result, 1, "directory entries allocation")?;
            result.push(name);
        }
        Ok(result)
    }
    fn open_component(parent: &File, name: &[u8], directory: bool) -> Result<File> {
        let flags = O_PATH | O_NOFOLLOW | O_CLOEXEC | if directory { O_DIRECTORY } else { 0 };
        // SAFETY: parent remains owned, name is a live NUL-terminated vector
        // without interior NUL, and flags never create a file. On success this
        // is the sole owner of the returned descriptor.
        let raw = unsafe { openat(parent.as_raw_fd(), name.as_ptr().cast(), flags) };
        if raw < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: a nonnegative openat result is a new descriptor owned here.
        Ok(unsafe { File::from_raw_fd(raw) })
    }
    pub(super) fn relative(root: &File, parts: &[Text], budget: &mut Budget) -> Result<File> {
        let mut parent = None;
        for (index, part) in parts.iter().enumerate() {
            hook(index);
            let name = name_bytes(part, budget)?;
            let final_part = index + 1 == parts.len();
            let opened = open_component(parent.as_ref().unwrap_or(root), &name, !final_part)?;
            if final_part {
                let metadata = opened.metadata()?;
                if metadata.is_dir() {
                    return Err(io::Error::from(io::ErrorKind::IsADirectory).into());
                }
                if !metadata.is_file() {
                    return Err(Error::Path(
                        "indexed owner is a symlink or non-regular file",
                    ));
                }
                hook(parts.len());
                // O_PATH lets us reject FIFOs/devices before opening them for
                // data I/O. This is a retained-FD procfs capability, not an
                // untrusted path or a canonicalize-then-reopen sequence.
                let readable = fd_path(&opened, |path| File::open(path))?;
                let actual = readable.metadata()?;
                if !actual.is_file()
                    || metadata.dev() != actual.dev()
                    || metadata.ino() != actual.ino()
                {
                    return Err(Error::Path("retained capability identity changed"));
                }
                return Ok(readable);
            }
            parent = Some(opened);
        }
        Err(Error::Path("indexed path has no filename"))
    }
    #[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
    pub(super) fn handle_count() -> io::Result<u64> {
        Ok(std::fs::read_dir("/proc/self/fd")?.count() as u64)
    }
    #[cfg(test)]
    pub(super) fn make_fifo(path: &Path) -> io::Result<()> {
        unsafe extern "C" {
            fn mkfifo(path: *const c_char, mode: u32) -> c_int;
        }
        let name = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: test-only creation, valid terminated path and mode_t on the
        // gated x86_64 Linux ABI. No handle is returned or left open.
        let result = unsafe { mkfifo(name.as_ptr(), 0o600) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[cfg(windows)]
mod host {
    use super::*;
    use std::ffi::c_void;
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    type Handle = *mut c_void;
    #[repr(C)]
    struct UnicodeString {
        length: u16,
        maximum_length: u16,
        buffer: *mut u16,
    }
    #[repr(C)]
    struct ObjectAttributes {
        length: u32,
        root: Handle,
        name: *mut UnicodeString,
        attributes: u32,
        security_descriptor: *mut c_void,
        security_quality: *mut c_void,
    }
    #[repr(C)]
    struct IoStatusBlock {
        status_or_pointer: usize,
        information: usize,
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtCreateFile(
            file: *mut Handle,
            access: u32,
            attributes: *mut ObjectAttributes,
            status: *mut IoStatusBlock,
            allocation_size: *mut i64,
            file_attributes: u32,
            share_access: u32,
            disposition: u32,
            options: u32,
            ea_buffer: *mut c_void,
            ea_length: u32,
        ) -> i32;
        fn NtQueryDirectoryFile(
            file: Handle,
            event: Handle,
            apc: *mut c_void,
            context: *mut c_void,
            status: *mut IoStatusBlock,
            information: *mut c_void,
            length: u32,
            class: u32,
            single: u8,
            name: *mut UnicodeString,
            restart: u8,
        ) -> i32;
        fn RtlNtStatusToDosError(status: i32) -> u32;
    }
    const STATUS_NO_MORE_FILES: i32 = 0x80000006u32 as i32;
    const STATUS_PENDING: i32 = 0x00000103;
    fn native_error(status: i32) -> Error {
        if status == 0xc00000bau32 as i32 {
            return io::Error::from(io::ErrorKind::IsADirectory).into();
        }
        // SAFETY: pure native-status translation, no pointers or ownership.
        let code = unsafe { RtlNtStatusToDosError(status) };
        Error::Io(io::Error::from_raw_os_error(code as i32))
    }
    fn wide_path(parts: &[Text], budget: &mut Budget) -> Result<Vec<u16>> {
        let units = parts
            .iter()
            .try_fold(parts.len().saturating_sub(1), |total, part| {
                part.codepoints().try_fold(total, |total, code| {
                    total.checked_add(if code > 0xffff { 2 } else { 1 })
                })
            })
            .ok_or(Error::Limit("UTF-16 path length"))?;
        let bytes = units
            .checked_mul(2)
            .ok_or(Error::Limit("UTF-16 path bytes"))?;
        if bytes > budget.limits.max_path_bytes || bytes > 65534 {
            return Err(Error::Limit("native UTF-16 path bytes"));
        }
        budget.text(bytes)?;
        let mut wide = Vec::new();
        reserve(&mut wide, units, "UTF-16 path allocation")?;
        for (index, part) in parts.iter().enumerate() {
            if index > 0 {
                wide.push(b'\\' as u16);
            }
            for code in part.codepoints() {
                if code > 0xffff {
                    let reduced = code - 0x10000;
                    wide.push((0xd800 + (reduced >> 10)) as u16);
                    wide.push((0xdc00 + (reduced & 1023)) as u16);
                } else {
                    wide.push(code as u16);
                }
            }
        }
        Ok(wide)
    }
    pub(super) fn relative(root: &File, parts: &[Text], budget: &mut Budget) -> Result<File> {
        let mut wide = wide_path(parts, budget)?;
        let length =
            u16::try_from(wide.len() * 2).map_err(|_| Error::Limit("native path length"))?;
        let mut name = UnicodeString {
            length,
            maximum_length: length,
            buffer: wide.as_mut_ptr(),
        };
        let mut attributes = ObjectAttributes {
            length: std::mem::size_of::<ObjectAttributes>() as u32,
            root: root.as_raw_handle(),
            name: &mut name,
            // OBJ_CASE_INSENSITIVE | OBJ_DONT_REPARSE.
            attributes: 0x40 | 0x1000,
            security_descriptor: std::ptr::null_mut(),
            security_quality: std::ptr::null_mut(),
        };
        let mut status_block = IoStatusBlock {
            status_or_pointer: 0,
            information: 0,
        };
        let mut handle: Handle = std::ptr::null_mut();
        hook(0);
        // SAFETY: all ABI records are repr(C), pointers remain live for this
        // synchronous call, UnicodeString lengths fit the vector, and the root
        // is a retained directory. No path component can reparse or escape.
        // FILE_GENERIC_READ includes SYNCHRONIZE; FILE_SHARE_READ denies
        // write/delete. FILE_OPEN never creates/truncates. Options require a
        // non-directory synchronous owner (FILE_NON_DIRECTORY_FILE|...NONALERT).
        let status = unsafe {
            NtCreateFile(
                &mut handle,
                0x0012_0089,
                &mut attributes,
                &mut status_block,
                std::ptr::null_mut(),
                0,
                1,
                1,
                0x40 | 0x20,
                std::ptr::null_mut(),
                0,
            )
        };
        if status < 0 {
            return Err(native_error(status));
        }
        if handle.is_null() || handle == (-1isize as Handle) {
            return Err(Error::Invalid(
                "native open returned an invalid file handle",
            ));
        }
        // SAFETY: a successful NtCreateFile transferred one handle; File now
        // closes it even if a following invariant or metadata check fails.
        let owner = unsafe { File::from_raw_handle(handle) };
        if status == STATUS_PENDING {
            return Err(Error::Invalid(
                "synchronous native open unexpectedly pending",
            ));
        }
        if !owner.metadata()?.is_file() {
            return Err(Error::Path("indexed owner is not a regular file"));
        }
        hook(1);
        Ok(owner)
    }
    pub(super) fn entries(root: &File, budget: &mut Budget) -> Result<Vec<Text>> {
        let mut storage = Vec::<u32>::new();
        reserve(&mut storage, 16384, "native directory buffer")?;
        storage.resize(16384, 0);
        let mut result = Vec::new();
        let mut restart = 1;
        loop {
            let mut status_block = IoStatusBlock {
                status_or_pointer: 0,
                information: 0,
            };
            // SAFETY: root is synchronous and retained; the initialized u32
            // buffer is writable, 4-byte aligned and exactly 65536 bytes.
            // No asynchronous completion routine/event is requested.
            let status = unsafe {
                NtQueryDirectoryFile(
                    root.as_raw_handle(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut status_block,
                    storage.as_mut_ptr().cast(),
                    65536,
                    12,
                    0,
                    std::ptr::null_mut(),
                    restart,
                )
            };
            restart = 0;
            if status == STATUS_NO_MORE_FILES {
                break;
            }
            if status < 0 {
                return Err(native_error(status));
            }
            if status == STATUS_PENDING
                || status_block.information == 0
                || status_block.information > 65536
            {
                return Err(Error::Invalid(
                    "native directory result has invalid synchronous length",
                ));
            }
            // SAFETY: every u32 has initialized bytes and the validated returned
            // byte length is within the allocation. Byte alignment is weaker.
            let bytes = unsafe {
                std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), status_block.information)
            };
            let mut at = 0usize;
            loop {
                let header = bytes
                    .get(at..at.saturating_add(12))
                    .ok_or(Error::Invalid("native directory entry truncated"))?;
                let next = u32::from_le_bytes(
                    header[0..4]
                        .try_into()
                        .map_err(|_| Error::Invalid("directory next"))?,
                ) as usize;
                let length = u32::from_le_bytes(
                    header[8..12]
                        .try_into()
                        .map_err(|_| Error::Invalid("directory length"))?,
                ) as usize;
                if !length.is_multiple_of(2) {
                    return Err(Error::Invalid("odd native filename length"));
                }
                let end = at
                    .checked_add(12)
                    .and_then(|n| n.checked_add(length))
                    .ok_or(Error::Limit("native directory entry overflow"))?;
                let encoded = bytes
                    .get(at + 12..end)
                    .ok_or(Error::Invalid("native filename truncated"))?;
                let codepoints = char::decode_utf16(
                    encoded
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| u16::from_le_bytes(*b)),
                )
                .map(|result| result.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from));
                let name = Text::from_codepoints(codepoints, budget)?;
                if name.as_wtf8() != b"." && name.as_wtf8() != b".." {
                    budget.entry()?;
                    reserve(&mut result, 1, "directory entries allocation")?;
                    result.push(name);
                }
                if next == 0 {
                    break;
                }
                if next < 12 + length || !next.is_multiple_of(4) {
                    return Err(Error::Invalid("native directory record displacement"));
                }
                at = at
                    .checked_add(next)
                    .ok_or(Error::Limit("directory cursor overflow"))?;
                if at >= bytes.len() {
                    return Err(Error::Invalid("native directory next entry outside result"));
                }
            }
        }
        Ok(result)
    }
    #[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
    pub(super) fn handle_count() -> io::Result<u64> {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> Handle;
            fn GetProcessHandleCount(process: Handle, count: *mut u32) -> i32;
        }
        let mut count = 0;
        // SAFETY: pseudo-handle requires no close, count is a live output u32.
        let ok = unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut count) };
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(u64::from(count))
        }
    }
}

#[cfg(not(any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
mod host {
    use super::*;
    pub(super) fn entries(_: &File, _: &mut Budget) -> Result<Vec<Text>> {
        Err(Error::Unsupported(
            "directory capabilities require the verified x86_64 Linux or Windows opener",
        ))
    }
    pub(super) fn relative(_: &File, _: &[Text], _: &mut Budget) -> Result<File> {
        Err(Error::Unsupported(
            "relative capabilities require the verified x86_64 Linux or Windows opener",
        ))
    }
}
#[cfg(all(test, any(windows, all(target_os = "linux", target_arch = "x86_64"))))]
pub(crate) fn handle_count() -> io::Result<u64> {
    host::handle_count()
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn make_fifo(path: &Path) -> io::Result<()> {
    host::make_fifo(path)
}
