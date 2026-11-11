//! Safe identity comparison across paths, with OS calls isolated in this crate.
//! Windows uses the documented 128-bit FileIdInfo plus volume number, avoiding
//! the 64-bit legacy identity ambiguity on ReFS. No file contents are read or
//! written, no locks are acquired, and normal read/write/delete sharing stays on.
use std::{io, path::Path};

/// Compare existing paths by OS file identity. Symbolic links are followed.
/// This is a point-in-time comparison, not protection against a hostile path
/// replacement after the call. Missing paths and unsupported queries are errors.
pub fn same_file(left: impl AsRef<Path>, right: impl AsRef<Path>) -> io::Result<bool> {
    native::same_file(left.as_ref(), right.as_ref())
}
#[cfg(windows)]
mod native {
    use super::*;
    use std::{
        ffi::c_void,
        fs::File,
        os::windows::{fs::OpenOptionsExt, io::AsRawHandle},
    };
    const FILE_ID_INFO_CLASS: i32 = 18;
    #[repr(C)]
    #[derive(Default, PartialEq, Eq)]
    struct FileIdInfo {
        volume_serial_number: u64,
        file_id: [u8; 16],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandleEx(
            handle: *mut c_void,
            class: i32,
            info: *mut c_void,
            size: u32,
        ) -> i32;
    }
    fn open(path: &Path) -> io::Result<File> {
        // Zero desired access queries metadata without requiring payload-read
        // permission. BACKUP_SEMANTICS also permits directory identity queries.
        File::options()
            .read(true)
            .access_mode(0)
            .share_mode(1 | 2 | 4)
            .custom_flags(0x0200_0000)
            .open(path)
    }
    fn identity(file: &File) -> io::Result<FileIdInfo> {
        let mut info = FileIdInfo::default();
        // SAFETY: file owns a live Windows handle throughout the synchronous
        // call. The repr(C) buffer has the installed SDK's FILE_ID_INFO layout:
        // u64 volume followed by 16 identifier bytes, 24 bytes/alignment 8 on
        // the supported x64 host. We supply its exact size, and the system call
        // neither retains the pointer nor closes the handle. Failed queries
        // return the last OS error without observing partial output.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle().cast(),
                FILE_ID_INFO_CLASS,
                (&mut info as *mut FileIdInfo).cast(),
                std::mem::size_of::<FileIdInfo>() as u32,
            )
        };
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(info)
        }
    }
    pub fn same_file(left: &Path, right: &Path) -> io::Result<bool> {
        let left = open(left)?;
        let right = open(right)?;
        Ok(identity(&left)? == identity(&right)?)
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn installed_sdk_file_identity_abi() {
            let native: std::collections::BTreeMap<_, _> =
                include_str!("../tests/fixtures/file-identity-sdk.tsv")
                    .lines()
                    .map(|line| {
                        let (key, value) = line.split_once('\t').unwrap();
                        (key, value.parse::<usize>().unwrap())
                    })
                    .collect();
            assert_eq!(
                std::mem::size_of::<FileIdInfo>(),
                native["sizeof_file_id_info"]
            );
            assert_eq!(
                std::mem::align_of::<FileIdInfo>(),
                native["alignof_file_id_info"]
            );
            assert_eq!(
                std::mem::offset_of!(FileIdInfo, volume_serial_number),
                native["offset_volume"]
            );
            assert_eq!(
                std::mem::offset_of!(FileIdInfo, file_id),
                native["offset_id"]
            );
            assert_eq!(std::mem::size_of::<[u8; 16]>(), native["sizeof_id"]);
            assert_eq!(std::mem::size_of::<i32>(), native["sizeof_bool"]);
            assert_eq!(std::mem::size_of::<*mut c_void>(), native["sizeof_handle"]);
            assert_eq!(FILE_ID_INFO_CLASS as usize, native["file_id_info_class"]);
        }
    }
}
#[cfg(unix)]
mod native {
    use super::*;
    use std::os::unix::fs::MetadataExt;
    pub fn same_file(left: &Path, right: &Path) -> io::Result<bool> {
        let left = std::fs::metadata(left)?;
        let right = std::fs::metadata(right)?;
        Ok(left.dev() == right.dev() && left.ino() == right.ino())
    }
}
#[cfg(not(any(unix, windows)))]
mod native {
    use super::*;
    pub fn same_file(_: &Path, _: &Path) -> io::Result<bool> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "file identity is unsupported on this target",
        ))
    }
}
#[cfg(all(test, any(unix, windows)))]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn identifies_hard_links_without_confusing_equal_contents() {
        let directory = Directory(std::env::temp_dir().join(format!(
            "llama-file-identity-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        fs::create_dir(&directory.0).unwrap();
        let input = directory.0.join("input");
        let alias = directory.0.join("alias");
        let copy = directory.0.join("copy");
        fs::write(&input, b"same bytes").unwrap();
        fs::hard_link(&input, &alias).unwrap();
        fs::copy(&input, &copy).unwrap();
        assert!(same_file(&input, &input).unwrap());
        assert!(same_file(&input, &alias).unwrap());
        assert!(!same_file(&input, &copy).unwrap());
        assert!(same_file(&input, directory.0.join("missing")).is_err());
        assert_eq!(fs::read(&input).unwrap(), b"same bytes");
        #[cfg(windows)]
        {
            // The identity query must coexist with ordinary shared read handles.
            let _reader = fs::File::open(&input).unwrap();
            assert!(same_file(&input, &alias).unwrap());
        }
    }
}
