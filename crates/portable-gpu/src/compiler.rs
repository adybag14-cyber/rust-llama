//! Optional external shader compiler, not a rewritten backend or GPU result.
//!
//! shaderc is a separately installed compiler boundary. It is not bundled,
//! linked at build time or required for the default Rust-only semantic tests.
use crate::*;
use llama_platform_loader::DynamicLibrary;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr::NonNull;

type Init = unsafe extern "C" fn() -> *mut c_void;
type Release = unsafe extern "C" fn(*mut c_void);
type Compile = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    usize,
    c_int,
    *const c_char,
    *const c_char,
    *const c_void,
) -> *mut c_void;
type StatusFn = unsafe extern "C" fn(*const c_void) -> c_int;
type Length = unsafe extern "C" fn(*const c_void) -> usize;
type Bytes = unsafe extern "C" fn(*const c_void) -> *const c_char;

pub struct ShaderCompiler {
    library: DynamicLibrary,
    handle: NonNull<c_void>,
    release: Release,
    compile: Compile,
    result_release: Release,
    status: StatusFn,
    length: Length,
    bytes: Bytes,
    error: Bytes,
}
#[derive(Debug, Clone)]
pub struct SpirvArtifact {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub compiler_library: String,
    pub source_sha256: String,
}
impl ShaderCompiler {
    pub fn open() -> Result<Self> {
        let library = DynamicLibrary::open(if cfg!(windows) {
            &["shaderc_shared.dll"]
        } else if cfg!(target_os = "macos") {
            &["libshaderc_shared.dylib", "libshaderc.dylib"]
        } else {
            &["libshaderc.so.1", "libshaderc_shared.so", "libshaderc.so"]
        })?;
        // SAFETY: each type below is the documented shaderc C ABI. The library
        // remains owned by Self until compiler/result objects have been released.
        unsafe {
            let initialize: Init = library.symbol(c"shaderc_compiler_initialize")?;
            let release = library.symbol(c"shaderc_compiler_release")?;
            let compile = library.symbol(c"shaderc_compile_into_spv")?;
            let result_release = library.symbol(c"shaderc_result_release")?;
            let status = library.symbol(c"shaderc_result_get_compilation_status")?;
            let length = library.symbol(c"shaderc_result_get_length")?;
            let bytes = library.symbol(c"shaderc_result_get_bytes")?;
            let error = library.symbol(c"shaderc_result_get_error_message")?;
            let handle = NonNull::new(initialize())
                .ok_or_else(|| Error("shaderc initialization failed".into()))?;
            Ok(Self {
                library,
                handle,
                release,
                compile,
                result_release,
                status,
                length,
                bytes,
                error,
            })
        }
    }
    pub fn library(&self) -> &str {
        self.library.name()
    }
    pub fn compile_glsl(&self, source: &llama_kernel_ir::Artifact) -> Result<SpirvArtifact> {
        if source.target != llama_kernel_ir::Target::Glsl {
            return Err(Error(
                "shaderc path accepts Vulkan GLSL artifacts only".into(),
            ));
        }
        let actual_source_hash = llama_kernel_ir::sha256_hex(source.source.as_bytes());
        if source.sha256 != actual_source_hash {
            return Err(Error(
                "shader source hash differs from artifact provenance".into(),
            ));
        }
        let entry = CString::new(source.entry_point.as_str())
            .map_err(|_| Error("NUL in entry point".into()))?;
        // SAFETY: compiler is live, source length exactly describes its bytes,
        // shader kind 2 is compute, filename and entry are terminated strings.
        let result = NonNull::new(unsafe {
            (self.compile)(
                self.handle.as_ptr(),
                source.source.as_ptr().cast(),
                source.source.len(),
                2,
                c"rust-ir.comp".as_ptr(),
                entry.as_ptr(),
                std::ptr::null(),
            )
        })
        .ok_or_else(|| Error("shaderc returned a null result".into()))?;
        struct Guard {
            pointer: NonNull<c_void>,
            release: Release,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                // SAFETY: exactly one release of the owned shaderc result.
                unsafe { (self.release)(self.pointer.as_ptr()) };
            }
        }
        let result = Guard {
            pointer: result,
            release: self.result_release,
        };
        // SAFETY: result is live for every accessor and slices are copied before release.
        unsafe {
            let code = (self.status)(result.pointer.as_ptr());
            if code != 0 {
                let error = (self.error)(result.pointer.as_ptr());
                let message = if error.is_null() {
                    "no compiler diagnostic".into()
                } else {
                    CStr::from_ptr(error).to_string_lossy().into_owned()
                };
                return Err(Error(format!("shaderc status {code}: {message}")));
            }
            let length = (self.length)(result.pointer.as_ptr());
            let pointer = (self.bytes)(result.pointer.as_ptr());
            if !(20..=64 * 1024 * 1024).contains(&length)
                || !length.is_multiple_of(4)
                || pointer.is_null()
            {
                return Err(Error("invalid SPIR-V result size/pointer".into()));
            }
            let bytes = std::slice::from_raw_parts(pointer.cast::<u8>(), length).to_vec();
            if bytes[..4] != 0x07230203u32.to_le_bytes() {
                return Err(Error("compiler output lacks SPIR-V magic".into()));
            }
            Ok(SpirvArtifact {
                sha256: llama_kernel_ir::sha256_hex(&bytes),
                bytes,
                compiler_library: self.library.name().into(),
                source_sha256: actual_source_hash,
            })
        }
    }
}
impl Drop for ShaderCompiler {
    fn drop(&mut self) {
        // SAFETY: compiler is owned, and all method-scoped results are already released.
        unsafe { (self.release)(self.handle.as_ptr()) };
    }
}
