//! Explicit comparison-only FFI driver for the separately built frozen native
//! library. The product's library code never calls or links original ggml.
use llama_platform_loader::DynamicLibrary;
use std::env;
use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io;
use std::net::SocketAddrV4;
use std::path::PathBuf;
use std::process::Command;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Ptr = *mut c_void;
type Free = unsafe extern "C" fn(Ptr);
#[repr(C)]
#[derive(Clone, Copy)]
struct InitParams {
    mem_size: usize,
    mem_buffer: Ptr,
    no_alloc: bool,
}

struct Api {
    // The library outlives every function pointer and all NativeResources.
    _library: DynamicLibrary,
    cpu_reg: unsafe extern "C" fn() -> Ptr,
    reg_dev: unsafe extern "C" fn(Ptr, usize) -> Ptr,
    reg_count: unsafe extern "C" fn(Ptr) -> usize,
    server: unsafe extern "C" fn(*const c_char, *const c_char, usize, usize, *mut Ptr),
    rpc_init: unsafe extern "C" fn(*const c_char, u32) -> Ptr,
    add_server: unsafe extern "C" fn(*const c_char) -> Ptr,
    device_memory: unsafe extern "C" fn(*const c_char, u32, *mut usize, *mut usize),
    init: unsafe extern "C" fn(InitParams) -> Ptr,
    tensor_1d: unsafe extern "C" fn(Ptr, c_int, i64) -> Ptr,
    set_name: unsafe extern "C" fn(Ptr, *const c_char) -> Ptr,
    allocate: unsafe extern "C" fn(Ptr, Ptr) -> Ptr,
    base: unsafe extern "C" fn(Ptr) -> Ptr,
    clear: unsafe extern "C" fn(Ptr, u8),
    set: unsafe extern "C" fn(Ptr, *const c_void, usize, usize),
    get: unsafe extern "C" fn(Ptr, Ptr, usize, usize),
    buffer_free: Free,
    context_free: Free,
    backend_free: Free,
}
impl Api {
    fn load(directory: &std::path::Path) -> Result<Self> {
        let library_path = directory.join("libggml-reference-rpc.so");
        let path = library_path
            .to_str()
            .ok_or("non-UTF8 comparison library path")?;
        let library = DynamicLibrary::open(&[path])?;
        // SAFETY: each exact signature below is declared by the frozen public
        // ggml.h, ggml-backend.h, ggml-alloc.h, ggml-cpu.h or ggml-rpc.h, checked
        // by the builder's input manifest. No private tensor layout is assumed.
        unsafe {
            Ok(Self {
                cpu_reg: library.symbol(c"ggml_backend_cpu_reg")?,
                reg_dev: library.symbol(c"ggml_backend_reg_dev_get")?,
                reg_count: library.symbol(c"ggml_backend_reg_dev_count")?,
                server: library.symbol(c"ggml_backend_rpc_start_server")?,
                rpc_init: library.symbol(c"ggml_backend_rpc_init")?,
                add_server: library.symbol(c"ggml_backend_rpc_add_server")?,
                device_memory: library.symbol(c"ggml_backend_rpc_get_device_memory")?,
                init: library.symbol(c"ggml_init")?,
                tensor_1d: library.symbol(c"ggml_new_tensor_1d")?,
                set_name: library.symbol(c"ggml_set_name")?,
                allocate: library.symbol(c"ggml_backend_alloc_ctx_tensors")?,
                base: library.symbol(c"ggml_backend_buffer_get_base")?,
                clear: library.symbol(c"ggml_backend_buffer_clear")?,
                set: library.symbol(c"ggml_backend_tensor_set")?,
                get: library.symbol(c"ggml_backend_tensor_get")?,
                buffer_free: library.symbol(c"ggml_backend_buffer_free")?,
                context_free: library.symbol(c"ggml_free")?,
                backend_free: library.symbol(c"ggml_backend_free")?,
                _library: library,
            })
        }
    }
}

struct NativeResources<'a> {
    api: &'a Api,
    buffer: Ptr,
    context: Ptr,
    backend: Ptr,
}
impl Drop for NativeResources<'_> {
    fn drop(&mut self) {
        // SAFETY: these are exclusively owned native handles. Free the remote
        // buffer before tensor metadata and dispatcher/backend; Api stays live.
        unsafe {
            if !self.buffer.is_null() {
                (self.api.buffer_free)(self.buffer);
            }
            if !self.context.is_null() {
                (self.api.context_free)(self.context);
            }
            if !self.backend.is_null() {
                (self.api.backend_free)(self.backend);
            }
        }
    }
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}
fn nonnull(pointer: Ptr, what: &str) -> Result<Ptr> {
    if pointer.is_null() {
        Err(invalid(format!("original native {what} returned null")))
    } else {
        Ok(pointer)
    }
}
fn endpoint(text: &str) -> Result<CString> {
    let address: SocketAddrV4 = text.parse()?;
    if !address.ip().is_loopback() || address.port() == 0 {
        return Err(invalid(
            "comparison driver accepts only explicit IPv4 loopback endpoints with nonzero ports",
        ));
    }
    Ok(CString::new(text)?)
}

fn main() -> Result<()> {
    if !cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_endian = "little"
    )) {
        return Err(invalid("comparison FFI is validated only on x86-64 Linux"));
    }
    let directory = match env::var_os("T08_REFERENCE_DIR") {
        Some(path) => PathBuf::from(path),
        None => env::current_exe()?
            .parent()
            .ok_or("missing executable parent")?
            .to_path_buf(),
    };
    let args: Vec<String> = env::args().skip(1).collect();
    let layout = Command::new(directory.join("rpc-layout")).output()?;
    if !layout.status.success() {
        return Err(invalid("isolated native layout probe failed"));
    }
    let layout_text = String::from_utf8(layout.stdout)?;
    // The only native struct passed by value is checked against actual C++ ABI.
    for fragment in [
        format!("\"init_params_size\":{}", std::mem::size_of::<InitParams>()),
        format!(
            "\"offset_init_no_alloc\":{}",
            std::mem::offset_of!(InitParams, no_alloc)
        ),
    ] {
        if !layout_text.contains(&fragment) {
            return Err(invalid("native ggml_init_params layout mismatch"));
        }
    }
    if args == ["--layout"] {
        print!("{layout_text}");
        return Ok(());
    }
    let api = Api::load(&directory)?;
    match args.as_slice() {
        [mode, target] if mode == "--client" => run_client(&api, &endpoint(target)?),
        [target] => {
            let endpoint = endpoint(target)?;
            // SAFETY: original registry/device remain valid for library lifetime,
            // null cache_dir disables cache, one device pointer lives through
            // this blocking server call. Test parent terminates the process.
            unsafe {
                let reg = nonnull((api.cpu_reg)(), "CPU registry")?;
                if (api.reg_count)(reg) == 0 {
                    return Err(invalid("original CPU registry is empty"));
                }
                let mut device = nonnull((api.reg_dev)(reg, 0), "CPU device")?;
                (api.server)(endpoint.as_ptr(), std::ptr::null(), 2, 1, &mut device);
            }
            Err(invalid("original native server unexpectedly returned"))
        }
        _ => Err(invalid(
            "usage: rpc-oracle --layout | --client 127.0.0.1:PORT | 127.0.0.1:PORT",
        )),
    }
}

fn run_client(api: &Api, endpoint: &CStr) -> Result<()> {
    let mut native = NativeResources {
        api,
        buffer: std::ptr::null_mut(),
        context: std::ptr::null_mut(),
        backend: std::ptr::null_mut(),
    };
    // SAFETY: original public C API owns opaque tensor layout and serialization.
    // All pointers are checked, array lifetimes span synchronous native calls,
    // and 16 bytes fit the 16-element F32 allocation. NativeResources orders
    // frees and keeps both backend dispatcher and shared library alive.
    unsafe {
        native.backend = nonnull((api.rpc_init)(endpoint.as_ptr(), 0), "RPC backend")?;
        let reg = nonnull((api.add_server)(endpoint.as_ptr()), "RPC registry")?;
        if (api.reg_count)(reg) != 1 {
            return Err(invalid("comparison expects exactly one remote device"));
        }
        native.context = nonnull(
            (api.init)(InitParams {
                mem_size: 1024 * 1024,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            }),
            "tensor context",
        )?;
        let tensor = nonnull((api.tensor_1d)(native.context, 0, 16), "F32 tensor")?;
        (api.set_name)(tensor, c"native-public-api-client".as_ptr());
        native.buffer = nonnull(
            (api.allocate)(native.context, native.backend),
            "RPC allocation",
        )?;
        nonnull((api.base)(native.buffer), "remote virtual base")?;
        (api.clear)(native.buffer, 0);
        let expected = [1.25f32, -2.5, 3.75, 4.0];
        let mut actual = [0.0f32; 4];
        (api.set)(
            tensor,
            expected.as_ptr().cast(),
            0,
            std::mem::size_of_val(&expected),
        );
        (api.get)(
            tensor,
            actual.as_mut_ptr().cast(),
            0,
            std::mem::size_of_val(&actual),
        );
        if actual.map(f32::to_bits) != expected.map(f32::to_bits) {
            return Err(invalid(format!(
                "original client tensor roundtrip mismatch: {actual:?}"
            )));
        }
        (api.buffer_free)(native.buffer);
        native.buffer = std::ptr::null_mut();
        let (mut free, mut total) = (0, 0);
        // A responding command confirms that FREE_BUFFER reached the server.
        (api.device_memory)(endpoint.as_ptr(), 0, &mut free, &mut total);
        if free > total {
            return Err(invalid(
                "original client received invalid memory accounting",
            ));
        }
    }
    drop(native);
    println!(
        "original C++ RPC client via Rust public-ABI driver -> Rust server: HELLO, DEVICE_COUNT, alignment/max, ALLOC_BUFFER, BUFFER_GET_BASE, INIT_TENSOR, BUFFER_CLEAR, SET_TENSOR, GET_TENSOR, FREE_BUFFER, GET_DEVICE_MEMORY barrier: PASS"
    );
    Ok(())
}
