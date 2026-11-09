//! OpenCL C API host path with library/context/queue/program/buffer ownership.
//! Source compilation is delegated to the external device compiler, with source
//! authored exclusively by the Rust kernel IR. No vendor kernel source is loaded.
use crate::*;
use llama_kernel_ir::{Access, BufferData, Target, kernels::DispatchPlan};
use llama_platform_loader::DynamicLibrary;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr::{self, NonNull};

type Handle = *mut c_void;
type GetPlatforms = unsafe extern "system" fn(u32, *mut Handle, *mut u32) -> c_int;
type GetDevices = unsafe extern "system" fn(Handle, u64, u32, *mut Handle, *mut u32) -> c_int;
type GetInfo = unsafe extern "system" fn(Handle, u32, usize, *mut c_void, *mut usize) -> c_int;
type ContextCallback =
    Option<unsafe extern "system" fn(*const c_char, *const c_void, usize, *mut c_void)>;
type CreateContext = unsafe extern "system" fn(
    *const isize,
    u32,
    *const Handle,
    ContextCallback,
    *mut c_void,
    *mut c_int,
) -> Handle;
type CreateQueue = unsafe extern "system" fn(Handle, Handle, u64, *mut c_int) -> Handle;
type Release = unsafe extern "system" fn(Handle) -> c_int;
type CreateBuffer =
    unsafe extern "system" fn(Handle, u64, usize, *mut c_void, *mut c_int) -> Handle;
type Transfer = unsafe extern "system" fn(
    Handle,
    Handle,
    u32,
    usize,
    usize,
    *mut c_void,
    u32,
    *const Handle,
    *mut Handle,
) -> c_int;
type CreateProgram = unsafe extern "system" fn(
    Handle,
    u32,
    *const *const c_char,
    *const usize,
    *mut c_int,
) -> Handle;
type BuildProgram = unsafe extern "system" fn(
    Handle,
    u32,
    *const Handle,
    *const c_char,
    Option<unsafe extern "system" fn(Handle, *mut c_void)>,
    *mut c_void,
) -> c_int;
type BuildInfo =
    unsafe extern "system" fn(Handle, Handle, u32, usize, *mut c_void, *mut usize) -> c_int;
type CreateKernel = unsafe extern "system" fn(Handle, *const c_char, *mut c_int) -> Handle;
type SetArg = unsafe extern "system" fn(Handle, u32, usize, *const c_void) -> c_int;
type Enqueue = unsafe extern "system" fn(
    Handle,
    Handle,
    u32,
    *const usize,
    *const usize,
    *const usize,
    u32,
    *const Handle,
    *mut Handle,
) -> c_int;

fn check(code: c_int, action: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error(format!("{action} returned OpenCL status {code}")))
    }
}

struct Api {
    library: DynamicLibrary,
    platforms: GetPlatforms,
    devices: GetDevices,
    info: GetInfo,
    create_context: CreateContext,
    create_queue: CreateQueue,
    release_context: Release,
    release_queue: Release,
    create_buffer: CreateBuffer,
    release_mem: Release,
    write: Transfer,
    read: Transfer,
    create_program: CreateProgram,
    build: BuildProgram,
    build_info: BuildInfo,
    release_program: Release,
    create_kernel: CreateKernel,
    release_kernel: Release,
    set_arg: SetArg,
    enqueue: Enqueue,
    finish: Release,
}
impl Api {
    fn load() -> Result<Self> {
        let library = DynamicLibrary::open(if cfg!(windows) {
            &["OpenCL.dll"]
        } else if cfg!(target_os = "macos") {
            &["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
        } else {
            &["libOpenCL.so.1", "libOpenCL.so"]
        })?;
        // SAFETY: the typedefs above match the Khronos OpenCL C API. Library
        // ownership encloses all resolved symbol uses and OpenCL objects.
        unsafe {
            Ok(Self {
                platforms: library.symbol(c"clGetPlatformIDs")?,
                devices: library.symbol(c"clGetDeviceIDs")?,
                info: library.symbol(c"clGetDeviceInfo")?,
                create_context: library.symbol(c"clCreateContext")?,
                create_queue: library.symbol(c"clCreateCommandQueue")?,
                release_context: library.symbol(c"clReleaseContext")?,
                release_queue: library.symbol(c"clReleaseCommandQueue")?,
                create_buffer: library.symbol(c"clCreateBuffer")?,
                release_mem: library.symbol(c"clReleaseMemObject")?,
                write: library.symbol(c"clEnqueueWriteBuffer")?,
                read: library.symbol(c"clEnqueueReadBuffer")?,
                create_program: library.symbol(c"clCreateProgramWithSource")?,
                build: library.symbol(c"clBuildProgram")?,
                build_info: library.symbol(c"clGetProgramBuildInfo")?,
                release_program: library.symbol(c"clReleaseProgram")?,
                create_kernel: library.symbol(c"clCreateKernel")?,
                release_kernel: library.symbol(c"clReleaseKernel")?,
                set_arg: library.symbol(c"clSetKernelArg")?,
                enqueue: library.symbol(c"clEnqueueNDRangeKernel")?,
                finish: library.symbol(c"clFinish")?,
                library,
            })
        }
    }
    fn text(&self, device: Handle, key: u32) -> Result<String> {
        let mut length = 0usize;
        // SAFETY: valid enumerated device, null output is the size query form.
        check(
            unsafe { (self.info)(device, key, 0, ptr::null_mut(), &mut length) },
            "clGetDeviceInfo size",
        )?;
        if length == 0 || length > 1024 * 1024 {
            return Err(Error("OpenCL info length outside limit".into()));
        }
        let mut bytes = vec![0u8; length];
        let mut actual = 0;
        // SAFETY: byte allocation is exactly length; device stays valid.
        check(
            unsafe { (self.info)(device, key, length, bytes.as_mut_ptr().cast(), &mut actual) },
            "clGetDeviceInfo text",
        )?;
        if actual > bytes.len() {
            return Err(Error("OpenCL info size changed".into()));
        }
        Ok(CStr::from_bytes_until_nul(&bytes)
            .map_err(|_| Error("unterminated OpenCL info".into()))?
            .to_string_lossy()
            .into_owned())
    }
    fn number<T: Default + Copy>(&self, device: Handle, key: u32) -> Result<T> {
        let mut result = T::default();
        let mut actual = 0usize;
        // SAFETY: private callers use only documented primitive info types, and
        // the driver may write at most size_of::<T>() into this live value.
        check(
            unsafe {
                (self.info)(
                    device,
                    key,
                    std::mem::size_of::<T>(),
                    (&mut result as *mut T).cast(),
                    &mut actual,
                )
            },
            "clGetDeviceInfo numeric",
        )?;
        if actual != std::mem::size_of::<T>() {
            return Err(Error("OpenCL info ABI width mismatch".into()));
        }
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub driver: String,
    pub extensions: String,
    pub il_version: Option<String>,
    pub max_workgroup: usize,
    pub max_allocation: u64,
    pub global_memory: u64,
    pub little_endian: bool,
    pub compiler_available: bool,
}
struct Device {
    handle: Handle,
    info: DeviceInfo,
}
pub struct OpenCl {
    api: Api,
    devices: Vec<Device>,
}

impl OpenCl {
    pub fn open() -> Result<Self> {
        let api = Api::load()?;
        let mut count = 0;
        // SAFETY: null platform output with live count is a valid size query.
        let code = unsafe { (api.platforms)(0, ptr::null_mut(), &mut count) };
        if code == -1001 {
            return Err(Error("clGetPlatformIDs returned CL_PLATFORM_NOT_FOUND_KHR (-1001); loader exists but no OpenCL platform is installed".into()));
        }
        check(code, "clGetPlatformIDs")?;
        if count > 1024 {
            return Err(Error("OpenCL platform count exceeds limit".into()));
        }
        let mut platforms = vec![ptr::null_mut(); count as usize];
        if count > 0 {
            // SAFETY: platform storage has the requested capacity.
            check(
                unsafe { (api.platforms)(count, platforms.as_mut_ptr(), ptr::null_mut()) },
                "clGetPlatformIDs list",
            )?;
        }
        let mut devices = Vec::new();
        for platform in platforms {
            if platform.is_null() {
                return Err(Error("null OpenCL platform handle".into()));
            }
            let mut n = 0;
            // SAFETY: platform is an enumerated live root, ALL includes CPU/GPU.
            let code = unsafe { (api.devices)(platform, 0xffff_ffff, 0, ptr::null_mut(), &mut n) };
            if code == -1 {
                continue;
            } // CL_DEVICE_NOT_FOUND
            check(code, "clGetDeviceIDs")?;
            if n > 1024 || devices.len() + n as usize > 4096 {
                return Err(Error("OpenCL device count exceeds limit".into()));
            }
            let mut handles = vec![ptr::null_mut(); n as usize];
            if n > 0 {
                // SAFETY: array matches requested capacity; root devices need no release.
                check(
                    unsafe {
                        (api.devices)(
                            platform,
                            0xffff_ffff,
                            n,
                            handles.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    },
                    "clGetDeviceIDs list",
                )?;
            }
            for handle in handles {
                if handle.is_null() {
                    return Err(Error("null OpenCL device handle".into()));
                }
                let info = DeviceInfo {
                    name: api.text(handle, 0x102b)?,
                    vendor: api.text(handle, 0x102c)?,
                    version: api.text(handle, 0x102f)?,
                    driver: api.text(handle, 0x102d)?,
                    extensions: api.text(handle, 0x1030)?,
                    il_version: api.text(handle, 0x105b).ok(),
                    max_workgroup: api.number::<usize>(handle, 0x1004)?,
                    max_allocation: api.number::<u64>(handle, 0x1010)?,
                    global_memory: api.number::<u64>(handle, 0x101f)?,
                    little_endian: api.number::<u32>(handle, 0x1026)? != 0,
                    compiler_available: api.number::<u32>(handle, 0x1028)? != 0,
                };
                devices.push(Device { handle, info });
            }
        }
        Ok(Self { api, devices })
    }
    pub fn library(&self) -> &str {
        self.api.library.name()
    }
    pub fn devices(&self) -> Vec<DeviceInfo> {
        self.devices.iter().map(|d| d.info.clone()).collect()
    }
    pub fn session(&self, index: usize) -> Result<Session<'_>> {
        let device = self
            .devices
            .get(index)
            .ok_or_else(|| Error("OpenCL device index out of range".into()))?;
        if !device.info.compiler_available {
            return Err(Error(
                "selected OpenCL device has no source compiler; IL path pending".into(),
            ));
        }
        if !device.info.little_endian {
            return Err(Error(
                "big-endian OpenCL storage conversion is not implemented".into(),
            ));
        }
        let mut error = 0;
        // SAFETY: one valid root device, no callback/userdata, context is owned.
        let context = unsafe {
            (self.api.create_context)(
                ptr::null(),
                1,
                &device.handle,
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        check(error, "clCreateContext")?;
        let context = Owned::new(context, self.api.release_context, "clCreateContext")?;
        // SAFETY: device belongs to context; properties 0 gives an in-order queue.
        let queue = unsafe { (self.api.create_queue)(context.raw(), device.handle, 0, &mut error) };
        check(error, "clCreateCommandQueue")?;
        let queue = Owned::new(queue, self.api.release_queue, "clCreateCommandQueue")?;
        Ok(Session {
            api: &self.api,
            device,
            queue,
            context,
        })
    }
}

struct Owned {
    handle: NonNull<c_void>,
    release: Release,
}
impl Owned {
    fn new(handle: Handle, release: Release, action: &str) -> Result<Self> {
        Ok(Self {
            handle: NonNull::new(handle)
                .ok_or_else(|| Error(format!("{action} returned null despite success")))?,
            release,
        })
    }
    fn raw(&self) -> Handle {
        self.handle.as_ptr()
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: one release of this owned OpenCL object; containing borrows retain the API.
        unsafe { (self.release)(self.raw()) };
    }
}

pub struct Session<'a> {
    api: &'a Api,
    device: &'a Device,
    queue: Owned,
    context: Owned,
}
#[derive(Debug, Clone)]
pub struct DeviceExecution {
    pub device: String,
    pub kernel: String,
    pub source_sha256: String,
    pub workgroups: u32,
    pub transferred_bytes: usize,
}
impl Session<'_> {
    /// Synchronous transaction: buffers are staged and only copied back after
    /// every kernel/read completes. This intentionally avoids implicit fallback.
    pub fn execute(
        &self,
        plan: &DispatchPlan,
        buffers: &mut BTreeMap<String, BufferData>,
    ) -> Result<DeviceExecution> {
        // Arbitrary mutable public IR cannot enter the safe device path: only
        // sealed, checked built-in plans establish bounds and write ownership.
        let kernel = plan.kernel();
        kernel.validate()?;
        if kernel.workgroup_size as usize > self.device.info.max_workgroup {
            return Err(Error(
                "kernel workgroup exceeds selected device limit".into(),
            ));
        }
        let mut total = 0usize;
        for b in &kernel.buffers {
            let data = buffers
                .get(&b.name)
                .ok_or_else(|| Error(format!("missing buffer {}", b.name)))?;
            if data.scalar() != b.scalar || data.len() < b.min_elements as usize {
                return Err(Error(format!("buffer {} type/size mismatch", b.name)));
            }
            let bytes = data
                .len()
                .checked_mul(4)
                .ok_or_else(|| Error("OpenCL buffer size overflow".into()))?;
            if bytes as u64 > self.device.info.max_allocation {
                return Err(Error("buffer exceeds device allocation limit".into()));
            }
            total = total
                .checked_add(bytes)
                .ok_or_else(|| Error("OpenCL total allocation overflow".into()))?;
        }
        if total > 256 * 1024 * 1024 {
            return Err(Error(
                "checkpoint execution budget is 256 MiB per dispatch".into(),
            ));
        }
        let artifact = llama_kernel_ir::emit(kernel, Target::OpenCl)?;
        let source_ptr = artifact.source.as_ptr().cast::<c_char>();
        let source_len = artifact.source.len();
        let mut error = 0;
        // SAFETY: exact length is provided, context and source are live.
        let program = unsafe {
            (self.api.create_program)(self.context.raw(), 1, &source_ptr, &source_len, &mut error)
        };
        check(error, "clCreateProgramWithSource")?;
        let program = Owned::new(
            program,
            self.api.release_program,
            "clCreateProgramWithSource",
        )?;
        // SAFETY: synchronous build on the session device. No fast-math flags.
        let code = unsafe {
            (self.api.build)(
                program.raw(),
                1,
                &self.device.handle,
                c"-cl-std=CL1.2".as_ptr(),
                None,
                ptr::null_mut(),
            )
        };
        if code != 0 {
            return Err(Error(format!(
                "clBuildProgram status {code}: {}",
                self.build_log(program.raw())
                    .unwrap_or_else(|e| e.to_string())
            )));
        }
        let entry = CString::new(artifact.entry_point.as_str())
            .map_err(|_| Error("NUL in OpenCL entry point".into()))?;
        // SAFETY: program was built and entry is a terminated valid identifier.
        let executable =
            unsafe { (self.api.create_kernel)(program.raw(), entry.as_ptr(), &mut error) };
        check(error, "clCreateKernel")?;
        let executable = Owned::new(executable, self.api.release_kernel, "clCreateKernel")?;
        let mut device_buffers = Vec::new();
        for (index, b) in kernel.buffers.iter().enumerate() {
            let mut bytes = buffers[&b.name].to_le_bytes();
            // SAFETY: size is bounded and nonzero, host pointer null with no COPY flag.
            let buffer = unsafe {
                (self.api.create_buffer)(
                    self.context.raw(),
                    if b.access == Access::Read { 4 } else { 1 },
                    bytes.len(),
                    ptr::null_mut(),
                    &mut error,
                )
            };
            check(error, "clCreateBuffer")?;
            let buffer = Owned::new(buffer, self.api.release_mem, "clCreateBuffer")?;
            // SAFETY: blocking transfer keeps bytes alive until it completes,
            // allocated buffer covers the full transfer, no event dependencies.
            check(
                unsafe {
                    (self.api.write)(
                        self.queue.raw(),
                        buffer.raw(),
                        1,
                        0,
                        bytes.len(),
                        bytes.as_mut_ptr().cast(),
                        0,
                        ptr::null(),
                        ptr::null_mut(),
                    )
                },
                "clEnqueueWriteBuffer",
            )?;
            let handle = buffer.raw();
            // SAFETY: argument value is a cl_mem handle with exact ABI width.
            check(
                unsafe {
                    (self.api.set_arg)(
                        executable.raw(),
                        index as u32,
                        std::mem::size_of::<Handle>(),
                        (&handle as *const Handle).cast(),
                    )
                },
                "clSetKernelArg",
            )?;
            device_buffers.push(buffer);
        }
        let groups = kernel.workgroups()?;
        let global = (groups as usize)
            .checked_mul(kernel.workgroup_size as usize)
            .ok_or_else(|| Error("OpenCL global size overflow".into()))?;
        let local = kernel.workgroup_size as usize;
        // SAFETY: 1D work sizes are nonzero and divisible; every kernel argument
        // is set, and all memory/program objects outlive synchronous completion.
        check(
            unsafe {
                (self.api.enqueue)(
                    self.queue.raw(),
                    executable.raw(),
                    1,
                    ptr::null(),
                    &global,
                    &local,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "clEnqueueNDRangeKernel",
        )?;
        let mut staged = BTreeMap::new();
        let mut transferred_bytes = total;
        for (b, device_buffer) in kernel.buffers.iter().zip(&device_buffers) {
            if b.access == Access::ReadWrite {
                let mut bytes = vec![0u8; buffers[&b.name].len() * 4];
                // SAFETY: blocking read after kernel in the same in-order queue,
                // destination capacity and source allocation equal transfer size.
                check(
                    unsafe {
                        (self.api.read)(
                            self.queue.raw(),
                            device_buffer.raw(),
                            1,
                            0,
                            bytes.len(),
                            bytes.as_mut_ptr().cast(),
                            0,
                            ptr::null(),
                            ptr::null_mut(),
                        )
                    },
                    "clEnqueueReadBuffer",
                )?;
                staged.insert(b.name.clone(), BufferData::from_le_bytes(b.scalar, &bytes)?);
                transferred_bytes = transferred_bytes
                    .checked_add(bytes.len())
                    .ok_or_else(|| Error("OpenCL transfer accounting overflow".into()))?;
            }
        }
        // SAFETY: session owns a valid queue; finish precedes result publication.
        check(unsafe { (self.api.finish)(self.queue.raw()) }, "clFinish")?;
        buffers.extend(staged);
        Ok(DeviceExecution {
            device: self.device.info.name.clone(),
            kernel: kernel.name.clone(),
            source_sha256: artifact.sha256,
            workgroups: groups,
            transferred_bytes,
        })
    }
    fn build_log(&self, program: Handle) -> Result<String> {
        let mut size = 0;
        // SAFETY: valid program/device pair and standard build-log size query.
        check(
            unsafe {
                (self.api.build_info)(
                    program,
                    self.device.handle,
                    0x1183,
                    0,
                    ptr::null_mut(),
                    &mut size,
                )
            },
            "clGetProgramBuildInfo size",
        )?;
        if size == 0 {
            return Ok(String::new());
        }
        if size > 4 * 1024 * 1024 {
            return Err(Error("OpenCL build log exceeds limit".into()));
        }
        let mut bytes = vec![0u8; size];
        // SAFETY: live byte allocation has exactly the requested capacity.
        check(
            unsafe {
                (self.api.build_info)(
                    program,
                    self.device.handle,
                    0x1183,
                    size,
                    bytes.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            },
            "clGetProgramBuildInfo log",
        )?;
        Ok(String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .into())
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: valid queue is finished before queue/context release; best effort on errors.
        unsafe { (self.api.finish)(self.queue.raw()) };
    }
}

pub fn probe() -> ProbeReport {
    match OpenCl::open() {
        Ok(driver)=>ProbeReport{backend:"OpenCL",status:if driver.devices.is_empty(){Status::Unavailable}else{Status::Available},library:Some(driver.library().into()),devices:driver.devices.iter().map(|d|format!("{} | {} | {} | compiler={} | IL={}",d.info.name,d.info.vendor,d.info.version,d.info.compiler_available,d.info.il_version.as_deref().unwrap_or("not reported"))).collect(),detail:"Real platform/device enumeration. Source compilation and buffer/queue execution path implemented; enumeration alone does not establish kernel execution.".into()},
        Err(error)=>ProbeReport{backend:"OpenCL",status:if error.0.contains("-1001")||error.0.contains("cannot open")||error.0.contains("not found"){Status::Unavailable}else{Status::Failed},library:None,devices:vec![],detail:error.0},
    }
}
