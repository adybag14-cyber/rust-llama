//! CUDA Driver API ownership. Every object retains its driver and context.
//!
//! No blanket Send/Sync implementations: context stacks and driver handles are
//! kept on their creating thread. Public generated-kernel execution is blocking
//! so buffers cannot disappear while work is in flight. Raw launch is unsafe.
use crate::probe::symbol;
use crate::{DeviceError, Family, check, checked_range};
use llama_platform_loader::DynamicLibrary;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;
use std::rc::Rc;

type Handle = *mut c_void;
type Init = unsafe extern "system" fn(u32) -> c_int;
type GetDevice = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
type PrimaryRetain = unsafe extern "system" fn(*mut Handle, c_int) -> c_int;
type PrimaryRelease = unsafe extern "system" fn(c_int) -> c_int;
type ContextPush = unsafe extern "system" fn(Handle) -> c_int;
type ContextPop = unsafe extern "system" fn(*mut Handle) -> c_int;
type ContextSync = unsafe extern "system" fn() -> c_int;
type MemAlloc = unsafe extern "system" fn(*mut u64, usize) -> c_int;
type MemFree = unsafe extern "system" fn(u64) -> c_int;
type HostAlloc = unsafe extern "system" fn(*mut Handle, usize) -> c_int;
type HostFree = unsafe extern "system" fn(Handle) -> c_int;
type CopyHtoD = unsafe extern "system" fn(u64, *const c_void, usize) -> c_int;
type CopyDtoH = unsafe extern "system" fn(*mut c_void, u64, usize) -> c_int;
type Memset = unsafe extern "system" fn(u64, u8, usize) -> c_int;
type Create = unsafe extern "system" fn(*mut Handle, u32) -> c_int;
type OneHandle = unsafe extern "system" fn(Handle) -> c_int;
type EventRecord = unsafe extern "system" fn(Handle, Handle) -> c_int;
type StreamWait = unsafe extern "system" fn(Handle, Handle, u32) -> c_int;
type ModuleLoad = unsafe extern "system" fn(*mut Handle, *const c_void) -> c_int;
type GetFunction = unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> c_int;
type Launch = unsafe extern "system" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

struct Api {
    _library: DynamicLibrary,
    device_get: GetDevice,
    primary_retain: PrimaryRetain,
    primary_release: PrimaryRelease,
    context_push: ContextPush,
    context_pop: ContextPop,
    context_sync: ContextSync,
    mem_alloc: MemAlloc,
    mem_free: MemFree,
    host_alloc: HostAlloc,
    host_free: HostFree,
    copy_htod: CopyHtoD,
    copy_dtoh: CopyDtoH,
    memset: Memset,
    stream_create: Create,
    stream_destroy: OneHandle,
    stream_sync: OneHandle,
    event_create: Create,
    event_destroy: OneHandle,
    event_record: EventRecord,
    event_sync: OneHandle,
    stream_wait_event: StreamWait,
    module_load: ModuleLoad,
    module_unload: OneHandle,
    module_function: GetFunction,
    launch: Launch,
}

#[derive(Clone)]
pub struct CudaDriver {
    api: Rc<Api>,
}
impl CudaDriver {
    pub fn load() -> Result<Self, DeviceError> {
        #[cfg(windows)]
        let names = &["nvcuda.dll"][..];
        #[cfg(not(windows))]
        let names = &["libcuda.so.1", "libcuda.so"][..];
        let library =
            DynamicLibrary::open(names).map_err(|e| DeviceError::DriverUnavailable(e.0))?;
        // SAFETY: CUDA's documented driver ABI types are declared above. Every
        // copied function pointer is held beside its owning library.
        let api = unsafe {
            let init: Init = symbol(&library, c"cuInit")?;
            check(Family::Cuda, "cuInit", init(0))?;
            Api {
                device_get: symbol(&library, c"cuDeviceGet")?,
                primary_retain: symbol(&library, c"cuDevicePrimaryCtxRetain")?,
                primary_release: symbol(&library, c"cuDevicePrimaryCtxRelease_v2")?,
                context_push: symbol(&library, c"cuCtxPushCurrent_v2")?,
                context_pop: symbol(&library, c"cuCtxPopCurrent_v2")?,
                context_sync: symbol(&library, c"cuCtxSynchronize")?,
                mem_alloc: symbol(&library, c"cuMemAlloc_v2")?,
                mem_free: symbol(&library, c"cuMemFree_v2")?,
                host_alloc: symbol(&library, c"cuMemAllocHost_v2")?,
                host_free: symbol(&library, c"cuMemFreeHost")?,
                copy_htod: symbol(&library, c"cuMemcpyHtoD_v2")?,
                copy_dtoh: symbol(&library, c"cuMemcpyDtoH_v2")?,
                memset: symbol(&library, c"cuMemsetD8_v2")?,
                stream_create: symbol(&library, c"cuStreamCreate")?,
                stream_destroy: symbol(&library, c"cuStreamDestroy_v2")?,
                stream_sync: symbol(&library, c"cuStreamSynchronize")?,
                event_create: symbol(&library, c"cuEventCreate")?,
                event_destroy: symbol(&library, c"cuEventDestroy_v2")?,
                event_record: symbol(&library, c"cuEventRecord")?,
                event_sync: symbol(&library, c"cuEventSynchronize")?,
                stream_wait_event: symbol(&library, c"cuStreamWaitEvent")?,
                module_load: symbol(&library, c"cuModuleLoadData")?,
                module_unload: symbol(&library, c"cuModuleUnload")?,
                module_function: symbol(&library, c"cuModuleGetFunction")?,
                launch: symbol(&library, c"cuLaunchKernel")?,
                _library: library,
            }
        };
        Ok(Self { api: Rc::new(api) })
    }

    pub fn context(&self, ordinal: i32) -> Result<CudaContext, DeviceError> {
        if ordinal < 0 {
            return Err(DeviceError::InvalidArgument(
                "negative device ordinal".into(),
            ));
        }
        let (mut device, mut handle) = (0, std::ptr::null_mut());
        // SAFETY: output pointers are valid; retain does not change the thread's
        // current context. Primary release is paired in ContextInner::drop.
        unsafe {
            check(
                Family::Cuda,
                "cuDeviceGet",
                (self.api.device_get)(&mut device, ordinal),
            )?;
            check(
                Family::Cuda,
                "cuDevicePrimaryCtxRetain",
                (self.api.primary_retain)(&mut handle, device),
            )?;
        }
        let Some(handle) = NonNull::new(handle) else {
            // SAFETY: balance the successful retain even on a broken driver.
            unsafe {
                (self.api.primary_release)(device);
            }
            return Err(DeviceError::InvalidArgument(
                "CUDA returned null context".into(),
            ));
        };
        Ok(CudaContext {
            inner: Rc::new(ContextInner {
                api: self.api.clone(),
                device,
                handle,
            }),
        })
    }
}

struct ContextInner {
    api: Rc<Api>,
    device: i32,
    handle: NonNull<c_void>,
}
impl ContextInner {
    fn enter(&self) -> Result<ContextGuard<'_>, DeviceError> {
        // SAFETY: retained context and library remain alive through the guard.
        check(Family::Cuda, "cuCtxPushCurrent_v2", unsafe {
            (self.api.context_push)(self.handle.as_ptr())
        })?;
        Ok(ContextGuard { context: self })
    }
}
impl Drop for ContextInner {
    fn drop(&mut self) {
        // SAFETY: all child resources retain this Rc, so they have been dropped
        // first. No API resets shared primary contexts.
        unsafe {
            (self.api.primary_release)(self.device);
        }
    }
}
struct ContextGuard<'a> {
    context: &'a ContextInner,
}
impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let mut popped = std::ptr::null_mut();
        // SAFETY: exactly one matching pop for this guard's successful push.
        unsafe {
            (self.context.api.context_pop)(&mut popped);
        }
    }
}

#[derive(Clone)]
pub struct CudaContext {
    inner: Rc<ContextInner>,
}
impl CudaContext {
    pub fn allocate(&self, bytes: usize) -> Result<CudaBuffer, DeviceError> {
        validate_allocation(bytes)?;
        let guard = self.inner.enter()?;
        let mut address = 0;
        // SAFETY: pointer is writable, active context is correct, nonzero size.
        check(Family::Cuda, "cuMemAlloc_v2", unsafe {
            (self.inner.api.mem_alloc)(&mut address, bytes)
        })?;
        let buffer = CudaBuffer {
            context: self.inner.clone(),
            address,
            bytes,
        };
        if address == 0 {
            return Err(DeviceError::InvalidArgument(
                "CUDA returned zero device address".into(),
            ));
        }
        // SAFETY: allocation covers the entire initialized range.
        check(Family::Cuda, "cuMemsetD8_v2", unsafe {
            (self.inner.api.memset)(address, 0, bytes)
        })?;
        check(Family::Cuda, "cuCtxSynchronize", unsafe {
            (self.inner.api.context_sync)()
        })?;
        drop(guard);
        Ok(buffer)
    }

    pub fn allocate_pinned(&self, bytes: usize) -> Result<PinnedBuffer, DeviceError> {
        validate_allocation(bytes)?;
        let _guard = self.inner.enter()?;
        let mut pointer = std::ptr::null_mut();
        // SAFETY: valid output pointer; size checked to fit Rust slice limits.
        check(Family::Cuda, "cuMemAllocHost_v2", unsafe {
            (self.inner.api.host_alloc)(&mut pointer, bytes)
        })?;
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
            DeviceError::InvalidArgument("CUDA returned null pinned allocation".into())
        })?;
        // SAFETY: vendor allocated bytes writable host bytes; initialize before
        // exposing a safe Rust slice.
        unsafe {
            pointer.as_ptr().write_bytes(0, bytes);
        }
        Ok(PinnedBuffer {
            context: self.inner.clone(),
            pointer,
            bytes,
        })
    }

    pub fn stream(&self) -> Result<CudaStream, DeviceError> {
        let _guard = self.inner.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: CU_STREAM_NON_BLOCKING = 1; valid context and output pointer.
        check(Family::Cuda, "cuStreamCreate", unsafe {
            (self.inner.api.stream_create)(&mut handle, 1)
        })?;
        Ok(CudaStream {
            context: self.inner.clone(),
            handle: require_handle(handle, "stream")?,
        })
    }

    pub fn event(&self) -> Result<CudaEvent, DeviceError> {
        let _guard = self.inner.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: CU_EVENT_DISABLE_TIMING = 2; output pointer is valid.
        check(Family::Cuda, "cuEventCreate", unsafe {
            (self.inner.api.event_create)(&mut handle, 2)
        })?;
        Ok(CudaEvent {
            context: self.inner.clone(),
            handle: require_handle(handle, "event")?,
            recorded: false,
        })
    }

    /// Load externally supplied PTX through the pointer-only CUDA module API.
    /// Prefer `compile_generated` when the module comes from this crate's IR.
    ///
    /// # Safety
    /// The bytes must be complete, NUL-terminated textual PTX, not a truncated
    /// cubin/fatbin or another binary image disguised as a C string. CUDA's
    /// loader sniffs image formats without accepting a byte length, so CStr
    /// alone cannot establish the vendor parser's full memory contract.
    pub unsafe fn module(&self, ptx: &CStr) -> Result<CudaModule, DeviceError> {
        let _guard = self.inner.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: terminated PTX source lives through JIT loading. Loading is
        // distinct from executing it, whose arbitrary-pointer ABI is unsafe.
        check(Family::Cuda, "cuModuleLoadData", unsafe {
            (self.inner.api.module_load)(&mut handle, ptx.as_ptr().cast())
        })?;
        Ok(CudaModule {
            inner: Rc::new(ModuleInner {
                context: self.inner.clone(),
                handle: require_handle(handle, "module")?,
            }),
        })
    }

    /// Safe JIT entry restricted to an immutable artifact created by this
    /// crate's typed generator. Callers cannot replace its source bytes.
    pub fn compile_generated(
        &self,
        generated: &crate::ptx::GeneratedPtx,
    ) -> Result<CudaModule, DeviceError> {
        let source = std::ffi::CString::new(generated.source())
            .map_err(|_| DeviceError::InvalidArgument("NUL in emitted PTX".into()))?;
        // SAFETY: GeneratedPtx's source is private and produced only by the
        // typed PTX emitter. It is complete text, never a binary image prefix.
        unsafe { self.module(&source) }
    }

    /// Execute only this crate's validated generated kernel, keeping all device
    /// allocations, module, arguments and stream alive until synchronization.
    /// The operation is blocking; asynchronous submission is still a T02 gate.
    pub fn run_generated(
        &self,
        generated: &crate::ptx::GeneratedPtx,
        buffers: &mut [Vec<u8>],
    ) -> Result<(), DeviceError> {
        generated
            .validate_buffers(buffers)
            .map_err(|e| DeviceError::InvalidArgument(e.to_string()))?;
        let name = std::ffi::CString::new(generated.entry_point())
            .map_err(|_| DeviceError::InvalidArgument("NUL in emitted entry name".into()))?;
        let module = self.compile_generated(generated)?;
        let function = module.function(&name)?;
        let mut stream = self.stream()?;
        let mut device = Vec::with_capacity(buffers.len());
        for bytes in buffers.iter() {
            let mut buffer = self.allocate(bytes.len())?;
            buffer.copy_from(0, bytes)?;
            device.push(buffer);
        }
        let mut addresses: Vec<u64> = device.iter().map(|buffer| buffer.address).collect();
        let mut arguments: Vec<*mut c_void> = addresses
            .iter_mut()
            .map(|address| (address as *mut u64).cast())
            .collect();
        let (grid, block) = generated.launch_dimensions();
        // SAFETY: generated code's complete buffer contract was checked above;
        // each parameter is a u64 device address. All resources are retained
        // until launch_sync has waited even if launch reports an error.
        unsafe {
            function.launch_sync(&mut stream, [grid, 1, 1], [block, 1, 1], 0, &mut arguments)?;
        }
        let out = generated.output_buffer();
        device[out].copy_to(0, &mut buffers[out])?;
        Ok(())
    }
}

fn validate_allocation(bytes: usize) -> Result<(), DeviceError> {
    if bytes == 0 || bytes > isize::MAX as usize {
        Err(DeviceError::InvalidArgument(
            "allocation must be in 1..=isize::MAX".into(),
        ))
    } else {
        Ok(())
    }
}
fn require_handle(handle: Handle, kind: &str) -> Result<NonNull<c_void>, DeviceError> {
    NonNull::new(handle)
        .ok_or_else(|| DeviceError::InvalidArgument(format!("CUDA returned null {kind}")))
}

pub struct CudaBuffer {
    context: Rc<ContextInner>,
    address: u64,
    bytes: usize,
}
impl CudaBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    pub fn device_address(&self) -> u64 {
        self.address
    }
    pub fn copy_from(&mut self, offset: usize, bytes: &[u8]) -> Result<(), DeviceError> {
        checked_range(offset, bytes.len(), self.bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let address = self
            .address
            .checked_add(offset as u64)
            .ok_or_else(|| DeviceError::InvalidArgument("address overflow".into()))?;
        let _guard = self.context.enter()?;
        // SAFETY: bounds checked; host bytes remain borrowed until the copy
        // and explicit context completion have both returned.
        check(Family::Cuda, "cuMemcpyHtoD_v2", unsafe {
            (self.context.api.copy_htod)(address, bytes.as_ptr().cast(), bytes.len())
        })?;
        // Pageable HtoD may return after staging, before its default-stream DMA
        // completes. Nonblocking kernel streams do not implicitly wait on that
        // stream. Complete the upload before exposing a blocking copy result.
        // SAFETY: correct active context and retained input/device allocations.
        check(Family::Cuda, "cuCtxSynchronize after HtoD", unsafe {
            (self.context.api.context_sync)()
        })
    }
    pub fn copy_to(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DeviceError> {
        checked_range(offset, bytes.len(), self.bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let address = self
            .address
            .checked_add(offset as u64)
            .ok_or_else(|| DeviceError::InvalidArgument("address overflow".into()))?;
        let _guard = self.context.enter()?;
        // SAFETY: destination is exclusively borrowed and range is in bounds.
        check(Family::Cuda, "cuMemcpyDtoH_v2", unsafe {
            (self.context.api.copy_dtoh)(bytes.as_mut_ptr().cast(), address, bytes.len())
        })
    }
}
impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if let Ok(_guard) = self.context.enter() {
            // SAFETY: generated execution always synchronizes before return;
            // raw callers must satisfy launch_sync's lifetime requirements.
            unsafe {
                (self.context.api.mem_free)(self.address);
            }
        }
    }
}

pub struct PinnedBuffer {
    context: Rc<ContextInner>,
    pointer: NonNull<u8>,
    bytes: usize,
}
impl PinnedBuffer {
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is initialized, live, and not asynchronously
        // exposed by any safe method. Slice lifetime is bounded by self.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.bytes) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive self borrow and initialized, writable host bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.bytes) }
    }
}
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if let Ok(_guard) = self.context.enter() {
            // SAFETY: the allocator/free pair and context are retained.
            unsafe {
                (self.context.api.host_free)(self.pointer.as_ptr().cast());
            }
        }
    }
}

pub struct CudaStream {
    context: Rc<ContextInner>,
    handle: NonNull<c_void>,
}
impl CudaStream {
    pub fn synchronize(&mut self) -> Result<(), DeviceError> {
        let _guard = self.context.enter()?;
        // SAFETY: stream exists and is in the active retained context.
        check(Family::Cuda, "cuStreamSynchronize", unsafe {
            (self.context.api.stream_sync)(self.handle.as_ptr())
        })
    }
    pub fn wait_event(&mut self, event: &CudaEvent) -> Result<(), DeviceError> {
        if !Rc::ptr_eq(&self.context, &event.context) || !event.recorded {
            return Err(DeviceError::InvalidArgument(
                "event must be recorded in the same context".into(),
            ));
        }
        let _guard = self.context.enter()?;
        // SAFETY: both handles have live ownership and flags must be zero.
        check(Family::Cuda, "cuStreamWaitEvent", unsafe {
            (self.context.api.stream_wait_event)(self.handle.as_ptr(), event.handle.as_ptr(), 0)
        })
    }
}
impl Drop for CudaStream {
    fn drop(&mut self) {
        if let Ok(_guard) = self.context.enter() {
            // SAFETY: synchronize before single destruction of the stream.
            unsafe {
                (self.context.api.stream_sync)(self.handle.as_ptr());
                (self.context.api.stream_destroy)(self.handle.as_ptr());
            }
        }
    }
}

pub struct CudaEvent {
    context: Rc<ContextInner>,
    handle: NonNull<c_void>,
    recorded: bool,
}
impl CudaEvent {
    pub fn record(&mut self, stream: &mut CudaStream) -> Result<(), DeviceError> {
        if !Rc::ptr_eq(&self.context, &stream.context) {
            return Err(DeviceError::InvalidArgument("context mismatch".into()));
        }
        let _guard = self.context.enter()?;
        // SAFETY: both handles live in this context, event outlives recording.
        check(Family::Cuda, "cuEventRecord", unsafe {
            (self.context.api.event_record)(self.handle.as_ptr(), stream.handle.as_ptr())
        })?;
        self.recorded = true;
        Ok(())
    }
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        if !self.recorded {
            return Err(DeviceError::InvalidArgument(
                "event has not been recorded".into(),
            ));
        }
        let _guard = self.context.enter()?;
        // SAFETY: event has been recorded and remains owned.
        check(Family::Cuda, "cuEventSynchronize", unsafe {
            (self.context.api.event_sync)(self.handle.as_ptr())
        })
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        if let Ok(_guard) = self.context.enter() {
            // SAFETY: finish a recorded event before single destruction.
            unsafe {
                if self.recorded {
                    (self.context.api.event_sync)(self.handle.as_ptr());
                }
                (self.context.api.event_destroy)(self.handle.as_ptr());
            }
        }
    }
}

struct ModuleInner {
    context: Rc<ContextInner>,
    handle: NonNull<c_void>,
}
impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Ok(_guard) = self.context.enter() {
            // SAFETY: functions retain ModuleInner; all safe launches block.
            unsafe {
                (self.context.api.module_unload)(self.handle.as_ptr());
            }
        }
    }
}
pub struct CudaModule {
    inner: Rc<ModuleInner>,
}
impl CudaModule {
    pub fn function(&self, name: &CStr) -> Result<CudaFunction, DeviceError> {
        let _guard = self.inner.context.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: live module, writable output, terminated function name.
        check(Family::Cuda, "cuModuleGetFunction", unsafe {
            (self.inner.context.api.module_function)(
                &mut handle,
                self.inner.handle.as_ptr(),
                name.as_ptr(),
            )
        })?;
        Ok(CudaFunction {
            module: self.inner.clone(),
            handle: require_handle(handle, "function")?,
        })
    }
}
pub struct CudaFunction {
    module: Rc<ModuleInner>,
    handle: NonNull<c_void>,
}
impl CudaFunction {
    /// Launch an arbitrary module function and wait before returning.
    ///
    /// # Safety
    /// Parameters must have the function's exact device ABI. Every accessible
    /// buffer must belong to this context, be large enough and remain alive;
    /// writable regions must not overlap host access. Kernel code itself must
    /// be valid for these allocations. The parameter array is borrowed through
    /// submission; all resources must survive the blocking synchronization.
    pub unsafe fn launch_sync(
        &self,
        stream: &mut CudaStream,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        parameters: &mut [*mut c_void],
    ) -> Result<(), DeviceError> {
        if !Rc::ptr_eq(&self.module.context, &stream.context) {
            return Err(DeviceError::InvalidArgument("context mismatch".into()));
        }
        if grid.contains(&0) || block.contains(&0) {
            return Err(DeviceError::InvalidArgument(
                "launch dimensions must be positive".into(),
            ));
        }
        let _guard = self.module.context.enter()?;
        let api = &self.module.context.api;
        // SAFETY: caller guarantees kernel ABI, memory, launch configuration,
        // and lifetime requirements. Handles belong to the same context.
        let launched = unsafe {
            (api.launch)(
                self.handle.as_ptr(),
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                stream.handle.as_ptr(),
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        // SAFETY: wait even when launch returned an error, before any callers
        // may release retained memory. Failed sync is returned, never ignored.
        let synced = unsafe { (api.stream_sync)(stream.handle.as_ptr()) };
        check(Family::Cuda, "cuLaunchKernel", launched)?;
        check(Family::Cuda, "cuStreamSynchronize", synced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_and_oversize_allocations_are_rejected() {
        assert!(validate_allocation(0).is_err());
        assert!(validate_allocation(1).is_ok());
        assert!(validate_allocation(usize::MAX).is_err());
    }
}
