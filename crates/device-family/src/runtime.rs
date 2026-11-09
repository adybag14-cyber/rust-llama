//! Independently loaded HIP/MUSA runtime resources, with synchronous copies.
//!
//! This is host implementation, not a claim that either family executes PTX.
//! Binary generation, module loading and kernels remain explicit family gates.
use crate::probe::symbol;
use crate::{DeviceError, Family, check, checked_range};
use llama_platform_loader::DynamicLibrary;
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::rc::Rc;

type GetDevice = unsafe extern "system" fn(*mut c_int) -> c_int;
type SetDevice = unsafe extern "system" fn(c_int) -> c_int;
type Allocate = unsafe extern "system" fn(*mut *mut c_void, usize) -> c_int;
type Free = unsafe extern "system" fn(*mut c_void) -> c_int;
type Copy = unsafe extern "system" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type Memset = unsafe extern "system" fn(*mut c_void, c_int, usize) -> c_int;
type Synchronize = unsafe extern "system" fn() -> c_int;
type StreamCreate = unsafe extern "system" fn(*mut *mut c_void, u32) -> c_int;
type StreamOperation = unsafe extern "system" fn(*mut c_void) -> c_int;
type EventRecord = unsafe extern "system" fn(*mut c_void, *mut c_void) -> c_int;

struct Api {
    _library: DynamicLibrary,
    family: Family,
    get_device: GetDevice,
    set_device: SetDevice,
    allocate: Allocate,
    free: Free,
    copy: Copy,
    memset: Memset,
    sync: Synchronize,
    stream_create: StreamCreate,
    stream_destroy: StreamOperation,
    stream_sync: StreamOperation,
    event_create: StreamCreate,
    event_record: EventRecord,
    event_sync: StreamOperation,
    event_destroy: StreamOperation,
}

pub struct RuntimeDriver {
    api: Rc<Api>,
}
impl RuntimeDriver {
    pub fn load(family: Family) -> Result<Self, DeviceError> {
        if family == Family::Cuda {
            return Err(DeviceError::InvalidArgument(
                "use CudaDriver for CUDA".into(),
            ));
        }
        #[cfg(windows)]
        let libraries: &[&str] = match family {
            Family::Hip => &["amdhip64.dll"],
            Family::Musa => &["musart.dll"],
            Family::Cuda => unreachable!(),
        };
        #[cfg(not(windows))]
        let libraries: &[&str] = match family {
            Family::Hip => &["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"],
            Family::Musa => &["libmusart.so", "libmusart.so.1"],
            Family::Cuda => unreachable!(),
        };
        let library =
            DynamicLibrary::open(libraries).map_err(|e| DeviceError::DriverUnavailable(e.0))?;
        let prefix = match family {
            Family::Hip => "hip",
            Family::Musa => "musa",
            Family::Cuda => unreachable!(),
        };
        macro_rules! load {
            ($suffix:literal) => {{
                let name = std::ffi::CString::new(format!("{prefix}{}", $suffix))
                    .expect("static ASCII name");
                // SAFETY: each expansion is assigned its documented runtime C
                // ABI type; the family-specific library is retained in Api.
                unsafe { symbol(&library, &name)? }
            }};
        }
        let api = Api {
            family,
            get_device: load!("GetDevice"),
            set_device: load!("SetDevice"),
            allocate: load!("Malloc"),
            free: load!("Free"),
            copy: load!("Memcpy"),
            memset: load!("Memset"),
            sync: load!("DeviceSynchronize"),
            stream_create: load!("StreamCreateWithFlags"),
            stream_destroy: load!("StreamDestroy"),
            stream_sync: load!("StreamSynchronize"),
            event_create: load!("EventCreateWithFlags"),
            event_record: load!("EventRecord"),
            event_sync: load!("EventSynchronize"),
            event_destroy: load!("EventDestroy"),
            _library: library,
        };
        Ok(Self { api: Rc::new(api) })
    }
    pub fn family(&self) -> Family {
        self.api.family
    }
    pub fn device(&self, ordinal: i32) -> Result<RuntimeDevice, DeviceError> {
        if ordinal < 0 {
            return Err(DeviceError::InvalidArgument(
                "negative device ordinal".into(),
            ));
        }
        let device = RuntimeDevice {
            api: self.api.clone(),
            ordinal,
        };
        {
            let _guard = device.enter()?;
        }
        Ok(device)
    }
}

#[derive(Clone)]
pub struct RuntimeDevice {
    api: Rc<Api>,
    ordinal: i32,
}
struct DeviceGuard<'a> {
    api: &'a Api,
    previous: i32,
    changed: bool,
}
impl Drop for DeviceGuard<'_> {
    fn drop(&mut self) {
        if self.changed {
            // SAFETY: previous ordinal came from the same runtime; restore the
            // caller's thread-local device after this scoped operation.
            unsafe {
                (self.api.set_device)(self.previous);
            }
        }
    }
}
impl RuntimeDevice {
    fn enter(&self) -> Result<DeviceGuard<'_>, DeviceError> {
        let mut previous = 0;
        // SAFETY: writable device ordinal output; scalar ordinal is checked by
        // the actual runtime, not an assumed CUDA-equivalent device list.
        unsafe {
            check(
                self.api.family,
                "GetDevice",
                (self.api.get_device)(&mut previous),
            )?;
            let changed = previous != self.ordinal;
            if changed {
                check(
                    self.api.family,
                    "SetDevice",
                    (self.api.set_device)(self.ordinal),
                )?;
            }
            Ok(DeviceGuard {
                api: &self.api,
                previous,
                changed,
            })
        }
    }
    fn same(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && Rc::ptr_eq(&self.api, &other.api)
    }
    pub fn allocate(&self, bytes: usize) -> Result<RuntimeBuffer, DeviceError> {
        if bytes == 0 || bytes > isize::MAX as usize {
            return Err(DeviceError::InvalidArgument(
                "allocation must fit a nonempty Rust slice".into(),
            ));
        }
        let _guard = self.enter()?;
        let mut pointer = std::ptr::null_mut();
        // SAFETY: size and output pointer valid, selected runtime device active.
        check(self.api.family, "Malloc", unsafe {
            (self.api.allocate)(&mut pointer, bytes)
        })?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            DeviceError::InvalidArgument("runtime returned null allocation".into())
        })?;
        let buffer = RuntimeBuffer {
            device: self.clone(),
            pointer,
            bytes,
        };
        // SAFETY: allocated device range fully covers memset; wait before
        // exposing the allocation to another operation.
        check(self.api.family, "Memset", unsafe {
            (self.api.memset)(pointer.as_ptr(), 0, bytes)
        })?;
        check(self.api.family, "DeviceSynchronize", unsafe {
            (self.api.sync)()
        })?;
        Ok(buffer)
    }
    pub fn stream(&self) -> Result<RuntimeStream, DeviceError> {
        let _guard = self.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: both runtimes document nonblocking flag 1; scalar ABI has no
        // device-property or compiler-dependent struct layout.
        check(self.api.family, "StreamCreateWithFlags", unsafe {
            (self.api.stream_create)(&mut handle, 1)
        })?;
        let handle = NonNull::new(handle)
            .ok_or_else(|| DeviceError::InvalidArgument("runtime returned null stream".into()))?;
        Ok(RuntimeStream {
            device: self.clone(),
            handle,
        })
    }
    pub fn event(&self) -> Result<RuntimeEvent, DeviceError> {
        let _guard = self.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: timing-disabled event flag 2, valid writable output pointer.
        check(self.api.family, "EventCreateWithFlags", unsafe {
            (self.api.event_create)(&mut handle, 2)
        })?;
        let handle = NonNull::new(handle)
            .ok_or_else(|| DeviceError::InvalidArgument("runtime returned null event".into()))?;
        Ok(RuntimeEvent {
            device: self.clone(),
            handle,
            recorded: false,
        })
    }
    /// A real allocate → copy → event → copy-back check. Does not launch a
    /// kernel and must never be reported as compute correctness evidence.
    pub fn memory_roundtrip(&self, input: &[u8]) -> Result<Vec<u8>, DeviceError> {
        let mut buffer = self.allocate(input.len())?;
        buffer.copy_from(0, input)?;
        let mut stream = self.stream()?;
        let mut event = self.event()?;
        event.record(&mut stream)?;
        event.synchronize()?;
        let mut output = vec![0; input.len()];
        buffer.copy_to(0, &mut output)?;
        Ok(output)
    }
}

pub struct RuntimeBuffer {
    device: RuntimeDevice,
    pointer: NonNull<c_void>,
    bytes: usize,
}
impl RuntimeBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    pub fn copy_from(&mut self, offset: usize, bytes: &[u8]) -> Result<(), DeviceError> {
        checked_range(offset, bytes.len(), self.bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let _guard = self.device.enter()?;
        // SAFETY: address arithmetic stays in the owned device allocation,
        // input bytes live through the synchronous copy. Both runtimes define
        // HostToDevice as 1; this is not a cast of a guessed property enum.
        check(self.device.api.family, "Memcpy(HostToDevice)", unsafe {
            (self.device.api.copy)(
                self.pointer
                    .as_ptr()
                    .cast::<u8>()
                    .wrapping_byte_add(offset)
                    .cast(),
                bytes.as_ptr().cast(),
                bytes.len(),
                1,
            )
        })?;
        // Preserve the explicitly blocking Rust contract independently of
        // each vendor's pageable-memory staging implementation. Future async
        // copies need a submission object retaining memory and dependencies.
        // SAFETY: the owning device is selected until this guard is dropped.
        check(
            self.device.api.family,
            "DeviceSynchronize after HtoD",
            unsafe { (self.device.api.sync)() },
        )
    }
    pub fn copy_to(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DeviceError> {
        checked_range(offset, bytes.len(), self.bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let _guard = self.device.enter()?;
        // SAFETY: exclusive host output and bounded source. DeviceToHost = 2.
        check(self.device.api.family, "Memcpy(DeviceToHost)", unsafe {
            (self.device.api.copy)(
                bytes.as_mut_ptr().cast(),
                self.pointer
                    .as_ptr()
                    .cast::<u8>()
                    .wrapping_byte_add(offset)
                    .cast(),
                bytes.len(),
                2,
            )
        })
    }
}
impl Drop for RuntimeBuffer {
    fn drop(&mut self) {
        if let Ok(_guard) = self.device.enter() {
            // SAFETY: one free of the owned, synchronously accessed allocation.
            unsafe {
                (self.device.api.free)(self.pointer.as_ptr());
            }
        }
    }
}

pub struct RuntimeStream {
    device: RuntimeDevice,
    handle: NonNull<c_void>,
}
impl RuntimeStream {
    pub fn synchronize(&mut self) -> Result<(), DeviceError> {
        let _guard = self.device.enter()?;
        // SAFETY: stream belongs to the active runtime/device and is retained.
        check(self.device.api.family, "StreamSynchronize", unsafe {
            (self.device.api.stream_sync)(self.handle.as_ptr())
        })
    }
}
impl Drop for RuntimeStream {
    fn drop(&mut self) {
        if let Ok(_guard) = self.device.enter() {
            // SAFETY: completion then destruction in the owning device.
            unsafe {
                (self.device.api.stream_sync)(self.handle.as_ptr());
                (self.device.api.stream_destroy)(self.handle.as_ptr());
            }
        }
    }
}

pub struct RuntimeEvent {
    device: RuntimeDevice,
    handle: NonNull<c_void>,
    recorded: bool,
}
impl RuntimeEvent {
    pub fn record(&mut self, stream: &mut RuntimeStream) -> Result<(), DeviceError> {
        if !self.device.same(&stream.device) {
            return Err(DeviceError::InvalidArgument(
                "runtime event/device mismatch".into(),
            ));
        }
        let _guard = self.device.enter()?;
        // SAFETY: live handles in the same runtime and device.
        check(self.device.api.family, "EventRecord", unsafe {
            (self.device.api.event_record)(self.handle.as_ptr(), stream.handle.as_ptr())
        })?;
        self.recorded = true;
        Ok(())
    }
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        if !self.recorded {
            return Err(DeviceError::InvalidArgument("event not recorded".into()));
        }
        let _guard = self.device.enter()?;
        // SAFETY: recorded, live event in the active runtime/device.
        check(self.device.api.family, "EventSynchronize", unsafe {
            (self.device.api.event_sync)(self.handle.as_ptr())
        })
    }
}
impl Drop for RuntimeEvent {
    fn drop(&mut self) {
        if let Ok(_guard) = self.device.enter() {
            // SAFETY: finish use before releasing the single owned event.
            unsafe {
                if self.recorded {
                    (self.device.api.event_sync)(self.handle.as_ptr());
                }
                (self.device.api.event_destroy)(self.handle.as_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_is_not_disguised_as_hip_or_musa() {
        assert!(matches!(
            RuntimeDriver::load(Family::Cuda),
            Err(DeviceError::InvalidArgument(_))
        ));
    }
}
