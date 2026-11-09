use crate::{DeviceError, DeviceInfo, Family, ProbeReport, ProbeStatus, check};
use llama_platform_loader::DynamicLibrary;
use std::ffi::{CStr, c_char, c_int};

type Init = unsafe extern "system" fn(u32) -> c_int;
type GetVersion = unsafe extern "system" fn(*mut c_int) -> c_int;
type GetCount = unsafe extern "system" fn(*mut c_int) -> c_int;
type GetDevice = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
type GetName = unsafe extern "system" fn(*mut c_char, c_int, c_int) -> c_int;
type GetMemory = unsafe extern "system" fn(*mut usize, c_int) -> c_int;
type GetAttribute = unsafe extern "system" fn(*mut c_int, c_int, c_int) -> c_int;

/// A read-only discovery attempt for each family, performed independently.
pub fn probe_all() -> Vec<ProbeReport> {
    [Family::Cuda, Family::Hip, Family::Musa]
        .into_iter()
        .map(probe)
        .collect()
}

pub fn probe(family: Family) -> ProbeReport {
    let mut report = ProbeReport {
        family,
        library: None,
        driver_version: None,
        devices: Vec::new(),
        status: ProbeStatus::Unavailable,
        detail: String::new(),
    };
    match discover(family, &mut report) {
        Ok(()) if report.devices.is_empty() => {
            report.detail =
                "Driver initialized; it reported zero devices. No device execution attempted."
                    .into();
        }
        Ok(()) => {
            report.status = ProbeStatus::Available;
            report.detail = "Device enumeration succeeded. Allocation, kernels and full backend parity are separate gates.".into();
        }
        Err(e) => {
            report.status = if matches!(
                e,
                DeviceError::DriverUnavailable(_) | DeviceError::Api { code: 100, .. }
            ) {
                ProbeStatus::Unavailable
            } else {
                ProbeStatus::Failed
            };
            report.detail = e.to_string();
        }
    }
    report
}

fn discover(family: Family, report: &mut ProbeReport) -> Result<(), DeviceError> {
    #[cfg(target_os = "windows")]
    let names: &[&str] = match family {
        Family::Cuda => &["nvcuda.dll"],
        Family::Hip => &["amdhip64.dll"],
        Family::Musa => &["musa.dll"],
    };
    #[cfg(not(target_os = "windows"))]
    let names: &[&str] = match family {
        Family::Cuda => &["libcuda.so.1", "libcuda.so"],
        Family::Hip => &["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"],
        Family::Musa => &["libmusa.so.1", "libmusa.so"],
    };
    let lib = DynamicLibrary::open(names).map_err(|e| DeviceError::DriverUnavailable(e.0))?;
    report.library = Some(lib.name().into());
    let (init_name, version_name, count_name, device_name, name_name, memory_names) = match family {
        Family::Cuda => (
            c"cuInit",
            c"cuDriverGetVersion",
            c"cuDeviceGetCount",
            c"cuDeviceGet",
            c"cuDeviceGetName",
            &[c"cuDeviceTotalMem_v2"][..],
        ),
        Family::Hip => (
            c"hipInit",
            c"hipDriverGetVersion",
            c"hipGetDeviceCount",
            c"hipDeviceGet",
            c"hipDeviceGetName",
            &[c"hipDeviceTotalMem"][..],
        ),
        Family::Musa => (
            c"muInit",
            c"muDriverGetVersion",
            c"muDeviceGetCount",
            c"muDeviceGet",
            c"muDeviceGetName",
            &[c"muDeviceTotalMem_v2", c"muDeviceTotalMem"][..],
        ),
    };
    // SAFETY: these are the documented scalar/pointer C ABIs. No vendor device
    // property struct is allocated with a guessed size. The library stays alive.
    let (init, version, count, get_device, get_name, memory): (
        Init,
        GetVersion,
        GetCount,
        GetDevice,
        GetName,
        GetMemory,
    ) = unsafe {
        (
            symbol(&lib, init_name)?,
            symbol(&lib, version_name)?,
            symbol(&lib, count_name)?,
            symbol(&lib, device_name)?,
            symbol(&lib, name_name)?,
            symbol_any(&lib, memory_names)?,
        )
    };
    // SAFETY: flags are zero as required and all output pointers are writable.
    unsafe {
        check(family, "initialize", init(0))?;
        let mut value = 0;
        check(family, "driver_version", version(&mut value))?;
        report.driver_version = Some(value);
        let mut n = 0;
        check(family, "device_count", count(&mut n))?;
        if !(0..=4096).contains(&n) {
            return Err(DeviceError::InvalidArgument(
                "driver returned implausible device count".into(),
            ));
        }
        for ordinal in 0..n {
            let mut device = 0;
            check(family, "device_get", get_device(&mut device, ordinal))?;
            let mut bytes = [0u8; 256];
            check(
                family,
                "device_name",
                get_name(bytes.as_mut_ptr().cast(), bytes.len() as i32, device),
            )?;
            let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
            let mut memory_bytes = 0;
            check(family, "device_memory", memory(&mut memory_bytes, device))?;
            let mut info = DeviceInfo {
                ordinal,
                name: String::from_utf8_lossy(&bytes[..end]).into_owned(),
                memory_bytes,
                subgroup_width: None,
                compute_capability: None,
            };
            // CUDA enumerants are not reused as HIP or MUSA enumerants.
            if family == Family::Cuda {
                let attribute: GetAttribute = symbol(&lib, c"cuDeviceGetAttribute")?;
                let mut width = 0;
                check(family, "warp_size", attribute(&mut width, 10, device))?;
                if width > 0 {
                    info.subgroup_width = Some(width as u32);
                }
                let (mut major, mut minor) = (0, 0);
                check(
                    family,
                    "compute_capability_major",
                    attribute(&mut major, 75, device),
                )?;
                check(
                    family,
                    "compute_capability_minor",
                    attribute(&mut minor, 76, device),
                )?;
                if major > 0 && minor >= 0 {
                    info.compute_capability = Some((major as u32, minor as u32));
                }
            }
            report.devices.push(info);
        }
    }
    Ok(())
}

/// # Safety
/// `T` must be the exact ABI pointer type and never outlive `library`.
pub(crate) unsafe fn symbol<T: Copy>(
    library: &DynamicLibrary,
    name: &CStr,
) -> Result<T, DeviceError> {
    // SAFETY: forwarded type/lifetime contract is the caller's obligation.
    unsafe { library.symbol(name) }
        .map_err(|e| DeviceError::MissingSymbol(format!("{}: {}", name.to_string_lossy(), e.0)))
}

/// # Safety
/// Every alternate must have exactly the same ABI as `T` on this target.
pub(crate) unsafe fn symbol_any<T: Copy>(
    library: &DynamicLibrary,
    names: &[&CStr],
) -> Result<T, DeviceError> {
    let mut errors = Vec::new();
    for name in names {
        // SAFETY: all names refer to the same ABI according to the caller.
        match unsafe { symbol(library, name) } {
            Ok(v) => return Ok(v),
            Err(e) => errors.push(e.to_string()),
        }
    }
    Err(DeviceError::MissingSymbol(errors.join("; ")))
}
