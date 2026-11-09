//! Loader/instance/physical-device feasibility probe, with no compute fallback.
//! Resource, descriptor, command and pipeline execution adapters remain pending.
use crate::*;
use llama_platform_loader::DynamicLibrary;
use std::ffi::{c_char, c_void};
use std::ptr;

type Handle = *mut c_void;
#[repr(C)]
struct ApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    application_name: *const c_char,
    application_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}
#[repr(C)]
struct InstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    application_info: *const ApplicationInfo,
    layer_count: u32,
    layers: *const *const c_char,
    extension_count: u32,
    extensions: *const *const c_char,
}
type EnumerateVersion = unsafe extern "system" fn(*mut u32) -> i32;
type CreateInstance =
    unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut Handle) -> i32;
type DestroyInstance = unsafe extern "system" fn(Handle, *const c_void);
type EnumerateDevices = unsafe extern "system" fn(Handle, *mut u32, *mut Handle) -> i32;

pub fn probe() -> ProbeReport {
    let mut report = ProbeReport {
        backend: "Vulkan",
        status: Status::Unavailable,
        library: None,
        devices: vec![],
        detail: String::new(),
    };
    let result = (|| -> Result<()> {
        let library = DynamicLibrary::open(if cfg!(windows) {
            &["vulkan-1.dll"]
        } else if cfg!(target_os = "macos") {
            &["libvulkan.1.dylib", "libMoltenVK.dylib"]
        } else {
            &["libvulkan.so.1", "libvulkan.so"]
        })?;
        report.library = Some(library.name().into());
        // SAFETY: these function types and repr(C) structures match the core
        // Vulkan C ABI. The library outlives the instance guard and all calls.
        unsafe {
            let create: CreateInstance = library.symbol(c"vkCreateInstance")?;
            let destroy: DestroyInstance = library.symbol(c"vkDestroyInstance")?;
            let devices: EnumerateDevices = library.symbol(c"vkEnumeratePhysicalDevices")?;
            let mut version = 1 << 22;
            if let Ok(enumerate) = library.symbol::<EnumerateVersion>(c"vkEnumerateInstanceVersion")
            {
                let code = enumerate(&mut version);
                if code != 0 {
                    return Err(Error(format!("vkEnumerateInstanceVersion returned {code}")));
                }
            }
            let version_text = format!(
                "{}.{}.{}",
                version >> 22,
                (version >> 12) & 0x3ff,
                version & 0xfff
            );
            let app = ApplicationInfo {
                s_type: 0,
                p_next: ptr::null(),
                application_name: c"Rust T07 feasibility probe".as_ptr(),
                application_version: 1,
                engine_name: c"llama-rust-rewrite".as_ptr(),
                engine_version: 1,
                api_version: 1 << 22,
            };
            let info = InstanceCreateInfo {
                s_type: 1,
                p_next: ptr::null(),
                flags: 0,
                application_info: &app,
                layer_count: 0,
                layers: ptr::null(),
                extension_count: 0,
                extensions: ptr::null(),
            };
            let mut instance = ptr::null_mut();
            let code = create(&info, ptr::null(), &mut instance);
            if code != 0 {
                return Err(Error(format!(
                    "Vulkan loader {version_text}; vkCreateInstance returned {code}{}",
                    if code == -9 {
                        " (VK_ERROR_INCOMPATIBLE_DRIVER; no compatible ICD/device)"
                    } else {
                        ""
                    }
                )));
            }
            if instance.is_null() {
                return Err(Error("vkCreateInstance returned a null instance".into()));
            }
            struct Instance {
                handle: Handle,
                destroy: DestroyInstance,
            }
            impl Drop for Instance {
                fn drop(&mut self) {
                    // SAFETY: exactly one destruction, default allocator matches creation.
                    unsafe { (self.destroy)(self.handle, ptr::null()) };
                }
            }
            let instance = Instance {
                handle: instance,
                destroy,
            };
            let mut handles = None;
            for _ in 0..3 {
                let mut count = 0;
                let code = devices(instance.handle, &mut count, ptr::null_mut());
                if code != 0 {
                    return Err(Error(format!(
                        "vkEnumeratePhysicalDevices count returned {code}"
                    )));
                }
                if count > 4096 {
                    return Err(Error("Vulkan device count exceeds probe limit".into()));
                }
                if count == 0 {
                    handles = Some(Vec::new());
                    break;
                }
                let mut list = vec![ptr::null_mut(); count as usize];
                let capacity = count;
                let code = devices(instance.handle, &mut count, list.as_mut_ptr());
                if code == 5 {
                    continue;
                } // VK_INCOMPLETE: the device set changed.
                if code != 0 || count > capacity {
                    return Err(Error(format!(
                        "vkEnumeratePhysicalDevices list returned {code}, count {count}, capacity {capacity}"
                    )));
                }
                list.truncate(count as usize);
                if list.iter().any(|h| h.is_null()) {
                    return Err(Error("null Vulkan physical device".into()));
                }
                handles = Some(list);
                break;
            }
            let handles = handles.ok_or_else(|| {
                Error("Vulkan device enumeration changed on all three attempts".into())
            })?;
            report.devices = (0..handles.len())
                .map(|i| format!("physical device {i}; properties/capability queries pending"))
                .collect();
            report.status = if handles.is_empty() {
                Status::Unavailable
            } else {
                Status::Available
            };
            report.detail = format!(
                "Vulkan loader {version_text}; {} actual physical devices. Generated GLSL/SPIR-V path is separate; Vulkan buffer/pipeline execution remains unimplemented.",
                handles.len()
            );
            Ok(())
        }
    })();
    if let Err(error) = result {
        report.status =
            if report.library.is_none() || error.0.contains("VK_ERROR_INCOMPATIBLE_DRIVER") {
                Status::Unavailable
            } else {
                Status::Failed
            };
        report.detail = error.0;
    }
    report
}
