//! Executable feasibility probes. Finding a shared library is not device parity.
use llama_platform_loader::DynamicLibrary;
use std::ffi::{CStr, c_void};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStatus {
    RustSubsetAvailable,
    ExternalSymbolsPresent,
    ExternalUnavailable,
    AbiGateOpen,
    RuntimeInitialized,
}
#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub family: &'static str,
    pub status: ProbeStatus,
    pub detail: String,
    pub remaining_gate: &'static str,
}
fn library_probe(
    family: &'static str,
    names: &[&str],
    symbols: &[&CStr],
    gate: &'static str,
) -> ProbeReport {
    match DynamicLibrary::open(names) {
        Ok(lib) => {
            let missing: Vec<_> = symbols
                .iter()
                .filter_map(|name| {
                    // SAFETY: the returned value is inspected only for existence;
                    // it is never called/dereferenced and cannot escape this loader.
                    unsafe { lib.symbol::<*mut c_void>(name) }
                        .err()
                        .map(|e| format!("{}: {e}", name.to_string_lossy()))
                })
                .collect();
            if missing.is_empty() {
                ProbeReport {
                    family,
                    status: ProbeStatus::ExternalSymbolsPresent,
                    detail: format!(
                        "{} loaded; {} required symbol(s) present. No device execution claimed.",
                        lib.name(),
                        symbols.len()
                    ),
                    remaining_gate: gate,
                }
            } else {
                ProbeReport {
                    family,
                    status: ProbeStatus::AbiGateOpen,
                    detail: format!("{} loaded; missing {}", lib.name(), missing.join(", ")),
                    remaining_gate: gate,
                }
            }
        }
        Err(e) => ProbeReport {
            family,
            status: ProbeStatus::ExternalUnavailable,
            detail: e.to_string(),
            remaining_gate: gate,
        },
    }
}

/// Execute the historical Level Zero initialization/enumeration pair in this
/// isolated probe. Never mix this with zeInitDrivers in the same process.
/// Signatures: oneAPI Level Zero core 1.9.1 / driver and device APIs 1.0.
pub fn level_zero_probe() -> ProbeReport {
    const GATE: &str = "Level Zero discovery is only a feasibility probe; SYCL/XMX/DPAS kernels, graph dispatch and hardware execution remain";
    let outcome = (|| -> crate::Result<(ProbeStatus, String)> {
        let lib = DynamicLibrary::open(&["libze_loader.so.1", "libze_loader.so"])?;
        type Init = unsafe extern "C" fn(u32) -> u32;
        type Drivers = unsafe extern "C" fn(*mut u32, *mut *mut c_void) -> u32;
        type Devices = unsafe extern "C" fn(*mut c_void, *mut u32, *mut *mut c_void) -> u32;
        // SAFETY: C signatures verified against official Level Zero docs; the
        // library remains owned throughout, and no handle escapes the probe.
        let (init, drivers, devices) = unsafe {
            (
                lib.symbol::<Init>(c"zeInit")?,
                lib.symbol::<Drivers>(c"zeDriverGet")?,
                lib.symbol::<Devices>(c"zeDeviceGet")?,
            )
        };
        // SAFETY: zero is a valid initialization flag mask; this probe uses only
        // zeInit + zeDriverGet, never the mutually exclusive zeInitDrivers API.
        let code = unsafe { init(0) };
        if code != 0 {
            return Ok((
                ProbeStatus::ExternalUnavailable,
                format!(
                    "{} loaded; zeInit(0) returned 0x{code:08x}; no driver/device execution established",
                    lib.name()
                ),
            ));
        }
        let mut count = 0u32;
        // SAFETY: live count pointer and optional-null handle array query.
        let code = unsafe { drivers(&mut count, std::ptr::null_mut()) };
        if code != 0 || count > 256 {
            return Ok((
                ProbeStatus::ExternalUnavailable,
                format!(
                    "zeDriverGet count result 0x{code:08x}, count={count}; bounded probe stopped"
                ),
            ));
        }
        if count == 0 {
            return Ok((
                ProbeStatus::ExternalUnavailable,
                "zeInit success; zero Level Zero drivers".into(),
            ));
        }
        let capacity = count;
        let mut handles = vec![std::ptr::null_mut(); capacity as usize];
        // SAFETY: count is the allocated array capacity, bounded to 256 handles.
        let code = unsafe { drivers(&mut count, handles.as_mut_ptr()) };
        if code != 0 || count > capacity {
            return Err(crate::Error::Library(format!(
                "zeDriverGet handles result 0x{code:08x}; count {count} exceeds capacity {capacity} or failed"
            )));
        }
        let mut total = 0u64;
        for handle in &handles[..count as usize] {
            if handle.is_null() {
                return Err(crate::Error::Library(
                    "Level Zero returned null driver handle".into(),
                ));
            }
            let mut n = 0u32;
            // SAFETY: valid driver from zeDriverGet; optional-null array query.
            let code = unsafe { devices(*handle, &mut n, std::ptr::null_mut()) };
            if code != 0 {
                return Err(crate::Error::Library(format!(
                    "zeDeviceGet count failed: 0x{code:08x}"
                )));
            }
            total += u64::from(n);
        }
        Ok((
            ProbeStatus::RuntimeInitialized,
            format!(
                "{}: zeInit success; {count} driver(s), {total} device(s) enumerated; no kernel execution claimed",
                lib.name()
            ),
        ))
    })();
    match outcome {
        Ok((status, detail)) => ProbeReport {
            family: "SYCL",
            status,
            detail,
            remaining_gate: GATE,
        },
        Err(error) => ProbeReport {
            family: "SYCL",
            status: ProbeStatus::ExternalUnavailable,
            detail: error.to_string(),
            remaining_gate: GATE,
        },
    }
}

/// Reports all ten T08 families separately, based on live local evidence.
/// The probes do not call C++ class entry points or substitute an ordinary CPU
/// path for a vendor backend. Library candidates follow the Linux snapshot.
pub fn probe_all() -> Vec<ProbeReport> {
    let mut out = vec![
        library_probe(
            "BLAS",
            &["libblas.so.3", "libopenblas.so.0"],
            &[c"cblas_sgemm"],
            "F32 adapter available; quant conversions, batch scheduling, vendor/thread matrix and T02 integration remain",
        ),
        library_probe(
            "CANN",
            &["libascendcl.so"],
            &[
                c"aclInit",
                c"aclrtGetDeviceCount",
                c"aclrtMalloc",
                c"aclrtFree",
            ],
            "Ascend hardware, ACLNN operation lowerings and workspace/resource integration",
        ),
        level_zero_probe(),
        library_probe(
            "zDNN",
            &["libzdnn.so"],
            &[c"zdnn_init", c"zdnn_matmul_op"],
            "IBM Z/NNPA execution, tensor transforms and dtype/operation lowering",
        ),
        library_probe(
            "OpenVINO",
            &["libopenvino_c.so"],
            &[c"ov_core_create", c"ov_core_free", c"ov_core_read_model"],
            "GGML-to-vendor graph construction, state/quantization and real CPU/GPU/NPU inference",
        ),
        library_probe(
            "Hexagon",
            &["libcdsprpc.so"],
            &[
                c"remote_handle64_open",
                c"remote_handle64_invoke",
                c"remote_handle64_close",
                c"rpcmem_alloc",
                c"rpcmem_free",
            ],
            "FastRPC generated IDL ABI, DSP kernels, HVX/HMX and physical load/execute; Rust ELF/SWIV utility is implemented",
        ),
    ];
    let et = std::env::var_os("ET_PLATFORM")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/opt/et".into());
    out.push(ProbeReport{family:"ET",status:ProbeStatus::AbiGateOpen,detail:format!("ET platform directory present: {}. Snapshot imports C++ rt::IRuntime and deviceLayer::IDeviceLayer; no verified C ABI.",et.is_dir()),remaining_gate:"Vendor runtime C++ ABI/source, custom RISC-V toolchain/startup, firmware or sysemu and physical execution; Rust forbidden-instruction text guard is implemented"});
    let zendnn = DynamicLibrary::open(&["libzendnnl.so", "libzendnn.so"]);
    out.push(ProbeReport{family:"ZenDNN",status:ProbeStatus::AbiGateOpen,detail:match zendnn{Ok(l)=>format!("{} found, but snapshot uses C++ zendnnl.hpp matmul_direct; no C ABI established",l.name()),Err(e)=>format!("{e}; pinned source uses C++ zendnnl.hpp matmul_direct")},remaining_gate:"Snapshot pin 1f399a75cc0993778374a51bea49b64a57879595; establish audited ABI or Rust behavior without a project C++ shim"});
    out.push(ProbeReport{family:"RPC",status:ProbeStatus::RustSubsetAvailable,detail:"Rust TCP 8.0.1 memory and bounded F32 graph subset compiled. Run self_check for live loopback evidence.".into(),remaining_gate:"Complete graph op/dtype integration, persistent tensor caching, async scheduler, Linux/Apple RDMA and collectives"});
    let mut vg = library_probe(
        "VirtGPU",
        &["libdrm.so.2"],
        &[c"drmGetVersion", c"drmFreeVersion", c"drmIoctl"],
        "APIR guest/host transport, shared memory and synchronization; real virtio GPU + virglrenderer required",
    );
    vg.detail.push_str(&format!(
        " /dev/dri present: {}. Rust APIR handshake codecs are implemented.",
        std::path::Path::new("/dev/dri").is_dir()
    ));
    out.push(vg);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_ten_families_are_accounted() {
        let reports = probe_all();
        assert_eq!(reports.len(), 10);
        let names: std::collections::HashSet<_> = reports.iter().map(|p| p.family).collect();
        assert_eq!(names.len(), 10);
        assert!(reports.iter().all(|p| !p.remaining_gate.is_empty()));
    }
    #[test]
    fn missing_vendor_is_never_success() {
        let p = library_probe(
            "fixture",
            &["lib-nonexistent-t08-fixture.so"],
            &[c"missing"],
            "pending",
        );
        assert_eq!(p.status, ProbeStatus::ExternalUnavailable);
    }
}
