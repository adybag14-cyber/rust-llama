//! Portable accelerator progress: real OpenCL host calls, Vulkan enumeration,
//! four typed-IR emitters and optional external GLSL compilation.
//!
//! Device absence and unimplemented host paths are separate states. This crate
//! never routes a GPU label to the scalar validation interpreter.
pub mod compiler;
pub mod opencl;
pub mod vulkan;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<llama_platform_loader::LoadError> for Error {
    fn from(value: llama_platform_loader::LoadError) -> Self {
        Self(value.to_string())
    }
}
impl From<llama_kernel_ir::Error> for Error {
    fn from(value: llama_kernel_ir::Error) -> Self {
        Self(value.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Available,
    Unavailable,
    NotImplemented,
    Failed,
}
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub backend: &'static str,
    pub status: Status,
    pub library: Option<String>,
    pub devices: Vec<String>,
    pub detail: String,
}

pub fn probe_all() -> Vec<ProbeReport> {
    vec![opencl::probe(),vulkan::probe(),
        ProbeReport{backend:"Metal",status:Status::NotImplemented,library:None,devices:vec![],detail:"MSL source generation implemented. Objective-C/Metal resource and execution adapter pending; this host is not a Metal execution runner.".into()},
        ProbeReport{backend:"WebGPU",status:Status::NotImplemented,library:None,devices:vec![],detail:"WGSL source generation implemented. Native/browser adapter, async mapping and device-loss lifecycle pending; no WebGPU execution claimed.".into()},
    ]
}
