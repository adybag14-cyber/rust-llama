//! T08: bounded RPC 8.0.1 transport, external CBLAS adapter, and vendor probes.
//!
//! This checkpoint is not a complete replacement for the ten upstream backends.
//! External vendor math remains external. RPC's local server has a deliberately
//! explicit, small F32 graph executor; unavailable operations return errors.
pub mod blas;
pub mod dsp;
pub mod probe;
pub mod rpc;
pub mod virtgpu;

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Protocol(String),
    Bounds(&'static str),
    Unsupported(&'static str),
    Library(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO: {e}"),
            Self::Protocol(e) => write!(f, "RPC protocol: {e}"),
            Self::Bounds(e) => write!(f, "bounds: {e}"),
            Self::Unsupported(e) => write!(f, "not implemented: {e}"),
            Self::Library(e) => write!(f, "external library: {e}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<llama_platform_loader::LoadError> for Error {
    fn from(value: llama_platform_loader::LoadError) -> Self {
        Self::Library(value.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
pub use probe::{ProbeReport, ProbeStatus, probe_all};

#[derive(Debug)]
pub struct SelfCheckReport {
    pub rpc: String,
    pub blas: String,
}
/// Runs a real loopback TCP session and, when installed, a real CBLAS call.
/// Missing BLAS is recorded as unavailable instead of replaced with a fake.
pub fn self_check() -> Result<SelfCheckReport> {
    let rpc = rpc::loopback_self_check()?;
    let blas = match blas::Blas::open() {
        Ok(provider) => {
            let a = blas::MatrixView::contiguous(&[1., 2., 3., 4., 5., 6.], 2, 3)?;
            let b = blas::MatrixView::contiguous(&[7., 8., 9., 10., 11., 12.], 3, 2)?;
            let result = provider.matmul(a, b)?;
            if result != [58., 64., 139., 154.] {
                return Err(Error::Protocol(
                    "CBLAS self-check numerical mismatch".into(),
                ));
            }
            format!(
                "PASS: {} cblas_sgemm, output {:?}",
                provider.provider(),
                result
            )
        }
        Err(error) => format!("UNAVAILABLE: {error}"),
    };
    Ok(SelfCheckReport { rpc, blas })
}
