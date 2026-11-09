//! APIR's control payloads only. DRM resources, shared buffers and the
//! guest/host submission engine are still pending; no virtual GPU is emulated.
use crate::{Error, Result};
pub const PROTOCOL_MAJOR: u32 = 0;
pub const PROTOCOL_MINOR: u32 = 2;
pub const HANDSHAKE_MAGIC: u32 = 0xab1e;
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlCommand {
    Handshake = 0,
    LoadLibrary = 1,
    Forward = 2,
}
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendCommand {
    DeviceGetDeviceCount = 0,
    DeviceGetCount = 1,
    DeviceGetName = 2,
    DeviceGetDescription = 3,
    DeviceGetType = 4,
    DeviceGetMemory = 5,
    DeviceSupportsOp = 6,
    DeviceGetBufferType = 7,
    DeviceGetProps = 8,
    DeviceBufferFromPtr = 9,
    BufferTypeGetName = 10,
    BufferTypeGetAlignment = 11,
    BufferTypeGetMaxSize = 12,
    BufferTypeIsHost = 13,
    BufferTypeAllocBuffer = 14,
    BufferTypeGetAllocSize = 15,
    BufferGetBase = 16,
    BufferSetTensor = 17,
    BufferGetTensor = 18,
    BufferCopyTensor = 19,
    BufferClear = 20,
    BufferFree = 21,
    BackendGraphCompute = 22,
}
pub const BACKEND_COMMANDS: [BackendCommand; 23] = [
    BackendCommand::DeviceGetDeviceCount,
    BackendCommand::DeviceGetCount,
    BackendCommand::DeviceGetName,
    BackendCommand::DeviceGetDescription,
    BackendCommand::DeviceGetType,
    BackendCommand::DeviceGetMemory,
    BackendCommand::DeviceSupportsOp,
    BackendCommand::DeviceGetBufferType,
    BackendCommand::DeviceGetProps,
    BackendCommand::DeviceBufferFromPtr,
    BackendCommand::BufferTypeGetName,
    BackendCommand::BufferTypeGetAlignment,
    BackendCommand::BufferTypeGetMaxSize,
    BackendCommand::BufferTypeIsHost,
    BackendCommand::BufferTypeAllocBuffer,
    BackendCommand::BufferTypeGetAllocSize,
    BackendCommand::BufferGetBase,
    BackendCommand::BufferSetTensor,
    BackendCommand::BufferGetTensor,
    BackendCommand::BufferCopyTensor,
    BackendCommand::BufferClear,
    BackendCommand::BufferFree,
    BackendCommand::BackendGraphCompute,
];
impl TryFrom<u32> for BackendCommand {
    type Error = Error;
    fn try_from(v: u32) -> Result<Self> {
        BACKEND_COMMANDS.get(v as usize).copied().ok_or_else(|| {
            Error::Protocol(format!("APIR backend command {v} outside dispatch table"))
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionRelation {
    Exact,
    MinorMismatch,
    MajorMismatch,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandshakeReply {
    pub host_major: u32,
    pub host_minor: u32,
    pub relation: VersionRelation,
}
/// Payload written by virtgpu_handshake through apir_encode_uint32_t twice.
pub fn handshake_request() -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&PROTOCOL_MAJOR.to_le_bytes());
    bytes[4..].copy_from_slice(&PROTOCOL_MINOR.to_le_bytes());
    bytes
}
/// The reference treats version mismatches as diagnostics after validating the
/// return magic. Preserve that distinction; this does not open a GPU transport.
pub fn decode_handshake(return_code: u32, payload: &[u8]) -> Result<HandshakeReply> {
    if return_code != HANDSHAKE_MAGIC {
        return Err(Error::Protocol(format!(
            "APIR handshake return code {return_code:#x}"
        )));
    }
    if payload.len() != 8 {
        return Err(Error::Protocol(
            "APIR handshake payload length must be 8".into(),
        ));
    }
    let major = u32::from_le_bytes(payload[..4].try_into().expect("length checked"));
    let minor = u32::from_le_bytes(payload[4..].try_into().expect("length checked"));
    let relation = if major != PROTOCOL_MAJOR {
        VersionRelation::MajorMismatch
    } else if minor != PROTOCOL_MINOR {
        VersionRelation::MinorMismatch
    } else {
        VersionRelation::Exact
    };
    Ok(HandshakeReply {
        host_major: major,
        host_minor: minor,
        relation,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handshake_fixture_and_version_diagnostics() {
        assert_eq!(handshake_request(), [0, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(
            decode_handshake(HANDSHAKE_MAGIC, &handshake_request())
                .unwrap()
                .relation,
            VersionRelation::Exact
        );
        assert_eq!(
            decode_handshake(HANDSHAKE_MAGIC, &[1, 0, 0, 0, 2, 0, 0, 0])
                .unwrap()
                .relation,
            VersionRelation::MajorMismatch
        );
        assert!(decode_handshake(1, &handshake_request()).is_err());
        assert!(decode_handshake(HANDSHAKE_MAGIC, &[0; 7]).is_err());
    }
    #[test]
    fn all_dispatch_ids_fixed() {
        for (id, command) in BACKEND_COMMANDS.iter().enumerate() {
            assert_eq!(*command as usize, id);
            assert_eq!(BackendCommand::try_from(id as u32).unwrap(), *command);
        }
        assert!(BackendCommand::try_from(23).is_err());
    }
}
