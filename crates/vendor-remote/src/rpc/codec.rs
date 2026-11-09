//! Explicit encoding of the frozen x86-64 little-endian RPC v8.0.1 wire.
//! The reference sends native integers, so cross-endian interoperability is not
//! established; unlike native struct casts this codec has no padding dependency.
use crate::{Error, Result};
use std::io::{Read, Write};

pub const VERSION: [u8; 3] = [8, 0, 1];
pub const CAPABILITY_BYTES: usize = 24;
pub const TENSOR_BYTES: usize = 296;
pub const GRAPH_CACHE_MAX: usize = 1024;
pub const COMMAND_COUNT: u8 = 24;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    AllocBuffer = 0,
    GetAlignment = 1,
    GetMaxSize = 2,
    BufferGetBase = 3,
    FreeBuffer = 4,
    BufferClear = 5,
    SetTensor = 6,
    SetTensorHash = 7,
    GetTensor = 8,
    CopyTensor = 9,
    GraphCompute = 10,
    GetDeviceMemory = 11,
    InitTensor = 12,
    GetAllocSize = 13,
    Hello = 14,
    DeviceCount = 15,
    GraphRecompute = 16,
    MemsetTensor = 17,
    SetTensor2d = 18,
    GetTensor2d = 19,
    CommInit = 20,
    CommAllreduce = 21,
    CommFree = 22,
    None = 23,
}
impl TryFrom<u8> for Command {
    type Error = Error;
    fn try_from(value: u8) -> Result<Self> {
        use Command::*;
        Ok(match value {
            0 => AllocBuffer,
            1 => GetAlignment,
            2 => GetMaxSize,
            3 => BufferGetBase,
            4 => FreeBuffer,
            5 => BufferClear,
            6 => SetTensor,
            7 => SetTensorHash,
            8 => GetTensor,
            9 => CopyTensor,
            10 => GraphCompute,
            11 => GetDeviceMemory,
            12 => InitTensor,
            13 => GetAllocSize,
            14 => Hello,
            15 => DeviceCount,
            16 => GraphRecompute,
            17 => MemsetTensor,
            18 => SetTensor2d,
            19 => GetTensor2d,
            20 => CommInit,
            21 => CommAllreduce,
            22 => CommFree,
            23 => None,
            _ => return Err(Error::Protocol(format!("unknown command {value}"))),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RpcTensor {
    pub id: u64,
    pub dtype: u32,
    pub buffer: u64,
    pub ne: [u32; 4],
    pub nb: [u32; 4],
    pub op: u32,
    pub op_params: [i32; 16],
    pub flags: i32,
    pub src: [u64; 10],
    pub view_src: u64,
    pub view_offs: u64,
    pub data: u64,
    pub name: [u8; 64],
    pub use_count: i32,
}
impl Default for RpcTensor {
    fn default() -> Self {
        Self {
            id: 0,
            dtype: 0,
            buffer: 0,
            ne: [1; 4],
            nb: [4; 4],
            op: 0,
            op_params: [0; 16],
            flags: 0,
            src: [0; 10],
            view_src: 0,
            view_offs: 0,
            data: 0,
            name: [0; 64],
            use_count: 0,
        }
    }
}
impl RpcTensor {
    pub fn contiguous_f32(
        id: u64,
        buffer: u64,
        data: u64,
        shape: [u32; 4],
        name: &str,
    ) -> Result<Self> {
        let mut nb = [0u32; 4];
        nb[0] = 4;
        for i in 1..4 {
            nb[i] = nb[i - 1]
                .checked_mul(shape[i - 1])
                .ok_or(Error::Bounds("RPC stride exceeds u32"))?;
        }
        if name.as_bytes().contains(&0) || name.len() >= 64 {
            return Err(Error::Bounds(
                "RPC tensor name must be under 64 non-NUL bytes",
            ));
        }
        let mut result = Self {
            id,
            buffer,
            data,
            ne: shape,
            nb,
            ..Self::default()
        };
        result.name[..name.len()].copy_from_slice(name.as_bytes());
        result.byte_len()?;
        Ok(result)
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TENSOR_BYTES);
        u64_to(&mut out, self.id);
        u32_to(&mut out, self.dtype);
        u64_to(&mut out, self.buffer);
        for n in self.ne {
            u32_to(&mut out, n);
        }
        for n in self.nb {
            u32_to(&mut out, n);
        }
        u32_to(&mut out, self.op);
        for n in self.op_params {
            u32_to(&mut out, n as u32);
        }
        u32_to(&mut out, self.flags as u32);
        for n in self.src {
            u64_to(&mut out, n);
        }
        u64_to(&mut out, self.view_src);
        u64_to(&mut out, self.view_offs);
        u64_to(&mut out, self.data);
        out.extend_from_slice(&self.name);
        u32_to(&mut out, self.use_count as u32);
        debug_assert_eq!(out.len(), TENSOR_BYTES);
        out
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != TENSOR_BYTES {
            return Err(Error::Protocol(format!(
                "tensor size {} != {TENSOR_BYTES}",
                bytes.len()
            )));
        }
        let mut r = Reader::new(bytes);
        let id = r.u64()?;
        let dtype = r.u32()?;
        let buffer = r.u64()?;
        let mut ne = [0; 4];
        for n in &mut ne {
            *n = r.u32()?;
        }
        let mut nb = [0; 4];
        for n in &mut nb {
            *n = r.u32()?;
        }
        let op = r.u32()?;
        let mut op_params = [0; 16];
        for n in &mut op_params {
            *n = r.u32()? as i32;
        }
        let flags = r.u32()? as i32;
        let mut src = [0; 10];
        for n in &mut src {
            *n = r.u64()?;
        }
        let view_src = r.u64()?;
        let view_offs = r.u64()?;
        let data = r.u64()?;
        let mut name = [0; 64];
        name.copy_from_slice(r.take(64)?);
        let use_count = r.u32()? as i32;
        let result = Self {
            id,
            dtype,
            buffer,
            ne,
            nb,
            op,
            op_params,
            flags,
            src,
            view_src,
            view_offs,
            data,
            name,
            use_count,
        };
        if !result.name.contains(&0) {
            return Err(Error::Protocol("tensor name lacks NUL termination".into()));
        }
        if result.op >= 102 {
            return Err(Error::Protocol("invalid operation ID".into()));
        }
        result.byte_len()?;
        Ok(result)
    }
    /// Validate the byte span of the implemented transfer types. Unimplemented
    /// quant layouts are rejected instead of guessed; T03 will supply all types.
    pub fn byte_len(&self) -> Result<u64> {
        let (block, bytes) = match self.dtype {
            0 => (1, 4),
            1 => (1, 2),
            2 => (32, 18),
            3 => (32, 20),
            6 => (32, 22),
            7 => (32, 24),
            8 => (32, 34),
            9 => (32, 36),
            24 => (1, 1),
            25 => (1, 2),
            26 => (1, 4),
            27 => (1, 8),
            28 => (1, 8),
            30 => (1, 2),
            _ => {
                return Err(Error::Unsupported(
                    "RPC dtype layout not yet integrated from T03",
                ));
            }
        };
        if self.ne.contains(&0) {
            return Ok(0);
        }
        if !self.ne[0].is_multiple_of(block) {
            return Err(Error::Bounds("quantized dimension is not block aligned"));
        }
        if self.nb[0] < bytes {
            return Err(Error::Bounds("RPC element/block stride is too small"));
        }
        let mut n = u64::from(bytes);
        for i in 0..4 {
            let count = if i == 0 {
                self.ne[0] / block
            } else {
                self.ne[i]
            };
            n = n
                .checked_add(
                    u64::from(count - 1)
                        .checked_mul(u64::from(self.nb[i]))
                        .ok_or(Error::Bounds("tensor span multiplication overflow"))?,
                )
                .ok_or(Error::Bounds("tensor span overflow"))?;
        }
        Ok(n)
    }
    pub fn elements(&self) -> Result<usize> {
        self.ne.iter().try_fold(1usize, |n, d| {
            n.checked_mul(*d as usize)
                .ok_or(Error::Bounds("tensor element count overflow"))
        })
    }
    pub fn is_contiguous_f32(&self) -> bool {
        if self.dtype != 0 {
            return false;
        }
        let mut stride = 4u64;
        for i in 0..4 {
            if u64::from(self.nb[i]) != stride {
                return false;
            }
            let Some(next) = stride.checked_mul(u64::from(self.ne[i])) else {
                return false;
            };
            stride = next;
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub device: u32,
    pub uid: u64,
    pub nodes: Vec<u64>,
    pub tensors: Vec<RpcTensor>,
}
impl Graph {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let nodes =
            u32::try_from(self.nodes.len()).map_err(|_| Error::Bounds("too many graph nodes"))?;
        let tensors = u32::try_from(self.tensors.len())
            .map_err(|_| Error::Bounds("too many graph tensors"))?;
        let mut out = Vec::new();
        u32_to(&mut out, self.device);
        u64_to(&mut out, self.uid);
        u32_to(&mut out, nodes);
        for &id in &self.nodes {
            u64_to(&mut out, id);
        }
        u32_to(&mut out, tensors);
        for tensor in &self.tensors {
            out.extend(tensor.encode());
        }
        Ok(out)
    }
    pub fn decode(input: &[u8], max_nodes: usize, max_tensors: usize) -> Result<Self> {
        let mut r = Reader::new(input);
        let device = r.u32()?;
        let uid = r.u64()?;
        let n = r.u32()? as usize;
        if n == 0 || n > max_nodes {
            return Err(Error::Bounds("graph node count"));
        }
        let ids = r.take(
            n.checked_mul(8)
                .ok_or(Error::Bounds("graph ID size overflow"))?,
        )?;
        let nodes = ids
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| u64::from_le_bytes(*p))
            .collect();
        let count = r.u32()? as usize;
        if count > max_tensors {
            return Err(Error::Bounds("graph tensor count"));
        }
        let bytes = r.take(
            count
                .checked_mul(TENSOR_BYTES)
                .ok_or(Error::Bounds("graph tensor size overflow"))?,
        )?;
        r.end()?;
        let tensors = bytes
            .as_chunks::<TENSOR_BYTES>()
            .0
            .iter()
            .map(|bytes| RpcTensor::decode(bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device,
            uid,
            nodes,
            tensors,
        })
    }
}

pub(crate) fn u32_to(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
pub(crate) fn u64_to(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    pub fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(Error::Bounds("message offset overflow"))?;
        let out = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::Protocol("truncated message".into()))?;
        self.pos = end;
        Ok(out)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("length checked"),
        ))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("length checked"),
        ))
    }
    pub fn tensor(&mut self) -> Result<RpcTensor> {
        RpcTensor::decode(self.take(TENSOR_BYTES)?)
    }
    pub fn rest(&mut self) -> &'a [u8] {
        let out = &self.bytes[self.pos..];
        self.pos = self.bytes.len();
        out
    }
    pub fn end(&self) -> Result<()> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::Protocol("trailing message bytes".into()))
        }
    }
}

pub fn write_request(
    w: &mut impl Write,
    command: Command,
    body: &[u8],
    limit: usize,
) -> Result<()> {
    if body.len() > limit {
        return Err(Error::Bounds("request exceeds frame limit"));
    }
    w.write_all(&[command as u8])?;
    write_response(w, body, limit)
}
pub fn write_response(w: &mut impl Write, body: &[u8], limit: usize) -> Result<()> {
    if body.len() > limit {
        return Err(Error::Bounds("response exceeds frame limit"));
    }
    w.write_all(&(body.len() as u64).to_le_bytes())?;
    w.write_all(body)?;
    w.flush()?;
    Ok(())
}
pub fn read_response(r: &mut impl Read, limit: usize) -> Result<Vec<u8>> {
    let mut size = [0u8; 8];
    r.read_exact(&mut size)?;
    let size = usize::try_from(u64::from_le_bytes(size))
        .map_err(|_| Error::Bounds("frame does not fit address space"))?;
    if size > limit {
        return Err(Error::Bounds("incoming frame exceeds limit"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(size)
        .map_err(|_| Error::Bounds("incoming frame allocation failed"))?;
    out.resize(size, 0);
    r.read_exact(&mut out)?;
    Ok(out)
}
pub fn read_request(r: &mut impl Read, limit: usize) -> Result<Option<(Command, Vec<u8>)>> {
    let mut command = [0];
    loop {
        match r.read(&mut command) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let command = Command::try_from(command[0])?;
    let body = read_response(r, limit)?;
    Ok(Some((command, body)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn fixed_wire_offsets() {
        let mut t = RpcTensor::contiguous_f32(
            0x0102030405060708,
            9,
            0x8877665544332211,
            [2, 3, 1, 1],
            "layout",
        )
        .unwrap();
        t.use_count = -7;
        let b = t.encode();
        assert_eq!(b.len(), 296);
        assert_eq!(&b[0..8], &[8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(&b[220..228], &t.data.to_le_bytes());
        assert_eq!(&b[228..235], b"layout\0");
        assert_eq!(&b[292..296], &(-7i32).to_le_bytes());
        assert_eq!(RpcTensor::decode(&b).unwrap(), t);
    }
    #[test]
    fn exact_command_ids() {
        assert_eq!(Command::Hello as u8, 14);
        assert_eq!(Command::CommFree as u8, 22);
        assert!(Command::try_from(24).is_err());
        for b in 0..24 {
            assert_eq!(Command::try_from(b).unwrap() as u8, b);
        }
    }
    #[test]
    fn little_endian_frame_fixture() {
        let mut b = Vec::new();
        write_request(&mut b, Command::Hello, &[0; 24], 100).unwrap();
        assert_eq!(&b[..9], &[14, 24, 0, 0, 0, 0, 0, 0, 0]);
        let decoded = read_request(&mut Cursor::new(b), 100).unwrap().unwrap();
        assert_eq!(decoded, (Command::Hello, vec![0; 24]));
    }
    #[test]
    fn malformed_frames_are_bounded() {
        assert!(read_response(&mut Cursor::new(u64::MAX.to_le_bytes()), 1024).is_err());
        assert!(read_response(&mut Cursor::new([3, 0, 0, 0, 0, 0, 0, 0, 1]), 1024).is_err());
        assert!(read_request(&mut Cursor::new([24]), 1024).is_err());
        assert!(read_request(&mut Cursor::new([]), 1024).unwrap().is_none());
    }
    #[test]
    fn malformed_tensors_are_rejected() {
        let t = RpcTensor::default();
        let mut b = t.encode();
        b[228..292].fill(65);
        assert!(RpcTensor::decode(&b).is_err());
        let mut t = RpcTensor {
            dtype: 4,
            ..RpcTensor::default()
        };
        assert!(t.byte_len().is_err());
        t.dtype = 2;
        t.ne[0] = 33;
        assert!(t.byte_len().is_err());
        t.ne[0] = 32;
        t.nb[0] = 18;
        t.nb[1..].fill(18);
        assert_eq!(t.byte_len().unwrap(), 18);
    }
    #[test]
    fn graph_decode_checks_counts_and_tail() {
        let t = RpcTensor::default();
        let g = Graph {
            device: 0,
            uid: 3,
            nodes: vec![1],
            tensors: vec![t],
        };
        let mut bytes = g.encode().unwrap();
        assert_eq!(Graph::decode(&bytes, 8, 8).unwrap().uid, 3);
        bytes.push(0);
        assert!(Graph::decode(&bytes, 8, 8).is_err());
        assert!(Graph::decode(&bytes, 0, 0).is_err());
    }
}
