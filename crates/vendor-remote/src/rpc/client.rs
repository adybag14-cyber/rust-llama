use super::Limits;
use super::codec::*;
use crate::{Error, Result};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteBuffer {
    pub handle: u64,
    pub base: u64,
    pub size: u64,
}
impl RemoteBuffer {
    pub fn tensor_f32(
        self,
        id: u64,
        offset: u64,
        shape: [u32; 4],
        name: &str,
    ) -> Result<RpcTensor> {
        let data = self
            .base
            .checked_add(offset)
            .ok_or(Error::Bounds("tensor base overflow"))?;
        let tensor = RpcTensor::contiguous_f32(id, self.handle, data, shape, name)?;
        if offset > self.size || tensor.byte_len()? > self.size - offset {
            return Err(Error::Bounds("tensor does not fit remote allocation"));
        }
        Ok(tensor)
    }
}
/// A synchronous client for the snapshot's existing TCP wire. It advertises no
/// RDMA features. A failed response poisons the connection to avoid framing drift.
pub struct RpcClient {
    stream: TcpStream,
    limits: Limits,
    poisoned: bool,
    pub server_version: [u8; 3],
}
impl RpcClient {
    pub fn connect(address: SocketAddr, limits: Limits) -> Result<Self> {
        limits.validate()?;
        let stream = TcpStream::connect_timeout(&address, Duration::from_secs(5))?;
        Self::from_stream(stream, limits)
    }
    pub fn from_stream(stream: TcpStream, limits: Limits) -> Result<Self> {
        limits.validate()?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let mut client = Self {
            stream,
            limits,
            poisoned: false,
            server_version: [0; 3],
        };
        let reply = client.request(Command::Hello, &[0; CAPABILITY_BYTES], Some(28))?;
        if reply[0] != VERSION[0] || reply[1] > VERSION[1] {
            client.poison();
            return Err(Error::Protocol(format!(
                "server version {}.{}.{} incompatible with 8.0.1",
                reply[0], reply[1], reply[2]
            )));
        }
        client.server_version.copy_from_slice(&reply[..3]);
        Ok(client)
    }
    fn poison(&mut self) {
        self.poisoned = true;
        let _ = self.stream.shutdown(Shutdown::Both);
    }
    fn send(&mut self, cmd: Command, body: &[u8]) -> Result<()> {
        if self.poisoned {
            return Err(Error::Protocol("connection is poisoned".into()));
        }
        let result = write_request(&mut self.stream, cmd, body, self.limits.max_frame_bytes);
        if result.is_err() {
            self.poison();
        }
        result
    }
    fn request(&mut self, cmd: Command, body: &[u8], length: Option<usize>) -> Result<Vec<u8>> {
        self.send(cmd, body)?;
        match read_response(&mut self.stream, self.limits.max_frame_bytes) {
            Ok(reply) if length.is_none_or(|n| reply.len() == n) => Ok(reply),
            Ok(reply) => {
                self.poison();
                Err(Error::Protocol(format!(
                    "response length {} != {:?}",
                    reply.len(),
                    length
                )))
            }
            Err(error) => {
                self.poison();
                Err(error)
            }
        }
    }
    pub fn device_count(&mut self) -> Result<u32> {
        let reply = self.request(Command::DeviceCount, &[], Some(4))?;
        Ok(u32::from_le_bytes(
            reply.try_into().expect("length checked"),
        ))
    }
    /// A responding command is an ordered barrier for prior no-response writes.
    pub fn barrier(&mut self) -> Result<()> {
        self.device_count()?;
        Ok(())
    }
    pub fn alignment(&mut self, device: u32) -> Result<u64> {
        let reply = self.request(Command::GetAlignment, &device.to_le_bytes(), Some(8))?;
        Ok(u64::from_le_bytes(
            reply.try_into().expect("length checked"),
        ))
    }
    pub fn max_size(&mut self, device: u32) -> Result<u64> {
        let reply = self.request(Command::GetMaxSize, &device.to_le_bytes(), Some(8))?;
        Ok(u64::from_le_bytes(
            reply.try_into().expect("length checked"),
        ))
    }
    pub fn device_memory(&mut self, device: u32) -> Result<(u64, u64)> {
        let reply = self.request(Command::GetDeviceMemory, &device.to_le_bytes(), Some(16))?;
        let mut r = Reader::new(&reply);
        Ok((r.u64()?, r.u64()?))
    }
    pub fn alloc(&mut self, device: u32, size: u64) -> Result<RemoteBuffer> {
        let mut body = device.to_le_bytes().to_vec();
        u64_to(&mut body, size);
        let reply = self.request(Command::AllocBuffer, &body, Some(16))?;
        let mut r = Reader::new(&reply);
        let handle = r.u64()?;
        let allocated = r.u64()?;
        if handle == 0 || allocated < size {
            return Err(Error::Bounds("remote allocation failed or too small"));
        }
        let reply = self.request(Command::BufferGetBase, &handle.to_le_bytes(), Some(8))?;
        let base = u64::from_le_bytes(reply.try_into().expect("length checked"));
        if base.checked_add(allocated).is_none() {
            self.poison();
            return Err(Error::Protocol(
                "remote allocation address overflows".into(),
            ));
        }
        Ok(RemoteBuffer {
            handle,
            base,
            size: allocated,
        })
    }
    pub fn free(&mut self, buffer: RemoteBuffer) -> Result<()> {
        self.send(Command::FreeBuffer, &buffer.handle.to_le_bytes())?;
        self.barrier()
    }
    pub fn clear(&mut self, buffer: RemoteBuffer, value: u8) -> Result<()> {
        let mut body = buffer.handle.to_le_bytes().to_vec();
        body.push(value);
        self.send(Command::BufferClear, &body)?;
        self.barrier()
    }
    pub fn init_tensor(&mut self, tensor: &RpcTensor) -> Result<()> {
        self.send(Command::InitTensor, &tensor.encode())?;
        self.barrier()
    }
    pub fn alloc_size(&mut self, device: u32, tensor: &RpcTensor) -> Result<u64> {
        let mut body = device.to_le_bytes().to_vec();
        body.extend(tensor.encode());
        body.resize(4 + 11 * TENSOR_BYTES, 0);
        let reply = self.request(Command::GetAllocSize, &body, Some(8))?;
        Ok(u64::from_le_bytes(
            reply.try_into().expect("length checked"),
        ))
    }
    pub fn set_tensor(&mut self, t: &RpcTensor, offset: u64, data: &[u8]) -> Result<()> {
        if offset > t.byte_len()? || data.len() as u64 > t.byte_len()? - offset {
            return Err(Error::Bounds("write exceeds declared tensor"));
        }
        let mut body = t.encode();
        body.push(0);
        u64_to(&mut body, offset);
        body.extend_from_slice(data);
        self.send(Command::SetTensor, &body)?;
        self.barrier()
    }
    pub fn get_tensor(&mut self, t: &RpcTensor, offset: u64, size: usize) -> Result<Vec<u8>> {
        if size > self.limits.max_frame_bytes {
            return Err(Error::Bounds("requested read exceeds frame limit"));
        }
        if offset > t.byte_len()? || size as u64 > t.byte_len()? - offset {
            return Err(Error::Bounds("read exceeds declared tensor"));
        }
        let mut body = t.encode();
        u64_to(&mut body, offset);
        u64_to(&mut body, size as u64);
        self.request(Command::GetTensor, &body, Some(size))
    }
    pub fn set_f32(&mut self, t: &RpcTensor, values: &[f32]) -> Result<()> {
        if !t.is_contiguous_f32() || values.len() != t.elements()? {
            return Err(Error::Bounds("F32 upload layout/length mismatch"));
        }
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.set_tensor(t, 0, &data)
    }
    pub fn get_f32(&mut self, t: &RpcTensor) -> Result<Vec<f32>> {
        if !t.is_contiguous_f32() {
            return Err(Error::Unsupported("F32 download layout"));
        }
        let size =
            usize::try_from(t.byte_len()?).map_err(|_| Error::Bounds("F32 download length"))?;
        Ok(self
            .get_tensor(t, 0, size)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    pub fn memset_tensor(
        &mut self,
        t: &RpcTensor,
        offset: u64,
        size: u64,
        value: u8,
    ) -> Result<()> {
        let mut body = t.encode();
        u64_to(&mut body, offset);
        u64_to(&mut body, size);
        body.push(value);
        self.send(Command::MemsetTensor, &body)?;
        self.barrier()
    }
    pub fn copy_tensor(&mut self, src: &RpcTensor, dst: &RpcTensor) -> Result<bool> {
        let mut body = src.encode();
        body.extend(dst.encode());
        let reply = self.request(Command::CopyTensor, &body, Some(1))?;
        match reply[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => {
                self.poison();
                Err(Error::Protocol("COPY_TENSOR returned invalid bool".into()))
            }
        }
    }
    pub fn set_tensor_2d(
        &mut self,
        t: &RpcTensor,
        offset: u64,
        width: u64,
        copies: u64,
        stride: u64,
        packed: &[u8],
    ) -> Result<()> {
        if width.checked_mul(copies) != Some(packed.len() as u64) {
            return Err(Error::Bounds("2D upload length mismatch"));
        }
        let mut body = t.encode();
        for n in [offset, width, copies, stride] {
            u64_to(&mut body, n);
        }
        body.extend_from_slice(packed);
        self.send(Command::SetTensor2d, &body)?;
        self.barrier()
    }
    pub fn get_tensor_2d(
        &mut self,
        t: &RpcTensor,
        offset: u64,
        width: u64,
        copies: u64,
        stride: u64,
    ) -> Result<Vec<u8>> {
        let size = width
            .checked_mul(copies)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::Bounds("2D read overflow"))?;
        if size > self.limits.max_frame_bytes {
            return Err(Error::Bounds("2D read exceeds frame limit"));
        }
        let mut body = t.encode();
        for n in [offset, width, copies, stride] {
            u64_to(&mut body, n);
        }
        self.request(Command::GetTensor2d, &body, Some(size))
    }
    pub fn graph_compute(&mut self, graph: &Graph) -> Result<()> {
        self.send(Command::GraphCompute, &graph.encode()?)?;
        self.barrier()
    }
    pub fn graph_recompute(&mut self, device: u32, uid: u64) -> Result<()> {
        let mut body = device.to_le_bytes().to_vec();
        u64_to(&mut body, uid);
        self.send(Command::GraphRecompute, &body)?;
        self.barrier()
    }
}
