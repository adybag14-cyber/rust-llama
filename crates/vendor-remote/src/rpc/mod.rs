//! Frozen RPC 8.0.1 TCP wire, explicit packed codecs, bounded server and client.
mod client;
mod codec;
mod server;
use crate::{Error, Result};
pub use client::{RemoteBuffer, RpcClient};
pub use codec::{
    CAPABILITY_BYTES, COMMAND_COUNT, Command, GRAPH_CACHE_MAX, Graph, RpcTensor, TENSOR_BYTES,
    VERSION, read_request, read_response, write_request, write_response,
};
pub use server::{Limits, ServerSession};

/// Deterministic probe usable against either the frozen native server or the
/// Rust server. It performs memory round trips, 2D writes, graph matmul/replay,
/// copy, memset and explicit frees using the existing protocol.
pub fn exercise_client(client: &mut RpcClient) -> Result<String> {
    if client.device_count()? == 0 {
        return Err(Error::Protocol("server has no devices".into()));
    }
    let align = client.alignment(0)?;
    if align == 0 || !align.is_power_of_two() {
        return Err(Error::Protocol("invalid remote alignment".into()));
    }
    let buffer = client.alloc(0, 1024)?;
    let a = buffer.tensor_f32(1001, 0, [3, 2, 1, 1], "weights")?;
    let b = buffer.tensor_f32(1002, 64, [3, 2, 1, 1], "inputs")?;
    let mut c = buffer.tensor_f32(1003, 128, [2, 2, 1, 1], "product")?;
    c.op = 29;
    c.flags = 18;
    c.src[0] = a.id;
    c.src[1] = b.id;
    let copy = buffer.tensor_f32(1004, 192, [2, 2, 1, 1], "copy")?;
    client.init_tensor(&a)?;
    client.init_tensor(&b)?;
    client.init_tensor(&c)?;
    if client.alloc_size(0, &a)? < 24 {
        return Err(Error::Protocol("invalid remote alloc size".into()));
    }
    client.set_f32(&a, &[1., 2., 3., 4., 5., 6.])?;
    client.set_f32(&b, &[7., 9., 11., 8., 10., 12.])?;
    let graph = Graph {
        device: 0,
        uid: 0x544f3831,
        nodes: vec![c.id],
        tensors: vec![a.clone(), b.clone(), c.clone()],
    };
    client.graph_compute(&graph)?;
    let actual = client.get_f32(&c)?;
    if actual != [58., 139., 64., 154.] {
        return Err(Error::Protocol(format!(
            "remote matmul mismatch: {actual:?}"
        )));
    }
    client.set_f32(&b, &[1., 0., 0., 0., 1., 0.])?;
    client.graph_recompute(0, graph.uid)?;
    if client.get_f32(&c)? != [1., 4., 2., 5.] {
        return Err(Error::Protocol("remote replay mismatch".into()));
    }
    if !client.copy_tensor(&c, &copy)? || client.get_f32(&copy)? != [1., 4., 2., 5.] {
        return Err(Error::Protocol("remote copy mismatch".into()));
    }
    client.memset_tensor(&copy, 0, copy.byte_len()?, 0)?;
    if client.get_f32(&copy)? != [0., 0., 0., 0.] {
        return Err(Error::Protocol("remote memset mismatch".into()));
    }
    let packed: Vec<u8> = [3.0f32, 7.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    client.set_tensor_2d(&copy, 0, 4, 2, 8, &packed)?;
    if client.get_tensor_2d(&copy, 0, 4, 2, 8)? != packed {
        return Err(Error::Protocol("remote 2D transfer mismatch".into()));
    }
    client.clear(buffer, 0)?;
    client.free(buffer)?;
    Ok(format!(
        "PASS: TCP RPC {}.{}.{}, alignment {align}, allocation/transfer/2D/copy/memset/F32 matmul/replay/free",
        client.server_version[0], client.server_version[1], client.server_version[2]
    ))
}

pub fn loopback_self_check() -> Result<String> {
    use std::net::TcpListener;
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    let thread = std::thread::spawn(move || -> Result<(usize, u64)> {
        let (mut stream, _) = listener.accept()?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
        let mut session = ServerSession::new(Limits::default())?;
        session.serve(&mut stream)?;
        Ok((session.allocated_bytes(), session.commands_processed))
    });
    let result = (|| {
        let mut client = RpcClient::connect(address, Limits::default())?;
        exercise_client(&mut client)
    })();
    let (remaining, commands) = thread
        .join()
        .map_err(|_| Error::Protocol("RPC server thread panicked".into()))??;
    if remaining != 0 {
        return Err(Error::Protocol(
            "RPC explicit frees left remote allocation".into(),
        ));
    }
    Ok(format!(
        "{}; {commands} commands; 0 bytes retained",
        result?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session() -> ServerSession {
        let mut s = ServerSession::new(Limits::default()).unwrap();
        s.handle(Command::Hello, &[0; 24]).unwrap();
        s
    }
    fn allocate(s: &mut ServerSession, n: u64) -> RemoteBuffer {
        let mut b = 0u32.to_le_bytes().to_vec();
        b.extend(n.to_le_bytes());
        let p = s.handle(Command::AllocBuffer, &b).unwrap().unwrap();
        let id = u64::from_le_bytes(p[0..8].try_into().unwrap());
        let size = u64::from_le_bytes(p[8..].try_into().unwrap());
        let p = s
            .handle(Command::BufferGetBase, &id.to_le_bytes())
            .unwrap()
            .unwrap();
        RemoteBuffer {
            handle: id,
            base: u64::from_le_bytes(p.try_into().unwrap()),
            size,
        }
    }
    fn set(s: &mut ServerSession, t: &RpcTensor, data: &[f32]) {
        let mut b = t.encode();
        b.push(0);
        b.extend(0u64.to_le_bytes());
        for v in data {
            b.extend(v.to_le_bytes());
        }
        s.handle(Command::SetTensor, &b).unwrap();
    }
    fn get(s: &mut ServerSession, t: &RpcTensor) -> Vec<f32> {
        let mut b = t.encode();
        b.extend(0u64.to_le_bytes());
        b.extend(t.byte_len().unwrap().to_le_bytes());
        s.handle(Command::GetTensor, &b)
            .unwrap()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect()
    }
    #[test]
    fn tcp_round_trip_and_real_scalar_graph() {
        assert!(loopback_self_check().unwrap().contains("0 bytes retained"));
    }
    #[test]
    fn handshake_required_and_no_repeat() {
        let mut s = ServerSession::new(Limits::default()).unwrap();
        assert!(s.handle(Command::DeviceCount, &[]).is_err());
        assert!(s.handle(Command::Hello, &[0; 23]).is_err());
        s.handle(Command::Hello, &[0; 24]).unwrap();
        assert!(s.handle(Command::Hello, &[0; 24]).is_err());
    }
    #[test]
    fn allocation_limit_and_device_validation() {
        let mut s = session();
        let mut b = 0u32.to_le_bytes().to_vec();
        b.extend(u64::MAX.to_le_bytes());
        assert_eq!(
            s.handle(Command::AllocBuffer, &b).unwrap().unwrap(),
            vec![0; 16]
        );
        assert_eq!(s.allocated_bytes(), 0);
        assert!(
            s.handle(Command::GetAlignment, &1u32.to_le_bytes())
                .is_err()
        );
    }
    #[test]
    fn buffers_are_session_local_and_double_free_fails() {
        let mut a = session();
        let b = allocate(&mut a, 128);
        assert_eq!(a.allocated_bytes(), 128);
        let mut other = session();
        assert!(
            other
                .handle(Command::BufferGetBase, &b.handle.to_le_bytes())
                .is_err()
        );
        a.handle(Command::FreeBuffer, &b.handle.to_le_bytes())
            .unwrap();
        assert_eq!(a.allocated_bytes(), 0);
        assert!(
            a.handle(Command::FreeBuffer, &b.handle.to_le_bytes())
                .is_err()
        );
    }
    #[test]
    fn malformed_transfer_does_not_write() {
        let mut s = session();
        let b = allocate(&mut s, 128);
        let t = b.tensor_f32(1, 0, [4, 1, 1, 1], "target").unwrap();
        set(&mut s, &t, &[1., 2., 3., 4.]);
        let mut body = t.encode();
        body.push(0);
        body.extend(15u64.to_le_bytes());
        body.extend([1, 2, 3, 4]);
        assert!(s.handle(Command::SetTensor, &body).is_err());
        assert_eq!(get(&mut s, &t), vec![1., 2., 3., 4.]);
    }
    #[test]
    fn out_of_allocation_and_overflow_are_rejected() {
        let mut s = session();
        let b = allocate(&mut s, 128);
        let mut t = b.tensor_f32(1, 0, [4, 1, 1, 1], "target").unwrap();
        t.data = b.base - 1;
        assert!(s.handle(Command::InitTensor, &t.encode()).is_err());
        t.data = u64::MAX;
        assert!(s.handle(Command::InitTensor, &t.encode()).is_err());
        let mut body = t.encode();
        for n in [0u64, 8, u64::MAX, 8] {
            body.extend(n.to_le_bytes());
        }
        assert!(s.handle(Command::GetTensor2d, &body).is_err());
    }
    #[test]
    fn graph_error_rolls_back_prior_computed_nodes() {
        let mut s = session();
        let b = allocate(&mut s, 256);
        let a = b.tensor_f32(1, 0, [4, 1, 1, 1], "a").unwrap();
        let mut c = b.tensor_f32(2, 64, [4, 1, 1, 1], "c").unwrap();
        let mut d = b.tensor_f32(3, 128, [4, 1, 1, 1], "d").unwrap();
        c.op = 9;
        c.flags = 16;
        c.src[0] = a.id;
        d.op = 100;
        d.flags = 16;
        d.src[0] = c.id;
        set(&mut s, &a, &[1., 2., 3., 4.]);
        set(&mut s, &c, &[9., 9., 9., 9.]);
        let g = Graph {
            device: 0,
            uid: 1,
            nodes: vec![2, 3],
            tensors: vec![a, c.clone(), d],
        };
        assert!(
            s.handle(Command::GraphCompute, &g.encode().unwrap())
                .is_err()
        );
        assert_eq!(get(&mut s, &c), vec![9.; 4]);
        assert_eq!(s.cached_graphs(), 0);
    }
    #[test]
    fn graph_rejects_cycles_missing_nodes_and_aliases() {
        let mut s = session();
        let b = allocate(&mut s, 128);
        let mut a = b.tensor_f32(1, 0, [4, 1, 1, 1], "a").unwrap();
        a.op = 9;
        a.flags = 16;
        a.src[0] = a.id;
        let g = Graph {
            device: 0,
            uid: 1,
            nodes: vec![1],
            tensors: vec![a],
        };
        assert!(
            s.handle(Command::GraphCompute, &g.encode().unwrap())
                .is_err()
        );
        let g = Graph {
            device: 0,
            uid: 1,
            nodes: vec![900],
            tensors: vec![],
        };
        assert!(
            s.handle(Command::GraphCompute, &g.encode().unwrap())
                .is_err()
        );
    }
    #[test]
    fn freeing_buffers_invalidates_graphs() {
        let mut s = session();
        let b = allocate(&mut s, 128);
        let a = b.tensor_f32(1, 0, [4, 1, 1, 1], "a").unwrap();
        let mut c = b.tensor_f32(2, 64, [4, 1, 1, 1], "c").unwrap();
        c.op = 9;
        c.flags = 16;
        c.src[0] = a.id;
        let g = Graph {
            device: 0,
            uid: 55,
            nodes: vec![2],
            tensors: vec![a, c],
        };
        s.handle(Command::GraphCompute, &g.encode().unwrap())
            .unwrap();
        assert_eq!(s.cached_graphs(), 1);
        s.handle(Command::FreeBuffer, &b.handle.to_le_bytes())
            .unwrap();
        assert_eq!(s.cached_graphs(), 0);
        let mut body = 0u32.to_le_bytes().to_vec();
        body.extend(55u64.to_le_bytes());
        assert!(s.handle(Command::GraphRecompute, &body).is_err());
    }
    #[test]
    fn graph_broadcast_and_affine_scale() {
        let mut s = session();
        let b = allocate(&mut s, 256);
        let a = b.tensor_f32(1, 0, [2, 2, 1, 1], "a").unwrap();
        let bias = b.tensor_f32(2, 64, [2, 1, 1, 1], "bias").unwrap();
        let mut c = b.tensor_f32(3, 128, [2, 2, 1, 1], "c").unwrap();
        c.op = 2;
        c.flags = 16;
        c.src[0] = 1;
        c.src[1] = 2;
        set(&mut s, &a, &[1., 2., 3., 4.]);
        set(&mut s, &bias, &[10., 20.]);
        let g = Graph {
            device: 0,
            uid: 0,
            nodes: vec![3],
            tensors: vec![a, bias, c.clone()],
        };
        s.handle(Command::GraphCompute, &g.encode().unwrap())
            .unwrap();
        assert_eq!(get(&mut s, &c), vec![11., 22., 13., 24.]);
    }
    #[test]
    fn unsupported_collectives_are_errors() {
        let mut s = session();
        for cmd in [
            Command::CommInit,
            Command::CommAllreduce,
            Command::CommFree,
            Command::None,
        ] {
            assert!(s.handle(cmd, &[]).is_err());
        }
    }
    #[test]
    fn compute_flag_is_required_like_frozen_ggml() {
        let mut s = session();
        let b = allocate(&mut s, 128);
        let a = b.tensor_f32(1, 0, [2, 1, 1, 1], "input").unwrap();
        let mut out = b.tensor_f32(2, 64, [2, 1, 1, 1], "output").unwrap();
        out.op = 9;
        out.src[0] = 1;
        set(&mut s, &a, &[2., 3.]);
        set(&mut s, &out, &[77., 88.]);
        let g = Graph {
            device: 0,
            uid: 0,
            nodes: vec![2],
            tensors: vec![a, out.clone()],
        };
        s.handle(Command::GraphCompute, &g.encode().unwrap())
            .unwrap();
        assert_eq!(get(&mut s, &out), vec![77., 88.]);
    }
    #[test]
    fn division_by_zero_uses_ieee_values() {
        let mut s = session();
        let b = allocate(&mut s, 256);
        let a = b.tensor_f32(1, 0, [2, 1, 1, 1], "a").unwrap();
        let z = b.tensor_f32(2, 64, [2, 1, 1, 1], "zero").unwrap();
        let mut out = b.tensor_f32(3, 128, [2, 1, 1, 1], "out").unwrap();
        out.op = 8;
        out.flags = 16;
        out.src[0] = 1;
        out.src[1] = 2;
        set(&mut s, &a, &[1., 0.]);
        set(&mut s, &z, &[0., 0.]);
        let g = Graph {
            device: 0,
            uid: 0,
            nodes: vec![3],
            tensors: vec![a, z, out.clone()],
        };
        s.handle(Command::GraphCompute, &g.encode().unwrap())
            .unwrap();
        let value = get(&mut s, &out);
        assert_eq!(value[0], f32::INFINITY);
        assert!(value[1].is_nan());
    }
    #[test]
    fn graph_aliases_cannot_multiply_workspace_without_bound() {
        let limits = Limits {
            max_buffer_bytes: 128,
            max_total_buffer_bytes: 128,
            ..Limits::default()
        };
        let mut s = ServerSession::new(limits).unwrap();
        s.handle(Command::Hello, &[0; 24]).unwrap();
        let b = allocate(&mut s, 128);
        let tensors = (1..=5)
            .map(|id| b.tensor_f32(id, 0, [8, 1, 1, 1], "alias").unwrap())
            .collect();
        let g = Graph {
            device: 0,
            uid: 0,
            nodes: vec![1, 2, 3, 4, 5],
            tensors,
        };
        let e = s
            .handle(Command::GraphCompute, &g.encode().unwrap())
            .unwrap_err();
        assert!(e.to_string().contains("workspace"));
    }
    #[test]
    fn disconnect_without_free_releases_session_memory() {
        use std::net::{Ipv4Addr, TcpListener};
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut s = ServerSession::new(Limits::default()).unwrap();
            s.serve(&mut stream).unwrap();
            (s.allocated_bytes(), s.allocation_count())
        });
        {
            let mut client = RpcClient::connect(addr, Limits::default()).unwrap();
            client.alloc(0, 4096).unwrap();
        }
        assert_eq!(server.join().unwrap(), (0, 0));
    }
    #[test]
    fn unknown_command_closes_session_and_frees_allocations() {
        use std::io::Write;
        use std::net::{Ipv4Addr, TcpListener, TcpStream};
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut s = ServerSession::new(Limits::default()).unwrap();
            let failed = s.serve(&mut stream).is_err();
            (failed, s.allocated_bytes(), s.allocation_count())
        });
        let mut stream = TcpStream::connect(addr).unwrap();
        write_request(&mut stream, Command::Hello, &[0; 24], 1024).unwrap();
        read_response(&mut stream, 1024).unwrap();
        let mut body = 0u32.to_le_bytes().to_vec();
        body.extend(32u64.to_le_bytes());
        write_request(&mut stream, Command::AllocBuffer, &body, 1024).unwrap();
        read_response(&mut stream, 1024).unwrap();
        stream.write_all(&[255]).unwrap();
        drop(stream);
        assert_eq!(server.join().unwrap(), (true, 0, 0));
    }
}
