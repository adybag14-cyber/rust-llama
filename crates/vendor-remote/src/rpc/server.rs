use super::codec::*;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_frame_bytes: usize,
    pub max_buffer_bytes: usize,
    pub max_total_buffer_bytes: usize,
    pub max_graph_nodes: usize,
    pub max_graph_tensors: usize,
    pub max_graph_cache_bytes: usize,
    pub max_graph_operations: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 8 * 1024 * 1024,
            max_buffer_bytes: 32 * 1024 * 1024,
            max_total_buffer_bytes: 64 * 1024 * 1024,
            max_graph_nodes: 4096,
            max_graph_tensors: 8192,
            max_graph_cache_bytes: 16 * 1024 * 1024,
            max_graph_operations: 64 * 1024 * 1024,
        }
    }
}
impl Limits {
    pub fn validate(&self) -> Result<()> {
        if self.max_frame_bytes < 28
            || self.max_buffer_bytes == 0
            || self.max_total_buffer_bytes < self.max_buffer_bytes
            || self.max_graph_nodes == 0
            || self.max_graph_tensors < self.max_graph_nodes
            || self.max_graph_cache_bytes < 20
            || self.max_graph_operations == 0
        {
            return Err(Error::Bounds("invalid RPC limits"));
        }
        Ok(())
    }
}
#[derive(Debug)]
struct Allocation {
    base: u64,
    bytes: Vec<u8>,
}

/// One session owns all remote allocations and cached graphs. Opaque handles
/// and virtual base addresses never expose or dereference native Rust pointers.
pub struct ServerSession {
    limits: Limits,
    hello: bool,
    closed: bool,
    allocations: HashMap<u64, Allocation>,
    allocated: usize,
    next_id: u64,
    next_base: u64,
    graphs: HashMap<u64, Graph>,
    graph_bytes: usize,
    pub commands_processed: u64,
}
impl ServerSession {
    pub fn new(limits: Limits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            limits,
            hello: false,
            closed: false,
            allocations: HashMap::new(),
            allocated: 0,
            next_id: 1,
            next_base: 0x1_0000_0000,
            graphs: HashMap::new(),
            graph_bytes: 0,
            commands_processed: 0,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        self.allocated
    }
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }
    pub fn cached_graphs(&self) -> usize {
        self.graphs.len()
    }
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
    /// Synchronous execution makes a responding command a barrier for preceding
    /// no-response commands. A protocol failure closes the stream in the caller.
    pub fn serve(&mut self, io: &mut (impl Read + Write)) -> Result<()> {
        let result = (|| {
            while let Some((cmd, body)) = read_request(io, self.limits.max_frame_bytes)? {
                if let Some(reply) = self.handle(cmd, &body)? {
                    write_response(io, &reply, self.limits.max_frame_bytes)?;
                }
            }
            Ok(())
        })();
        // Session-owned remote resources never survive disconnect, including a
        // malformed command, truncated frame, timeout or failed response write.
        self.allocations.clear();
        self.allocated = 0;
        self.graphs.clear();
        self.graph_bytes = 0;
        self.closed = true;
        result
    }
    fn device(&self, id: u32) -> Result<()> {
        if id == 0 {
            Ok(())
        } else {
            Err(Error::Protocol(format!("unknown device {id}")))
        }
    }
    fn allocation(&self, id: u64) -> Result<&Allocation> {
        self.allocations
            .get(&id)
            .ok_or_else(|| Error::Protocol("unknown/freed buffer handle".into()))
    }
    fn region(
        &self,
        t: &RpcTensor,
        offset: u64,
        size: u64,
    ) -> Result<(u64, std::ops::Range<usize>)> {
        let a = self.allocation(t.buffer)?;
        let tensor_bytes = t.byte_len()?;
        if offset > tensor_bytes || size > tensor_bytes - offset {
            return Err(Error::Bounds("tensor transfer exceeds declared tensor"));
        }
        let origin = t
            .data
            .checked_sub(a.base)
            .ok_or(Error::Bounds("tensor data before allocation"))?;
        let allocation_size = a.bytes.len() as u64;
        if origin > allocation_size || tensor_bytes > allocation_size - origin {
            return Err(Error::Bounds("declared tensor exceeds allocation"));
        }
        let start = origin
            .checked_add(offset)
            .ok_or(Error::Bounds("tensor transfer offset overflow"))?;
        let end = start
            .checked_add(size)
            .ok_or(Error::Bounds("tensor transfer end overflow"))?;
        if end > allocation_size {
            return Err(Error::Bounds("tensor transfer exceeds allocation"));
        }
        Ok((t.buffer, start as usize..end as usize))
    }
    fn validate_tensor(&self, t: &RpcTensor) -> Result<()> {
        self.region(t, 0, t.byte_len()?)?;
        Ok(())
    }
    fn two_dimensional_region(
        &self,
        t: &RpcTensor,
        offset: u64,
        size: u64,
        count: u64,
        stride: u64,
    ) -> Result<(u64, usize, usize, usize, usize)> {
        if count == 0 || size == 0 {
            return Err(Error::Bounds("2D copy size/count must be positive"));
        }
        let span = (count - 1)
            .checked_mul(stride)
            .and_then(|s| s.checked_add(size))
            .ok_or(Error::Bounds("2D transfer span overflow"))?;
        let packed = size
            .checked_mul(count)
            .ok_or(Error::Bounds("2D transfer packed size overflow"))?;
        if packed > self.limits.max_frame_bytes as u64 {
            return Err(Error::Bounds("2D transfer exceeds frame limit"));
        }
        let (id, r) = self.region(t, offset, span)?;
        let step =
            usize::try_from(stride).map_err(|_| Error::Bounds("2D stride does not fit usize"))?;
        Ok((id, r.start, size as usize, count as usize, step))
    }
    /// Exact v8.0.1 command payloads for the implemented subset. The wire lacks
    /// an error reply; an error here must terminate the connection, as upstream.
    pub fn handle(&mut self, cmd: Command, body: &[u8]) -> Result<Option<Vec<u8>>> {
        if self.closed {
            return Err(Error::Protocol("session has closed".into()));
        }
        if body.len() > self.limits.max_frame_bytes {
            return Err(Error::Bounds("request exceeds frame limit"));
        }
        if !self.hello {
            if cmd != Command::Hello || body.len() != CAPABILITY_BYTES {
                return Err(Error::Protocol(
                    "first request must be HELLO with 24 capability bytes".into(),
                ));
            }
            self.hello = true;
            self.commands_processed += 1;
            let mut out = vec![8, 0, 1, 0];
            out.extend_from_slice(&[0; CAPABILITY_BYTES]);
            return Ok(Some(out));
        }
        let mut r = Reader::new(body);
        let mut out = Vec::new();
        let response = match cmd {
            Command::Hello => {
                return Err(Error::Protocol("HELLO cannot repeat in a session".into()));
            }
            Command::DeviceCount => {
                r.end()?;
                u32_to(&mut out, 1);
                true
            }
            Command::GetAlignment | Command::GetMaxSize | Command::GetDeviceMemory => {
                self.device(r.u32()?)?;
                r.end()?;
                match cmd {
                    Command::GetAlignment => u64_to(&mut out, 64),
                    Command::GetMaxSize => u64_to(&mut out, self.limits.max_buffer_bytes as u64),
                    _ => {
                        u64_to(
                            &mut out,
                            (self.limits.max_total_buffer_bytes - self.allocated) as u64,
                        );
                        u64_to(&mut out, self.limits.max_total_buffer_bytes as u64);
                    }
                }
                true
            }
            Command::AllocBuffer => {
                self.device(r.u32()?)?;
                let size = r.u64()?;
                r.end()?;
                let requested = usize::try_from(size).ok();
                let within = requested.filter(|n| {
                    *n > 0
                        && *n <= self.limits.max_buffer_bytes
                        && *n <= self.limits.max_total_buffer_bytes - self.allocated
                });
                if let Some(size) = within {
                    let mut bytes = Vec::new();
                    if bytes.try_reserve_exact(size).is_ok() {
                        bytes.resize_with(size, || 0);
                        let id = self.next_id;
                        let base = self.next_base;
                        self.next_id = self
                            .next_id
                            .checked_add(1)
                            .ok_or(Error::Bounds("remote handle counter exhausted"))?;
                        self.next_base = self
                            .next_base
                            .checked_add(
                                (size as u64)
                                    .checked_add(127)
                                    .ok_or(Error::Bounds("remote address overflow"))?
                                    & !63,
                            )
                            .ok_or(Error::Bounds("remote address counter exhausted"))?;
                        self.allocations.insert(id, Allocation { base, bytes });
                        self.allocated += size;
                        u64_to(&mut out, id);
                        u64_to(&mut out, size as u64);
                    } else {
                        out.resize(16, 0);
                    }
                } else {
                    out.resize(16, 0);
                }
                true
            }
            Command::BufferGetBase => {
                let id = r.u64()?;
                r.end()?;
                u64_to(&mut out, self.allocation(id)?.base);
                true
            }
            Command::FreeBuffer => {
                let id = r.u64()?;
                r.end()?;
                let a = self
                    .allocations
                    .remove(&id)
                    .ok_or_else(|| Error::Protocol("free of unknown/freed buffer".into()))?;
                self.allocated -= a.bytes.len();
                self.graphs.clear();
                self.graph_bytes = 0;
                false
            }
            Command::BufferClear => {
                let id = r.u64()?;
                let value = r.u8()?;
                r.end()?;
                self.allocations
                    .get_mut(&id)
                    .ok_or_else(|| Error::Protocol("clear of unknown buffer".into()))?
                    .bytes
                    .fill(value);
                false
            }
            Command::InitTensor => {
                let t = r.tensor()?;
                r.end()?;
                self.validate_tensor(&t)?;
                false
            }
            Command::GetAllocSize => {
                self.device(r.u32()?)?;
                let t = r.tensor()?;
                for _ in 0..10 {
                    let bytes = r.take(TENSOR_BYTES)?;
                    let id = u64::from_le_bytes(bytes[..8].try_into().expect("fixed prefix"));
                    if id != 0 {
                        RpcTensor::decode(bytes)?;
                    }
                }
                r.end()?;
                u64_to(&mut out, t.byte_len()?);
                true
            }
            Command::SetTensor => {
                let t = r.tensor()?;
                let flag = r.u8()?;
                if flag > 1 {
                    return Err(Error::Protocol("invalid tensor-cache flag".into()));
                }
                let offset = r.u64()?;
                let data = r.rest();
                let (id, range) = self.region(&t, offset, data.len() as u64)?;
                self.allocations
                    .get_mut(&id)
                    .expect("validated handle")
                    .bytes[range]
                    .copy_from_slice(data);
                false
            }
            Command::GetTensor => {
                let t = r.tensor()?;
                let offset = r.u64()?;
                let size = r.u64()?;
                r.end()?;
                if size > self.limits.max_frame_bytes as u64 {
                    return Err(Error::Bounds("tensor read exceeds frame limit"));
                }
                let (id, range) = self.region(&t, offset, size)?;
                out.extend_from_slice(&self.allocation(id)?.bytes[range]);
                true
            }
            Command::SetTensorHash => {
                let t = r.tensor()?;
                let offset = r.u64()?;
                let _hash = r.u64()?;
                r.end()?;
                self.region(&t, offset, 0)?;
                // No persistent hash cache is configured in this checkpoint,
                // matching the reference's cache_dir == nullptr miss behavior.
                out.push(0);
                true
            }
            Command::MemsetTensor => {
                let t = r.tensor()?;
                let offset = r.u64()?;
                let size = r.u64()?;
                let value = r.u8()?;
                r.end()?;
                let (id, range) = self.region(&t, offset, size)?;
                self.allocations
                    .get_mut(&id)
                    .expect("validated handle")
                    .bytes[range]
                    .fill(value);
                false
            }
            Command::SetTensor2d | Command::GetTensor2d => {
                let t = r.tensor()?;
                let offset = r.u64()?;
                let size = r.u64()?;
                let count = r.u64()?;
                let stride = r.u64()?;
                let (id, start, width, count, stride) =
                    self.two_dimensional_region(&t, offset, size, count, stride)?;
                if cmd == Command::SetTensor2d {
                    let payload = r.rest();
                    if payload.len() != width * count {
                        return Err(Error::Protocol("2D tensor data size mismatch".into()));
                    }
                    let a = self.allocations.get_mut(&id).expect("validated handle");
                    for i in 0..count {
                        a.bytes[start + i * stride..start + i * stride + width]
                            .copy_from_slice(&payload[i * width..(i + 1) * width]);
                    }
                    false
                } else {
                    r.end()?;
                    let a = self.allocation(id)?;
                    out.reserve(width * count);
                    for i in 0..count {
                        out.extend_from_slice(
                            &a.bytes[start + i * stride..start + i * stride + width],
                        );
                    }
                    true
                }
            }
            Command::CopyTensor => {
                let src = r.tensor()?;
                let dst = r.tensor()?;
                r.end()?;
                if src.dtype != dst.dtype || src.ne != dst.ne || src.nb != dst.nb {
                    out.push(0);
                } else {
                    let bytes = src.byte_len()?;
                    let (a, ar) = self.region(&src, 0, bytes)?;
                    let (b, br) = self.region(&dst, 0, bytes)?;
                    if a == b {
                        self.allocations
                            .get_mut(&a)
                            .expect("validated handle")
                            .bytes
                            .copy_within(ar, br.start);
                    } else {
                        let data = self.allocation(a)?.bytes[ar].to_vec();
                        self.allocations
                            .get_mut(&b)
                            .expect("validated handle")
                            .bytes[br]
                            .copy_from_slice(&data);
                    }
                    out.push(1);
                }
                true
            }
            Command::GraphCompute => {
                let graph = Graph::decode(
                    body,
                    self.limits.max_graph_nodes,
                    self.limits.max_graph_tensors,
                )?;
                self.device(graph.device)?;
                let reset = graph.uid != 0 && self.graphs.len() >= GRAPH_CACHE_MAX;
                let old_bytes = if reset { 0 } else { self.graph_bytes };
                let replaced = if reset {
                    0
                } else {
                    self.graphs
                        .get(&graph.uid)
                        .map(|g| 20 + 8 * g.nodes.len() + TENSOR_BYTES * g.tensors.len())
                        .unwrap_or(0)
                };
                if graph.uid != 0
                    && old_bytes - replaced + body.len() > self.limits.max_graph_cache_bytes
                {
                    return Err(Error::Bounds("graph cache memory limit"));
                }
                self.execute(&graph)?;
                if graph.uid != 0 {
                    if reset {
                        self.graphs.clear();
                        self.graph_bytes = 0;
                    }
                    self.graph_bytes = self.graph_bytes - replaced + body.len();
                    self.graphs.insert(graph.uid, graph);
                }
                false
            }
            Command::GraphRecompute => {
                self.device(r.u32()?)?;
                let uid = r.u64()?;
                r.end()?;
                let graph = self
                    .graphs
                    .get(&uid)
                    .ok_or_else(|| Error::Protocol("unknown/invalidated graph UID".into()))?
                    .clone();
                self.execute(&graph)?;
                false
            }
            Command::CommInit | Command::CommAllreduce | Command::CommFree => {
                return Err(Error::Unsupported("RPC collectives/RDMA"));
            }
            Command::None => {
                return Err(Error::Protocol(
                    "NONE is client dispatcher sentinel, not a server request".into(),
                ));
            }
        };
        self.commands_processed += 1;
        Ok(if response { Some(out) } else { None })
    }

    fn read_f32(&self, t: &RpcTensor) -> Result<Vec<f32>> {
        if !t.is_contiguous_f32() {
            return Err(Error::Unsupported(
                "RPC scalar executor requires contiguous F32 tensors",
            ));
        }
        let (id, range) = self.region(t, 0, t.byte_len()?)?;
        Ok(self.allocation(id)?.bytes[range]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    fn execute(&mut self, g: &Graph) -> Result<()> {
        let mut tensors = HashMap::new();
        let mut workspace_bytes = 0u64;
        for t in &g.tensors {
            if t.id == 0 || tensors.insert(t.id, t).is_some() {
                return Err(Error::Protocol("zero/duplicate graph tensor ID".into()));
            }
            self.validate_tensor(t)?;
            if t.ne.contains(&0) {
                return Err(Error::Bounds("graph tensors must have positive dimensions"));
            }
            workspace_bytes = workspace_bytes
                .checked_add(t.byte_len()?)
                .ok_or(Error::Bounds("graph workspace overflow"))?;
            if workspace_bytes > self.limits.max_total_buffer_bytes as u64 {
                return Err(Error::Bounds("graph scalar workspace limit"));
            }
            if !t.is_contiguous_f32() {
                return Err(Error::Unsupported("RPC graph supports contiguous F32 only"));
            }
            if t.view_src != 0 || t.view_offs != 0 {
                return Err(Error::Unsupported("RPC graph view aliases"));
            }
        }
        let mut seen = HashSet::new();
        let mut node_ids = HashSet::new();
        let mut values: HashMap<u64, Vec<f32>> = HashMap::new();
        let mut staged = Vec::new();
        let mut work = 0u64;
        // Computed outputs cannot overlap another tensor's region until T02
        // alias scheduling is integrated. Transfer commands can handle overlap.
        for id in &g.nodes {
            if !node_ids.insert(*id) {
                return Err(Error::Protocol("duplicate graph execution node".into()));
            }
            let t = *tensors
                .get(id)
                .ok_or_else(|| Error::Protocol("missing graph node descriptor".into()))?;
            if t.op != 0 && t.flags & 16 != 0 {
                let (_, a) = self.region(t, 0, t.byte_len()?)?;
                for other in &g.tensors {
                    if other.id != t.id && other.buffer == t.buffer {
                        let (_, b) = self.region(other, 0, other.byte_len()?)?;
                        if a.start < b.end && b.start < a.end {
                            return Err(Error::Unsupported("RPC graph overlapping tensor storage"));
                        }
                    }
                }
            }
        }
        for id in &g.nodes {
            let t = *tensors.get(id).expect("validated node");
            for source in t.src.iter().filter(|&&id| id != 0) {
                let st = *tensors
                    .get(source)
                    .ok_or_else(|| Error::Protocol("missing graph source descriptor".into()))?;
                if st.op != 0 && st.flags & 16 != 0 && !seen.contains(source) {
                    return Err(Error::Protocol(
                        "graph sources must precede consumer; cycle or missing execution node"
                            .into(),
                    ));
                }
                if !values.contains_key(source) {
                    values.insert(*source, self.read_f32(st)?);
                }
            }
            let result = if t.op == 0 || t.flags & 16 == 0 {
                self.read_f32(t)?
            } else {
                let src0 = *tensors
                    .get(&t.src[0])
                    .ok_or_else(|| Error::Protocol("operation requires src0".into()))?;
                let a = values
                    .get(&t.src[0])
                    .ok_or_else(|| Error::Protocol("missing src0 values".into()))?;
                let elements = t.elements()?;
                work = work
                    .checked_add(elements as u64)
                    .ok_or(Error::Bounds("graph work overflow"))?;
                if work > self.limits.max_graph_operations {
                    return Err(Error::Bounds("graph work limit"));
                }
                match t.op {
                    1 | 34 | 35 => {
                        if a.len() != elements {
                            return Err(Error::Bounds("copy graph shape mismatch"));
                        }
                        a.clone()
                    }
                    2 | 4 | 6 | 7 | 8 => {
                        let st = *tensors.get(&t.src[1]).ok_or_else(|| {
                            Error::Protocol("binary operation requires src1".into())
                        })?;
                        let b = values
                            .get(&t.src[1])
                            .ok_or_else(|| Error::Protocol("missing src1 values".into()))?;
                        if src0.ne != t.ne
                            || st.ne.contains(&0)
                            || (t.op == 4 && b.len() != 1)
                            || (0..4).any(|d| !t.ne[d].is_multiple_of(st.ne[d]))
                        {
                            return Err(Error::Bounds("binary broadcast shape mismatch"));
                        }
                        (0..elements)
                            .map(|i| {
                                let mut index = i;
                                let mut bi = 0usize;
                                let mut stride = 1usize;
                                for d in 0..4 {
                                    let coord = index % (t.ne[d] as usize);
                                    index /= t.ne[d] as usize;
                                    bi += coord % (st.ne[d] as usize) * stride;
                                    stride *= st.ne[d] as usize;
                                }
                                match t.op {
                                    2 | 4 => a[i] + b[bi],
                                    6 => a[i] - b[bi],
                                    7 => a[i] * b[bi],
                                    8 => a[i] / b[bi],
                                    _ => unreachable!(),
                                }
                            })
                            .collect()
                    }
                    9 | 10 | 11 | 12 | 13 | 32 => {
                        if src0.ne != t.ne {
                            return Err(Error::Bounds("unary shape mismatch"));
                        }
                        if t.op == 32 && t.op_params[2..].iter().any(|&p| p != 0) {
                            return Err(Error::Unsupported("SCALE extra parameters"));
                        }
                        let scale = f32::from_bits(t.op_params[0] as u32);
                        let bias = f32::from_bits(t.op_params[1] as u32);
                        a.iter()
                            .map(|x| match t.op {
                                9 => x * x,
                                10 => x.sqrt(),
                                11 => x.ln(),
                                12 => x.sin(),
                                13 => x.cos(),
                                32 => x * scale + bias,
                                _ => unreachable!(),
                            })
                            .collect()
                    }
                    29 => {
                        let st = *tensors
                            .get(&t.src[1])
                            .ok_or_else(|| Error::Protocol("MUL_MAT requires src1".into()))?;
                        let b = values
                            .get(&t.src[1])
                            .ok_or_else(|| Error::Protocol("missing matmul values".into()))?;
                        if src0.ne[0] != st.ne[0]
                            || src0.ne.contains(&0)
                            || st.ne.contains(&0)
                            || !st.ne[2].is_multiple_of(src0.ne[2])
                            || !st.ne[3].is_multiple_of(src0.ne[3])
                            || t.ne != [src0.ne[1], st.ne[1], st.ne[2], st.ne[3]]
                        {
                            return Err(Error::Bounds("MUL_MAT dimensions/broadcast mismatch"));
                        }
                        if t.op_params.iter().any(|&v| v != 0) {
                            return Err(Error::Unsupported("MUL_MAT precision/hint parameters"));
                        }
                        let k = src0.ne[0] as usize;
                        let m = src0.ne[1] as usize;
                        let n = st.ne[1] as usize;
                        work = work
                            .checked_add(
                                (elements as u64)
                                    .checked_mul(k as u64)
                                    .ok_or(Error::Bounds("matmul work overflow"))?,
                            )
                            .ok_or(Error::Bounds("graph work overflow"))?;
                        if work > self.limits.max_graph_operations {
                            return Err(Error::Bounds("graph work limit"));
                        }
                        let mut out = vec![0.; elements];
                        for i3 in 0..st.ne[3] as usize {
                            for i2 in 0..st.ne[2] as usize {
                                let a3 = i3 / (st.ne[3] / src0.ne[3]) as usize;
                                let a2 = i2 / (st.ne[2] / src0.ne[2]) as usize;
                                let aoff = (a3 * src0.ne[2] as usize + a2) * m * k;
                                let boff = (i3 * st.ne[2] as usize + i2) * n * k;
                                let coff = (i3 * st.ne[2] as usize + i2) * n * m;
                                for j in 0..n {
                                    for i in 0..m {
                                        let mut sum = 0.;
                                        for p in 0..k {
                                            sum += a[aoff + i * k + p] * b[boff + j * k + p];
                                        }
                                        out[coff + j * m + i] = sum;
                                    }
                                }
                            }
                        }
                        out
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "RPC graph operation not implemented in scalar checkpoint",
                        ));
                    }
                }
            };
            seen.insert(*id);
            if t.op != 0 && t.flags & 16 != 0 {
                staged.push((t, result.clone()));
            }
            values.insert(*id, result);
        }
        // Commit only after every node validates and computes successfully.
        for (t, value) in staged {
            let (id, range) = self.region(t, 0, t.byte_len()?)?;
            let out = &mut self
                .allocations
                .get_mut(&id)
                .expect("validated handle")
                .bytes[range];
            for (b, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(value) {
                b.copy_from_slice(&v.to_le_bytes());
            }
        }
        Ok(())
    }
}
