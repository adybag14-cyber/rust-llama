use super::{Cell, NativeContext, SequenceSet, invalid, zeroes};
use crate::{Result, llama::KvLayer};

/// Raw frozen memory serialization, before sequence-I/O or session headers.
/// For ON_DEVICE, tensor bytes are held separately in the owning CPU context.
#[derive(Debug, Clone)]
pub struct NativeState {
    pub bytes: Vec<u8>,
    pub tensors: Vec<u8>,
    pub write_boundaries: Vec<usize>,
}
#[derive(Default, Clone, Copy)]
struct Shape {
    cells: usize,
    ranges: usize,
    ids: usize,
}
fn shape(stream: &super::Stream, wanted: Option<i32>) -> Shape {
    let mut result = Shape::default();
    let mut prior = None;
    for (index, cell) in stream.cells.iter().enumerate() {
        if !cell.occupied() || wanted.is_some_and(|id| !cell.sequences.contains(&id)) {
            continue;
        }
        result.cells += 1;
        result.ids += if wanted.is_some() {
            1
        } else {
            cell.sequences.len()
        };
        if prior != Some(index) {
            result.ranges += 1;
        }
        prior = Some(index + 1);
    }
    result
}
#[derive(Default, Clone, Copy)]
struct Size {
    bytes: usize,
    tensors: usize,
    boundaries: usize,
    scratch: usize,
}
fn add(target: &mut usize, a: usize, b: usize) -> Result<()> {
    *target = a
        .checked_mul(b)
        .and_then(|n| target.checked_add(n))
        .ok_or_else(|| invalid("native serialization size overflow"))?;
    Ok(())
}
fn estimate(context: &NativeContext, sequence: Option<i32>, flags: u32) -> Result<Size> {
    let mut result = Size::default();
    if sequence.is_some() && flags & 1 != 0 {
        return Ok(result);
    }
    if let Some(id) = sequence.filter(|id| *id != -1) {
        context.stream_id(id)?;
    }
    let wanted = sequence.filter(|id| *id != -1);
    if sequence.is_none() {
        result.bytes = 9;
        result.boundaries = 2;
    }
    add(&mut result.bytes, 1, 4)?;
    add(&mut result.boundaries, 1, 1)?;
    for stream in &context.streams {
        add(&mut result.bytes, 1, 4)?;
        add(&mut result.boundaries, 1, 1)?;
        let s = shape(stream, wanted);
        if s.cells == 0 {
            continue;
        }
        add(&mut result.bytes, s.cells, 8)?;
        add(&mut result.bytes, s.ids, 4)?;
        add(&mut result.bytes, 1, 16)?;
        add(&mut result.bytes, stream.layers.len(), 24)?;
        add(&mut result.boundaries, s.cells, 2)?;
        add(&mut result.boundaries, s.ids, 1)?;
        add(&mut result.boundaries, 1, 4)?;
        add(&mut result.boundaries, stream.layers.len(), 5)?;
        let tensor = s
            .cells
            .checked_mul(context.width)
            .and_then(|n| n.checked_mul(stream.layers.len()))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("native tensor serialization size overflow"))?;
        if flags & 2 != 0 {
            add(&mut result.tensors, tensor, 1)?;
        } else {
            add(&mut result.bytes, tensor, 1)?;
            let chunks = s
                .ranges
                .checked_mul(context.width + 1)
                .ok_or_else(|| invalid("native tensor chunk count overflow"))?;
            add(&mut result.boundaries, chunks, stream.layers.len())?;
        }
        let scratch = s
            .cells
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|n| n.checked_add(s.ranges * std::mem::size_of::<std::ops::Range<usize>>()))
            .ok_or_else(|| invalid("native serializer scratch size overflow"))?;
        result.scratch = result.scratch.max(scratch);
    }
    Ok(result)
}
struct Writer {
    state: NativeState,
    device: bool,
}
impl Writer {
    fn new(size: Size, device: bool) -> Result<Self> {
        let mut state = NativeState {
            bytes: Vec::new(),
            tensors: Vec::new(),
            write_boundaries: Vec::new(),
        };
        state
            .bytes
            .try_reserve_exact(
                size.bytes
                    .checked_add(8)
                    .ok_or_else(|| invalid("native byte reserve overflow"))?,
            )
            .map_err(|_| invalid("native serialization allocation failed"))?;
        state
            .tensors
            .try_reserve_exact(size.tensors)
            .map_err(|_| invalid("native tensor snapshot allocation failed"))?;
        state
            .write_boundaries
            .try_reserve_exact(
                size.boundaries
                    .checked_add(2)
                    .ok_or_else(|| invalid("native boundary reserve overflow"))?,
            )
            .map_err(|_| invalid("native write boundary allocation failed"))?;
        Ok(Self { state, device })
    }
    fn raw(&mut self, value: &[u8]) {
        self.state.bytes.extend_from_slice(value);
        self.state.write_boundaries.push(self.state.bytes.len());
    }
    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }
    fn i32(&mut self, value: i32) {
        self.raw(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }
    fn tensor(&mut self, values: impl Iterator<Item = f32>) {
        if self.device {
            for value in values {
                self.state.tensors.extend_from_slice(&value.to_le_bytes());
            }
        } else {
            for value in values {
                self.state.bytes.extend_from_slice(&value.to_le_bytes());
            }
            self.state.write_boundaries.push(self.state.bytes.len());
        }
    }
}
pub(super) fn encoded_size(
    context: &NativeContext,
    sequence: Option<i32>,
    flags: u32,
) -> Result<usize> {
    Ok(estimate(context, sequence, flags)?.bytes)
}
pub(super) fn encode(
    context: &NativeContext,
    sequence: Option<i32>,
    flags: u32,
) -> Result<NativeState> {
    let size = estimate(context, sequence, flags)?;
    let work = size
        .bytes
        .checked_add(size.tensors)
        .and_then(|n| n.checked_add(size.scratch))
        .and_then(|n| n.checked_add(size.boundaries.checked_mul(std::mem::size_of::<usize>())?))
        .and_then(|n| n.checked_add(32))
        .ok_or_else(|| invalid("native serialization work overflow"))?;
    context.check_work(work)?;
    let mut out = Writer::new(size, flags & 2 != 0)?;
    if sequence.is_some() && flags & 1 != 0 {
        return Ok(out.state);
    }
    let wanted = sequence.filter(|id| *id != -1);
    if sequence.is_none() {
        out.u32(5);
        out.raw(b"llama");
    }
    out.u32(context.streams.len() as u32);
    for stream in &context.streams {
        let expected = shape(stream, wanted);
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(expected.cells)
            .map_err(|_| invalid("native selected cell allocation failed"))?;
        selected.extend(
            stream
                .cells
                .iter()
                .enumerate()
                .filter(|(_, cell)| {
                    cell.occupied() && wanted.is_none_or(|id| cell.sequences.contains(&id))
                })
                .map(|(i, _)| i),
        );
        out.u32(selected.len() as u32);
        if selected.is_empty() {
            continue;
        }
        let mut ranges = Vec::<std::ops::Range<usize>>::new();
        ranges
            .try_reserve_exact(expected.ranges)
            .map_err(|_| invalid("native cell range allocation failed"))?;
        for &index in &selected {
            if let Some(last) = ranges.last_mut().filter(|r| r.end == index) {
                last.end += 1;
            } else {
                ranges.push(index..index + 1);
            }
        }
        for &index in &selected {
            let cell = &stream.cells[index];
            out.i32(cell.position);
            out.u32(if wanted.is_some() {
                1
            } else {
                cell.sequences.len()
            } as u32);
            for id in cell
                .sequences
                .iter()
                .copied()
                .filter(|id| wanted.is_none_or(|w| *id == w))
            {
                out.i32(id);
            }
        }
        out.u32(1);
        out.u32(stream.layers.len() as u32);
        out.u32(0);
        out.u32(0);
        for layer in &stream.layers {
            out.i32(0);
            out.u64((context.width * 4) as u64);
            for range in &ranges {
                out.tensor(
                    layer.keys[range.start * context.width..range.end * context.width]
                        .iter()
                        .copied(),
                );
            }
        }
        for layer in &stream.layers {
            out.i32(0);
            out.u32(4);
            out.u32(context.width as u32);
            for dimension in 0..context.width {
                for range in &ranges {
                    out.tensor(
                        range
                            .clone()
                            .map(|index| layer.values[index * context.width + dimension]),
                    );
                }
            }
        }
    }
    debug_assert_eq!(out.state.bytes.len(), size.bytes);
    debug_assert_eq!(out.state.tensors.len(), size.tensors);
    debug_assert_eq!(out.state.write_boundaries.len(), size.boundaries);
    Ok(out.state)
}
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    tensors: Option<&'a [u8]>,
    tensor_at: usize,
}
impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| invalid("truncated native state"))?;
        let result = &self.bytes[self.at..end];
        self.at = end;
        Ok(result)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("four bytes"),
        ))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.bytes(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn floats(&mut self, count: usize) -> Result<Vec<f32>> {
        let n = count
            .checked_mul(4)
            .ok_or_else(|| invalid("native tensor byte overflow"))?;
        let raw = if let Some(tensors) = self.tensors {
            let end = self
                .tensor_at
                .checked_add(n)
                .filter(|end| *end <= tensors.len())
                .ok_or_else(|| invalid("native device snapshot tensor data is truncated"))?;
            let raw = &tensors[self.tensor_at..end];
            self.tensor_at = end;
            raw
        } else {
            self.bytes(n)?
        };
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| invalid("native restored tensor allocation failed"))?;
        values.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|raw| f32::from_le_bytes(*raw)),
        );
        Ok(values)
    }
}
pub(super) fn decode(
    context: &mut NativeContext,
    bytes: &[u8],
    sequence: Option<i32>,
    flags: u32,
    tensors: Option<&[u8]>,
) -> Result<usize> {
    if sequence.is_some() && flags & 1 != 0 {
        return Ok(0);
    }
    if flags & 2 != 0 && tensors.is_none() {
        return Err(invalid(
            "native device state requires its owning context snapshot",
        ));
    }
    let wanted = sequence.filter(|id| *id != -1);
    if let Some(id) = wanted {
        context.stream_id(id)?;
    }
    let mut input = Reader {
        bytes,
        at: 0,
        tensors: if flags & 2 != 0 { tensors } else { None },
        tensor_at: 0,
    };
    if sequence.is_none() {
        let length = input.u32()? as usize;
        if input.bytes(length)? != b"llama" {
            return Err(invalid("wrong native state architecture"));
        }
    }
    let streams = input.u32()? as usize;
    if streams != context.streams.len() {
        return Err(invalid("native state stream count mismatch"));
    }
    let input_work = bytes
        .len()
        .checked_add(tensors.map_or(0, <[u8]>::len))
        .ok_or_else(|| invalid("native input accounting overflow"))?;
    context.check_work(
        context
            .storage_bytes()?
            .checked_add(input_work)
            .ok_or_else(|| invalid("native clone accounting overflow"))?,
    )?;
    let mut staged = context.try_clone()?;
    if wanted.is_none() {
        staged.clear(true);
    }
    for source in 0..streams {
        let count = input.u32()? as usize;
        if count == 0 {
            continue;
        }
        if count > staged.capacity {
            return Err(invalid("native state cell count exceeds capacity"));
        }
        let destination = match wanted {
            Some(id) => staged.stream_id(id)?,
            None => source,
        };
        let temporary = count
            .checked_mul(staged.width)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_mul(staged.model.layer_count() * 2 + 1))
            .and_then(|n| n.checked_add(count * std::mem::size_of::<Cell>()))
            .ok_or_else(|| invalid("native restore temporary size overflow"))?;
        context.check_work(
            context
                .storage_bytes()?
                .checked_add(input_work)
                .and_then(|n| n.checked_add(temporary))
                .ok_or_else(|| invalid("native restore work overflow"))?,
        )?;
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(count)
            .map_err(|_| invalid("native restore cell allocation failed"))?;
        for _ in 0..count {
            let position = input.i32()?;
            if position < 0 {
                return Err(invalid("negative occupied state position"));
            }
            let ids = input.u32()? as usize;
            if ids > staged.sequence_id_limit() || wanted.is_some() && ids != 1 {
                return Err(invalid("invalid native state cell sequence count"));
            }
            let mut sequences = SequenceSet::new();
            for _ in 0..ids {
                let id = input.i32()?;
                if let Some(destination) = wanted {
                    sequences.insert(destination);
                } else {
                    staged.stream_id(id)?;
                    if !sequences.insert(id) {
                        return Err(invalid("duplicate state sequence ID"));
                    }
                }
            }
            cells.push(Cell {
                position,
                shift: 0,
                sequences,
            });
        }
        if input.u32()? != 1 {
            return Err(invalid("native V transposition mismatch"));
        }
        let layers = input.u32()? as usize;
        if layers != staged.model.layer_count() {
            return Err(invalid("native state layer count mismatch"));
        }
        if input.u32()? != 0 || input.u32()? != 0 {
            return Err(invalid("native state rotation mismatch"));
        }
        let elements = count
            .checked_mul(staged.width)
            .ok_or_else(|| invalid("native state shape overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(layers)
            .map_err(|_| invalid("native restore layer allocation failed"))?;
        for _ in 0..layers {
            if input.i32()? != 0 || input.u64()? != (staged.width * 4) as u64 {
                return Err(invalid("native state key type or row size mismatch"));
            }
            data.push(KvLayer {
                keys: input.floats(elements)?,
                values: Vec::new(),
            });
        }
        for layer in &mut data {
            if input.i32()? != 0 || input.u32()? != 4 || input.u32()? != staged.width as u32 {
                return Err(invalid("native state value shape mismatch"));
            }
            let transposed = input.floats(elements)?;
            layer.values = zeroes(elements)?;
            for dimension in 0..staged.width {
                for index in 0..count {
                    layer.values[index * staged.width + dimension] =
                        transposed[dimension * count + index];
                }
            }
        }
        let slots = if let Some(id) = wanted {
            staged.remove_range(id, -1, -1)?;
            staged.find_slots(destination, count)?
        } else {
            let mut slots = Vec::new();
            slots
                .try_reserve_exact(count)
                .map_err(|_| invalid("native restore slot allocation failed"))?;
            slots.extend(0..count);
            slots
        };
        let stream = &mut staged.streams[destination];
        for ((index, cell), &slot) in cells.into_iter().enumerate().zip(&slots) {
            stream.cells[slot] = cell;
            for (src, dst) in data.iter().zip(&mut stream.layers) {
                dst.keys[slot * staged.width..(slot + 1) * staged.width]
                    .copy_from_slice(&src.keys[index * staged.width..(index + 1) * staged.width]);
                dst.values[slot * staged.width..(slot + 1) * staged.width]
                    .copy_from_slice(&src.values[index * staged.width..(index + 1) * staged.width]);
            }
        }
        stream.head = if wanted.is_some() {
            slots.last().copied().unwrap_or(0) + 1
        } else {
            0
        };
    }
    if let Some(tensors) = input.tensors
        && (input.tensor_at != tensors.len() || input.at != bytes.len())
    {
        return Err(invalid("native device snapshot and metadata sizes differ"));
    }
    *context = staged;
    Ok(input.at)
}
