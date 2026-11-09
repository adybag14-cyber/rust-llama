//! Source-compatible indexed CPU operations. Inputs remain immutable.
use crate::graph::check_cancel;
use crate::{DType, Op, Result, SortOrder, Tensor, TensorDesc, TensorError, reserved, zeros};
use std::sync::atomic::AtomicBool;

pub(super) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    use Op::*;
    Some(match op {
        AddId => add_id(src[0], src[1], src[2], cancel),
        MatMulId => matmul_id(src[0], src[1], src[2], cancel),
        SetRows => set_rows(src[0], src[1], src[2], cancel),
        Argsort { order } => select(src[0], *order, None, cancel),
        TopK { count } => select(src[0], SortOrder::Descending, Some(*count), cancel),
        TopKMask {
            with_scores,
            with_blocks,
            with_extra,
        } => top_k_mask(src, [*with_scores, *with_blocks, *with_extra], cancel),
        Acc { strides, offset } => update(src[0], src[1], *strides, *offset, true, cancel),
        Set { strides, offset } => update(src[0], src[1], *strides, *offset, false, cancel),
        Cast { dtype } => cast(src[0], *dtype, cancel),
        _ => return None,
    })
}
fn count(shape: [usize; 4]) -> Result<usize> {
    TensorDesc::contiguous(DType::F32, shape)?.elements()
}
fn flat(i: [usize; 4], n: [usize; 4]) -> usize {
    i[0] + n[0] * (i[1] + n[1] * (i[2] + n[2] * i[3]))
}
fn row_coords(mut row: usize, n: [usize; 4]) -> [usize; 3] {
    let mut i = [0; 3];
    for axis in 0..3 {
        i[axis] = row % n[axis + 1];
        row /= n[axis + 1];
    }
    i
}
fn require_type(t: &Tensor, expected: DType) -> Result<()> {
    if t.dtype() != expected {
        Err(TensorError::Type {
            expected,
            actual: t.dtype(),
        })
    } else {
        Ok(())
    }
}
fn contiguous_rows(t: &Tensor) -> Result<()> {
    if t.desc.nb[0] != t.dtype().type_size() {
        Err(TensorError::Layout(
            "source CPU operation requires contiguous blocks within rows",
        ))
    } else {
        Ok(())
    }
}
fn copy_span(t: &Tensor) -> Result<Vec<u8>> {
    let span = t.desc.byte_span()?;
    let mut bytes = reserved(span)?;
    bytes.extend_from_slice(&t.data[t.offset..t.offset + span]);
    Ok(bytes)
}

fn add_id(a: &Tensor, bias: &Tensor, ids: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    require_type(a, DType::F32)?;
    require_type(bias, DType::F32)?;
    contiguous_rows(a)?;
    contiguous_rows(bias)?;
    let an = a.shape();
    let bn = bias.shape();
    let inn = ids.shape();
    if an[0] != bn[0] || an[1] != inn[0] || an[2] != inn[1] {
        return Err(TensorError::Shape("ADD_ID dimensions are incompatible"));
    }
    let indices = ids.to_i32()?;
    let mut out = a.to_f32()?;
    if out.is_empty() {
        return Tensor::from_f32(an, &out);
    }
    if inn[2] == 0 || inn[3] == 0 || bn[2] == 0 || bn[3] == 0 {
        return Err(TensorError::Bounds);
    }
    // The source reads only the first IDs/bias slab, even for a batched a.
    for &id in &indices[..inn[0] * inn[1]] {
        if id < -1 || (id >= 0 && id as usize >= bn[1]) {
            return Err(TensorError::Bounds);
        }
    }
    let b = bias.to_f32()?;
    for i3 in 0..an[3] {
        for i2 in 0..an[2] {
            for i1 in 0..an[1] {
                check_cancel(cancel)?;
                let id = indices[i1 + inn[0] * i2];
                if id == -1 {
                    continue;
                }
                let at = flat([0, i1, i2, i3], an);
                let bi = bn[0] * id as usize;
                for j in 0..an[0] {
                    out[at + j] += b[bi + j];
                }
            }
        }
    }
    Tensor::from_f32(an, &out)
}

/// Availability of the frozen CPU from_float function, not generic quantizers.
fn has_cpu_encoder(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::F16
            | DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Q8_0
            | DType::Q8_1
            | DType::Q2_K
            | DType::Q3_K
            | DType::Q4_K
            | DType::Q5_K
            | DType::Q6_K
            | DType::Q8_K
            | DType::IQ4_NL
            | DType::IQ4_XS
            | DType::I32
            | DType::BF16
            | DType::TQ1_0
            | DType::TQ2_0
            | DType::MXFP4
            | DType::NVFP4
            | DType::Q1_0
            | DType::Q2_0
            | DType::F8_E4M3
    )
}
fn encode_cpu_row(dtype: DType, row: &[f32]) -> Result<Vec<u8>> {
    if !has_cpu_encoder(dtype) {
        return Err(TensorError::Unsupported(
            "destination dtype has no frozen CPU from_float converter",
        ));
    }
    if dtype == DType::I32 {
        Ok(llama_quant::convert_f32_to_integer(dtype, row)?)
    } else {
        Ok(llama_quant::quantize(dtype, row)?)
    }
}
fn encode_rows(
    dtype: DType,
    shape: [usize; 4],
    values: &[f32],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    let desc = TensorDesc::contiguous(dtype, shape)?;
    let mut bytes = reserved(desc.byte_span()?)?;
    if shape[0] != 0 {
        for row in values.chunks_exact(shape[0]) {
            check_cancel(cancel)?;
            bytes.extend_from_slice(&encode_cpu_row(dtype, row)?);
        }
    }
    Tensor::from_bytes(desc, bytes)
}

fn matmul_id(a: &Tensor, b: &Tensor, ids: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    let an = a.shape();
    let bn = b.shape();
    let inn = ids.shape();
    let activation = llama_quant::matvec_activation_dtype(a.dtype())?;
    contiguous_rows(a)?;
    contiguous_rows(b)?;
    if a.desc.nb[0] > a.desc.nb[1] {
        return Err(TensorError::Layout(
            "MUL_MAT_ID does not accept a transposed matrix",
        ));
    }
    if an[0] == 0
        || an[2] == 0
        || bn[1] == 0
        || an[0] != bn[0]
        || an[3] != 1
        || bn[3] != 1
        || inn[2] != 1
        || inn[3] != 1
        || inn[1] != bn[2]
        || !inn[0].is_multiple_of(bn[1])
    {
        return Err(TensorError::Shape(
            "MUL_MAT_ID expert, slot or token dimensions are incompatible",
        ));
    }
    if b.dtype() != DType::F32 && b.dtype() != activation {
        return Err(TensorError::Unsupported(
            "MUL_MAT_ID RHS must be F32 or the exact native activation dtype",
        ));
    }
    let indices = ids.to_i32()?;
    for &id in &indices {
        if id < -1 || (id >= 0 && id as usize >= an[2]) {
            return Err(TensorError::Bounds);
        }
    }
    let shape = [an[1], inn[0], inn[1], 1];
    let mut out = zeros(count(shape)?)?;
    if out.is_empty() {
        return Tensor::from_f32(shape, &out);
    }
    // Native MUL_MAT_ID quantizes every activation row once before expert work.
    let conversion;
    let right = if b.dtype() != activation {
        conversion = encode_rows(activation, bn, &b.to_f32()?, cancel)?;
        &conversion
    } else {
        b
    };
    let left_desc = TensorDesc::with_strides(a.dtype(), [an[0], an[1], 1, 1], a.desc.nb)?;
    let right_desc = TensorDesc::with_strides(right.dtype(), [bn[0], 1, 1, 1], right.desc.nb)?;
    for token in 0..inn[1] {
        for slot in 0..inn[0] {
            check_cancel(cancel)?;
            let id = indices[slot + token * inn[0]];
            if id == -1 {
                continue;
            }
            let left = a.view(left_desc, id as usize * a.desc.nb[2])?;
            // Selection uses the slot, not the selected expert, for RHS broadcasting.
            let rhs = right.view(
                right_desc,
                (slot % bn[1]) * right.desc.nb[1] + token * right.desc.nb[2],
            )?;
            let values = crate::cpu_extra::matmul(&left, &rhs, cancel)?;
            let base = an[1] * (slot + inn[0] * token);
            for (dst, bytes) in out[base..base + an[1]]
                .iter_mut()
                .zip(values.data.as_chunks::<4>().0.iter())
            {
                *dst = f32::from_le_bytes(*bytes);
            }
        }
    }
    Tensor::from_f32(shape, &out)
}

fn set_rows(base: &Tensor, values: &Tensor, ids: &Tensor, cancel: &AtomicBool) -> Result<Tensor> {
    if !matches!(values.dtype(), DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "SET_ROWS values must be F32 or F16",
        ));
    }
    if !has_cpu_encoder(base.dtype()) {
        return Err(TensorError::Unsupported(
            "SET_ROWS destination has no frozen CPU converter",
        ));
    }
    contiguous_rows(base)?;
    contiguous_rows(values)?;
    let an = base.shape();
    let bn = values.shape();
    let inn = ids.shape();
    if an[0] != bn[0]
        || an[2] != bn[2]
        || an[3] != bn[3]
        || bn[1] != inn[0]
        || inn[1] == 0
        || inn[2] == 0
        || !bn[2].is_multiple_of(inn[1])
        || !bn[3].is_multiple_of(inn[2])
        || inn[3] != 1
    {
        return Err(TensorError::Shape(
            "SET_ROWS destination, row or index batch dimensions differ",
        ));
    }
    let indices = if ids.dtype() == DType::I64 {
        ids.to_i64()?
    } else {
        let mut all = reserved(ids.desc.elements()?)?;
        all.extend(ids.to_i32()?.into_iter().map(i64::from));
        all
    };
    for &id in &indices {
        if id < 0 || id as u64 >= an[1] as u64 {
            return Err(TensorError::Bounds);
        }
    }
    let mut out = copy_span(base)?;
    if bn.contains(&0) {
        return Tensor::from_bytes(base.desc, out);
    }
    let raw_half = values.dtype() == DType::F16 && base.dtype() == DType::F16;
    let decoded = if raw_half {
        None
    } else {
        Some(values.to_f32()?)
    };
    let row_bytes = base.dtype().row_bytes(an[0])?;
    for i3 in 0..bn[3] {
        for i2 in 0..bn[2] {
            for row in 0..bn[1] {
                check_cancel(cancel)?;
                let id = indices[flat([row, i2 % inn[1], i3 % inn[2], 0], inn)] as usize;
                let at = id * base.desc.nb[1] + i2 * base.desc.nb[2] + i3 * base.desc.nb[3];
                if raw_half {
                    let from = values.offset
                        + row * values.desc.nb[1]
                        + i2 * values.desc.nb[2]
                        + i3 * values.desc.nb[3];
                    out[at..at + row_bytes].copy_from_slice(&values.data[from..from + row_bytes]);
                } else {
                    let from = flat([0, row, i2, i3], bn);
                    let floats = decoded.as_ref().expect("converted input");
                    let encoded = encode_cpu_row(base.dtype(), &floats[from..from + bn[0]])?;
                    out[at..at + row_bytes].copy_from_slice(&encoded);
                }
            }
        }
    }
    Tensor::from_bytes(base.desc, out)
}

fn cast(a: &Tensor, dtype: DType, cancel: &AtomicBool) -> Result<Tensor> {
    if a.dtype() == dtype {
        return a.contiguous();
    }
    let supported = match a.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => has_cpu_encoder(dtype),
        DType::F8_E4M3 => matches!(dtype, DType::F32 | DType::F16 | DType::BF16),
        DType::I32 => dtype == DType::F32,
        DType::Q8_1 | DType::Q8_K => false, // No source to_float trait.
        t if t.block_size() > 1 => dtype == DType::F32,
        _ => false,
    };
    if !supported {
        return Err(TensorError::Unsupported(
            "conversion is absent from the frozen CPU CPY dispatcher",
        ));
    }
    encode_rows(dtype, a.shape(), &a.to_f32()?, cancel)
}

fn update(
    base: &Tensor,
    values: &Tensor,
    strides: [usize; 3],
    offset: usize,
    accumulate: bool,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    if !matches!(base.dtype(), DType::F32 | DType::I32)
        || (accumulate && base.dtype() != DType::F32)
    {
        return Err(TensorError::Unsupported(
            "ACC requires F32; SET supports F32 or I32",
        ));
    }
    require_type(values, base.dtype())?;
    if !base.desc.is_contiguous() {
        return Err(TensorError::Layout("ACC/SET base must be contiguous"));
    }
    contiguous_rows(values)?;
    if values.desc.elements()? > base.desc.elements()? {
        return Err(TensorError::Shape(
            "ACC/SET update has more elements than its base",
        ));
    }
    if offset >= 1 << 30
        || !offset.is_multiple_of(4)
        || strides
            .iter()
            .any(|&s| s > i32::MAX as usize || !s.is_multiple_of(4))
    {
        return Err(TensorError::Parameters(
            "ACC/SET byte offset and strides must be aligned and fit source signed parameters",
        ));
    }
    let shape = values.shape();
    let view =
        TensorDesc::with_strides(base.dtype(), shape, [4, strides[0], strides[1], strides[2]])?;
    let end = offset
        .checked_add(view.byte_span()?)
        .ok_or(TensorError::Overflow)?;
    if end > base.desc.byte_span()? {
        return Err(TensorError::Bounds);
    }
    let mut out = copy_span(base)?;
    if view.elements()? == 0 {
        return Tensor::from_bytes(base.desc, out);
    }
    for i3 in 0..shape[3] {
        for i2 in 0..shape[2] {
            for i1 in 0..shape[1] {
                check_cancel(cancel)?;
                let dest = offset + i1 * strides[0] + i2 * strides[1] + i3 * strides[2];
                let from = values.offset
                    + i1 * values.desc.nb[1]
                    + i2 * values.desc.nb[2]
                    + i3 * values.desc.nb[3];
                for i0 in 0..shape[0] {
                    let d = dest + i0 * 4;
                    let s = from + i0 * 4;
                    if accumulate {
                        let original = f32::from_le_bytes(
                            base.data[base.offset + d..base.offset + d + 4]
                                .try_into()
                                .expect("F32 base"),
                        );
                        let add = f32::from_le_bytes(
                            values.data[s..s + 4].try_into().expect("F32 input"),
                        );
                        out[d..d + 4].copy_from_slice(&(original + add).to_le_bytes());
                    } else {
                        out[d..d + 4].copy_from_slice(&values.data[s..s + 4]);
                    }
                }
            }
        }
    }
    Tensor::from_bytes(base.desc, out)
}

fn select(
    a: &Tensor,
    order: SortOrder,
    top_k: Option<usize>,
    cancel: &AtomicBool,
) -> Result<Tensor> {
    require_type(a, DType::F32)?;
    let shape = a.shape();
    let width = shape[0];
    let keep = top_k.unwrap_or(width);
    if width > i32::MAX as usize || keep > width {
        return Err(TensorError::Parameters(
            "sort width must fit I32 and top-k cannot exceed it",
        ));
    }
    let mut out_shape = shape;
    out_shape[0] = keep;
    let mut out = reserved(count(out_shape)?)?;
    if keep == 0 || shape.contains(&0) {
        return Tensor::from_i32(out_shape, &out);
    }
    crate::native_sort::check_platform()?;
    let rows = shape[1]
        .checked_mul(shape[2])
        .and_then(|n| n.checked_mul(shape[3]))
        .ok_or(TensorError::Overflow)?;
    // The original sort kernels use flattened row*nb[1], regardless of nb[2/3].
    // Preserve that addressing, but reject any access outside the source span.
    let row_bytes = width.checked_mul(4).ok_or(TensorError::Overflow)?;
    let end = (rows - 1)
        .checked_mul(a.desc.nb[1])
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or(TensorError::Overflow)?;
    if end > a.desc.byte_span()? {
        return Err(TensorError::Bounds);
    }
    let mut keys = reserved(width)?;
    let mut indices = reserved(width)?;
    for row in 0..rows {
        check_cancel(cancel)?;
        keys.clear();
        indices.clear();
        let start = a.offset + row * a.desc.nb[1];
        for bytes in a.data[start..start + row_bytes].as_chunks::<4>().0 {
            keys.push(f32::from_le_bytes(*bytes));
        }
        if keys.iter().any(|v| v.is_nan()) {
            return Err(TensorError::Parameters(
                "NaN sort keys violate the source comparator's strict weak ordering",
            ));
        }
        indices.extend((0..width).map(|i| i as i32));
        if top_k.is_some() {
            crate::native_sort::partial(&mut indices, keep, &keys);
            if keep > 1 {
                indices.swap(0, 1);
            }
        } else {
            crate::native_sort::full(&mut indices, &keys, order);
        }
        out.extend_from_slice(&indices[..keep]);
    }
    Tensor::from_i32(out_shape, &out)
}

fn top_k_mask(src: &[&Tensor], flags: [bool; 3], cancel: &AtomicBool) -> Result<Tensor> {
    let mask = src[0];
    let top = src[1];
    if !matches!(mask.dtype(), DType::F32 | DType::F16) {
        return Err(TensorError::Unsupported(
            "TOP_K_MASK requires an F32 or F16 mask",
        ));
    }
    contiguous_rows(mask)?;
    require_type(top, DType::I32)?;
    let mn = mask.shape();
    let tn = top.shape();
    if (1..4).any(|axis| mn[axis] == 0 || !tn[axis].is_multiple_of(mn[axis])) {
        return Err(TensorError::Shape(
            "TOP_K_MASK mask batch dimensions must divide index batches",
        ));
    }
    let mut at = 2;
    let mut optional = [None; 3];
    for (slot, enabled) in optional.iter_mut().zip(flags) {
        if enabled {
            *slot = Some(src[at]);
            at += 1;
        }
    }
    let [score, blocks, extra] = optional;
    if let Some(s) = score {
        require_type(s, DType::F32)?;
        if s.shape()[1..] != tn[1..] {
            return Err(TensorError::Shape("TOP_K_MASK score batch differs"));
        }
    }
    if let Some(b) = blocks {
        require_type(b, DType::I32)?;
        if b.shape()[2] != 1
            || b.shape()[3] != 1
            || score.is_some_and(|s| b.shape()[1] != s.shape()[0])
        {
            return Err(TensorError::Shape("TOP_K_MASK block shape differs"));
        }
    }
    if let Some(e) = extra {
        require_type(e, DType::I32)?;
        if e.shape()[1..] != tn[1..] {
            return Err(TensorError::Shape("TOP_K_MASK extra batch differs"));
        }
    }
    let n_pick = blocks.map_or_else(|| score.map_or(mn[0], |s| s.shape()[0]), |b| b.shape()[1]);
    let n_block = blocks.map_or(1, |b| b.shape()[0]);
    let n_extra = extra.map_or(0, |e| e.shape()[0]);
    let scores = score.map(Tensor::to_f32).transpose()?;
    let block_ids = blocks.map(Tensor::to_i32).transpose()?;
    let extra_ids = extra.map(Tensor::to_i32).transpose()?;
    let top_ids = top.to_i32()?;
    let original = mask.to_bytes()?;
    let shape = [mn[0], tn[1], tn[2], tn[3]];
    let desc = TensorDesc::contiguous(mask.dtype(), shape)?;
    let mut bytes = reserved(desc.byte_span()?)?;
    let ts = mask.dtype().type_size();
    let ninf = if mask.dtype() == DType::F16 {
        vec![0x00, 0xfc]
    } else {
        f32::NEG_INFINITY.to_le_bytes().to_vec()
    };
    for _ in 0..desc.elements()? {
        bytes.extend_from_slice(&ninf);
    }
    if shape.contains(&0) {
        return Tensor::from_bytes(desc, bytes);
    }
    let rows = desc.elements()? / mn[0];
    for row in 0..rows {
        check_cancel(cancel)?;
        let [i1, i2, i3] = row_coords(row, shape);
        let source = flat([0, i1 % mn[1], i2 % mn[2], i3 % mn[3]], mn) * ts;
        let dest = row * mn[0] * ts;
        let mut keep = |column: i32| {
            if column >= 0 && (column as usize) < mn[0] {
                let c = column as usize * ts;
                bytes[dest + c..dest + c + ts]
                    .copy_from_slice(&original[source + c..source + c + ts]);
            }
        };
        for &p in &top_ids[row * tn[0]..(row + 1) * tn[0]] {
            if p < 0 || p as usize >= n_pick {
                continue;
            }
            if let Some(s) = &scores {
                let value = s[row * n_pick + p as usize];
                if value <= f32::NEG_INFINITY || value.is_nan() {
                    continue;
                }
            }
            if let Some(b) = &block_ids {
                for &column in &b[p as usize * n_block..(p as usize + 1) * n_block] {
                    keep(column);
                }
            } else {
                keep(p);
            }
        }
        if let Some(e) = &extra_ids {
            for &column in &e[row * n_extra..(row + 1) * n_extra] {
                keep(column);
            }
        }
    }
    Tensor::from_bytes(desc, bytes)
}
