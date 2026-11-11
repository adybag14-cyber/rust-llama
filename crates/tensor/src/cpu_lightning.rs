//! Frozen CPU Lightning Indexer: decoded key rows, scalar dot and F16 masks.
use crate::graph::check_cancel;
use crate::{DType, Result, Tensor, TensorDesc, TensorError, zeros};
use std::sync::atomic::AtomicBool;

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(TensorError::Overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(TensorError::Overflow)
}
fn require_type(tensor: &Tensor, dtype: DType) -> Result<()> {
    if tensor.dtype() != dtype {
        return Err(TensorError::Type {
            expected: dtype,
            actual: tensor.dtype(),
        });
    }
    Ok(())
}
fn raw(tensor: &Tensor) -> Result<&[u8]> {
    tensor
        .data
        .get(tensor.offset..add(tensor.offset, tensor.desc().byte_span()?)?)
        .ok_or(TensorError::Bounds)
}
fn bytes_at(bytes: &[u8], offset: usize, count: usize) -> Result<&[u8]> {
    bytes
        .get(offset..add(offset, count)?)
        .ok_or(TensorError::Bounds)
}
fn native_key_decoder(dtype: DType) -> bool {
    // These seven active storage types have no original CPU to_float trait.
    // F32 is the direct original row path; the other 28 use original decoders.
    !matches!(
        dtype,
        DType::Q8_1 | DType::Q8_K | DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::F64
    )
}

pub(super) fn evaluate(src: &[&Tensor], cancel: &AtomicBool) -> Result<Tensor> {
    let [q, k, w, m] = [src[0], src[1], src[2], src[3]];
    require_type(q, DType::F32)?;
    require_type(w, DType::F32)?;
    require_type(m, DType::F16)?;
    for tensor in src {
        if tensor.shape().into_iter().any(|n| n > i32::MAX as usize) {
            return Err(TensorError::Parameters(
                "Lightning Indexer dimensions must fit signed I32",
            ));
        }
    }
    let [embedding, heads, tokens, streams] = q.shape();
    let keys = k.shape()[2];
    let mask_streams = m.shape()[3];
    if k.shape() != [embedding, 1, keys, streams]
        || w.shape() != [heads, tokens, 1, streams]
        || m.shape() != [keys, tokens, 1, mask_streams]
        || mask_streams == 0
        || !streams.is_multiple_of(mask_streams)
    {
        return Err(TensorError::Shape(
            "Lightning Indexer Q/K/weight/mask geometry does not match",
        ));
    }
    let shape = [keys, tokens, 1, streams];
    let desc = TensorDesc::contiguous(DType::F32, shape)?;
    let output_count = desc.elements()?;
    // Original empty destination dispatch skips the kernel, including its
    // key-decoder and physical-row assertions.
    if output_count == 0 {
        return Tensor::from_bytes(desc, Vec::new());
    }
    for tensor in [q, k, w, m] {
        if tensor.desc().nb[0] != tensor.dtype().type_size() {
            return Err(TensorError::Layout(
                "Lightning Indexer requires contiguous inner rows",
            ));
        }
    }
    for tensor in [q, w] {
        if tensor
            .desc()
            .nb
            .iter()
            .any(|stride| !stride.is_multiple_of(4))
        {
            return Err(TensorError::Layout(
                "Lightning Indexer Q/weight strides must be F32-aligned",
            ));
        }
    }
    if m.desc().nb.iter().any(|stride| !stride.is_multiple_of(2)) {
        return Err(TensorError::Layout(
            "Lightning Indexer mask strides must be F16-aligned",
        ));
    }
    if !native_key_decoder(k.dtype()) {
        return Err(TensorError::Unsupported(
            "original Lightning Indexer has no decoder for this key dtype",
        ));
    }
    let decoded_row_bytes = mul(embedding, 4)?;
    if desc.byte_span()? > 256 * 1024 * 1024 || decoded_row_bytes > 256 * 1024 * 1024 {
        return Err(TensorError::Unsupported(
            "Lightning Indexer output or decoded row exceeds the 256 MiB allocation limit",
        ));
    }
    // Count key decoding, dot products and weighted-head reductions before
    // an aliased view can amplify a small buffer into unbounded scalar work.
    let work_per_key = add(mul(embedding, add(heads, 1)?)?, heads)?;
    if mul(output_count, work_per_key)? > 256 * 1024 * 1024 {
        return Err(TensorError::Unsupported(
            "Lightning Indexer exceeds the 256 Mi scalar-work limit",
        ));
    }
    let qb = raw(q)?;
    let kb = raw(k)?;
    let wb = raw(w)?;
    let mb = raw(m)?;
    let key_row_bytes = k.dtype().row_bytes(embedding)?;
    let mut key_row = zeros(embedding)?;
    let mut result = zeros(output_count)?;
    for stream in 0..streams {
        for token in 0..tokens {
            check_cancel(cancel)?;
            let weight_offset = add(mul(stream, w.desc().nb[3])?, mul(token, w.desc().nb[1])?)?;
            // With no heads, the original kernel does not read weights.
            let weights = if heads == 0 {
                &[][..]
            } else {
                bytes_at(wb, weight_offset, mul(heads, 4)?)?
            };
            for word in weights.as_chunks::<4>().0 {
                if !f32::from_le_bytes(*word).is_finite() {
                    return Err(TensorError::Unsupported(
                        "Lightning Indexer requires finite weights and pre-mask scores",
                    ));
                }
            }
            let mask_offset = add(
                mul(stream % mask_streams, m.desc().nb[3])?,
                mul(token, m.desc().nb[1])?,
            )?;
            let masks = bytes_at(mb, mask_offset, mul(keys, 2)?)?;
            for key in 0..keys {
                check_cancel(cancel)?;
                if embedding != 0 {
                    let offset = add(mul(stream, k.desc().nb[3])?, mul(key, k.desc().nb[2])?)?;
                    let row = bytes_at(kb, offset, key_row_bytes)?;
                    llama_quant::dequantize_into(k.dtype(), row, &mut key_row)?;
                }
                let mut score = 0.0_f32;
                for (head, weight) in weights.as_chunks::<4>().0.iter().enumerate() {
                    let mut sum = 0.0_f64;
                    if embedding != 0 {
                        let offset = add(
                            add(mul(stream, q.desc().nb[3])?, mul(token, q.desc().nb[2])?)?,
                            mul(head, q.desc().nb[1])?,
                        )?;
                        let query = bytes_at(qb, offset, decoded_row_bytes)?;
                        for (query, key) in query.as_chunks::<4>().0.iter().zip(&key_row) {
                            sum += (f32::from_le_bytes(*query) * *key) as f64;
                        }
                    }
                    let dot = sum as f32;
                    // Source MAX(dot, 0) returns +0 for NaN and either zero sign.
                    let positive = if dot > 0.0 { dot } else { 0.0 };
                    score += positive * f32::from_le_bytes(*weight);
                }
                if !score.is_finite() {
                    return Err(TensorError::Unsupported(
                        "Lightning Indexer requires finite weights and pre-mask scores",
                    ));
                }
                let bits =
                    u16::from_le_bytes(masks[key * 2..key * 2 + 2].try_into().expect("two bytes"));
                let mask = llama_quant::f16_to_f32(bits);
                // The source's single NaN comes from the half mask when the
                // score is finite; preserve its measured quiet payload.
                let masked = if mask.is_nan() { mask } else { score + mask };
                result[key + keys * (token + tokens * stream)] = masked;
            }
        }
    }
    Tensor::from_f32(shape, &result)
}
