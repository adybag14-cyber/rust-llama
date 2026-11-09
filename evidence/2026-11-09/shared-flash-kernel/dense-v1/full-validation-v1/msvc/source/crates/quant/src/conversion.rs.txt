//! Checked F32-to-integer assignments corresponding to ggml_set_f32_1d.
//! These are scalar tensor conversions, not GGML model quantizer capabilities.
use crate::{DType, QuantError};

fn bounds(dtype: DType) -> Result<(f64, f64), QuantError> {
    match dtype {
        DType::I8 => Ok((i8::MIN as f64, i8::MAX as f64)),
        DType::I16 => Ok((i16::MIN as f64, i16::MAX as f64)),
        DType::I32 => Ok((i32::MIN as f64, i32::MAX as f64)),
        _ => Err(QuantError::UnsupportedIntegerConversion(dtype)),
    }
}
/// Truncate F32 toward zero to I8/I16/I32, matching defined source assignments.
/// Nonfinite and out-of-range inputs are rejected instead of invoking C UB.
/// The source F32 setter does not support I64.
pub fn convert_f32_to_integer(dtype: DType, values: &[f32]) -> Result<Vec<u8>, QuantError> {
    bounds(dtype)?;
    let bytes = dtype.row_bytes(values.len())?;
    let mut out = Vec::new();
    out.try_reserve_exact(bytes)
        .map_err(|_| QuantError::AllocationFailed)?;
    out.resize(bytes, 0);
    convert_f32_to_integer_into(dtype, values, &mut out)?;
    Ok(out)
}
/// Validate the entire conversion before changing a caller-owned destination.
pub fn convert_f32_to_integer_into(
    dtype: DType,
    values: &[f32],
    out: &mut [u8],
) -> Result<(), QuantError> {
    let (min, max) = bounds(dtype)?;
    crate::validate(dtype, out.len(), values.len())?;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantError::NonFiniteInput { index });
        }
        let truncated = value.trunc() as f64;
        if truncated < min || truncated > max {
            return Err(QuantError::IntegerOutOfRange { dtype, index });
        }
    }
    for (&value, bytes) in values.iter().zip(out.chunks_exact_mut(dtype.type_size())) {
        match dtype {
            DType::I8 => bytes[0] = value as i8 as u8,
            DType::I16 => bytes.copy_from_slice(&(value as i16).to_le_bytes()),
            DType::I32 => bytes.copy_from_slice(&(value as i32).to_le_bytes()),
            _ => unreachable!("validated source integer assignment"),
        }
    }
    Ok(())
}
