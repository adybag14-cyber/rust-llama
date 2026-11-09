//! Q8_K activation conversion from quantize_row_q8_K_ref.
use crate::QuantError;

fn inverse_scale(x: &[f32]) -> f32 {
    let mut absolute = 0.0_f32;
    let mut maximum = 0.0_f32;
    for &value in x {
        if value.abs() > absolute {
            absolute = value.abs();
            maximum = value;
        }
    }
    if absolute == 0.0 {
        0.0
    } else {
        -127.0 / maximum
    }
}

pub(crate) fn preflight_q8k(values: &[f32]) -> Result<(), QuantError> {
    for (block, x) in values.as_chunks::<256>().0.iter().enumerate() {
        let inverse = inverse_scale(x);
        if !inverse.is_finite() || (inverse != 0.0 && !(1.0 / inverse).is_finite()) {
            return Err(QuantError::UnrepresentableScale { block });
        }
    }
    Ok(())
}

pub(crate) fn encode_q8k(x: &[f32], b: &mut [u8]) {
    b.fill(0);
    let inverse = inverse_scale(x);
    if inverse == 0.0 {
        // The C reference leaves bsums untouched for zero blocks. Canonical
        // zero sums make the owned Rust encoding deterministic and reusable.
        return;
    }
    for j in 0..256 {
        // nearest_int's float-bias trick is ties-to-even within this range.
        let q = (inverse * x[j]).round_ties_even() as i32;
        b[4 + j] = q.min(127) as i8 as u8;
    }
    for group in 0..16 {
        let mut sum = 0_i16;
        for j in 0..16 {
            sum += b[4 + group * 16 + j] as i8 as i16;
        }
        b[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
    }
    b[..4].copy_from_slice(&(1.0 / inverse).to_le_bytes());
}
