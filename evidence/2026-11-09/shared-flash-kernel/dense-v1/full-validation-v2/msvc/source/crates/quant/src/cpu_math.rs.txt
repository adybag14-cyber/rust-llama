//! Frozen scalar/SSE2 F32 row policies, expressed in safe Rust.
//!
//! The SSE2 policy follows the no-FMA branch in the original ggml vec.h.
//! Four-element prefixes and scalar tails are part of the numerical contract.
//! Environment: round-to-nearest ties-to-even and gradual underflow. The API
//! does not promise to reproduce floating exception status flags.
#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathPolicy {
    Scalar,
    Sse2,
}

pub const fn native_math_policy() -> MathPolicy {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        MathPolicy::Sse2
    } else {
        MathPolicy::Scalar
    }
}

/// SiLU on one source row. Do not split a row into arbitrary scalar calls.
pub fn silu_f32_in_place(values: &mut [f32]) {
    silu_f32_with_policy(values, native_math_policy());
}

/// Select the vector-prefix policy explicitly. Scalar tail exp uses the target
/// platform's normal Rust/libm lowering, even for an explicitly selected policy.
pub fn silu_f32_with_policy(values: &mut [f32], policy: MathPolicy) {
    let prefix = if policy == MathPolicy::Sse2 {
        values.len() / 4 * 4
    } else {
        0
    };
    let (vector, tail) = values.split_at_mut(prefix);
    for group in vector.as_chunks_mut::<4>().0 {
        let exponent = sse2_exp_f32x4(group.map(|v| 0.0 - v));
        for (value, exp) in group.iter_mut().zip(exponent) {
            *value /= 1.0 + exp;
        }
    }
    for value in tail {
        *value /= 1.0 + (-*value).exp();
    }
}

/// Replace one row with exp(value-maximum) and return the original F64 sum.
/// The caller retains source maximum validation and final reciprocal scaling.
pub fn softmax_exp_sum_f32_in_place(values: &mut [f32], maximum: f32) -> f64 {
    softmax_exp_sum_f32_with_policy(values, maximum, native_math_policy())
}

pub fn softmax_exp_sum_f32_with_policy(
    values: &mut [f32],
    maximum: f32,
    policy: MathPolicy,
) -> f64 {
    let prefix = if policy == MathPolicy::Sse2 {
        values.len() / 4 * 4
    } else {
        0
    };
    let (vector, tail) = values.split_at_mut(prefix);
    let mut sum = 0.0_f64;
    for group in vector.as_chunks_mut::<4>().0 {
        let output = sse2_exp_f32x4(group.map(|v| v - maximum));
        group.copy_from_slice(&output);
        // Original no-AVX SSE2 shuffle: adjacent pairs in F32, then one F32 add.
        let pairs = ordered_add_f32(
            ordered_add_f32(output[0], output[1]),
            ordered_add_f32(output[2], output[3]),
        );
        sum = ordered_add_f64(sum, pairs as f64);
    }
    for value in tail {
        *value = (*value - maximum).exp();
        sum = if policy == MathPolicy::Sse2 {
            ordered_add_f64(sum, *value as f64)
        } else {
            sum + *value as f64
        };
    }
    sum
}

/// Original four-lane SSE2 exponential. Constants are exact source F32 bits.
/// This intentionally reproduces the original approximation and scaling, rather
/// than replacing it with a differently rounded scalar exponential.
pub fn sse2_exp_f32x4(input: [f32; 4]) -> [f32; 4] {
    let parts = input.map(|x| {
        let z = x * f32::from_bits(0x3fb8_aa3b) + f32::from_bits(0x4b40_0000);
        let n = z - f32::from_bits(0x4b40_0000);
        let b = (x - n * f32::from_bits(0x3f31_7200)) - n * f32::from_bits(0x35bf_be8e);
        let exponent = z.to_bits().wrapping_shl(23);
        let scale = f32::from_bits(exponent.wrapping_add(0x3f80_0000));
        let squared = b * b;
        let first = f32::from_bits(0x3c07_2010) * b + f32::from_bits(0x3d2b_9f17);
        let second = f32::from_bits(0x3e2a_af33) * b + f32::from_bits(0x3eff_fedb);
        let polynomial = (first * squared + second) * squared + f32::from_bits(0x3f7f_fff6) * b;
        (n, exponent, scale, polynomial)
    });
    // The source branches once for the complete vector, not once per element.
    if !parts.iter().any(|p| p.0.abs() > 126.0) {
        return parts.map(|(_, _, scale, polynomial)| polynomial * scale + scale);
    }
    parts.map(|(n, exponent, scale, polynomial)| {
        let shift = if n <= 0.0 { 0x8200_0000_u32 } else { 0 };
        let high = f32::from_bits(shift.wrapping_add(0x7f00_0000));
        let low = f32::from_bits(exponent.wrapping_sub(shift));
        if n.abs() > 192.0 {
            high * high
        } else if n.abs() > 126.0 {
            (low * polynomial + low) * high
        } else {
            scale * polynomial + scale
        }
    })
}

// Scalar Rust/LLVM may commute an add and choose a different NaN payload.
// The frozen GNU SSE2 shuffle and F64 accumulator preserve the first NaN.
fn ordered_add_f32(left: f32, right: f32) -> f32 {
    if left.is_nan() {
        f32::from_bits(left.to_bits() | 0x0040_0000)
    } else if right.is_nan() {
        f32::from_bits(right.to_bits() | 0x0040_0000)
    } else {
        left + right
    }
}
fn ordered_add_f64(left: f64, right: f64) -> f64 {
    if left.is_nan() {
        f64::from_bits(left.to_bits() | 0x0008_0000_0000_0000)
    } else if right.is_nan() {
        f64::from_bits(right.to_bits() | 0x0008_0000_0000_0000)
    } else {
        left + right
    }
}

/// Center a whole contiguous F32 row using the original variance reduction.
/// The return value is the F64 square sum divided by the row width. A NORM
/// caller rounds it to F32 before adding epsilon and computing its scale.
/// An empty row returns the original floating zero-divided-by-zero result.
pub fn center_f32_and_variance(values: &mut [f32], mean: f32) -> f64 {
    center_f32_and_variance_with_policy(values, mean, native_math_policy())
}

pub fn center_f32_and_variance_with_policy(
    values: &mut [f32],
    mean: f32,
    policy: MathPolicy,
) -> f64 {
    let width = values.len();
    let prefix = width / 4 * 4;
    let (vector, tail) = values.split_at_mut(prefix);
    let mut sum = 0.0_f64;
    for group in vector.as_chunks_mut::<4>().0 {
        for value in group.iter_mut() {
            *value -= mean;
        }
        let squares = group.map(|value| value * value);
        if policy == MathPolicy::Sse2 {
            let paired = ordered_add_f32(
                ordered_add_f32(squares[0], squares[1]),
                ordered_add_f32(squares[2], squares[3]),
            );
            sum = ordered_add_f64(sum, paired as f64);
        } else {
            // The frozen MSVC scalar function unrolls four iterations. Each
            // F64 add places the new square in the first operand; its scalar
            // tail places the running sum first. Numeric reduction order is
            // unchanged, but the distinction preserves observed NaN payloads.
            for square in squares {
                sum = ordered_add_f64(square as f64, sum);
            }
        }
    }
    for value in tail {
        *value -= mean;
        let squared = *value * *value;
        sum = ordered_add_f64(sum, squared as f64);
    }
    sum / width as f64
}

/// F32 quick GELU on a source row, including its distinct vector division and
/// scalar reciprocal-multiply rounding. Half/BF16 activation kernels do not
/// use this F32 row policy.
pub fn gelu_quick_f32_in_place(values: &mut [f32]) {
    gelu_quick_f32_with_policy(values, native_math_policy());
}

pub fn gelu_quick_f32_with_policy(values: &mut [f32], policy: MathPolicy) {
    let prefix = if policy == MathPolicy::Sse2 {
        values.len() / 4 * 4
    } else {
        0
    };
    let (vector, tail) = values.split_at_mut(prefix);
    for group in vector.as_chunks_mut::<4>().0 {
        let exponent = sse2_exp_f32x4(group.map(|value| value * -1.702_f32));
        for (value, exp) in group.iter_mut().zip(exponent) {
            *value /= 1.0 + exp;
        }
    }
    for value in tail {
        *value *= 1.0 / (1.0 + (-1.702_f32 * *value).exp());
    }
}
