// Copyright Naoki Shibata and contributors 2010 - 2024.
// Distributed under the Boost Software License, Version 1.0.
// Scalar Rust transcription of the FMA/AVX512 single-precision xpowf path
// in SLEEF commit 5a1d179df9cf652951b59010a2d2075372d67f68.
// This is reference-profile arithmetic, not a general public powf API.
// Coefficient literals and rounding points are retained from the pinned source.
#![allow(clippy::excessive_precision, clippy::approx_constant)]
#[derive(Clone, Copy)]
struct Pair {
    hi: f32,
    lo: f32,
}
impl Pair {
    fn new(hi: f32, lo: f32) -> Self {
        Self { hi, lo }
    }
    fn norm(self) -> Self {
        let s = self.hi + self.lo;
        Self::new(s, (self.hi - s) + self.lo)
    }
    fn scale(self, y: f32) -> Self {
        Self::new(self.hi * y, self.lo * y)
    }
    fn add(self, y: Self) -> Self {
        let s = self.hi + y.hi;
        Self::new(s, ((self.hi - s) + y.hi + self.lo) + y.lo)
    }
    fn add2(self, y: Self) -> Self {
        let s = self.hi + y.hi;
        let v = s - self.hi;
        let t = (self.hi - (s - v)) + (y.hi - v);
        Self::new(s, t + (self.lo + y.lo))
    }
    fn add_scalar2(self, y: f32) -> Self {
        let s = self.hi + y;
        let v = s - self.hi;
        let t = (self.hi - (s - v)) + (y - v);
        Self::new(s, t + self.lo)
    }
    fn scalar_add2(x: f32, y: f32) -> Self {
        let s = x + y;
        let v = s - x;
        Self::new(s, (x - (s - v)) + (y - v))
    }
    fn scalar_add(x: f32, y: Self) -> Self {
        let s = x + y.hi;
        Self::new(s, ((x - s) + y.hi) + y.lo)
    }
    fn mul(self, y: Self) -> Self {
        let s = self.hi * y.hi;
        Self::new(
            s,
            self.hi
                .mul_add(y.lo, self.lo.mul_add(y.hi, self.hi.mul_add(y.hi, -s))),
        )
    }
    fn mul_scalar(self, y: f32) -> Self {
        let s = self.hi * y;
        Self::new(s, self.lo.mul_add(y, self.hi.mul_add(y, -s)))
    }
    fn square(self) -> Self {
        let s = self.hi * self.hi;
        Self::new(
            s,
            (self.hi + self.hi).mul_add(self.lo, self.hi.mul_add(self.hi, -s)),
        )
    }
    fn div(self, d: Self) -> Self {
        let t = 1.0 / d.hi;
        let s = self.hi * t;
        let u = t.mul_add(self.hi, -s);
        let v = (-d.lo).mul_add(t, (-d.hi).mul_add(t, 1.0));
        Self::new(s, s.mul_add(v, self.lo.mul_add(t, u)))
    }
}
fn exponent(value: f32) -> i32 {
    let bits = value.to_bits() & 0x7fff_ffff;
    let e = (bits >> 23) as i32;
    if e == 0 {
        if bits == 0 {
            i32::MIN
        } else {
            31 - bits.leading_zeros() as i32 - 149
        }
    } else if e == 255 {
        128
    } else {
        e - 127
    }
}
fn mantissa(value: f32) -> f32 {
    let bits = value.to_bits() & 0x7fff_ffff;
    let normalized = if bits >> 23 == 0 {
        let shift = bits.leading_zeros() - 8;
        (bits << shift) & 0x007f_ffff
    } else {
        bits & 0x007f_ffff
    };
    let base = f32::from_bits(normalized | 0x3f80_0000);
    if base >= 1.5 { base * 0.5 } else { base }
}
fn logarithm(value: f32) -> Pair {
    let e = exponent(value * (1.0f32 / 0.75f32)) as f32;
    let m = mantissa(value);
    let x = Pair::scalar_add2(-1.0, m).div(Pair::scalar_add2(1.0, m));
    let x2 = x.square();
    let mut t = 0.240320354700088500976562f32;
    t = t.mul_add(x2.hi, 0.285112679004669189453125f32);
    t = t.mul_add(x2.hi, 0.400007992982864379882812f32);
    let c = Pair::new(
        0.66666662693023681640625f32,
        3.69183861259614332084311e-9f32,
    );
    let s = Pair::new(0.69314718246459960938f32, -1.904654323148236017e-9f32).mul_scalar(e);
    let s = s.add(x.scale(2.0));
    s.add(x2.mul(x).mul(x2.mul_scalar(t).add2(c)))
}
fn scale_exponent(mut x: f32, mut q: i32) -> f32 {
    let sign = q >> 31;
    let mut m = (((sign + q) >> 6) - sign) << 4;
    q -= m << 2;
    m = (m + 127).clamp(0, 255);
    let scale = f32::from_bits((m as u32) << 23);
    x = (((x * scale) * scale) * scale) * scale;
    x * f32::from_bits(((q + 127) as u32) << 23)
}
fn exponential(d: Pair) -> f32 {
    let u = (d.hi + d.lo) * 1.442695040888963407359924681001892137426645954152985934135449406931f32;
    let q = u.round_ties_even() as i32;
    let s = d.add_scalar2(q as f32 * -0.693145751953125f32);
    let s = s.add_scalar2(q as f32 * -1.428606765330187045e-6f32).norm();
    let mut u = 0.00136324646882712841033936f32;
    u = u.mul_add(s.hi, 0.00836596917361021041870117f32);
    u = u.mul_add(s.hi, 0.0416710823774337768554688f32);
    u = u.mul_add(s.hi, 0.166665524244308471679688f32);
    u = u.mul_add(s.hi, 0.499999850988388061523438f32);
    let t = s.add(s.square().mul_scalar(u));
    let t = Pair::scalar_add(1.0, t);
    if d.hi < -104.0 {
        0.0
    } else {
        scale_exponent(t.hi + t.lo, q)
    }
}
pub(super) fn vector_pow(x: f32, y: f32, domain_nan: u32) -> f32 {
    // The converter's exponent is finite and in [0,1]; its finite F64 base can narrow to infinity.
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f32::from_bits(u32::MAX);
    }
    let integer = y.trunc() == y;
    let odd = integer && y.abs() < 16777216.0 && (y as i32 & 1) != 0;
    if x == 0.0 || x.is_infinite() {
        let zero = y.is_sign_negative() ^ (x == 0.0);
        let value = if zero { 0.0 } else { f32::INFINITY };
        return if odd { value.copysign(x) } else { value };
    }
    if x < 0.0 && !integer {
        return f32::from_bits(domain_nan);
    }
    let mut r = exponential(logarithm(x.abs()).mul_scalar(y));
    if r.is_nan() {
        r = f32::INFINITY;
    }
    if x < 0.0 && odd { -r } else { r }
}
