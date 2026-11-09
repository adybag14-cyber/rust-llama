//! Comparisons use explicit numeric budgets; NaN never accidentally passes.
use std::fmt;
#[derive(Debug, Clone, PartialEq)]
pub struct Difference {
    pub index: usize,
    pub expected: String,
    pub actual: String,
    pub reason: &'static str,
}
impl fmt::Display for Difference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at index {}: expected {}, observed {}",
            self.reason, self.index, self.expected, self.actual
        )
    }
}
impl std::error::Error for Difference {}
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
}
impl Tolerance {
    pub fn new(absolute: f64, relative: f64) -> Result<Self, &'static str> {
        if !absolute.is_finite() || !relative.is_finite() || absolute < 0.0 || relative < 0.0 {
            return Err("tolerances must be finite and nonnegative");
        }
        Ok(Self { absolute, relative })
    }
}
pub fn exact(expected: &[u8], actual: &[u8]) -> Result<(), Difference> {
    if let Some(i) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        return Err(Difference {
            index: i,
            expected: format!("0x{:02x}", expected[i]),
            actual: format!("0x{:02x}", actual[i]),
            reason: "byte mismatch",
        });
    }
    if expected.len() != actual.len() {
        return Err(Difference {
            index: expected.len().min(actual.len()),
            expected: expected.len().to_string(),
            actual: actual.len().to_string(),
            reason: "length mismatch",
        });
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, Default)]
pub struct NumericReport {
    pub compared: usize,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
}
pub fn floats(
    expected: &[f64],
    actual: &[f64],
    tolerance: Tolerance,
) -> Result<NumericReport, Difference> {
    if Tolerance::new(tolerance.absolute, tolerance.relative).is_err() {
        return Err(Difference {
            index: 0,
            expected: "valid tolerance".into(),
            actual: format!("{tolerance:?}"),
            reason: "invalid budget",
        });
    }
    if expected.len() != actual.len() {
        return Err(Difference {
            index: expected.len().min(actual.len()),
            expected: expected.len().to_string(),
            actual: actual.len().to_string(),
            reason: "length mismatch",
        });
    }
    let mut report = NumericReport::default();
    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        if e.is_nan() || a.is_nan() {
            return Err(Difference {
                index: i,
                expected: e.to_string(),
                actual: a.to_string(),
                reason: "NaN rejected",
            });
        }
        if e == a {
            report.compared += 1;
            continue;
        }
        if !e.is_finite() || !a.is_finite() {
            return Err(Difference {
                index: i,
                expected: e.to_string(),
                actual: a.to_string(),
                reason: "nonfinite mismatch",
            });
        }
        let error = (e - a).abs();
        let bound = tolerance.absolute + tolerance.relative * e.abs();
        if !bound.is_finite() {
            return Err(Difference {
                index: i,
                expected: "finite error budget".into(),
                actual: bound.to_string(),
                reason: "numeric budget overflow",
            });
        }
        if !error.is_finite() || error > bound {
            return Err(Difference {
                index: i,
                expected: e.to_string(),
                actual: a.to_string(),
                reason: "numeric budget exceeded",
            });
        }
        report.max_absolute_error = report.max_absolute_error.max(error);
        report.max_relative_error = report.max_relative_error.max(if e == 0.0 {
            f64::INFINITY
        } else {
            error / e.abs()
        });
        report.compared += 1;
    }
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_mutation_and_truncation() {
        assert!(exact(b"token\0\xff", b"token\0\xfe").is_err());
        assert!(exact(b"token", b"toke").is_err());
        assert!(exact(b"A\xc2\xa3", b"A\xc2\xa3").is_ok());
    }
    #[test]
    fn numeric_budget_edges() {
        let t = Tolerance::new(0.001, 0.01).unwrap();
        assert!(floats(&[100.0, 0.0], &[101.0, 0.001], t).is_ok());
        assert!(floats(&[100.0], &[101.01], t).is_err());
        assert!(floats(&[0.0], &[f64::NAN], t).is_err());
        assert!(floats(&[f64::INFINITY], &[f64::INFINITY], t).is_ok());
        assert!(floats(&[f64::INFINITY], &[f64::NEG_INFINITY], t).is_err());
        assert!(Tolerance::new(-1.0, 0.0).is_err());
    }
}
