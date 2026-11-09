//! Repeatable host microbenchmarks. These do not establish inference speed.
use std::time::Instant;
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    pub samples: usize,
    pub minimum_ns: f64,
    pub median_ns: f64,
    pub p90_ns: f64,
    pub mean_ns: f64,
    pub standard_deviation_ns: f64,
}
impl Distribution {
    pub fn from_nanoseconds(samples: &[u128]) -> Result<Self, &'static str> {
        if samples.is_empty() {
            return Err("benchmark requires at least one sample");
        }
        if samples.len() > 100_000 {
            return Err("too many benchmark samples");
        }
        let mut values: Vec<f64> = samples.iter().map(|n| *n as f64).collect();
        values.sort_by(f64::total_cmp);
        let len = values.len();
        let mean = values.iter().sum::<f64>() / len as f64;
        let variance = values.iter().map(|n| (n - mean).powi(2)).sum::<f64>() / len as f64;
        let median = if len.is_multiple_of(2) {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        };
        let p90 = values[(len * 9).div_ceil(10) - 1];
        Ok(Self {
            samples: len,
            minimum_ns: values[0],
            median_ns: median,
            p90_ns: p90,
            mean_ns: mean,
            standard_deviation_ns: variance.sqrt(),
        })
    }
}
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub warmups: usize,
    pub measurements: Vec<u128>,
    pub distribution: Distribution,
}
pub fn measure<T>(
    warmups: usize,
    runs: usize,
    mut operation: impl FnMut() -> T,
) -> Result<BenchmarkReport, &'static str> {
    if warmups > 1000 || runs == 0 || runs > 100_000 {
        return Err("benchmark sample budget outside supported range");
    }
    for _ in 0..warmups {
        std::hint::black_box(operation());
    }
    let mut measurements = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        std::hint::black_box(operation());
        measurements.push(start.elapsed().as_nanos());
    }
    Ok(BenchmarkReport {
        warmups,
        distribution: Distribution::from_nanoseconds(&measurements)?,
        measurements,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn statistics_use_sorted_samples_and_population_variance() {
        let r = Distribution::from_nanoseconds(&[40, 10, 30, 20]).unwrap();
        assert_eq!(r.minimum_ns, 10.0);
        assert_eq!(r.median_ns, 25.0);
        assert_eq!(r.p90_ns, 40.0);
        assert!((r.standard_deviation_ns - 125.0_f64.sqrt()).abs() < 1e-10);
    }
    #[test]
    fn warmups_are_excluded() {
        let mut n = 0;
        let r = measure(2, 3, || {
            n += 1;
            n
        })
        .unwrap();
        assert_eq!(n, 5);
        assert_eq!(r.measurements.len(), 3);
        assert_eq!(r.distribution.samples, 3);
        assert!(measure(0, 0, || 0).is_err());
    }
}
