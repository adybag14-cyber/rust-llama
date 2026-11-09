//! Requires a real CUDA driver/device. Exit 77 means unavailable, not passed.
use llama_device_family::{Family, ProbeStatus, driver::CudaDriver, probe, ptx::*};
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaDriver::load()?.context(0)?;
    let mut pinned = context.allocate_pinned(16)?;
    pinned.as_mut_slice().copy_from_slice(b"Rust CUDA probe!");
    let mut device = context.allocate(16)?;
    device.copy_from(0, pinned.as_slice())?;
    let mut roundtrip = vec![0; 16];
    device.copy_to(0, &mut roundtrip)?;
    if roundtrip != pinned.as_slice() {
        return Err("pinned memory roundtrip mismatch".into());
    }
    let mut stream = context.stream()?;
    let mut event = context.event()?;
    event.record(&mut stream)?;
    event.synchronize()?;
    let cases = vec![
        (
            KernelKind::AddF32 {
                elements: 3,
                lhs_stride: 2,
                rhs_stride: 0,
                output_stride: 2,
            },
            vec![
                bytes(&[1., 777., 2., 777., 3.]),
                bytes(&[10.]),
                bytes(&[0., 777., 0., 777., 0.]),
            ],
        ),
        (
            KernelKind::ScaleF32 {
                elements: 3,
                input_stride: 1,
                output_stride: 1,
                scale: -0.25,
            },
            vec![bytes(&[4., -8., 12.]), vec![0; 12]],
        ),
        (
            KernelKind::SumF32 {
                elements: 3,
                input_stride: 1,
            },
            vec![bytes(&[1., 2., 3.]), vec![0; 4]],
        ),
        (
            KernelKind::Q4MatVec {
                rows: 1,
                columns: 32,
                row_stride_bytes: 18,
                block_stride_bytes: 18,
                vector_stride: 1,
            },
            {
                let mut q = vec![0x99; 18];
                q[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                vec![q, bytes(&[1.; 32]), vec![0; 4]]
            },
        ),
    ];
    for (kind, mut buffers) in cases {
        let kernel = generate(&KernelSpec { kind: kind.clone() }, PtxTarget::default())?;
        let mut interpreted = buffers.clone();
        kernel.interpret(&mut interpreted, 100_000)?;
        context.run_generated(&kernel, &mut buffers)?;
        let out = kernel.output_buffer();
        for (actual, expected) in values(&buffers[out]).iter().zip(values(&interpreted[out])) {
            if !actual.is_finite() || (*actual - expected).abs() > 1e-5 * (1.0 + expected.abs()) {
                return Err(format!("device mismatch: {actual} vs {expected}").into());
            }
        }
        println!("CUDA execution passed: {kind:?}");
    }
    Ok(())
}
fn main() {
    let report = probe(Family::Cuda);
    if report.status != ProbeStatus::Available {
        eprintln!("CUDA EXECUTION NOT RUN: {}", report.detail);
        std::process::exit(if report.status == ProbeStatus::Unavailable {
            77
        } else {
            1
        });
    }
    if let Err(error) = run() {
        eprintln!("CUDA EXECUTION FAILED: {error}");
        std::process::exit(1);
    }
}
