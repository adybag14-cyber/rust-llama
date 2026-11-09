//! Opt-in native CUDA acceptance. Requires an actual CUDA device; absence is
//! an error (exit 77), never a CPU fallback. Reference values below use direct
//! arithmetic or frozen original GGML output, not the PTX interpreter.
use llama_device_family::{Family, ProbeStatus, driver::CudaDriver, probe, ptx::*};

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn floats(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect()
}
fn check(
    context: &llama_device_family::driver::CudaContext,
    target: PtxTarget,
    kind: KernelKind,
    mut buffers: Vec<Vec<u8>>,
    expected: &[f32],
    label: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let generated = generate(&KernelSpec { kind }, target)?;
    let inputs = buffers.clone();
    context.run_generated(&generated, &mut buffers)?;
    let output = generated.output_buffer();
    for (index, buffer) in buffers.iter().enumerate() {
        if index != output && buffer != &inputs[index] {
            return Err(format!("{label}: modified input {index}").into());
        }
    }
    let actual = floats(&buffers[output]);
    if actual.len() != expected.len() {
        return Err(format!("{label}: output size differs").into());
    }
    let before = floats(&inputs[output]);
    let mut maximum = 0.0f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if before[index].to_bits() == expected.to_bits() && actual.to_bits() != expected.to_bits() {
            return Err(format!("{label}: untouched output padding changed at {index}").into());
        }
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        if !actual.is_finite() || error > 2e-6 * (1.0 + expected.abs()) {
            return Err(format!(
                "{label}: index {index}, actual={actual}, reference={expected}, error={error}"
            )
            .into());
        }
    }
    Ok(maximum)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let report = probe(Family::Cuda);
    println!("device_report={report:?}");
    let context = CudaDriver::load()?.context(0)?;
    let packed = include_bytes!("../fixtures/q4_0_reference_input.bin");
    let decoded = floats(include_bytes!("../fixtures/q4_0_reference_f32.bin"));
    let mut checks = 0usize;
    let mut maximum = 0.0f32;
    let capability = report
        .devices
        .first()
        .and_then(|device| device.compute_capability)
        .ok_or("CUDA device did not report compute capability")?;
    if capability < (6, 0) {
        return Err(
            "this scalar PTX implementation requires compute capability 6.0 or later".into(),
        );
    }
    let mut targets = vec![PtxTarget::default()];
    if capability >= (8, 9) {
        targets.push(PtxTarget {
            sm: 89,
            isa_major: 7,
            isa_minor: 8,
        });
    }
    for target in targets {
        // Multiple workgroups, noncontiguous inputs, broadcast rhs, untouched
        // output gaps and allocation tail. All values are exact binary fractions.
        let count = 259u32;
        let left: Vec<f32> = (0..(count - 1) * 2 + 1)
            .map(|i| (i as i32 % 29 - 14) as f32 * 0.125)
            .collect();
        let mut expected = vec![-911.0f32; ((count - 1) * 3 + 7) as usize];
        for index in 0..count as usize {
            expected[index * 3] = left[index * 2] - 0.5;
        }
        maximum = maximum.max(check(
            &context,
            target,
            KernelKind::AddF32 {
                elements: count,
                lhs_stride: 2,
                rhs_stride: 0,
                output_stride: 3,
            },
            vec![
                bytes(&left),
                bytes(&[-0.5]),
                bytes(&vec![-911.0; expected.len()]),
            ],
            &expected,
            "add-tail-padding",
        )?);
        checks += 1;

        let mut expected_scale = vec![-811.0f32; ((count - 1) * 2 + 6) as usize];
        for index in 0..count as usize {
            expected_scale[index * 2] = left[index * 2] * -0.25;
        }
        maximum = maximum.max(check(
            &context,
            target,
            KernelKind::ScaleF32 {
                elements: count,
                input_stride: 2,
                output_stride: 2,
                scale: -0.25,
            },
            vec![bytes(&left), bytes(&vec![-811.0; expected_scale.len()])],
            &expected_scale,
            "scale-tail-padding",
        )?);
        checks += 1;

        let sum = (0..count as usize).map(|i| left[i * 2] as f64).sum::<f64>() as f32;
        maximum = maximum.max(check(
            &context,
            target,
            KernelKind::SumF32 {
                elements: count,
                input_stride: 2,
            },
            vec![bytes(&left), bytes(&[-711.0])],
            &[sum],
            "sum-strided",
        )?);
        checks += 1;

        let rows = 259u32;
        let cols = 17u32;
        let row_stride = 39u32;
        let matrix: Vec<f32> = (0..((rows - 1) * row_stride + (cols - 1) * 2 + 1))
            .map(|i| (i as i32 % 37 - 18) as f32 * 0.0625)
            .collect();
        let vector: Vec<f32> = (0..(cols - 1) * 3 + 1)
            .map(|i| (i as i32 % 13 - 6) as f32 * 0.125)
            .collect();
        let mut expected_matvec = vec![-611.0f32; ((rows - 1) * 2 + 5) as usize];
        for row in 0..rows as usize {
            expected_matvec[row * 2] = (0..cols as usize)
                .map(|col| {
                    matrix[row * row_stride as usize + col * 2] as f64 * vector[col * 3] as f64
                })
                .sum::<f64>() as f32;
        }
        maximum = maximum.max(check(
            &context,
            target,
            KernelKind::MatVecF32 {
                rows,
                columns: cols,
                row_stride,
                column_stride: 2,
                vector_stride: 3,
                output_stride: 2,
            },
            vec![
                bytes(&matrix),
                bytes(&vector),
                bytes(&vec![-611.0; expected_matvec.len()]),
            ],
            &expected_matvec,
            "matvec-strides-tail-padding",
        )?);
        checks += 1;

        // Every original native dequantized scalar is independently exercised
        // by a basis-vector dot product through the real CUDA driver.
        for column in 0..64 {
            let mut vector = vec![0.0f32; 64];
            vector[column] = 1.0;
            let expected: Vec<f32> = (0..4).map(|row| decoded[row * 64 + column]).collect();
            maximum = maximum.max(check(
                &context,
                target,
                KernelKind::Q4MatVec {
                    rows: 4,
                    columns: 64,
                    row_stride_bytes: 36,
                    block_stride_bytes: 18,
                    vector_stride: 1,
                },
                vec![packed.to_vec(), bytes(&vector), vec![0; 16]],
                &expected,
                "q4-frozen-native-basis",
            )?);
            checks += 1;
        }
        let mut padded = vec![0xabu8; 4 * 46];
        for row in 0..4 {
            for block in 0..2 {
                padded[row * 46 + block * 22..row * 46 + block * 22 + 18]
                    .copy_from_slice(&packed[row * 36 + block * 18..row * 36 + block * 18 + 18]);
            }
        }
        let vector: Vec<f32> = (0..127).map(|i| (i % 17 - 8) as f32 * 0.0625).collect();
        let expected: Vec<f32> = (0..4)
            .map(|row| {
                (0..64)
                    .map(|col| decoded[row * 64 + col] as f64 * vector[col * 2] as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        maximum = maximum.max(check(
            &context,
            target,
            KernelKind::Q4MatVec {
                rows: 4,
                columns: 64,
                row_stride_bytes: 46,
                block_stride_bytes: 22,
                vector_stride: 2,
            },
            vec![padded, bytes(&vector), vec![0; 16]],
            &expected,
            "q4-padded-blocks-rows",
        )?);
        checks += 1;
        println!(
            "target_sm={} isa={}.{} passed=69 native_reference_values=256",
            target.sm, target.isa_major, target.isa_minor
        );
    }
    // Wrong buffer length is rejected before a device dispatch and preserves
    // every caller buffer; it must not turn into an out-of-bounds GPU access.
    let generated = generate(
        &KernelSpec {
            kind: KernelKind::MatVecF32 {
                rows: 2,
                columns: 3,
                row_stride: 3,
                column_stride: 1,
                vector_stride: 1,
                output_stride: 1,
            },
        },
        PtxTarget::default(),
    )?;
    let mut invalid = vec![vec![0; 4], vec![0; 12], bytes(&[-1.0, -2.0])];
    let before = invalid.clone();
    if context.run_generated(&generated, &mut invalid).is_ok() || invalid != before {
        return Err("invalid-buffer rejection failed".into());
    }
    println!(
        "CUDA_CONFORMANCE_PASS dispatches={checks} max_abs_error={maximum:e} invalid_buffer_rollback=true"
    );
    Ok(())
}
fn main() {
    let report = probe(Family::Cuda);
    if report.status != ProbeStatus::Available {
        eprintln!("CUDA_CONFORMANCE_UNAVAILABLE: {}", report.detail);
        std::process::exit(if report.status == ProbeStatus::Unavailable {
            77
        } else {
            1
        });
    }
    if let Err(error) = run() {
        eprintln!("CUDA_CONFORMANCE_FAILED: {error}");
        std::process::exit(1);
    }
}
