//! Opt-in OpenCL device acceptance for all nine implemented built-ins.
//! Expected results use independent formulas and frozen native GGML data.
//! The IR interpreter is deliberately not used as the numerical oracle.
use llama_kernel_ir::{BufferData, kernels::*};
use llama_portable_gpu::opencl::{OpenCl, Session};
use std::collections::BTreeMap;

type Buffers = BTreeMap<String, BufferData>;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn check(
    session: &Session<'_>,
    plan: DispatchPlan,
    mut buffers: Buffers,
    expected: &[f32],
    label: &str,
) -> Result<f32> {
    let initial = buffers.clone();
    let execution = session.execute(&plan, &mut buffers)?;
    let BufferData::F32(actual) = &buffers["dst"] else {
        return Err("wrong output type".into());
    };
    let BufferData::F32(before) = &initial["dst"] else {
        return Err("wrong initial type".into());
    };
    if actual.len() != expected.len() {
        return Err(format!("{label}: wrong output length").into());
    }
    let mut max_error = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        if !actual.is_finite() || error > 3e-6 * (1.0 + expected.abs()) {
            return Err(
                format!("{label}: index {index}: actual {actual} reference {expected}").into(),
            );
        }
        if before[index].to_bits() == expected.to_bits() && actual.to_bits() != expected.to_bits() {
            return Err(format!("{label}: untouched padding changed at {index}").into());
        }
    }
    for (name, value) in initial {
        if name != "dst" && buffers[&name] != value {
            return Err(format!("{label}: input buffer {name} changed").into());
        }
    }
    println!(
        "OPENCL_CASE_PASS label={label} kernel={} device={} max_abs_error={max_error:e} source_sha256={}",
        execution.kernel, execution.device, execution.source_sha256
    );
    Ok(max_error)
}
fn on_device(session: &Session<'_>) -> Result<(usize, f32)> {
    let mut checks = 0;
    let mut maximum = 0.0f32;
    // Noncontiguous operands, broadcast, partial final workgroup and guard
    // values before, between and after the output locations.
    let n = 259u32;
    let lhs_layout = VectorLayout {
        length: n,
        offset: 1,
        stride: 2,
    };
    let rhs_layout = VectorLayout {
        length: n,
        offset: 2,
        stride: 0,
    };
    let out_layout = VectorLayout {
        length: n,
        offset: 3,
        stride: 3,
    };
    let mut lhs = vec![991.0f32; lhs_layout.storage_elements()? as usize];
    for i in 0..n as usize {
        lhs[1 + i * 2] = (i as i32 % 23 - 11) as f32 * 0.125;
    }
    let rhs = vec![881.0, 881.0, -0.5];
    for kind in [
        Pointwise::Add,
        Pointwise::Mul,
        Pointwise::Silu,
        Pointwise::Scale,
    ] {
        let mut expected = vec![-771.0f32; out_layout.storage_elements()? as usize + 5];
        for i in 0..n as usize {
            let x = lhs[1 + i * 2] as f64;
            expected[3 + i * 3] = match kind {
                Pointwise::Add => x - 0.5,
                Pointwise::Mul => x * -0.5,
                Pointwise::Silu => x / (1.0 + (-x).exp()),
                Pointwise::Scale => x * 0.25,
            } as f32;
        }
        let binary = matches!(kind, Pointwise::Add | Pointwise::Mul).then_some(rhs_layout);
        let plan = DispatchPlan::pointwise(kind, lhs_layout, binary, out_layout, 0.25)?;
        let buffers = Buffers::from([
            ("lhs".into(), BufferData::F32(lhs.clone())),
            ("rhs".into(), BufferData::F32(rhs.clone())),
            ("dst".into(), BufferData::F32(vec![-771.0; expected.len()])),
        ]);
        maximum = maximum.max(check(
            session,
            plan,
            buffers,
            &expected,
            &format!("{kind:?}-strided-tail"),
        )?);
        checks += 1;
    }
    let layout = MatrixLayout {
        rows: 3,
        cols: 13,
        offset: 2,
        row_stride: 43,
        col_stride: 3,
    };
    let mut source = vec![661.0f32; layout.storage_elements()? as usize];
    for row in 0..3 {
        for col in 0..13 {
            source[2 + row * 43 + col * 3] = (row as f32 - 1.0) + (col as f32 - 6.0) * 0.125;
        }
    }
    let mut sums = vec![];
    let mut rms = vec![];
    let mut probabilities = vec![];
    for row in 0..3 {
        let values: Vec<f64> = (0..13)
            .map(|c| source[2 + row * 43 + c * 3] as f64)
            .collect();
        sums.push(values.iter().sum::<f64>() as f32);
        let inverse = (values.iter().map(|x| x * x).sum::<f64>() / 13.0 + 1e-5)
            .sqrt()
            .recip();
        rms.extend(values.iter().map(|x| (x * inverse) as f32));
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials: Vec<f64> = values.iter().map(|x| (x - maximum).exp()).collect();
        let denominator = exponentials.iter().sum::<f64>();
        probabilities.extend(exponentials.iter().map(|x| (x / denominator) as f32));
    }
    for (plan, expected, label) in [
        (DispatchPlan::row_sum(layout)?, sums, "row-sum"),
        (
            DispatchPlan::rms_norm(layout, 1e-5)?,
            rms,
            "rms-normalization",
        ),
        (
            DispatchPlan::softmax(layout, 1.0)?,
            probabilities,
            "softmax",
        ),
    ] {
        maximum = maximum.max(check(
            session,
            plan,
            Buffers::from([
                ("src".into(), BufferData::F32(source.clone())),
                ("dst".into(), BufferData::F32(vec![-551.0; expected.len()])),
            ]),
            &expected,
            label,
        )?);
        checks += 1;
    }
    let a_layout = MatrixLayout {
        rows: 3,
        cols: 5,
        offset: 1,
        row_stride: 12,
        col_stride: 2,
    };
    let b_layout = MatrixLayout {
        rows: 5,
        cols: 7,
        offset: 3,
        row_stride: 1,
        col_stride: 6,
    };
    let a: Vec<f32> = (0..a_layout.storage_elements()?)
        .map(|i| (i as f32 - 8.0) * 0.125)
        .collect();
    let b: Vec<f32> = (0..b_layout.storage_elements()?)
        .map(|i| (i as f32 + 1.0) * 0.03125)
        .collect();
    let mut expected = vec![];
    for row in 0..3 {
        for col in 0..7 {
            expected.push(
                (0..5)
                    .map(|k| a[1 + row * 12 + k * 2] as f64 * b[3 + k + col * 6] as f64)
                    .sum::<f64>() as f32,
            );
        }
    }
    maximum = maximum.max(check(
        session,
        DispatchPlan::matmul(a_layout, b_layout)?,
        Buffers::from([
            ("lhs".into(), BufferData::F32(a)),
            ("rhs".into(), BufferData::F32(b)),
            ("dst".into(), BufferData::F32(vec![-441.0; expected.len()])),
        ]),
        &expected,
        "matmul-transposed-strides",
    )?);
    checks += 1;

    let packed = include_bytes!("../../device-family/fixtures/q4_0_reference_input.bin");
    let decoded: Vec<f32> = include_bytes!("../../device-family/fixtures/q4_0_reference_f32.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut padded = vec![0xabu8; 3 + 3 * 41 + 36];
    for row in 0..4 {
        padded[3 + row * 41..3 + row * 41 + 36].copy_from_slice(&packed[row * 36..row * 36 + 36]);
    }
    let x: Vec<f32> = (0..128).map(|i| (i % 17 - 8) as f32 * 0.0625).collect();
    let expected: Vec<f32> = (0..4)
        .map(|row| {
            (0..64)
                .map(|col| decoded[row * 64 + col] as f64 * x[1 + col * 2] as f64)
                .sum::<f64>() as f32
        })
        .collect();
    let plan = DispatchPlan::q4_0_matvec(
        4,
        64,
        3,
        41,
        VectorLayout {
            length: 64,
            offset: 1,
            stride: 2,
        },
    )?;
    maximum = maximum.max(check(
        session,
        plan,
        Buffers::from([
            ("packed".into(), BufferData::U32(bytes_to_words(&padded))),
            ("vector".into(), BufferData::F32(x)),
            ("dst".into(), BufferData::F32(vec![-331.0; 4])),
        ]),
        &expected,
        "q4-native-fixture-unaligned-padded",
    )?);
    checks += 1;

    // Stable large-logit acceptance cannot be inferred from ordinary rows.
    let logits = vec![10_000.0, 10_001.0, 9999.0, -10_000.0, -9999.0, -10_001.0];
    let denominator = 1.0 + (-1f64).exp() + (-2f64).exp();
    let row = [
        ((-1f64).exp() / denominator) as f32,
        (1.0 / denominator) as f32,
        ((-2f64).exp() / denominator) as f32,
    ];
    maximum = maximum.max(check(
        session,
        DispatchPlan::softmax(MatrixLayout::contiguous(2, 3), 1.0)?,
        Buffers::from([
            ("src".into(), BufferData::F32(logits)),
            ("dst".into(), BufferData::F32(vec![-221.0; 6])),
        ]),
        &[row, row].concat(),
        "softmax-large-finite-logits",
    )?);
    checks += 1;

    let invalid_plan = DispatchPlan::row_sum(MatrixLayout::contiguous(2, 3))?;
    let mut invalid = Buffers::from([
        ("src".into(), BufferData::F32(vec![1.0; 2])),
        ("dst".into(), BufferData::F32(vec![-111.0; 2])),
    ]);
    let before = invalid.clone();
    if session.execute(&invalid_plan, &mut invalid).is_ok() || invalid != before {
        return Err("invalid-buffer rejection did not preserve buffers".into());
    }
    Ok((checks, maximum))
}
fn run() -> Result<()> {
    let opencl = OpenCl::open()?;
    let devices = opencl.devices();
    if devices.is_empty() {
        return Err("OpenCL reported no devices; no execution performed".into());
    }
    for (index, device) in devices.iter().enumerate() {
        println!("OPENCL_DEVICE index={index} info={device:?}");
        let session = opencl.session(index)?;
        let (checks, error) = on_device(&session)?;
        println!(
            "OPENCL_CONFORMANCE_PASS device={} dispatches={checks} max_abs_error={error:e} invalid_buffer_rollback=true",
            device.name
        );
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("OPENCL_CONFORMANCE_FAILED: {error}");
        std::process::exit(1);
    }
}
