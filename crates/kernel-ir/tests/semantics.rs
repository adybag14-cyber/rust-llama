use llama_kernel_ir::kernels::*;
use llama_kernel_ir::*;
use std::collections::BTreeMap;

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "element {index}: {a} != {b}"
        );
    }
}
fn output(buffers: &BTreeMap<String, BufferData>) -> &[f32] {
    match &buffers["dst"] {
        BufferData::F32(v) => v,
        _ => panic!("wrong output type"),
    }
}
#[test]
fn strided_pointwise_preserves_gaps_and_tail() {
    let kernel = pointwise(
        Pointwise::Add,
        VectorLayout {
            length: 37,
            offset: 1,
            stride: 2,
        },
        Some(VectorLayout {
            length: 37,
            offset: 2,
            stride: 3,
        }),
        VectorLayout {
            length: 37,
            offset: 3,
            stride: 2,
        },
        1.0,
    )
    .unwrap();
    let mut lhs = vec![999.0; 74];
    let mut rhs = vec![999.0; 111];
    let mut expected = vec![-9.0; 77];
    for i in 0..37 {
        lhs[1 + i * 2] = i as f32;
        rhs[2 + i * 3] = (i as f32) * 0.5;
        expected[3 + i * 2] = (i as f32) * 1.5;
    }
    let mut buffers = BTreeMap::from([
        ("lhs".into(), BufferData::F32(lhs)),
        ("rhs".into(), BufferData::F32(rhs)),
        ("dst".into(), BufferData::F32(vec![-9.0; 77])),
    ]);
    let report = execute(&kernel, &mut buffers, 10000).unwrap();
    assert_eq!(report.writes, 37);
    assert_eq!(kernel.workgroups().unwrap(), 1);
    assert_eq!(output(&buffers), expected);
}
#[test]
fn nonlinear_and_scale_kernels_match_direct_formulas() {
    let values = vec![-8.0, -1.0, 0.0, 1.0, 8.0];
    for kind in [Pointwise::Silu, Pointwise::Scale, Pointwise::Mul] {
        let binary = if kind == Pointwise::Mul {
            Some(VectorLayout::contiguous(5))
        } else {
            None
        };
        let kernel = pointwise(
            kind,
            VectorLayout::contiguous(5),
            binary,
            VectorLayout::contiguous(5),
            0.25,
        )
        .unwrap();
        let mut buffers = BTreeMap::from([
            ("lhs".into(), BufferData::F32(values.clone())),
            ("rhs".into(), BufferData::F32(values.clone())),
            ("dst".into(), BufferData::F32(vec![0.0; 5])),
        ]);
        execute(&kernel, &mut buffers, 10000).unwrap();
        let expected: Vec<f32> = values
            .iter()
            .map(|&x| match kind {
                Pointwise::Silu => x / (1.0 + (-x).exp()),
                Pointwise::Scale => x * 0.25,
                _ => x * x,
            })
            .collect();
        close(output(&buffers), &expected, 1e-6);
    }
}
#[test]
fn strided_row_sum_and_rms_match_f64_oracle() {
    let layout = MatrixLayout {
        rows: 3,
        cols: 13,
        offset: 2,
        row_stride: 43,
        col_stride: 3,
    };
    let mut values = vec![100.0; layout.storage_elements().unwrap() as usize];
    for row in 0..3 {
        for col in 0..13 {
            values[2 + row * 43 + col * 3] = (row as f32 - 1.0) + (col as f32 - 6.0) * 0.125;
        }
    }
    let mut sums = vec![];
    let mut normalized = vec![];
    for row in 0..3 {
        let vals: Vec<f64> = (0..13)
            .map(|col| values[2 + row * 43 + col * 3] as f64)
            .collect();
        sums.push(vals.iter().sum::<f64>() as f32);
        let factor = (vals.iter().map(|x| x * x).sum::<f64>() / 13.0 + 1e-5)
            .sqrt()
            .recip();
        normalized.extend(vals.iter().map(|x| (x * factor) as f32));
    }
    for (kernel, expected) in [
        (row_sum(layout).unwrap(), sums),
        (rms_norm(layout, 1e-5).unwrap(), normalized),
    ] {
        let mut buffers = BTreeMap::from([
            ("src".into(), BufferData::F32(values.clone())),
            ("dst".into(), BufferData::F32(vec![0.0; expected.len()])),
        ]);
        execute(&kernel, &mut buffers, 100000).unwrap();
        close(output(&buffers), &expected, 2e-6);
    }
}
#[test]
fn softmax_stable_at_large_finite_logits() {
    let values = vec![10000.0, 10001.0, 9999.0, -10000.0, -9999.0, -10001.0];
    let kernel = softmax(MatrixLayout::contiguous(2, 3), 1.0).unwrap();
    let mut buffers = BTreeMap::from([
        ("src".into(), BufferData::F32(values)),
        ("dst".into(), BufferData::F32(vec![0.0; 6])),
    ]);
    execute(&kernel, &mut buffers, 10000).unwrap();
    let denominator = 1.0 + (-1f64).exp() + (-2f64).exp();
    let expected = [
        ((-1f64).exp() / denominator) as f32,
        (1.0 / denominator) as f32,
        ((-2f64).exp() / denominator) as f32,
    ];
    close(&output(&buffers)[..3], &expected, 1e-6);
    close(&output(&buffers)[3..], &expected, 1e-6);
    for row in output(&buffers).chunks(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
#[test]
fn matmul_transposed_strides_match_scalar_dot_products() {
    let lhs = MatrixLayout {
        rows: 3,
        cols: 5,
        offset: 1,
        row_stride: 12,
        col_stride: 2,
    };
    let rhs = MatrixLayout {
        rows: 5,
        cols: 7,
        offset: 3,
        row_stride: 1,
        col_stride: 6,
    };
    let a: Vec<f32> = (0..lhs.storage_elements().unwrap())
        .map(|i| (i as f32 - 8.0) * 0.1)
        .collect();
    let b: Vec<f32> = (0..rhs.storage_elements().unwrap())
        .map(|i| (i as f32 + 1.0) * 0.025)
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
    let mut buffers = BTreeMap::from([
        ("lhs".into(), BufferData::F32(a)),
        ("rhs".into(), BufferData::F32(b)),
        ("dst".into(), BufferData::F32(vec![0.0; 21])),
    ]);
    execute(&matmul(lhs, rhs).unwrap(), &mut buffers, 100000).unwrap();
    close(output(&buffers), &expected, 2e-6);
}
#[test]
fn q4_exact_nibble_order_unaligned_scale_and_vector_stride() {
    let rows = 3usize;
    let cols = 64usize;
    let mut packed = vec![0xa5u8; 3 + (rows - 1) * 41 + 36];
    let mut x = vec![-999.0; 128];
    for i in 0..cols {
        x[1 + i * 2] = (i as f32 - 30.0) * 0.125;
    }
    let scales = [
        0x3c00u16, 0xb800u16, 0x3400u16, 0x4000u16, 0x0001u16, 0x8000u16,
    ];
    let mut expected = vec![0.0; rows];
    for row in 0..rows {
        for block in 0..2 {
            let base = 3 + row * 41 + block * 18;
            packed[base..base + 2].copy_from_slice(&scales[row * 2 + block].to_le_bytes());
            let scale = half_to_float(scales[row * 2 + block]);
            for j in 0..16 {
                let low = ((j + row + block) % 16) as u8;
                let high = ((15 - j + row) % 16) as u8;
                packed[base + 2 + j] = low | (high << 4);
                expected[row] += scale * (low as i32 - 8) as f32 * x[1 + (block * 32 + j) * 2];
                expected[row] +=
                    scale * (high as i32 - 8) as f32 * x[1 + (block * 32 + j + 16) * 2];
            }
        }
    }
    let kernel = q4_0_matvec(
        rows as u32,
        cols as u32,
        3,
        41,
        VectorLayout {
            length: 64,
            offset: 1,
            stride: 2,
        },
    )
    .unwrap();
    let mut buffers = BTreeMap::from([
        ("packed".into(), BufferData::U32(bytes_to_words(&packed))),
        ("vector".into(), BufferData::F32(x)),
        ("dst".into(), BufferData::F32(vec![0.0; rows])),
    ]);
    execute(&kernel, &mut buffers, 100000).unwrap();
    close(output(&buffers), &expected, 0.0);
}
#[test]
fn half_conversion_exhaustive_matches_integer_normalization() {
    for bits in 0u32..=65535 {
        let sign = (bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 31;
        let mut mantissa = bits & 1023;
        let expected = if exponent == 0 && mantissa != 0 {
            let mut exponent = 113;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 1023) << 13)
        } else if exponent == 0 {
            sign
        } else if exponent == 31 {
            sign | 0x7f800000 | (mantissa << 13)
        } else {
            sign | ((exponent + 112) << 23) | (mantissa << 13)
        };
        assert_eq!(
            half_to_float(bits as u16).to_bits(),
            expected,
            "half 0x{bits:04x}"
        );
    }
}
#[test]
fn validation_rejects_shape_overflow_type_confusion_and_writes_to_inputs() {
    assert!(
        VectorLayout {
            length: 2,
            offset: u32::MAX,
            stride: 1
        }
        .storage_elements()
        .is_err()
    );
    assert!(MatrixLayout::contiguous(0, 1).storage_elements().is_err());
    assert!(q4_0_matvec(1, 33, 0, 18, VectorLayout::contiguous(33)).is_err());
    assert!(rms_norm(MatrixLayout::contiguous(1, 2), f32::NAN).is_err());
    let mut kernel = row_sum(MatrixLayout::contiguous(1, 2)).unwrap();
    kernel.name = "injected; kernel".into();
    assert!(emit(&kernel, Target::OpenCl).is_err());
    kernel.name = "sum_kernel".into();
    kernel.body = vec![Stmt::Store {
        buffer: "src".into(),
        index: Expr::Uint(0),
        value: Expr::Float(0.0),
    }];
    assert!(kernel.validate().is_err());
}
#[test]
fn failures_roll_back_buffers_and_bound_work() {
    let kernel = row_sum(MatrixLayout::contiguous(2, 3)).unwrap();
    let mut buffers = BTreeMap::from([
        ("src".into(), BufferData::F32(vec![1.0; 6])),
        ("dst".into(), BufferData::F32(vec![-99.0; 2])),
    ]);
    let original = buffers.clone();
    assert!(execute(&kernel, &mut buffers, 2).is_err());
    assert_eq!(buffers, original);
    let mut bad = kernel.clone();
    bad.body = vec![Stmt::Store {
        buffer: "dst".into(),
        index: Expr::Uint(0),
        value: Expr::Float(1.0),
    }];
    assert!(
        execute(&bad, &mut buffers, 100)
            .unwrap_err()
            .0
            .contains("race")
    );
    assert_eq!(buffers, original);
    bad.invocations = 1;
    bad.body = vec![Stmt::Store {
        buffer: "dst".into(),
        index: Expr::Uint(100),
        value: Expr::Float(1.0),
    }];
    assert!(
        execute(&bad, &mut buffers, 100)
            .unwrap_err()
            .0
            .contains("out-of-bounds")
    );
    assert_eq!(buffers, original);
}

#[test]
fn loop_iteration_budget_cannot_underflow_after_body_exhaustion() {
    let mut kernel = row_sum(MatrixLayout::contiguous(1, 2)).unwrap();
    kernel.body = vec![Stmt::For {
        index: "iteration".into(),
        end: Expr::Uint(2),
        body: vec![
            Stmt::Declare {
                name: "value".into(),
                ty: Scalar::F32,
                value: Expr::Float(2.0),
            },
            Stmt::Store {
                buffer: "dst".into(),
                index: Expr::Uint(0),
                value: Expr::var("value"),
            },
        ],
    }];
    let mut buffers = BTreeMap::from([
        ("src".into(), BufferData::F32(vec![1.0; 2])),
        ("dst".into(), BufferData::F32(vec![-1.0])),
    ]);
    let original = buffers.clone();
    assert!(
        execute(&kernel, &mut buffers, 5)
            .unwrap_err()
            .0
            .contains("budget exhausted")
    );
    assert_eq!(buffers, original);
}

#[test]
fn changing_outer_loop_bound_is_snapshotted_in_every_emitter() {
    let mut kernel = row_sum(MatrixLayout::contiguous(1, 3)).unwrap();
    kernel.buffers[1].min_elements = 3;
    kernel.body = vec![
        Stmt::Declare {
            name: "count".into(),
            ty: Scalar::U32,
            value: Expr::Uint(3),
        },
        Stmt::For {
            index: "iteration".into(),
            end: Expr::var("count"),
            body: vec![
                Stmt::Assign {
                    name: "count".into(),
                    value: Expr::Uint(0),
                },
                Stmt::Store {
                    buffer: "dst".into(),
                    index: Expr::var("iteration"),
                    value: Expr::Float(1.0),
                },
            ],
        },
    ];
    let mut buffers = BTreeMap::from([
        ("src".into(), BufferData::F32(vec![0.0; 3])),
        ("dst".into(), BufferData::F32(vec![0.0; 3])),
    ]);
    execute(&kernel, &mut buffers, 100).unwrap();
    assert_eq!(output(&buffers), &[1.0, 1.0, 1.0]);
    for target in Target::ALL {
        let text = emit(&kernel, target).unwrap().source;
        assert!(text.contains("= v_count;"));
        assert!(text.contains("v_iteration < ir_bound_1"));
        assert!(!text.contains("v_iteration < v_count"));
    }
}

#[test]
fn nested_select_validation_visits_each_branch_once() {
    let mut kernel = row_sum(MatrixLayout::contiguous(1, 1)).unwrap();
    let mut value = Expr::Float(1.0);
    for _ in 0..96 {
        value = Expr::Select {
            condition: Box::new(Expr::Bool(true)),
            yes: Box::new(value),
            no: Box::new(Expr::Float(0.0)),
        };
    }
    kernel.body = vec![Stmt::Store {
        buffer: "dst".into(),
        index: Expr::Uint(0),
        value,
    }];
    kernel.validate().unwrap();
    for target in Target::ALL {
        assert!(emit(&kernel, target).unwrap().source.len() < 10000);
    }
}

#[test]
fn empty_kernel_cannot_bypass_dispatch_budget() {
    let mut kernel = row_sum(MatrixLayout::contiguous(1, 1)).unwrap();
    kernel.body.clear();
    kernel.invocations = u32::MAX;
    kernel.workgroup_size = 1;
    let mut buffers = BTreeMap::from([
        ("src".into(), BufferData::F32(vec![0.0])),
        ("dst".into(), BufferData::F32(vec![0.0])),
    ]);
    assert!(
        execute(&kernel, &mut buffers, 0)
            .unwrap_err()
            .0
            .contains("budget")
    );
}
#[test]
fn all_four_emitters_are_deterministic_and_include_tail_guards() {
    let kernels = conformance_kernels().unwrap();
    assert_eq!(kernels.len(), 9);
    for kernel in &kernels {
        for target in Target::ALL {
            let a = emit(kernel, target).unwrap();
            let b = emit(kernel, target).unwrap();
            assert_eq!(a.source, b.source);
            assert_eq!(a.sha256, sha256_hex(a.source.as_bytes()));
            assert!(a.source.contains("if (ir_gid >="));
            assert!(!a.source.contains("#include \""));
        }
    }
    let wgsl = emit(
        &rms_norm(MatrixLayout::contiguous(2, 3), 1e-5).unwrap(),
        Target::Wgsl,
    )
    .unwrap();
    assert!(wgsl.source.contains("inverseSqrt("));
}
#[test]
fn digest_and_cache_keys_have_known_and_distinct_results() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    let key = CacheIdentity {
        artifact_sha256: sha256_hex(b"source"),
        compiler: "compiler-v1".into(),
        target: "glsl450".into(),
        device: "device-A".into(),
        driver: "driver-v1".into(),
        capabilities: "scalar-f32".into(),
        flags: vec!["strict".into()],
    };
    let first = key.key().unwrap();
    let mut changed = key.clone();
    changed.driver = "driver-v2".into();
    assert_ne!(first, changed.key().unwrap());
    changed = key.clone();
    changed.flags = vec!["str".into(), "ict".into()];
    assert_ne!(first, changed.key().unwrap());
}
