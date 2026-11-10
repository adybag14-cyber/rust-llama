//! Checked-domain regressions for source inputs that would assert, access
//! outside their view, divide by zero, or return unwritten recurrent state.
use llama_tensor::{DType, Graph, Op, Result, Tensor, TensorDesc, TensorError};
use std::sync::atomic::AtomicBool;

fn f32_tensor(shape: [usize; 4], value: f32) -> Tensor {
    Tensor::from_f32(shape, &vec![value; shape.iter().product()]).unwrap()
}
fn layout(shape: [usize; 4], strides: [usize; 4]) -> Tensor {
    let desc = TensorDesc::with_strides(DType::F32, shape, strides).unwrap();
    Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap()
}
fn evaluate(op: Op, inputs: &[Tensor]) -> Result<Tensor> {
    let mut graph = Graph::new();
    let ids = inputs
        .iter()
        .cloned()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(op, &ids)?;
    graph.evaluate(target)
}
fn scan() -> Vec<Tensor> {
    vec![
        f32_tensor([2, 1, 2, 3], 0.5),
        f32_tensor([1, 2, 3, 2], 0.25),
        f32_tensor([2, 3, 2, 1], -0.5),
        f32_tensor([1, 2, 1, 1], -0.25),
        f32_tensor([2, 1, 3, 2], 0.75),
        f32_tensor([2, 1, 3, 2], -0.125),
        Tensor::from_i32([2, 1, 1, 1], &[2, 0]).unwrap(),
    ]
}
fn recurrent(op: &Op) -> Vec<Tensor> {
    let input_count = match op {
        Op::RwkvWkv6 => 5,
        Op::GatedLinearAttn { .. } => 4,
        Op::RwkvWkv7 => 6,
        _ => unreachable!(),
    };
    let mut inputs = vec![f32_tensor([3, 2, 6, 1], 0.125); input_count];
    if matches!(op, Op::RwkvWkv6) {
        inputs[3] = f32_tensor([3, 2, 1, 1], 0.25);
    }
    inputs.push(f32_tensor([18, 2, 1, 1], -0.125));
    inputs
}
fn recurrent_ops() -> [Op; 3] {
    [
        Op::RwkvWkv6,
        Op::GatedLinearAttn { scale: 0.25 },
        Op::RwkvWkv7,
    ]
}

#[test]
fn recurrent_registry_ids_and_arities_are_executable() {
    for (op, id, count) in [
        (Op::SsmConv, 76, 2),
        (Op::SsmScan { snapshots: 1 }, 77, 7),
        (Op::RwkvWkv6, 82, 6),
        (Op::GatedLinearAttn { scale: 1.0 }, 83, 5),
        (Op::RwkvWkv7, 84, 7),
    ] {
        assert_eq!(op.code() as u32, id);
        assert!(matches!(evaluate(op, &[]),
            Err(TensorError::InvalidArity { expected, actual: 0 }) if expected == count));
    }
}

#[test]
fn convolution_checks_source_shapes_and_physical_accesses() {
    let valid = vec![
        f32_tensor([4, 2, 2, 1], 0.5),
        f32_tensor([3, 2, 1, 1], 0.25),
    ];
    assert_eq!(evaluate(Op::SsmConv, &valid).unwrap().shape(), [2, 2, 2, 1]);
    let mut inputs = valid.clone();
    inputs[1] = f32_tensor([0, 2, 1, 1], 0.0);
    assert!(matches!(
        evaluate(Op::SsmConv, &inputs),
        Err(TensorError::Shape(_))
    ));
    inputs[1] = f32_tensor([6, 2, 1, 1], 0.0);
    assert!(matches!(
        evaluate(Op::SsmConv, &inputs),
        Err(TensorError::Shape(_))
    ));
    inputs[1] = f32_tensor([3, 1, 1, 1], 0.0);
    assert!(matches!(
        evaluate(Op::SsmConv, &inputs),
        Err(TensorError::Shape(_))
    ));
    inputs = valid.clone();
    inputs[0] = layout([4, 2, 2, 1], [4, 20, 40, 80]);
    assert!(matches!(
        evaluate(Op::SsmConv, &inputs),
        Err(TensorError::Layout(_))
    ));
    inputs = valid.clone();
    inputs[1] = layout([3, 2, 1, 1], [8, 24, 48, 48]);
    assert!(matches!(
        evaluate(Op::SsmConv, &inputs),
        Err(TensorError::Layout(_))
    ));
    // Source flattens weights despite nb1; its read must not escape a small
    // declared view merely because an owning allocation could be larger.
    let parent = f32_tensor([32, 1, 1, 1], 0.25);
    let overlapping = TensorDesc::with_strides(DType::F32, [3, 2, 1, 1], [4, 0, 0, 0]).unwrap();
    inputs[1] = parent.view(overlapping, 4).unwrap();
    assert_eq!(
        evaluate(Op::SsmConv, &inputs).unwrap_err(),
        TensorError::Bounds
    );
    inputs = valid;
    inputs[1] = Tensor::from_i32([3, 2, 1, 1], &[0; 6]).unwrap();
    assert!(matches!(
        evaluate(Op::SsmConv, &inputs),
        Err(TensorError::Type {
            expected: DType::F32,
            ..
        })
    ));
    let empty = [f32_tensor([2, 3, 2, 1], 0.0), f32_tensor([3, 3, 1, 1], 0.0)];
    assert_eq!(evaluate(Op::SsmConv, &empty).unwrap().to_f32().unwrap(), []);
}

#[test]
fn scan_rejects_unwritten_snapshots_bad_ids_and_mismatched_groups() {
    let valid = scan();
    assert!(evaluate(Op::SsmScan { snapshots: 3 }, &valid).is_ok());
    for snapshots in [0, 4] {
        assert!(matches!(
            evaluate(Op::SsmScan { snapshots }, &valid),
            Err(TensorError::Parameters(_))
        ));
    }
    let mut inputs = valid.clone();
    inputs[3] = f32_tensor([2, 2, 1, 1], -0.25);
    assert!(evaluate(Op::SsmScan { snapshots: 1 }, &inputs).is_ok());
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 2 }, &inputs),
        Err(TensorError::Parameters(_))
    ));
    inputs[3] = f32_tensor([3, 2, 1, 1], -0.25);
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 1 }, &inputs),
        Err(TensorError::Shape(_))
    ));
    for bad in [-1, 3, i32::MAX] {
        inputs = valid.clone();
        inputs[6] = Tensor::from_i32([2, 1, 1, 1], &[bad, 0]).unwrap();
        assert_eq!(
            evaluate(Op::SsmScan { snapshots: 1 }, &inputs).unwrap_err(),
            TensorError::Bounds
        );
    }
    inputs = valid.clone();
    inputs[4] = f32_tensor([2, 3, 3, 2], 0.25);
    inputs[5] = inputs[4].clone();
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 1 }, &inputs),
        Err(TensorError::Shape(_))
    ));
    inputs = valid.clone();
    inputs[1] = f32_tensor([1, 2, 0, 2], 0.0);
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 1 }, &inputs),
        Err(TensorError::Shape(_))
    ));
    inputs = valid.clone();
    inputs[4] = layout([2, 1, 3, 2], [4, 16, 16, 48]);
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 1 }, &inputs),
        Err(TensorError::Layout(_))
    ));
    inputs = valid;
    inputs[6] = f32_tensor([2, 1, 1, 1], 0.0);
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 1 }, &inputs),
        Err(TensorError::Type {
            expected: DType::I32,
            ..
        })
    ));
}

#[test]
fn scan_rejects_ambiguous_unit_axis_state_destination_strides() {
    let mut inputs = scan();
    inputs[0] = layout([2, 1, 2, 1], [4, 8, 8, 0]);
    inputs[6] = Tensor::from_i32([2, 1, 1, 1], &[0, 0]).unwrap();
    assert!(matches!(
        evaluate(Op::SsmScan { snapshots: 1 }, &inputs),
        Err(TensorError::Layout(_))
    ));
    let too_wide = TensorDesc::contiguous(DType::F32, [i32::MAX as usize + 1, 0, 1, 1]).unwrap();
    let x = Tensor::from_bytes(too_wide, vec![]).unwrap();
    assert!(matches!(
        evaluate(Op::SsmConv, &[x, f32_tensor([1, 0, 1, 1], 0.0)]),
        Err(TensorError::Unsupported(_))
    ));
}

#[test]
fn rwkv_and_gla_reject_empty_or_unequal_sequence_runs_and_invalid_state_storage() {
    for op in recurrent_ops() {
        let valid = recurrent(&op);
        assert_eq!(evaluate(op.clone(), &valid).unwrap().shape(), [6, 12, 1, 1]);
        let last = valid.len() - 1;
        let mut inputs = valid.clone();
        inputs[last] = f32_tensor([18, 4, 1, 1], 0.0);
        assert!(matches!(
            evaluate(op.clone(), &inputs),
            Err(TensorError::Shape(_))
        ));
        inputs[last] = f32_tensor([18, 0, 1, 1], 0.0);
        assert!(matches!(
            evaluate(op.clone(), &inputs),
            Err(TensorError::Shape(_))
        ));
        inputs[last] = f32_tensor([17, 2, 1, 1], 0.0);
        assert!(matches!(
            evaluate(op.clone(), &inputs),
            Err(TensorError::Shape(_))
        ));
        inputs = valid.clone();
        inputs[1] = f32_tensor([2, 2, 6, 1], 0.25);
        assert!(matches!(
            evaluate(op.clone(), &inputs),
            Err(TensorError::Shape(_))
        ));
        inputs = valid.clone();
        inputs[0] = layout([3, 2, 6, 1], [4, 16, 32, 192]);
        assert!(matches!(
            evaluate(op.clone(), &inputs),
            Err(TensorError::Layout(_))
        ));
        inputs = valid;
        inputs[0] = f32_tensor([3, 2, 0, 1], 0.0);
        assert!(matches!(evaluate(op, &inputs), Err(TensorError::Shape(_))));
    }
    let mut inputs = recurrent(&Op::RwkvWkv6);
    inputs[3] = f32_tensor([1, 1, 1, 1], 0.25);
    assert_eq!(
        evaluate(Op::RwkvWkv6, &inputs).unwrap_err(),
        TensorError::Bounds
    );
}

#[test]
fn competing_nans_are_checked_at_arithmetic_and_quiet_single_nan_is_preserved() {
    let a = f32::from_bits(0x7f81_2345);
    let b = f32::from_bits(0xffc5_4321);
    let result = evaluate(
        Op::SsmConv,
        &[f32_tensor([1, 1, 1, 1], a), f32_tensor([1, 1, 1, 1], 1.0)],
    )
    .unwrap();
    assert_eq!(result.to_f32().unwrap()[0].to_bits(), 0x7fc1_2345);
    let result = evaluate(
        Op::SsmConv,
        &[f32_tensor([1, 1, 1, 1], a), f32_tensor([1, 1, 1, 1], b)],
    );
    assert!(matches!(
        result,
        Err(TensorError::Unsupported(
            "recurrent arithmetic with competing NaN payloads is unsupported"
        ))
    ));
    // Different NaNs in independent rows do not collide and are not rejected.
    let x = Tensor::from_f32([1, 2, 1, 1], &[a, b]).unwrap();
    let result = evaluate(Op::SsmConv, &[x, f32_tensor([1, 2, 1, 1], 1.0)]).unwrap();
    let bits = result
        .to_f32()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bits, [0x7fc1_2345, 0xffc5_4321]);
    // Invalid arithmetic can create a second payload even when only one input
    // contains a NaN. Its eventual collision is checked in the reduction.
    let x = Tensor::from_f32([2, 1, 1, 1], &[0.0, a]).unwrap();
    let weights = Tensor::from_f32([2, 1, 1, 1], &[f32::INFINITY, 1.0]).unwrap();
    assert!(matches!(
        evaluate(Op::SsmConv, &[x, weights]),
        Err(TensorError::Unsupported(
            "recurrent arithmetic with competing NaN payloads is unsupported"
        ))
    ));
}

#[test]
fn cancellation_does_not_publish_output_or_mutate_recurrent_inputs() {
    for (op, inputs) in [
        (
            Op::SsmConv,
            vec![
                f32_tensor([4, 2, 2, 1], 0.5),
                f32_tensor([3, 2, 1, 1], 0.25),
            ],
        ),
        (Op::SsmScan { snapshots: 2 }, scan()),
        (Op::RwkvWkv6, recurrent(&Op::RwkvWkv6)),
        (
            Op::GatedLinearAttn { scale: 0.25 },
            recurrent(&Op::GatedLinearAttn { scale: 0.25 }),
        ),
        (Op::RwkvWkv7, recurrent(&Op::RwkvWkv7)),
    ] {
        let before = inputs
            .iter()
            .map(|t| t.to_bytes().unwrap())
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let ids = inputs
            .iter()
            .cloned()
            .map(|t| graph.input(t))
            .collect::<Vec<_>>();
        let target = graph.op(op, &ids).unwrap();
        assert_eq!(
            graph
                .evaluate_with_cancel(target, &AtomicBool::new(true))
                .unwrap_err(),
            TensorError::Cancelled
        );
        for (input, bytes) in inputs.iter().zip(before) {
            assert_eq!(input.to_bytes().unwrap(), bytes);
        }
        assert!(graph.evaluate(target).is_ok());
    }
}
