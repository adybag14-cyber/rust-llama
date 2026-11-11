use llama_tensor::{DType, Graph, Op, OpCode, Tensor, TensorDesc, TensorError};
use std::sync::atomic::AtomicBool;

fn values(shape: [usize; 4], value: f32) -> Tensor {
    Tensor::from_f32(shape, &vec![value; shape.iter().product()]).unwrap()
}
fn run(op: Op, tensors: &[Tensor]) -> Result<Tensor, TensorError> {
    let mut graph = Graph::new();
    let ids = tensors
        .iter()
        .cloned()
        .map(|tensor| graph.input(tensor))
        .collect::<Vec<_>>();
    let target = graph.op(op, &ids)?;
    graph.evaluate(target)
}
fn delta() -> Vec<Tensor> {
    vec![
        values([2, 1, 2, 1], 0.25),
        values([2, 3, 2, 2], 0.5),
        values([2, 2, 2, 2], 0.75),
        values([1, 2, 2, 2], -0.125),
        values([1, 2, 2, 2], 0.25),
        values([2, 2, 2, 2], 0.125),
    ]
}
#[test]
fn constructors_check_source_arity_and_keep_original_ids() {
    for (op, code, arity) in [
        (
            Op::GatedDeltaNet { snapshots: 1 },
            OpCode::GATED_DELTA_NET,
            6,
        ),
        (
            Op::Dsv4HcComb {
                epsilon: 0.001,
                iterations: 3,
            },
            OpCode::DSV4_HC_COMB,
            3,
        ),
        (Op::Dsv4HcPre, OpCode::DSV4_HC_PRE, 2),
        (Op::Dsv4HcPreGated { scale: 0.5 }, OpCode::DSV4_HC_PRE, 2),
        (Op::Dsv4HcPost { with_comb: false }, OpCode::DSV4_HC_POST, 3),
        (Op::Dsv4HcPost { with_comb: true }, OpCode::DSV4_HC_POST, 4),
    ] {
        assert_eq!(op.code(), code);
        let mut graph = Graph::new();
        let input = graph.input(values([1; 4], 0.0));
        assert!(
            matches!(graph.op(op.clone(), &vec![input;arity-1]), Err(TensorError::InvalidArity { expected, actual }) if expected == arity && actual == arity-1)
        );
        assert!(graph.op(op, &vec![input; arity]).is_ok());
    }
}
#[test]
fn delta_checks_group_shapes_physical_rows_and_state_layouts() {
    let inputs = delta();
    assert!(run(Op::GatedDeltaNet { snapshots: 2 }, &inputs).is_ok());
    for index in [0, 1] {
        let mut bad = inputs.clone();
        // Three stored sequences do not divide the two destination sequences.
        bad[index] = values([2, 1, 2, 3], 0.0);
        assert!(matches!(
            run(Op::GatedDeltaNet { snapshots: 1 }, &bad),
            Err(TensorError::Shape(_))
        ));
        bad[index] = values([1, 1, 2, 1], 0.0);
        assert!(matches!(
            run(Op::GatedDeltaNet { snapshots: 1 }, &bad),
            Err(TensorError::Shape(_))
        ));
        let desc = TensorDesc::with_strides(DType::F32, [2, 1, 2, 1], [8, 16, 16, 32]).unwrap();
        bad[index] = Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap();
        assert!(matches!(
            run(Op::GatedDeltaNet { snapshots: 1 }, &bad),
            Err(TensorError::Layout(_))
        ));
    }
    let mut bad = inputs.clone();
    bad[3] = values([3, 2, 2, 2], 0.0);
    assert!(run(Op::GatedDeltaNet { snapshots: 1 }, &bad).is_err());
    bad = inputs.clone();
    bad[3] = bad[3].permute([0, 2, 1, 3]).unwrap();
    assert!(matches!(
        run(Op::GatedDeltaNet { snapshots: 1 }, &bad),
        Err(TensorError::Layout(_))
    ));
    bad = inputs.clone();
    bad[4] = values([1, 1, 2, 2], 0.0);
    assert!(run(Op::GatedDeltaNet { snapshots: 1 }, &bad).is_err());
    bad = inputs.clone();
    bad[5] = values([2, 2, 1, 2], 0.0);
    assert!(run(Op::GatedDeltaNet { snapshots: 1 }, &bad).is_err());
}
#[test]
fn hyperconnection_shapes_and_scalar_alignment_are_checked() {
    let x = values([3, 4, 2, 1], 0.5);
    assert!(run(Op::Dsv4HcPre, &[x.clone(), values([4, 3, 1, 1], 1.0)]).is_err());
    assert!(
        run(
            Op::Dsv4HcPreGated { scale: 1.0 },
            &[x.clone(), values([4, 2, 1, 1], 0.0)]
        )
        .is_err()
    );
    let desc = TensorDesc::with_strides(DType::F32, [3, 4, 2, 1], [5, 15, 60, 120]).unwrap();
    let unaligned = Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap();
    assert!(matches!(
        run(Op::Dsv4HcPre, &[unaligned, values([4, 2, 1, 1], 1.0)]),
        Err(TensorError::Layout(_))
    ));
    assert!(
        run(
            Op::Dsv4HcPost { with_comb: true },
            &[
                values([3, 2, 1, 1], 0.0),
                x,
                values([4, 2, 1, 1], 0.5),
                values([4, 3, 2, 1], 0.25)
            ]
        )
        .is_err()
    );
    assert!(
        run(
            Op::Dsv4HcComb {
                epsilon: 0.001,
                iterations: 2
            },
            &[
                values([24, 2, 1, 1], 0.0),
                values([2, 1, 1, 1], 1.0),
                values([24, 1, 1, 1], 0.0)
            ]
        )
        .is_err()
    );
    assert!(matches!(
        run(
            Op::Dsv4HcPre,
            &[Tensor::from_i32([1; 4], &[1]).unwrap(), values([1; 4], 1.0)]
        ),
        Err(TensorError::Type { .. })
    ));
}
#[test]
fn invalid_parameters_and_unwritten_snapshot_slots_are_rejected() {
    for snapshots in [0, 3, i32::MAX as usize + 1, usize::MAX] {
        assert!(matches!(
            run(Op::GatedDeltaNet { snapshots }, &delta()),
            Err(TensorError::Parameters(_))
        ));
    }
    let inputs = [
        values([24, 2, 1, 1], 0.0),
        values([3, 1, 1, 1], 1.0),
        values([24, 1, 1, 1], 0.0),
    ];
    for iterations in [0, i32::MAX as usize + 1, usize::MAX] {
        assert!(matches!(
            run(
                Op::Dsv4HcComb {
                    epsilon: 0.001,
                    iterations
                },
                &inputs
            ),
            Err(TensorError::Parameters(_))
        ));
    }
    for epsilon in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            run(
                Op::Dsv4HcComb {
                    epsilon,
                    iterations: 2
                },
                &inputs
            ),
            Err(TensorError::Unsupported(_))
        ));
    }
    assert!(
        run(
            Op::Dsv4HcPre,
            &[values([0, 4, 2, 1], 0.0), values([4, 2, 1, 1], 0.0)]
        )
        .is_err()
    );
}
#[test]
fn snapshot_and_broadcast_output_amplification_is_bounded() {
    let tokens = 32_768;
    let width = 128;
    let desc =
        TensorDesc::with_strides(DType::F32, [width, 1, tokens, 1], [4, width * 4, 0, 0]).unwrap();
    let aliased = Tensor::from_bytes(desc, vec![0; width * 4]).unwrap();
    let inputs = [
        aliased.clone(),
        aliased.clone(),
        aliased,
        values([1, 1, tokens, 1], 0.0),
        values([1, 1, tokens, 1], 0.0),
        values([width, width, 1, 1], 0.0),
    ];
    // Valid native geometry would create a multi-GiB output from small aliased
    // input storage. Reject before allocation, copying or recurrent work.
    assert!(matches!(
        run(Op::GatedDeltaNet { snapshots: tokens }, &inputs),
        Err(TensorError::Unsupported(_))
    ));
    let huge = i32::MAX as usize;
    let mix_desc = TensorDesc::with_strides(DType::F32, [24, huge, 1, 1], [4, 0, 0, 0]).unwrap();
    let mixes = Tensor::from_bytes(mix_desc, vec![0; 96]).unwrap();
    assert!(matches!(
        run(
            Op::Dsv4HcComb {
                epsilon: 0.001,
                iterations: 1
            },
            &[mixes, values([3, 1, 1, 1], 1.0), values([24, 1, 1, 1], 0.0)]
        ),
        Err(TensorError::Unsupported(_))
    ));
}
#[test]
fn cancellation_and_checked_failures_preserve_aliased_inputs() {
    let inputs = delta();
    let original = inputs
        .iter()
        .map(|t| t.to_bytes().unwrap())
        .collect::<Vec<_>>();
    let mut graph = Graph::new();
    let nodes = inputs
        .iter()
        .cloned()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph
        .op(Op::GatedDeltaNet { snapshots: 2 }, &nodes)
        .unwrap();
    assert!(matches!(
        graph.evaluate_with_cancel(target, &AtomicBool::new(true)),
        Err(TensorError::Cancelled)
    ));
    let mut invalid = inputs.clone();
    invalid[4] = values([1, 2, 2, 2], f32::NAN);
    assert!(matches!(
        run(Op::GatedDeltaNet { snapshots: 2 }, &invalid),
        Err(TensorError::Unsupported(_))
    ));
    for (input, expected) in inputs.iter().zip(original) {
        assert_eq!(input.to_bytes().unwrap(), expected);
    }
}
