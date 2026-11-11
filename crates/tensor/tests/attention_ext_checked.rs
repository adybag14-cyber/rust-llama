use llama_tensor::{
    DType, Graph, Op, OpCode, RopeExtMode, RopeExtOptions, Tensor, TensorDesc, TensorError,
};
use std::sync::atomic::AtomicBool;
fn values(shape: [usize; 4], value: f32) -> Tensor {
    Tensor::from_f32(shape, &vec![value; shape.iter().product()]).unwrap()
}
fn operation(mask: bool, sinks: bool) -> Op {
    Op::SoftmaxExt {
        scale: 0.25,
        max_bias: if mask { 8.0 } else { 0.0 },
        with_mask: mask,
        with_sinks: sinks,
    }
}
fn run(op: Op, inputs: &[Tensor]) -> Result<Tensor, TensorError> {
    let mut graph = Graph::new();
    let ids = inputs
        .iter()
        .cloned()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let node = graph.op(op, &ids)?;
    graph.evaluate(node)
}
#[test]
fn original_constructor_failures_are_checked_errors() {
    let input = values([4, 2, 3, 1], 1.0);
    let positive_without_mask = Op::SoftmaxExt {
        scale: 1.0,
        max_bias: 8.0,
        with_mask: false,
        with_sinks: false,
    };
    assert!(matches!(
        run(positive_without_mask, std::slice::from_ref(&input)),
        Err(TensorError::Parameters(_))
    ));
    assert!(matches!(
        run(
            operation(true, false),
            &[input.clone(), values([4, 1, 1, 1], 0.0)]
        ),
        Err(TensorError::Shape(_))
    ));
    assert!(matches!(
        run(
            operation(false, true),
            &[input.clone(), values([2, 1, 1, 1], 0.0)]
        ),
        Err(TensorError::Shape(_))
    ));
    assert!(matches!(
        run(
            operation(true, false),
            &[input.clone(), values([4, 2, 2, 1], 0.0)]
        ),
        Err(TensorError::Shape(_))
    ));
    assert!(matches!(
        run(operation(true, false), &[input, values([3, 2, 1, 1], 0.0)]),
        Err(TensorError::Shape(_))
    ));
}
#[test]
fn mask_types_noncontiguity_and_sink_backing_spans_are_checked() {
    let input = values([4, 2, 3, 1], 1.0);
    let mask = Tensor::from_i32([4, 2, 1, 1], &[0; 8]).unwrap();
    assert!(matches!(
        run(operation(true, false), &[input.clone(), mask]),
        Err(TensorError::Unsupported(_))
    ));
    let desc = TensorDesc::with_strides(DType::F32, [4, 2, 1, 1], [4, 20, 40, 40]).unwrap();
    let mask = Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap();
    assert!(matches!(
        run(operation(true, false), &[input.clone(), mask]),
        Err(TensorError::Layout(_))
    ));
    let desc = TensorDesc::with_strides(DType::F32, [3, 0, 1, 1], [4, 12, 0, 0]).unwrap();
    let sinks = Tensor::from_bytes(desc, Vec::new()).unwrap();
    assert!(matches!(
        run(operation(false, true), &[input, sinks]),
        Err(TensorError::Bounds)
    ));
    let desc = TensorDesc::with_strides(DType::F32, [4, 2, 3, 1], [4, 20, 40, 120]).unwrap();
    let input = Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap();
    assert!(matches!(
        run(operation(false, false), &[input]),
        Err(TensorError::Layout(_))
    ));
}
#[test]
fn optional_input_arity_and_frozen_operation_ids_are_preserved() {
    for mask in [false, true] {
        for sinks in [false, true] {
            let op = operation(mask, sinks);
            assert_eq!(op.code(), OpCode::SOFT_MAX);
            let mut graph = Graph::new();
            let id = graph.input(values([1; 4], 1.0));
            let count = 1 + usize::from(mask) + usize::from(sinks);
            assert!(matches!(
                graph.op(op.clone(), &vec![id; count + 1]),
                Err(TensorError::InvalidArity { .. })
            ));
            assert!(graph.op(op, &vec![id; count]).is_ok());
        }
    }
    let op = Op::RopeExt {
        options: RopeExtOptions::new(4, RopeExtMode::Normal),
        with_factors: true,
    };
    assert_eq!(op.code(), OpCode::ROPE);
    let mut graph = Graph::new();
    let id = graph.input(values([1; 4], 1.0));
    assert!(matches!(
        graph.op(op, &[id, id]),
        Err(TensorError::InvalidArity {
            expected: 3,
            actual: 2
        })
    ));
}
#[test]
fn cancellation_and_later_row_failure_leave_input_bytes_unchanged() {
    let input =
        Tensor::from_f32([4, 2, 1, 1], &[1., 2., 3., 4., 1., f32::INFINITY, 2., 3.]).unwrap();
    let original = input.to_bytes().unwrap();
    let mut graph = Graph::new();
    let id = graph.input(input.clone());
    let target = graph.op(operation(false, false), &[id]).unwrap();
    assert!(matches!(
        graph.evaluate_with_cancel(target, &AtomicBool::new(true)),
        Err(TensorError::Cancelled)
    ));
    assert!(matches!(
        graph.evaluate(target),
        Err(TensorError::Unsupported(_))
    ));
    assert_eq!(input.to_bytes().unwrap(), original);
    let values = values([8, 2, 3, 1], 0.25);
    let positions = Tensor::from_i32([3, 1, 1, 1], &[-7, 0, 13]).unwrap();
    let before = values.to_bytes().unwrap();
    let mut graph = Graph::new();
    let a = graph.input(values.clone());
    let p = graph.input(positions);
    let target = graph
        .op(
            Op::RopeExt {
                options: RopeExtOptions::new(4, RopeExtMode::Normal),
                with_factors: false,
            },
            &[a, p],
        )
        .unwrap();
    assert!(matches!(
        graph.evaluate_with_cancel(target, &AtomicBool::new(true)),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(values.to_bytes().unwrap(), before);
}
#[test]
fn amplified_forward_rotary_output_is_rejected_before_allocation() {
    let desc =
        TensorDesc::with_strides(DType::F32, [2, i32::MAX as usize, 1, 1], [4, 0, 0, 0]).unwrap();
    let input = Tensor::from_bytes(desc, vec![0; 8]).unwrap();
    let positions = Tensor::from_i32([1; 4], &[1]).unwrap();
    assert!(matches!(
        run(
            Op::RopeExt {
                options: RopeExtOptions::new(2, RopeExtMode::Normal),
                with_factors: false
            },
            &[input, positions]
        ),
        Err(TensorError::Unsupported(_))
    ));
}
