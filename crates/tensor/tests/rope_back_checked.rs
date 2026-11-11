use llama_tensor::{
    DType, Graph, Op, OpCode, RopeBackMode, RopeBackOptions, Tensor, TensorDesc, TensorError,
};
use std::sync::atomic::AtomicBool;
fn values(shape: [usize; 4], value: f32) -> Tensor {
    Tensor::from_f32(shape, &vec![value; shape.iter().product()]).unwrap()
}
fn view(dtype: DType, shape: [usize; 4], strides: [usize; 4]) -> Tensor {
    let desc = TensorDesc::with_strides(dtype, shape, strides).unwrap();
    Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap()
}
fn args() -> Vec<Tensor> {
    vec![
        values([8, 2, 3, 1], 0.25),
        Tensor::from_i32([3, 1, 1, 1], &[-7, 0, 13]).unwrap(),
    ]
}
fn run(options: RopeBackOptions, inputs: &[Tensor]) -> Result<Tensor, TensorError> {
    let mut graph = Graph::new();
    let ids = inputs
        .iter()
        .cloned()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(
        Op::RopeBack {
            options,
            with_factors: inputs.len() == 3,
        },
        &ids,
    )?;
    graph.evaluate(target)
}
fn options() -> RopeBackOptions {
    RopeBackOptions::new(4, RopeBackMode::Normal)
}

#[test]
fn original_backward_id_modes_and_optional_factor_arity_are_preserved() {
    for (mode, id) in [
        (RopeBackMode::Normal, 0),
        (RopeBackMode::NeoX, 2),
        (RopeBackMode::Multi, 8),
        (RopeBackMode::Vision, 24),
        (RopeBackMode::Interleaved, 40),
    ] {
        assert_eq!(mode as u32, id);
    }
    for with_factors in [false, true] {
        let op = Op::RopeBack {
            options: options(),
            with_factors,
        };
        assert_eq!(op.code(), OpCode::ROPE_BACK);
        assert_eq!(op.code() as u32, 49);
        let mut graph = Graph::new();
        let node = graph.input(values([1; 4], 1.0));
        let expected = if with_factors { 3 } else { 2 };
        assert!(
            matches!(graph.op(op.clone(),&vec![node;expected-1]),Err(TensorError::InvalidArity {expected:e,actual}) if e==expected && actual==expected-1)
        );
        assert!(graph.op(op, &vec![node; expected]).is_ok());
    }
}
#[test]
fn input_types_and_flat_position_frequency_contracts_are_checked() {
    let valid = args();
    assert!(run(options(), &valid).is_ok());
    let mut bad = valid.clone();
    bad[0] = Tensor::from_i32([8, 2, 3, 1], &[0; 48]).unwrap();
    assert!(matches!(
        run(options(), &bad),
        Err(TensorError::Unsupported(_))
    ));
    bad = valid.clone();
    bad[1] = values([3, 1, 1, 1], 0.0);
    assert!(matches!(
        run(options(), &bad),
        Err(TensorError::Type { .. })
    ));
    bad = valid.clone();
    bad[1] = Tensor::from_i32([1, 3, 1, 1], &[0; 3]).unwrap();
    assert!(matches!(run(options(), &bad), Err(TensorError::Shape(_))));
    bad = valid.clone();
    bad.push(Tensor::from_i32([4, 1, 1, 1], &[1; 4]).unwrap());
    assert!(matches!(
        run(options(), &bad),
        Err(TensorError::Type { .. })
    ));
    bad = valid.clone();
    bad.push(values([1; 4], 1.0));
    assert!(matches!(run(options(), &bad), Err(TensorError::Shape(_))));
    let mut multi = options();
    multi.mode = RopeBackMode::Multi;
    multi.sections = [1, 1, 1, 1];
    assert!(matches!(run(multi, &valid), Err(TensorError::Shape(_))));
}
#[test]
fn rotation_offsets_sections_and_signed_parameter_bounds_are_checked() {
    for (dimensions, offset) in [(0, 0), (3, 0), (4, 1), (4, 6), (10, 0), (usize::MAX, 0)] {
        let mut bad = options();
        bad.dimensions = dimensions;
        bad.offset = offset;
        assert!(matches!(run(bad, &args()), Err(TensorError::Parameters(_))));
    }
    let odd = [
        values([7, 1, 1, 1], 1.0),
        Tensor::from_i32([1; 4], &[1]).unwrap(),
    ];
    assert!(matches!(
        run(options(), &odd),
        Err(TensorError::Parameters(_))
    ));
    let mut vision = RopeBackOptions::new(2, RopeBackMode::Vision);
    vision.sections = [1, 1, 1, 1];
    assert!(matches!(
        run(vision, &args()),
        Err(TensorError::Parameters(_))
    ));
    for sections in [[0, 0, 0, 1], [1, 4, 4, 0], [usize::MAX, 0, 0, 0]] {
        let mut bad = options();
        bad.mode = RopeBackMode::Multi;
        bad.sections = sections;
        assert!(matches!(run(bad, &args()), Err(TensorError::Parameters(_))));
    }
    let mut bad = options();
    bad.context_original = i32::MAX as usize + 1;
    assert!(matches!(run(bad, &args()), Err(TensorError::Parameters(_))));
}
#[test]
fn declared_flat_prefix_spans_and_physical_scalar_rows_are_enforced() {
    let mut short = args();
    // The constructor accepts dimensions/2 factors, but the original cache
    // reads width/2 words. Reject a declared backing span that is too short.
    short.push(values([2, 1, 1, 1], 1.0));
    assert!(matches!(run(options(), &short), Err(TensorError::Bounds)));
    short[2] = values([2, 2, 1, 1], 1.0);
    assert!(run(options(), &short).is_ok());
    let mut bad = args();
    bad[0] = view(DType::F32, [8, 2, 3, 1], [8, 64, 128, 384]);
    assert!(matches!(run(options(), &bad), Err(TensorError::Layout(_))));
    bad[0] = view(DType::F32, [8, 2, 3, 1], [4, 33, 66, 198]);
    assert!(matches!(run(options(), &bad), Err(TensorError::Layout(_))));
}
#[test]
fn aliased_output_amplification_cancellation_and_failures_do_not_mutate_inputs() {
    let huge = i32::MAX as usize;
    let aliased = [
        view(DType::F32, [2, huge, 1, 1], [4, 0, 0, 0]),
        Tensor::from_i32([1; 4], &[1]).unwrap(),
    ];
    let tiny = RopeBackOptions::new(2, RopeBackMode::Normal);
    assert!(matches!(
        run(tiny, &aliased),
        Err(TensorError::Unsupported(_))
    ));
    let inputs = args();
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
    let target = graph
        .op(
            Op::RopeBack {
                options: options(),
                with_factors: false,
            },
            &ids,
        )
        .unwrap();
    assert!(matches!(
        graph.evaluate_with_cancel(target, &AtomicBool::new(true)),
        Err(TensorError::Cancelled)
    ));
    let mut bad = options();
    bad.frequency_base = 0.0;
    assert!(matches!(
        run(bad, &inputs),
        Err(TensorError::Unsupported(_))
    ));
    for (input, bytes) in inputs.iter().zip(before) {
        assert_eq!(input.to_bytes().unwrap(), bytes);
    }
}
