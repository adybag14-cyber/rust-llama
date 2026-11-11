use llama_tensor::{DType, Graph, Op, Tensor, TensorDesc, TensorError};
use std::sync::atomic::AtomicBool;
fn run(op: Op, inputs: &[Tensor]) -> Result<Tensor, TensorError> {
    let mut g = Graph::new();
    let ids = inputs
        .iter()
        .cloned()
        .map(|t| g.input(t))
        .collect::<Vec<_>>();
    let id = g.op(op, &ids)?;
    g.evaluate(id)
}
fn zeros(shape: [usize; 4]) -> Tensor {
    Tensor::from_f32(shape, &vec![0.; shape.iter().product()]).unwrap()
}
#[test]
fn geometry_and_dtype_errors_do_not_mutate_shared_inputs() {
    let a = zeros([3, 5, 7, 1]);
    let before = a.to_bytes().unwrap();
    for window in [0, usize::MAX, i32::MAX as usize + 1] {
        assert!(run(Op::WindowPartition { window }, std::slice::from_ref(&a)).is_err());
    }
    assert!(run(Op::WindowPartition { window: 3 }, &[zeros([3, 5, 7, 2])]).is_err());
    assert!(
        run(
            Op::WindowUnpartition {
                width: 7,
                height: 8,
                window: 3
            },
            &[zeros([3, 1, 1, 1])]
        )
        .is_err()
    );
    assert!(
        run(
            Op::WindowPartition { window: 3 },
            &[Tensor::from_i32([1; 4], &[7]).unwrap()]
        )
        .is_err()
    );
    assert!(
        run(
            Op::GetRelativePosition { query: 2, key: 3 },
            std::slice::from_ref(&a)
        )
        .is_err()
    );
    assert_eq!(a.to_bytes().unwrap(), before);
}
#[test]
fn flat_window_reads_cannot_escape_the_declared_view() {
    let bytes = vec![0; 400];
    let owned = Tensor::from_bytes(
        TensorDesc::contiguous(DType::F32, [100, 1, 1, 1]).unwrap(),
        bytes,
    )
    .unwrap();
    // Source flat indexing wants 4*3*3 words, but this overlapping view spans 9.
    let view = owned
        .view(
            TensorDesc::with_strides(DType::F32, [4, 3, 3, 1], [4, 4, 4, 4]).unwrap(),
            0,
        )
        .unwrap();
    assert!(matches!(
        run(Op::WindowPartition { window: 3 }, &[view]),
        Err(TensorError::Bounds)
    ));
    // Relative-position flat copies also stay within their view, even if the
    // underlying owner could satisfy the original out-of-view access.
    let half = Tensor::from_bytes(
        TensorDesc::contiguous(DType::F16, [100, 1, 1, 1]).unwrap(),
        vec![0; 200],
    )
    .unwrap();
    let view = half
        .view(
            TensorDesc::with_strides(DType::F16, [4, 5, 1, 1], [2, 2, 2, 2]).unwrap(),
            0,
        )
        .unwrap();
    assert!(matches!(
        run(Op::GetRelativePosition { query: 3, key: 3 }, &[view]),
        Err(TensorError::Bounds)
    ));
}
#[test]
fn relative_positions_preserve_bfloat_payloads_as_half_output() {
    let input = Tensor::from_bytes(
        TensorDesc::contiguous(DType::BF16, [2, 3, 1, 1]).unwrap(),
        [0x3f80u16, 0x7f81, 0x8000, 0x7fc1, 0x4000, 0xffff]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    )
    .unwrap();
    let out = run(Op::GetRelativePosition { query: 2, key: 2 }, &[input]).unwrap();
    assert_eq!(out.dtype(), DType::F16);
    assert_eq!(
        out.to_bytes().unwrap(),
        [
            0x8000u16, 0x7fc1, 0x3f80, 0x7f81, 0x4000, 0xffff, 0x8000, 0x7fc1
        ]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>()
    );
}
#[test]
fn triangular_solve_checks_singular_and_mismatched_systems() {
    let bad = Tensor::from_f32([2, 2, 1, 1], &[1., 99., 2., 0.]).unwrap();
    let b = zeros([3, 2, 1, 1]);
    let before = bad.to_bytes().unwrap();
    assert!(matches!(
        run(Op::SolveTriangular, &[bad.clone(), b.clone()]),
        Err(TensorError::Parameters(_))
    ));
    assert_eq!(bad.to_bytes().unwrap(), before);
    assert!(run(Op::SolveTriangular, &[zeros([2, 3, 1, 1]), b.clone()]).is_err());
    assert!(run(Op::SolveTriangular, &[zeros([2, 2, 2, 1]), b]).is_err());
    let a = Tensor::from_f32([2, 2, 1, 1], &[1., 77., 2., 3.]).unwrap();
    let b = Tensor::from_f32([1, 2, 1, 1], &[4., 11.]).unwrap();
    assert_eq!(
        run(Op::SolveTriangular, &[a, b]).unwrap().to_f32().unwrap(),
        [4., 1.]
    );
}
#[test]
fn relative_addition_rejects_inconsistent_or_noncontiguous_shapes() {
    assert!(
        run(
            Op::AddRelativePosition,
            &[
                zeros([4, 6, 1, 1]),
                zeros([2, 2, 3, 1]),
                zeros([2, 3, 3, 1])
            ]
        )
        .is_err()
    );
    let input = zeros([4, 6, 1, 1]).permute([1, 0, 2, 3]).unwrap();
    assert!(
        run(
            Op::AddRelativePosition,
            &[input, zeros([2, 2, 3, 1]), zeros([2, 2, 3, 1])]
        )
        .is_err()
    );
}
#[test]
fn cross_entropy_checks_native_row_addresses_and_nonfinite_domains() {
    let a = Tensor::from_f32([3, 2, 1, 1], &[1., 2., 3., 4., 5., 6.]).unwrap();
    assert!(run(Op::CrossEntropyLoss, &[a.clone(), zeros([3, 1, 1, 1])]).is_err());
    let nan = Tensor::from_f32([3, 2, 1, 1], &[1., 2., f32::NAN, 4., 5., 6.]).unwrap();
    assert!(run(Op::CrossEntropyLoss, &[a.clone(), nan.clone()]).is_err());
    assert!(
        run(
            Op::CrossEntropyLossBack,
            &[zeros([2, 1, 1, 1]), a.clone(), nan]
        )
        .is_err()
    );
    assert!(
        run(
            Op::CrossEntropyLossBack,
            &[
                zeros([1; 4]),
                a.permute([1, 0, 2, 3]).unwrap(),
                zeros([2, 3, 1, 1])
            ]
        )
        .is_err()
    );
    // ne[2] overlaps the first physical plane: flattened native row reads may
    // not extend to later bytes of the owner outside this view's exact span.
    let owner = zeros([100, 1, 1, 1]);
    let view = owner
        .view(
            TensorDesc::with_strides(DType::F32, [3, 2, 3, 1], [4, 12, 12, 12]).unwrap(),
            0,
        )
        .unwrap();
    assert!(matches!(
        run(Op::CrossEntropyLoss, &[view.clone(), view]),
        Err(TensorError::Bounds)
    ));
}
#[test]
fn cancellation_and_empty_domains_do_not_publish_outputs() {
    let mut graph = Graph::new();
    let input = graph.input(zeros([3, 5, 7, 1]));
    let target = graph
        .op(Op::WindowPartition { window: 3 }, &[input])
        .unwrap();
    assert!(matches!(
        graph.evaluate_with_cancel(target, &AtomicBool::new(true)),
        Err(TensorError::Cancelled)
    ));
    for op in [
        Op::WindowPartition { window: 2 },
        Op::WindowUnpartition {
            width: 2,
            height: 2,
            window: 2,
        },
    ] {
        assert!(run(op, &[zeros([3, 0, 2, 1])]).is_err());
    }
}

#[test]
fn flat_spans_and_large_padding_are_checked_before_allocating() {
    let huge = i32::MAX as usize;
    let overlap = Tensor::from_bytes(
        TensorDesc::with_strides(DType::F32, [1, huge, 1, 1], [4, 0, 0, 0]).unwrap(),
        vec![0; 4],
    )
    .unwrap();
    assert!(matches!(
        run(
            Op::WindowPartition { window: 1 },
            std::slice::from_ref(&overlap)
        ),
        Err(TensorError::Bounds)
    ));
    assert!(matches!(
        run(
            Op::WindowUnpartition {
                width: huge,
                height: 1,
                window: 1
            },
            &[zeros([1; 4])]
        ),
        Err(TensorError::Bounds)
    ));
    let half = Tensor::from_bytes(
        TensorDesc::with_strides(DType::F16, [1, huge, 1, 1], [2, 0, 0, 0]).unwrap(),
        vec![0; 2],
    )
    .unwrap();
    assert!(matches!(
        run(
            Op::GetRelativePosition {
                query: huge.div_ceil(2),
                key: huge.div_ceil(2)
            },
            &[half]
        ),
        Err(TensorError::Bounds)
    ));
    assert!(matches!(
        run(Op::WindowPartition { window: 1_000_000 }, &[zeros([1; 4])]),
        Err(TensorError::Unsupported(_))
    ));
    assert!(matches!(
        run(Op::CrossEntropyLoss, &[overlap.clone(), overlap]),
        Err(TensorError::Unsupported(_))
    ));
    // When stored windows are smaller, the greatest physical read may belong
    // to a complete earlier block rather than the final output coordinate.
    assert!(
        run(
            Op::WindowUnpartition {
                width: 4,
                height: 4,
                window: 3
            },
            &[zeros([1, 1, 1, 5])]
        )
        .is_ok()
    );
    assert!(matches!(
        run(
            Op::WindowUnpartition {
                width: 4,
                height: 4,
                window: 3
            },
            &[zeros([1, 1, 1, 4])]
        ),
        Err(TensorError::Bounds)
    ));
}
