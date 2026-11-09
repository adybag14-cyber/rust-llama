use llama_tensor::{
    Conv2dParams, Conv3dParams, DType, Graph, Op, PoolMode, ResizeMode, Tensor, TensorDesc,
    TensorError,
};
use std::sync::atomic::AtomicBool;
fn p2() -> Conv2dParams {
    Conv2dParams {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    }
}
fn p3() -> Conv3dParams {
    Conv3dParams {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
    }
}
fn values(shape: [usize; 4]) -> Tensor {
    let count = shape.iter().product();
    Tensor::from_f32(
        shape,
        &(0..count)
            .map(|i| i as f32 * 0.125 - 1.0)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn typed(shape: [usize; 4], dtype: DType) -> Tensor {
    let source = values(shape).to_f32().unwrap();
    Tensor::from_bytes(
        TensorDesc::contiguous(dtype, shape).unwrap(),
        llama_quant::quantize(dtype, &source).unwrap(),
    )
    .unwrap()
}
fn evaluate(op: Op, inputs: Vec<Tensor>) -> llama_tensor::Result<Tensor> {
    let mut graph = Graph::new();
    let inputs = inputs
        .into_iter()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(op, &inputs)?;
    graph.evaluate(target)
}
#[test]
fn image_kernels_reject_native_undefined_dimensions_and_parameters_before_work() {
    let image = values([4, 3, 2, 1]);
    let kernel = values([2, 2, 2, 3]);
    for params in [
        Conv2dParams {
            stride: [0, 1],
            ..p2()
        },
        Conv2dParams {
            stride: [1, -1],
            ..p2()
        },
        Conv2dParams {
            dilation: [0, 1],
            ..p2()
        },
        Conv2dParams {
            dilation: [1, -1],
            ..p2()
        },
        Conv2dParams {
            padding: [i32::MAX, 0],
            ..p2()
        },
    ] {
        assert!(evaluate(Op::Conv2d { params }, vec![kernel.clone(), image.clone()]).is_err());
        assert!(
            evaluate(
                Op::Im2col {
                    params,
                    is_2d: true,
                    dtype: DType::F32
                },
                vec![kernel.clone(), image.clone()]
            )
            .is_err()
        );
    }
    for op in [
        Op::Pool1d {
            mode: PoolMode::Max,
            kernel: 0,
            stride: 1,
            padding: 0,
        },
        Op::Pool1d {
            mode: PoolMode::Average,
            kernel: 1,
            stride: 0,
            padding: 0,
        },
        Op::Pool1d {
            mode: PoolMode::Average,
            kernel: usize::MAX,
            stride: 1,
            padding: 0,
        },
        Op::Pool1d {
            mode: PoolMode::Max,
            kernel: 1,
            stride: 1,
            padding: i32::MIN,
        },
        Op::Pool2d {
            mode: PoolMode::Max,
            kernel: [2, 2],
            stride: [1, 0],
            padding: [0., 0.],
        },
        Op::Pool2d {
            mode: PoolMode::Average,
            kernel: [2, 2],
            stride: [1, 1],
            padding: [f32::NAN, 0.],
        },
        Op::Pool2d {
            mode: PoolMode::Max,
            kernel: [2, 2],
            stride: [1, 1],
            padding: [0., f32::INFINITY],
        },
        Op::Resize {
            shape: [0, 2, 2, 1],
            mode: ResizeMode::Nearest,
            align_corners: false,
            antialias: false,
        },
        Op::Resize {
            shape: [usize::MAX, 2, 2, 1],
            mode: ResizeMode::Nearest,
            align_corners: false,
            antialias: false,
        },
        Op::Resize {
            shape: [4, 3, 2, 1],
            mode: ResizeMode::Nearest,
            align_corners: false,
            antialias: true,
        },
    ] {
        assert!(evaluate(op, vec![image.clone()]).is_err());
    }
    assert_eq!(
        evaluate(
            Op::Resize {
                shape: [i32::MAX as usize; 4],
                mode: ResizeMode::Nearest,
                align_corners: false,
                antialias: false
            },
            vec![image.clone()]
        )
        .unwrap_err(),
        TensorError::Overflow
    );
    let empty = Tensor::from_bytes(
        TensorDesc::with_strides(
            DType::F32,
            [0, usize::MAX, usize::MAX, usize::MAX],
            [4, 0, 0, 0],
        )
        .unwrap(),
        Vec::new(),
    )
    .unwrap();
    assert!(matches!(
        evaluate(
            Op::Pool1d {
                mode: PoolMode::Max,
                kernel: 1,
                stride: 1,
                padding: 0
            },
            vec![empty]
        ),
        Err(TensorError::Shape(_))
    ));
    assert!(
        evaluate(
            Op::ConvTranspose1d { stride: 0 },
            vec![values([2, 1, 2, 1]), values([4, 2, 1, 1])]
        )
        .is_err()
    );
    assert!(
        evaluate(
            Op::ConvTranspose2d { stride: usize::MAX },
            vec![kernel, image]
        )
        .is_err()
    );
}
#[test]
fn packed_channel_contracts_and_actual_source_dtypes_are_enforced() {
    let im = values([4, 3, 2, 1]);
    let k = values([2, 2, 2, 3]);
    assert!(matches!(
        evaluate(
            Op::Conv2d { params: p2() },
            vec![typed([2, 2, 2, 3], DType::BF16), im.clone()]
        ),
        Err(TensorError::Unsupported(_))
    ));
    assert!(matches!(
        evaluate(
            Op::Conv2d { params: p2() },
            vec![k.clone(), typed([4, 3, 2, 1], DType::F16)]
        ),
        Err(TensorError::Type {
            expected: DType::F32,
            actual: DType::F16
        })
    ));
    assert!(matches!(
        evaluate(
            Op::Im2col {
                params: p2(),
                is_2d: true,
                dtype: DType::F32
            },
            vec![k, typed([4, 3, 2, 1], DType::F16)]
        ),
        Err(TensorError::Unsupported(_))
    ));
    assert!(
        evaluate(
            Op::Conv2dDepthwise { params: p2() },
            vec![values([2, 2, 2, 2]), im]
        )
        .is_err()
    );
    for channels in [0, 3] {
        assert!(
            evaluate(
                Op::Conv3d {
                    params: p3(),
                    input_channels: channels
                },
                vec![values([2, 2, 2, 4]), values([4, 3, 3, 2])]
            )
            .is_err()
        );
        assert!(
            evaluate(
                Op::Im2col3d {
                    params: p3(),
                    input_channels: channels,
                    dtype: DType::F32
                },
                vec![values([2, 2, 2, 4]), values([4, 3, 3, 2])]
            )
            .is_err()
        );
    }
    for (stride, channels, padding) in [
        (0, 2, 0),
        (1, 0, 0),
        (1, 3, 0),
        (1, 2, usize::MAX),
        (1, 2, 100),
    ] {
        assert!(
            evaluate(
                Op::Col2im1d {
                    stride,
                    output_channels: channels,
                    padding
                },
                vec![values([4, 3, 1, 1])]
            )
            .is_err()
        );
    }
    assert!(matches!(
        evaluate(
            Op::Pool2dBack {
                mode: PoolMode::Max,
                kernel: [2, 2],
                stride: [1, 1],
                padding: [0., 0.]
            },
            vec![values([3, 2, 1, 1]), typed([4, 3, 1, 1], DType::F16)]
        ),
        Err(TensorError::Type {
            expected: DType::F32,
            actual: DType::F16
        })
    ));
    assert!(
        evaluate(
            Op::Im2colBack {
                params: p2(),
                is_2d: true,
                shape: [4, 3, 2, 1]
            },
            vec![values([3, 2, 1, 1]), values([2, 2, 2, 1])]
        )
        .is_err()
    );
}
#[test]
fn flat_native_addressing_cannot_escape_a_declared_view_or_mutate_aliases() {
    let storage = values([64, 1, 1, 1]);
    let original = storage.to_bytes().unwrap();
    // Both logical rows are inside 32 bytes, but native POOL_1D flattens higher
    // axes using nb1=100. Its second physical row would escape this view even
    // though the larger shared allocation has bytes at that address.
    let view = storage
        .view(
            TensorDesc::with_strides(DType::F32, [4, 1, 2, 1], [4, 100, 16, 32]).unwrap(),
            0,
        )
        .unwrap();
    let before = view.to_bytes().unwrap();
    assert_eq!(
        evaluate(
            Op::Pool1d {
                mode: PoolMode::Average,
                kernel: 2,
                stride: 1,
                padding: 0
            },
            vec![view.clone()]
        )
        .unwrap_err(),
        TensorError::Bounds
    );
    assert_eq!(view.to_bytes().unwrap(), before);
    assert_eq!(storage.to_bytes().unwrap(), original);
    assert!(view.shares_storage(&storage));
    let planes = storage
        .view(
            TensorDesc::with_strides(DType::F32, [4, 2, 1, 2], [4, 16, 128, 32]).unwrap(),
            0,
        )
        .unwrap();
    assert!(matches!(
        evaluate(
            Op::Pool2d {
                mode: PoolMode::Max,
                kernel: [2, 2],
                stride: [1, 1],
                padding: [0., 0.]
            },
            vec![planes]
        ),
        Err(TensorError::Layout(_))
    ));
    let misaligned = storage
        .view(
            TensorDesc::with_strides(DType::F32, [2, 2, 1, 1], [4, 9, 18, 18]).unwrap(),
            0,
        )
        .unwrap();
    assert!(matches!(
        evaluate(
            Op::Resize {
                shape: [2, 2, 1, 1],
                mode: ResizeMode::Nearest,
                align_corners: false,
                antialias: false
            },
            vec![misaligned]
        ),
        Err(TensorError::Layout(_))
    ));
    let output = evaluate(
        Op::Pool1d {
            mode: PoolMode::Average,
            kernel: 2,
            stride: 2,
            padding: 0,
        },
        vec![storage.clone()],
    )
    .unwrap();
    assert!(!output.shares_storage(&storage));
    assert_eq!(storage.to_bytes().unwrap(), original);
}
#[test]
fn cancellation_precedes_allocation_and_each_new_operation_keeps_input_storage() {
    let source = values([4, 3, 2, 1]);
    let before = source.to_bytes().unwrap();
    let pairs = vec![
        (
            Op::Im2col {
                params: p2(),
                is_2d: true,
                dtype: DType::F32,
            },
            vec![values([2, 2, 2, 3]), source.clone()],
        ),
        (
            Op::Im2colBack {
                params: p2(),
                is_2d: true,
                shape: [4, 3, 2, 1],
            },
            vec![values([8, 3, 2, 1]), values([2, 2, 2, 3])],
        ),
        (
            Op::Im2col3d {
                params: p3(),
                input_channels: 2,
                dtype: DType::F16,
            },
            vec![values([2, 2, 2, 4]), values([4, 3, 3, 2])],
        ),
        (
            Op::Col2im1d {
                stride: 1,
                output_channels: 2,
                padding: 0,
            },
            vec![values([4, 3, 1, 1])],
        ),
        (
            Op::ConvTranspose1d { stride: 1 },
            vec![values([2, 3, 2, 1]), values([4, 2, 1, 1])],
        ),
        (
            Op::Conv2d { params: p2() },
            vec![values([2, 2, 2, 3]), source.clone()],
        ),
        (
            Op::Conv3d {
                params: p3(),
                input_channels: 2,
            },
            vec![values([2, 2, 2, 4]), values([4, 3, 3, 2])],
        ),
        (
            Op::Conv2dDepthwise { params: p2() },
            vec![values([2, 2, 1, 2]), source.clone()],
        ),
        (
            Op::ConvTranspose2d { stride: 1 },
            vec![values([2, 2, 3, 2]), source.clone()],
        ),
        (
            Op::Pool1d {
                mode: PoolMode::Max,
                kernel: 2,
                stride: 1,
                padding: 0,
            },
            vec![source.clone()],
        ),
        (
            Op::Pool2d {
                mode: PoolMode::Average,
                kernel: [2, 2],
                stride: [1, 1],
                padding: [0., 0.],
            },
            vec![source.clone()],
        ),
        (
            Op::Pool2dBack {
                mode: PoolMode::Average,
                kernel: [2, 2],
                stride: [1, 1],
                padding: [0., 0.],
            },
            vec![values([3, 2, 2, 1]), source.clone()],
        ),
        (
            Op::Resize {
                shape: [i32::MAX as usize; 4],
                mode: ResizeMode::Bilinear,
                align_corners: false,
                antialias: false,
            },
            vec![source.clone()],
        ),
    ];
    for (op, inputs) in pairs {
        let mut graph = Graph::new();
        let ids = inputs
            .into_iter()
            .map(|t| graph.input(t))
            .collect::<Vec<_>>();
        let node = graph.op(op, &ids).unwrap();
        assert_eq!(
            graph
                .evaluate_with_cancel(node, &AtomicBool::new(true))
                .unwrap_err(),
            TensorError::Cancelled
        );
    }
    assert_eq!(source.to_bytes().unwrap(), before);
}
