use llama_tensor::{DType, Graph, Op, OpCode, Tensor, TensorDesc, TensorError};

fn values(shape: [usize; 4], value: f32) -> Tensor {
    Tensor::from_f32(shape, &vec![value; shape.iter().product()]).unwrap()
}
fn storage(dtype: DType, shape: [usize; 4]) -> Tensor {
    let desc = TensorDesc::contiguous(dtype, shape).unwrap();
    Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap()
}
fn view(dtype: DType, shape: [usize; 4], strides: [usize; 4]) -> Tensor {
    let desc = TensorDesc::with_strides(dtype, shape, strides).unwrap();
    Tensor::from_bytes(desc, vec![0; desc.byte_span().unwrap()]).unwrap()
}
fn run(inputs: &[Tensor]) -> Result<Tensor, TensorError> {
    let mut graph = Graph::new();
    let nodes = inputs
        .iter()
        .cloned()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(Op::LightningIndexer, &nodes)?;
    graph.evaluate(target)
}
fn inputs() -> Vec<Tensor> {
    vec![
        values([3, 2, 2, 4], 0.25),
        values([3, 1, 5, 4], 0.5),
        values([2, 2, 1, 4], 0.75),
        storage(DType::F16, [5, 2, 1, 2]),
    ]
}

#[test]
fn original_id_arity_and_missing_native_decoders_are_explicit() {
    assert_eq!(Op::LightningIndexer.code(), OpCode::LIGHTNING_INDEXER);
    assert_eq!(OpCode::LIGHTNING_INDEXER as u32, 87);
    let mut graph = Graph::new();
    let node = graph.input(values([1; 4], 0.0));
    for count in [0, 3, 5] {
        assert!(matches!(
            graph.op(Op::LightningIndexer, &vec![node; count]),
            Err(TensorError::InvalidArity { expected: 4, actual }) if actual == count
        ));
    }
    for dtype in [
        DType::Q8_1,
        DType::Q8_K,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F64,
    ] {
        let args = [
            values([256, 1, 1, 1], 0.25),
            storage(dtype, [256, 1, 1, 1]),
            values([1; 4], 1.0),
            storage(DType::F16, [1; 4]),
        ];
        assert!(
            matches!(run(&args), Err(TensorError::Unsupported(_))),
            "{dtype:?}"
        );
    }
}

#[test]
fn query_key_weight_and_mask_groups_are_checked() {
    let valid = inputs();
    assert!(run(&valid).is_ok());
    for (index, dtype, shape) in [
        (0, DType::F32, [4, 2, 2, 4]),
        (1, DType::F32, [3, 2, 5, 4]),
        (1, DType::F32, [3, 1, 5, 2]),
        (2, DType::F32, [3, 2, 1, 4]),
        (2, DType::F32, [2, 3, 1, 4]),
        (2, DType::F32, [2, 2, 2, 4]),
        (2, DType::F32, [2, 2, 1, 2]),
        (3, DType::F16, [5, 2, 1, 3]),
        (3, DType::F16, [5, 2, 1, 0]),
        (3, DType::F16, [4, 2, 1, 2]),
        (3, DType::F16, [5, 2, 2, 2]),
    ] {
        let mut bad = valid.clone();
        bad[index] = storage(dtype, shape);
        assert!(
            matches!(run(&bad), Err(TensorError::Shape(_))),
            "{index}: {shape:?}"
        );
    }
    for (index, dtype) in [(0, DType::F16), (2, DType::I32), (3, DType::BF16)] {
        let mut bad = valid.clone();
        bad[index] = storage(dtype, bad[index].shape());
        assert!(matches!(run(&bad), Err(TensorError::Type { .. })));
    }
}

#[test]
fn physical_inner_rows_and_scalar_alignment_are_required() {
    let scalar = vec![
        values([1; 4], 0.25),
        storage(DType::F16, [1; 4]),
        values([1; 4], 1.0),
        storage(DType::F16, [1; 4]),
    ];
    // Native nb0 assertions apply even to axes containing a single element.
    for index in 0..4 {
        let mut bad = scalar.clone();
        let dtype = bad[index].dtype();
        let size = dtype.type_size();
        bad[index] = view(dtype, [1; 4], [size * 2, size, size, size]);
        assert!(matches!(run(&bad), Err(TensorError::Layout(_))));
    }
    for index in [0, 2, 3] {
        let mut bad = inputs();
        let desc = *bad[index].desc();
        let mut strides = desc.nb;
        strides[1] += 1;
        bad[index] = view(desc.dtype, desc.ne, strides);
        assert!(matches!(run(&bad), Err(TensorError::Layout(_))));
    }
}

#[test]
fn aliased_shapes_cannot_amplify_output_or_scalar_work_without_bound() {
    let huge = i32::MAX as usize;
    let args = [
        storage(DType::F32, [0, 0, huge, 1]),
        storage(DType::F32, [0, 1, 1, 1]),
        storage(DType::F32, [0, huge, 1, 1]),
        view(DType::F16, [1, huge, 1, 1], [2, 0, 0, 0]),
    ];
    // Only two input bytes would otherwise produce almost 8 GiB of output.
    assert!(matches!(run(&args), Err(TensorError::Unsupported(_))));
    let mut too_wide = args;
    too_wide[0] = storage(DType::F32, [0, 0, huge + 1, 1]);
    assert!(matches!(run(&too_wide), Err(TensorError::Parameters(_))));

    let width = 32_768;
    let args = [
        view(DType::F32, [width, width, 1, 1], [4, 0, 0, 0]),
        values([width, 1, 1, 1], 0.5),
        values([width, 1, 1, 1], 0.5),
        storage(DType::F16, [1; 4]),
    ];
    // One tiny output with repeated query rows would require over a billion
    // scalar products, despite using less than 400 KiB of input storage.
    assert!(matches!(run(&args), Err(TensorError::Unsupported(_))));
}

#[test]
fn finite_score_boundary_keeps_source_relu_and_mask_behavior() {
    let original = vec![
        values([1; 4], f32::NAN),
        values([1; 4], 1.0),
        values([1; 4], 1.0),
        storage(DType::F16, [1; 4]),
    ];
    assert_eq!(run(&original).unwrap().to_f32().unwrap(), [0.0]);
    for bits in [0x7c00u16, 0xfc00, 0x7e51] {
        let mut args = original.clone();
        args[3] = Tensor::from_bytes(*args[3].desc(), bits.to_le_bytes().to_vec()).unwrap();
        let output = run(&args).unwrap().to_f32().unwrap()[0];
        let expected = match bits {
            0x7c00 => 0x7f80_0000,
            0xfc00 => 0xff80_0000,
            _ => 0x7fca_2000,
        };
        assert_eq!(output.to_bits(), expected);
    }
    for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args = original.clone();
        args[2] = values([1; 4], weight);
        assert!(matches!(run(&args), Err(TensorError::Unsupported(_))));
    }
    let mut overflow = original.clone();
    overflow[0] = values([1; 4], 1e30);
    overflow[1] = values([1; 4], 1e30);
    assert!(matches!(run(&overflow), Err(TensorError::Unsupported(_))));
    let before = original
        .iter()
        .map(|t| t.to_bytes().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(run(&original).unwrap().to_f32().unwrap(), [0.0]);
    for (input, bytes) in original.iter().zip(before) {
        assert_eq!(input.to_bytes().unwrap(), bytes);
    }
}
