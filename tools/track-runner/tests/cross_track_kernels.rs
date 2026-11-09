//! T06/T07 integration checks on the overlapping, explicitly supported subset.
//! Both are scalar semantic executions; no GPU result is implied.
use llama_device_family::ptx::{KernelKind, KernelSpec, PtxTarget, generate};
use llama_kernel_ir::{BufferData, execute, kernels::*};
use std::collections::BTreeMap;

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

#[test]
fn cuda_and_portable_ir_agree_on_broadcast_strides_and_launch_tails() {
    let count = 259u32;
    let spec = KernelSpec {
        kind: KernelKind::AddF32 {
            elements: count,
            lhs_stride: 2,
            rhs_stride: 0,
            output_stride: 3,
        },
    };
    let ptx = generate(&spec, PtxTarget::default()).unwrap();
    let lhs: Vec<f32> = (0..(count - 1) * 2 + 1).map(|i| i as f32 * 0.125).collect();
    let rhs = vec![-0.5];
    let out = vec![-123.0; ((count - 1) * 3 + 1) as usize];
    let mut cuda = vec![bytes(&lhs), bytes(&rhs), bytes(&out)];
    ptx.interpret(&mut cuda, 100000).unwrap();
    let kernel = pointwise(
        Pointwise::Add,
        VectorLayout {
            length: count,
            offset: 0,
            stride: 2,
        },
        Some(VectorLayout {
            length: count,
            offset: 0,
            stride: 0,
        }),
        VectorLayout {
            length: count,
            offset: 0,
            stride: 3,
        },
        1.0,
    )
    .unwrap();
    let mut portable = BTreeMap::from([
        ("lhs".into(), BufferData::F32(lhs)),
        ("rhs".into(), BufferData::F32(rhs)),
        ("dst".into(), BufferData::F32(out)),
    ]);
    execute(&kernel, &mut portable, 100000).unwrap();
    assert_eq!(portable["dst"], BufferData::F32(floats(&cuda[2])));
}

#[test]
fn cuda_and_portable_q4_ir_agree_on_frozen_blocks_and_padded_rows() {
    let fixture = include_bytes!("../../../crates/device-family/fixtures/q4_0_reference_input.bin");
    let mut packed = vec![0xabu8; 3 * 40 + 36];
    for row in 0..4 {
        packed[row * 40..row * 40 + 36].copy_from_slice(&fixture[row * 36..row * 36 + 36]);
    }
    let vector: Vec<f32> = (0..127).map(|i| ((i % 17) as f32 - 8.0) * 0.0625).collect();
    let spec = KernelSpec {
        kind: KernelKind::Q4MatVec {
            rows: 4,
            columns: 64,
            row_stride_bytes: 40,
            block_stride_bytes: 18,
            vector_stride: 2,
        },
    };
    let ptx = generate(&spec, PtxTarget::default()).unwrap();
    let mut cuda = vec![packed.clone(), bytes(&vector), vec![0; 16]];
    ptx.interpret(&mut cuda, 100000).unwrap();
    let kernel = q4_0_matvec(
        4,
        64,
        0,
        40,
        VectorLayout {
            length: 64,
            offset: 0,
            stride: 2,
        },
    )
    .unwrap();
    let mut portable = BTreeMap::from([
        ("packed".into(), BufferData::U32(bytes_to_words(&packed))),
        ("vector".into(), BufferData::F32(vector)),
        ("dst".into(), BufferData::F32(vec![0.0; 4])),
    ]);
    execute(&kernel, &mut portable, 100000).unwrap();
    let BufferData::F32(actual) = &portable["dst"] else {
        panic!("output type")
    };
    for (a, b) in actual.iter().zip(floats(&cuda[2])) {
        assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()));
    }
}
