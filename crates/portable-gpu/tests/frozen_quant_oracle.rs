//! Opt-in differential tests only. The production crates never load ggml.
use llama_kernel_ir::{BufferData, execute, half_to_float, kernels};
use llama_platform_loader::DynamicLibrary;
use std::collections::BTreeMap;
use std::ffi::c_void;

#[repr(C)]
struct InitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ4 {
    scale: u16,
    quants: [u8; 16],
}
type Init = unsafe extern "C" fn(InitParams) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Dequant = unsafe extern "C" fn(*const BlockQ4, *mut f32, i64);
type Half = unsafe extern "C" fn(u16) -> f32;

#[test]
#[ignore = "requires LLAMA_REFERENCE_LIBRARY pointing at an isolated frozen GGML reference build"]
fn q4_and_half_match_frozen_ggml_symbols() {
    let path =
        std::env::var("LLAMA_REFERENCE_LIBRARY").expect("set isolated reference library path");
    let library = DynamicLibrary::open(&[&path]).unwrap();
    // SAFETY: exact frozen C signatures, repr(C) block/init layout, and the
    // library is retained until every call and reference context is released.
    unsafe {
        let init: Init = library.symbol(c"ggml_init").unwrap();
        let free: Free = library.symbol(c"ggml_free").unwrap();
        let dequant: Dequant = library.symbol(c"dequantize_row_q4_0").unwrap();
        let half: Half = library.symbol(c"ggml_fp16_to_fp32").unwrap();
        let context = init(InitParams {
            mem_size: 1024 * 1024,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        });
        assert!(!context.is_null());
        assert_eq!(std::mem::size_of::<BlockQ4>(), 18);
        for bits in 0..=u16::MAX {
            let actual = half_to_float(bits);
            let expected = half(bits);
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "half {bits:04x}");
            }
        }
        let mut x = vec![0.0f32; 128];
        for i in 0..64 {
            x[1 + i * 2] = (i as f32 - 30.0) * 0.125;
        }
        let mut storage = vec![0x99u8; 3 + 2 * 41 + 36];
        let mut expected = vec![];
        for row in 0..3usize {
            let mut blocks = [BlockQ4 {
                scale: 0,
                quants: [0; 16],
            }; 2];
            for (block, b) in blocks.iter_mut().enumerate() {
                b.scale = [0x3c00, 0xb800, 0x3400, 0x4000, 0x0001, 0x8000][row * 2 + block];
                for (j, q) in b.quants.iter_mut().enumerate() {
                    *q = (((j + row + block) % 16) as u8) | ((((15 - j + row) % 16) as u8) << 4);
                }
                let base = 3 + row * 41 + block * 18;
                storage[base..base + 2].copy_from_slice(&b.scale.to_le_bytes());
                storage[base + 2..base + 18].copy_from_slice(&b.quants);
            }
            let mut expanded = [0f32; 64];
            dequant(blocks.as_ptr(), expanded.as_mut_ptr(), 64);
            expected.push(
                expanded
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (*w as f64) * (x[1 + i * 2] as f64))
                    .sum::<f64>() as f32,
            );
        }
        let kernel = kernels::q4_0_matvec(
            3,
            64,
            3,
            41,
            kernels::VectorLayout {
                length: 64,
                offset: 1,
                stride: 2,
            },
        )
        .unwrap();
        let mut buffers = BTreeMap::from([
            (
                "packed".into(),
                BufferData::U32(kernels::bytes_to_words(&storage)),
            ),
            ("vector".into(), BufferData::F32(x)),
            ("dst".into(), BufferData::F32(vec![0.0; 3])),
        ]);
        execute(&kernel, &mut buffers, 100000).unwrap();
        let BufferData::F32(actual) = &buffers["dst"] else {
            panic!("wrong output type")
        };
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} != {b}");
        }
        println!("65536 binary16 encodings matched frozen GGML (NaNs compared by class)");
        println!(
            "unaligned/strided Q4_0 matvec: Rust={actual:?}; frozen GGML dequant + F64 dot={expected:?}"
        );
        free(context);
    }
}
