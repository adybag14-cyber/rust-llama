use llama_tensor::{
    Conv2dParams, Conv3dParams, DType, Graph, Op, PoolMode, ResizeMode, Tensor, TensorDesc,
};
struct Reader {
    bytes: &'static [u8],
    at: usize,
}
impl Reader {
    fn take(&mut self, n: usize) -> &'static [u8] {
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        out
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn i64(&mut self) -> i64 {
        i64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn string(&mut self) -> &'static str {
        let n = self.u32() as usize;
        std::str::from_utf8(self.take(n)).unwrap()
    }
    fn tensor(&mut self, strided: bool) -> Tensor {
        let dtype = DType::from_id(self.u32()).unwrap();
        let shape = std::array::from_fn(|_| self.u64());
        let desc = if strided {
            TensorDesc::with_strides(dtype, shape, std::array::from_fn(|_| self.u64())).unwrap()
        } else {
            TensorDesc::contiguous(dtype, shape).unwrap()
        };
        let bytes = self.u64();
        Tensor::from_bytes(desc, self.take(bytes).to_vec()).unwrap()
    }
}

fn evaluate(op: Op, inputs: Vec<Tensor>) -> llama_tensor::Result<Tensor> {
    let mut graph = Graph::new();
    let ids = inputs
        .into_iter()
        .map(|t| graph.input(t))
        .collect::<Vec<_>>();
    let target = graph.op(op, &ids)?;
    graph.evaluate(target)
}
fn params2(ints: &[i64]) -> Conv2dParams {
    Conv2dParams {
        stride: [ints[0] as i32, ints[1] as i32],
        padding: [ints[2] as i32, ints[3] as i32],
        dilation: [ints[4] as i32, ints[5] as i32],
    }
}
fn params3(ints: &[i64]) -> Conv3dParams {
    Conv3dParams {
        stride: std::array::from_fn(|i| ints[i] as i32),
        padding: std::array::from_fn(|i| ints[3 + i] as i32),
        dilation: std::array::from_fn(|i| ints[6 + i] as i32),
    }
}
fn mode(i: i64) -> PoolMode {
    match i {
        0 => PoolMode::Max,
        1 => PoolMode::Average,
        _ => panic!("invalid pool mode"),
    }
}
fn operation(tag: &str, ints: &[i64], floats: &[f32]) -> Op {
    match tag {
        "im2col" => Op::Im2col {
            params: params2(ints),
            is_2d: ints[6] != 0,
            dtype: DType::from_id(ints[7] as u32).unwrap(),
        },
        "im2col_back" => Op::Im2colBack {
            params: params2(ints),
            is_2d: ints[6] != 0,
            shape: std::array::from_fn(|i| ints[7 + i] as usize),
        },
        "im2col_3d" => Op::Im2col3d {
            params: params3(&ints[1..]),
            input_channels: ints[0] as usize,
            dtype: DType::from_id(ints[10] as u32).unwrap(),
        },
        "col2im_1d" => Op::Col2im1d {
            stride: ints[0] as usize,
            output_channels: ints[1] as usize,
            padding: ints[2] as usize,
        },
        "conv_transpose_1d" => Op::ConvTranspose1d {
            stride: ints[0] as usize,
        },
        "conv_transpose_2d" => Op::ConvTranspose2d {
            stride: ints[0] as usize,
        },
        "conv_2d_direct" => Op::Conv2d {
            params: params2(ints),
        },
        "conv_2d_dw_direct" => Op::Conv2dDepthwise {
            params: params2(ints),
        },
        "conv_3d_direct" => Op::Conv3d {
            params: params3(ints),
            input_channels: ints[9] as usize,
        },
        "pool_1d" => Op::Pool1d {
            mode: mode(ints[0]),
            kernel: ints[1] as usize,
            stride: ints[2] as usize,
            padding: ints[3] as i32,
        },
        "pool_2d" => Op::Pool2d {
            mode: mode(ints[0]),
            kernel: [ints[1] as usize, ints[2] as usize],
            stride: [ints[3] as usize, ints[4] as usize],
            padding: [floats[0], floats[1]],
        },
        "pool_2d_back" => Op::Pool2dBack {
            mode: mode(ints[0]),
            kernel: [ints[1] as usize, ints[2] as usize],
            stride: [ints[3] as usize, ints[4] as usize],
            padding: [floats[0], floats[1]],
        },
        "interpolate" => Op::Resize {
            shape: std::array::from_fn(|i| ints[i] as usize),
            mode: match ints[4] & 0xff {
                0 => ResizeMode::Nearest,
                1 => ResizeMode::Bilinear,
                2 => ResizeMode::Bicubic,
                _ => panic!("invalid resize mode"),
            },
            align_corners: ints[4] & (1 << 8) != 0,
            antialias: ints[4] & (1 << 9) != 0,
        },
        _ => panic!("unknown native operation {tag}"),
    }
}
fn compare_fixture(bytes: &'static [u8], expected_cases: u32) {
    let mut r = Reader { bytes, at: 0 };
    assert_eq!(r.take(8), b"GGTR0002");
    let cases = r.u32();
    let mut compared_bytes = 0;
    let mut failures = Vec::new();
    for _ in 0..cases {
        let name = r.string();
        let tag = r.string();
        assert_eq!(r.u32(), 0, "exact source byte capture");
        let ni = r.u32();
        let ints = (0..ni).map(|_| r.i64()).collect::<Vec<_>>();
        let nf = r.u32();
        let floats = (0..nf).map(|_| r.f32()).collect::<Vec<_>>();
        let nt = r.u32();
        let inputs = (0..nt).map(|_| r.tensor(true)).collect::<Vec<_>>();
        let expected = r.tensor(false);
        let split = ints.len() - 4;
        let expected_nb: [usize; 4] = std::array::from_fn(|i| ints[split + i] as usize);
        let op = operation(tag, &ints[..split], &floats);
        let result = match evaluate(op, inputs) {
            Ok(result) => result,
            Err(error) => {
                failures.push(format!("{name}: error {error}"));
                continue;
            }
        };
        assert_eq!(result.dtype(), expected.dtype(), "{name} dtype");
        assert_eq!(result.shape(), expected.shape(), "{name} shape");
        assert_eq!(result.desc().nb, expected_nb, "{name} native byte strides");
        let expected = expected.to_bytes().unwrap();
        let actual = result.to_bytes().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name} length");
        compared_bytes += expected.len();
        if expected != actual {
            let at = actual
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap();
            failures.push(format!(
                "{name}: byte {at}: got {:?}, expected {:?}",
                &actual[at..(at + 16).min(actual.len())],
                &expected[at..(at + 16).min(expected.len())]
            ));
        }
    }
    assert_eq!(r.at, bytes.len());
    assert_eq!(cases, expected_cases);
    println!(
        "original graph cases={cases}, compared bytes={compared_bytes}, mismatches={}",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn convolution_extraction_pooling_resize_match_original_graph_bytes_and_strides() {
    compare_fixture(
        include_bytes!("fixtures/cpu-convolution-reference.bin"),
        446,
    );
}
#[test]
fn native_reduction_payloads_and_unit_axis_dispatch_preserve_source_bits() {
    compare_fixture(
        include_bytes!("fixtures/cpu-convolution-edges-reference.bin"),
        490,
    );
}

#[test]
fn native_signed_convolution_sizes_and_fractional_pool_padding_keep_original_geometry() {
    compare_fixture(
        include_bytes!("fixtures/cpu-convolution-corners-reference.bin"),
        25,
    );
}
