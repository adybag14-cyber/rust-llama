use llama_quant::DType;
use llama_quant::flash_attention::{Inputs, KvRows, Params, View4};

struct Reader {
    bytes: &'static [u8],
    at: usize,
}
impl Reader {
    fn take(&mut self, count: usize) -> &'static [u8] {
        let bytes = &self.bytes[self.at..self.at + count];
        self.at += count;
        bytes
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn usize(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn string(&mut self) -> &'static str {
        let n = self.u32() as usize;
        std::str::from_utf8(self.take(n)).unwrap()
    }
    fn view(&mut self) -> RawInput {
        let dtype = DType::from_id(self.u32()).unwrap();
        let shape = std::array::from_fn(|_| self.usize());
        let strides = std::array::from_fn(|_| self.usize());
        let size = self.usize();
        let raw = self.take(size);
        RawInput {
            dtype,
            shape,
            strides,
            raw,
        }
    }
}
#[derive(Clone, Copy)]
pub struct RawInput {
    pub dtype: DType,
    pub shape: [usize; 4],
    pub strides: [usize; 4],
    pub raw: &'static [u8],
}
impl RawInput {
    pub fn view(self) -> View4<'static> {
        View4::from_bytes(self.dtype, self.raw, self.shape, self.strides, 0).unwrap()
    }
}
pub struct Case {
    pub cohort: &'static str,
    pub name: &'static str,
    pub inputs: Inputs<'static>,
    pub raw_inputs: Vec<RawInput>,
    pub rows: KvRows<'static>,
    pub params: Params,
    pub shape: [usize; 4],
    pub expected: &'static [u8],
    pub forced: &'static [u8],
}
pub fn load() -> Vec<Case> {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("../fixtures/flash-rows/msvc-reference.bin").as_slice();
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("../fixtures/flash-rows/gnu-reference.bin").as_slice();
    load_files(
        include_bytes!("../fixtures/flash-rows/inputs.bin"),
        bytes,
        None,
        576,
    )
}
pub fn load_gemm_order() -> Vec<Case> {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("../fixtures/flash-rows-gemm-order/msvc-reference.bin").as_slice();
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("../fixtures/flash-rows-gemm-order/gnu-reference.bin").as_slice();
    load_files(
        include_bytes!("../fixtures/flash-rows-gemm-order/inputs.bin"),
        bytes,
        None,
        552,
    )
}
pub fn load_qk_order() -> Vec<Case> {
    #[cfg(target_os = "windows")]
    let bytes = include_bytes!("../fixtures/flash-rows-qk-order/msvc-reference.bin").as_slice();
    #[cfg(target_os = "linux")]
    let bytes = include_bytes!("../fixtures/flash-rows-qk-order/gnu-reference.bin").as_slice();
    load_files(
        include_bytes!("../fixtures/flash-rows-qk-order/inputs.bin"),
        bytes,
        None,
        402,
    )
}
fn load_files(
    pool_bytes: &'static [u8],
    bytes: &'static [u8],
    expected_inputs: Option<u32>,
    expected_cases: u32,
) -> Vec<Case> {
    let mut pool = Reader {
        bytes: pool_bytes,
        at: 0,
    };
    assert_eq!(pool.take(8), b"FAIN0001");
    let count = pool.u32();
    if let Some(expected_inputs) = expected_inputs {
        assert_eq!(count, expected_inputs);
    }
    let tensors = (0..count).map(|_| pool.view()).collect::<Vec<_>>();
    assert_eq!(pool.at, pool_bytes.len());

    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(reader.take(8), b"FARW0001");
    let count = reader.u32();
    assert_eq!(count, expected_cases);
    let mut cases = Vec::new();
    for _ in 0..count {
        let cohort = reader.string();
        let name = reader.string();
        let workers = reader.u32() as usize;
        let mask = reader.u32() != 0;
        let sinks = reader.u32() != 0;
        let scale = reader.f32();
        let max_bias = reader.f32();
        let logit_softcap = reader.f32();
        let ni = reader.u32();
        assert_eq!(ni, 4 + u32::from(mask) + u32::from(sinks));
        let raw_inputs = (0..ni)
            .map(|_| tensors[reader.u32() as usize])
            .collect::<Vec<_>>();
        let row_data = raw_inputs.last().copied().unwrap();
        assert_eq!(row_data.dtype, DType::I32);
        assert_eq!(&row_data.shape[2..], &[1, 1]);
        assert_eq!(row_data.strides[0], 4);
        assert_eq!(row_data.strides[1], row_data.shape[0] * 4);
        let rows =
            KvRows::from_bytes(row_data.raw, [row_data.shape[0], row_data.shape[1]], 0).unwrap();
        let inputs = raw_inputs[..raw_inputs.len() - 1]
            .iter()
            .copied()
            .map(RawInput::view)
            .collect::<Vec<_>>();
        let shape = std::array::from_fn(|_| reader.usize());
        let size = reader.usize();
        let expected = reader.take(size);
        let size = reader.usize();
        let forced = reader.take(size);
        assert_eq!(expected.len(), forced.len());
        cases.push(Case {
            cohort,
            name,
            shape,
            expected,
            forced,
            raw_inputs,
            rows,
            inputs: Inputs {
                q: inputs[0],
                k: inputs[1],
                v: inputs[2],
                mask: mask.then(|| inputs[3]),
                sinks: sinks.then(|| inputs[3 + usize::from(mask)]),
            },
            params: Params {
                workers,
                scale,
                max_bias,
                logit_softcap,
                ..Params::default()
            },
        });
    }
    assert_eq!(reader.at, bytes.len());
    cases
}
