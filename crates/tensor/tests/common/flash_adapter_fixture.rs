use llama_quant::flash_attention::Params;
use llama_tensor::{DType, FlashAttentionOptions, Tensor, TensorDesc};

struct Reader {
    bytes: &'static [u8],
    at: usize,
}
impl Reader {
    fn take(&mut self, count: usize) -> &'static [u8] {
        let end = self.at.checked_add(count).unwrap();
        let out = self.bytes.get(self.at..end).unwrap();
        self.at = end;
        out
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn usize(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn float(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn string(&mut self) -> &'static str {
        let length = self.u32() as usize;
        std::str::from_utf8(self.take(length)).unwrap()
    }
    fn tensor(&mut self) -> Tensor {
        let dtype = DType::from_id(self.u32()).unwrap();
        let shape = std::array::from_fn(|_| self.usize());
        let strides = std::array::from_fn(|_| self.usize());
        let origin = self.usize();
        let length = self.usize();
        let raw = self.take(length);
        assert!(length.is_multiple_of(dtype.type_size()));
        let owner_elements = length / dtype.type_size() * dtype.block_size();
        let owner = Tensor::from_bytes(
            TensorDesc::contiguous(dtype, [owner_elements, 1, 1, 1]).unwrap(),
            raw.to_vec(),
        )
        .unwrap();
        owner
            .view(
                TensorDesc::with_strides(dtype, shape, strides).unwrap(),
                origin,
            )
            .unwrap()
    }
}
pub struct Case {
    pub name: &'static str,
    pub domain: u32,
    pub flags: u32,
    pub options: FlashAttentionOptions,
    pub branch: u32,
    pub inputs: Vec<Tensor>,
    pub shape: [usize; 4],
    pub expected: Option<&'static [u8]>,
    pub forced: Option<&'static [u8]>,
}
pub fn load(bytes: &'static [u8], rejected: bool) -> Vec<Case> {
    let mut reader = Reader { bytes, at: 0 };
    assert_eq!(
        reader.take(8),
        if rejected { b"TFAR0001" } else { b"TFAT0001" }
    );
    let count = reader.u32();
    let mut cases = Vec::new();
    for _ in 0..count {
        let name = reader.string();
        let domain = reader.u32();
        let workers = reader.u32() as usize;
        let flags = reader.u32();
        let precision = reader.i32();
        let n_kv_max = reader.i32();
        let scale = reader.float();
        let max_bias = reader.float();
        let logit_softcap = reader.float();
        let branch = reader.u32();
        let inputs = reader.u32();
        assert_eq!(inputs, 3 + (flags & 7).count_ones());
        let inputs = (0..inputs).map(|_| reader.tensor()).collect::<Vec<_>>();
        let shape;
        let expected;
        let forced;
        if rejected {
            shape = [0; 4];
            expected = None;
            forced = None;
        } else {
            assert_eq!(reader.u32(), DType::F32.id());
            shape = std::array::from_fn(|_| reader.usize());
            let length = reader.usize();
            expected = Some(reader.take(length));
            let length = reader.usize();
            forced = Some(reader.take(length));
        }
        cases.push(Case {
            name,
            domain,
            flags,
            branch,
            inputs,
            shape,
            expected,
            forced,
            options: FlashAttentionOptions {
                params: Params {
                    scale,
                    max_bias,
                    logit_softcap,
                    workers,
                    ..Params::default()
                },
                precision,
                n_kv_max,
                ..FlashAttentionOptions::default()
            },
        });
    }
    assert_eq!(reader.at, bytes.len());
    cases
}
