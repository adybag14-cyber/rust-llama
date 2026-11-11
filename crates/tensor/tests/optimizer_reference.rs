use llama_tensor::{
    AdamWState, DType, Graph, Op, OptimizerLimits, Tensor, TensorDesc, TensorError, adamw_step,
    sgd_step,
};
use std::sync::atomic::AtomicBool;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.bytes[self.at..self.at + n];
        self.at += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn usize(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn string(&mut self) -> String {
        let n = self.u32() as usize;
        String::from_utf8(self.take(n).to_vec()).unwrap()
    }
    fn tensor(&mut self) -> Tensor {
        assert_eq!(self.u32(), 0);
        let ne = std::array::from_fn(|_| self.usize());
        let nb = std::array::from_fn(|_| self.usize());
        let n = self.usize();
        let raw = self.take(n).to_vec();
        Tensor::from_bytes(TensorDesc::with_strides(DType::F32, ne, nb).unwrap(), raw).unwrap()
    }
}
fn fixture() -> &'static [u8] {
    if cfg!(all(target_arch = "x86_64", target_env = "msvc")) {
        include_bytes!("fixtures/optimizer/windows-msvc.bin")
    } else {
        include_bytes!("fixtures/optimizer/linux-gnu.bin")
    }
}
fn physical(t: &Tensor) -> Vec<u8> {
    let count = t.desc().byte_span().unwrap() / 4;
    t.view(
        TensorDesc::contiguous(DType::F32, [count, 1, 1, 1]).unwrap(),
        0,
    )
    .unwrap()
    .to_bytes()
    .unwrap()
}
fn compare(name: &str, actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.desc(), expected.desc(), "{name}: shape/stride");
    let a = physical(actual);
    let e = physical(expected);
    if a != e {
        let first = a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(e.as_chunks::<4>().0)
            .position(|(a, e)| a != e)
            .unwrap();
        panic!(
            "{name}: word {first}: got {:08x}, expected {:08x}",
            u32::from_le_bytes(a[first * 4..first * 4 + 4].try_into().unwrap()),
            u32::from_le_bytes(e[first * 4..first * 4 + 4].try_into().unwrap())
        );
    }
}

#[test]
fn independent_native_optimizer_state_and_graphs() {
    let mut r = Reader {
        bytes: fixture(),
        at: 0,
    };
    assert_eq!(r.take(8), b"GGOS0001");
    let count = r.u32();
    assert_eq!(count, 300);
    let cancel = AtomicBool::new(false);
    let limits = OptimizerLimits::default();
    let mut output_bytes = 0;
    for _ in 0..count {
        let name = r.string();
        let kind = r.u32();
        let _threads = r.u32();
        let steps = r.u32();
        let n = r.u32();
        let inputs = (0..n).map(|_| r.tensor()).collect::<Vec<_>>();
        let n = r.u32();
        let expected = (0..n).map(|_| r.tensor()).collect::<Vec<_>>();
        output_bytes += expected
            .iter()
            .map(|t| t.desc().byte_span().unwrap())
            .sum::<usize>();
        if kind == 0 {
            let mut w = inputs[0].clone();
            for _ in 0..steps {
                w = sgd_step(&w, &inputs[1], &inputs[2], limits, &cancel).unwrap();
            }
            compare(&name, &w, &expected[0]);
        } else {
            let mut state =
                AdamWState::new(inputs[0].clone(), inputs[2].clone(), inputs[3].clone()).unwrap();
            for _ in 0..steps {
                state.step(&inputs[1], &inputs[4], limits, &cancel).unwrap();
            }
            for (i, t) in [&state.weights, &state.first_moment, &state.second_moment]
                .into_iter()
                .enumerate()
            {
                compare(&format!("{name} result {i}"), t, &expected[i]);
            }
        }
        // Real graph state edges feed every subsequent step. Moment projections
        // expose the same execution's state; no independently computed oracle.
        let mut g = Graph::new();
        let ids = inputs
            .iter()
            .cloned()
            .map(|t| g.input(t))
            .collect::<Vec<_>>();
        let targets = if kind == 0 {
            let mut w = ids[0];
            for _ in 0..steps {
                w = g
                    .op(Op::OptStepSgd { limits }, &[w, ids[1], ids[2]])
                    .unwrap();
            }
            vec![w]
        } else {
            let mut w = ids[0];
            let mut m = ids[2];
            let mut v = ids[3];
            for _ in 0..steps {
                let next = g.adamw_step([w, ids[1], m, v, ids[4]], limits).unwrap();
                [w, m, v] = next.as_array();
            }
            vec![w, m, v]
        };
        let actual = g.evaluate_many(&targets).unwrap();
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            compare(&format!("{name} graph {i}"), a, e);
        }
        for (i, t) in inputs.iter().enumerate() {
            compare(
                &format!("{name} immutable input {i}"),
                &g.evaluate(ids[i]).unwrap(),
                t,
            );
        }
    }
    assert_eq!(r.at, r.bytes.len());
    assert_eq!(output_bytes, 440704);
}

fn t(v: &[f32]) -> Tensor {
    Tensor::from_f32([v.len(), 1, 1, 1], v).unwrap()
}
fn state() -> AdamWState {
    AdamWState::new(t(&[1., 2., 3.]), t(&[0., 0., 0.]), t(&[0., 0., 0.])).unwrap()
}
fn parameters() -> Tensor {
    t(&[0.01, 0.9, 0.999, 1e-8, 0.01, 10., 1000.])
}

#[test]
fn failed_optimizer_updates_are_transactional() {
    let mut s = state();
    let before = [
        physical(&s.weights),
        physical(&s.first_moment),
        physical(&s.second_moment),
    ];
    let grad = t(&[0.1, 0.2, 0.3]);
    let p = parameters();
    let no = AtomicBool::new(false);
    let yes = AtomicBool::new(true);
    assert_eq!(
        s.step(&grad, &p, OptimizerLimits::default(), &yes),
        Err(TensorError::Cancelled)
    );
    let limits = OptimizerLimits {
        max_output_bytes: 35,
        max_elements: 3,
    };
    assert!(matches!(
        s.step(&grad, &p, limits, &no),
        Err(TensorError::Parameters(_))
    ));
    assert!(matches!(
        s.step(
            &grad,
            &p,
            OptimizerLimits {
                max_elements: 2,
                ..OptimizerLimits::default()
            },
            &no
        ),
        Err(TensorError::Parameters(_))
    ));
    assert!(
        s.step(&t(&[1., 2.]), &p, OptimizerLimits::default(), &no)
            .is_err()
    );
    assert!(
        s.step(&grad, &t(&[1., 2.]), OptimizerLimits::default(), &no)
            .is_err()
    );
    assert_eq!(
        [
            physical(&s.weights),
            physical(&s.first_moment),
            physical(&s.second_moment)
        ],
        before
    );
}

#[test]
fn optimizer_physical_bounds_aliases_and_empty_shapes() {
    let no = AtomicBool::new(false);
    let limits = OptimizerLimits::default();
    let p = parameters();
    let w = t(&[1., 2., 3.]);
    let m = t(&[0., 0., 0.]);
    let v = t(&[0., 0., 0.]);
    assert!(matches!(
        adamw_step(&w, &w, &m, &v, &p, limits, &no),
        Err(TensorError::Unsupported(_))
    ));
    let wd = TensorDesc::with_strides(DType::F32, [2, 2, 1, 1], [4, 12, 24, 24]).unwrap();
    let padded = Tensor::from_bytes(wd, vec![0; 20]).unwrap();
    let compact = Tensor::from_f32([2, 2, 1, 1], &[0.; 4]).unwrap();
    assert_eq!(
        sgd_step(&padded, &compact, &t(&[0.1, 0.]), limits, &no).unwrap_err(),
        TensorError::Bounds
    );
    let empty = Tensor::from_f32([0, 2, 1, 1], &[]).unwrap();
    let eg = Tensor::from_f32([0, 2, 1, 1], &[]).unwrap();
    assert_eq!(
        sgd_step(&empty, &eg, &t(&[0.1, 0.]), limits, &no)
            .unwrap()
            .shape(),
        [0, 2, 1, 1]
    );
}

#[test]
fn multi_target_graph_checks_identity_order_and_liveness() {
    let mut g = Graph::new();
    let a = g.input(t(&[2., 4.]));
    let b = g
        .op(
            Op::Scale {
                scale: 2.,
                bias: 1.,
            },
            &[a],
        )
        .unwrap();
    let other = Graph::new().input(t(&[9.]));
    assert!(matches!(
        g.evaluate_many(&[a, other]),
        Err(TensorError::InvalidNode(_))
    ));
    let outputs = g.evaluate_many(&[b, a, b]).unwrap();
    assert_eq!(outputs[0].to_f32().unwrap(), [5., 9.]);
    assert_eq!(outputs[1].to_f32().unwrap(), [2., 4.]);
    assert!(outputs[0].shares_storage(&outputs[2]));
    assert!(g.evaluate_many(&[]).unwrap().is_empty());
    let mismatch = g.input(t(&[1., 2., 3.]));
    let bad = g.op(Op::MatMul, &[a, mismatch]).unwrap();
    assert!(g.evaluate_many(&[a, bad]).is_err());
    assert_eq!(g.evaluate(a).unwrap().to_f32().unwrap(), [2., 4.]);
}
