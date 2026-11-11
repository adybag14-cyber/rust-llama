use llama_tensor::{
    CustomKernel, CustomOptions, CustomTask, DType, Graph, Op, Tensor, TensorDesc, TensorError,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let r = &self.bytes[self.at..self.at + n];
        self.at += n;
        r
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn size(&mut self) -> usize {
        usize::try_from(u64::from_le_bytes(self.take(8).try_into().unwrap())).unwrap()
    }
    fn text(&mut self) -> String {
        let n = self.u32() as usize;
        String::from_utf8(self.take(n).to_vec()).unwrap()
    }
    fn tensor(&mut self) -> Tensor {
        let d = DType::from_id(self.u32()).unwrap();
        let ne = std::array::from_fn(|_| self.size());
        let nb = std::array::from_fn(|_| self.size());
        let n = self.size();
        Tensor::from_bytes(
            TensorDesc::with_strides(d, ne, nb).unwrap(),
            self.take(n).to_vec(),
        )
        .unwrap()
    }
}
fn fixture() -> &'static [u8] {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        include_bytes!("fixtures/custom/windows-msvc.bin")
    } else {
        include_bytes!("fixtures/custom/linux-gnu.bin")
    }
}
#[test]
fn original_custom_dispatch_layouts_tasks_userdata_and_outputs() {
    let mut r = Reader {
        bytes: fixture(),
        at: 0,
    };
    assert_eq!(r.take(8), b"GGCU0001");
    let count = r.u32();
    assert_eq!(count, 260);
    let mut total_bytes = 0;
    for _ in 0..count {
        let name = r.text();
        let kind = r.u32();
        let n_tasks = r.i32();
        let workers = r.u32() as usize;
        let seed = r.u32() as usize;
        let ni = r.u32();
        let inputs = (0..ni).map(|_| r.tensor()).collect::<Vec<_>>();
        let expected = r.tensor();
        let actual_tasks = r.u32() as usize;
        total_bytes += expected.desc().byte_span().unwrap();
        let traces = Arc::new(Mutex::new(Vec::new()));
        let capture = Arc::clone(&traces);
        let descriptions = inputs.iter().map(|t| *t.desc()).collect::<Vec<_>>();
        let kernel = CustomKernel::new("native byte callback", move |task: CustomTask<'_>| {
            assert_eq!(task.input_count(), ni as usize);
            for (i, desc) in descriptions.iter().enumerate() {
                assert_eq!(task.input(i)?.desc(), desc);
            }
            capture
                .lock()
                .unwrap()
                .push((task.ith, task.nth, std::thread::current().id()));
            for j in (task.ith..task.output_bytes_len()).step_by(task.nth) {
                let mut value = (seed + j) as u8;
                for k in 0..task.input_count() {
                    let len = task.input_bytes_len(k)?;
                    if len > 0 {
                        let mut byte = [0];
                        task.read_input(k, (j + k * 7) % len, &mut byte)?;
                        value ^= byte[0];
                    }
                }
                task.write_output(j, &[value])?;
            }
            Ok(())
        });
        let options = CustomOptions {
            n_tasks,
            workers,
            ..CustomOptions::default()
        };
        let op = match kind {
            0 => Op::MapCustom1 { kernel, options },
            1 => Op::MapCustom2 { kernel, options },
            2 => Op::MapCustom3 { kernel, options },
            3 => Op::Custom {
                kernel,
                options,
                dtype: expected.dtype(),
                shape: expected.shape(),
                input_count: inputs.len(),
            },
            _ => panic!("unknown callback family"),
        };
        let originals = inputs
            .iter()
            .map(|t| t.to_bytes().unwrap())
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let ids = inputs
            .iter()
            .cloned()
            .map(|t| graph.input(t))
            .collect::<Vec<_>>();
        let target = graph.op(op, &ids).unwrap();
        let outputs = graph.evaluate_many(&[target, target]).unwrap();
        assert_eq!(outputs[0].desc(), expected.desc(), "{name}");
        assert_eq!(
            outputs[0].to_bytes().unwrap(),
            expected.to_bytes().unwrap(),
            "{name}"
        );
        assert!(outputs[0].shares_storage(&outputs[1]));
        let trace = traces.lock().unwrap();
        let mut invocations = trace.iter().map(|&(i, n, _)| (i, n)).collect::<Vec<_>>();
        invocations.sort();
        assert_eq!(
            invocations,
            (0..actual_tasks)
                .map(|i| (i, actual_tasks))
                .collect::<Vec<_>>(),
            "{name}"
        );
        assert_eq!(
            trace
                .iter()
                .map(|&(_, _, t)| t)
                .collect::<HashSet<_>>()
                .len(),
            actual_tasks,
            "actual worker threads: {name}"
        );
        for (input, bytes) in inputs.iter().zip(originals) {
            assert_eq!(input.to_bytes().unwrap(), bytes);
            assert!(!outputs[0].shares_storage(input));
        }
    }
    assert_eq!(r.at, r.bytes.len());
    assert_eq!(total_bytes, 30064);
}

fn single(
    kernel: CustomKernel,
    options: CustomOptions,
    shape: [usize; 4],
) -> (Graph, llama_tensor::NodeId) {
    let mut g = Graph::new();
    let n = g
        .op(
            Op::Custom {
                kernel,
                options,
                dtype: DType::F32,
                shape,
                input_count: 0,
            },
            &[],
        )
        .unwrap();
    (g, n)
}
#[test]
fn custom_callback_errors_panics_and_uninitialized_bytes_are_checked() {
    for (kernel, expected) in [
        (
            CustomKernel::new("unwritten", |_| Ok(())),
            TensorError::Custom("custom callback left output bytes unwritten"),
        ),
        (
            CustomKernel::new("uninitialized read", |c| c.read_output(0, &mut [0])),
            TensorError::Custom("custom callback read uninitialized output"),
        ),
        (
            CustomKernel::new("bad write", |c| c.write_output(4, &[1])),
            TensorError::Bounds,
        ),
        (
            CustomKernel::new("overflow", |c| c.write_output(usize::MAX, &[1])),
            TensorError::Overflow,
        ),
        (
            CustomKernel::new("bad source", |c| c.read_input(0, 0, &mut [0])),
            TensorError::Bounds,
        ),
        (
            CustomKernel::new("panic", |_| panic!("owned test callback panic")),
            TensorError::Custom("custom callback panicked"),
        ),
    ] {
        let (g, n) = single(kernel, CustomOptions::default(), [1, 1, 1, 1]);
        assert_eq!(g.evaluate(n).unwrap_err(), expected);
    }
}

#[test]
fn parallel_callback_failure_joins_every_started_worker() {
    let entered = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&entered);
    let kernel = CustomKernel::new("worker failure", move |c| {
        count.fetch_add(1, Ordering::SeqCst);
        if c.ith == 1 {
            return Err(TensorError::Parameters("deliberate callback rejection"));
        }
        c.write_output(c.ith, &[c.ith as u8])
    });
    let (g, n) = single(
        kernel,
        CustomOptions {
            workers: 4,
            ..CustomOptions::default()
        },
        [1, 1, 1, 1],
    );
    assert_eq!(
        g.evaluate(n).unwrap_err(),
        TensorError::Parameters("deliberate callback rejection")
    );
    assert_eq!(entered.load(Ordering::SeqCst), 4);
}

#[test]
fn callbacks_are_owned_and_cancellation_does_not_publish_buffers() {
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicUsize::new(0));
    let owner = Owner(Arc::clone(&dropped));
    let cancel = Arc::new(AtomicBool::new(false));
    let token = Arc::clone(&cancel);
    let kernel = CustomKernel::new("owned cancellation", move |c| {
        let _ = &owner;
        c.write_output(0, &[0, 0, 0, 0])?;
        token.store(true, Ordering::SeqCst);
        Ok(())
    });
    let (g, n) = single(kernel, CustomOptions::default(), [1, 1, 1, 1]);
    assert_eq!(
        g.evaluate_with_cancel(n, &cancel).unwrap_err(),
        TensorError::Cancelled
    );
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(g);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn custom_limits_and_empty_outputs_never_invoke_userdata() {
    let called = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&called);
    let kernel = CustomKernel::new("must not run", move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    for options in [
        CustomOptions {
            workers: 0,
            ..CustomOptions::default()
        },
        CustomOptions {
            workers: 65,
            ..CustomOptions::default()
        },
        CustomOptions {
            n_tasks: 0,
            ..CustomOptions::default()
        },
        CustomOptions {
            n_tasks: -2,
            ..CustomOptions::default()
        },
        CustomOptions {
            max_output_bytes: 3,
            ..CustomOptions::default()
        },
    ] {
        let (g, n) = single(kernel.clone(), options, [1, 1, 1, 1]);
        assert!(matches!(g.evaluate(n), Err(TensorError::Parameters(_))));
    }
    let (g, n) = single(
        kernel.clone(),
        CustomOptions {
            workers: 3,
            ..CustomOptions::default()
        },
        [0, 4, 1, 1],
    );
    assert!(g.evaluate(n).unwrap().to_bytes().unwrap().is_empty());
    let mut g = Graph::new();
    let a = g.input(Tensor::from_f32([1, 1, 1, 1], &[1.]).unwrap());
    let n = g
        .op(
            Op::Custom {
                kernel,
                options: CustomOptions::default(),
                dtype: DType::F32,
                shape: [1, 1, 1, 1],
                input_count: 10,
            },
            &[a; 10],
        )
        .unwrap();
    assert!(matches!(g.evaluate(n), Err(TensorError::Parameters(_))));
    assert_eq!(called.load(Ordering::SeqCst), 0);
}
