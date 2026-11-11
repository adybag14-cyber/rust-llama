//! Owned Rust callbacks for the frozen out-of-place custom operation family.
use crate::graph::check_cancel;
use crate::{Op, Result, Tensor, TensorDesc, TensorError, reserved};
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};

/// -1 means all configured workers; positive values are capped to workers.
/// The actual worker count is limited to 64. The full graph is not parallelized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CustomOptions {
    pub n_tasks: i32,
    pub workers: usize,
    pub max_output_bytes: usize,
}
impl Default for CustomOptions {
    fn default() -> Self {
        Self {
            n_tasks: -1,
            workers: 1,
            max_output_bytes: 256 * 1024 * 1024,
        }
    }
}
type Callback = dyn for<'a> Fn(CustomTask<'a>) -> Result<()> + Send + Sync;

/// Owned callback and captured userdata. Clones share the same callback owner.
#[derive(Clone)]
pub struct CustomKernel {
    name: Arc<str>,
    callback: Arc<Callback>,
}
impl CustomKernel {
    pub fn new<F>(name: impl Into<String>, callback: F) -> Self
    where
        F: for<'a> Fn(CustomTask<'a>) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            name: Arc::from(name.into()),
            callback: Arc::new(callback),
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}
impl fmt::Debug for CustomKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomKernel")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
struct Storage {
    bytes: Vec<u8>,
    written: Vec<u8>,
}
struct Output {
    desc: TensorDesc,
    len: usize,
    storage: Mutex<Storage>,
}

/// Safe callback view. Physical byte access includes source view padding.
/// Output accesses are synchronized; no lock guard escapes to callback code.
/// Callbacks should cooperate with cancellation and assign disjoint output
/// regions when they need deterministic execution across multiple workers.
#[derive(Clone, Copy)]
pub struct CustomTask<'a> {
    pub ith: usize,
    pub nth: usize,
    inputs: &'a [&'a Tensor],
    output: &'a Output,
    cancel: &'a AtomicBool,
}
impl CustomTask<'_> {
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }
    pub fn input(&self, index: usize) -> Result<&Tensor> {
        self.inputs.get(index).copied().ok_or(TensorError::Bounds)
    }
    pub fn input_bytes_len(&self, index: usize) -> Result<usize> {
        self.input(index)?.desc.byte_span()
    }
    pub fn read_input(&self, index: usize, offset: usize, destination: &mut [u8]) -> Result<()> {
        check_cancel(self.cancel)?;
        let t = self.input(index)?;
        let end = offset
            .checked_add(destination.len())
            .ok_or(TensorError::Overflow)?;
        if end > t.desc.byte_span()? {
            return Err(TensorError::Bounds);
        }
        destination.copy_from_slice(&t.data[t.offset + offset..t.offset + end]);
        Ok(())
    }
    pub fn output_desc(&self) -> &TensorDesc {
        &self.output.desc
    }
    pub fn output_bytes_len(&self) -> usize {
        self.output.len
    }
    pub fn check_cancel(&self) -> Result<()> {
        check_cancel(self.cancel)
    }

    pub fn write_output(&self, offset: usize, source: &[u8]) -> Result<()> {
        check_cancel(self.cancel)?;
        let end = offset
            .checked_add(source.len())
            .ok_or(TensorError::Overflow)?;
        if end > self.output.len {
            return Err(TensorError::Bounds);
        }
        let mut s = self
            .output
            .storage
            .lock()
            .map_err(|_| TensorError::Custom("custom output lock poisoned"))?;
        s.bytes[offset..end].copy_from_slice(source);
        for i in offset..end {
            s.written[i / 8] |= 1 << (i % 8);
        }
        Ok(())
    }
    /// Reading bytes not yet initialized by a callback returns a checked error.
    pub fn read_output(&self, offset: usize, destination: &mut [u8]) -> Result<()> {
        check_cancel(self.cancel)?;
        let end = offset
            .checked_add(destination.len())
            .ok_or(TensorError::Overflow)?;
        if end > self.output.len {
            return Err(TensorError::Bounds);
        }
        let s = self
            .output
            .storage
            .lock()
            .map_err(|_| TensorError::Custom("custom output lock poisoned"))?;
        if (offset..end).any(|i| s.written[i / 8] & (1 << (i % 8)) == 0) {
            return Err(TensorError::Custom(
                "custom callback read uninitialized output",
            ));
        }
        destination.copy_from_slice(&s.bytes[offset..end]);
        Ok(())
    }
}

fn run_task(
    kernel: &CustomKernel,
    src: &[&Tensor],
    output: &Output,
    cancel: &AtomicBool,
    ith: usize,
    nth: usize,
) -> Result<()> {
    check_cancel(cancel)?;
    catch_unwind(AssertUnwindSafe(|| {
        (kernel.callback)(CustomTask {
            ith,
            nth,
            inputs: src,
            output,
            cancel,
        })
    }))
    .map_err(|_| TensorError::Custom("custom callback panicked"))?
}

pub(crate) fn evaluate(op: &Op, src: &[&Tensor], cancel: &AtomicBool) -> Option<Result<Tensor>> {
    let (kernel, options, desc) = match op {
        Op::MapCustom1 { kernel, options }
        | Op::MapCustom2 { kernel, options }
        | Op::MapCustom3 { kernel, options } => {
            let a = src[0];
            (
                kernel,
                *options,
                TensorDesc::contiguous(a.dtype(), a.shape()),
            )
        }
        Op::Custom {
            kernel,
            options,
            dtype,
            shape,
            input_count,
        } => {
            if *input_count >= 10 {
                return Some(Err(TensorError::Parameters(
                    "CUSTOM accepts at most nine inputs",
                )));
            }
            (kernel, *options, TensorDesc::contiguous(*dtype, *shape))
        }
        _ => return None,
    };
    Some(desc.and_then(|desc| execute(kernel, options, desc, src, cancel)))
}
fn execute(
    kernel: &CustomKernel,
    options: CustomOptions,
    desc: TensorDesc,
    src: &[&Tensor],
    cancel: &AtomicBool,
) -> Result<Tensor> {
    check_cancel(cancel)?;
    if options.workers == 0 || options.workers > i32::MAX as usize {
        return Err(TensorError::Parameters(
            "custom worker count must fit a positive source integer",
        ));
    }
    let nth = match options.n_tasks {
        -1 => options.workers,
        n if n > 0 => (n as usize).min(options.workers),
        _ => {
            return Err(TensorError::Parameters(
                "custom task count must be positive or -1",
            ));
        }
    };
    if nth > 64 {
        return Err(TensorError::Parameters(
            "custom operation exceeds 64 actual workers",
        ));
    }
    let len = desc.byte_span()?;
    if len > options.max_output_bytes {
        return Err(TensorError::Parameters("custom output byte limit exceeded"));
    }
    let mut bytes = reserved(len)?;
    bytes.resize(len, 0u8);
    // The original CPU graph skips every callback for an empty output.
    if len == 0 {
        return Tensor::from_bytes(desc, bytes);
    }
    let map_len = len.checked_add(7).ok_or(TensorError::Overflow)? / 8;
    let mut written = reserved(map_len)?;
    written.resize(map_len, 0u8);
    let output = Output {
        desc,
        len,
        storage: Mutex::new(Storage { bytes, written }),
    };
    if nth == 1 {
        run_task(kernel, src, &output, cancel, 0, 1)?;
    } else {
        // All started workers are joined on every path. Failure never publishes
        // the partially written buffer. User-owned external side effects cannot
        // be rolled back by the graph and are not part of this transaction.
        let start = (Mutex::new(None), Condvar::new());
        std::thread::scope(|scope| -> Result<()> {
            let mut joins = reserved(nth - 1)?;
            let mut failure = None;
            let start_ref = &start;
            let output_ref = &output;
            for ith in 1..nth {
                match std::thread::Builder::new().spawn_scoped(scope, move || {
                    let mut gate = start_ref
                        .0
                        .lock()
                        .map_err(|_| TensorError::Custom("custom launch lock poisoned"))?;
                    while gate.is_none() {
                        gate = start_ref
                            .1
                            .wait(gate)
                            .map_err(|_| TensorError::Custom("custom launch lock poisoned"))?;
                    }
                    let enabled = gate.expect("released custom launch gate");
                    drop(gate);
                    if enabled {
                        run_task(kernel, src, output_ref, cancel, ith, nth)
                    } else {
                        Ok(())
                    }
                }) {
                    Ok(handle) => joins.push(handle),
                    Err(_) => {
                        failure = Some(TensorError::Custom("custom worker creation failed"));
                        break;
                    }
                }
            }
            *start
                .0
                .lock()
                .map_err(|_| TensorError::Custom("custom launch lock poisoned"))? =
                Some(failure.is_none());
            start.1.notify_all();
            if failure.is_none()
                && let Err(e) = run_task(kernel, src, &output, cancel, 0, nth)
            {
                failure = Some(e);
            }
            for handle in joins {
                let result = handle
                    .join()
                    .unwrap_or(Err(TensorError::Custom("custom worker panicked")));
                if let Err(e) = result
                    && failure.is_none()
                {
                    failure = Some(e);
                }
            }
            if let Some(e) = failure {
                Err(e)
            } else {
                Ok(())
            }
        })?;
    }
    check_cancel(cancel)?;
    let storage = output
        .storage
        .into_inner()
        .map_err(|_| TensorError::Custom("custom output lock poisoned"))?;
    if (0..len).any(|i| storage.written[i / 8] & (1 << (i % 8)) == 0) {
        return Err(TensorError::Custom(
            "custom callback left output bytes unwritten",
        ));
    }
    Tensor::from_bytes(desc, storage.bytes)
}
