//! Context timing for the synchronous Rust runtime, with frozen reporting boundaries.
use crate::*;

pub(crate) struct Performance {
    no_perf: bool,
    start_us: i64,
    load_us: i64,
    prompt_us: i64,
    eval_us: i64,
    prompt_count: i32,
    eval_count: i32,
    queued: i64,
    compute_start_us: Option<i64>,
    evaluated: bool,
}

impl Performance {
    pub(crate) fn new(start_us: i64, load_us: i64, no_perf: bool) -> Self {
        Self {
            no_perf,
            start_us,
            load_us,
            prompt_us: 0,
            eval_us: 0,
            prompt_count: 0,
            eval_count: 0,
            queued: 0,
            compute_start_us: None,
            evaluated: false,
        }
    }

    /// Keep reporting deferred until the source API's synchronization boundary.
    /// The actual Rust compute is synchronous; pending here describes accounting.
    pub(crate) fn queue(&mut self, tokens: usize) -> Result<()> {
        let tokens = i64::try_from(tokens).map_err(|_| "performance token count overflow")?;
        let queued = self
            .queued
            .checked_add(tokens)
            .filter(|n| *n <= i32::MAX as i64)
            .ok_or("pending performance count exceeds INT32_MAX")?;
        if queued > 1 {
            self.prompt_count
                .checked_add(queued as i32)
                .ok_or("prompt performance count exceeds INT32_MAX")?;
        } else if queued == 1 {
            self.eval_count
                .checked_add(1)
                .ok_or("evaluation performance count exceeds INT32_MAX")?;
        }
        if tokens > 0 && self.compute_start_us.is_none() {
            self.compute_start_us = Some(monotonic_time_us());
        }
        self.queued = queued;
        Ok(())
    }

    pub(crate) fn synchronize(&mut self) -> Result<()> {
        if self.queued == 0 {
            return Ok(());
        }
        let now = monotonic_time_us();
        let elapsed = now
            .checked_sub(self.compute_start_us.ok_or("missing performance start")?)
            .filter(|n| *n >= 0)
            .ok_or("performance clock moved backwards")?;
        let measured = if self.no_perf { 0 } else { elapsed };
        let (prompt_count, eval_count, prompt_us, eval_us) = if self.queued == 1 {
            (
                self.prompt_count,
                self.eval_count
                    .checked_add(1)
                    .ok_or("evaluation count overflow")?,
                self.prompt_us,
                self.eval_us
                    .checked_add(measured)
                    .ok_or("evaluation timer overflow")?,
            )
        } else {
            (
                self.prompt_count
                    .checked_add(i32::try_from(self.queued).map_err(|_| "prompt count overflow")?)
                    .ok_or("prompt count overflow")?,
                self.eval_count,
                self.prompt_us
                    .checked_add(measured)
                    .ok_or("prompt timer overflow")?,
                self.eval_us,
            )
        };
        let load_us = if self.evaluated {
            self.load_us
        } else {
            now.checked_sub(self.start_us)
                .filter(|n| *n >= 0)
                .ok_or("model loading timer overflow")?
        };
        self.prompt_count = prompt_count;
        self.eval_count = eval_count;
        self.prompt_us = prompt_us;
        self.eval_us = eval_us;
        self.load_us = load_us;
        self.evaluated = true;
        self.queued = 0;
        self.compute_start_us = None;
        Ok(())
    }

    fn data(&self) -> LlamaPerfContextData {
        LlamaPerfContextData {
            t_start_ms: 1e-3 * self.start_us as f64,
            t_load_ms: 1e-3 * self.load_us as f64,
            t_p_eval_ms: 1e-3 * self.prompt_us as f64,
            t_eval_ms: 1e-3 * self.eval_us as f64,
            n_p_eval: self.prompt_count.max(1),
            n_eval: self.eval_count.max(1),
            // This direct Rust executor does not allocate or reuse ggml graphs.
            // Reporting zero is truthful, not a prediction of original ggml stats.
            n_reused: 0,
            n_reallocs_same: 0,
            n_reallocs_diff: 0,
        }
    }

    fn reset(&mut self) {
        self.start_us = monotonic_time_us();
        self.prompt_us = 0;
        self.eval_us = 0;
        self.prompt_count = 0;
        self.eval_count = 0;
        // Native reset retains model load duration, first-evaluation state,
        // and unsynchronized queued work (including its original start).
    }
}

ffi!(
    llama_perf_context(context: *const ContextHandle) -> LlamaPerfContextData,
    LlamaPerfContextData::default(),
    {
        if context.is_null() {
            Ok(LlamaPerfContextData::default())
        } else {
            Ok(unsafe { reference(context)? }.performance.data())
        }
    }
);

ffi!(
    llama_perf_context_reset(context: *mut ContextHandle) -> (),
    (),
    {
        unsafe { reference_mut(context)? }.performance.reset();
        Ok(())
    }
);

#[cfg(any(target_os = "linux", all(target_os = "windows", target_env = "msvc")))]
#[cfg_attr(
    all(target_os = "windows", target_env = "msvc"),
    link(name = "legacy_stdio_definitions")
)]
unsafe extern "C" {
    fn snprintf(buffer: *mut c_char, capacity: usize, format: *const c_char, ...) -> i32;
}

#[cfg(any(target_os = "linux", all(target_os = "windows", target_env = "msvc")))]
fn formatted(write: impl FnOnce(*mut c_char, usize) -> i32) -> Result<CString> {
    let mut bytes = [0u8; 512];
    let size = write(bytes.as_mut_ptr().cast(), bytes.len());
    if size < 0 || size as usize >= bytes.len() {
        return Err("performance log exceeds its bounded buffer".into());
    }
    CString::new(&bytes[..size as usize]).map_err(|_| "performance log contains NUL".into())
}

fn messages(data: LlamaPerfContextData, end_ms: f64) -> Result<[CString; 6]> {
    let total = data
        .n_p_eval
        .checked_add(data.n_eval)
        .ok_or("printed token count overflow")?;
    #[cfg(any(target_os = "linux", all(target_os = "windows", target_env = "msvc")))]
    {
        // The format strings are constant and each promoted variadic argument
        // exactly matches its specifier. snprintf writes at most capacity bytes.
        // Only the platform C runtime is linked, never original llama/ggml.
        Ok([
            formatted(|p, n| unsafe {
                snprintf(
                    p,
                    n,
                    c"llama_perf_context_print:        load time = %10.2f ms\n".as_ptr(),
                    data.t_load_ms,
                )
            })?,
            formatted(|p, n| unsafe {
                snprintf(p,n,c"llama_perf_context_print: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n".as_ptr(),data.t_p_eval_ms,data.n_p_eval,data.t_p_eval_ms/data.n_p_eval as f64,1e3/data.t_p_eval_ms*data.n_p_eval as f64)
            })?,
            formatted(|p, n| unsafe {
                snprintf(p,n,c"llama_perf_context_print:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n".as_ptr(),data.t_eval_ms,data.n_eval,data.t_eval_ms/data.n_eval as f64,1e3/data.t_eval_ms*data.n_eval as f64)
            })?,
            formatted(|p, n| unsafe {
                snprintf(
                    p,
                    n,
                    c"llama_perf_context_print:       total time = %10.2f ms / %5d tokens\n"
                        .as_ptr(),
                    end_ms - data.t_start_ms,
                    total,
                )
            })?,
            formatted(|p, n| unsafe {
                snprintf(
                    p,
                    n,
                    c"llama_perf_context_print:    graphs reused = %10d\n".as_ptr(),
                    data.n_reused,
                )
            })?,
            formatted(|p, n| unsafe {
                snprintf(p,n,c"llama_perf_context_print:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n".as_ptr(),data.n_reallocs_same,data.n_reallocs_diff)
            })?,
        ])
    }
    #[cfg(not(any(target_os = "linux", all(target_os = "windows", target_env = "msvc"))))]
    {
        let raw = [
            format!(
                "llama_perf_context_print:        load time = {:10.2} ms\n",
                data.t_load_ms
            ),
            format!(
                "llama_perf_context_print: prompt eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)\n",
                data.t_p_eval_ms,
                data.n_p_eval,
                data.t_p_eval_ms / data.n_p_eval as f64,
                1e3 / data.t_p_eval_ms * data.n_p_eval as f64
            ),
            format!(
                "llama_perf_context_print:        eval time = {:10.2} ms / {:5} runs   ({:8.2} ms per token, {:8.2} tokens per second)\n",
                data.t_eval_ms,
                data.n_eval,
                data.t_eval_ms / data.n_eval as f64,
                1e3 / data.t_eval_ms * data.n_eval as f64
            ),
            format!(
                "llama_perf_context_print:       total time = {:10.2} ms / {:5} tokens\n",
                end_ms - data.t_start_ms,
                total
            ),
            format!(
                "llama_perf_context_print:    graphs reused = {:10}\n",
                data.n_reused
            ),
            format!(
                "llama_perf_context_print:   graph reallocs = {:10}    / {:5}        (same size) / (changed size)\n",
                data.n_reallocs_same, data.n_reallocs_diff
            ),
        ];
        let mut messages = Vec::with_capacity(6);
        for value in raw {
            messages.push(CString::new(value).map_err(|_| "performance log contains NUL")?);
        }
        messages
            .try_into()
            .map_err(|_| "performance log count".into())
    }
}

ffi!(
    llama_perf_context_print(context: *const ContextHandle) -> (),
    (),
    {
        let data = if context.is_null() {
            LlamaPerfContextData::default()
        } else {
            unsafe { reference(context)? }.performance.data()
        };
        let output = messages(data, 1e-3 * monotonic_time_us() as f64)?;
        for message in output {
            crate::logging::emit(2, &message)?;
        }
        Ok(())
    }
);

ffi!(llama_print_system_info() -> *const c_char, ptr::null(), {
    // Advertise only features of the registered C-facade CPU implementation.
    // Hardware discovery does not imply that this runtime executes its ISA/GPU.
    Ok(if cfg!(all(
        target_arch = "x86_64",
        any(
            all(target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", target_env = "gnu")
        )
    )) {
        c"CPU : RUST = 1 | SCALAR = 1 | FLASH_ATTN = 1 | THREADS = 1 | ".as_ptr()
    } else {
        c"CPU : RUST = 1 | SCALAR = 1 | FLASH_ATTN = 0 | THREADS = 1 | ".as_ptr()
    })
});

#[cfg(test)]
mod tests;
