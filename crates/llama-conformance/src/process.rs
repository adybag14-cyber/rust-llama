//! Bounded external probes for isolated oracles, compilers and diagnostic CLIs.
use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Probe {
    pub program: OsString,
    pub arguments: Vec<OsString>,
    pub input: Vec<u8>,
    pub timeout: Duration,
    pub output_limit: usize,
}
#[derive(Debug)]
pub struct ProbeOutput {
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
    pub output_truncated: bool,
    pub elapsed: Duration,
}
impl Probe {
    pub fn new(program: impl AsRef<OsStr>) -> Self {
        Self {
            program: program.as_ref().into(),
            arguments: Vec::new(),
            input: Vec::new(),
            timeout: Duration::from_secs(30),
            output_limit: 4 * 1024 * 1024,
        }
    }
}
fn drain(mut stream: impl Read, limit: usize) -> io::Result<(Vec<u8>, bool)> {
    let mut out = Vec::new();
    let mut clipped = false;
    let mut buffer = [0u8; 8192];
    loop {
        let n = stream.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        let take = n.min(limit.saturating_sub(out.len()));
        out.extend_from_slice(&buffer[..take]);
        clipped |= take < n;
    }
    Ok((out, clipped))
}
pub fn run(probe: &Probe) -> io::Result<ProbeOutput> {
    if probe.timeout.is_zero()
        || probe.timeout > Duration::from_secs(3600)
        || probe.output_limit > 64 * 1024 * 1024
        || probe.input.len() > 64 * 1024 * 1024
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "probe budgets outside permitted range",
        ));
    }
    let start = Instant::now();
    let mut child = Command::new(&probe.program)
        .args(&probe.arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let input = probe.input.clone();
    let limit = probe.output_limit;
    let input_thread = thread::spawn(move || stdin.write_all(&input));
    let output_thread = thread::spawn(move || drain(stdout, limit));
    let error_thread = thread::spawn(move || drain(stderr, limit));
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= probe.timeout {
            timed_out = true;
            child.kill()?;
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(5));
    };
    // Descendant processes that inherit pipes are outside this single-process
    // probe's contract. Use executables known not to leave descendants holding
    // these pipes. Compiler drivers (including Cargo/GCC) can spawn descendants
    // and require a separate process-tree supervisor for a hard deadline.
    let input_result = input_thread
        .join()
        .map_err(|_| io::Error::other("stdin worker panicked"))?;
    if let Err(e) = input_result
        && e.kind() != io::ErrorKind::BrokenPipe
        && !timed_out
    {
        return Err(e);
    }
    let (stdout, out_clipped) = output_thread
        .join()
        .map_err(|_| io::Error::other("stdout worker panicked"))??;
    let (stderr, err_clipped) = error_thread
        .join()
        .map_err(|_| io::Error::other("stderr worker panicked"))??;
    Ok(ProbeOutput {
        status: status.code(),
        stdout,
        stderr,
        timed_out,
        output_truncated: out_clipped || err_clipped,
        elapsed: start.elapsed(),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn binary_stdin_and_bounded_capture_do_not_deadlock() {
        let mut probe = Probe::new("/bin/cat");
        probe.input = (0..256).cycle().take(200_000).map(|x| x as u8).collect();
        probe.output_limit = 1024;
        let result = run(&probe).unwrap();
        assert_eq!(result.status, Some(0));
        assert_eq!(result.stdout, &probe.input[..1024]);
        assert!(result.output_truncated);
        assert!(!result.timed_out);
    }
    #[test]
    fn timeout_terminates_single_process() {
        let mut probe = Probe::new("/bin/sleep");
        probe.arguments.push("2".into());
        probe.timeout = Duration::from_millis(30);
        let result = run(&probe).unwrap();
        assert!(result.timed_out);
        assert!(result.elapsed < Duration::from_secs(1));
    }
    #[test]
    fn missing_program_and_invalid_budget_are_errors() {
        assert!(run(&Probe::new("/definitely-not-present/llama-probe")).is_err());
        let mut probe = Probe::new("/bin/cat");
        probe.timeout = Duration::ZERO;
        assert!(run(&probe).is_err());
    }
}
