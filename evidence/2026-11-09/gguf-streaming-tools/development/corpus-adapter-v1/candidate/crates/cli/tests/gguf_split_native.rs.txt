//! Black-box regressions captured only from the frozen C++ gguf-split tool.
//! See fixtures/gguf-split/provenance.json for UB, partial-output and safe-delete
//! qualifications. Original-invalid successful GGUF bytes are compared directly.
use llama_behavior::json::{self, Value};
use llama_conformance::digest::sha256;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const REFERENCE_SHA: &str = "8e215c082250d335dd38ec5256f61484d8a0f9f6473c9bbfb54357336863ec0c";
const DATA_SHA: &str = "f34d665678bb6dce9a87b9ff5f30c63e925715f36be59eb2846356049d693d58";
const MAX_CAPTURE: u64 = 4 * 1024 * 1024;
const ORIGINAL_DELETE_HELP: &str = "  --delete-splits         delete the split files during merge to free up disk space WARNING: this option is unsafe and will leave you in an unrecoverable state if something fails during the merge\n";
const SAFE_DELETE_HELP: &str = "  --delete-splits         delete split files only after the complete merge is flushed and synced\n";

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn string(value: &Value) -> &str {
    value.as_str().expect("fixture string")
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture array")
}
fn number(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("fixture nonnegative integer")).unwrap()
}
fn boolean(value: &Value) -> bool {
    value.as_bool().expect("fixture boolean")
}
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gguf-split")
        .join(name)
}
fn pinned(name: &str, expected: &str) -> Vec<u8> {
    let raw = fs::read(fixture(name)).unwrap();
    assert_eq!(sha256(&raw), expected, "fixture {name}");
    raw
}
fn relative(name: &str) -> &Path {
    let path = Path::new(name);
    assert!(
        !name.is_empty()
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "unsafe fixture path {name:?}"
    );
    path
}
struct Corpus {
    reference: Value,
    data: Vec<u8>,
}
impl Corpus {
    fn new() -> Self {
        let reference = json::parse(
            std::str::from_utf8(&pinned("native-reference.json", REFERENCE_SHA)).unwrap(),
        )
        .unwrap();
        let data = pinned("native-data.bin", DATA_SHA);
        assert_eq!(
            data.len(),
            number(field(field(&reference, "data"), "bytes"))
        );
        assert_eq!(array(field(&reference, "cases")).len(), 103);
        for (name, hash) in [
            (
                "native-msvc-report.json",
                "fc59723ff92d5a358533ad6736fcb3b6ac37c63c77c6e611eec0db6f9c207be1",
            ),
            (
                "native-gnu-report.json",
                "47dbdd19f6f96cd10d23cd1d4f5cccf0c160ac4bc43c4c3984dcecc18ce136a8",
            ),
            (
                "original-input-index.json",
                "dcfce851b54b8fefb3db53346953051cdcac41e55da5c9c734783acf3d44b565",
            ),
        ] {
            pinned(name, hash);
        }
        Self { reference, data }
    }
    fn blob(&self, record: &Value) -> &[u8] {
        let start = number(field(record, "offset"));
        let end = start.checked_add(number(field(record, "bytes"))).unwrap();
        let raw = self.data.get(start..end).expect("bounded fixture blob");
        assert_eq!(sha256(raw), string(field(record, "sha256")));
        raw
    }
    fn cases(&self) -> &[Value] {
        array(field(&self.reference, "cases"))
    }
    fn case(&self, name: &str) -> &Value {
        self.cases()
            .iter()
            .find(|case| string(field(case, "name")) == name)
            .unwrap()
    }
    fn materialize(&self, case: &Value) -> WorkingDirectory {
        let work = WorkingDirectory::new();
        for record in array(field(case, "input_files")) {
            let path = work.root.join(relative(string(field(record, "name"))));
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, self.blob(record)).unwrap();
        }
        work
    }
}
struct WorkingDirectory {
    root: PathBuf,
}
impl WorkingDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "rust-llama-split-native-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self { root }
    }
}
impl Drop for WorkingDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Captured {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}
fn limited_text(path: &Path) -> Result<String, String> {
    let mut raw = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(MAX_CAPTURE + 1)
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() as u64 > MAX_CAPTURE {
        return Err("CLI transcript exceeded 4 MiB".into());
    }
    let program = env!("CARGO_BIN_EXE_llama-gguf-split");
    Ok(String::from_utf8_lossy(&raw)
        .replace(program, "$EXE")
        .replace("\r\n", "\n"))
}
fn run(work: &WorkingDirectory, args: &[String]) -> Result<Captured, String> {
    let stdout_path = work.root.join("__capture_stdout.txt");
    let stderr_path = work.root.join("__capture_stderr.txt");
    let mut child = ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_llama-gguf-split"))
            .args(args)
            .current_dir(&work.root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                File::create(&stdout_path).map_err(|e| e.to_string())?,
            ))
            .stderr(Stdio::from(
                File::create(&stderr_path).map_err(|e| e.to_string())?,
            ))
            .spawn()
            .map_err(|e| e.to_string())?,
    );
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.0.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        if started.elapsed() >= Duration::from_secs(30) {
            return Err("CLI exceeded its 30-second direct-child deadline".into());
        }
        thread::sleep(Duration::from_millis(5));
    };
    Ok(Captured {
        status,
        stdout: limited_text(&stdout_path)?,
        stderr: limited_text(&stderr_path)?,
    })
}
fn disk_files(root: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    fn visit(
        root: &Path,
        directory: &Path,
        result: &mut BTreeMap<String, PathBuf>,
    ) -> Result<(), String> {
        for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_symlink() {
                return Err(format!("unexpected output symlink {}", path.display()));
            }
            if kind.is_dir() {
                visit(root, &path, result)?;
            } else if kind.is_file() {
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace('\\', "/");
                if !matches!(
                    name.as_str(),
                    "__capture_stdout.txt" | "__capture_stderr.txt"
                ) {
                    result.insert(name, path);
                }
            } else {
                return Err(format!("unexpected output file type {}", path.display()));
            }
        }
        Ok(())
    }
    let mut result = BTreeMap::new();
    visit(root, root, &mut result)?;
    Ok(result)
}
fn compare_file(path: &Path, expected: &[u8]) -> Result<(), String> {
    let size = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size != expected.len() as u64 {
        return Err(format!(
            "{}: {size} bytes, expected {}",
            path.display(),
            expected.len()
        ));
    }
    let actual = fs::read(path).map_err(|e| e.to_string())?;
    if actual != expected {
        return Err(format!(
            "{}: SHA {} != expected {}",
            path.display(),
            sha256(&actual),
            sha256(expected)
        ));
    }
    Ok(())
}
fn compare_snapshot(
    corpus: &Corpus,
    disk: &BTreeMap<String, PathBuf>,
    expected: &[Value],
) -> Result<usize, String> {
    let names: BTreeSet<_> = expected
        .iter()
        .map(|r| string(field(r, "name")).to_owned())
        .collect();
    if disk.keys().cloned().collect::<BTreeSet<_>>() != names {
        return Err(format!(
            "file names differ: actual {:?}, expected {names:?}",
            disk.keys().collect::<Vec<_>>()
        ));
    }
    let mut bytes = 0;
    for record in expected {
        let name = string(field(record, "name"));
        let expected = corpus.blob(record);
        compare_file(&disk[name], expected)?;
        bytes += expected.len();
    }
    Ok(bytes)
}
fn without_deletions(text: &str) -> (String, Vec<&str>) {
    let mut ordinary = String::new();
    let mut deletions = Vec::new();
    for line in text.split_inclusive('\n') {
        if line.starts_with("gguf_merge: deleted file ") {
            deletions.push(line);
        } else {
            ordinary.push_str(line);
        }
    }
    (ordinary, deletions)
}
fn compare_success_text(case: &Value, captured: &Captured, native: &Value) -> Result<(), String> {
    let name = string(field(case, "name"));
    let expected_stdout = string(field(field(native, "stdout"), "normalized"));
    let expected_stderr = string(field(field(native, "stderr"), "normalized"));
    if name == "parse-version" {
        let banner = format!("{}{}", captured.stdout, captured.stderr);
        if !banner.to_ascii_lowercase().contains("rust") {
            return Err(format!(
                "version must identify the Rust implementation: {banner:?}"
            ));
        }
        return Ok(());
    }
    let expected_stdout = if name == "parse-help" {
        assert_eq!(expected_stdout.matches(ORIGINAL_DELETE_HELP).count(), 1);
        expected_stdout.replace(ORIGINAL_DELETE_HELP, SAFE_DELETE_HELP)
    } else {
        expected_stdout.to_owned()
    };
    if captured.stdout != expected_stdout {
        return Err(format!(
            "successful stdout differs:\nactual {:?}\nexpected {:?}",
            captured.stdout, expected_stdout
        ));
    }
    if name == "merge-delete" {
        // Completion is now flushed/synced before deletion. Preserve all text,
        // and the order within each class, while allowing those lines to move.
        if without_deletions(&captured.stderr) != without_deletions(expected_stderr) {
            return Err(format!(
                "successful merge-delete stderr differs beyond deletion timing:\nactual {:?}\nexpected {:?}",
                captured.stderr, expected_stderr
            ));
        }
    } else if captured.stderr != expected_stderr {
        return Err(format!(
            "successful stderr differs:\nactual {:?}\nexpected {:?}",
            captured.stderr, expected_stderr
        ));
    }
    Ok(())
}
fn verify_failure_files(
    corpus: &Corpus,
    case: &Value,
    work: &WorkingDirectory,
) -> Result<(), String> {
    let disk = disk_files(&work.root)?;
    let initial = array(field(case, "input_files"));
    if string(field(case, "filesystem_contract")) == "checked-error-exact-inputs" {
        compare_snapshot(corpus, &disk, initial)?;
    } else {
        let inputs: BTreeSet<_> = initial
            .iter()
            .map(|r| string(field(r, "name")).to_owned())
            .collect();
        let permitted: BTreeSet<_> = array(field(case, "optional_partial_outputs"))
            .iter()
            .map(|r| string(r).to_owned())
            .collect();
        for record in initial {
            let name = string(field(record, "name"));
            let path = disk
                .get(name)
                .ok_or_else(|| format!("input {name} disappeared on failure"))?;
            compare_file(path, corpus.blob(record))?;
        }
        for (name, path) in &disk {
            if !inputs.contains(name) {
                if !permitted.contains(name) {
                    return Err(format!("unexpected failure artifact {name}"));
                }
                if fs::metadata(path).map_err(|e| e.to_string())?.len() > MAX_CAPTURE {
                    return Err(format!("failed output {name} exceeds 4 MiB"));
                }
            }
        }
    }
    Ok(())
}
fn check_case(corpus: &Corpus, case: &Value) -> Result<usize, String> {
    let work = corpus.materialize(case);
    let args: Vec<_> = array(field(case, "args"))
        .iter()
        .map(|v| string(v).to_owned())
        .collect();
    let captured = run(&work, &args)?;
    let success = boolean(field(case, "rust_success"));
    if success {
        if captured.status.code() != Some(0) {
            return Err(format!(
                "expected success, got {:?}: {}",
                captured.status.code(),
                captured.stderr
            ));
        }
        let native = field(
            field(case, "native"),
            if cfg!(windows) { "windows" } else { "linux" },
        );
        let bytes = compare_snapshot(
            corpus,
            &disk_files(&work.root)?,
            array(field(native, "after_files")),
        )?;
        compare_success_text(case, &captured, native)?;
        Ok(bytes)
    } else {
        // A native abort, signal or UB observation is deliberately not replayed.
        // Require the ordinary Rust error exit, so a panic/abort cannot pass.
        if captured.status.code() != Some(1) || captured.stderr.is_empty() {
            return Err(format!(
                "expected checked error exit1, got {:?}, stderr={:?}",
                captured.status.code(),
                captured.stderr
            ));
        }
        verify_failure_files(corpus, case, &work)?;
        Ok(0)
    }
}

#[test]
fn frozen_original_split_merge_cli_cases() {
    let corpus = Corpus::new();
    let mut failures = Vec::new();
    let mut success = 0;
    let mut errors = 0;
    let mut exact_bytes = 0;
    for case in corpus.cases() {
        let name = string(field(case, "name"));
        match check_case(&corpus, case) {
            Ok(bytes) => {
                exact_bytes += bytes;
                if boolean(field(case, "rust_success")) {
                    success += 1;
                } else {
                    errors += 1;
                }
            }
            Err(error) => failures.push(format!("{name}: {error}")),
        }
    }
    eprintln!(
        "gguf-split-native: cases=103 success={success} checked_errors={errors} exact_after_file_bytes={exact_bytes}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert_eq!((success, errors), (63, 40));
}

/// An intentional stronger Rust ownership contract, using unchanged original
/// truncated input bytes. The original incremental delete loop is not an oracle
/// for preserving early splits when a later payload fails.
#[test]
fn failed_delete_merge_preserves_every_original_split() {
    let corpus = Corpus::new();
    let case = corpus.case("merge-truncated-secondary");
    let work = corpus.materialize(case);
    let mut args: Vec<_> = array(field(case, "args"))
        .iter()
        .map(|v| string(v).to_owned())
        .collect();
    args.insert(1, "--delete-splits".into());
    let captured = run(&work, &args).unwrap();
    assert_eq!(captured.status.code(), Some(1), "{}", captured.stderr);
    assert!(!captured.stderr.is_empty());
    verify_failure_files(&corpus, case, &work).unwrap();
}
