//! Rust-authored build, inventory, validation and deterministic packaging tasks.
#![forbid(unsafe_code)]
use llama_conformance::{
    benchmark, compare, digest, fixtures as fixture_integrity, inventory, package, process,
};
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Duration;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

const HELP: &str = "llama rewrite xtask (Rust 1.99.1; offline)\n\nCommands:\n  doctor                                compiler identity and installed targets\n  baseline REFERENCE_DIR [OUTPUT_DIR]     hash and assign every original file; extract registries\n  verify-source REFERENCE_DIR [LEDGER]    report added, removed and changed files\n  audit [--write]                        audit source languages and offline Cargo dependency graph\n  fixtures                              verify independent fixture manifests, when present\n  check | test | build                   offline, locked workspace operations\n  fmt                                   rustfmt check\n  compare EXPECTED ACTUAL                exact byte comparison\n  numeric EXPECTED ACTUAL ABS REL        numeric whitespace/CSV comparison\n  capture OUTPUT_STEM PROGRAM [ARGS...]  single-process evidence (30-second direct-child timeout)\n  storage [DIR]                          count disk bytes, without deleting anything\n  benchmark-hash                         2 warmups and 9 SHA-256 host measurements\n  release-check                         fail while mandatory release gates remain open\n  package OUTPUT.zip                    deterministic source ZIP; refuses overwrite\n  verify-package [ROOT]                  verify every extracted package file hash\n";
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}
fn require_len(args: &[String], min: usize, max: usize) -> Result<()> {
    if args.len() < min || args.len() > max {
        Err("incorrect arguments; use --help".into())
    } else {
        Ok(())
    }
}
fn cargo() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}
fn run_status(program: impl AsRef<std::ffi::OsStr>, args: &[&str]) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .current_dir(root())
        .status()?;
    if !status.success() {
        return Err(format!("command exited with {status}").into());
    }
    Ok(())
}
fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
fn baseline(args: &[String]) -> Result<()> {
    require_len(args, 1, 2)?;
    let reference = Path::new(&args[0]);
    let output = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("ledger"));
    let files = inventory::scan(reference)?;
    let counts = inventory::counts(&files);
    let bytes: u64 = files.iter().map(|r| r.bytes).sum();
    for (i, n) in counts.iter().enumerate() {
        println!("T{:02}: {n}", i + 1);
    }
    println!("{} files, {bytes} source bytes", files.len());
    if counts != inventory::EXPECTED_COUNTS || files.len() != 4082 || bytes != 183_804_212 {
        return Err(
            "baseline ownership/size differs from frozen plan; no ledger was overwritten".into(),
        );
    }
    let registries = inventory::registries(reference)?;
    let registry_counts = inventory::registry_counts(&registries);
    for (kind, n) in &registry_counts {
        println!("{kind}: {n}");
    }
    for (kind, count) in [
        ("architecture", 173),
        ("dtype", 36),
        ("operation", 102),
        ("unary", 22),
        ("glu", 7),
        ("model_source", 176),
        ("multimodal_builder", 58),
        ("backend", 18),
    ] {
        if registry_counts.get(kind) != Some(&count) {
            return Err(format!("registry {kind} differs from frozen count {count}").into());
        }
    }
    fs::create_dir_all(&output)?;
    inventory::write_ledger(&files, &output.join("baseline-files.tsv"))?;
    inventory::write_registries(&registries, &output.join("baseline-registries.tsv"))?;
    let counts_json = registry_counts
        .iter()
        .map(|(k, v)| format!("  {}: {v}", json_string(k)))
        .collect::<Vec<_>>()
        .join(",\n");
    let data = format!(
        "{{\n\"archive_sha256\":{},\n\"toolchain_archive_sha256\":{},\n\"files\":{},\n\"uncompressed_bytes\":{},\n\"track_file_counts\":{:?},\n\"registry_unique_counts\":{{\n{}\n}},\n\"note\":\"Registry extraction is lexical evidence, not a full semantic completeness proof; planned baseline rows are separate from implementation capability ledgers.\"\n}}\n",
        json_string(inventory::BASELINE_SHA256),
        json_string(inventory::TOOLCHAIN_SHA256),
        files.len(),
        bytes,
        counts,
        counts_json
    );
    fs::write(output.join("baseline-summary.json"), data)?;
    println!("Baseline identity and registries recorded.");
    Ok(())
}
fn audit(write: bool) -> Result<()> {
    let files = package::gather(&root())?;
    let result = package::audit(&files);
    println!(
        "Rust source: {} files, {} physical lines",
        result.rust_files, result.rust_lines
    );
    println!(
        "Non-Rust data/config/docs: {} files; total {} bytes",
        result.non_rust_files, result.total_bytes
    );
    for item in &result.unsafe_mentions {
        println!("Unsafe review candidate: {item}");
    }
    for item in &result.errors {
        eprintln!("FAIL: {item}");
    }
    let metadata = Command::new(cargo())
        .args(["metadata", "--offline", "--locked", "--format-version", "1"])
        .current_dir(root())
        .output()?;
    if !metadata.status.success() {
        return Err(format!(
            "Cargo dependency audit failed: {}",
            String::from_utf8_lossy(&metadata.stderr)
        )
        .into());
    }
    let metadata = String::from_utf8(metadata.stdout)?;
    // Cargo metadata emits compact JSON. There are no external crates in this
    // checkpoint. Source-less workspace/path packages are the only allowed graph.
    if metadata.contains("\"source\":\"registry+") || metadata.contains("\"source\":\"git+") {
        return Err("external dependency requires an explicit audit before release".into());
    }
    if metadata.contains("\"kind\":[\"custom-build\"]") {
        return Err("custom build scripts require a separate native-dependency audit".into());
    }
    if !result.errors.is_empty() {
        return Err("source-language audit failed".into());
    }
    if write {
        let selected: Vec<_> = files
            .into_iter()
            .filter(|f| f.path != "release/non-rust-allowlist.tsv")
            .collect();
        fs::create_dir_all(root().join("release"))?;
        fs::write(
            root().join("release/non-rust-allowlist.tsv"),
            package::non_rust_allowlist(&selected),
        )?;
    }
    println!(
        "Source audit passed; Cargo graph contains only local packages. This is a host-checkpoint audit, not all-feature/backend closure."
    );
    Ok(())
}
fn fixtures() -> Result<()> {
    let files = package::gather(&root())?;
    let report = fixture_integrity::verify(&files)?;
    println!(
        "{} fixture files covered by {} integrity declarations in {} manifests ({} original size/purpose rows). Numerical parity is checked separately by domain tests; historical provenance is preserved.",
        report.fixture_files, report.entries, report.manifests, report.size_and_purpose_entries
    );
    Ok(())
}

fn numeric(path: &str) -> Result<Vec<f64>> {
    let text = fs::read_to_string(path)?;
    text.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|x| Ok(x.parse()?))
        .collect()
}
fn capture(args: &[String]) -> Result<()> {
    require_len(args, 2, usize::MAX)?;
    let stem = Path::new(&args[0]);
    if let Some(parent) = stem.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut probe = process::Probe::new(&args[1]);
    probe.arguments = args[2..].iter().map(OsString::from).collect();
    probe.timeout = Duration::from_secs(30);
    let result = process::run(&probe)?;
    fs::write(stem.with_extension("stdout.txt"), &result.stdout)?;
    fs::write(stem.with_extension("stderr.txt"), &result.stderr)?;
    let command = args[1..]
        .iter()
        .map(|s| json_string(s))
        .collect::<Vec<_>>()
        .join(",");
    let data = format!(
        "{{\"command\":[{command}],\"exit_code\":{},\"timed_out\":{},\"output_truncated\":{},\"elapsed_ms\":{},\"stdout_sha256\":{},\"stderr_sha256\":{}}}\n",
        result
            .status
            .map(|x| x.to_string())
            .unwrap_or("null".into()),
        result.timed_out,
        result.output_truncated,
        result.elapsed.as_millis(),
        json_string(&digest::sha256(&result.stdout)),
        json_string(&digest::sha256(&result.stderr))
    );
    fs::write(stem.with_extension("json"), data)?;
    if result.status != Some(0) || result.timed_out || result.output_truncated {
        return Err("probe failed, timed out or exceeded output budget; evidence retained".into());
    }
    println!("Probe succeeded; stdout/stderr and SHA-256 recorded.");
    Ok(())
}
fn storage(path: &Path) -> Result<()> {
    let files = inventory::relative_files(path)?;
    let mut groups: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for f in files {
        let bytes = fs::metadata(path.join(&f))?.len();
        let group = f
            .components()
            .next()
            .unwrap()
            .as_os_str()
            .to_string_lossy()
            .into_owned();
        let entry = groups.entry(group).or_default();
        entry.0 += 1;
        entry.1 += bytes;
    }
    let mut total = 0;
    for (name, (n, bytes)) in groups {
        println!("{name}\t{n} files\t{bytes} bytes");
        total += bytes;
    }
    println!("Total: {total} bytes");
    Ok(())
}
fn release_check() -> Result<()> {
    let text = fs::read_to_string(root().join("release/gates.tsv"))?;
    let mut open = 0;
    for (i, line) in text.lines().enumerate().skip(1) {
        let cols: Vec<_> = line.split('\t').collect();
        if cols.len() < 4 {
            return Err(format!("malformed gate row {}", i + 1).into());
        }
        if cols[2] != "validated" {
            println!("OPEN {} [{}]: {}", cols[0], cols[2], cols[3]);
            open += 1;
        }
    }
    if open > 0 {
        return Err(
            format!("{open} mandatory gates remain open; full rewrite release is blocked").into(),
        );
    }
    println!("All recorded gates validated. Review their evidence before a release claim.");
    Ok(())
}
fn execute() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("{HELP}");
        return Ok(());
    }
    let command = args.remove(0);
    match command.as_str() {
        "help" | "--help" | "-h" => {
            println!("{HELP}");
            Ok(())
        }
        "doctor" => {
            require_len(&args, 0, 0)?;
            run_status("rustc", &["-vV"])?;
            run_status(cargo(), &["--version"])?;
            run_status("rustc", &["--print", "target-libdir"])?;
            println!("Only installed target libraries establish build availability.");
            Ok(())
        }
        "baseline" => baseline(&args),
        "verify-source" => {
            require_len(&args, 1, 2)?;
            let ledger = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root().join("ledger/baseline-files.tsv"));
            let errors = inventory::verify_ledger(Path::new(&args[0]), &ledger)?;
            for e in &errors {
                eprintln!("{e}");
            }
            if !errors.is_empty() {
                Err(format!("{} baseline drift errors", errors.len()).into())
            } else {
                println!("Every baseline file matches its recorded SHA-256, size and owner.");
                Ok(())
            }
        }
        "audit" => {
            require_len(&args, 0, 1)?;
            if !args.is_empty() && args[0] != "--write" {
                return Err("audit accepts only --write".into());
            }
            audit(!args.is_empty())
        }
        "fixtures" => {
            require_len(&args, 0, 0)?;
            fixtures()
        }
        "check" => {
            require_len(&args, 0, 0)?;
            run_status(
                cargo(),
                &[
                    "check",
                    "--workspace",
                    "--all-targets",
                    "--offline",
                    "--locked",
                ],
            )
        }
        "test" => {
            require_len(&args, 0, 0)?;
            run_status(cargo(), &["test", "--workspace", "--offline", "--locked"])
        }
        "build" => {
            require_len(&args, 0, 0)?;
            run_status(
                cargo(),
                &["build", "--workspace", "--release", "--offline", "--locked"],
            )
        }
        "fmt" => {
            require_len(&args, 0, 0)?;
            run_status(cargo(), &["fmt", "--all", "--", "--check"])
        }
        "verify-package" => {
            require_len(&args, 0, 1)?;
            let location = args.first().map(PathBuf::from).unwrap_or_else(root);
            let errors = package::verify_extracted_manifest(&location)?;
            for error in &errors {
                eprintln!("{error}");
            }
            if errors.is_empty() {
                println!("Extracted package integrity verified.");
                Ok(())
            } else {
                Err(format!("{} package integrity errors", errors.len()).into())
            }
        }
        "package" => {
            require_len(&args, 1, 1)?;
            audit(false)?;
            let r = package::write_zip(&root(), Path::new(&args[0]))?;
            println!(
                "{} files; {} source bytes; {} ZIP bytes\nSHA-256 {}",
                r.files, r.source_bytes, r.zip_bytes, r.sha256
            );
            Ok(())
        }
        "compare" => {
            require_len(&args, 2, 2)?;
            compare::exact(&fs::read(&args[0])?, &fs::read(&args[1])?)?;
            println!("Exact byte comparison passed.");
            Ok(())
        }
        "numeric" => {
            require_len(&args, 4, 4)?;
            let t = compare::Tolerance::new(args[2].parse()?, args[3].parse()?)?;
            let r = compare::floats(&numeric(&args[0])?, &numeric(&args[1])?, t)?;
            println!(
                "Compared {} values; max absolute error {}; max relative error {}",
                r.compared, r.max_absolute_error, r.max_relative_error
            );
            Ok(())
        }
        "capture" => capture(&args),
        "benchmark-hash" => {
            require_len(&args, 0, 0)?;
            let data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 251) as u8).collect();
            let r = benchmark::measure(2, 9, || digest::sha256(std::hint::black_box(&data)))?;
            println!(
                "{{\"benchmark\":\"safe-rust-sha256-1MiB\",\"kind\":\"host-microbenchmark-not-inference\",\"warmups\":{},\"samples\":{:?},\"unit\":\"nanoseconds\",\"median_ns\":{},\"p90_ns\":{},\"mean_ns\":{},\"stddev_ns\":{}}}",
                r.warmups,
                r.measurements,
                r.distribution.median_ns,
                r.distribution.p90_ns,
                r.distribution.mean_ns,
                r.distribution.standard_deviation_ns
            );
            Ok(())
        }
        "storage" => {
            require_len(&args, 0, 1)?;
            storage(&args.first().map(PathBuf::from).unwrap_or_else(root))
        }
        "release-check" => {
            require_len(&args, 0, 0)?;
            release_check()
        }
        "hash" => {
            require_len(&args, 1, 1)?;
            let (bytes, hash) = digest::hash_reader(File::open(&args[0])?)?;
            println!("{hash}  {} ({bytes} bytes)", args[0]);
            Ok(())
        }
        _ => Err(format!("unknown task {command}; use --help").into()),
    }
}
fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("xtask: {error}");
            ExitCode::from(1)
        }
    }
}
