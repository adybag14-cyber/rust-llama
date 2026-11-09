//! Deterministic source packages with fixed timestamps and a content manifest.
use crate::digest::{crc32, sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: String,
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub role: &'static str,
}
#[derive(Debug, Clone, Default)]
pub struct AuditReport {
    pub rust_files: usize,
    pub rust_lines: usize,
    pub non_rust_files: usize,
    pub total_bytes: u64,
    pub errors: Vec<String>,
    pub unsafe_mentions: Vec<String>,
}
pub fn role(path: &str) -> Option<&'static str> {
    if path.ends_with(".rs") {
        return Some("Rust implementation or test");
    }
    if matches!(
        path,
        "Cargo.toml"
            | "Cargo.lock"
            | "rust-toolchain.toml"
            | ".gitignore"
            | ".gitattributes"
            | ".cargo/config.toml"
    ) || path.ends_with("/Cargo.toml")
    {
        return Some("declarative build configuration");
    }
    if matches!(
        path,
        "LICENSE"
            | "LICENSE-UPSTREAM"
            | "NOTICE"
            | "README.md"
            | "PROGRESS.md"
            | "SHA256SUMS"
            | "docs/imports/t05-t08/LICENSE"
            | "docs/imports/t09-t12/LICENSE"
            | "docs/imports/t09-t12/NOTICE"
    ) || path.ends_with(".md")
    {
        return Some("documentation or license");
    }
    if (path.starts_with("evidence/")
        || path.starts_with("ledger/")
        || path.starts_with("release/")
        || path.starts_with("docs/"))
        && matches!(
            Path::new(path).extension().and_then(|s| s.to_str()),
            Some("json" | "jsonl" | "tsv" | "csv" | "txt" | "log" | "sha256")
        )
    {
        return Some("provenance, inventory or validation data");
    }
    if (path.starts_with("fixtures/") || path.contains("/fixtures/"))
        && matches!(
            Path::new(path).extension().and_then(|s| s.to_str()),
            Some(
                "json"
                    | "jsonl"
                    | "tsv"
                    | "csv"
                    | "txt"
                    | "gbnf"
                    | "jinja"
                    | "bin"
                    | "gguf"
                    | "ppm"
                    | "pgm"
                    | "pnm"
                    | "bmp"
                    | "wav"
                    | "hex"
                    | "png"
                    | "rgb"
            )
        )
    {
        return Some("test input or independent golden data");
    }
    if path == "crates/model-runtime/tests/data/ggml_reference.tsv" {
        return Some("frozen GGML synthetic model graph numerical reference data");
    }
    // These four tiny ELF objects are synthetic parser/rewriter inputs and
    // outputs, hash-pinned in docs/fixture-manifest.tsv. They are never loaded as
    // product implementations. Other ELF/executable paths remain unclassified.
    if matches!(
        path,
        "crates/vendor-remote/tests/fixtures/hexagon32-input.elf"
            | "crates/vendor-remote/tests/fixtures/hexagon32-reference-swiv.elf"
            | "crates/vendor-remote/tests/fixtures/hexagon64-input.elf"
            | "crates/vendor-remote/tests/fixtures/hexagon64-reference-swiv.elf"
    ) {
        return Some("pinned synthetic ELF parser/rewriter fixture; not executable product code");
    }
    if path == "crates/vendor-remote/examples/reference/inputs.sha256" {
        return Some("frozen external reference input hash inventory");
    }
    if path == ".github/workflows/ci.yml" {
        return Some("declarative CI entry point; substantive tasks are Rust");
    }
    None
}
pub(crate) fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && !name.contains(':')
        && !name.chars().any(char::is_control)
        && Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}
pub fn gather(root: &Path) -> io::Result<Vec<SourceFile>> {
    fn visit(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
        for item in fs::read_dir(dir)? {
            let item = item?;
            let path = item.path();
            let rel = path.strip_prefix(root).unwrap();
            // Recomputed when an extracted source package is packaged again.
            if rel == Path::new("PACKAGE_MANIFEST.tsv") {
                continue;
            }
            let name = item.file_name();
            let ty = item.file_type()?;
            let top_level = rel.components().count() == 1;
            if top_level && (ty.is_dir() || ty.is_file()) && name == ".git" {
                continue;
            }
            // A name alone must not hide source. Recognize only root Cargo
            // caches with Cargo's standard cache-directory signature.
            if top_level
                && ty.is_dir()
                && name
                    .to_str()
                    .is_some_and(|name| name == "target" || name.starts_with("target-"))
                && fs::read(path.join("CACHEDIR.TAG")).is_ok_and(|tag| {
                    tag.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55")
                })
            {
                continue;
            }
            if ty.is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("source symlink rejected: {}", rel.display()),
                ));
            }
            if ty.is_dir() {
                visit(root, &path, out)?;
            } else if ty.is_file() {
                out.push(rel.to_owned());
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "special source file rejected",
                ));
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    visit(root, root, &mut paths)?;
    paths.sort();
    let mut out = Vec::new();
    for path in paths {
        let name = path
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "source path must be UTF-8"))?
            .replace('\\', "/");
        if !safe_name(&name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsafe archive path",
            ));
        }
        let len = fs::metadata(root.join(&path))?.len();
        if len > 64 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("source file exceeds 64 MiB: {name}"),
            ));
        }
        let bytes = fs::read(root.join(&path))?;
        out.push(SourceFile {
            sha256: sha256(&bytes),
            bytes,
            role: role(&name).unwrap_or("UNCLASSIFIED"),
            path: name,
        });
    }
    Ok(out)
}
pub fn audit(files: &[SourceFile]) -> AuditReport {
    let mut out = AuditReport::default();
    for file in files {
        out.total_bytes += file.bytes.len() as u64;
        if file.role == "UNCLASSIFIED" {
            out.errors
                .push(format!("unclassified non-Rust item: {}", file.path));
        }
        if file.path.ends_with(".rs") {
            out.rust_files += 1;
            let text = String::from_utf8_lossy(&file.bytes);
            out.rust_lines += text.lines().count();
            for (i, line) in text.lines().enumerate() {
                let code = line.trim();
                if !code.starts_with("//")
                    && (code.starts_with("unsafe ") || code.contains(" unsafe "))
                    && (code.contains("unsafe {")
                        || code.contains("unsafe fn ")
                        || code.contains("unsafe impl ")
                        || code.contains("unsafe extern "))
                {
                    out.unsafe_mentions.push(format!("{}:{}", file.path, i + 1));
                }
                if (code.contains("include_str!") || code.contains("include_bytes!"))
                    && [".cpp\"", ".c\"", ".py\"", ".js\"", ".cu\"", ".wgsl\""]
                        .iter()
                        .any(|suffix| code.contains(suffix))
                {
                    out.errors.push(format!(
                        "foreign implementation include candidate at {}:{}",
                        file.path,
                        i + 1
                    ));
                }
            }
        } else {
            out.non_rust_files += 1;
        }
    }
    out
}
pub fn non_rust_allowlist(files: &[SourceFile]) -> String {
    let mut out = String::from("path\trole\tsha256\n");
    for f in files {
        if !f.path.ends_with(".rs") {
            out.push_str(&format!("{}\t{}\t{}\n", f.path, f.role, f.sha256));
        }
    }
    out
}

/// Check an extracted checkpoint's integrity, including missing or added files.
/// Build caches and the generated manifest itself are intentionally excluded.
pub fn verify_extracted_manifest(root: &Path) -> io::Result<Vec<String>> {
    let manifest = fs::read_to_string(root.join("PACKAGE_MANIFEST.tsv"))?;
    let mut expected = std::collections::BTreeMap::new();
    if manifest.lines().next() != Some("sha256\tbytes\tpath") {
        return Err(zip_error("invalid package manifest header"));
    }
    for line in manifest.lines().skip(1) {
        let parts: Vec<_> = line.split('\t').collect();
        if parts.len() != 3 || !safe_name(parts[2]) {
            return Err(zip_error("invalid package manifest row"));
        }
        let bytes: usize = parts[1]
            .parse()
            .map_err(|_| zip_error("invalid manifest byte count"))?;
        if parts[0].len() != 64 || !parts[0].bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(zip_error("invalid manifest digest"));
        }
        if expected
            .insert(parts[2].to_string(), (parts[0].to_string(), bytes))
            .is_some()
        {
            return Err(zip_error("duplicate package manifest path"));
        }
    }
    if expected.is_empty() {
        return Err(zip_error("empty package manifest"));
    }
    let mut errors = Vec::new();
    for file in gather(root)? {
        match expected.remove(&file.path) {
            None => errors.push(format!("unlisted file: {}", file.path)),
            Some((hash, len)) => {
                if hash != file.sha256 || len != file.bytes.len() {
                    errors.push(format!("changed file: {}", file.path));
                }
            }
        }
    }
    for path in expected.keys() {
        errors.push(format!("missing file: {path}"));
    }
    Ok(errors)
}
fn u16le(w: &mut impl Write, n: u16) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn u32le(w: &mut impl Write, n: u32) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn zip_error(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s)
}
#[derive(Debug, Clone)]
pub struct PackageReport {
    pub files: usize,
    pub source_bytes: u64,
    pub zip_bytes: u64,
    pub sha256: String,
}
pub fn write_zip(root: &Path, output: &Path) -> io::Result<PackageReport> {
    let mut files = gather(root)?;
    let report = audit(&files);
    if !report.errors.is_empty() {
        return Err(zip_error(&report.errors.join("\n")));
    }
    let mut manifest = String::from("sha256\tbytes\tpath\n");
    for f in &files {
        manifest.push_str(&format!("{}\t{}\t{}\n", f.sha256, f.bytes.len(), f.path));
    }
    files.push(SourceFile {
        path: "PACKAGE_MANIFEST.tsv".into(),
        sha256: sha256(manifest.as_bytes()),
        bytes: manifest.into_bytes(),
        role: "generated integrity manifest",
    });
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.windows(2).any(|w| w[0].path == w[1].path) {
        return Err(zip_error("duplicate ZIP path"));
    }
    if files.len() > u16::MAX as usize {
        return Err(zip_error("ZIP64 entry count unsupported"));
    }
    if output.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "output already exists; choose a new checkpoint name",
        ));
    }
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let partial = output.with_extension("zip.partial");
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    let result = (|| {
        let mut offset = 0u64;
        let mut directory = Vec::new();
        for f in &files {
            let name = f.path.as_bytes();
            let name_len =
                u16::try_from(name.len()).map_err(|_| zip_error("ZIP filename too long"))?;
            let size = u32::try_from(f.bytes.len())
                .map_err(|_| zip_error("ZIP64 file size unsupported"))?;
            let start = u32::try_from(offset).map_err(|_| zip_error("ZIP64 offset unsupported"))?;
            let crc = crc32(&f.bytes);
            u32le(&mut writer, 0x04034b50)?;
            u16le(&mut writer, 20)?;
            u16le(&mut writer, 0x800)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0x21)?;
            u32le(&mut writer, crc)?;
            u32le(&mut writer, size)?;
            u32le(&mut writer, size)?;
            u16le(&mut writer, name_len)?;
            u16le(&mut writer, 0)?;
            writer.write_all(name)?;
            writer.write_all(&f.bytes)?;
            offset += 30 + u64::from(name_len) + u64::from(size);
            directory.push((name, name_len, size, start, crc));
        }
        let central_start =
            u32::try_from(offset).map_err(|_| zip_error("ZIP64 central offset unsupported"))?;
        for (name, name_len, size, start, crc) in directory {
            u32le(&mut writer, 0x02014b50)?;
            u16le(&mut writer, (3 << 8) | 20)?;
            u16le(&mut writer, 20)?;
            u16le(&mut writer, 0x800)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0x21)?;
            u32le(&mut writer, crc)?;
            u32le(&mut writer, size)?;
            u32le(&mut writer, size)?;
            u16le(&mut writer, name_len)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0)?;
            u16le(&mut writer, 0)?;
            u32le(&mut writer, 0o100644 << 16)?;
            u32le(&mut writer, start)?;
            writer.write_all(name)?;
            offset += 46 + u64::from(name_len);
        }
        let central_size = u32::try_from(offset - u64::from(central_start))
            .map_err(|_| zip_error("ZIP64 directory size unsupported"))?;
        u32le(&mut writer, 0x06054b50)?;
        u16le(&mut writer, 0)?;
        u16le(&mut writer, 0)?;
        u16le(&mut writer, files.len() as u16)?;
        u16le(&mut writer, files.len() as u16)?;
        u32le(&mut writer, central_size)?;
        u32le(&mut writer, central_start)?;
        u16le(&mut writer, 0)?;
        writer.flush()?;
        writer.sync_all()?;
        Ok(())
    })();
    drop(writer);
    if let Err(e) = result {
        let _ = fs::remove_file(&partial);
        return Err(e);
    }
    // The caller owns this output path. A hard-link install refuses races that
    // create it between the existence check and finalization.
    fs::hard_link(&partial, output)?;
    fs::remove_file(&partial)?;
    let (zip_bytes, hash) = crate::digest::hash_reader(File::open(output)?)?;
    Ok(PackageReport {
        files: files.len(),
        source_bytes: report.total_bytes,
        zip_bytes,
        sha256: hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn temp() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "llama-package-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }
    #[test]
    fn archive_reproducible_and_no_caches() {
        let t = temp();
        let r = t.join("src");
        fs::create_dir_all(r.join("target")).unwrap();
        fs::write(r.join("README.md"), b"checkpoint").unwrap();
        fs::write(r.join("target/secret.bin"), b"cache").unwrap();
        fs::write(
            r.join("target/CACHEDIR.TAG"),
            b"Signature: 8a477f597d28d172789f06886806bc55\n",
        )
        .unwrap();
        let a = write_zip(&r, &t.join("a.zip")).unwrap();
        let b = write_zip(&r, &t.join("b.zip")).unwrap();
        assert_eq!(a.sha256, b.sha256);
        assert_eq!(a.files, 2);
        assert!(write_zip(&r, &t.join("a.zip")).is_err());
        fs::remove_dir_all(t).unwrap();
    }
    #[test]
    fn source_audit_rejects_foreign_code() {
        let f = SourceFile {
            path: "crates/app/main.cpp".into(),
            bytes: b"int main() {}".to_vec(),
            sha256: String::new(),
            role: "UNCLASSIFIED",
        };
        assert_eq!(audit(&[f]).errors.len(), 1);
        assert!(!safe_name("../escape"));
        assert!(!safe_name("/abs"));
        assert!(!safe_name("a\\b"));
        assert!(!safe_name("C:/windows"));
    }
    #[test]
    fn manifest_detects_same_length_corruption_and_extra_files() {
        let t = temp();
        fs::write(t.join("README.md"), "hello").unwrap();
        let manifest = format!("sha256\tbytes\tpath\n{}\t5\tREADME.md\n", sha256(b"hello"));
        fs::write(t.join("PACKAGE_MANIFEST.tsv"), manifest).unwrap();
        assert!(verify_extracted_manifest(&t).unwrap().is_empty());
        fs::write(t.join("README.md"), "jello").unwrap();
        assert_eq!(
            verify_extracted_manifest(&t).unwrap(),
            ["changed file: README.md"]
        );
        fs::write(t.join("extra.md"), "extra").unwrap();
        assert_eq!(verify_extracted_manifest(&t).unwrap().len(), 2);
        fs::remove_dir_all(t).unwrap();
    }
    #[test]
    fn cache_exclusions_require_root_location_and_cargo_signature() {
        let t = temp();
        fs::create_dir_all(t.join("target-marked")).unwrap();
        fs::write(
            t.join("target-marked/CACHEDIR.TAG"),
            b"Signature: 8a477f597d28d172789f06886806bc55\n",
        )
        .unwrap();
        fs::write(t.join("target-marked/build.cpp"), b"generated cache").unwrap();
        fs::create_dir_all(t.join("target-unmarked")).unwrap();
        fs::write(t.join("target-unmarked/hidden.cpp"), b"int hidden;").unwrap();
        fs::create_dir_all(t.join("crates/target-marked")).unwrap();
        fs::write(
            t.join("crates/target-marked/CACHEDIR.TAG"),
            b"Signature: 8a477f597d28d172789f06886806bc55\n",
        )
        .unwrap();
        fs::write(t.join("crates/target-marked/hidden.cpp"), b"int hidden;").unwrap();
        let files = gather(&t).unwrap();
        assert!(!files.iter().any(|f| f.path.starts_with("target-marked/")));
        assert!(files.iter().any(|f| f.path == "target-unmarked/hidden.cpp"));
        assert!(
            files
                .iter()
                .any(|f| f.path == "crates/target-marked/hidden.cpp")
        );
        assert!(!audit(&files).errors.is_empty());
        fs::remove_dir_all(t).unwrap();
    }

    #[test]
    fn special_data_roles_do_not_allow_arbitrary_foreign_implementations() {
        for path in [
            "LICENSE-UPSTREAM",
            "docs/imports/t05-t08/LICENSE",
            "docs/imports/t09-t12/NOTICE",
            ".github/workflows/ci.yml",
            "fixtures/inference/ggml-vocab-llama-spm.gguf",
            "crates/vendor-remote/tests/fixtures/hexagon32-input.elf",
        ] {
            assert!(role(path).is_some(), "{path}");
        }
        for path in [
            "crates/runtime/runtime.cpp",
            "fixtures/unreviewed.elf",
            "fixtures/old.dll",
            ".github/workflows/unknown.yml",
            "docs/imports/old/LICENSE.cpp",
        ] {
            assert!(role(path).is_none(), "{path}");
        }
    }
}
