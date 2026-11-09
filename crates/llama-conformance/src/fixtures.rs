//! Explicit fixture-manifest schemas and complete fixture-data integrity checks.
use crate::package::{SourceFile, safe_name};
use std::collections::{BTreeMap, BTreeSet};
use std::io;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureEntry {
    pub path: String,
    pub sha256: String,
    pub bytes: Option<u64>,
    /// Preserves the original declaration; does not infer numerical parity.
    pub provenance: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FixtureReport {
    pub manifests: usize,
    pub entries: usize,
    pub fixture_files: usize,
    pub size_and_purpose_entries: usize,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Supported historical schemas are identified by their exact complete header.
/// The size/purpose schema is an integrity declaration, not a substitute for an
/// independently captured numerical reference and its domain provenance.
pub fn parse_manifest(text: &str) -> io::Result<Vec<FixtureEntry>> {
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| invalid("empty fixture manifest"))?;
    let sized = match header {
        "path\tsha256\torigin\tsource_identity" => false,
        "path\tbytes\tsha256\tpurpose" => true,
        _ => {
            return Err(invalid(format!(
                "unsupported fixture-manifest header: {header}"
            )));
        }
    };
    let mut result = Vec::new();
    let mut paths = BTreeSet::new();
    for (line_number, line) in lines.enumerate() {
        let columns: Vec<_> = line.split('\t').collect();
        if columns.len() != 4 || columns.iter().any(|value| value.is_empty()) {
            return Err(invalid(format!("invalid fixture row {}", line_number + 2)));
        }
        if !safe_name(columns[0]) || !paths.insert(columns[0]) {
            return Err(invalid(format!(
                "unsafe or duplicate fixture path: {}",
                columns[0]
            )));
        }
        let sha = columns[if sized { 2 } else { 1 }];
        if sha.len() != 64
            || !sha
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(invalid(format!("invalid SHA-256 for {}", columns[0])));
        }
        let bytes = if sized {
            if !columns[1].bytes().all(|c| c.is_ascii_digit()) {
                return Err(invalid(format!("invalid byte count for {}", columns[0])));
            }
            Some(
                columns[1]
                    .parse::<u64>()
                    .map_err(|_| invalid("fixture size exceeds u64"))?,
            )
        } else {
            None
        };
        result.push(FixtureEntry {
            path: columns[0].into(),
            sha256: sha.into(),
            bytes,
            provenance: if sized {
                columns[3].into()
            } else {
                format!("{}; source identity: {}", columns[2], columns[3])
            },
        });
    }
    if result.is_empty() {
        return Err(invalid("fixture manifest has no data rows"));
    }
    Ok(result)
}

pub fn is_fixture_data(path: &str) -> bool {
    (path.starts_with("fixtures/")
        || path.contains("/fixtures/")
        || path == "crates/model-runtime/tests/data/ggml_reference.tsv")
        && !path.ends_with("fixture-manifest.tsv")
}

/// Verify all declared hashes and optional byte counts, reject missing files and
/// omissions, and preserve both imported schemas without rewriting provenance.
pub fn verify(files: &[SourceFile]) -> io::Result<FixtureReport> {
    let by_path: BTreeMap<_, _> = files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect();
    if by_path.len() != files.len() {
        return Err(invalid("duplicate source paths in fixture inventory"));
    }
    let mut report = FixtureReport::default();
    let mut covered = BTreeSet::new();
    for manifest in files
        .iter()
        .filter(|file| file.path.ends_with("fixture-manifest.tsv"))
    {
        report.manifests += 1;
        let text = std::str::from_utf8(&manifest.bytes)
            .map_err(|_| invalid(format!("non-UTF-8 fixture manifest: {}", manifest.path)))?;
        for entry in parse_manifest(text).map_err(|e| invalid(format!("{}: {e}", manifest.path)))? {
            let file = by_path.get(entry.path.as_str()).ok_or_else(|| {
                invalid(format!("{}: missing fixture {}", manifest.path, entry.path))
            })?;
            if file.sha256 != entry.sha256 {
                return Err(invalid(format!("fixture SHA-256 drift: {}", entry.path)));
            }
            if entry
                .bytes
                .is_some_and(|size| size != file.bytes.len() as u64)
            {
                return Err(invalid(format!("fixture byte-count drift: {}", entry.path)));
            }
            report.entries += 1;
            report.size_and_purpose_entries += usize::from(entry.bytes.is_some());
            covered.insert(entry.path);
        }
    }
    if report.manifests == 0 {
        return Err(invalid(
            "no fixture manifests; independent fixture provenance has not been recorded",
        ));
    }
    for file in files.iter().filter(|file| is_fixture_data(&file.path)) {
        if !covered.contains(&file.path) {
            return Err(invalid(format!(
                "fixture has no integrity/provenance declaration: {}",
                file.path
            )));
        }
        report.fixture_files += 1;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::sha256;
    fn file(path: &str, bytes: &[u8]) -> SourceFile {
        SourceFile {
            path: path.into(),
            bytes: bytes.into(),
            sha256: sha256(bytes),
            role: "test input or independent golden data",
        }
    }
    #[test]
    fn both_imported_schemas_validate_their_actual_fields() {
        let data = file("fixtures/input.bin", b"sample");
        let hash = &data.sha256;
        for text in [
            format!(
                "path\tsha256\torigin\tsource_identity\nfixtures/input.bin\t{hash}\toriginal oracle output\tfrozen source identity\n"
            ),
            format!(
                "path\tbytes\tsha256\tpurpose\nfixtures/input.bin\t6\t{hash}\tsynthetic frozen-oracle input\n"
            ),
        ] {
            let manifest = file("docs/example-fixture-manifest.tsv", text.as_bytes());
            let report = verify(&[data.clone(), manifest]).unwrap();
            assert_eq!(report.fixture_files, 1);
            assert_eq!(report.entries, 1);
        }
    }
    #[test]
    fn malformed_headers_paths_hashes_and_duplicate_rows_are_rejected() {
        let hash = sha256(b"sample");
        assert!(parse_manifest("path\tsha\torigin\tsource\n").is_err());
        for path in ["../escape", "/absolute", "C:/outside", "dir\\file"] {
            let text = format!("path\tbytes\tsha256\tpurpose\n{path}\t6\t{hash}\torigin\n");
            assert!(parse_manifest(&text).is_err(), "{path}");
        }
        let row = format!("fixtures/input.bin\t6\t{hash}\torigin\n");
        assert!(parse_manifest(&format!("path\tbytes\tsha256\tpurpose\n{row}{row}")).is_err());
        assert!(
            parse_manifest("path\tbytes\tsha256\tpurpose\nfixtures/a.bin\t-1\tx\ty\n").is_err()
        );
    }
    #[test]
    fn drift_wrong_size_missing_data_and_unrecorded_fixtures_fail() {
        let data = file("fixtures/input.bin", b"sample");
        let manifest = file(
            "docs/example-fixture-manifest.tsv",
            format!(
                "path\tbytes\tsha256\tpurpose\nfixtures/input.bin\t6\t{}\torigin\n",
                data.sha256
            )
            .as_bytes(),
        );
        assert!(verify(std::slice::from_ref(&manifest)).is_err());
        assert!(verify(&[file("fixtures/input.bin", b"tamper"), manifest.clone()]).is_err());
        assert!(
            verify(&[
                data.clone(),
                manifest.clone(),
                file("fixtures/unrecorded.bin", b"x")
            ])
            .is_err()
        );
        let wrong_size = file(
            "docs/example-fixture-manifest.tsv",
            format!(
                "path\tbytes\tsha256\tpurpose\nfixtures/input.bin\t7\t{}\torigin\n",
                data.sha256
            )
            .as_bytes(),
        );
        assert!(verify(&[data, wrong_size]).is_err());
    }
}
