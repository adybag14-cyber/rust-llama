use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn approved_json() -> Value {
    let raw = include_bytes!("../../../release/reviewed-dependencies.json");
    assert_eq!(digest::sha256(raw), POLICY_SHA256);
    parse(raw).unwrap()
}
fn lock_text(approved: &[Approved<'_>]) -> String {
    let mut output = String::from("version = 4\n");
    for p in approved {
        output.push_str(&format!(
            "\n[[package]]\nname = {:?}\nversion = {:?}\nsource = {:?}\nchecksum = {:?}\n",
            p.name, p.version, p.source, p.checksum
        ));
    }
    output
}
fn metadata_text(approved: &[Approved<'_>]) -> String {
    let mut packages = Vec::new();
    let mut nodes = Vec::new();
    for p in approved {
        packages.push(format!(
            "{{\"name\":{:?},\"version\":{:?},\"id\":{:?},\"source\":{:?},\"license\":{:?},\"links\":null,\"targets\":[{{\"kind\":[\"lib\"]}}],\"manifest_path\":\"cache/Cargo.toml\"}}",
            p.name,p.version,p.name,p.source,p.license));
        let features = p
            .features
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",");
        nodes.push(format!("{{\"id\":{:?},\"features\":[{features}]}}", p.name));
    }
    format!(
        "{{\"version\":1,\"packages\":[{}],\"resolve\":{{\"nodes\":[{}]}}}}",
        packages.join(","),
        nodes.join(",")
    )
}
#[test]
fn policy_and_current_lock_have_exact_reviewed_identities() {
    let value = approved_json();
    let approved = policy(&value).unwrap();
    assert_eq!(approved.len(), 5);
    assert_eq!(approved.iter().map(|p| p.files.len()).sum::<usize>(), 132);
    locked(include_bytes!("../../../Cargo.lock"), &approved).unwrap();
    assert!(relative("../outside.rs").is_err());
    assert!(relative("a//b").is_err());
    assert!(relative("C:/outside.rs").is_err());
    assert!(relative("nested/source.rs").is_ok());
    assert!(sha("0000").is_err());
    assert!(is_native("hidden/source.CPP"));
}
#[test]
fn lock_rejects_unknown_version_checksum_source_and_duplicate_drift() {
    let value = approved_json();
    let approved = policy(&value).unwrap();
    let baseline = lock_text(&approved);
    assert_eq!(locked(baseline.as_bytes(), &approved).unwrap().len(), 5);
    for (from, to) in [
        ("name = \"arraydeque\"", "name = \"unreviewed\""),
        ("version = \"0.5.1\"", "version = \"0.5.2\""),
        (
            approved[0].checksum,
            "0000000000000000000000000000000000000000000000000000000000000000",
        ),
        (SOURCE, "git+https://invalid.example/replacement"),
        ("version = 4", "version = 3"),
    ] {
        let changed = baseline.replacen(from, to, 1);
        assert_ne!(changed, baseline);
        assert!(locked(changed.as_bytes(), &approved).is_err(), "{from}");
    }
    let first = baseline.split("[[package]]").nth(1).unwrap();
    assert!(
        locked(
            format!("{baseline}\n[[package]]{first}").as_bytes(),
            &approved
        )
        .is_err()
    );
    assert!(
        locked(
            b"version = 4\n[[package]]\nname = \"local\"\nversion = \"1\"\n",
            &approved
        )
        .is_err()
    );
}
#[test]
fn metadata_rejects_features_native_builds_links_and_source_drift() {
    let value = approved_json();
    let approved = policy(&value).unwrap();
    let lock = locked(lock_text(&approved).as_bytes(), &approved).unwrap();
    let baseline = metadata_text(&approved);
    let metadata = parse(baseline.as_bytes()).unwrap();
    assert_eq!(
        graph(&metadata, &lock, &approved).unwrap().registry.len(),
        5
    );
    for (from, to) in [
        (
            "\"features\":[\"default\",\"std\"]",
            "\"features\":[\"default\",\"encoding\",\"std\"]",
        ),
        ("\"kind\":[\"lib\"]", "\"kind\":[\"custom-build\"]"),
        ("\"links\":null", "\"links\":\"native-library\""),
        (
            "\"source\":\"registry+https://github.com/rust-lang/crates.io-index\"",
            "\"source\":null",
        ),
        (
            "\"license\":\"MIT/Apache-2.0\"",
            "\"license\":\"unreviewed-license\"",
        ),
    ] {
        let changed = baseline.replacen(from, to, 1);
        assert_ne!(changed, baseline);
        let metadata = parse(changed.as_bytes()).unwrap();
        assert!(graph(&metadata, &lock, &approved).is_err(), "{from}");
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rust-llama-dependency-audit-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn source_inventory_rejects_mutation_untracked_files_and_cache_marker_drift() {
    let temp = Temp::new();
    let content = b"[package]\nname = \"test\"\n";
    fs::write(temp.0.join("Cargo.toml"), content).unwrap();
    let hash = digest::sha256(content);
    let approved = Approved {
        name: "test",
        version: "1",
        source: SOURCE,
        checksum: &hash,
        license: "MIT",
        features: BTreeSet::new(),
        archive_bytes: 1,
        files: vec![FileIdentity {
            path: "Cargo.toml",
            bytes: content.len() as u64,
            sha256: &hash,
        }],
    };
    verify_source(&temp.0, &approved).unwrap();
    fs::write(temp.0.join(".cargo-ok"), b"{\"v\":1}").unwrap();
    verify_source(&temp.0, &approved).unwrap();
    fs::write(temp.0.join(".cargo-ok"), b"other").unwrap();
    assert!(verify_source(&temp.0, &approved).is_err());
    fs::write(temp.0.join(".cargo-ok"), b"{\"v\":1}").unwrap();
    fs::write(temp.0.join("Cargo.toml"), b"[package]\nname = \"evil\"\n").unwrap();
    assert!(verify_source(&temp.0, &approved).is_err());
    fs::write(temp.0.join("Cargo.toml"), content).unwrap();
    fs::write(temp.0.join("extra.rs"), b"pub fn extra() {}").unwrap();
    assert!(verify_source(&temp.0, &approved).is_err());
    fs::remove_file(temp.0.join("extra.rs")).unwrap();
    fs::write(temp.0.join("build.rs"), b"fn main() {}").unwrap();
    assert!(verify_source(&temp.0, &approved).is_err());
}
