//! Exact, local-only audit of the reviewed default Cargo dependency graph.
//! Cargo fetch is a separate operation. This module neither downloads nor builds
//! third-party packages, and rejects source replacement, feature and cache drift.
use crate::Result;
use llama_checkpoint::{
    Limits,
    json::{self, Object, Value},
};
use llama_conformance::{digest, process};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

const POLICY_SHA256: &str = "85ee48966a886f28fe38ed8dba424b72f6dd0ea3e2ae606fa8af0c8ad2beddaa";
const REVIEW_SHA256: &str = "1d3194f47ce918062d4471324d679e8771c400178afd4cd6526fe935432ea19f";
const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_POLICY_BYTES: usize = 256 * 1024;
const MAX_LOCK_BYTES: usize = 1024 * 1024;
const MAX_PACKAGES: usize = 256;
const MAX_CACHE_FILES: usize = 4096;
const MAX_CACHE_BYTES: u64 = 16 * 1024 * 1024;
const SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";

#[derive(Debug)]
struct FileIdentity<'a> {
    path: &'a str,
    bytes: u64,
    sha256: &'a str,
}
#[derive(Debug)]
struct Approved<'a> {
    name: &'a str,
    version: &'a str,
    source: &'a str,
    checksum: &'a str,
    license: &'a str,
    features: BTreeSet<&'a str>,
    archive_bytes: u64,
    files: Vec<FileIdentity<'a>>,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Identity {
    name: String,
    version: String,
    source: Option<String>,
}
#[derive(Debug)]
struct Locked {
    identity: Identity,
    checksum: Option<String>,
}
#[derive(Debug)]
struct RegistryPackage<'a> {
    approved: &'a Approved<'a>,
    manifest: PathBuf,
}
#[derive(Debug)]
struct Graph<'a> {
    registry: Vec<RegistryPackage<'a>>,
    local_manifests: Vec<PathBuf>,
}
pub struct Report {
    local_packages: usize,
    registry_packages: usize,
    source_files: usize,
    source_bytes: u64,
    archive_bytes: u64,
}
impl Report {
    pub fn print(&self) {
        println!(
            "Dependency audit passed: {} local packages; {} exactly reviewed registry packages; {} source files ({} bytes); {} archive bytes. Locked identities, checksums, resolved default features, licenses and actual cache bytes match the pinned review. No build scripts or native links are admitted.",
            self.local_packages,
            self.registry_packages,
            self.source_files,
            self.source_bytes,
            self.archive_bytes
        );
        println!(
            "Policy SHA-256: {POLICY_SHA256}. This checks the default graph resolved by the pinned Cargo; it does not establish all-feature/backend closure."
        );
    }
}
fn regular_file(path: &Path) -> Result<File> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "dependency audit requires a regular non-symlink file: {}",
            path.display()
        )
        .into());
    }
    Ok(File::open(path)?)
}
fn read_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = regular_file(path)?;
    if file.metadata()?.len() > limit as u64 {
        return Err(format!(
            "dependency audit input exceeds {limit} bytes: {}",
            path.display()
        )
        .into());
    }
    let mut output = Vec::new();
    output.try_reserve_exact(limit + 1)?;
    file.take((limit + 1) as u64).read_to_end(&mut output)?;
    if output.len() > limit {
        return Err("dependency audit input grew beyond its byte limit".into());
    }
    Ok(output)
}
fn parse(bytes: &[u8]) -> Result<Value> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err("dependency audit JSON exceeds its byte limit".into());
    }
    Ok(json::parse(
        bytes,
        Limits {
            max_header_bytes: MAX_JSON_BYTES as u64,
            max_total_header_bytes: MAX_JSON_BYTES as u64,
            max_json_nodes: 200_000,
            max_text_bytes: 8 * 1024 * 1024,
            max_string_bytes: 256 * 1024,
            max_json_depth: 32,
            max_integer_digits: 32,
            ..Limits::default()
        },
    )?)
}
fn object(value: &Value) -> Result<&Object> {
    value
        .as_object()
        .ok_or_else(|| "dependency audit expected a JSON object".into())
}
fn field<'a>(value: &'a Object, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| format!("dependency audit missing JSON field {name}").into())
}
fn text(value: &Value) -> Result<&str> {
    value
        .as_text()
        .and_then(|v| v.as_str())
        .ok_or_else(|| "dependency audit expected Unicode text".into())
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .ok_or_else(|| "dependency audit expected a JSON array".into())
}
fn number(value: &Value) -> Result<u64> {
    if let Value::Number(n) = value
        && let Some(value) = n.as_integer().and_then(|n| n.as_u64())
    {
        return Ok(value);
    }
    Err("dependency audit expected an unsigned integer".into())
}
fn keys(value: &Object, expected: &[&str]) -> Result<()> {
    let actual = value
        .entries()
        .map(|(k, _)| k.as_str().ok_or("non-Unicode JSON key"))
        .collect::<std::result::Result<BTreeSet<_>, _>>()?;
    if actual != expected.iter().copied().collect() {
        return Err("dependency audit policy has missing or unknown fields".into());
    }
    Ok(())
}
fn sha(value: &str) -> Result<&str> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("dependency audit expected a lowercase SHA-256".into());
    }
    Ok(value)
}
fn relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.contains(['\\', ':', '\0'])
        || value.len() > 1024
        || value
            .split('/')
            .any(|v| v.is_empty() || v == "." || v == "..")
        || Path::new(value)
            .components()
            .any(|v| !matches!(v, Component::Normal(_)))
    {
        return Err("dependency audit found an unsafe relative source path".into());
    }
    Ok(())
}
fn is_native(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|v| v.to_str())
        .is_some_and(|ext| {
            [
                "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "cu", "cuh", "s", "asm", "o",
                "obj", "a", "lib", "dll", "so", "dylib", "exe", "wasm",
            ]
            .iter()
            .any(|v| ext.eq_ignore_ascii_case(v))
        })
}
fn policy(value: &Value) -> Result<Vec<Approved<'_>>> {
    let top = object(value)?;
    keys(top, &["schema", "cohort", "review_sha256", "packages"])?;
    if number(field(top, "schema")?)? != 1
        || text(field(top, "cohort")?)? != "yaml-rust2-0.13.0-without-default-features"
        || text(field(top, "review_sha256")?)? != REVIEW_SHA256
    {
        return Err("dependency policy does not bind the approved review".into());
    }
    let packages = array(field(top, "packages")?)?;
    if packages.len() != 5 {
        return Err("dependency policy must contain the five reviewed packages".into());
    }
    let mut output = Vec::new();
    let mut names = BTreeSet::new();
    let mut total_bytes = 0_u64;
    let mut total_files = 0;
    for value in packages {
        let p = object(value)?;
        keys(
            p,
            &[
                "name",
                "version",
                "source",
                "checksum",
                "features",
                "license",
                "archive_bytes",
                "files",
            ],
        )?;
        let name = text(field(p, "name")?)?;
        let version = text(field(p, "version")?)?;
        let source = text(field(p, "source")?)?;
        if !names.insert(name)
            || source != SOURCE
            || name.is_empty()
            || version.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            || !version
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b".+-".contains(&b))
        {
            return Err("duplicate or invalid reviewed package identity".into());
        }
        let mut features = BTreeSet::new();
        for feature in array(field(p, "features")?)? {
            if !features.insert(text(feature)?) || features.len() > 16 {
                return Err("duplicate or excessive reviewed features".into());
            }
        }
        let mut files = Vec::new();
        let mut paths = BTreeSet::new();
        for value in array(field(p, "files")?)? {
            let file = object(value)?;
            keys(file, &["path", "bytes", "sha256"])?;
            let path = text(field(file, "path")?)?;
            relative(path)?;
            if !paths.insert(path)
                || is_native(path)
                || path == "build.rs"
                || matches!(path, ".cargo-ok" | ".cargo-checksum.json")
            {
                return Err("duplicate, native or generated file in reviewed archive".into());
            }
            let bytes = number(field(file, "bytes")?)?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or("reviewed file size overflow")?;
            total_files += 1;
            if total_bytes > MAX_CACHE_BYTES || total_files > MAX_CACHE_FILES {
                return Err("reviewed archive inventory exceeds audit bounds".into());
            }
            files.push(FileIdentity {
                path,
                bytes,
                sha256: sha(text(field(file, "sha256")?)?)?,
            });
        }
        if !paths.contains("Cargo.toml") || files.is_empty() {
            return Err("reviewed package lacks its manifest".into());
        }
        let archive_bytes = number(field(p, "archive_bytes")?)?;
        if archive_bytes == 0 || archive_bytes > MAX_CACHE_BYTES {
            return Err("invalid archive byte limit".into());
        }
        output.push(Approved {
            name,
            version,
            source,
            checksum: sha(text(field(p, "checksum")?)?)?,
            license: text(field(p, "license")?)?,
            features,
            archive_bytes,
            files,
        });
    }
    if total_files != 132 {
        return Err("reviewed source inventory must contain 132 files".into());
    }
    Ok(output)
}
fn quoted(value: &str) -> Result<&str> {
    let text = value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or("noncanonical Cargo.lock string")?;
    if text.len() > 8192
        || !text
            .bytes()
            .all(|b| (0x20..=0x7e).contains(&b) && b != b'"' && b != b'\\')
    {
        return Err("unsupported Cargo.lock string escape or encoding".into());
    }
    Ok(text)
}
fn finish_locked(fields: BTreeMap<String, String>) -> Result<Locked> {
    let get = |name: &str| {
        fields
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Cargo.lock package missing {name}"))
    };
    let name = get("name")?;
    let version = get("version")?;
    if name.is_empty() || version.is_empty() {
        return Err("empty Cargo.lock package identity".into());
    }
    let source = fields.get("source").cloned();
    let checksum = fields.get("checksum").cloned();
    if source.is_none() && checksum.is_some() {
        return Err("local Cargo.lock package has an external checksum".into());
    }
    Ok(Locked {
        identity: Identity {
            name,
            version,
            source,
        },
        checksum,
    })
}
fn locked(bytes: &[u8], approved: &[Approved<'_>]) -> Result<Vec<Locked>> {
    if bytes.len() > MAX_LOCK_BYTES {
        return Err("Cargo.lock exceeds audit byte limit".into());
    }
    let input = std::str::from_utf8(bytes)?;
    let mut version_seen = false;
    let mut current: Option<BTreeMap<String, String>> = None;
    let mut fields_seen = BTreeSet::new();
    let mut dependency_array = false;
    let mut output = Vec::new();
    for raw in input.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if dependency_array {
            if line == "]" {
                dependency_array = false;
            } else {
                quoted(line.strip_suffix(',').unwrap_or(line))?;
            }
            continue;
        }
        if line == "[[package]]" {
            if !version_seen {
                return Err("Cargo.lock lacks format version 4".into());
            }
            if let Some(fields) = current.take() {
                output.push(finish_locked(fields)?);
            }
            if output.len() >= MAX_PACKAGES {
                return Err("Cargo.lock has excessive packages".into());
            }
            current = Some(BTreeMap::new());
            fields_seen.clear();
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or("unsupported Cargo.lock syntax")?;
        let (key, value) = (key.trim(), value.trim());
        if let Some(fields) = current.as_mut() {
            if !fields_seen.insert(key.to_owned()) {
                return Err("duplicate Cargo.lock package field".into());
            }
            if key == "dependencies" {
                match value {
                    "[" => dependency_array = true,
                    "[]" => (),
                    _ => return Err("unsupported Cargo.lock dependency array".into()),
                }
            } else if ["name", "version", "source", "checksum"].contains(&key) {
                fields.insert(key.to_owned(), quoted(value)?.to_owned());
            } else {
                return Err("unreviewed Cargo.lock field or section".into());
            }
        } else if key == "version" && value == "4" && !version_seen {
            version_seen = true;
        } else {
            return Err("unsupported Cargo.lock header".into());
        }
    }
    if dependency_array {
        return Err("unterminated Cargo.lock dependency array".into());
    }
    if let Some(fields) = current {
        output.push(finish_locked(fields)?);
    }
    if !version_seen || output.is_empty() {
        return Err("empty Cargo.lock graph".into());
    }
    let mut seen = BTreeSet::new();
    let mut external = BTreeSet::new();
    for package in &output {
        if !seen.insert(&package.identity) {
            return Err("duplicate Cargo.lock package".into());
        }
        if let Some(source) = &package.identity.source {
            let p = approved
                .iter()
                .find(|p| p.name == package.identity.name)
                .ok_or("unreviewed Cargo.lock dependency name")?;
            if p.version != package.identity.version
                || p.source != source
                || package.checksum.as_deref() != Some(p.checksum)
                || !external.insert(p.name)
            {
                return Err("Cargo.lock version/source/checksum differs from review".into());
            }
        }
    }
    if external.len() != approved.len() {
        return Err("Cargo.lock is missing a reviewed dependency".into());
    }
    Ok(output)
}
fn graph<'a>(
    metadata: &'a Value,
    lock: &[Locked],
    approved: &'a [Approved<'a>],
) -> Result<Graph<'a>> {
    let top = object(metadata)?;
    if number(field(top, "version")?)? != 1 {
        return Err("unsupported Cargo metadata version".into());
    }
    let packages = array(field(top, "packages")?)?;
    if packages.len() != lock.len() || packages.len() > MAX_PACKAGES {
        return Err("Cargo metadata and lock package sets differ".into());
    }
    let nodes = array(field(object(field(top, "resolve")?)?, "nodes")?)?;
    let mut resolved = BTreeMap::new();
    for node in nodes {
        let node = object(node)?;
        let id = text(field(node, "id")?)?;
        let mut features = BTreeSet::new();
        for value in array(field(node, "features")?)? {
            if !features.insert(text(value)?) {
                return Err("duplicate resolved Cargo feature".into());
            }
        }
        if resolved.insert(id, features).is_some() {
            return Err("duplicate Cargo resolve node".into());
        }
    }
    if resolved.len() != packages.len() {
        return Err("Cargo resolve nodes and packages differ".into());
    }
    let mut registry = Vec::new();
    let mut local_manifests = Vec::new();
    let mut seen = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for value in packages {
        let p = object(value)?;
        let name = text(field(p, "name")?)?;
        let version = text(field(p, "version")?)?;
        let id = text(field(p, "id")?)?;
        if !ids.insert(id) {
            return Err("duplicate Cargo package id".into());
        }
        let source = match field(p, "source")? {
            Value::Null => None,
            other => Some(text(other)?),
        };
        let identity = Identity {
            name: name.to_owned(),
            version: version.to_owned(),
            source: source.map(str::to_owned),
        };
        if !seen.insert(identity.clone()) || !lock.iter().any(|p| p.identity == identity) {
            return Err("Cargo package is absent from the exact locked graph".into());
        }
        if !matches!(field(p, "links")?, Value::Null) {
            return Err("native Cargo links require a separate review".into());
        }
        for target in array(field(p, "targets")?)? {
            for kind in array(field(object(target)?, "kind")?)? {
                if text(kind)? == "custom-build" {
                    return Err("Cargo custom build scripts require a separate review".into());
                }
            }
        }
        let features = resolved
            .get(id)
            .ok_or("Cargo package lacks its resolved feature node")?;
        let manifest = PathBuf::from(text(field(p, "manifest_path")?)?);
        if let Some(source) = source {
            let policy = approved
                .iter()
                .find(|p| p.name == name)
                .ok_or("unreviewed Cargo metadata dependency")?;
            if version != policy.version
                || source != policy.source
                || features != &policy.features
                || text(field(p, "license")?)? != policy.license
            {
                return Err(format!(
                    "Cargo resolved identity/features/license differ from review: {name}"
                )
                .into());
            }
            registry.push(RegistryPackage {
                approved: policy,
                manifest,
            });
        } else {
            local_manifests.push(manifest);
        }
    }
    if registry.len() != approved.len() {
        return Err("Cargo metadata omitted a reviewed dependency".into());
    }
    Ok(Graph {
        registry,
        local_manifests,
    })
}
fn same_file(path: &Path, bytes: u64, sha256: &str) -> Result<()> {
    let file = regular_file(path)?;
    if file.metadata()?.len() != bytes {
        return Err(format!("reviewed file size differs: {}", path.display()).into());
    }
    let measured =
        digest::hash_reader(file.take(bytes.checked_add(1).ok_or("file length overflow")?))?;
    if measured.0 != bytes || measured.1 != sha256 {
        return Err(format!("reviewed file hash differs: {}", path.display()).into());
    }
    Ok(())
}
fn walk(
    root: &Path,
    at: &Path,
    depth: usize,
    entries: &mut usize,
    files: &mut BTreeSet<String>,
) -> Result<()> {
    if depth > 16 {
        return Err("Cargo source directory depth exceeds audit limit".into());
    }
    for entry in fs::read_dir(at)? {
        *entries += 1;
        if *entries > MAX_CACHE_FILES {
            return Err("Cargo source directory entries exceed audit limit".into());
        }
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("linked file/directory in reviewed Cargo source".into());
        }
        if kind.is_dir() {
            walk(root, &path, depth + 1, entries, files)?;
        } else if kind.is_file() {
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .ok_or("non-Unicode Cargo source path")?
                .replace('\\', "/");
            if is_native(&relative) || relative == "build.rs" {
                return Err("native source or build script in Cargo source cache".into());
            }
            files.insert(relative);
        } else {
            return Err("non-regular entry in reviewed Cargo source".into());
        }
    }
    Ok(())
}
fn verify_source(root: &Path, approved: &Approved<'_>) -> Result<()> {
    let mut files = BTreeSet::new();
    walk(root, root, 0, &mut 0, &mut files)?;
    if files.remove(".cargo-ok") && read_limited(&root.join(".cargo-ok"), 32)? != b"{\"v\":1}" {
        return Err("unrecognized Cargo extraction marker".into());
    }
    if files.remove(".cargo-checksum.json") {
        let value = parse(&read_limited(
            &root.join(".cargo-checksum.json"),
            MAX_POLICY_BYTES,
        )?)?;
        let value = object(&value)?;
        keys(value, &["files", "package"])?;
        if text(field(value, "package")?)? != approved.checksum {
            return Err("Cargo checksum bookkeeping package differs".into());
        }
        let recorded = object(field(value, "files")?)?;
        if recorded.len() != approved.files.len() {
            return Err("Cargo checksum bookkeeping inventory differs".into());
        }
        for file in &approved.files {
            if text(field(recorded, file.path)?)? != file.sha256 {
                return Err("Cargo checksum bookkeeping file differs".into());
            }
        }
    }
    let expected = approved
        .files
        .iter()
        .map(|f| f.path.to_owned())
        .collect::<BTreeSet<_>>();
    if files != expected {
        return Err("Cargo extracted source file set differs from reviewed archive".into());
    }
    for file in &approved.files {
        same_file(&root.join(file.path), file.bytes, file.sha256)?;
    }
    Ok(())
}
fn cargo_home() -> Result<PathBuf> {
    let path = if let Some(path) = std::env::var_os("CARGO_HOME") {
        PathBuf::from(path)
    } else {
        let key = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        PathBuf::from(std::env::var_os(key).ok_or("cannot locate Cargo home")?).join(".cargo")
    };
    Ok(fs::canonicalize(path)?)
}
fn verify_materialized(root: &Path, home: &Path, graph: &Graph<'_>) -> Result<()> {
    let source_root = fs::canonicalize(home.join("registry/src"))?;
    for manifest in &graph.local_manifests {
        let canonical = fs::canonicalize(manifest)?;
        if !canonical.starts_with(root) || canonical.file_name() != Some(OsStr::new("Cargo.toml")) {
            return Err("local Cargo dependency escapes the audited workspace".into());
        }
        regular_file(&canonical)?;
    }
    for package in &graph.registry {
        let p = package.approved;
        let manifest = fs::canonicalize(&package.manifest)?;
        if manifest.file_name() != Some(OsStr::new("Cargo.toml")) {
            return Err("unexpected Cargo registry manifest name".into());
        }
        let owner = manifest.parent().ok_or("missing Cargo source owner")?;
        let registry = owner.parent().ok_or("missing Cargo source registry")?;
        let key = registry
            .file_name()
            .ok_or("missing Cargo registry cache key")?;
        if registry.parent() != Some(source_root.as_path())
            || owner.file_name() != Some(OsStr::new(&format!("{}-{}", p.name, p.version)))
        {
            return Err("Cargo dependency is not the reviewed registry cache source".into());
        }
        let archive = home
            .join("registry/cache")
            .join(key)
            .join(format!("{}-{}.crate", p.name, p.version));
        same_file(&archive, p.archive_bytes, p.checksum)?;
        verify_source(owner, p)?;
    }
    Ok(())
}
pub fn audit(root: &Path, cargo: &OsStr) -> Result<Report> {
    let root = fs::canonicalize(root)?;
    let policy_bytes = read_limited(
        &root.join("release/reviewed-dependencies.json"),
        MAX_POLICY_BYTES,
    )?;
    if digest::sha256(&policy_bytes) != POLICY_SHA256 {
        return Err("dependency policy bytes differ from the pinned review".into());
    }
    let review = read_limited(
        &root.join("docs/dependencies/yaml-rust2-review.json"),
        MAX_POLICY_BYTES,
    )?;
    if digest::sha256(&review) != REVIEW_SHA256 {
        return Err("dependency review bytes differ from the pinned review".into());
    }
    let policy_value = parse(&policy_bytes)?;
    let approved = policy(&policy_value)?;
    let lock_bytes = read_limited(&root.join("Cargo.lock"), MAX_LOCK_BYTES)?;
    let lock = locked(&lock_bytes, &approved)?;
    let mut probe = process::Probe::new(cargo);
    probe.arguments = [
        "metadata",
        "--offline",
        "--locked",
        "--format-version",
        "1",
        "--manifest-path",
    ]
    .iter()
    .map(|s| (*s).into())
    .chain([root.join("Cargo.toml").into_os_string()])
    .collect();
    probe.timeout = Duration::from_secs(30);
    probe.output_limit = MAX_JSON_BYTES;
    let result = process::run(&probe)?;
    if result.status != Some(0) || result.timed_out || result.output_truncated {
        return Err(format!(
            "offline locked Cargo metadata failed (status {:?}, timeout {}, truncated {}): {}",
            result.status,
            result.timed_out,
            result.output_truncated,
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    let metadata = parse(&result.stdout)?;
    if fs::canonicalize(text(field(object(&metadata)?, "workspace_root")?)?)? != root {
        return Err("Cargo metadata resolved a different workspace".into());
    }
    let graph = graph(&metadata, &lock, &approved)?;
    verify_materialized(&root, &cargo_home()?, &graph)?;
    if read_limited(&root.join("Cargo.lock"), MAX_LOCK_BYTES)? != lock_bytes
        || read_limited(
            &root.join("release/reviewed-dependencies.json"),
            MAX_POLICY_BYTES,
        )? != policy_bytes
    {
        return Err("dependency policy/lock changed during the audit".into());
    }
    Ok(Report {
        local_packages: graph.local_manifests.len(),
        registry_packages: graph.registry.len(),
        source_files: approved.iter().map(|p| p.files.len()).sum(),
        source_bytes: approved
            .iter()
            .flat_map(|p| &p.files)
            .map(|f| f.bytes)
            .sum(),
        archive_bytes: approved.iter().map(|p| p.archive_bytes).sum(),
    })
}

#[cfg(test)]
mod tests;
