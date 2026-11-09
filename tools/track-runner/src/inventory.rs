use super::*;
use std::collections::BTreeSet;

pub fn walk(root: &Path, base: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(base)?.collect::<std::result::Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == ".git" || name == "target" {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(format!(
                "symlink is outside archive policy: {}",
                entry.path().display()
            )
            .into());
        }
        if kind.is_dir() {
            walk(root, &entry.path(), files)?;
        } else if kind.is_file() {
            files.push(entry.path().strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}
fn classification(path: &str) -> Option<&'static str> {
    if path.starts_with("tools/rpc/") {
        return Some("T08");
    }
    for (track, families) in [
        ("T06", &["cuda", "hip", "musa"][..]),
        ("T07", &["metal", "vulkan", "opencl", "webgpu"][..]),
        (
            "T08",
            &[
                "blas", "cann", "et", "sycl", "zdnn", "hexagon", "zendnn", "openvino", "rpc",
                "virtgpu",
            ][..],
        ),
    ] {
        if families.iter().any(|family| {
            path.starts_with(&format!("ggml/src/ggml-{family}/"))
                || path == format!("ggml/include/ggml-{family}.h")
        }) {
            return Some(track);
        }
    }
    if let Some(name) = path.strip_prefix("src/") {
        if name.starts_with("models/") {
            return Some("T05");
        }
        if [
            "CMakeLists.txt",
            "llama-version.h.in",
            "llama.cpp",
            "llama-ext.h",
        ]
        .contains(&name)
        {
            return None;
        }
        if [
            "llama-impl",
            "llama-mmap",
            "llama-chat",
            "llama-grammar",
            "llama-sampler",
            "llama-vocab",
            "unicode",
            "llama-model-loader",
            "llama-model-saver",
            "llama-io",
            "llama-quant",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
        {
            return None;
        }
        return Some("T05");
    }
    None
}
pub fn audit_reference(root: &Path, output: &Path) -> Result<()> {
    if !root.join("ggml/include/ggml.h").is_file() || !root.join("src/llama-arch.h").is_file() {
        return Err("reference root does not look like the frozen llama.cpp archive".into());
    }
    if output
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("reference-audit output path must not contain parent traversal".into());
    }
    let absolute_output = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    let mut ancestor = absolute_output.as_path();
    while !ancestor.exists() {
        ancestor = ancestor.parent().ok_or("output has no existing parent")?;
    }
    let prospective_output = ancestor
        .canonicalize()?
        .join(absolute_output.strip_prefix(ancestor)?);
    if prospective_output.starts_with(root.canonicalize()?) {
        return Err("reference audit output must be outside the frozen source tree".into());
    }
    let mut files = vec![];
    walk(root, root, &mut files)?;
    files.sort();
    fs::create_dir_all(output)?;
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut ledger = String::from("track\tpath\tbytes\tsha256\tdisposition\tclosure\trust_area\n");
    for relative in files {
        let path = relative.to_string_lossy().replace('\\', "/");
        if let Some(track) = classification(&path) {
            let bytes = fs::read(root.join(&relative))?;
            let extension = relative.extension().and_then(|e| e.to_str()).unwrap_or("");
            let disposition = if path.ends_with("CMakeLists.txt")
                || extension == "cmake"
                || path.ends_with(".cmake.in")
            {
                "supersede-build-plumbing"
            } else if extension == "in" {
                "rewrite-template-semantics-and-generate-from-Rust"
            } else if ["md", "txt", "json", "csv", "png", "jpg", "svg"].contains(&extension) {
                "retain-or-regenerate-data-docs"
            } else {
                "rewrite"
            };
            let area = match track {
                "T05" => "crates/model-runtime",
                "T06" => "crates/device-family",
                "T07" => "crates/kernel-ir + crates/portable-gpu",
                _ => "crates/vendor-remote",
            };
            writeln!(
                ledger,
                "{track}\t{path}\t{}\t{}\t{disposition}\tOPEN: track partial; this row does not assert replacement parity\t{area}",
                bytes.len(),
                sha256_hex(&bytes)
            )?;
            *counts.entry(track).or_default() += 1;
        }
    }
    fs::write(output.join("reference-path-ledger.tsv"), ledger)?;
    let summary = format!(
        "# Scoped frozen-source inventory\n\nExpected archive SHA-256: `2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122`. This command hashes the supplied source paths; it does not independently validate the original ZIP identity. The delivery's separate input manifest records that archive verification.\n\nCounts from the plan's explicit path rules: {counts:?}.\n\nEvery row remains open at capability-completion level. The ledger is a hashed responsibility map, not a file-conversion percentage. `src/llama-version.h.in` is excluded as T12 build/version metadata per Appendix A. All four path counts match the plan (218/295/601/571). The model implementation `src/models/llama.cpp` belongs to T05; the distinct facade file `src/llama.cpp` belongs to T01.\n"
    );
    fs::write(output.join("REFERENCE-INVENTORY.md"), summary)?;
    println!("Frozen scoped source counts: {counts:?}");
    Ok(())
}

pub fn audit_product(root: &Path) -> Result<()> {
    let mut files = vec![];
    walk(root, root, &mut files)?;
    let mut generated = BTreeMap::<String, String>::new();
    let artifact_path = root.join("generated/artifact-manifest.tsv");
    if artifact_path.exists() {
        for line in fs::read_to_string(&artifact_path)?.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 6 {
                return Err("malformed generated-artifact manifest".into());
            }
            generated.insert(format!("generated/{}", fields[0]), fields[2].into());
        }
    }
    let mut fixture_hashes = BTreeMap::<String, String>::new();
    let fixture_path = root.join("docs/fixture-manifest.tsv");
    if fixture_path.exists() {
        for line in fs::read_to_string(fixture_path)?.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 4 {
                return Err("malformed fixture manifest".into());
            }
            if fields[0].starts_with('/') || fields[0].split('/').any(|p| p == ".." || p.is_empty())
            {
                return Err("unsafe fixture path".into());
            }
            fixture_hashes.insert(fields[0].into(), fields[2].into());
        }
    }
    let mut extensions = BTreeMap::<String, usize>::new();
    let mut rs_files = 0;
    let mut rs_lines = 0;
    let mut external = vec![];
    let mut referenced = BTreeSet::new();
    for relative in files {
        let path = relative.to_string_lossy().replace('\\', "/");
        let bytes = fs::read(root.join(&relative))?;
        let extension = relative.extension().and_then(|e| e.to_str()).unwrap_or("");
        *extensions.entry(extension.into()).or_default() += 1;
        if extension == "rs" {
            rs_files += 1;
            rs_lines += bytes.iter().filter(|&&b| b == b'\n').count();
            continue;
        }
        if let Some(expected) = generated.get(&path) {
            if &sha256_hex(&bytes) != expected {
                return Err(format!("generated artifact hash mismatch: {path}").into());
            }
            referenced.insert(path);
            continue;
        }
        if let Some(expected) = fixture_hashes.get(&path) {
            if &sha256_hex(&bytes) != expected {
                return Err(format!("fixture hash mismatch: {path}").into());
            }
            referenced.insert(path);
            continue;
        }
        if path == "LICENSE"
            || path == ".gitignore"
            || path == "Cargo.lock"
            || path.ends_with("Cargo.toml")
            || ["md", "json", "tsv", "txt", "log", "sha256"].contains(&extension)
        {
            continue;
        }
        if path == "bin/track-runner-linux-x86_64" {
            external.push(path);
            continue;
        }
        return Err(format!("unapproved non-Rust implementation/artifact path: {path}").into());
    }
    for name in generated.keys() {
        if !referenced.contains(name) {
            return Err(format!("generated artifact is missing: {name}").into());
        }
    }
    for name in fixture_hashes.keys() {
        if !referenced.contains(name) {
            return Err(format!("listed fixture is missing: {name}").into());
        }
    }
    println!(
        "Rust source files={rs_files}; physical Rust lines={rs_lines}; extensions={extensions:?}"
    );
    println!(
        "Generated files plus binary fixtures verified by SHA-256={}; derived Rust binaries={external:?}",
        referenced.len()
    );
    println!(
        "No maintained C/C++/CUDA/Python/JS/TS/shell source accepted. Semantic and dependency review still required; this is a path/provenance audit, not a full rewrite-completion proof."
    );
    Ok(())
}
