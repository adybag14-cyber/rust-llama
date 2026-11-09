//! Frozen-source ownership, registries and drift detection.
//!
//! Dispositions are work instructions, not completion claims. A baseline file
//! may feed several capabilities; implementation evidence lives in track ledgers.
use crate::digest::hash_reader;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

pub const BASELINE_SHA256: &str =
    "2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122";
pub const TOOLCHAIN_SHA256: &str =
    "231fcae47ee950621374725b5fc3b5441d2fbda2955ac45dd7dc816d8876cfe7";
pub const EXPECTED_COUNTS: [usize; 12] = [8, 19, 79, 252, 218, 295, 601, 571, 108, 378, 1106, 447];
pub const BACKENDS: [&str; 18] = [
    "CPU", "CUDA", "HIP", "MUSA", "Metal", "Vulkan", "OpenCL", "WebGPU", "BLAS", "CANN", "ET",
    "SYCL", "zDNN", "Hexagon", "ZenDNN", "OpenVINO", "RPC", "VirtGPU",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRecord {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub owner: u8,
    pub disposition: &'static str,
    pub destination: &'static str,
}
fn any_prefix(s: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| s.starts_with(p))
}
fn backend_track(s: &str) -> Option<u8> {
    let name = s
        .strip_prefix("ggml-")
        .unwrap_or(s)
        .split('.')
        .next()
        .unwrap_or(s);
    match name {
        "cpu" => Some(3),
        "cuda" | "hip" | "musa" => Some(6),
        "metal" | "vulkan" | "opencl" | "webgpu" => Some(7),
        "blas" | "cann" | "et" | "sycl" | "zdnn" | "hexagon" | "zendnn" | "openvino" | "rpc"
        | "virtgpu" => Some(8),
        _ => None,
    }
}
pub fn owner(path: &str) -> u8 {
    if path.starts_with("include/") {
        return 1;
    }
    if let Some(p) = path.strip_prefix("src/") {
        if matches!(p, "llama.cpp" | "llama-ext.h") || any_prefix(p, &["llama-impl", "llama-mmap"])
        {
            return 1;
        }
        if any_prefix(p, &["llama-chat", "llama-grammar", "llama-sampler"]) {
            return 10;
        }
        if any_prefix(
            p,
            &[
                "llama-vocab",
                "unicode",
                "llama-model-loader",
                "llama-model-saver",
                "llama-io",
            ],
        ) {
            return 4;
        }
        if p.starts_with("llama-quant") {
            return 3;
        }
        if p.ends_with(".cpp") || p.ends_with(".c") || p.ends_with(".h") || p.ends_with(".hpp") {
            return 5;
        }
        return 12;
    }
    if let Some(p) = path.strip_prefix("ggml/src/") {
        if let Some((dir, _)) = p.split_once('/')
            && let Some(track) = backend_track(dir)
        {
            return track;
        }
        if any_prefix(p, &["ggml-quants", "ggml-common.h"]) {
            return 3;
        }
        if p == "gguf.cpp" {
            return 4;
        }
        if !p.contains('/')
            && (p.ends_with(".c")
                || p.ends_with(".cpp")
                || p.ends_with(".h")
                || p.ends_with(".hpp"))
        {
            return 2;
        }
        return 12;
    }
    if let Some(p) = path.strip_prefix("ggml/include/") {
        if p == "gguf.h" {
            return 4;
        }
        if let Some(track) = backend_track(p) {
            return track;
        }
        if p.ends_with(".h") {
            return 2;
        }
        return 12;
    }
    if path.starts_with("ggml/") {
        return 12;
    }
    if any_prefix(path, &["conversion/", "gguf-py/"])
        || (!path.contains('/') && path.starts_with("convert_") && path.ends_with(".py"))
    {
        return 4;
    }
    if path.starts_with("models/") {
        return if path.ends_with(".jinja") { 10 } else { 4 };
    }
    if any_prefix(path, &["common/", "grammars/", "app/"]) {
        return 10;
    }
    if let Some(p) = path.strip_prefix("tools/") {
        if any_prefix(p, &["mtmd/", "tts/"]) {
            return 9;
        }
        if any_prefix(p, &["server/", "ui/", "connect/"]) {
            return 11;
        }
        if p.starts_with("rpc/") {
            return 8;
        }
        return if p.contains('/') { 10 } else { 12 };
    }
    if let Some(p) = path.strip_prefix("examples/") {
        if any_prefix(p, &["llama.android/", "llama.swiftui/", "batched.swift/"]) {
            return 11;
        }
        if any_prefix(p, &["model-conversion/", "convert-llama2c-to-ggml/"]) {
            return 4;
        }
        return 10;
    }
    if any_prefix(path, &["vendor/stb/", "vendor/miniaudio/"]) {
        return 9;
    }
    if path.starts_with("vendor/cpp-httplib/") {
        return 11;
    }
    if any_prefix(
        path,
        &["vendor/nlohmann/", "vendor/sheredom/", "vendor/hash/"],
    ) {
        return 10;
    }
    12
}
pub fn destination(track: u8) -> &'static str {
    match track {
        1 => "contracts;platform;llama-ffi",
        2 => "tensor;graph;backend-api;scheduler;optim",
        3 => "quant;backend-cpu;cpu-kernels",
        4 => "gguf;checkpoint;convert;tokenizer;unicode",
        5 => "models;runtime;memory;adapters",
        6 => "backend-cuda;backend-hip;backend-musa",
        7 => "kernel-ir;kernel-codegen;portable-gpu-backends",
        8 => "vendor-backends;rpc;virtgpu",
        9 => "crates/llama-media;future-model-execution",
        10 => "crates/llama-behavior;future-runtime-tools",
        11 => "crates/llama-server;future-browser-native-clients",
        _ => "crates/llama-conformance;xtask;docs;release",
    }
}
pub fn disposition(path: &str) -> &'static str {
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    if name == "CMakeLists.txt"
        || name.starts_with("Makefile")
        || path.starts_with(".github/workflows/")
        || matches!(ext.as_str(), "cmake" | "dockerfile" | "mk")
        || name.ends_with(".Dockerfile")
    {
        return "superseded-build-plumbing";
    }
    if matches!(
        ext.as_str(),
        "c" | "cpp"
            | "cc"
            | "h"
            | "hpp"
            | "cu"
            | "cuh"
            | "m"
            | "metal"
            | "cl"
            | "comp"
            | "glsl"
            | "wgsl"
            | "s"
            | "py"
            | "ts"
            | "js"
            | "svelte"
            | "swift"
            | "kt"
            | "kts"
            | "sh"
            | "bat"
            | "ps1"
            | "awk"
            | "rs"
            | "html"
            | "css"
    ) {
        return "rewrite";
    }
    if name.ends_with(".h.in") {
        return "generate";
    }
    "retain-data-or-docs"
}
pub fn relative_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            let p = entry.path();
            if ty.is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("symlink rejected: {}", p.display()),
                ));
            }
            if ty.is_dir() {
                walk(root, &p, out)?;
            } else if ty.is_file() {
                out.push(p.strip_prefix(root).unwrap().to_path_buf());
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "special files are not supported",
                ));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort();
    Ok(out)
}
fn display_path(path: &Path) -> io::Result<String> {
    let p = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non UTF-8 path"))?
        .replace('\\', "/");
    if p.chars().any(char::is_control) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "control character in path",
        ));
    }
    Ok(p)
}
pub fn scan(root: &Path) -> io::Result<Vec<FileRecord>> {
    let mut out = Vec::new();
    for path in relative_files(root)? {
        let (bytes, sha256) = hash_reader(File::open(root.join(&path))?)?;
        let path = display_path(&path)?;
        let owner = owner(&path);
        let disposition = disposition(&path);
        out.push(FileRecord {
            path,
            bytes,
            sha256,
            owner,
            disposition,
            destination: destination(owner),
        });
    }
    Ok(out)
}
pub fn counts(records: &[FileRecord]) -> [usize; 12] {
    let mut out = [0; 12];
    for row in records {
        out[(row.owner - 1) as usize] += 1;
    }
    out
}
pub fn write_ledger(records: &[FileRecord], path: &Path) -> io::Result<()> {
    let mut out = String::from(
        "path\tbytes\tsha256\towner\tdisposition\tintended_destination\tbaseline_path_status\n",
    );
    for r in records {
        out.push_str(&format!(
            "{}\t{}\t{}\tT{:02}\t{}\t{}\tplanned\n",
            r.path, r.bytes, r.sha256, r.owner, r.disposition, r.destination
        ));
    }
    fs::write(path, out)
}
pub fn verify_ledger(root: &Path, ledger: &Path) -> io::Result<Vec<String>> {
    let data = fs::read_to_string(ledger)?;
    if data.lines().next()
        != Some(
            "path\tbytes\tsha256\towner\tdisposition\tintended_destination\tbaseline_path_status",
        )
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid baseline ledger header",
        ));
    }
    let mut old = BTreeMap::new();
    for (i, line) in data.lines().enumerate().skip(1) {
        let cols: Vec<_> = line.split('\t').collect();
        if cols.len() != 7 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ledger row {} malformed", i + 1),
            ));
        }
        if old
            .insert(
                cols[0].to_string(),
                (
                    cols[1].to_string(),
                    cols[2].to_string(),
                    cols[3].to_string(),
                ),
            )
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate ledger path",
            ));
        }
    }
    let mut errors = Vec::new();
    for row in scan(root)? {
        match old.remove(&row.path) {
            None => errors.push(format!("added: {}", row.path)),
            Some((bytes, hash, track)) => {
                if bytes != row.bytes.to_string() || hash != row.sha256 {
                    errors.push(format!("content changed: {}", row.path));
                }
                if track != format!("T{:02}", row.owner) {
                    errors.push(format!("owner changed: {}", row.path));
                }
            }
        }
    }
    for path in old.keys() {
        errors.push(format!("missing: {path}"));
    }
    Ok(errors)
}

#[derive(Debug, Clone)]
pub struct RegistryRow {
    pub kind: String,
    pub name: String,
    pub path: String,
    pub line: usize,
    pub owner: u8,
}
#[derive(Debug, Clone)]
struct Token {
    text: String,
    line: usize,
    string: bool,
}
// A small lexical scanner is used for discovery only; it is not a C++ parser.
fn lex(source: &str) -> Vec<Token> {
    let s = source.as_bytes();
    let mut i = 0;
    let mut line = 1;
    let mut out = Vec::new();
    while i < s.len() {
        if s[i] == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if s[i..].starts_with(b"//") {
            while i < s.len() && s[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if s[i..].starts_with(b"/*") {
            i += 2;
            while i < s.len() && !s[i..].starts_with(b"*/") {
                if s[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            i = (i + 2).min(s.len());
            continue;
        }
        if matches!(s[i], b'\'' | b'"') {
            let quote = s[i];
            let start = i + 1;
            let start_line = line;
            i += 1;
            while i < s.len() && s[i] != quote {
                if s[i] == b'\\' {
                    i += 1;
                }
                if i < s.len() && s[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            out.push(Token {
                text: String::from_utf8_lossy(&s[start..i.min(s.len())]).into_owned(),
                line: start_line,
                string: quote == b'"',
            });
            i = (i + 1).min(s.len());
            continue;
        }
        if s[i].is_ascii_alphanumeric() || s[i] == b'_' {
            let start = i;
            i += 1;
            while i < s.len() && (s[i].is_ascii_alphanumeric() || s[i] == b'_') {
                i += 1;
            }
            out.push(Token {
                text: String::from_utf8_lossy(&s[start..i]).into_owned(),
                line,
                string: false,
            });
            continue;
        }
        if !s[i].is_ascii_whitespace() {
            out.push(Token {
                text: (s[i] as char).to_string(),
                line,
                string: false,
            });
        }
        i += 1;
    }
    out
}
fn enum_rows(
    source: &str,
    enum_name: &str,
    prefix: &str,
    kind: &str,
    path: &str,
) -> Vec<RegistryRow> {
    let tokens = lex(source);
    let mut out = Vec::new();
    let Some(start) = tokens
        .windows(2)
        .position(|w| w[0].text == "enum" && w[1].text == enum_name)
    else {
        return out;
    };
    let mut body = false;
    for t in &tokens[start + 2..] {
        if t.text == "{" {
            body = true;
            continue;
        }
        if t.text == "}" {
            break;
        }
        if body && !t.string && t.text.starts_with(prefix) && !t.text.ends_with("_COUNT") {
            out.push(RegistryRow {
                kind: kind.into(),
                name: t.text.clone(),
                path: path.into(),
                line: t.line,
                owner: owner(path),
            });
        }
    }
    out
}
fn add(out: &mut Vec<RegistryRow>, kind: &str, name: impl Into<String>, path: &str, line: usize) {
    out.push(RegistryRow {
        kind: kind.into(),
        name: name.into(),
        path: path.into(),
        line,
        owner: owner(path),
    });
}

// CMake names contain hyphens and dots; C++ lexical tokens would truncate
// test-sampling.cpp to "test". Keep arguments intact and resolve NAME overrides.
fn cmake_registrations(source: &str, path: &str, out: &mut Vec<RegistryRow>) {
    let mut tokens: Vec<(String, usize)> = Vec::new();
    let mut chars = source.chars().peekable();
    let mut line = 1;
    while let Some(ch) = chars.next() {
        if ch == '\n' {
            line += 1;
            continue;
        }
        if ch.is_whitespace() {
            continue;
        }
        if ch == '#' {
            for c in chars.by_ref() {
                if c == '\n' {
                    line += 1;
                    break;
                }
            }
            continue;
        }
        if ch == '(' || ch == ')' {
            tokens.push((ch.to_string(), line));
            continue;
        }
        let start = line;
        let mut value = String::new();
        if ch == '"' {
            while let Some(c) = chars.next() {
                if c == '"' {
                    break;
                }
                if c == '\\' {
                    if let Some(c) = chars.next() {
                        value.push(c);
                    }
                    continue;
                }
                if c == '\n' {
                    line += 1;
                }
                value.push(c);
            }
        } else {
            value.push(ch);
            while chars
                .peek()
                .is_some_and(|c| !c.is_whitespace() && *c != '(' && *c != ')' && *c != '#')
            {
                value.push(chars.next().unwrap());
            }
        }
        tokens.push((value, start));
    }
    let mut i = 0;
    while i + 2 < tokens.len() {
        let func = &tokens[i].0;
        let kind = match func.as_str() {
            "option" => "build_option",
            "llama_option_depr" => "deprecated_build_option",
            "llama_build_and_test" | "llama_test" | "llama_test_cmd" | "add_test" => {
                "test_registration"
            }
            _ => {
                i += 1;
                continue;
            }
        };
        if tokens[i + 1].0 != "(" {
            i += 1;
            continue;
        }
        let end = (i + 2..tokens.len())
            .find(|j| tokens[*j].0 == ")")
            .unwrap_or(tokens.len());
        let args = &tokens[i + 2..end];
        let mut name = args.first().map(|t| t.0.as_str()).unwrap_or("");
        if kind == "test_registration"
            && let Some(n) = args.iter().position(|t| t.0 == "NAME")
        {
            name = args.get(n + 1).map(|t| t.0.as_str()).unwrap_or(name);
        }
        if kind == "deprecated_build_option" {
            name = args.get(1).map(|t| t.0.as_str()).unwrap_or(name);
        }
        if !name.is_empty() {
            add(out, kind, name, path, tokens[i].1);
        }
        i = end.saturating_add(1);
    }
}
pub fn registries(root: &Path) -> io::Result<Vec<RegistryRow>> {
    let mut out = Vec::new();
    for (path, enm, prefix, kind) in [
        ("src/llama-arch.h", "llm_arch", "LLM_ARCH_", "architecture"),
        ("ggml/include/ggml.h", "ggml_type", "GGML_TYPE_", "dtype"),
        ("ggml/include/ggml.h", "ggml_op", "GGML_OP_", "operation"),
        (
            "ggml/include/ggml.h",
            "ggml_unary_op",
            "GGML_UNARY_OP_",
            "unary",
        ),
        ("ggml/include/ggml.h", "ggml_glu_op", "GGML_GLU_OP_", "glu"),
    ] {
        out.extend(enum_rows(
            &fs::read_to_string(root.join(path))?,
            enm,
            prefix,
            kind,
            path,
        ));
    }
    for file in relative_files(root)? {
        let path = display_path(&file)?;
        if path.starts_with("src/models/") && path.ends_with(".cpp") {
            add(
                &mut out,
                "model_source",
                file.file_stem().unwrap().to_string_lossy(),
                &path,
                1,
            );
        }
        if path.starts_with("tools/mtmd/models/") && path.ends_with(".cpp") {
            add(
                &mut out,
                "multimodal_builder",
                file.file_stem().unwrap().to_string_lossy(),
                &path,
                1,
            );
        }
        let inspect = path == "common/arg.cpp"
            || path == "src/llama-chat.cpp"
            || path.starts_with("tools/server/") && path.ends_with(".cpp")
            || path.ends_with("CMakeLists.txt")
            || path.ends_with(".cmake")
            || path == "include/llama.h"
            || path.starts_with("ggml/include/") && path.ends_with(".h");
        if !inspect {
            continue;
        }
        let source = fs::read_to_string(root.join(&file))?;
        let tokens = lex(&source);
        if path == "common/arg.cpp" {
            for token in &tokens {
                if token.string
                    && token.text.starts_with('-')
                    && token.text.len() > 1
                    && !token.text.chars().any(char::is_whitespace)
                    && !token.text.contains('\\')
                {
                    add(&mut out, "cli_flag_literal", &token.text, &path, token.line);
                }
            }
        }
        if path == "src/llama-chat.cpp" {
            for w in tokens.windows(5) {
                if w[0].text == "{"
                    && w[1].string
                    && w[2].text == ","
                    && w[3].text.starts_with("LLM_CHAT_TEMPLATE_")
                    && w[4].text == "}"
                {
                    add(
                        &mut out,
                        "native_chat_template",
                        &w[1].text,
                        &path,
                        w[1].line,
                    );
                }
            }
        }
        if path.starts_with("tools/server/") {
            for w in tokens.windows(5) {
                if w[0].text == "."
                    && matches!(
                        w[1].text.as_str(),
                        "get" | "post" | "del" | "put" | "patch" | "ws"
                    )
                    && w[2].text == "("
                    && w[3].string
                    && w[3].text.starts_with('/')
                    && w[4].text == ","
                {
                    add(
                        &mut out,
                        "http_route_registration",
                        format!("{} {}", w[1].text.to_uppercase(), w[3].text),
                        &path,
                        w[3].line,
                    );
                }
            }
        }
        if path.ends_with("CMakeLists.txt") || path.ends_with(".cmake") {
            cmake_registrations(&source, &path, &mut out);
        }
        if path == "include/llama.h" || path.starts_with("ggml/include/") {
            for w in tokens.windows(2) {
                if !w[0].string
                    && (w[0].text.starts_with("llama_")
                        || w[0].text.starts_with("ggml_")
                        || w[0].text.starts_with("gguf_"))
                    && w[1].text == "("
                {
                    add(
                        &mut out,
                        "public_function_candidate",
                        &w[0].text,
                        &path,
                        w[0].line,
                    );
                }
            }
        }
    }
    for b in BACKENDS {
        let track = match b {
            "CPU" => 3,
            "CUDA" | "HIP" | "MUSA" => 6,
            "Metal" | "Vulkan" | "OpenCL" | "WebGPU" => 7,
            _ => 8,
        };
        out.push(RegistryRow {
            kind: "backend".into(),
            name: b.into(),
            path: "ggml/CMakeLists.txt".into(),
            line: 1,
            owner: track,
        });
    }
    out.sort_by(|a, b| {
        (&a.kind, &a.name, &a.path, a.line).cmp(&(&b.kind, &b.name, &b.path, b.line))
    });
    out.dedup_by(|a, b| {
        a.kind == b.kind && a.name == b.name && a.path == b.path && a.line == b.line
    });
    Ok(out)
}
pub fn write_registries(rows: &[RegistryRow], path: &Path) -> io::Result<()> {
    let mut s = String::from("kind\tname\tsource_path\tline\towner\tstatus\n");
    for row in rows {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\tT{:02}\tplanned\n",
            row.kind, row.name, row.path, row.line, row.owner
        ));
    }
    fs::write(path, s)
}
pub fn registry_counts(rows: &[RegistryRow]) -> BTreeMap<String, usize> {
    let mut kinds: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        kinds
            .entry(row.kind.clone())
            .or_default()
            .insert(row.name.clone());
    }
    kinds.into_iter().map(|(k, v)| (k, v.len())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ownership_specific_over_general() {
        for (p, t) in [
            ("include/llama.h", 1),
            ("src/llama-mmap.cpp", 1),
            ("src/llama-sampler.cpp", 10),
            ("src/llama-io.cpp", 4),
            ("src/models/llama.cpp", 5),
            ("src/CMakeLists.txt", 12),
            ("ggml/src/ggml-cuda/mmq.cu", 6),
            ("ggml/include/ggml-metal.h", 7),
            ("ggml/src/ggml-cpu/ggml-cpu.cpp", 3),
            ("ggml/src/ggml-alloc.c", 2),
            ("ggml/include/gguf.h", 4),
            ("ggml/src/ggml-virtgpu/CMakeLists.txt", 8),
            ("vendor/miniaudio/miniaudio.h", 9),
            ("tools/server/server.cpp", 11),
            ("tools/CMakeLists.txt", 12),
            ("examples/CMakeLists.txt", 10),
            ("skills/slice-model/slice_model.py", 12),
        ] {
            assert_eq!(owner(p), t, "{p}");
        }
    }
    #[test]
    fn removed_enum_comments_are_ignored() {
        let source = "enum ggml_type { GGML_TYPE_F32=0, /* GGML_TYPE_OLD=1, */ GGML_TYPE_F16=2, // GGML_TYPE_FAKE\n GGML_TYPE_COUNT };";
        let rows = enum_rows(
            source,
            "ggml_type",
            "GGML_TYPE_",
            "dtype",
            "ggml/include/ggml.h",
        );
        assert_eq!(
            rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            ["GGML_TYPE_F32", "GGML_TYPE_F16"]
        );
    }
    #[test]
    fn hidden_executable_paths_are_rewrite_obligations() {
        assert_eq!(disposition("skills/slice-model/slice_model.py"), "rewrite");
        assert_eq!(disposition("models/chat.jinja"), "retain-data-or-docs");
        assert_eq!(disposition("src/version.h.in"), "generate");
    }
    #[test]
    fn cmake_test_names_and_multiline_overrides_survive() {
        let mut rows = Vec::new();
        cmake_registrations(
            "# llama_test(fake)\nllama_build_and_test(test-sampling.cpp LABEL behavior)\nllama_test(test-tokenizer-0\n NAME test-tokenizer-qwen ARGS file.gguf)\noption(GGML_CPU \"description\" ON)",
            "tests/CMakeLists.txt",
            &mut rows,
        );
        assert_eq!(
            rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            ["test-sampling.cpp", "test-tokenizer-qwen", "GGML_CPU"]
        );
    }
    #[test]
    fn drift_detection_catches_added_removed_changed_bytes() {
        let dir =
            std::env::temp_dir().join(format!("llama-inventory-drift-{}", std::process::id()));
        fs::create_dir_all(dir.join("baseline")).unwrap();
        let baseline = dir.join("baseline");
        let ledger = dir.join("ledger.tsv");
        fs::write(baseline.join("keep.md"), "original").unwrap();
        fs::write(baseline.join("remove.md"), "remove").unwrap();
        write_ledger(&scan(&baseline).unwrap(), &ledger).unwrap();
        assert!(verify_ledger(&baseline, &ledger).unwrap().is_empty());
        fs::write(baseline.join("keep.md"), "modified").unwrap();
        fs::remove_file(baseline.join("remove.md")).unwrap();
        fs::write(baseline.join("added.md"), "added").unwrap();
        let errors = verify_ledger(&baseline, &ledger).unwrap();
        assert_eq!(errors.len(), 3);
        assert!(errors.iter().any(|e| e == "content changed: keep.md"));
        assert!(errors.iter().any(|e| e == "missing: remove.md"));
        assert!(errors.iter().any(|e| e == "added: added.md"));
        fs::remove_dir_all(dir).unwrap();
    }
}
