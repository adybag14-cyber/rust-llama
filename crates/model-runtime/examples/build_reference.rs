//! Optional isolated oracle build. This never builds native code for the Rust
//! runtime: callers must supply a separate frozen source tree and output dir.
use std::{
    path::{Path, PathBuf},
    process::Command,
};
const SOURCES: &[&str] = &[
    "ggml.c",
    "ggml.cpp",
    "ggml-alloc.c",
    "ggml-backend.cpp",
    "ggml-backend-meta.cpp",
    "ggml-backend-reg.cpp",
    "ggml-backend-dl.cpp",
    "ggml-threading.cpp",
    "ggml-quants.c",
    "gguf.cpp",
    "ggml-cpu/ggml-cpu.c",
    "ggml-cpu/ggml-cpu.cpp",
    "ggml-cpu/quants.c",
    "ggml-cpu/traits.cpp",
    "ggml-cpu/vec.cpp",
    "ggml-cpu/ops.cpp",
    "ggml-cpu/binary-ops.cpp",
    "ggml-cpu/unary-ops.cpp",
    "ggml-cpu/arch/x86/quants.c",
    "ggml-cpu/tiled/tiled.cpp",
    "ggml-cpu/tiled/tiled-kernel.cpp",
];
fn execute(command: &mut Command, log: &mut String) -> Result<(), Box<dyn std::error::Error>> {
    log.push_str(&format!("{command:?}\n"));
    let output = command.output()?;
    log.push_str(&String::from_utf8_lossy(&output.stdout));
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!(
            "compiler failed: {command:?}\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err("isolated reference recipe currently validated for Linux x86-64 only".into());
    }
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err(
            "usage: build_reference /path/llama.cpp-master /separate/output-directory".into(),
        );
    }
    let root = PathBuf::from(&args[0]).canonicalize()?;
    let report =
        llama_model_runtime::registry::audit_snapshot(&root).map_err(std::io::Error::other)?;
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&output)?;
    let output = output.canonicalize()?;
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or("missing workspace root")?
        .canonicalize()?;
    if output.starts_with(&workspace) || output.starts_with(&root) {
        return Err(
            "reference build output must be separate from both Rust and frozen source trees".into(),
        );
    }
    let ggml = root.join("ggml");
    let mut log = format!("Frozen registry: {report:?}\n");
    // Generated external-reference build metadata, not product implementation.
    std::fs::write(
        output.join("ggml-version.h"),
        "#define GGML_VERSION \"0.9.0-archive\"\n#define GGML_COMMIT \"2db1e23f-source-archive\"\n",
    )?;
    let mut objects = Vec::new();
    for source in SOURCES {
        let c = source.ends_with(".c");
        let compiler = if c { "gcc" } else { "g++" };
        let object = output.join(format!("{}.o", source.replace('/', "_")));
        let mut command = Command::new(compiler);
        command.args([
            "-O1",
            "-fPIC",
            if c { "-std=c11" } else { "-std=c++17" },
            "-D_GNU_SOURCE",
            "-DGGML_USE_CPU",
            "-DGGML_SCHED_MAX_COPIES=4",
        ]);
        for include in [
            ggml.join("include"),
            ggml.join("src"),
            ggml.join("src/ggml-cpu"),
            output.clone(),
        ] {
            command.arg("-I").arg(include);
        }
        command
            .arg("-c")
            .arg(ggml.join("src").join(source))
            .arg("-o")
            .arg(&object);
        if let Err(error) = execute(&mut command, &mut log) {
            std::fs::write(output.join("reference-build.log"), &log)?;
            return Err(error);
        }
        objects.push(object);
    }
    let library = output.join("libggml-reference.so");
    let mut link = Command::new("g++");
    link.args(["-shared", "-Wl,--no-undefined", "-o"])
        .arg(&library)
        .args(&objects)
        .args(["-lpthread", "-ldl", "-lm"]);
    let result = execute(&mut link, &mut log);
    std::fs::write(output.join("reference-build.log"), &log)?;
    result?;
    println!("{}", library.display());
    Ok(())
}
