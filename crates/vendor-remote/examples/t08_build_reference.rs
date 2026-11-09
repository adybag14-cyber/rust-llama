//! Opt-in, isolated reconstruction of the frozen native comparison library.
//! This is never called by Cargo build scripts or the Rust product runtime.
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const INPUTS: &str = include_str!("reference/inputs.sha256");
const SOURCES: &[&str] = &[
    "ggml/src/ggml.c",
    "ggml/src/ggml.cpp",
    "ggml/src/ggml-alloc.c",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-opt.cpp",
    "ggml/src/ggml-threading.cpp",
    "ggml/src/ggml-quants.c",
    "ggml/src/gguf.cpp",
    "ggml/src/ggml-backend-dl.cpp",
    "ggml/src/ggml-backend-reg.cpp",
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/ggml-cpu.cpp",
    "ggml/src/ggml-cpu/repack.cpp",
    "ggml/src/ggml-cpu/hbm.cpp",
    "ggml/src/ggml-cpu/quants.c",
    "ggml/src/ggml-cpu/traits.cpp",
    "ggml/src/ggml-cpu/amx/amx.cpp",
    "ggml/src/ggml-cpu/amx/mmq.cpp",
    "ggml/src/ggml-cpu/tiled/tiled.cpp",
    "ggml/src/ggml-cpu/tiled/tiled-kernel.cpp",
    "ggml/src/ggml-cpu/binary-ops.cpp",
    "ggml/src/ggml-cpu/unary-ops.cpp",
    "ggml/src/ggml-cpu/vec.cpp",
    "ggml/src/ggml-cpu/ops.cpp",
    "ggml/src/ggml-rpc/ggml-rpc.cpp",
    "ggml/src/ggml-rpc/log.cpp",
    "ggml/src/ggml-rpc/transport.cpp",
];

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

fn main() -> Result<()> {
    if !cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_endian = "little"
    )) {
        return Err(invalid(
            "this comparison recipe is validated only for x86-64 Linux",
        ));
    }
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err(invalid(
            "usage: t08_build_reference FROZEN_LLAMA_SOURCE EXTERNAL_BUILD_DIRECTORY",
        ));
    }
    let root = fs::canonicalize(&args[0])?;
    fs::create_dir_all(&args[1])?;
    let out = fs::canonicalize(&args[1])?;
    let workspace = fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
    if out.starts_with(&root) || out.starts_with(&workspace) {
        return Err(invalid(
            "reference build directory must be outside both original source and Rust workspace",
        ));
    }
    let driver = env::current_exe()?.with_file_name("t08_reference_driver");
    if !driver.is_file() {
        return Err(invalid(
            "first run cargo build --offline -p llama-vendor-remote --examples to build the Rust reference driver",
        ));
    }
    let manifest = out.join("inputs.sha256");
    fs::write(&manifest, INPUTS)?;
    let check = Command::new("sha256sum")
        .args(["--check", "--status"])
        .arg(&manifest)
        .current_dir(&root)
        .status()?;
    if !check.success() {
        return Err(invalid(format!(
            "frozen reference input hash mismatch; inspect {} with sha256sum --check from the source root",
            manifest.display()
        )));
    }
    println!(
        "Verified {} frozen native input files",
        INPUTS.lines().count()
    );
    let cc = env::var_os("CC").unwrap_or_else(|| "gcc".into());
    let cxx = env::var_os("CXX").unwrap_or_else(|| "g++".into());
    let mut commands = File::create(out.join("commands.txt"))?;
    for compiler in [&cc, &cxx] {
        let version = Command::new(compiler).arg("--version").output()?;
        if !version.status.success() {
            return Err(invalid(format!("compiler {:?} --version failed", compiler)));
        }
        fs::write(
            out.join(if compiler == &cc {
                "cc-version.txt"
            } else {
                "cxx-version.txt"
            }),
            version.stdout,
        )?;
    }
    let template = fs::read_to_string(root.join("ggml/src/ggml-version.h.in"))?;
    fs::write(
        out.join("ggml-version.h"),
        template
            .replace("@GGML_VERSION@", "0.29.0")
            .replace("@GGML_BUILD_COMMIT@", "frozen-2db1e23f"),
    )?;

    let mut objects = Vec::new();
    for chunk in SOURCES.chunks(4) {
        let mut children = Vec::new();
        for relative in chunk {
            let source = root.join(relative);
            let object = out.join(relative.replace('/', "_") + ".o");
            let log_path = object.with_extension("log");
            let log = File::create(&log_path)?;
            let mut command = compiler_command(
                if source.extension().is_some_and(|x| x == "c") {
                    &cc
                } else {
                    &cxx
                },
                &root,
                &out,
            );
            command.arg(if source.extension().is_some_and(|x| x == "c") {
                "-std=c11"
            } else {
                "-std=c++17"
            });
            command.args(["-c"]).arg(&source).arg("-o").arg(&object);
            writeln!(commands, "{command:?}")?;
            let child = command
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log))
                .spawn()?;
            children.push((relative, child, log_path));
            objects.push(object);
        }
        let mut failures = Vec::new();
        for (relative, mut child, log_path) in children {
            if child.wait()?.success() {
                println!("Compiled {relative}");
            } else {
                failures.push(log_path);
            }
        }
        if !failures.is_empty() {
            return Err(invalid(format!(
                "native compile failed; inspect {failures:?}"
            )));
        }
    }
    let library = out.join("libggml-reference-rpc.so");
    let mut link = Command::new(&cxx);
    link.args(["-shared", "-pthread", "-Wl,--no-undefined"])
        .args(&objects)
        .args(["-ldl", "-lm", "-o"])
        .arg(&library);
    run_logged(&mut link, &out.join("link.log"), &mut commands)?;

    // Only declarations and sizeof/offsetof comparisons are generated. No
    // implementation code, graph algorithm or transport shim is authored here.
    let rpc = fs::read_to_string(root.join("ggml/src/ggml-rpc/ggml-rpc.cpp"))?;
    let probe_source = out.join("rpc-layout.cpp");
    fs::write(&probe_source, layout_probe(&rpc)?)?;
    let mut probe = compiler_command(&cxx, &root, &out);
    probe
        .args(["-std=c++17"])
        .arg(&probe_source)
        .arg("-o")
        .arg(out.join("rpc-layout"));
    run_logged(&mut probe, &out.join("layout-build.log"), &mut commands)?;
    let layout = Command::new(out.join("rpc-layout")).output()?;
    if !layout.status.success() {
        return Err(invalid("layout comparison process failed"));
    }
    fs::write(out.join("native-layout.json"), &layout.stdout)?;
    print!("{}", String::from_utf8_lossy(&layout.stdout));
    fs::copy(driver, out.join("rpc-oracle"))?;
    let hashes = Command::new("sha256sum")
        .current_dir(&out)
        .args([
            "libggml-reference-rpc.so",
            "rpc-layout",
            "rpc-layout.cpp",
            "rpc-oracle",
            "inputs.sha256",
        ])
        .output()?;
    if !hashes.status.success() {
        return Err(invalid("failed to hash generated comparison artifacts"));
    }
    fs::write(out.join("artifacts.sha256"), hashes.stdout)?;
    println!(
        "PASS: isolated original library + Rust FFI driver: {}",
        out.join("rpc-oracle").display()
    );
    Ok(())
}

fn compiler_command(compiler: &OsString, root: &Path, out: &Path) -> Command {
    let mut command = Command::new(compiler);
    command.args([
        "-O1",
        "-fPIC",
        "-pthread",
        "-D_GNU_SOURCE",
        "-DGGML_CPU_GENERIC",
        "-DGGML_USE_CPU",
        "-DGGML_USE_RPC",
        "-DGGML_SCHED_MAX_COPIES=4",
    ]);
    for include in [
        root.join("ggml/include"),
        root.join("ggml/src"),
        root.join("ggml/src/ggml-cpu"),
        root.join("ggml/src/ggml-rpc"),
        out.to_path_buf(),
    ] {
        command.arg("-I").arg(include);
    }
    command
}

fn run_logged(command: &mut Command, log: &Path, commands: &mut File) -> Result<()> {
    writeln!(commands, "{command:?}")?;
    let output = command.output()?;
    let mut bytes = output.stdout;
    bytes.extend(output.stderr);
    fs::write(log, bytes)?;
    if output.status.success() {
        Ok(())
    } else {
        Err(invalid(format!(
            "comparison build failed; inspect {}",
            log.display()
        )))
    }
}

fn declaration<'a>(source: &'a str, marker: &str) -> Result<&'a str> {
    let start = source
        .find(marker)
        .ok_or_else(|| invalid(format!("missing original declaration {marker}")))?;
    if source[start + marker.len()..].contains(marker) {
        return Err(invalid(format!("ambiguous original declaration {marker}")));
    }
    let rest = &source[start..];
    let end = rest
        .find("\n};")
        .ok_or_else(|| invalid(format!("unterminated original declaration {marker}")))?
        + 3;
    Ok(&rest[..end])
}

fn layout_probe(rpc: &str) -> Result<String> {
    let mut source = String::from(
        "#include \"ggml.h\"\n#include \"transport.h\"\n#include <cstddef>\n#include <cstdio>\n#pragma pack(push, 1)\n",
    );
    for marker in [
        "struct rpc_tensor {",
        "enum rpc_cmd {",
        "struct rpc_msg_hello_req {",
        "struct rpc_msg_hello_rsp {",
    ] {
        source.push_str(declaration(rpc, marker)?);
        source.push('\n');
    }
    source.push_str("#pragma pack(pop)\nint main() {\n");
    let fields = [
        ("tensor_size", "sizeof(rpc_tensor)"),
        ("offset_data", "offsetof(rpc_tensor, data)"),
        ("offset_name", "offsetof(rpc_tensor, name)"),
        ("offset_use_count", "offsetof(rpc_tensor, use_count)"),
        ("hello_request_size", "sizeof(rpc_msg_hello_req)"),
        ("hello_response_size", "sizeof(rpc_msg_hello_rsp)"),
        ("command_count", "RPC_CMD_COUNT"),
        ("init_params_size", "sizeof(ggml_init_params)"),
        (
            "offset_init_no_alloc",
            "offsetof(ggml_init_params, no_alloc)",
        ),
    ];
    let format = fields
        .iter()
        .map(|(key, _)| format!("\\\"{key}\\\":%zu"))
        .collect::<Vec<_>>()
        .join(",");
    let expressions = fields
        .iter()
        .map(|(_, expression)| format!("static_cast<size_t>({expression})"))
        .collect::<Vec<_>>()
        .join(", ");
    source.push_str(&format!(
        "std::printf(\"{{{format}}}\\n\", {expressions});\nreturn 0;\n}}\n"
    ));
    Ok(source)
}
