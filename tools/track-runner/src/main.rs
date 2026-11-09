//! Rust-only generation, inventory, inspection and packaging for this checkpoint.
mod inventory;
mod package;
use llama_kernel_ir::{BufferData, Target, emit, execute, kernels, sha256_hex};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("probe") if args.len() == 1 => {
            println!("PROBE ONLY: availability is not backend parity or device execution proof.");
            for p in llama_device_family::probe_all() {
                println!(
                    "{} | {:?} | {:?} | devices={} | {}",
                    p.family,
                    p.status,
                    p.library,
                    p.devices.len(),
                    p.detail
                );
            }
            for p in llama_portable_gpu::probe_all() {
                println!(
                    "{} | {:?} | {:?} | devices={} | {}",
                    p.backend,
                    p.status,
                    p.library,
                    p.devices.len(),
                    p.detail
                );
            }
            for p in llama_vendor_remote::probe_all() {
                println!("{p:#?}");
            }
        }
        Some("smoke") if args.len() == 1 => {
            println!(
                "T05 scalar model validation slices: {:#?}",
                llama_model_runtime::smoke_demo()?
            );
            println!(
                "T08 real loopback TCP and external CBLAS: {:#?}",
                llama_vendor_remote::self_check()?
            );
            let kernel = kernels::matmul(
                kernels::MatrixLayout::contiguous(2, 3),
                kernels::MatrixLayout::contiguous(3, 2),
            )?;
            let mut buffers = BTreeMap::from([
                ("lhs".into(), BufferData::F32(vec![1., 2., 3., 4., 5., 6.])),
                (
                    "rhs".into(),
                    BufferData::F32(vec![7., 8., 9., 10., 11., 12.]),
                ),
                ("dst".into(), BufferData::F32(vec![0.; 4])),
            ]);
            let report = execute(&kernel, &mut buffers, 10000)?;
            if buffers["dst"] != BufferData::F32(vec![58., 64., 139., 154.]) {
                return Err("IR matrix multiplication mismatch".into());
            }
            println!(
                "T07 scalar IR validation (not GPU execution): {report:?}; output={:?}",
                buffers["dst"]
            );
        }
        Some("generate-kernels") if (2..=3).contains(&args.len()) => {
            let compile = args.get(2).is_some_and(|a| a == "--compile-vulkan");
            if args.len() == 3 && !compile {
                return Err("unknown generation option".into());
            }
            generate(Path::new(&args[1]), compile)?;
        }
        Some("audit-reference") if args.len() == 3 => {
            inventory::audit_reference(Path::new(&args[1]), Path::new(&args[2]))?
        }
        Some("audit-product") if args.len() == 2 => inventory::audit_product(Path::new(&args[1]))?,
        Some("package") if args.len() == 3 => {
            package::write_zip(Path::new(&args[1]), Path::new(&args[2]))?
        }
        Some("opencl-smoke") if args.len() == 1 => opencl_smoke()?,
        None | Some("help") | Some("--help") => println!(
            "T05–T08 Rust progress runner\n\nCommands:\n  probe                         Query 17 non-CPU backend families\n  smoke                         Run model/IR validation, TCP RPC and CBLAS\n  generate-kernels DIR [--compile-vulkan]\n                                Emit 36 Rust-derived shader sources; optional external shaderc\n  opencl-smoke                  Execute IR-generated kernels on a real OpenCL device\n  audit-reference SOURCE OUT    Hash and map frozen T05–T08 paths\n  audit-product ROOT            Audit Rust source/artifact policy\n  package ROOT OUTPUT.zip       Produce a deterministic ZIP without build caches\n\nHardware/compiler absence returns explicit errors for requested execution.\nThis checkpoint is not a complete llama.cpp rewrite."
        ),
        _ => return Err("invalid command or arguments; run track-runner help".into()),
    }
    Ok(())
}

fn generate(output: &Path, compile: bool) -> Result<()> {
    fs::create_dir_all(output)?;
    let compiler = if compile {
        Some(llama_portable_gpu::compiler::ShaderCompiler::open()?)
    } else {
        None
    };
    let mut manifest = String::from("path\tbytes\tsha256\ttarget\tentry_point\tprovenance\n");
    let mut source_count = 0;
    let mut binary_count = 0;
    for kernel in kernels::conformance_kernels()? {
        for target in Target::ALL {
            let artifact = emit(&kernel, target)?;
            let name = format!("{}.{}", kernel.name, target.extension());
            fs::write(output.join(&name), artifact.source.as_bytes())?;
            writeln!(
                manifest,
                "{name}\t{}\t{}\t{}\t{}\tRust IR v{}",
                artifact.source.len(),
                artifact.sha256,
                target.name(),
                artifact.entry_point,
                artifact.ir_version
            )?;
            source_count += 1;
            if target == Target::Glsl
                && let Some(compiler) = &compiler
            {
                let binary = compiler.compile_glsl(&artifact)?;
                let name = format!("{}.spv", kernel.name);
                fs::write(output.join(&name), &binary.bytes)?;
                writeln!(
                    manifest,
                    "{name}\t{}\t{}\tvulkan-spirv\t{}\texternal {} from source {}",
                    binary.bytes.len(),
                    binary.sha256,
                    artifact.entry_point,
                    binary.compiler_library,
                    binary.source_sha256
                )?;
                binary_count += 1;
            }
        }
    }
    use llama_device_family::ptx::{KernelKind, KernelSpec, PtxTarget};
    let ptx_cases = [
        (
            "cuda_add_f32",
            KernelKind::AddF32 {
                elements: 259,
                lhs_stride: 2,
                rhs_stride: 0,
                output_stride: 3,
            },
        ),
        (
            "cuda_scale_f32",
            KernelKind::ScaleF32 {
                elements: 65,
                input_stride: 2,
                output_stride: 1,
                scale: -0.25,
            },
        ),
        (
            "cuda_sum_f32",
            KernelKind::SumF32 {
                elements: 127,
                input_stride: 2,
            },
        ),
        (
            "cuda_q4_0_matvec",
            KernelKind::Q4MatVec {
                rows: 4,
                columns: 64,
                row_stride_bytes: 40,
                block_stride_bytes: 18,
                vector_stride: 2,
            },
        ),
    ];
    for (name, kind) in ptx_cases {
        let artifact =
            llama_device_family::ptx::generate(&KernelSpec { kind }, PtxTarget::default())?;
        let name = format!("{name}.ptx");
        fs::write(output.join(&name), artifact.source())?;
        writeln!(
            manifest,
            "{name}\t{}\t{}\tnvidia-ptx-7.0-sm60\t{}\tRust typed instruction IR v1; assembler and GPU unverified",
            artifact.source().len(),
            sha256_hex(artifact.source().as_bytes()),
            artifact.entry_point()
        )?;
    }
    fs::write(output.join("artifact-manifest.tsv"), manifest)?;
    println!(
        "Generated {source_count} portable shader sources, 4 PTX sources and {binary_count} SPIR-V binaries in {}",
        output.display()
    );
    Ok(())
}

fn opencl_smoke() -> Result<()> {
    let driver = llama_portable_gpu::opencl::OpenCl::open()?;
    let session = driver.session(0)?;
    let corpus = kernels::DispatchPlan::conformance_corpus()?;
    for plan in corpus {
        let kernel = plan.kernel();
        let mut buffers = BTreeMap::new();
        for b in &kernel.buffers {
            let data = match b.scalar {
                llama_kernel_ir::Scalar::F32 => BufferData::F32(
                    (0..b.min_elements)
                        .map(|i| ((i % 19) as f32 - 9.0) * 0.125)
                        .collect(),
                ),
                llama_kernel_ir::Scalar::U32 => {
                    BufferData::U32(vec![0x1234_5678; b.min_elements as usize])
                }
                _ => return Err("unexpected smoke-corpus scalar".into()),
            };
            buffers.insert(b.name.clone(), data);
        }
        // Avoid arbitrary packed-half NaNs while retaining unaligned rows.
        if kernel.name == "q4_0_matvec" {
            let mut bytes = vec![0x87u8; 3 + 2 * 41 + 36];
            for row in 0..3 {
                for block in 0..2 {
                    let base = 3 + row * 41 + block * 18;
                    bytes[base..base + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                }
            }
            buffers.insert(
                "packed".into(),
                BufferData::U32(kernels::bytes_to_words(&bytes)),
            );
        }
        let mut expected = buffers.clone();
        execute(kernel, &mut expected, 10_000_000)?;
        let result = session.execute(&plan, &mut buffers)?;
        for b in &kernel.buffers {
            if b.access == llama_kernel_ir::Access::ReadWrite {
                let (BufferData::F32(a), BufferData::F32(e)) =
                    (&buffers[&b.name], &expected[&b.name])
                else {
                    return Err("unexpected smoke output type".into());
                };
                for (i, (a, e)) in a.iter().zip(e).enumerate() {
                    if !a.is_finite() || (a - e).abs() > 2e-5 * (1.0 + e.abs()) {
                        return Err(format!("{} output[{i}] {a} != {e}", kernel.name).into());
                    }
                }
            }
        }
        println!("ACTUAL OPENCL EXECUTION PASS {result:?}");
    }
    Ok(())
}
