//! Optional independent fixture recorder, NEVER part of the production path.
//! Usage: record_q4_reference /path/to/frozen/llama.cpp /temporary/build /output
//! Compiles only the supplied original quantization C translation unit in an
//! isolated directory, then garbage-collects every function except the oracle.
//! Requires a C compiler/linker solely for this reference-data capture.
use llama_platform_loader::DynamicLibrary;
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    process::Command,
};
type Dequantize = unsafe extern "C" fn(*const c_void, *mut f32, i64);
fn command(command: &mut Command) -> Result<(), Box<dyn std::error::Error>> {
    let status = command.status()?;
    if !status.success() {
        return Err(format!("oracle compiler command failed: {status}").into());
    }
    Ok(())
}
fn run(source: &Path, temporary: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !cfg!(target_endian = "little") {
        return Err("oracle recorder requires little-endian host".into());
    }
    std::fs::create_dir_all(temporary)?;
    std::fs::create_dir_all(output)?;
    let object = temporary.join("ggml-quants.o");
    let isolated = temporary.join("q4-oracle.o");
    let library = temporary.join("libq4-oracle.so");
    command(
        Command::new("cc")
            .args([
                "-std=c11",
                "-D_GNU_SOURCE",
                "-O2",
                "-fPIC",
                "-ffunction-sections",
                "-fdata-sections",
                "-c",
            ])
            .arg(source.join("ggml/src/ggml-quants.c"))
            .arg("-I")
            .arg(source.join("ggml/include"))
            .arg("-I")
            .arg(source.join("ggml/src"))
            .arg("-o")
            .arg(&object),
    )?;
    command(
        Command::new("ld")
            .args(["-r", "--gc-sections", "-u", "dequantize_row_q4_0"])
            .arg(&object)
            .arg("-o")
            .arg(&isolated),
    )?;
    command(
        Command::new("cc")
            .arg("-shared")
            .arg(&isolated)
            .arg("-lm")
            .arg("-o")
            .arg(&library),
    )?;
    let library = DynamicLibrary::open(&[library.to_str().ok_or("non-UTF8 oracle path")?])?;
    // SAFETY: this is the exact signature in the frozen ggml-quants.h. The
    // isolated library stays alive until all reference calls have returned.
    let dequantize: Dequantize = unsafe { library.symbol(c"dequantize_row_q4_0")? };
    let scales = [
        0x3c00u16, 0x3800, 0xbc00, 0x3400, 0x0001, 0x0400, 0xc000, 0x4000,
    ];
    let mut input = Vec::new();
    for (block, scale) in scales.iter().enumerate() {
        input.extend_from_slice(&scale.to_le_bytes());
        for j in 0..16 {
            input.push(((j + block) % 16) as u8 | ((((15 - j) + block * 3) % 16) as u8) << 4);
        }
    }
    // block_q4_0 has alignment 2, so use u16-backed storage rather than casting
    // a byte vector that has no guaranteed two-byte alignment.
    let words: Vec<u16> = input
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let mut result = vec![0f32; 256];
    // SAFETY: eight complete 18-byte blocks of aligned storage and 256 writable
    // output floats. Oracle length is a multiple of its 32-value block size.
    unsafe {
        dequantize(
            words.as_ptr().cast(),
            result.as_mut_ptr(),
            result.len() as i64,
        );
    }
    std::fs::write(output.join("q4_0_reference_input.bin"), input)?;
    let bytes: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(output.join("q4_0_reference_f32.bin"), bytes)?;
    println!("Captured 8 original Q4_0 blocks and 256 reference floats.");
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        return Err(
            "usage: record_q4_reference REFERENCE_SOURCE TEMPORARY_BUILD OUTPUT_DIRECTORY".into(),
        );
    }
    run(&args[0], &args[1], &args[2])
}
