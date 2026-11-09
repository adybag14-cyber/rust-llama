use llama_kernel_ir::{Artifact, Target, emit, kernels};
use llama_portable_gpu::compiler::ShaderCompiler;

#[test]
#[ignore = "requires a separately installed shaderc compiler; run explicitly for compiler evidence"]
fn glsl_corpus_compiles_to_real_spirv() {
    let compiler = ShaderCompiler::open().expect("external shaderc compiler required");
    for kernel in kernels::conformance_kernels().unwrap() {
        let source = emit(&kernel, Target::Glsl).unwrap();
        let binary = compiler
            .compile_glsl(&source)
            .unwrap_or_else(|error| panic!("{}: {error}", kernel.name));
        assert_eq!(&binary.bytes[..4], &0x07230203u32.to_le_bytes());
        assert_eq!(binary.source_sha256, source.sha256);
        assert!(binary.bytes.len() > 20);
        println!(
            "{}: {} SPIR-V bytes, sha256={}",
            kernel.name,
            binary.bytes.len(),
            binary.sha256
        );
    }
    let invalid = Artifact {
        target: Target::Glsl,
        entry_point: "main".into(),
        source: "#version 450\ninvalid shader".into(),
        sha256: llama_kernel_ir::sha256_hex(b"#version 450\ninvalid shader"),
        ir_version: 1,
    };
    assert!(compiler.compile_glsl(&invalid).is_err());
    let mut tampered = emit(&kernels::conformance_kernels().unwrap()[0], Target::Glsl).unwrap();
    tampered.source.push_str("\n// changed after emission\n");
    assert!(
        compiler
            .compile_glsl(&tampered)
            .unwrap_err()
            .0
            .contains("provenance")
    );
}

#[test]
fn non_glsl_input_is_rejected_before_compilation_when_compiler_present() {
    if let Ok(compiler) = ShaderCompiler::open() {
        let source = emit(&kernels::conformance_kernels().unwrap()[0], Target::Wgsl).unwrap();
        assert!(
            compiler
                .compile_glsl(&source)
                .unwrap_err()
                .0
                .contains("GLSL")
        );
    }
}
