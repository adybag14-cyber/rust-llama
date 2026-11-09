# Cargo integration
The focused patch adds one target-specific existing local path dependency:
llama-tensor -> llama-platform-loader for x86_64 Windows MSVC and Linux GNU.
No new crate, registry/git dependency, build script, or root workspace member is
needed. Update the product Cargo.lock's llama-tensor dependency list using Cargo
in the complete workspace. The isolated root Cargo.toml and Cargo.lock are test
infrastructure only and are deliberately excluded. Quant remains dependency-free.
The dependency inventory delta records the direct OS erff boundary separately.
