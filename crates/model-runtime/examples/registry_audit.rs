fn main() {
    let Some(root) = std::env::args_os().nth(1) else {
        eprintln!(
            "usage: cargo run -p llama-model-runtime --example registry_audit -- /path/to/llama.cpp-master"
        );
        std::process::exit(2);
    };
    match llama_model_runtime::registry::audit_snapshot(root) {
        Ok(audit) => println!("{audit:#?}"),
        Err(error) => {
            eprintln!("registry audit failed: {error}");
            std::process::exit(1);
        }
    }
}
