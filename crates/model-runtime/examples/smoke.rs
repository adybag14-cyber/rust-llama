fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{:#?}", llama_model_runtime::smoke_demo()?);
    Ok(())
}
