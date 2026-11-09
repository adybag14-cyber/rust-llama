use llama_gguf::{Gguf, Value, TensorData, write};
use llama_quant::DType;
use std::{collections::BTreeMap, fs::File, path::PathBuf};

fn tensor(name: &str, dimensions: &[u64], salt: usize, mode: u8) -> (String, Vec<u64>, Vec<u8>) {
    let count = dimensions.iter().product::<u64>() as usize;
    let values: Vec<f32> = (0..count).map(|i| {
        match mode {
            1 => 1.0 + ((i+salt)%5) as f32 / 64.0,
            2 => -0.25 - ((i+salt)%7) as f32 / 16.0,
            _ => (((i*7+salt*11)%29) as f32-14.0) / 64.0,
        }
    }).collect();
    (name.into(), dimensions.to_vec(), values.into_iter().flat_map(f32::to_le_bytes).collect())
}
fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    let vocab = Gguf::open(&args[1]).unwrap();
    let output=PathBuf::from(&args[2]);
    std::fs::create_dir_all(&output).unwrap();
    let mut metadata: BTreeMap<String, Value> = vocab.metadata.into_iter().filter(|(key,_)| key.starts_with("tokenizer.")).collect();
    let size = metadata["tokenizer.ggml.tokens"].as_array().unwrap().len();
    assert_eq!(size,512);
    metadata.insert("general.architecture".into(),Value::String("mamba".into()));
    metadata.insert("general.type".into(),Value::String("model".into()));
    for (key,value) in [
        ("context_length",32),("embedding_length",4),("block_count",2),
        ("feed_forward_length",0),("attention.head_count",0),
        ("ssm.conv_kernel",4),("ssm.inner_size",8),("ssm.state_size",3),
        ("ssm.time_step_rank",2),
    ] { metadata.insert(format!("mamba.{key}"),Value::U32(value)); }
    metadata.insert("mamba.attention.layer_norm_rms_epsilon".into(),Value::F32(1e-5));
    let mut tensors=vec![tensor("token_embd.weight",&[4,512],1,0),tensor("output_norm.weight",&[4],2,1)];
    for layer in 0..2 {
        let key=|name:&str|format!("blk.{layer}.{name}");
        for (name,dimensions,mode) in [
            ("attn_norm.weight",vec![4],1),
            ("ssm_in.weight",vec![4,16],0),
            ("ssm_conv1d.weight",vec![4,8],0),
            ("ssm_conv1d.bias",vec![8],0),
            ("ssm_x.weight",vec![8,8],0),
            ("ssm_dt.weight",vec![2,8],0),
            ("ssm_dt.bias",vec![8],0),
            ("ssm_a",vec![3,8],2),
            ("ssm_d",vec![8],0),
            ("ssm_out.weight",vec![8,4],0),
        ] {
            let salt=tensors.len()+layer*37;
            tensors.push(tensor(&key(name),&dimensions,salt,mode));
        }
    }
    for (name,normalized,untied) in [
        ("tiny-mamba-default",None,false),
        ("tiny-mamba-rms",Some(true),false),
        ("tiny-mamba-untied",Some(false),true),
    ] {
        let mut metadata=metadata.clone();
        metadata.insert("general.name".into(),Value::String(name.into()));
        if let Some(value)=normalized { metadata.insert("mamba.ssm.dt_b_c_rms".into(),Value::Bool(value)); }
        let mut tensors=tensors.clone();
        if untied { tensors.push(tensor("output.weight",&[4,512],83,0)); }
        let borrowed:Vec<_>=tensors.iter().map(|(name,dimensions,data)|TensorData{name,dimensions,dtype:DType::F32,data}).collect();
        let path=output.join(format!("{name}.gguf"));
        write(File::create(&path).unwrap(),&metadata,&borrowed).unwrap();
        println!("{} {}",path.display(),path.metadata().unwrap().len());
    }
}
