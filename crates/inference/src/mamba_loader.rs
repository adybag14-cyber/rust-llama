//! Frozen Mamba-1 GGUF loading. Recurrent arithmetic lives in the Rust runtime.
use crate::loader::{details, float, matrix, split_count, uint, unsupported, vector};
use crate::{Error, Executor, LoadOptions, Model, Result};
use llama_gguf::{Gguf, Value};
use llama_model_runtime::{MambaConfig, MambaLayer, MambaModel, MambaWeights, Matrix};
use llama_tokenizer::Tokenizer;
use std::{collections::BTreeSet, fs::File, path::Path, sync::Arc};

fn scalar_matrix(
    g: &mut Gguf<File>,
    used: &mut BTreeSet<String>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let tensor = g
        .tensor(name)
        .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))?;
    // The frozen SSM convolution and scan kernels require scalar F32 storage.
    // Matrix products use the separate native packed arithmetic path.
    if tensor.dtype.id() != 0 {
        return Err(unsupported(format!(
            "Mamba recurrent tensor {name} requires F32"
        )));
    }
    matrix(g, used, name, rows, cols)
}

pub(super) fn load(mut g: Gguf<File>, path: &Path, options: &LoadOptions) -> Result<Model> {
    let metadata = &g.metadata;
    match metadata.get("general.type") {
        None => {}
        Some(Value::String(kind)) if kind == "model" => {}
        Some(Value::String(kind)) => {
            return Err(unsupported(format!("GGUF general.type={kind}")));
        }
        Some(_) => return Err(Error::Invalid("general.type must be a string".into())),
    }
    let splits = split_count(metadata)?;
    if splits > 1 {
        return Err(unsupported("split GGUF model loading"));
    }
    let tokenizer = Tokenizer::from_metadata(metadata)?;
    let layers = uint(metadata, "mamba.block_count", None)?;
    if layers == 0 || layers > 512 {
        return Err(Error::Invalid("layer count must be 1..=512".into()));
    }
    let train_context = uint(metadata, "mamba.context_length", None)?;
    let normalize_dt_b_c = match metadata.get("mamba.ssm.dt_b_c_rms") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(Error::Invalid(
                "mamba.ssm.dt_b_c_rms must be a boolean".into(),
            ));
        }
    };
    if !metadata.contains_key("mamba.attention.layer_norm_rms_epsilon") {
        return Err(Error::Invalid(
            "missing mamba.attention.layer_norm_rms_epsilon".into(),
        ));
    }
    let config = MambaConfig {
        vocabulary: tokenizer.vocabulary_size(),
        width: uint(metadata, "mamba.embedding_length", None)?,
        inner: uint(metadata, "mamba.ssm.inner_size", None)?,
        state_size: uint(metadata, "mamba.ssm.state_size", None)?,
        convolution_width: uint(metadata, "mamba.ssm.conv_kernel", None)?,
        time_step_rank: uint(metadata, "mamba.ssm.time_step_rank", None)?,
        max_context: train_context.min(options.context_length),
        rms_epsilon: float(metadata, "mamba.attention.layer_norm_rms_epsilon", 1e-5)?,
        normalize_dt_b_c,
    };
    // Validate multiplication bounds before deriving any tensor dimensions.
    config.validate()?;
    let input_rows = config
        .inner
        .checked_mul(2)
        .ok_or_else(|| Error::Invalid("Mamba input dimension overflow".into()))?;
    let projection_rows = config
        .state_size
        .checked_mul(2)
        .and_then(|n| n.checked_add(config.time_step_rank))
        .ok_or_else(|| Error::Invalid("Mamba projection dimension overflow".into()))?;
    let (id, template, decoded_weight_bytes) = details(&g, path, options)?;
    let mut used = BTreeSet::new();
    let token_embedding = matrix(
        &mut g,
        &mut used,
        "token_embd.weight",
        config.vocabulary,
        config.width,
    )?;
    let output_norm = vector(&mut g, &mut used, "output_norm.weight", config.width)?;
    let output = if g.tensor("output.weight").is_some() {
        Some(matrix(
            &mut g,
            &mut used,
            "output.weight",
            config.vocabulary,
            config.width,
        )?)
    } else {
        None
    };
    let mut weights = Vec::with_capacity(layers);
    for index in 0..layers {
        let key = |name: &str| format!("blk.{index}.{name}");
        weights.push(MambaLayer {
            norm: vector(&mut g, &mut used, &key("attn_norm.weight"), config.width)?,
            input: matrix(
                &mut g,
                &mut used,
                &key("ssm_in.weight"),
                input_rows,
                config.width,
            )?,
            convolution: scalar_matrix(
                &mut g,
                &mut used,
                &key("ssm_conv1d.weight"),
                config.inner,
                config.convolution_width,
            )?,
            convolution_bias: vector(&mut g, &mut used, &key("ssm_conv1d.bias"), config.inner)?,
            x_projection: matrix(
                &mut g,
                &mut used,
                &key("ssm_x.weight"),
                projection_rows,
                config.inner,
            )?,
            dt_projection: matrix(
                &mut g,
                &mut used,
                &key("ssm_dt.weight"),
                config.inner,
                config.time_step_rank,
            )?,
            dt_bias: vector(&mut g, &mut used, &key("ssm_dt.bias"), config.inner)?,
            a: scalar_matrix(
                &mut g,
                &mut used,
                &key("ssm_a"),
                config.inner,
                config.state_size,
            )?,
            d: vector(&mut g, &mut used, &key("ssm_d"), config.inner)?,
            output: matrix(
                &mut g,
                &mut used,
                &key("ssm_out.weight"),
                config.width,
                config.inner,
            )?,
            // The frozen Mamba-1 loader does not load optional dt/B/C norm
            // tensors. Its boolean metadata flag uses unweighted RMS norms.
            dt_norm: None,
            b_norm: None,
            c_norm: None,
        });
    }
    let unconsumed: Vec<_> = g
        .tensors
        .iter()
        .filter(|tensor| !used.contains(&tensor.name))
        .map(|tensor| tensor.name.as_str())
        .collect();
    if !unconsumed.is_empty() {
        return Err(unsupported(format!(
            "model contains unimplemented tensors: {}",
            unconsumed.join(", ")
        )));
    }
    let model = MambaModel::new(
        config,
        MambaWeights {
            token_embedding,
            layers: weights,
            output_norm,
            output,
        },
    )?;
    Ok(Model {
        runtime: Executor::Decoder(Arc::new(llama_model_runtime::Model::Mamba(model))),
        tokenizer,
        id,
        template,
        decoded_weight_bytes,
        context_limit: train_context.min(options.context_length),
        embedding_pooling: None,
    })
}
