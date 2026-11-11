//! Dense Qwen converter configuration and sentence-transformer sidecars.
use crate::codec::SourceProfile;
use crate::config::{Architecture, Config};
use crate::python_unicode::lower_contains_ascii;
use crate::util::{JsonBudget, Metadata, integer, object, present, text, truth};
use crate::{Error, Result, string};
use llama_checkpoint::{Object, SourceDirectory, Value as Json};

/// AutoConfig falls back to the raw JSON after a layer-type validation error.
pub(crate) fn valid_auto_config(root: &Object) -> bool {
    let layers = match root.get("num_hidden_layers") {
        Some(v) => match integer(v) {
            Ok(n) if n >= 0 => n,
            _ => return false,
        },
        None => 32,
    };
    fn allowed(value: &str) -> bool {
        matches!(
            value,
            "full_attention" | "sliding_attention" | "chunked_attention" | "linear_attention"
        )
    }
    if let Some(types) = present(root.get("layer_types")) {
        // Python validates an iterable: dictionaries iterate their keys.
        let valid = match types {
            Json::Array(types) => {
                types.len() as i128 == i128::from(layers)
                    && types.iter().all(|v| text(v).is_ok_and(allowed))
            }
            Json::Object(types) => {
                types.len() as i128 == i128::from(layers)
                    && types
                        .entries()
                        .all(|(key, _)| key.as_str().is_some_and(allowed))
            }
            Json::Text(value) => layers == 0 && value.as_wtf8().is_empty(),
            _ => false,
        };
        if !valid {
            return false;
        }
    } else if layers > 0
        && root.get("use_sliding_window").is_some_and(truth)
        // The constructor defaults an omitted window to 4096; only explicit
        // None prevents the per-layer threshold comparison.
        && !matches!(root.get("sliding_window"), Some(Json::Null))
        && root.get("max_window_layers").is_some_and(|v| !matches!(v, Json::Number(_) | Json::Bool(_)))
    {
        return false;
    }
    true
}

pub(crate) fn validate_variant(
    source: &mut SourceDirectory,
    config: &Config,
    folder: &str,
    budget: &mut JsonBudget,
    profile: SourceProfile,
) -> Result<()> {
    if !matches!(config.family, Architecture::Qwen2 | Architecture::Qwen3) {
        return Ok(());
    }
    if let Some(raw) = source.read_optional("tokenizer_config.json")? {
        let value = budget.parse(raw)?;
        if present(object(&value)?.get("tokenizer_class"))
            .map(text)
            .transpose()?
            == Some("HybridDNATokenizer")
        {
            return Err(Error::Unsupported("HybridDNA tokenizer conversion"));
        }
    }
    if config.family == Architecture::Qwen3 {
        let origin = present(config.root().get("architectures"))
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .map(text)
            .transpose()?
            .unwrap_or("");
        if origin == "InternS1ForConditionalGeneration" {
            return Err(Error::Unsupported(
                "InternS1 specialized vocabulary conversion",
            ));
        }
        let hints_rerank = |hint: &str| -> Result<bool> {
            Ok(lower_contains_ascii(hint, "qwen3-reranker", profile)?
                || lower_contains_ascii(hint, "qwen3-vl-reranker", profile)?)
        };
        let mut rerank = hints_rerank(folder)?
            || hints_rerank(origin)?
            || lower_contains_ascii(origin, "sequenceclassification", profile)?;
        for name in ["_name_or_path", "model_type"] {
            if let Some(value) = present(config.get(name)?) {
                rerank |= hints_rerank(text(value)?)?;
            }
        }
        if let Some(raw) = source.read_optional("README.md")? {
            let readme = std::str::from_utf8(raw)
                .map_err(|_| Error::Invalid("Qwen3 model card is not UTF-8"))?;
            rerank |= lower_contains_ascii(readme, "# qwen3-reranker", profile)?
                || lower_contains_ascii(readme, "# qwen3-vl-reranker", profile)?;
        }
        if rerank {
            return Err(Error::Unsupported("Qwen3 reranker head conversion"));
        }
    }
    Ok(())
}

fn key(config: &Config, suffix: &str) -> Result<String> {
    let mut result = string(config.family.name())?;
    result
        .try_reserve(suffix.len() + 1)
        .map_err(|_| Error::Limit("Qwen metadata key"))?;
    result.push('.');
    result.push_str(suffix);
    Ok(result)
}
fn pool_name(value: &str) -> Option<u32> {
    match value {
        "mean" => Some(1),
        "cls" => Some(2),
        "lasttoken" => Some(3),
        _ => None,
    }
}
pub(crate) fn pooling_metadata(
    source: &mut SourceDirectory,
    config: &Config,
    budget: &mut JsonBudget,
    out: &mut Metadata,
) -> Result<()> {
    if !matches!(config.family, Architecture::Qwen2 | Architecture::Qwen3) {
        return Ok(());
    }
    let modules = source
        .read_optional("modules.json")?
        .map(|raw| budget.parse(raw))
        .transpose()?;
    let mut selected = None;
    if let Some(modules) = modules {
        let modules = modules.as_array().ok_or(Error::Invalid(
            "sentence-transformer modules must be an array",
        ))?;
        for module in modules {
            let module = object(module)?;
            let kind = text(
                module
                    .get("type")
                    .ok_or(Error::Invalid("module type is absent"))?,
            )?;
            if kind.ends_with("Pooling") {
                selected = Some(string(text(
                    module
                        .get("path")
                        .ok_or(Error::Invalid("pooling module path is absent"))?,
                )?)?);
                break;
            }
        }
    }
    if let Some(mut path) = selected {
        path.try_reserve(12)
            .map_err(|_| Error::Limit("pooling module path"))?;
        if !path.is_empty() && !path.ends_with('/') {
            path.push('/');
        }
        path.push_str("config.json");
        let raw = source.read_required(&path)?;
        let pool = budget.parse(raw)?;
        let pool = object(&pool)?;
        let mode = if pool.get("pooling_mode_mean_tokens").is_some_and(truth) {
            1
        } else if pool.get("pooling_mode_cls_token").is_some_and(truth) {
            2
        } else if pool.get("pooling_mode_lasttoken").is_some_and(truth) {
            3
        } else {
            present(pool.get("pooling_mode"))
                .map(text)
                .transpose()?
                .and_then(pool_name)
                .ok_or(Error::Unsupported(
                    "only mean, cls and lasttoken pooling conversion is implemented",
                ))?
        };
        out.uint(&key(config, "pooling_type")?, mode)?;
    } else if let Some(raw) = source.read_optional("embedding_config.json")? {
        let value = budget.parse(raw)?;
        if let Some(mode) = present(object(&value)?.get("pooling")) {
            if text(mode)? != "last_token" {
                return Err(Error::Unsupported("embedding pooling conversion"));
            }
            out.uint(&key(config, "pooling_type")?, 3)?;
        }
    }
    if let Some(value) = present(config.get("classifier_pooling")?) {
        let mode = match text(value)? {
            "mean" => 1,
            "cls" => 2,
            _ => return Err(Error::Unsupported("classifier pooling conversion")),
        };
        out.uint(&key(config, "classifier.pooling_type")?, mode)?;
    }
    if let Some(value) = present(config.get("classifier_activation")?) {
        let value = text(value)?;
        if !matches!(value, "gelu" | "silu" | "tanh") {
            return Err(Error::Unsupported("classifier activation conversion"));
        }
        out.string(&key(config, "classifier.activation")?, value)?;
    }
    Ok(())
}
